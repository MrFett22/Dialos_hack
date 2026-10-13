"""Motor del simulador de la página (sin Streamlit, para poder probarlo).

Un lanzamiento descrito por el usuario (físico + ubicación + contexto + ambiente) se evalúa con los
modelos FINALES del pipeline:

- Pitching+ v3 (`src/models/artifacts/pitching_plus_v3.pkl`): probabilidades de los 10 eventos y E[Δ].
- Stuff+ Physical (`stuff_plus_physical.pkl`): P(whiff | swing) con el lado del bateador marginalizado.
- Location+ v2 (`location_plus_v2.pkl`): valor de la ubicación respecto de la típica de su contexto.

El movimiento se interpreta como medido en el ambiente elegido. `trajectory.synthesize_pitch` reconstruye
la trayectoria que reproduce exactamente ese movimiento y la vuela en los otros dos ambientes, escalando
drag y Magnus con los ratios medidos (contrafactual físico). Nada se entrena aquí.

Escala de un lanzamiento (`data/web/simulator_scales.json`): 100 = lanzamiento promedio de la liga y
10 puntos = 1 desviación estándar entre lanzamientos, con el mismo ajuste por ambiente (por categoría)
que los índices publicados.
"""

import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import scoring  # noqa: E402
from src.features import trajectory  # noqa: E402
from src.models import location_plus_v2 as locp  # noqa: E402
from src.models import run_value_chain as chain  # noqa: E402
from src.models import stuff_plus_physical as spp  # noqa: E402
from src.models import stuff_plus_v2  # noqa: E402

WEB = ROOT / "data" / "web"
CATS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
CAT_LABEL = {"No Altitude": "No Altitude", "Medium Altitude": "Medium", "Extreme Altitude": "Extreme"}
PITCH_TYPES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
# input del usuario → columna del dominio (simulator_domain.json)
PHYS_INPUTS = {"velo": "RelSpeed", "ivb": "InducedVertBreak", "hb_arm": "hb_arm", "spin_rate": "SpinRate",
               "extension": "Extension", "rel_height": "RelHeight", "rel_side_arm": "rel_side_arm"}
LABELS = {"velo": "velocidad (mph)", "ivb": "movimiento vertical inducido (pulg.)", "hb_arm": "movimiento horizontal (pulg., + = lado del brazo)",
          "spin_rate": "spin rate (rpm)", "extension": "extensión (pies)", "rel_height": "altura de salida (pies)",
          "rel_side_arm": "salida lateral (pies, + = lado del brazo)", "spin_axis": "eje de giro (°)"}
EVENT_LABELS = {"ball": "bola", "called_strike": "strike cantado", "hbp": "golpeado", "whiff": "whiff (swing fallado)", "foul": "foul",
                "out": "out en juego", "single": "sencillo", "double": "doble", "triple": "triple", "home_run": "jonrón"}
LOC_MIN_CELL = 20          # pitches reales del tipo en la celda de 0.25 pies para considerar la ubicación "con soporte"
IMPROVE_SD = 0.5           # tamaño de los cambios de "¿qué podría mejorar?" (en sd del tipo)


@dataclass
class Result:
    inputs: dict
    env: str
    stuff_plus: float
    location_plus: float
    pitching_plus: float
    probs: dict
    p_swing: float
    p_whiff_swing_model: float
    p_whiff_swing_stuff: float
    p_bip: float
    e_delta_bip: float
    xrv_per_100: float
    p_whiff: float                 # whiff por pitch
    contact_damage: float          # P(BIP) × E[RV | BIP], carreras por pitch
    region: str                    # corazón / borde / persecución / desperdicio (location_plus_v2.region)
    by_env: pd.DataFrame
    percentiles: dict
    percentiles_tipo: dict
    warnings: list = field(default_factory=list)
    imputed: list = field(default_factory=list)


class Engine:
    def __init__(self):
        with open(chain.FINAL_MODEL_PATHS["pitching"], "rb") as fh:
            self.pit = pickle.load(fh)
        with open(spp.FINAL_PATH, "rb") as fh:
            self.stf = pickle.load(fh)
        with open(locp.FINAL_PATH, "rb") as fh:
            self.loc = pickle.load(fh)
        self.scales = json.loads((WEB / "simulator_scales.json").read_text(encoding="utf-8"))
        self.domain = json.loads((WEB / "simulator_domain.json").read_text(encoding="utf-8"))
        self.ratios = scoring.load_ratios()
        self.defaults = scoring.compute_pitch_defaults()
        self.count_shares = scoring.load_count_shares()
        summary = json.loads((WEB / "summary.json").read_text(encoding="utf-8"))
        self.type_effects = pd.read_csv(WEB / "pitch_type_altitude_effects.csv").set_index("tipo")
        self.type_rules = summary["tipos_pitch"]
        self.regions = pd.DataFrame(summary["location"]["regiones"]).set_index("region")
        env_path = WEB / "pitching_env_values.json"
        self.env_values = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else None

    # ------------------------------------------------------------------ dominio
    def type_domain(self, pitch_type: str) -> dict:
        return self.domain["tipos"][pitch_type]

    def median_inputs(self, pitch_type: str, p_throws: str = "Right", b_side: str = "Right", env: str = "No Altitude") -> dict:
        p = self.type_domain(pitch_type)["percentiles"]
        x = {k: p[c]["50"] for k, c in PHYS_INPUTS.items()}
        return {"pitch_type": pitch_type, "p_throws": p_throws, "b_side": b_side, **x, "spin_axis": None,
                "plate_x": 0.0, "plate_z": 2.5, "balls": 0, "strikes": 0, "env": env}

    @staticmethod
    def percentile_of(value: float, table: dict) -> float:
        qs = np.array([float(q) for q in table])
        vs = np.array(list(table.values()))
        order = np.argsort(vs)
        return float(np.interp(value, vs[order], qs[order]))

    def fill_missing(self, inputs: dict) -> tuple[dict, list]:
        """Inputs faltantes o no numéricos → mediana del tipo de pitch (se avisa)."""
        x = dict(inputs)
        if x.get("pitch_type") not in PITCH_TYPES:
            x["pitch_type"] = "Four-Seam"
        med = self.median_inputs(x["pitch_type"])
        imputed = []
        for k, v in med.items():
            cur = x.get(k)
            bad = cur is None or (isinstance(cur, float) and math.isnan(cur))
            if bad and v is not None:
                x[k] = v
                imputed.append(k)
            elif k not in x:
                x[k] = v
        if x.get("p_throws") not in ("Right", "Left"):
            x["p_throws"] = "Right"
        if x.get("b_side") not in ("Right", "Left"):
            x["b_side"] = "Right"
        if x.get("env") not in CATS:
            x["env"] = "No Altitude"
        x["balls"] = int(min(max(int(x["balls"]), 0), 3))
        x["strikes"] = int(min(max(int(x["strikes"]), 0), 2))
        return x, imputed

    def implied_axis(self, x: dict) -> float:
        return float(scoring.estimate_spin_axis(x["ivb"], x["hb_arm"]))

    def domain_check(self, x: dict) -> dict:
        """Percentil de cada input, distancia de Mahalanobis, eje de giro y soporte de la ubicación."""
        dom = self.type_domain(x["pitch_type"])
        rows, warn = [], []
        for k, c in PHYS_INPUTS.items():
            pct = self.percentile_of(x[k], dom["percentiles"][c])
            p = dom["percentiles"][c]
            outside = x[k] < p["1"] or x[k] > p["99"]
            rows.append({"variable": LABELS[k], "valor": x[k], "percentil": pct, "p5": p["5"], "p95": p["95"], "fuera_p1_p99": outside})
            if outside:
                warn.append(f"{LABELS[k]} = {x[k]:.1f} está fuera del rango común de los {x['pitch_type']} (p1–p99: {p['1']:.1f}–{p['99']:.1f}).")
        v = np.array([x[k] for k in PHYS_INPUTS], float) - np.array(dom["media"])
        dist = float(np.sqrt(v @ np.array(dom["cov_inv"]) @ v))
        q = dom["mahalanobis_cuantiles"]
        joint = dist > q["99"]
        if joint:
            warn.append(f"La combinación de variables está fuera del rango común observado para {x['pitch_type']} (distancia {dist:.1f} "
                        f"> percentil 99 de los lanzamientos reales, {q['99']:.1f}). La predicción tiene menor confiabilidad.")
        axis_dev = None
        if x.get("spin_axis") is not None:
            axis_dev = abs((x["spin_axis"] - self.implied_axis(x) + 180) % 360 - 180)
            if axis_dev > dom["desvio_eje_abs_cuantiles"]["99"]:
                warn.append(f"El eje de giro se aleja {axis_dev:.0f}° del que implica el movimiento (p99 real: "
                            f"{dom['desvio_eje_abs_cuantiles']['99']:.0f}°). Combinación poco común.")
        cell = f"{int(math.floor(x['plate_x'] / self.domain['celda_ubicacion_pies']))},{int(math.floor(x['plate_z'] / self.domain['celda_ubicacion_pies']))}"
        n_cell = int(dom["celdas_ubicacion"].get(cell, 0))
        if n_cell < LOC_MIN_CELL:
            warn.append(f"Muy pocos {x['pitch_type']} reales terminaron en esa ubicación ({n_cell} en una celda de 0.25 pies): "
                        "Location+ y Pitching+ son menos confiables ahí.")
        return {"tabla": pd.DataFrame(rows), "mahalanobis": dist, "mahalanobis_p99": q["99"],
                "mahalanobis_percentil": self.percentile_of(dist, q) if dist <= q["99.9"] else 99.9,
                "fuera_conjunto": joint, "desvio_eje": axis_dev, "pitches_en_celda": n_cell, "advertencias": warn}

    # ------------------------------------------------------------------ simulación
    def _features(self, x: dict, return_states: bool = False):
        sign = 1.0 if x["p_throws"] == "Right" else -1.0
        d = scoring._defaults_for(self.defaults, x["pitch_type"], x["p_throws"], x["env"])
        out = trajectory.synthesize_pitch(
            rel_speed_mph=x["velo"], ivb_in=x["ivb"], hb_in=x["hb_arm"] * sign, extension_ft=x["extension"],
            rel_height_ft=x["rel_height"], rel_side_ft=x["rel_side_arm"] * sign, drag_k=float(d["k_drag"]),
            target_side_ft=x["plate_x"], target_height_ft=x["plate_z"], reference_category=x["env"],
            pitch_type=x["pitch_type"], ratios=self.ratios, return_states=True)
        traj, states = out
        inputs = {"AutoPitchType": x["pitch_type"], "RelSpeed": x["velo"], "InducedVertBreak": x["ivb"], "HorzBreak": x["hb_arm"] * sign,
                  "SpinRate": x["spin_rate"], "Extension": x["extension"], "RelHeight": x["rel_height"],
                  "PitcherThrows": x["p_throws"], "BatterSide": x["b_side"]}
        rows = scoring._feature_rows(inputs, d, traj)
        rows["RelSide"] = x["rel_side_arm"] * sign
        axis_r = x["spin_axis"] if x.get("spin_axis") is not None else self.implied_axis(x)  # marco de un derecho
        rows["SpinAxis"] = axis_r if sign > 0 else 360 - axis_r
        rows["Balls"], rows["Strikes"] = x["balls"], x["strikes"]
        feats, _ = stuff_plus_v2.build_v2_features(rows, location_coefs=self.pit["location_coefs"])
        return (feats, traj, states) if return_states else (feats, traj)

    def _location_value(self, x: dict, same_side: str) -> float:
        loc = self.loc
        bat = x["b_side"]
        row = pd.DataFrame({"loc_in": [x["plate_x"] * (-1.0 if bat == "Left" else 1.0)], "PlateLocHeight": [x["plate_z"]],
                            "Balls": [x["balls"]], "Strikes": [x["strikes"]]})
        for c, v in (("AutoPitchType", x["pitch_type"]), ("same_side", same_side), ("bat_side", bat)):
            row[c] = pd.Categorical([v], categories=loc["categories"][c])
        X = row[loc["numeric_features"] + loc["categorical_features"]]
        pr = {}
        for name, _, _, n_classes in chain.SUBMODELS:
            p = loc["models"][name].predict_proba(X)
            pr[name] = p if n_classes > 2 else p[:, 1:2]
        D = loc["deltas_by_count"].loc[(x["balls"], x["strikes"])].to_numpy()[None, :]
        xrv = float(chain.expected_run_value(chain.event_probabilities(pr), D)[0])
        ref = loc["reference_by_context"].get((x["pitch_type"], same_side, bat, x["balls"], x["strikes"]))
        return xrv - ref

    def _plus(self, name: str, value: float, env: str | None = None) -> float:
        s = self.scales[name]
        v = value + (s["offset_por_ambiente"][env] if env and "offset_por_ambiente" in s else 0.0)
        z = (v - s["media_lanzamiento"]) / s["sd_lanzamiento"]
        return float(100 + 10 * (z if s["mayor_es_mejor"] else -z))

    def simulate(self, inputs: dict) -> Result:
        x, imputed = self.fill_missing(inputs)
        feats, traj = self._features(x)
        pred = chain.predict_expected_run_value(self.pit, feats)
        p_st = spp.marginal_predict(self.stf, feats)
        same_side = str(feats["same_side"].iloc[0])
        loc_val = self._location_value(x, same_side)
        loc_plus = self._plus("location", loc_val)
        rows = []
        for i, cat in enumerate(traj["altitude_category"]):
            pr = pred.iloc[i]
            p_swing = float(pr[["whiff", "foul"] + chain.BIP_CLASSES].sum())
            p_bip = float(pr[chain.BIP_CLASSES].sum())
            rows.append({"ambiente": cat, "stuff_plus": self._plus("stuff", p_st[i], cat), "location_plus": loc_plus,
                         "pitching_plus": self._plus("pitching", pr["xrv"], cat), "p_whiff_swing_stuff": float(p_st[i]),
                         "p_whiff": float(pr["whiff"]), "p_swing": p_swing, "p_whiff_swing_model": float(pr["whiff"]) / p_swing,
                         "p_bip": p_bip, "e_delta_bip": float(pr["comp_bip"]) / p_bip if p_bip > 0 else float("nan"),
                         "xrv_per_100": 100 * float(pr["xrv"]), "ivb": float(traj["InducedVertBreak"].iloc[i]),
                         "hb_arm": float(traj["HorzBreak"].iloc[i]) * (1 if x["p_throws"] == "Right" else -1),
                         "zone_speed": float(traj["ZoneSpeed"].iloc[i]), "vaa": float(traj["VertApprAngle"].iloc[i]),
                         **{f"p_{e}": float(pr[e]) for e in chain.EVENTS}})
        by_env = pd.DataFrame(rows)
        me = by_env.set_index("ambiente").loc[x["env"]]
        pct = {k: self.percentile_of(me[k], self.scales[n]["percentiles_lanzamientos"])
               for k, n in (("stuff_plus", "stuff"), ("location_plus", "location"), ("pitching_plus", "pitching"))}
        pct_t = {k: self.percentile_of(me[k], self.scales[n]["percentiles_por_tipo"][x["pitch_type"]])
                 for k, n in (("stuff_plus", "stuff"), ("location_plus", "location"), ("pitching_plus", "pitching"))}
        warnings = self.domain_check(x)["advertencias"]
        if imputed:
            warnings.append("Faltaban datos (" + ", ".join(LABELS.get(k, k) for k in imputed) + "); se usó la mediana del tipo de pitch.")
        return Result(inputs=x, env=x["env"], stuff_plus=float(me["stuff_plus"]), location_plus=loc_plus, pitching_plus=float(me["pitching_plus"]),
                      probs={e: float(me[f"p_{e}"]) for e in chain.EVENTS}, p_swing=float(me["p_swing"]),
                      p_whiff_swing_model=float(me["p_whiff_swing_model"]), p_whiff_swing_stuff=float(me["p_whiff_swing_stuff"]),
                      p_bip=float(me["p_bip"]), e_delta_bip=float(me["e_delta_bip"]), xrv_per_100=float(me["xrv_per_100"]),
                      p_whiff=float(me["p_whiff"]), contact_damage=float(me["p_bip"] * me["e_delta_bip"]),
                      region=self.region_of(x["plate_x"], x["plate_z"]), by_env=by_env, percentiles=pct, percentiles_tipo=pct_t, warnings=warnings, imputed=imputed)

    # ------------------------------------------------------------------ contexto histórico (observado)
    @staticmethod
    def region_of(plate_x: float, plate_z: float) -> str:
        return str(locp.region(pd.Series([float(plate_x)]), pd.Series([float(plate_z)])).iloc[0])

    def region_context(self, plate_x: float, plate_z: float) -> dict:
        """Cambio observado del valor de la región (Extreme − No Altitude, carreras por 100 pitches; positivo = pierde valor)."""
        reg = self.region_of(plate_x, plate_z)
        r = self.regions
        order = r["extreme_menos_no_por_100"].rank(ascending=False).astype(int)  # 1 = la que más valor pierde
        return {"region": reg, "extreme_menos_no": float(r.loc[reg, "extreme_menos_no_por_100"]), "ee": float(r.loc[reg, "ee_extreme"]),
                "medium_menos_no": float(r.loc[reg, "medium_menos_no_por_100"]), "no_altitude": float(r.loc[reg, "delta_medio_no_altitude_por_100"]),
                "orden_perdida": int(order[reg]), "n_regiones": len(r)}

    def pitch_type_context(self, pitch_type: str) -> dict:
        """Cambio promedio observado del tipo en Extreme vs. No Altitude (efectos fijos de pitcher, bateador, temporada y conteo)."""
        t = self.type_effects.loc[pitch_type]
        ok = self.type_effects[self.type_effects["pitches"] >= self.type_rules["min_pitches_comparacion"]]
        enough = bool(t["pitches"] >= self.type_rules["min_pitches_comparacion"])
        whiff_sig = abs(t["whiff_swing_E"]) > 1.96 * t["whiff_swing_E_se"]
        return {"tipo": pitch_type, "pitches": int(t["pitches"]), "muestra_suficiente": enough,
                "whiff_swing_E": float(t["whiff_swing_E"]), "whiff_swing_E_se": float(t["whiff_swing_E_se"]), "whiff_distinto_de_0": bool(whiff_sig),
                "costo_E": float(t["delta_por_100_E"]), "costo_E_se": float(t["delta_por_100_E_se"]),
                "ivb_E": float(t["ivb_E"]), "hb_abs_E": float(t["hb_abs_E"]),
                "orden_costo": int(ok["delta_por_100_E"].rank().get(pitch_type, 0)) if enough else None, "n_tipos": len(ok)}

    # ------------------------------------------------------------------ Pitching+ general vs. para altitud
    def general_vs_altitude(self, r: Result, alt: str = "extreme") -> dict:
        """El MISMO lanzamiento (mismas probabilidades) valuado con los valores de la liga y con los de `alt`.

        Valores por evento y conteo, calibración del costo de contacto y escala salen de `pitching_env_values.json`
        (paso `pitch_environment` del pipeline). Stuff+ y Location+ no cambian: es el mismo pitch.
        """
        E = self.env_values
        key = f"{r.inputs['balls']}-{r.inputs['strikes']}"
        p = np.array([r.probs[e] for e in chain.EVENTS])
        pbar = np.array([E["probabilidad_media_por_conteo"][key][e] for e in chain.EVENTS])
        bip = [chain.EVENTS.index(e) for e in chain.BIP_CLASSES]
        out = {}
        for k in ("general", alt):
            c = E["conjuntos"][k]
            d = np.array([c["deltas"][key][e] for e in chain.EVENTS])
            sh = c["shift_bip"]
            xrv = float(p @ d + sh * p[bip].sum())
            xbar = float(pbar @ d + sh * pbar[bip].sum())
            s = E["escalas"][k]
            dmg = float(p[bip] @ d[bip] + sh * p[bip].sum())
            term = np.array([c["terminales"][e] for e in chain.BIP_CLASSES])
            # costo de una bola en juego "típica" de este conteo frente a seguir el turno (Δ medio de la rama BIP)
            dbip_tipico = float(pbar[bip] @ d[bip] / pbar[bip].sum() + sh)
            out[k] = {"xrv": xrv, "pitching_plus": float(100 + 10 * (s["media"] - xrv) / s["sd"]),
                      "e_rv_bip": float(p[bip] @ term / p[bip].sum() + sh),  # run value del resultado (sin restar el conteo)
                      "e_delta_bip": dmg / p[bip].sum(), "dano_contacto": dmg, "relativo_conteo": xrv - xbar, "d": d, "shift": sh,
                      "delta_bip_tipico": dbip_tipico}
        groups = E["grupos"]
        dd = out[alt]["d"] - out["general"]["d"]
        contrib = {}
        for g, evs in groups.items():
            idx = [chain.EVENTS.index(e) for e in evs]
            contrib[g] = float(((p - pbar)[idx] * dd[idx]).sum())
        contrib["bola en juego"] += float((p[bip].sum() - pbar[bip].sum()) * (out[alt]["shift"] - out["general"]["shift"]))
        rel_change = -100 * sum(contrib.values())  # + = gana valor relativo frente al típico del conteo (carreras/100 pitches)
        main = max(contrib, key=lambda g: abs(contrib[g]))
        gains = contrib[main] < 0
        label = E["conjuntos"][alt]["etiqueta"]
        pb, pbb = p[bip].sum(), pbar[bip].sum()
        pk = p[[chain.EVENTS.index("whiff"), chain.EVENTS.index("called_strike")]].sum()
        pkb = pbar[[chain.EVENTS.index("whiff"), chain.EVENTS.index("called_strike")]].sum()
        two = r.inputs["strikes"] == 2
        if abs(rel_change) < 0.05:
            text = (f"Casi no cambia: frente al lanzamiento típico de este conteo, su valor relativo en {label} es prácticamente el mismo "
                    f"({rel_change:+.2f} carreras por 100 pitches). El costo distinto de los resultados casi no lo afecta.")
        elif main == "bola en juego":
            rel = "menos" if pb < pbb else "más"
            dif = out[alt]["delta_bip_tipico"] - out["general"]["delta_bip_tipico"]
            costo = "cuesta más" if dif > 0 else "cuesta menos"
            text = (f"{'Gana' if gains else 'Pierde'} valor relativo en {label} sobre todo por la bola en juego. En este conteo, terminar el turno "
                    f"con una bola en juego {costo} en {label} que en la liga, comparado con seguir el turno ({dif:+.3f} carreras): la bola en "
                    f"juego vale más para la ofensiva, pero en {label} seguir el turno también cuesta más. Este pitch tiene {rel} bola en juego "
                    f"({pb:.1%}) que el típico del conteo ({pbb:.1%}); en total, {rel_change:+.2f} carreras por 100 pitches frente al típico.")
        elif main == "ponche / strike":
            why = ("con dos strikes un swing fallado o un strike cantado es ponche, y el ponche vale más en " + label) if two else \
                  "el valor de un strike en este conteo cambia porque el resto del turno cuesta distinto"
            text = (f"{'Gana' if gains else 'Pierde'} valor relativo en {label} sobre todo por sus strikes y swings fallados ({pk:.1%} contra "
                    f"{pkb:.1%} del típico de este conteo): {why} ({rel_change:+.2f} carreras por 100 pitches).")
        else:
            text = (f"{'Gana' if gains else 'Pierde'} valor relativo en {label} sobre todo por el grupo «{main}» "
                    f"({rel_change:+.2f} carreras por 100 pitches frente al típico del conteo).")
        return {"general": {k: v for k, v in out["general"].items() if k not in ("d", "shift")},
                "alt": {k: v for k, v in out[alt].items() if k not in ("d", "shift")}, "alt_label": label,
                "contribuciones": contrib, "cambio_relativo_100": rel_change, "grupo_principal": main, "explicacion": text,
                "p_bip": float(pb), "p_bip_tipico": float(pbb), "p_whiff": float(r.probs["whiff"])}

    # ------------------------------------------------------------------ interpretación
    def baseline(self, x: dict) -> Result:
        """El lanzamiento mediano del mismo tipo, en la misma ubicación, conteo, manos y ambiente."""
        b = self.median_inputs(x["pitch_type"], x["p_throws"], x["b_side"], x["env"])
        b.update({k: x[k] for k in ("plate_x", "plate_z", "balls", "strikes")})
        return self.simulate(b)

    def interpret(self, r: Result, base: Result) -> dict:
        strengths, weaknesses = [], []

        pt = r.inputs["pitch_type"]
        for name, key, txt in (("Stuff+ Physical", "stuff_plus", "características físicas"), ("Location+", "location_plus", "ubicación")):
            v, q = getattr(r, key), r.percentiles_tipo[key]
            if q >= 75:
                strengths.append(f"{name} {v:.0f}: {txt} por encima de la mayoría de los {pt} (percentil {q:.0f} del tipo).")
            elif q <= 25:
                weaknesses.append(f"{name} {v:.0f}: {txt} por debajo de la mayoría de los {pt} (percentil {q:.0f} del tipo).")
        if r.p_whiff_swing_model >= base.p_whiff_swing_model * 1.1:
            strengths.append(f"P(whiff | swing) {r.p_whiff_swing_model:.0%}, por encima del {x_type(r)} mediano en el mismo contexto ({base.p_whiff_swing_model:.0%}).")
        elif r.p_whiff_swing_model <= base.p_whiff_swing_model * 0.9:
            weaknesses.append(f"P(whiff | swing) {r.p_whiff_swing_model:.0%}, por debajo del {x_type(r)} mediano en el mismo contexto ({base.p_whiff_swing_model:.0%}).")
        if r.p_bip <= base.p_bip * 0.9:
            strengths.append(f"Baja probabilidad de bola en juego ({r.p_bip:.1%} vs. {base.p_bip:.1%}).")
        elif r.p_bip >= base.p_bip * 1.1:
            weaknesses.append(f"Probabilidad de bola en juego alta ({r.p_bip:.1%} vs. {base.p_bip:.1%}).")
        if r.e_delta_bip <= base.e_delta_bip - 0.01:
            strengths.append(f"Si hay contacto, el daño esperado es menor que el típico ({r.e_delta_bip:+.3f} vs. {base.e_delta_bip:+.3f} carreras).")
        elif r.e_delta_bip >= base.e_delta_bip + 0.01:
            weaknesses.append(f"Si hay contacto, el daño esperado es mayor que el típico ({r.e_delta_bip:+.3f} vs. {base.e_delta_bip:+.3f} carreras).")
        if r.pitching_plus >= 108 and (r.p_bip <= base.p_bip or r.p_whiff_swing_model >= base.p_whiff_swing_model):
            summary = ("Según el modelo, este lanzamiento combina características que suelen ser más valiosas donde permitir contacto "
                       "cuesta más: más swing-and-miss o menos bolas en juego que el típico de su tipo.")
        elif r.pitching_plus <= 92:
            summary = "Según el modelo, este lanzamiento tiende a permitir más carreras que un lanzamiento promedio en el mismo contexto."
        else:
            summary = "Según el modelo, el valor esperado de este lanzamiento está cerca del promedio de la liga en ese contexto."
        return {"fortalezas": strengths, "debilidades": weaknesses, "interpretacion": summary,
                "vs_mediano": {"pitching_plus": base.pitching_plus, "p_whiff_swing": base.p_whiff_swing_model, "p_bip": base.p_bip,
                               "e_delta_bip": base.e_delta_bip}}

    def improvements(self, inputs: dict) -> pd.DataFrame:
        """Cambios pequeños (una variable a la vez) dentro del rango observado del tipo; Δ Pitching+ estimado."""
        x, _ = self.fill_missing(inputs)
        base = self.simulate(x).pitching_plus
        dom = self.type_domain(x["pitch_type"])
        cands = []
        for k in ("velo", "ivb", "hb_arm", "spin_rate", "extension"):
            c = PHYS_INPUTS[k]
            step = IMPROVE_SD * dom["sd"][c]
            lo, hi = dom["percentiles"][c]["5"], dom["percentiles"][c]["95"]
            for sgn in (+1, -1):
                new = x[k] + sgn * step
                if not (lo <= new <= hi):
                    continue
                cands.append((f"{LABELS[k]} {sgn * step:+.1f}", {**x, k: new}))
        for dx, dz, txt in ((0.25, 0, "0.25 pies hacia el lado + del plato"), (-0.25, 0, "0.25 pies hacia el lado − del plato"),
                            (0, 0.25, "0.25 pies más arriba"), (0, -0.25, "0.25 pies más abajo")):
            nx = {**x, "plate_x": x["plate_x"] + dx, "plate_z": x["plate_z"] + dz}
            cell = f"{int(math.floor(nx['plate_x'] / 0.25))},{int(math.floor(nx['plate_z'] / 0.25))}"
            if dom["celdas_ubicacion"].get(cell, 0) >= LOC_MIN_CELL:
                cands.append((f"ubicación: {txt}", nx))
        rows = []
        for name, cx in cands:
            r = self.simulate(cx)
            rows.append({"cambio": name, "pitching_plus": r.pitching_plus, "delta_pitching_plus": r.pitching_plus - base,
                         "stuff_plus": r.stuff_plus, "location_plus": r.location_plus})
        return pd.DataFrame(rows).sort_values("delta_pitching_plus", ascending=False).reset_index(drop=True)

    def trajectory_paths(self, inputs: dict, n: int = 40) -> pd.DataFrame:
        """Mismo lanzamiento (misma salida: posición y velocidad del ambiente elegido) volado en cada ambiente.

        Las diferencias de trayectoria y de punto de llegada son solo por drag y Magnus. Columnas en pies y, para
        comparar, la diferencia contra No Altitude en pulgadas.
        """
        x, _ = self.fill_missing(inputs)
        _, traj, states = self._features(x, return_states=True)
        ref = states[x["env"]]
        T_ref = float(traj.loc[traj["altitude_category"] == x["env"], "ZoneTime"].iloc[0])
        p, a, tg = ref["pos"], ref["acc"], ref["target"]
        v0 = np.array([(tg[0] - p[0] - 0.5 * a[0] * T_ref ** 2) / T_ref, ref["vy0"], (tg[1] - p[2] - 0.5 * a[2] * T_ref ** 2) / T_ref])
        out = []
        for cat, st in states.items():
            acc = st["acc"]
            T = float(trajectory._fly(p[None, :], v0[None, :], acc[None, :], None)["ZoneTime"][0])
            t = np.linspace(0, T, n)
            pos = p[None, :] + v0[None, :] * t[:, None] + 0.5 * acc[None, :] * t[:, None] ** 2
            out.append(pd.DataFrame({"ambiente": cat, "t": t, "lado": -pos[:, 0], "distancia": pos[:, 1], "altura": pos[:, 2]}))
        df = pd.concat(out, ignore_index=True)
        base = df[df["ambiente"] == "No Altitude"].reset_index(drop=True)
        df["dif_altura_pulg"] = 12 * (df["altura"].to_numpy() - np.tile(base["altura"].to_numpy(), 3))
        df["dif_lado_pulg"] = 12 * (df["lado"].to_numpy() - np.tile(base["lado"].to_numpy(), 3))
        return df

    @staticmethod
    def location_grid(step: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Rejilla fija de la zona (el orden de los puntos es el índice que devuelve la selección de la gráfica)."""
        xs = np.round(np.arange(-1.8, 1.8001, step), 2)
        zs = np.round(np.arange(0.6, 4.4001, step), 2)
        gx, gz = np.meshgrid(xs, zs)
        return gx.ravel(), gz.ravel()

    def location_surface(self, inputs: dict, step: float = 0.2) -> pd.DataFrame:
        """Location+ en una rejilla para el contexto del lanzamiento (tipo, manos, conteo)."""
        x, _ = self.fill_missing(inputs)
        same = "same" if x["p_throws"] == x["b_side"] else "opposite"
        gx, gz = self.location_grid(step)
        loc = self.loc
        bat = x["b_side"]
        row = pd.DataFrame({"loc_in": gx.ravel() * (-1.0 if bat == "Left" else 1.0), "PlateLocHeight": gz.ravel(),
                            "Balls": x["balls"], "Strikes": x["strikes"]})
        for c, v in (("AutoPitchType", x["pitch_type"]), ("same_side", same), ("bat_side", bat)):
            row[c] = pd.Categorical([v] * len(row), categories=loc["categories"][c])
        X = row[loc["numeric_features"] + loc["categorical_features"]]
        pr = {}
        for name, _, _, n_classes in chain.SUBMODELS:
            p = loc["models"][name].predict_proba(X)
            pr[name] = p if n_classes > 2 else p[:, 1:2]
        D = np.tile(loc["deltas_by_count"].loc[(x["balls"], x["strikes"])].to_numpy(), (len(row), 1))
        xrv = chain.expected_run_value(chain.event_probabilities(pr), D)
        ref = loc["reference_by_context"][(x["pitch_type"], same, bat, x["balls"], x["strikes"])]
        dom = self.type_domain(x["pitch_type"])
        support = [dom["celdas_ubicacion"].get(f"{int(math.floor(a / 0.25))},{int(math.floor(b / 0.25))}", 0) for a, b in zip(gx.ravel(), gz.ravel())]
        return pd.DataFrame({"plate_x": gx.ravel(), "plate_z": gz.ravel(),
                             "location_plus": [self._plus("location", v - ref) for v in xrv], "pitches_en_celda": support})


def x_type(r: Result) -> str:
    return r.inputs["pitch_type"]
