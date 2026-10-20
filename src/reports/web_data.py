"""Datos de la página web que no estaban en `data/web/` (paso `web_data` del pipeline).

Todo sale de outputs reproducibles del pipeline; ninguna cifra se escribe a mano:

- `summary.json`: las cifras que citan los textos de la página.
- `simulator_scales.json`: μ y τ de Stuff+ Physical, Location+ y Pitching+ (para poner un lanzamiento
  individual en la misma escala que los pitchers), desplazamientos por ambiente y percentiles de los
  puntajes de los lanzamientos reales.
- `simulator_domain.json`: soporte de los datos por tipo de pitch (percentiles, media, covarianza,
  cuantiles de la distancia de Mahalanobis, conteo de pitches por celda de ubicación) para las reglas
  de extrapolación del simulador.
- `pitcher_pitch_types.csv`: uso y calidad de cada tipo de pitch de los pitchers del ranking.
- `anon_stadiums.csv`: ratios físicos de cada estadio anónimo.

Convención de la página: movimiento horizontal y lado de salida en "lado del brazo" (positivo hacia el
brazo del pitcher; los zurdos se espejan), como en `stuff_plus_v2.MIRROR_SIGN_COLUMNS`.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import p2_common as pc
from src.analysis import pitcher_profiles as pp
from src.data import parks
from src.models import location_plus_v2 as locp
from src.models import pitching_plus_neutral as pn
from src.models import run_value_chain as chain
from src.models import stuff_plus_physical as spp
from src.models import stuff_plus_v2

ROOT = Path(__file__).resolve().parents[2]
WEB = pc.WEB_DIR
DOMAIN_COLS = ["RelSpeed", "InducedVertBreak", "hb_arm", "SpinRate", "Extension", "RelHeight", "rel_side_arm"]
PCTS = [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5]
CELL = 0.25
MIN_PITCHES_TYPE = 50

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _eb_params(d: pd.DataFrame, col: str) -> dict:
    """μ y τ del shrinkage con todas las temporadas juntas (empirical_bayes trata menor = mejor)."""
    _, params = pn.empirical_bayes(d.assign(year=0), col)
    return {"mu": params[0]["mu"], "tau": params[0]["tau"]}


def hand_sign(throws: pd.Series) -> np.ndarray:
    return np.where(throws.astype("object") == "Left", -1.0, 1.0)


# ---------------------------------------------------------------------------
# Escalas del simulador (con los modelos FINALES, que son los que usa la página)
# ---------------------------------------------------------------------------


def scales() -> dict:
    df = chain.load_data()
    frame = pc.load_pitch_frame()[["PitchUID", "env", "altitude_category"]]
    df = df.merge(frame, on="PitchUID", how="left")

    with open(chain.FINAL_MODEL_PATHS["pitching"], "rb") as fh:
        pit = pickle.load(fh)
    pred = chain.predict_expected_run_value(pit, df)
    xrv = pred["xrv"].to_numpy()
    with open(spp.FINAL_PATH, "rb") as fh:
        stf = pickle.load(fh)
    p_st = spp.marginal_predict(stf, df)

    # Location+: modelo final y referencia por contexto
    ld = locp.load_data()
    with open(locp.FINAL_PATH, "rb") as fh:
        loc = pickle.load(fh)
    X = ld[loc["numeric_features"] + loc["categorical_features"]]
    pr = {}
    for name, _, _, n_classes in chain.SUBMODELS:
        p = loc["models"][name].predict_proba(X)
        pr[name] = p if n_classes > 2 else p[:, 1:2]
    D = loc["deltas_by_count"]
    pos = {bs: i for i, bs in enumerate(D.index)}
    Dm = D.to_numpy()[[pos[bs] for bs in zip(ld["Balls"], ld["Strikes"])]]
    xrv_loc = chain.expected_run_value(chain.event_probabilities(pr), Dm)
    key = list(zip(ld["AutoPitchType"].astype(str), ld["same_side"].astype(str), ld["bat_side"].astype(str), ld["Balls"], ld["Strikes"]))
    loc_value = xrv_loc - np.array([loc["reference_by_context"][k] for k in key])

    # Ajuste por ambiente a nivel de categoría (el oficial usa temporada × estadio; el simulador solo conoce la categoría)
    cat = df["altitude_category"]
    st_off = {c: float(np.mean(p_st) - p_st[cat == c].mean()) for c in pc.CATS}
    pit_off = {c: float(-xrv[cat.to_numpy() == c].mean()) for c in pc.CATS}

    # μ y τ: los del índice publicado (OOF, todas las temporadas), para que la escala sea la misma que la de los pitchers
    st_oof = pd.read_parquet(spp.OOF_PATH.with_name("stuff_plus_physical_pitches.parquet"))
    st_par = _eb_params(st_oof.assign(neg=-st_oof["p_whiff_swing_adj"]), "neg")
    st_par["mu"] = -st_par["mu"]
    lo_oof = pd.read_parquet(locp.PITCHES_PATH, columns=["pitcher_anon_id", "year", "loc_value"])
    lo_par = _eb_params(lo_oof, "loc_value")
    pn_oof = pd.read_parquet(pn.OOF_PATH, columns=["pitcher_anon_id", "year", "xrv_obs", "altitude_category", "stadium_id"])
    pn_oof["xrv_m1b"] = pn_oof["xrv_obs"] - pn_oof.groupby([pn_oof["year"], pn_oof["stadium_id"].fillna(pn_oof["altitude_category"])])["xrv_obs"].transform("mean")
    pi_par = _eb_params(pn_oof, "xrv_m1b")

    # Escala de LANZAMIENTO: 100 = lanzamiento promedio de la liga, 10 = 1 sd entre lanzamientos (valores ajustados por ambiente).
    # La escala de PITCHER (μ, τ del shrinkage) daría a un lanzamiento suelto valores fuera de toda proporción, porque un
    # lanzamiento varía mucho más que el promedio de un pitcher; se guarda solo como referencia.
    v_st = p_st + cat.map(st_off).to_numpy()
    v_pi = xrv + cat.map(pit_off).to_numpy()
    out = {}
    for name, v, higher, par, extra in (("stuff", v_st, True, st_par, {"offset_por_ambiente": st_off}),
                                        ("location", loc_value, False, lo_par, {}),
                                        ("pitching", v_pi, False, pi_par, {"offset_por_ambiente": pit_off})):
        m, sd = float(np.mean(v)), float(np.std(v))
        plus = 100 + 10 * (1 if higher else -1) * (v - m) / sd
        types = (ld if name == "location" else df)["AutoPitchType"].astype(str).to_numpy()
        by_type = {t: {str(q): float(np.percentile(plus[types == t], q)) for q in range(1, 100)} for t in pc.MAIN_TYPES}
        out[name] = {"media_lanzamiento": m, "sd_lanzamiento": sd, "mayor_es_mejor": higher, **extra,
                     "escala_pitcher": par, "percentiles_lanzamientos": {str(q): float(np.percentile(plus, q)) for q in range(1, 100)},
                     "percentiles_por_tipo": by_type}
    out["liga"] = {"p_whiff_swing_media": float(np.mean(p_st)), "xrv_media": float(xrv.mean())}
    return out


# ---------------------------------------------------------------------------
# Dominio de los datos para el simulador
# ---------------------------------------------------------------------------


def domain() -> dict:
    from dashboard.scoring import estimate_spin_axis

    d = pd.read_parquet(pc.ARSENAL_PATH, columns=["AutoPitchType", "PitcherThrows", "RelSpeed", "InducedVertBreak", "HorzBreak",
                                                  "SpinRate", "SpinAxis", "Extension", "RelHeight", "RelSide", "PlateLocSide", "PlateLocHeight"])
    d["AutoPitchType"] = d["AutoPitchType"].astype("object")
    d = d[d["AutoPitchType"].isin(pc.MAIN_TYPES) & d["PitcherThrows"].isin(["Right", "Left"])].copy()
    sign = hand_sign(d["PitcherThrows"])
    d["hb_arm"] = d["HorzBreak"] * sign
    d["rel_side_arm"] = d["RelSide"] * sign
    axis_r = np.where(sign < 0, 360 - d["SpinAxis"], d["SpinAxis"])  # eje en el marco de un derecho
    implied = estimate_spin_axis(d["InducedVertBreak"].to_numpy(), d["hb_arm"].to_numpy())
    d["axis_dev"] = (axis_r - implied + 180) % 360 - 180
    out = {"columnas": DOMAIN_COLS, "celda_ubicacion_pies": CELL, "tipos": {}}
    for t, g in d.groupby("AutoPitchType"):
        g = g.dropna(subset=DOMAIN_COLS)
        X = g[DOMAIN_COLS].to_numpy(float)
        mu, cov = X.mean(0), np.cov(X, rowvar=False)
        inv = np.linalg.inv(cov)
        diff = X - mu
        dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
        loc = g.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        cells = (np.floor(loc["PlateLocSide"] / CELL).astype(int).astype(str) + "," + np.floor(loc["PlateLocHeight"] / CELL).astype(int).astype(str))
        out["tipos"][t] = {
            "n": int(len(g)),
            "percentiles": {c: {str(q): float(np.percentile(g[c], q)) for q in PCTS} for c in DOMAIN_COLS},
            "sd": {c: float(g[c].std()) for c in DOMAIN_COLS},
            "media": mu.tolist(), "cov_inv": inv.tolist(),
            "mahalanobis_cuantiles": {str(q): float(np.percentile(dist, q)) for q in (50, 90, 95, 99, 99.9)},
            "desvio_eje_abs_cuantiles": {str(q): float(np.nanpercentile(g["axis_dev"].abs(), q)) for q in (50, 90, 95, 99)},
            "celdas_ubicacion": cells.value_counts().to_dict(),
        }
    return out


# ---------------------------------------------------------------------------
# Pitch mix por pitcher
# ---------------------------------------------------------------------------


def pitcher_pitch_types(eligible: pd.Series) -> pd.DataFrame:
    st = pd.read_parquet(spp.OOF_PATH.with_name("stuff_plus_physical_pitches.parquet"), columns=["PitchUID", "pitcher_anon_id", "p_whiff_swing_adj"])
    lo = pd.read_parquet(locp.PITCHES_PATH, columns=["PitchUID", "loc_value"])
    f = pc.load_pitch_frame()[["PitchUID", "AutoPitchType", "is_swing", "is_whiff"]]
    d = st.merge(lo, on="PitchUID", how="left").merge(f, on="PitchUID", how="left")
    d["AutoPitchType"] = d["AutoPitchType"].where(d["AutoPitchType"].isin(pc.MAIN_TYPES), "Otro")
    g = d.groupby(["pitcher_anon_id", "AutoPitchType"]).agg(n=("PitchUID", "size"), p_whiff_swing=("p_whiff_swing_adj", "mean"),
                                                            loc_value=("loc_value", "mean"), swings=("is_swing", "sum"),
                                                            whiffs=("is_whiff", "sum")).reset_index()
    g["uso"] = g["n"] / g.groupby("pitcher_anon_id")["n"].transform("sum")
    g["whiff_swing"] = g["whiffs"] / g["swings"].where(g["swings"] > 0)
    # Escala por tipo: 100 = promedio de ese tipo en la liga; 10 = 1 sd entre pitchers (con ≥ 50 pitches del tipo)
    for col, out, higher in (("p_whiff_swing", "stuff_plus_tipo", True), ("loc_value", "location_plus_tipo", False)):
        ref = g[g["n"] >= MIN_PITCHES_TYPE].groupby("AutoPitchType")[col].agg(["mean", "std"])
        z = (g[col] - g["AutoPitchType"].map(ref["mean"])) / g["AutoPitchType"].map(ref["std"])
        g[out] = 100 + 10 * (z if higher else -z)
    g["muestra_suficiente"] = g["n"] >= MIN_PITCHES_TYPE
    return g[g["pitcher_anon_id"].isin(eligible)].drop(columns=["whiffs"]).sort_values(["pitcher_anon_id", "n"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Resumen de cifras
# ---------------------------------------------------------------------------


def summary(res: dict) -> dict:
    hyp = res["hyp"]
    rows = {r["variable"]: r for r in hyp["h1_h2"]}
    h3 = hyp["h3"]
    cons = res["consistencia"]
    lw = res["pesos"]
    dmg = res["dano"]
    dcat = dmg["tabla"][dmg["tabla"]["nivel"] == "categoria"].set_index("ambiente")
    dec = dmg["descomposicion"]
    e = res["pitchers"][res["elegibles"].to_numpy()]
    val = res["validacion"][100]
    sens = res["sensibilidad"]
    s_tab = sens["tabla"]
    stuff = json.loads(spp.JSON_PATH.read_text(encoding="utf-8"))
    loc = json.loads(locp.JSON_PATH.read_text(encoding="utf-8"))
    st_json = json.loads(parks.JSON_PATH.read_text(encoding="utf-8"))
    chain_meta = json.loads(chain.OOF_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    arch = res["arquetipos"]
    gN = h3["por_categoria"]["No Altitude"]["G"]

    def eff(label):
        fx = rows[label]["efectos_fijos"]
        return {"no_altitude": fx["media_no_altitude"], "medium": fx["beta_medium"],
                "medium_ic95": [fx["beta_medium"] - 1.96 * fx["se_medium"], fx["beta_medium"] + 1.96 * fx["se_medium"]],
                "extreme": fx["beta_extreme"], "extreme_ic95": fx["ic95_extreme"], "crudo_extreme": rows[label]["crudo"]["beta_extreme"],
                "consistencia_extreme": [cons["Extreme Altitude"][label]["en_direccion"], cons["Extreme Altitude"][label]["total"]],
                "consistencia_medium": [cons["Medium Altitude"][label]["en_direccion"], cons["Medium Altitude"][label]["total"]]}

    het = {r["categoria"]: r for r in st_json["heterogeneidad"]["drag"]}
    hetm = {r["categoria"]: r for r in st_json["heterogeneidad"]["magnus"]}
    y100 = stuff["yoy"]["100"]["res"]
    return {
        "efectos": {"whiff_swing": eff("whiff por swing"), "k_pa": eff("ponche por turno"), "bip_swing": eff("bola en juego por swing"),
                    "rv_bip": eff("run value RE24 por bola en juego")},
        "G": {c: {"G": h3["por_categoria"][c]["G"], "ic95": h3["por_categoria"][c]["ic95_G"]} for c in pc.CATS},
        "G_extreme_menos_no": {"valor": h3["extreme_menos_no"], "ic95": h3["ic95"]},
        "G_medium_menos_no": {"valor": h3["medium_menos_no"], "ic95": h3["ic95_medium"]},
        "G_aumento_relativo_extreme": h3["por_categoria"]["Extreme Altitude"]["G"] / gN - 1,
        "G_por_estadio_temporada": {c: {"mayor_que_no": int(sum(v > gN for v in cons[c]["G"].values())), "total": len(cons[c]["G"]),
                                        "valores": cons[c]["G"]} for c in ["Medium Altitude", "Extreme Altitude"]},
        "consistencia_detalle": {c: {k: {"betas": {p: x["beta"] for p, x in v["por_estadio_temporada"].items()}}
                                     for k, v in cons[c].items() if k != "G"} for c in ["Medium Altitude", "Extreme Altitude"]},
        "dano": {"por_100_no": 100 * dcat.loc["No Altitude", "dano_bip_por_pitch"], "por_100_medium": 100 * dcat.loc["Medium", "dano_bip_por_pitch"],
                 "por_100_extreme": 100 * dcat.loc["Extreme", "dano_bip_por_pitch"],
                 "fraccion_por_dano_extreme": dec["Extreme"]["por_dano"] / dec["Extreme"]["total"],
                 "fraccion_por_probabilidad_extreme": dec["Extreme"]["por_probabilidad"] / dec["Extreme"]["total"],
                 "fraccion_interaccion_extreme": dec["Extreme"]["interaccion"] / dec["Extreme"]["total"],
                 "descomposicion": dec},
        "pesos": {"k_menos_bip": lw["K_menos_BIP"].to_dict(), "bb_menos_bip": lw["BB_menos_BIP"].to_dict(),
                  "k_vs_bip_aumento_extreme": lw.loc["Extreme", "K_menos_BIP"] / lw.loc["No Altitude", "K_menos_BIP"] - 1},
        "hbp_run_value": float(chain_meta["terminal_values"]["hbp"]),
        "valores_terminales": chain_meta["terminal_values"],
        "stuff": {"eta2_ubicacion_physical": stuff["ubicacion_eta2"]["Stuff+ Physical"], "eta2_ubicacion_v1": stuff["ubicacion_eta2"]["Stuff+ v1"],
                  "eta2_ubicacion_v4": stuff["ubicacion_eta2"]["Stuff+ v4"],
                  "estabilidad": {k: y100[k]["estabilidad"]["r"] for k in ["stuff_plus_v1", "stuff_plus_v4", "stuff_plus_physical"]},
                  "whiff_t1": {k: y100[k]["whiff_swing_t1"]["r"] for k in ["stuff_plus_v1", "stuff_plus_v4", "stuff_plus_physical"]},
                  "delta_vs_v4_whiff": y100["stuff_plus_physical"]["delta_vs_v4_whiff"],
                  "auc": {k: v["auc_swings"] for k, v in stuff["pitch"].items()}, "r_oficial_neutral": stuff["r_oficial_neutral"]},
        "location": {"r_stuff": loc["validacion"]["r_stuff_physical"], "estabilidad": loc["validacion"]["100"]["estabilidad"]["r"],
                     "bb_t1": loc["validacion"]["100"]["bb_t1"]["r"], "bb_propio_t1": loc["validacion"]["100"]["bb_propio_t1"]["r"],
                     "regiones": loc["regiones"]},
        "aepp": {"r_altitud_t1": val["aepp"]["carreras_altitud_t1"]["r"], "r_altitud_t1_ic95": val["aepp"]["carreras_altitud_t1"]["ci95"],
                 "r_pitching_plus_altitud_t1": val["pitching_plus"]["carreras_altitud_t1"]["r"],
                 "p_mejor_que_pitching_plus": val["aepp"]["delta_altitud_vs_pitching_plus"]["p_mejor"],
                 "r_kbb_altitud_t1": val["k_bb"]["carreras_altitud_t1"]["r"],
                 "delta_vs_kbb": val["aepp"]["carreras_altitud_t1"]["r"] - val["k_bb"]["carreras_altitud_t1"]["r"],
                 "estabilidad": val["aepp"]["estabilidad"]["r"],
                 "rho_pesos_altitud_vs_no": float(e["aepp"].corr(e["aepp_no_altitude"], method="spearman")),
                 "r_bono_k": float(e["bono_altitud"].corr(e["k_pct"])), "r_bono_bb": float(e["bono_altitud"].corr(e["bb_pct"])),
                 "n_elegibles": int(len(e)), "min_pitches": pp.MIN_PITCHES_RANK, "min_pa": pp.MIN_PA_RANK,
                 "ancho_mediano_rango_80": float((e["rank_hi80"] - e["rank_lo80"]).median())},
        "sensibilidad": {"confiabilidad": sens["info"]["confiabilidad_spearman_brown"], "n_pitchers": sens["info"]["n_pitchers"],
                         "r_mod_emp": sens["r_mod_emp"], "pitches_para_0_5": sens["pitches_necesarios"].get(0.5),
                         "pitches_para_0_7": sens["pitches_necesarios"].get(0.7), "mediana_pitches_extreme": float(s_tab["n_extreme"].median()),
                         "intervalos_cruzan_0": int(((s_tab["s_shrunk_lo80"] < 0) & (s_tab["s_shrunk_hi80"] > 0)).sum()),
                         "r_rivales": sens["factores"]["rivales_E_menos_N"]["r_con_s_emp"]},
        "validacion_perfil": {c: {"etiqueta": lab, "estabilidad": val[c]["estabilidad"]["r"], "r_carreras_t1": val[c]["carreras_t1"]["r"],
                                  "r_altitud_t1": val[c]["carreras_altitud_t1"]["r"], "r_altitud_t1_ic95": val[c]["carreras_altitud_t1"]["ci95"]}
                              for c, lab in res["validacion"]["candidatos"].items()},
        "n_pares_validacion": {"todos": val["n_pares"], "altitud": val["n_pares_altitud"]},
        "estabilidad_metricas": res["estabilidad_metricas"].to_dict("records"),
        "arquetipos": {"silueta": arch["silueta"], "k": arch["k"], "ari": arch["ari_estabilidad"], "nombres_validos": arch["nombres_validos"]},
        "estadios": {**{k: v for k, v in st_json.items() if k != "heterogeneidad"},
                     "drag_extreme": [het["Extreme Altitude"]["min"], het["Extreme Altitude"]["max"]],
                     "magnus_extreme": [hetm["Extreme Altitude"]["min"], hetm["Extreme Altitude"]["max"]],
                     "drag_medium": [het["Medium Altitude"]["min"], het["Medium Altitude"]["max"]],
                     "I2_drag_extreme": het["Extreme Altitude"]["I2"], "I2_magnus_extreme": hetm["Extreme Altitude"]["I2"]},
        "validacion_temporal_p0": {mn: {k: v["r"] for k, v in blk["r"].items()} for mn, blk in json.loads(
            (ROOT / "reports" / "benchmark" / "validacion_temporal.json").read_text(encoding="utf-8"))["secondary_posthoc"].items()},
        # ---- ampliación de la página (prompt complementario) ----
        "valor_bip": {a: float(lw.loc[a, "v_BIP"]) for a in ["No Altitude", "Medium", "Extreme"]},
        "ahorro_rodado": {a: float(lw.loc[a, "v_aire"] - lw.loc[a, "v_GB"]) for a in ["No Altitude", "Medium", "Extreme"]},
        "stuff_versiones": {lab: {"r_pct_extreme": stuff["pitcher"][k]["r_pct_extreme"], "estabilidad": y100[k]["estabilidad"]["r"],
                                  "whiff_t1": y100[k]["whiff_swing_t1"]["r"]}
                            for lab, k in (("observado", "stuff_plus_physical_observado"), ("oficial", "stuff_plus_physical"),
                                           ("neutral", "stuff_plus_physical_neutral"))},
        "sensibilidad_extra": {"r_mod_emp_spearman": sens["r_mod_emp_spearman"],
                               "techo_correlacion": float(np.sqrt(sens["info"]["confiabilidad_spearman_brown"]))},
        "tipos_pitch": pitch_type_summary(res["tipos"]),
        "contrafactual_split_half": split_half(),
        "temporadas": seasons(),
        "ranking_top": {"lugar_1_intervalo": [float(e.loc[e["rank"] == 1, "rank_lo80"].iloc[0]), float(e.loc[e["rank"] == 1, "rank_hi80"].iloc[0])],
                        "n_p_top25_mayor_50": int((e["p_top25"] > 0.5).sum()), "simulaciones": pp.N_DRAWS},
    }


MIN_PITCHES_TYPE_COMPARE = 20_000  # tipos con muestra razonable para comparar costos (el splitter tiene ~7 mil)


def pitch_type_summary(t: pd.DataFrame) -> dict:
    """Hallazgos descriptivos por tipo (efectos con efectos fijos de pitcher, bateador, temporada y conteo)."""
    ok = t[t["pitches"] >= MIN_PITCHES_TYPE_COMPARE]
    small = t[t["pitches"] < MIN_PITCHES_TYPE_COMPARE]
    return {"min_pitches_comparacion": MIN_PITCHES_TYPE_COMPARE,
            "menor_costo_extreme": ok.loc[ok["delta_por_100_E"].idxmin(), "tipo"],
            "mayor_costo_extreme": ok.loc[ok["delta_por_100_E"].idxmax(), "tipo"],
            "mayor_perdida_whiff": ok.loc[ok["whiff_swing_E"].idxmin(), "tipo"],
            "menor_cambio_whiff": ok.loc[ok["whiff_swing_E"].abs().idxmin(), "tipo"],
            "mayor_perdida_hb": ok.sort_values("hb_abs_E")["tipo"].head(2).tolist(),
            "mayor_perdida_ivb": ok.loc[ok["ivb_E"].idxmin(), "tipo"],
            "mayor_aumento_rv_bip": ok.loc[ok["rv_bip_E"].idxmax(), "tipo"],
            "muestra_pequena": small["tipo"].tolist()}


def split_half() -> dict:
    """Validación split-half del contrafactual físico (P0): cambio observado vs. predicho, mediana entre grupos."""
    v = json.loads((ROOT / "reports" / "benchmark" / "altitude_counterfactual_v2.json").read_text(encoding="utf-8"))["validation"]
    out = {}
    for r in v:
        if r["categoria"] == "Extreme Altitude":
            out[r["feature"]] = {"observado": r["cambio_observado_mediana"], "predicho": r["cambio_predicho_mediana"],
                                 "corr_entre_grupos": r["corr_entre_grupos"]}
    ivb = out["InducedVertBreak"]
    return {"extreme": out, "sobreprediccion_ivb": ivb["predicho"] / ivb["observado"] - 1}


def seasons() -> dict:
    """Pitches por temporada (2026 parcial) y contacto duro por temporada (calibración del Trackman)."""
    f = pc.load_pitch_frame()
    g = f.groupby("year")
    return {"pitches": {str(k): int(v) for k, v in g.size().items()},
            "juegos": {str(k): int(v) for k, v in g["game_anon_id"].nunique().items()},
            "hard_pct_bip": {str(k): float(v) for k, v in g["is_hard"].mean().items()},
            "ev_media_bip": {str(k): float(v) for k, v in f[f["is_bip"] == 1].groupby("year")["ExitSpeed"].mean().items()},
            "pares_validacion": [f"{a}→{b}" for a, b in zip(sorted(g.groups)[:-1], sorted(g.groups)[1:])]}


def run() -> None:
    res = pp.run()
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "summary.json").write_text(json.dumps(summary(res), indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    elig = res["pitchers"].loc[res["elegibles"].to_numpy(), "pitcher_anon_id"]
    pc.write_web_table(pitcher_pitch_types(elig), "pitcher_pitch_types")
    r = parks.load_stadium_ratios().reset_index()
    cols = ["stadium_id", "categoria", "celdas", "drag_ratio", "drag_lo", "drag_hi", "magnus_ratio", "magnus_lo", "magnus_hi"]
    pc.write_web_table(r[[c for c in cols if c in r.columns]].sort_values(["categoria", "drag_ratio"]), "anon_stadiums")
    (WEB / "simulator_domain.json").write_text(json.dumps(domain(), default=float), encoding="utf-8")
    (WEB / "simulator_scales.json").write_text(json.dumps(scales(), indent=2, default=float), encoding="utf-8")
    files = {
        "summary.json": "cifras que citan los textos de la página (efectos, consistencia, daño, pesos, validación, sensibilidad, estadios)",
        "simulator_scales.json": "escalas de lanzamiento (media, sd, percentiles) y ajuste por ambiente del simulador",
        "simulator_domain.json": "soporte de los datos por tipo de pitch (percentiles, Mahalanobis, celdas de ubicación)",
        "pitcher_pitch_types.csv": "uso, Stuff+ y Location+ por tipo (100 = promedio del tipo) y whiff/swing de cada pitcher del ranking",
        "anon_stadiums.csv": "ratios de drag y Magnus de cada estadio anónimo (IC 95%)",
        "pitcher_altitude_profiles.csv": "ficha completa de los pitchers elegibles (P2)",
        "pitcher_rankings.csv": "ranking AEPP (subconjunto de la ficha, para descargar)",
        "pitcher_sensitivity.csv": "sensibilidad observada — exploratoria, con advertencia en cada fila",
        "stuff_plus_physical.csv": "Stuff+ Physical por pitcher-temporada",
        "location_plus.csv": "Location+ por pitcher-temporada",
        "location_plus_heatmap.csv": "mapas de calor de Location+ (tipo × enfrentamiento × grupo de conteo)",
        "pitch_type_altitude_effects.csv": "efecto de Extreme y Medium por tipo de pitch",
        "contact_damage_by_environment.csv": "P(BIP), E[RV|BIP] y daño por ambiente y por estadio anónimo",
        "linear_weights_by_environment.csv": "valor RE24 de cada resultado del turno por ambiente",
        "characteristic_value_by_environment.csv": "valor de cada característica por ambiente (diseño cruzado)",
    }
    write_readme(files, res["sensibilidad"]["info"]["confiabilidad_spearman_brown"])
    logger.info("Datos de la web en %s", WEB)


# Tablas del paso pitch_environment (extensión: Pitching+ general vs. altitud y óptimo por ambiente)
EXTENSION_FILES = {
    "pitching_env_values.json": "Δ(evento, conteo), valores terminales, calibración del costo de contacto y escalas: liga, No Altitude, Medium y Extreme",
    "pitching_general_vs_altitude.csv": "Pitching+ general vs. Extreme por tipo de pitch y por decil de P(BIP)",
    "pitching_relative_gain_by_whiff.csv": "ganancia de valor relativo en Extreme frente al típico del conteo, por quintil de P(whiff)",
    "optimal_pitch_profiles.csv": "perfil óptimo (top 1%) y mejor observado por tipo, objetivo y conteo",
    "optimal_pitch_comparison.csv": "cuánto cambia el óptimo entre objetivos (distancia, ruido bootstrap, rangos, top 1% compartido)",
    "optimal_pitch_location.csv": "mejor combinación física + ubicación por tipo, enfrentamiento, conteo y objetivo",
    "optimal_location_heatmap.csv": "Pitching+ por ubicación para un lanzamiento fijo de muy buen Stuff+ (general vs. Extreme)",
    "optimal_location_fixed_pitch.csv": "el lanzamiento fijo de cada tipo usado en el mapa de ubicaciones",
    "optimal_pitch_meta.json": "parámetros del análisis del óptimo (contextos, umbrales, rangos de los candidatos)",
    "validation_holdout.json": "validación fuera de muestra del modelo final: pitcher, season y park holdout contra el modelo nulo",
    "pitch_shape_altitude_effects.csv": "cambio en Extreme por forma de pitch: Stuff+ observado, movimiento, whiff, contacto duro, HR/BIP y valor",
    "movement_map.csv": "mapa de movimiento por pitcher × tipo (No Altitude y Extreme), Stuff+ del tipo y grupo más/menos efectivo",
    "arsenal_altitude.csv": "arsenales (tipos con uso ≥ 10%): Stuff+ con pitches de Extreme, AEPP, K% y whiff",
    "prospect_reference.json": "percentiles de referencia y parámetros de la calculadora de AEPP para prospectos",
}


def write_readme(files: dict | None = None, rel: float | None = None) -> None:
    """README de data/web con todas las tablas presentes (lo llaman web_data y pitch_environment)."""
    if files is None or rel is None:
        S = json.loads((WEB / "summary.json").read_text(encoding="utf-8"))
        rel = S["sensibilidad"]["confiabilidad"]
        prev = (WEB / "README.md").read_text(encoding="utf-8") if (WEB / "README.md").exists() else ""
        files = {}
        for line in prev.splitlines():
            if line.startswith("| `"):
                k, v = line[3:].split("` | ", 1)
                files[k] = v.rstrip(" |")
    files = {**files, **EXTENSION_FILES}
    lines = ["# Datos de la página (data/web/)", "",
             "Generados por el pipeline (`src/reports/p2_report.py`, `src/reports/web_data.py` y `src/analysis/pitch_environment.py`); los usa "
             "`web/app.py`. Identificadores anónimos. No editar a mano.", "", "| archivo | contenido |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in files.items() if (WEB / k).exists()]
    lines += ["", f"La sensibilidad observada es exploratoria (confiabilidad ≈ {rel:.2f}). El contrafactual físico es explicativo: no mostró "
              "ventaja predictiva frente a la normalización simple por estadio anónimo.", ""]
    (WEB / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
