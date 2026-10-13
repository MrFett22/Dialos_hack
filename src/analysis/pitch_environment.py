"""Extensión de la página: Pitching+ general vs. para altitud, y el lanzamiento óptimo por ambiente.

No cambia ningún índice oficial. Usa los modelos FINALES (Pitching+ v3, Stuff+ Physical, Location+ v2) y la misma
recursión RE24 por conteo de `run_value_chain.count_values`.

Parte B — ¿qué cambia en nuestro Pitching+?
    Pitching+ general  = valores Δ(evento, conteo) calculados con TODA la liga (son los del Pitching+ oficial; se
                         verifica numéricamente contra el artefacto).
    Pitching+ Extreme  = las mismas probabilidades del modelo (mismo pitch, misma física, misma ubicación, mismo
                         conteo) valuadas con Δ(evento, conteo) calculados SOLO con los pitches de Extreme: valores
                         terminales (promedio del run value RE24 de cada resultado) y frecuencias de transición del
                         conteo observadas en ese ambiente. No hay pesos ni bonos manuales.
    Escala: 100 + 10·(media − E[Δ])/sd, con media y sd de TODOS los lanzamientos reales de la liga valuados con cada
    juego de valores (sin desplazamiento por ambiente), para que la única diferencia sea el costo de los resultados.

Parte C — ¿es el mismo un buen lanzamiento en cualquier ambiente?
    Candidatos = lanzamientos REALES de cada tipo lanzados en No Altitude (marco de un derecho), así que todo
    óptimo está dentro del soporte observado y respeta la correlación entre variables. Se evalúan en contextos
    fijos (conteo 0-0, 0-2 y 2-0; mismo lado y lado contrario ponderados por la mezcla de la liga):
      - Stuff+ Physical;
      - Pitching+ general;
      - Pitching+ Extreme (solo costos: misma física, valores de Extreme) — capa validada;
      - Pitching+ Extreme con física aproximada (el mismo pitch re-simulado en Extreme con el contrafactual físico y
        valuado con valores de Extreme) — capa explicativa (sobreestima ≈ 20% el cambio de IVB).
    Experimento 1: física sola, ubicación típica del tipo. Experimento 2: física + ubicación sobre una rejilla con
    soporte real. El "óptimo" se reporta como (a) el mejor lanzamiento observado y (b) el perfil del top 1%
    (mediana, p10–p90), puntuado con sus vecinos reales más cercanos; la incertidumbre sale de un bootstrap por pitcher.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard import scoring
from src.analysis import altitude_counterfactual_v2 as cf
from src.analysis import p2_common as pc
from src.features import trajectory
from src.models import location_plus_v2 as locp
from src.models import run_value_chain as chain
from src.models import stuff_plus_physical as spp
from src.models import stuff_plus_v2

ROOT = Path(__file__).resolve().parents[2]
WEB = pc.WEB_DIR
REPORTS = ROOT / "reports"
RESULTS_PATH = ROOT / "data" / "processed" / "pitch_environment_results.pkl"
SEED = 2026
VALUE_SETS = {"general": None, "no_altitude": "No Altitude", "medium": "Medium Altitude", "extreme": "Extreme Altitude"}
SET_LABEL = {"general": "General (liga)", "no_altitude": "No Altitude", "medium": "Medium", "extreme": "Extreme"}
PROFILE_COLS = ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight", "RelSide"]
PROFILE_LABEL = {"RelSpeed": "velocidad (mph)", "InducedVertBreak": "IVB (pulg.)", "HorzBreak": "HB lado del brazo (pulg.)",
                 "SpinRate": "spin (rpm)", "Extension": "extensión (pies)", "RelHeight": "altura de salida (pies)",
                 "RelSide": "salida lateral (pies)"}
CONTEXTS = {"0-0": (0, 0), "0-2": (0, 2), "2-0": (2, 0)}
CONTEXT_LABEL = {"0-0": "0-0 (neutral)", "0-2": "0-2 (dos strikes)", "2-0": "2-0 (bateador adelante)"}
OBJECTIVES = {"stuff": "Stuff+ Physical", "general": "Pitching+ general", "extreme": "Pitching+ Extreme (costos)",
              "extreme_fisica": "Pitching+ Extreme (física aprox. + costos)", "no_altitude": "Pitching+ No Altitude (costos)"}
POOL_MAX = 4000            # candidatos reales por tipo (muestra aleatoria con semilla)
TOP_SHARE = 0.01           # el perfil óptimo es el top 1% de los candidatos
TOP_MIN = 25
K_NEIGHBORS = 25           # vecinos reales para puntuar un perfil y medir densidad
N_BOOT = 200
SUBPOOL_EACH = 100         # experimento 2: top-100 por objetivo + 100 al azar
GRID_STEP = 0.2
GRID_X = np.round(np.arange(-1.6, 1.6001, GRID_STEP), 2)
GRID_Z = np.round(np.arange(0.8, 4.2001, GRID_STEP), 2)
LOC_MIN = 20               # pitches reales del tipo en la celda (mismo enfrentamiento) para usarla

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valores por ambiente (Parte B)
# ---------------------------------------------------------------------------


def load_artifacts() -> dict:
    out = {}
    for name, path in (("pit", chain.FINAL_MODEL_PATHS["pitching"]), ("stf", spp.FINAL_PATH), ("loc", locp.FINAL_PATH)):
        with open(path, "rb") as fh:
            out[name] = pickle.load(fh)
    out["scales"] = json.loads((WEB / "simulator_scales.json").read_text(encoding="utf-8"))
    return out


BIP_IDX = [chain.EVENTS.index(e) for e in chain.BIP_CLASSES]


def delta_rows(deltas: pd.DataFrame, balls, strikes) -> np.ndarray:
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    return deltas[chain.EVENTS].to_numpy()[[pos[(int(b), int(s))] for b, s in zip(balls, strikes)]]


def environment_values(chain_df: pd.DataFrame, probs: np.ndarray) -> dict:
    """Δ(evento, conteo) con la recursión RE24 de run_value_chain, para la liga y para cada ambiente.

    Calibración del costo de contacto (P2 §7.2): el modelo no ve el ambiente, así que su reparto out/1B/2B/3B/HR dada una
    bola en juego es el de la liga y subestima el daño por bola en juego en altitud. En cada ambiente se suma a la rama
    de bola en juego la diferencia entre el Δ observado de sus bolas en juego y el que predice el modelo (ponderado por
    P(BIP)). La versión general es la oficial, sin ajuste; su brecha se reporta como control.
    """
    out = {}
    for name, cat in VALUE_SETS.items():
        m = np.ones(len(chain_df), bool) if cat is None else (chain_df["altitude_category"] == cat).to_numpy()
        d = chain_df[m]
        term = chain.terminal_values(d)
        vtab, deltas = chain.count_values(d, term)
        D = delta_rows(deltas, d["Balls"], d["Strikes"])
        P = probs[m]
        pred = float((P[:, BIP_IDX] * D[:, BIP_IDX]).sum() / P[:, BIP_IDX].sum())
        is_bip = d["event"].isin(chain.BIP_CLASSES).to_numpy()
        ev_idx = d["event"].map({e: i for i, e in enumerate(chain.EVENTS)}).to_numpy()
        obs = float(D[np.where(is_bip)[0], ev_idx[is_bip].astype(int)].mean())
        out[name] = {"terminal": term, "V": vtab["V"], "deltas": deltas, "n": int(len(d)), "e_bip_predicho": pred,
                     "e_bip_observado": obs, "brecha_bip": obs - pred, "shift_bip": 0.0 if cat is None else obs - pred}
    return out


def value_at(probs: np.ndarray, v: dict, balls: int, strikes: int) -> np.ndarray:
    """E[Δ] por fila en un conteo fijo con el juego de valores `v` (incluye la calibración de la rama de bola en juego)."""
    return probs @ v["deltas"].loc[(balls, strikes)][chain.EVENTS].to_numpy() + v["shift_bip"] * probs[:, BIP_IDX].sum(axis=1)


def bip_cost(probs: np.ndarray, v: dict, balls: int, strikes: int) -> np.ndarray:
    """Daño esperado de contacto P(BIP) × E[Δ | BIP] con el juego de valores `v`."""
    return probs[:, BIP_IDX] @ v["deltas"].loc[(balls, strikes)][chain.BIP_CLASSES].to_numpy() + v["shift_bip"] * probs[:, BIP_IDX].sum(axis=1)


def league_scales(chain_df: pd.DataFrame, values: dict, probs: np.ndarray, xrv_oficial: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """E[Δ] de todos los pitches reales con cada juego de valores; media y sd de la escala."""
    xr = pd.DataFrame(index=chain_df.index)
    scales = {}
    for name, v in values.items():
        D = delta_rows(v["deltas"], chain_df["Balls"], chain_df["Strikes"])
        xr[name] = (probs * D).sum(axis=1) + v["shift_bip"] * probs[:, BIP_IDX].sum(axis=1)
        scales[name] = {"media": float(xr[name].mean()), "sd": float(xr[name].std())}
    scales["check_general_vs_oficial"] = float(np.abs(xr["general"].to_numpy() - xrv_oficial).max())
    pbar = pd.DataFrame(probs, columns=chain.EVENTS).groupby([chain_df["Balls"].to_numpy(), chain_df["Strikes"].to_numpy()]).mean()
    pbar.index.names = ["Balls", "Strikes"]
    return scales, pbar, pd.concat([xr, pd.DataFrame(probs, columns=chain.EVENTS, index=chain_df.index)], axis=1)


def plus(xrv, scale: dict):
    return 100 + 10 * (scale["media"] - np.asarray(xrv)) / scale["sd"]


GROUPS = {"bola en juego": chain.BIP_CLASSES, "ponche / strike": ["called_strike", "whiff", "foul"], "base por bolas": ["ball"],
          "golpeado": ["hbp"]}


def contributions(p: np.ndarray, pbar: np.ndarray, d_alt: np.ndarray, d_gen: np.ndarray, shift_alt: float, shift_gen: float = 0.0) -> dict:
    """Σ (p − p̄)·(Δ_alt − Δ_gen) por grupo de eventos; la rama de bola en juego incluye la calibración del costo de contacto.

    Es cuánto cambia el valor RELATIVO del pitch frente al pitch típico del mismo conteo (carreras por pitch).
    Positivo = el pitch pierde valor relativo en el ambiente alterno."""
    diff = (p - pbar) * (d_alt - d_gen)
    out = {g: float(diff[[chain.EVENTS.index(e) for e in evs]].sum()) for g, evs in GROUPS.items()}
    out["bola en juego"] += float((p[BIP_IDX].sum() - pbar[BIP_IDX].sum()) * (shift_alt - shift_gen))
    return out


def part_b_tables(chain_df: pd.DataFrame, xr: pd.DataFrame, scales: dict) -> dict:
    d = chain_df[["PitchUID", "AutoPitchType", "altitude_category"]].copy()
    d["pp_general"] = plus(xr["general"], scales["general"])
    d["pp_extreme"] = plus(xr["extreme"], scales["extreme"])
    d["pp_no_altitude"] = plus(xr["no_altitude"], scales["no_altitude"])
    d["cambio"] = d["pp_extreme"] - d["pp_general"]
    d["p_bip"] = xr[chain.BIP_CLASSES].sum(axis=1)
    d["p_whiff"] = xr["whiff"]
    d["p_k_strike"] = xr[["called_strike", "whiff"]].sum(axis=1)
    d["decil_p_bip"] = pd.qcut(d["p_bip"], 10, labels=[f"D{i}" for i in range(1, 11)])
    rows = []
    for col, lab in (("AutoPitchType", "tipo de pitch"), ("decil_p_bip", "decil de P(BIP)")):
        g = d.groupby(col, observed=True)
        t = g.agg(n=("cambio", "size"), pp_general=("pp_general", "mean"), pp_extreme=("pp_extreme", "mean"),
                  cambio=("cambio", "mean"), p_bip=("p_bip", "mean"), p_whiff=("p_whiff", "mean")).reset_index()
        t = t.rename(columns={col: "grupo"})
        t.insert(0, "agrupacion", lab)
        rows.append(t[t["n"] >= 500] if col == "AutoPitchType" else t)
    tab = pd.concat(rows, ignore_index=True)
    stats = {
        "spearman_general_extreme": float(d["pp_general"].corr(d["pp_extreme"], method="spearman")),
        "spearman_no_extreme": float(d["pp_no_altitude"].corr(d["pp_extreme"], method="spearman")),
        "spearman_general_no": float(d["pp_general"].corr(d["pp_no_altitude"], method="spearman")),
        "cambio_abs_mediana": float(d["cambio"].abs().median()),
        "cambio_abs_p90": float(d["cambio"].abs().quantile(0.9)),
        "pct_cambio_mayor_1": float((d["cambio"].abs() > 1).mean()),
        "pct_cambio_mayor_3": float((d["cambio"].abs() > 3).mean()),
        "corr_cambio_p_bip": float(d["cambio"].corr(d["p_bip"])),
        "corr_cambio_p_whiff": float(d["cambio"].corr(d["p_whiff"])),
        "n_pitches": int(len(d)),
    }
    return {"tabla": tab, "stats": stats}


def within_count_gains(chain_df: pd.DataFrame, xr: pd.DataFrame, values: dict) -> dict:
    """Ganancia de valor RELATIVO en Extreme, comparando cada pitch con el típico de su mismo conteo.

    rel_k = E[Δ]_k − media de E[Δ]_k en el conteo; ganancia (carreras/100 pitches) = −100·(rel_extreme − rel_ref).
    Se separa antes de dos strikes y con dos strikes (donde un whiff es ponche), por quintil de P(whiff) relativo.
    """
    d = pd.DataFrame({"b": chain_df["Balls"].to_numpy(), "s": chain_df["Strikes"].to_numpy(),
                      "p_whiff": xr["whiff"].to_numpy(), "p_bip": xr[chain.BIP_CLASSES].sum(axis=1).to_numpy()})
    for k in ("general", "no_altitude", "extreme"):
        d[k] = xr[k].to_numpy()
        d[k + "_rel"] = d[k] - d.groupby(["b", "s"])[k].transform("mean")
    for c in ("p_whiff", "p_bip"):
        d[c + "_rel"] = d[c] - d.groupby(["b", "s"])[c].transform("mean")
    d["gana_vs_general"] = -100 * (d["extreme_rel"] - d["general_rel"])
    d["gana_vs_no"] = -100 * (d["extreme_rel"] - d["no_altitude_rel"])
    d["fase"] = np.where(d["s"] == 2, "dos strikes", "0 o 1 strike")
    rows, corr = [], {}
    for fase, x in d.groupby("fase"):
        q = pd.qcut(x["p_whiff_rel"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)]).rename("quintil_whiff")
        t = x.groupby(q, observed=True).agg(n=("gana_vs_no", "size"), p_whiff_rel=("p_whiff_rel", "mean"), p_bip_rel=("p_bip_rel", "mean"),
                                           gana_vs_general=("gana_vs_general", "mean"), gana_vs_no=("gana_vs_no", "mean")).reset_index()
        t.insert(0, "fase", fase)
        rows.append(t)
        corr[fase] = {"whiff_vs_no": float(x["gana_vs_no"].corr(x["p_whiff_rel"])), "whiff_vs_general": float(x["gana_vs_general"].corr(x["p_whiff_rel"])),
                      "bip_vs_no": float(x["gana_vs_no"].corr(x["p_bip_rel"])), "bip_vs_general": float(x["gana_vs_general"].corr(x["p_bip_rel"]))}
    tab = pd.concat(rows, ignore_index=True)
    pc.write_web_table(tab, "pitching_relative_gain_by_whiff")
    return {"tabla": tab, "correlaciones": corr,
            "sd_pitching_general": float(xr["general"].std())}


# ---------------------------------------------------------------------------
# Parte C: candidatos reales y contextos
# ---------------------------------------------------------------------------


def candidate_frames(art: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Features del modelo de los pitches de No Altitude: observadas y re-simuladas en Extreme (contrafactual)."""
    raw = cf.load_data()
    raw_no = raw[raw["altitude_category"] == "No Altitude"].reset_index(drop=True)
    coefs = art["pit"]["location_coefs"]
    obs, _ = stuff_plus_v2.build_v2_features(raw_no.copy(), location_coefs=coefs)
    proj = trajectory.recompute_fastball_diffs(trajectory.project_to_altitude(raw_no.copy(), "Extreme Altitude", scoring.load_ratios()))
    projf, _ = stuff_plus_v2.build_v2_features(proj, location_coefs=coefs)
    assert (obs["PitchUID"].to_numpy() == projf["PitchUID"].to_numpy()).all()
    # todas las ubicaciones reales (marco de un derecho) de la liga, para el soporte de la rejilla
    allf, _ = stuff_plus_v2.build_v2_features(raw.copy(), location_coefs=coefs)
    return obs, projf, allf[["AutoPitchType", "same_side", "PlateLocSide", "PlateLocHeight"]]


def with_context(f: pd.DataFrame, x, z, balls: int, strikes: int, same: str) -> pd.DataFrame:
    g = f.copy()
    g["PlateLocSide"] = x
    g["PlateLocHeight"] = z
    g["Balls"] = balls
    g["Strikes"] = strikes
    g["same_side"] = pd.Categorical([same] * len(g), categories=f["same_side"].cat.categories)
    return g


def event_probs(art: dict, f: pd.DataFrame) -> np.ndarray:
    return chain.predict_expected_run_value(art["pit"], f)[chain.EVENTS].to_numpy()


def stuff_plus_values(art: dict, f: pd.DataFrame, category: str) -> np.ndarray:
    s = art["scales"]["stuff"]
    p = spp.marginal_predict(art["stf"], f)
    return 100 + 10 * (p + s["offset_por_ambiente"][category] - s["media_lanzamiento"]) / s["sd_lanzamiento"]


def location_plus_values(art: dict, pitch_type: str, xs, zs, same: str, balls: int, strikes: int) -> np.ndarray:
    """Location+ (escala de lanzamiento del simulador) para ubicaciones en el marco de un derecho."""
    loc = art["loc"]
    bat = "Right" if same == "same" else "Left"
    xs, zs = np.atleast_1d(np.asarray(xs, float)), np.atleast_1d(np.asarray(zs, float))
    row = pd.DataFrame({"loc_in": xs * (-1.0 if bat == "Left" else 1.0), "PlateLocHeight": zs, "Balls": balls, "Strikes": strikes})
    for c, v in (("AutoPitchType", pitch_type), ("same_side", same), ("bat_side", bat)):
        row[c] = pd.Categorical([v] * len(row), categories=loc["categories"][c])
    X = row[loc["numeric_features"] + loc["categorical_features"]]
    pr = {}
    for name, _, _, n_classes in chain.SUBMODELS:
        p = loc["models"][name].predict_proba(X)
        pr[name] = p if n_classes > 2 else p[:, 1:2]
    D = np.tile(loc["deltas_by_count"].loc[(balls, strikes)].to_numpy(), (len(row), 1))
    xrv = chain.expected_run_value(chain.event_probabilities(pr), D)
    ref = loc["reference_by_context"][(pitch_type, same, bat, balls, strikes)]
    s = art["scales"]["location"]
    return 100 - 10 * ((xrv - ref) - s["media_lanzamiento"]) / s["sd_lanzamiento"]


def zdist(a: np.ndarray, b: np.ndarray, sd: np.ndarray) -> float:
    return float(np.sqrt(np.nansum(((a - b) / sd) ** 2)))


# ---------------------------------------------------------------------------
# Parte C: experimento 1 (física sola)
# ---------------------------------------------------------------------------


def evaluate_pool(art, values, scales, pool_obs, pool_proj, typical_loc, w_same) -> dict:
    """Puntajes de cada candidato por contexto (enfrentamientos ponderados)."""
    out = {"stuff": stuff_plus_values(art, pool_obs, "No Altitude")}
    extra = {}
    for ctx, (b, s) in CONTEXTS.items():
        acc = {k: np.zeros(len(pool_obs)) for k in ("general", "extreme", "no_altitude", "extreme_fisica")}
        det = {k: np.zeros(len(pool_obs)) for k in ("p_whiff", "p_bip", "comp_bip_general", "comp_bip_extreme")}
        for same, w in (("same", w_same), ("opposite", 1 - w_same)):
            x, z = typical_loc[same]
            po = event_probs(art, with_context(pool_obs, x, z, b, s, same))
            pp = event_probs(art, with_context(pool_proj, x, z, b, s, same))
            for k in ("general", "extreme", "no_altitude"):
                acc[k] += w * plus(value_at(po, values[k], b, s), scales[k])
            acc["extreme_fisica"] += w * plus(value_at(pp, values["extreme"], b, s), scales["extreme"])
            bip = [chain.EVENTS.index(e) for e in chain.BIP_CLASSES]
            det["p_whiff"] += w * po[:, chain.EVENTS.index("whiff")]
            det["p_bip"] += w * po[:, bip].sum(axis=1)
            for k in ("general", "extreme"):
                det[f"comp_bip_{k}"] += w * bip_cost(po, values[k], b, s)
        for k, v in acc.items():
            out[f"{k}|{ctx}"] = v
        extra[ctx] = det
    return out, extra


def top_index(score: np.ndarray, share: float = TOP_SHARE) -> np.ndarray:
    k = max(TOP_MIN, int(round(share * len(score))))
    return np.argsort(-score)[:k]


def profile(X: np.ndarray, idx: np.ndarray) -> dict:
    sub = X[idx]
    return {"mediana": np.nanmedian(sub, 0), "p10": np.nanpercentile(sub, 10, 0), "p90": np.nanpercentile(sub, 90, 0)}


def knn(Z: np.ndarray, point: np.ndarray, k: int = K_NEIGHBORS) -> tuple[np.ndarray, float]:
    d = np.sqrt(np.nansum((Z - point) ** 2, axis=1))
    order = np.argsort(d)[:k]
    return order, float(d[order[-1]])


def experiment_1(pt: str, art, values, scales, obs, proj, allf) -> dict:
    rng = np.random.default_rng(SEED)
    sel = np.where((obs["AutoPitchType"].astype(str) == pt).to_numpy() & obs[PROFILE_COLS].notna().all(axis=1).to_numpy())[0]
    if len(sel) > POOL_MAX:
        sel = np.sort(rng.choice(sel, POOL_MAX, replace=False))
    po, pp = obs.iloc[sel].reset_index(drop=True), proj.iloc[sel].reset_index(drop=True)
    X = po[PROFILE_COLS].to_numpy(float)
    typ = allf[allf["AutoPitchType"].astype(str) == pt]
    w_same = float((typ["same_side"] == "same").mean() / typ["same_side"].isin(["same", "opposite"]).mean())
    typical = {s: (float(typ.loc[typ["same_side"] == s, "PlateLocSide"].median()), float(typ.loc[typ["same_side"] == s, "PlateLocHeight"].median()))
               for s in ("same", "opposite")}
    scores, extra = evaluate_pool(art, values, scales, po, pp, typical, w_same)
    sd = np.nanstd(X, 0)
    Z = (X - np.nanmean(X, 0)) / sd
    # densidad de referencia: distancia al k-ésimo vecino de cada candidato (muestra de 1,500)
    ref_idx = rng.choice(len(Z), min(1500, len(Z)), replace=False)
    kdist_ref = np.array([knn(Z, Z[i], K_NEIGHBORS + 1)[1] for i in ref_idx])
    pitchers = po["pitcher_anon_id"].to_numpy()
    upitch = np.unique(pitchers)
    rows_of = {p: np.where(pitchers == p)[0] for p in upitch}
    boot_rng = np.random.default_rng(SEED + 1)
    boots = [np.concatenate([rows_of[p] for p in boot_rng.choice(upitch, len(upitch), replace=True)]) for _ in range(N_BOOT)]

    loc_tip = {ctx: float(sum(w * location_plus_values(art, pt, typical[sm][0], typical[sm][1], sm, b, s_)[0]
                              for sm, w in (("same", w_same), ("opposite", 1 - w_same)))) for ctx, (b, s_) in CONTEXTS.items()}
    res = {"tipo": pt, "n_candidatos": int(len(po)), "n_pitchers": int(len(upitch)), "w_same": w_same, "ubicacion_tipica": typical,
           "location_plus_tipica": loc_tip,
           "rango": {c: [float(np.nanpercentile(X[:, i], 1)), float(np.nanpercentile(X[:, i], 99)), float(np.nanmin(X[:, i])),
                         float(np.nanmax(X[:, i]))] for i, c in enumerate(PROFILE_COLS)},
           "sd": dict(zip(PROFILE_COLS, sd.tolist())), "perfiles": {}, "comparaciones": {}}
    top10 = {k: np.quantile(v, 0.9) for k, v in scores.items()}
    proj_cols = proj.iloc[sel].reset_index(drop=True)[["InducedVertBreak", "HorzBreak"]].to_numpy(float)
    for key, sc in scores.items():
        obj, ctx = (key.split("|") + ["todos"])[:2]
        idx = top_index(sc)
        prof = profile(X, idx)
        nn, kd = knn(Z, (prof["mediana"] - np.nanmean(X, 0)) / sd)
        best = int(np.argmax(sc))
        nn_best, _ = knn(Z, Z[best], K_NEIGHBORS + 1)
        nn_best = nn_best[1:]
        bx = {}
        for b in boots:
            bi = b[top_index(sc[b])]
            bx.setdefault("med", []).append(np.nanmedian(X[bi], 0))
        med_b = np.array(bx["med"])
        det = extra.get(ctx, {}) if ctx != "todos" else {}
        res["perfiles"][key] = {
            "objetivo": obj, "contexto": ctx, "n_top": int(len(idx)), "top_stuff_plus": float(np.mean(scores["stuff"][idx])),
            "top_puntaje": float(np.mean(sc[idx])),
            "perfil_mediana": dict(zip(PROFILE_COLS, prof["mediana"].tolist())),
            "perfil_p10": dict(zip(PROFILE_COLS, prof["p10"].tolist())), "perfil_p90": dict(zip(PROFILE_COLS, prof["p90"].tolist())),
            "perfil_boot_p05": dict(zip(PROFILE_COLS, np.nanpercentile(med_b, 5, 0).tolist())),
            "perfil_boot_p95": dict(zip(PROFILE_COLS, np.nanpercentile(med_b, 95, 0).tolist())),
            "puntaje_perfil_vecinos": float(np.mean(sc[nn])), "sd_vecinos_perfil": float(np.std(sc[nn])), "densidad_percentil": float((kdist_ref <= kd).mean() * 100),
            "mejor_observado": dict(zip(PROFILE_COLS, X[best].tolist())), "puntaje_mejor_observado": float(sc[best]),
            "puntaje_vecinos_del_mejor": float(np.median(sc[nn_best])), "umbral_top10": float(top10[key]),
            "mejor_estable": bool(np.median(sc[nn_best]) >= top10[key]),
            "ivb_hb_en_extreme_fisica": proj_cols[idx].mean(0).tolist() if obj == "extreme_fisica" else None,
            **({f"top_{k}": float(np.mean(v[idx])) for k, v in det.items()} if det else {}),
            "_idx": idx, "_boot_med": med_b,
        }
    for ctx in CONTEXTS:
        for alt in ("extreme", "extreme_fisica", "no_altitude"):
            g, a = scores[f"general|{ctx}"], scores[f"{alt}|{ctx}"]
            pg, pa = res["perfiles"][f"general|{ctx}"], res["perfiles"][f"{alt}|{ctx}"]
            d_obs = zdist(np.array(list(pg["perfil_mediana"].values())), np.array(list(pa["perfil_mediana"].values())), sd)
            # ruido: distancia entre dos réplicas bootstrap independientes del MISMO objetivo (general)
            mb = pg["_boot_med"]
            null = np.array([zdist(mb[i], mb[(i + 1) % len(mb)], sd) for i in range(len(mb))])
            d_boot = np.array([zdist(pg["_boot_med"][i], pa["_boot_med"][i], sd) for i in range(len(mb))])
            q95 = float(np.quantile(null, 0.95))
            clase = "Poco" if d_obs <= q95 else ("Moderado" if d_obs <= 2 * q95 else "Mucho")
            zdiff = (np.array(list(pa["perfil_mediana"].values())) - np.array(list(pg["perfil_mediana"].values()))) / sd
            top_g, top_a = set(pg["_idx"].tolist()), set(pa["_idx"].tolist())
            rank_a = pd.Series(a).rank(ascending=False, pct=True).to_numpy()
            res["comparaciones"][f"{alt}|{ctx}"] = {
                "spearman": float(pd.Series(g).corr(pd.Series(a), method="spearman")),
                "jaccard_top1": len(top_g & top_a) / len(top_g | top_a),
                "percentil_en_alt_del_mejor_general": float(100 * (1 - rank_a[int(np.argmax(g))])),
                "mismo_mejor_observado": bool(int(np.argmax(g)) == int(np.argmax(a))),
                "distancia_z": d_obs, "ruido_q95": q95, "distancia_boot_p05": float(np.quantile(d_boot, 0.05)),
                "distancia_boot_p95": float(np.quantile(d_boot, 0.95)), "clase": clase,
                "variable_que_mas_cambia": PROFILE_COLS[int(np.argmax(np.abs(zdiff)))],
                "cambio_z_por_variable": dict(zip(PROFILE_COLS, zdiff.tolist())),
                "stuff_vs_general_spearman": float(pd.Series(scores["stuff"]).corr(pd.Series(g), method="spearman")),
                "jaccard_stuff_general": len(set(res["perfiles"]["stuff"]["_idx"].tolist()) & top_g)
                / len(set(res["perfiles"]["stuff"]["_idx"].tolist()) | top_g),
            }
    res["_pool"] = (po, pp, X, Z, sd, scores, typical, w_same)
    return res


# ---------------------------------------------------------------------------
# Parte C: experimento 2 (física + ubicación)
# ---------------------------------------------------------------------------


def grid_support(allf: pd.DataFrame, pt: str, same: str) -> pd.DataFrame:
    t = allf[(allf["AutoPitchType"].astype(str) == pt) & (allf["same_side"] == same)]
    ix = np.round((t["PlateLocSide"] - GRID_X[0]) / GRID_STEP).astype(int)
    iz = np.round((t["PlateLocHeight"] - GRID_Z[0]) / GRID_STEP).astype(int)
    ok = ix.between(0, len(GRID_X) - 1) & iz.between(0, len(GRID_Z) - 1)
    cnt = pd.Series(1, index=pd.MultiIndex.from_arrays([ix[ok], iz[ok]])).groupby(level=[0, 1]).sum()
    cells = [(GRID_X[i], GRID_Z[j], int(n)) for (i, j), n in cnt.items() if n >= LOC_MIN]
    return pd.DataFrame(cells, columns=["x", "z", "n"])


def experiment_2(r1: dict, art, values, scales, allf) -> dict:
    po, pp, X, Z, sd, scores, typical, w_same = r1["_pool"]
    pt = r1["tipo"]
    rng = np.random.default_rng(SEED + 2)
    pick = set()
    for k in ("stuff", "general|0-0", "extreme|0-0", "extreme_fisica|0-0"):
        pick |= set(np.argsort(-scores[k])[:SUBPOOL_EACH].tolist())
    pick |= set(rng.choice(len(po), min(SUBPOOL_EACH, len(po)), replace=False).tolist())
    # pitch fijo para el mapa: el real más cercano al perfil de mejor Stuff+ (siempre entra a la sub-muestra)
    st = r1["perfiles"]["stuff"]["perfil_mediana"]
    fixed, _ = knn(Z, (np.array(list(st.values())) - np.nanmean(X, 0)) / sd, 1)
    fixed = int(fixed[0])
    pick.add(fixed)
    pick = np.array(sorted(pick))
    out = {"n_subpool": int(len(pick)), "por_enfrentamiento": {}, "heatmap": [],
           "pitch_fijo": {**dict(zip(PROFILE_COLS, X[fixed].tolist())), "stuff_plus": float(scores["stuff"][fixed])}}
    for same in ("same", "opposite"):
        cells = grid_support(allf, pt, same)
        if cells.empty:
            continue
        res_m = {}
        for ctx in ("0-0", "0-2"):
            b, s = CONTEXTS[ctx]
            nP, nC = len(pick), len(cells)
            base_o = po.iloc[np.repeat(pick, nC)].reset_index(drop=True)
            base_p = pp.iloc[np.repeat(pick, nC)].reset_index(drop=True)
            xs, zs = np.tile(cells["x"].to_numpy(), nP), np.tile(cells["z"].to_numpy(), nP)
            po_ = event_probs(art, with_context(base_o, xs, zs, b, s, same))
            pp_ = event_probs(art, with_context(base_p, xs, zs, b, s, same))
            sc = {k: plus(value_at(po_, values[k], b, s), scales[k]).reshape(nP, nC) for k in ("general", "extreme")}
            sc["extreme_fisica"] = plus(value_at(pp_, values["extreme"], b, s), scales["extreme"]).reshape(nP, nC)
            bip = [chain.EVENTS.index(e) for e in chain.BIP_CLASSES]
            probs_of = {"general": po_, "extreme": po_, "extreme_fisica": pp_}
            vals_of = {"general": "general", "extreme": "extreme", "extreme_fisica": "extreme"}
            best = {}
            for k, M in sc.items():
                i, j = np.unravel_index(int(np.argmax(M)), M.shape)
                flat = np.argsort(-M.ravel())[: max(TOP_MIN, int(TOP_SHARE * M.size))]
                ti, tj = np.unravel_index(flat, M.shape)
                P = probs_of[k].reshape(nP, nC, len(chain.EVENTS))[i, j]
                pb_ = float(P[bip].sum())
                comp = float(bip_cost(P[None, :], values[vals_of[k]], b, s)[0])
                best[k] = {"pitch": dict(zip(PROFILE_COLS, X[pick[i]].tolist())), "x": float(cells["x"].iloc[j]), "z": float(cells["z"].iloc[j]),
                           "pitching_plus": float(M[i, j]), "p_bip": pb_, "p_whiff": float(P[chain.EVENTS.index("whiff")]),
                           "e_rv_bip": float(comp / pb_), "stuff_plus": float(scores["stuff"][pick[i]]),
                           "location_plus": float(location_plus_values(art, pt, cells["x"].iloc[j], cells["z"].iloc[j], same, b, s)[0]),
                           "top_x_mediana": float(np.median(cells["x"].to_numpy()[tj])), "top_z_mediana": float(np.median(cells["z"].to_numpy()[tj])),
                           "top_perfil": dict(zip(PROFILE_COLS, np.nanmedian(X[pick[ti]], 0).tolist()))}
            # ubicación óptima del pitch fijo y ranking de ubicaciones entre ambientes
            fi = int(np.where(pick == fixed)[0][0]) if fixed in pick else None
            fixed_cmp = None
            if fi is not None:
                g, e = sc["general"][fi], sc["extreme"][fi]
                fixed_cmp = {"spearman_ubicaciones": float(pd.Series(g).corr(pd.Series(e), method="spearman")),
                             "mejor_celda_general": (float(cells["x"].iloc[int(np.argmax(g))]), float(cells["z"].iloc[int(np.argmax(g))])),
                             "mejor_celda_extreme": (float(cells["x"].iloc[int(np.argmax(e))]), float(cells["z"].iloc[int(np.argmax(e))])),
                             "mejor_celda_extreme_fisica": (float(cells["x"].iloc[int(np.argmax(sc["extreme_fisica"][fi]))]),
                                                            float(cells["z"].iloc[int(np.argmax(sc["extreme_fisica"][fi]))]))}
                if ctx == "0-0":
                    for j in range(len(cells)):
                        out["heatmap"].append({"tipo": pt, "enfrentamiento": same, "x": float(cells["x"].iloc[j]), "z": float(cells["z"].iloc[j]),
                                               "n_celda": int(cells["n"].iloc[j]), "pp_general": float(g[j]), "pp_extreme": float(e[j]),
                                               "pp_extreme_fisica": float(sc["extreme_fisica"][fi][j])})
            dloc = float(np.hypot(best["general"]["x"] - best["extreme"]["x"], best["general"]["z"] - best["extreme"]["z"]))
            dloc_top = float(np.hypot(best["general"]["top_x_mediana"] - best["extreme"]["top_x_mediana"],
                                      best["general"]["top_z_mediana"] - best["extreme"]["top_z_mediana"]))
            Mg, Me = sc["general"].ravel(), sc["extreme"].ravel()
            res_m[ctx] = {"mejor": best, "pitch_fijo": fixed_cmp, "distancia_ubicacion_pies": dloc, "distancia_ubicacion_top_pies": dloc_top,
                          "spearman_combos": float(pd.Series(Mg).corr(pd.Series(Me), method="spearman")),
                          "n_celdas": int(len(cells))}
        out["por_enfrentamiento"][same] = res_m
    return out


# ---------------------------------------------------------------------------
# Corrida y salidas
# ---------------------------------------------------------------------------


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run() -> dict:
    art = load_artifacts()
    logger.info("Valores por ambiente…")
    chain_df = chain.load_data()
    cats = pd.read_parquet(cf.RAW_PATH, columns=["PitchUID", "altitude_category"])
    chain_df = chain_df.merge(cats, on="PitchUID", how="left")
    pred = chain.predict_expected_run_value(art["pit"], chain_df)
    probs = pred[chain.EVENTS].to_numpy()
    values = environment_values(chain_df, probs)
    check_deltas = float(np.abs(values["general"]["deltas"][chain.EVENTS].to_numpy()
                                - art["pit"]["deltas_by_count"].loc[values["general"]["deltas"].index][chain.EVENTS].to_numpy()).max())
    scales, pbar, xr = league_scales(chain_df, values, probs, pred["xrv"].to_numpy())
    scales["check_deltas_general_vs_oficial"] = check_deltas
    logger.info("Δ general vs. artefacto oficial: %.2e; E[Δ] general vs. oficial: %.2e", check_deltas, scales["check_general_vs_oficial"])
    pb = part_b_tables(chain_df, xr, scales)
    wc = within_count_gains(chain_df, xr, values)

    env_json = {
        "definicion": "general = valores de toda la liga (los del Pitching+ oficial); extreme/no_altitude/medium = misma recursión RE24 "
                      "por conteo con los pitches de ese ambiente. Escala 100 + 10·(media − E[Δ])/sd sobre todos los lanzamientos reales.",
        "conjuntos": {k: {"etiqueta": SET_LABEL[k], "n_pitches": v["n"], "terminales": v["terminal"], "shift_bip": v["shift_bip"],
                          "e_bip_predicho": v["e_bip_predicho"], "e_bip_observado": v["e_bip_observado"], "brecha_bip": v["brecha_bip"],
                          "V": {f"{b}-{s}": float(x) for (b, s), x in v["V"].items()},
                          "deltas": {f"{b}-{s}": {e: float(v["deltas"].loc[(b, s), e]) for e in chain.EVENTS} for (b, s) in v["deltas"].index}}
                      for k, v in values.items()},
        "escalas": {k: scales[k] for k in VALUE_SETS},
        "verificacion": {"max_dif_deltas_general_vs_oficial": check_deltas, "max_dif_xrv_general_vs_oficial": scales["check_general_vs_oficial"]},
        "probabilidad_media_por_conteo": {f"{b}-{s}": {e: float(pbar.loc[(b, s), e]) for e in chain.EVENTS} for (b, s) in pbar.index},
        "grupos": GROUPS,
        "comparacion_lanzamientos": pb["stats"],
        "ganancia_relativa_mismo_conteo": {"correlaciones": wc["correlaciones"], "tabla": wc["tabla"].to_dict("records")},
    }
    (WEB / "pitching_env_values.json").write_text(json.dumps(env_json, indent=2, ensure_ascii=False), encoding="utf-8")
    pc.write_web_table(pb["tabla"], "pitching_general_vs_altitude")

    logger.info("Candidatos reales por tipo de pitch…")
    obs, proj, allf = candidate_frames(art)
    results, heat, prof_rows, cmp_rows, loc_rows, fixed_rows = {}, [], [], [], [], []
    for pt in pc.MAIN_TYPES:
        logger.info("  %s: experimento 1", pt)
        r1 = experiment_1(pt, art, values, scales, obs, proj, allf)
        logger.info("  %s: experimento 2", pt)
        r2 = experiment_2(r1, art, values, scales, allf)
        heat += r2.pop("heatmap")
        fixed_rows.append({"tipo": pt, **r2["pitch_fijo"]})
        results[pt] = {"exp1": r1, "exp2": r2}
        for key, p in r1["perfiles"].items():
            row = {"tipo": pt, "objetivo": p["objetivo"], "contexto": p["contexto"], "n_top": p["n_top"],
                   "puntaje_perfil_vecinos": p["puntaje_perfil_vecinos"], "sd_vecinos_perfil": p["sd_vecinos_perfil"], "densidad_percentil": p["densidad_percentil"],
                   "puntaje_mejor_observado": p["puntaje_mejor_observado"], "puntaje_vecinos_del_mejor": p["puntaje_vecinos_del_mejor"],
                   "mejor_estable": p["mejor_estable"], "top_stuff_plus": p["top_stuff_plus"], "top_puntaje": p["top_puntaje"],
                   "location_plus_tipica": r1["location_plus_tipica"].get(p["contexto"]),
                   "frontera": ";".join(c for c in PROFILE_COLS if not r1["rango"][c][0] <= p["perfil_mediana"][c] <= r1["rango"][c][1])}
            for c in PROFILE_COLS:
                row[f"{c}_med"], row[f"{c}_p10"], row[f"{c}_p90"] = p["perfil_mediana"][c], p["perfil_p10"][c], p["perfil_p90"][c]
                row[f"{c}_boot_p05"], row[f"{c}_boot_p95"] = p["perfil_boot_p05"][c], p["perfil_boot_p95"][c]
                row[f"{c}_mejor"] = p["mejor_observado"][c]
                row[f"{c}_pct"] = float((r1["_pool"][2][:, PROFILE_COLS.index(c)] <= p["perfil_mediana"][c]).mean() * 100)
            for k in ("top_p_whiff", "top_p_bip", "top_comp_bip_general", "top_comp_bip_extreme"):
                row[k] = p.get(k)
            prof_rows.append(row)
        for key, c in r1["comparaciones"].items():
            alt, ctx = key.split("|")
            cmp_rows.append({"tipo": pt, "alterno": alt, "contexto": ctx, **{k: v for k, v in c.items() if k != "cambio_z_por_variable"},
                             **{f"z_{k}": v for k, v in c["cambio_z_por_variable"].items()}})
        for same, by_ctx in r2["por_enfrentamiento"].items():
            for ctx, m in by_ctx.items():
                for obj, bst in m["mejor"].items():
                    loc_rows.append({"tipo": pt, "enfrentamiento": same, "contexto": ctx, "objetivo": obj, "x": bst["x"], "z": bst["z"],
                                     "pitching_plus": bst["pitching_plus"], "stuff_plus": bst["stuff_plus"], "location_plus": bst["location_plus"],
                                     "p_whiff": bst["p_whiff"], "p_bip": bst["p_bip"], "e_rv_bip": bst["e_rv_bip"],
                                     "top_x_mediana": bst["top_x_mediana"], "top_z_mediana": bst["top_z_mediana"],
                                     "distancia_ubicacion_pies": m["distancia_ubicacion_pies"], "distancia_ubicacion_top_pies": m["distancia_ubicacion_top_pies"],
                                     "spearman_combos": m["spearman_combos"],
                                     **{f"{c}": bst["pitch"][c] for c in PROFILE_COLS},
                                     **({f"fijo_{k}": str(v) for k, v in m["pitch_fijo"].items()} if m["pitch_fijo"] else {})})
    pc.write_web_table(pd.DataFrame(prof_rows), "optimal_pitch_profiles")
    pc.write_web_table(pd.DataFrame(cmp_rows), "optimal_pitch_comparison")
    pc.write_web_table(pd.DataFrame(loc_rows), "optimal_pitch_location")
    pc.write_web_table(pd.DataFrame(heat), "optimal_location_heatmap")
    pc.write_web_table(pd.DataFrame(fixed_rows), "optimal_location_fixed_pitch")
    meta = {"pool_max": POOL_MAX, "top_share": TOP_SHARE, "top_min": TOP_MIN, "k_vecinos": K_NEIGHBORS, "n_boot": N_BOOT,
            "contextos": CONTEXT_LABEL, "objetivos": OBJECTIVES, "variables": PROFILE_LABEL, "loc_min": LOC_MIN, "rejilla_pies": GRID_STEP,
            "clases": "Poco: distancia ≤ ruido q95; Moderado: ≤ 2× ruido q95; Mucho: > 2× ruido q95 (ruido = distancia entre dos réplicas "
                      "bootstrap por pitcher del mismo objetivo)",
            "tipos": {pt: {"n_candidatos": r["exp1"]["n_candidatos"], "n_pitchers": r["exp1"]["n_pitchers"], "w_same": r["exp1"]["w_same"],
                           "ubicacion_tipica": r["exp1"]["ubicacion_tipica"], "location_plus_tipica": r["exp1"]["location_plus_tipica"],
                           "rango_candidatos_p1_p99_min_max": r["exp1"]["rango"],
                           "sd": r["exp1"]["sd"]} for pt, r in results.items()}}
    (WEB / "optimal_pitch_meta.json").write_text(json.dumps(_clean(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    out = {"env_json": env_json, "part_b_stats": pb["stats"], "part_b_tabla": pb["tabla"], "meta": _clean(meta),
           "profiles": pd.DataFrame(prof_rows), "comparison": pd.DataFrame(cmp_rows), "location": pd.DataFrame(loc_rows),
           "heatmap": pd.DataFrame(heat), "fixed": pd.DataFrame(fixed_rows), "within_count": wc}
    with open(RESULTS_PATH, "wb") as fh:
        pickle.dump(out, fh)
    from src.reports import pitch_environment_report, web_data
    pitch_environment_report.write(out)
    web_data.write_readme()
    return out


if __name__ == "__main__":
    run()
