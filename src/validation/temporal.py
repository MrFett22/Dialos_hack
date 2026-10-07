"""Validación temporal ESTRICTA del índice neutral y de sus baselines.

Para cada temporada t ∈ {2025, 2026}, todo lo que se aprende de los datos se
estima SOLO con temporadas < t y se aplica a la temporada t:

| componente                                  | se estima con     |
|---------------------------------------------|-------------------|
| matriz RE24 → run_value de cada turno       | temporadas < t    |
| valores terminales, V(b,s), Δ(evento, conteo) | temporadas < t  |
| coeficientes del ajuste de VAA/HAA por ubicación | temporadas < t |
| ratios de densidad efectiva ρ̂ (drag, Magnus) | temporadas < t   |
| ratios de densidad por estadio anónimo (M3a) | temporadas < t   |
| grupos de estadio anónimos (quién es local)  | todas las temporadas (excepción: solo usa quién jugó con quién, no resultados) |
| los 5 sub-modelos LightGBM                  | temporadas < t    |
| recta base del arsenal (ivb/hb/velo vs. recta) | temporada t, solo features (no resultados) |
| shrinkage (μ, τ, σ)                         | predicciones de la temporada t (no usa resultados) |
| ajustes simples M1/M1b (promedios por entorno) | predicciones de la temporada t (no usan resultados) |
| hiperparámetros de LightGBM                 | GLOBALES (excepción, ver abajo) |

Excepción documentada: los hiperparámetros se buscaron con las 3 temporadas
(src/models/tuning.py, objetivo = log-loss de las etiquetas de eventos). Se
usan los mismos aquí. No usan resultados de carreras ni el benchmark como
objetivo, pero sí datos de todas las temporadas.

El resultado t+1 (lo que se quiere predecir) es una MEDICIÓN y se define igual
que en la validación original, para que la comparación sea justa.

Con t = 2024 no hay temporadas anteriores para entrenar, así que la única
pareja evaluable es 2025 → 2026. Las cifras se comparan con los scores
originales (no temporales) restringidos a la misma pareja y los mismos
pitchers.

Salidas: reports/validacion_temporal.md, reports/benchmark/validacion_temporal.json,
data/processed/temporal_strict_pitches.parquet.
"""

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.analysis import altitude_counterfactual_v2 as cf
from src.features import targets as targets_mod
from src.features import trajectory as tr
from src.models import pitching_plus_neutral as pn
from src.models import run_value_chain as chain
from src.models import stuff_plus_v2 as v2
from src.data import parks
from src.validation import baselines, benchmark

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / "data" / "processed" / "pitches_clean.parquet"
PITCHES_PATH = ROOT / "data" / "processed" / "temporal_strict_pitches.parquet"
REPORT_PATH = ROOT / "reports" / "validacion_temporal.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "validacion_temporal.json"
SCORED_SEASONS = [2025, 2026]
TARGET_COLS = ["PitchUID", "PitchCall", "KorBB", "Balls", "Strikes", "single", "double", "triple", "home_run"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_value_before(t: int) -> pd.Series:
    """run_value con una matriz RE24 estimada solo con temporadas < t (indexado por PitchUID)."""
    clean = pd.read_parquet(CLEAN_PATH)
    clean = clean.loc[clean["year"] < t].reset_index(drop=True)
    clean, _ = targets_mod.fix_is_hit_by_pitch(clean)
    rv, _ = targets_mod.add_run_value(clean)
    return rv.set_index("PitchUID")["run_value"]


def add_events(df: pd.DataFrame) -> pd.DataFrame:
    """Misma taxonomía de eventos y mismas etiquetas que run_value_chain.load_data."""
    calls = df["PitchCall"].astype("object")
    df["event_group"] = np.select([calls.isin(list(chain.TAKE_CALLS)), calls.isin(list(chain.SWING_CALLS))], ["take", "swing"], "unknown")
    df = df.loc[(df["event_group"] != "unknown") & df["Balls"].between(0, 3) & df["Strikes"].between(0, 2)].reset_index(drop=True)
    calls = df["PitchCall"].astype("object")
    swing_event = calls.map(chain.SWING_CALLS)
    bip = np.select([df["home_run"] == 1, df["triple"] == 1, df["double"] == 1, df["single"] == 1],
                    ["home_run", "triple", "double", "single"], default="out")
    df["event"] = np.where(df["event_group"] == "take", calls.map(chain.TAKE_CALLS), np.where(swing_event == "in_play", bip, swing_event))
    df["Balls"], df["Strikes"] = df["Balls"].astype(int), df["Strikes"].astype(int)
    df["y_swing"] = (df["event_group"] == "swing").astype(int)
    df["y_take"] = df["event"].map({c: i for i, c in enumerate(chain.TAKE_CLASSES)})
    df["y_whiff"] = np.where(df["event_group"] == "swing", (df["event"] == "whiff").astype(int), np.nan)
    contact = (df["event_group"] == "swing") & (df["event"] != "whiff")
    df["y_foul"] = np.where(contact, (df["event"] == "foul").astype(int), np.nan)
    df["y_bip"] = df["event"].map({c: i for i, c in enumerate(chain.BIP_CLASSES)})
    return df


def score_season(t: int, base: pd.DataFrame, raw: pd.DataFrame, tgt: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    past = base.loc[base["year"] < t].reset_index(drop=True)
    _, coefs = v2.build_v2_features(past)
    train, _ = v2.build_v2_features(past, location_coefs=coefs)
    rv = run_value_before(t)
    train = add_events(train.merge(tgt, on="PitchUID").merge(rv.rename("run_value"), left_on="PitchUID", right_index=True))
    term = chain.terminal_values(train)
    vtab, deltas = chain.count_values(train, term)

    ratios = tr.estimate_density_ratios(raw.loc[raw["year"] < t])
    cur_raw = raw.loc[raw["year"] == t].reset_index(drop=True)
    obs, _ = v2.build_v2_features(cur_raw, location_coefs=coefs)
    proj = tr.recompute_fastball_diffs(tr.project_to_altitude(cur_raw, "No Altitude", ratios))
    neu, _ = v2.build_v2_features(proj, location_coefs=coefs)
    # M3a: ratios por estadio anónimo estimados solo con temporadas < t (si un estadio no tiene, el de su categoría)
    resolved = games.loc[games["status"] == "resuelto"]
    game_stadium = resolved.set_index("game_anon_id")["stadium_id"]
    past_raw = raw.loc[raw["year"] < t]
    st_ratios = parks.stadium_ratios(parks.physics_frame(past_raw, games)).set_index("stadium_id")
    d_obs, m_obs, _ = parks.pitch_ratios(cur_raw, game_stadium, ratios, st_ratios)
    one = np.ones(len(cur_raw))
    proj_st = tr.recompute_fastball_diffs(tr.project_with_ratios(cur_raw, d_obs, m_obs, one, one))
    neu_st, _ = v2.build_v2_features(proj_st, location_coefs=coefs)
    obs = add_events(obs.merge(tgt.drop(columns=["Balls", "Strikes"]), on="PitchUID"))
    neu = neu.set_index("PitchUID").loc[obs["PitchUID"]].reset_index()
    neu_st = neu_st.set_index("PitchUID").loc[obs["PitchUID"]].reset_index()

    feats = chain.FEATURE_SETS["pitching"]
    cols = feats["numeric"] + feats["categorical"]
    masks = chain.universe_masks(train)
    preds = {"obs": {}, "neutral": {}, "neutral_estadio": {}}
    for name, universe, target, k in chain.SUBMODELS:
        sub = train.loc[masks[universe]]
        model = lgb.LGBMClassifier(verbosity=-1, **chain.submodel_params(name))
        model.fit(sub[cols], sub[target].astype(int), categorical_feature=feats["categorical"])
        for scen, X in (("obs", obs), ("neutral", neu), ("neutral_estadio", neu_st)):
            p = model.predict_proba(X[cols])
            preds[scen][name] = p if k > 2 else p[:, 1:2]
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    D = deltas.to_numpy()[[pos[bs] for bs in zip(obs["Balls"], obs["Strikes"])]]
    out = obs[["PitchUID", "pitcher_anon_id", "year", "altitude_category", "game_anon_id"]].copy()
    out["altitude_category"] = out["altitude_category"].astype("object")
    out["stadium_id"] = out["game_anon_id"].map(game_stadium)
    for scen in ("obs", "neutral", "neutral_estadio"):
        out[f"xrv_{scen}"] = chain.expected_run_value(chain.event_probabilities(preds[scen]), D)
    glob = ratios.loc[ratios["AutoPitchType"] == "__all__"].set_index("altitude_category")
    logger.info("t=%d: entrenado con %s pitches (< %d), puntuados %s; ρ_E = %.3f/%.3f; V(0,0) = %+.4f; Δ(hbp,0-0) = %+.4f (%.0fs)",
                t, f"{len(train):,}", t, f"{len(out):,}", glob.loc["Extreme Altitude", "drag_ratio"], glob.loc["Extreme Altitude", "magnus_ratio"],
                vtab.loc[(0, 0), "V"], deltas.loc[(0, 0), "hbp"], time.time() - t0)
    return out


MODELS = ["m0", "m1", "m1b", "m2", "m3a"]


def run() -> dict:
    base = v2.load_data()
    raw = cf.load_data().merge(pd.read_parquet(cf.RAW_PATH, columns=["PitchUID", "game_anon_id"]), on="PitchUID", how="left")
    tgt = pd.read_parquet(chain.TARGETS_PATH, columns=TARGET_COLS)
    games = pd.read_parquet(parks.OUT_PATH)
    pitches = pd.concat([score_season(t, base, raw, tgt, games) for t in SCORED_SEASONS], ignore_index=True)
    pitches.to_parquet(PITCHES_PATH, index=False)

    temporal = baselines.build_scores(pitches)[["pitcher_anon_id", "year"] + MODELS]
    temporal = temporal.rename(columns={m: f"t_{m}" for m in MODELS})
    original = baselines.build_scores(pd.read_parquet(pn.OOF_PATH))
    original = original[["pitcher_anon_id", "year", "n_pitches", "pct_extreme"] + MODELS].rename(columns={m: f"o_{m}" for m in MODELS})
    table = original.merge(temporal, on=["pitcher_anon_id", "year"], how="left")
    # Solo la pareja 2025 → 2026: los scores temporales no existen para 2024 y las pruebas exigen casos completos,
    # así que original y temporal se comparan exactamente sobre los mismos pares.
    names = [f"o_{m}" for m in MODELS] + [f"t_{m}" for m in MODELS]
    labels = {f"o_{m}": f"{m.upper()} original" for m in MODELS} | {f"t_{m}": f"{m.upper()} temporal estricto" for m in MODELS}
    refs = ["t_m0", "t_m1", "t_m1b"]

    result = {"secondary_posthoc": {}, "prespecified": {}}
    for ref in refs:
        res = pn.park_adjusted_test(table, scores=tuple(names), reference=ref)
        for mp, r in res.items():
            o = result["secondary_posthoc"].setdefault(mp, {"n_pairs": r["n_pairs"], "r": {k: v for k, v in r["r"].items()}, "deltas": {}})
            o["deltas"].update({f"{k} vs {ref}": v for k, v in r["delta_vs_reference"].items()})
    bt, _ = benchmark.build_metric_table(extra_models={n: table for n in names})
    for mp in benchmark.MIN_PITCHES_LEVELS:
        pairs = benchmark.build_pairs(bt, mp)
        pairs = pairs[pairs["pair"] == "2025->2026"]
        res = {ref: benchmark.correlation_suite(pairs, model_metrics={n: labels[n] for n in names},
                                                references={n: ref for n in names if n != ref}) for ref in refs}
        result["prespecified"][mp] = {
            "n_pairs": res[refs[0]]["n_pairs"],
            "r": {n: res[refs[0]]["predictive"][n]["obs_rv_prevented"] for n in names},
            "deltas": {f"{n} vs {ref}": res[ref]["deltas_vs_reference"][f"{n}|obs_rv_prevented"] for ref in res for n in names if n != ref},
        }
    big = table[(table["n_pitches"] >= 100)]
    result["neutralization"] = {n: {str(int(y)): float(g[n].corr(g["pct_extreme"])) for y, g in big.groupby("year") if g[n].notna().any()}
                                for n in [f"t_{m}" for m in MODELS]}
    REPORT_PATH.write_text(format_report(result, labels), encoding="utf-8")
    JSON_PATH.write_text(json.dumps({str(k): v for k, v in result.items()}, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return result


def _f(d: dict) -> str:
    return f"{d['r']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def _fd(d: dict) -> str:
    lo, hi = d["ci95"]
    ok = d["delta_r"] > 0 and d["p_model_better"] >= baselines.P_THRESHOLD
    return f"{d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] · P={d['p_model_better']:.0%} · {'✅' if ok else '❌'}"


def format_report(res: dict, labels: dict) -> str:
    L = [
        "# Validación temporal estricta (entrenamiento con temporadas < t)",
        "",
        "Generado por `src/validation/temporal.py`. Qué se estima con qué datos: ver el docstring del módulo "
        "(todo con temporadas < t, salvo los hiperparámetros, que son globales, y el shrinkage y el ajuste trivial, que usan "
        "solo predicciones de la temporada t, sin resultados).",
        "",
        "Única pareja evaluable: **2025 → 2026** (para puntuar 2024 no hay temporadas anteriores). *Original* = los scores del "
        "pipeline normal (OOF por pitcher con las 3 temporadas) restringidos a la misma pareja y los mismos pitchers.",
        "",
    ]
    for key, title in [("prespecified", "Resultado preespecificado: carreras observadas t+1"),
                       ("secondary_posthoc", "Resultado secundario (post-hoc): carreras ajustadas por entorno t+1")]:
        L += [f"## {title}", ""]
        for mp, r in res[key].items():
            L += [f"### Mínimo {mp} pitches ({r['n_pairs']} pares)", "", "| score | r [IC 95%] |", "|---|---|"]
            for n, lab in labels.items():
                L.append(f"| {lab} | {_f(r['r'][n])} |")
            L += ["", "| comparación | Δr [IC 95%] · P · cumple (P ≥ 0.90) |", "|---|---|"]
            for comp in ["t_m1 vs t_m0", "t_m2 vs t_m0", "t_m2 vs t_m1", "t_m3a vs t_m1", "t_m3a vs t_m1b", "t_m2 vs t_m1b",
                         "o_m2 vs t_m1"]:
                if comp in r["deltas"]:
                    a, b = comp.split(" vs ")
                    L.append(f"| {labels[a]} vs {labels[b]} | {_fd(r['deltas'][comp])} |")
            L.append("")
    L += ["## Neutralización (corr. con % de pitches en Extreme, ≥100 pitches)", "", "| score | " + " | ".join(next(iter(res["neutralization"].values()))) + " |",
          "|---|" + "---:|" * len(next(iter(res["neutralization"].values())))]
    for n, v in res["neutralization"].items():
        L.append(f"| {labels[n]} | " + " | ".join(f"{x:+.3f}" for x in v.values()) + " |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    run()
