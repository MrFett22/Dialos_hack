"""Pitching+ v3 neutral por altitud — índice oficial del proyecto.

Responde la pregunta central: ¿qué tan bueno es cada pitcher si todos
lanzaran al nivel del mar? Decisión del usuario (2026-10-03): se neutraliza
el EFECTO TOTAL de la altitud (calidad del pitch + entorno de bola en juego).

1. Cada pitch se proyecta a No Altitude con el contrafactual de trayectoria
   completa (src/features/trajectory.py; ratios de drag y Magnus medidos en
   los datos). Mismo pitcher, misma salida, mismo objetivo en el plato.
2. E[Δ carreras] con la cadena de Pitching+ v3, predicción OUT-OF-FOLD
   (GroupKFold por pitcher, k=5, mismos folds que el v3 oficial): los modelos
   se entrenan con los pitches observados y predicen los pitches proyectados
   de pitchers que nunca vieron.
3. Shrinkage empírico-bayesiano por pitcher-temporada (modelo normal-normal):
       x̄ = promedio de E[Δ] de sus pitches, varianza de muestreo σ²/n
       τ² = varianza "real" entre pitchers (momentos, por temporada, n ≥ 100)
       shrunk = μ + (1 − B)(x̄ − μ),  B = (σ²/n) / (σ²/n + τ²)
   Un pitcher con pocos pitches se acerca al promedio de la liga en vez de
   quedar fuera del ranking (antes: filtro duro de 100 pitches).
4. Escala: Pitching+ = 100 + 10·(μ − shrunk)/τ  → 10 puntos = 1 desviación
   estándar de la habilidad real entre pitchers en esa temporada; 100 = μ, la
   media de E[Δ] ponderada por pitches de los pitchers con ≥100 (el pitch
   promedio de la liga). Por eso el promedio simple de los pitchers no es
   exactamente 100 (≈98–99): los de poco volumen son peores en promedio.
   Intervalo de credibilidad empírico-bayesiano de 80% (posterior normal con
   μ, τ y σ fijos en sus estimaciones; NO incluye la incertidumbre de esos
   parámetros, de los modelos ML, de los ratios de densidad ni de los
   hiperparámetros).
5. Validación:
   - benchmark año-contra-año contra el v3 oficial (con altitud);
   - prueba de neutralización: correlación del score con el % de pitches del
     pitcher en altitud extrema (el v3 con altitud debería castigarlo; el
     neutral, no).
6. Costo de altitud personal: E[Δ](Extreme) − E[Δ](No Altitude) promediado
   sobre los pitches del pitcher (data/processed/altitude_counterfactual_v2_pitches.parquet).

Salidas:
- data/processed/pitching_plus_official.parquet (tabla oficial pitcher-temporada)
- data/processed/pitching_plus_neutral_oof_pitches.parquet
- reports/pitching_plus_neutral.md, reports/benchmark/pitching_plus_neutral.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.analysis import altitude_counterfactual_v2 as cf_v2
from src.data import parks
from src.features import trajectory
from src.models import run_value_chain as chain
from src.models import stuff_plus_v2
from src.validation import benchmark

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "data" / "processed" / "pitching_plus_neutral_oof_pitches.parquet"
OFFICIAL_PATH = ROOT / "data" / "processed" / "pitching_plus_official.parquet"
REPORT_PATH = ROOT / "reports" / "pitching_plus_neutral.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "pitching_plus_neutral.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NEUTRAL_CATEGORY = "No Altitude"
MIN_PITCHES_FOR_TAU = 100
Z_80 = 1.2816


# ---------------------------------------------------------------------------
# 1-2. Predicciones OOF observadas y proyectadas a nivel del mar
# ---------------------------------------------------------------------------


def neutral_features(chain_df: pd.DataFrame, by_stadium: bool = False) -> pd.DataFrame:
    """Features v2 de cada pitch proyectado a No Altitude, alineadas con `chain_df` por PitchUID.

    `by_stadium=False` (Modelo 2): ratio de origen = el de la categoría de altitud × tipo de pitch.
    `by_stadium=True` (Modelo 3a): ratio de origen = el del estadio anónimo del juego (src/data/parks.py), con el ajuste
    de tipo de pitch de su categoría; si el juego no tiene estadio resuelto, el de la categoría. En ambos casos el
    destino es el promedio de No Altitude (ratio 1), así que el Modelo 3a también neutraliza las diferencias entre
    estadios de No Altitude.
    """
    raw = cf_v2.load_data()
    ratios = trajectory.estimate_density_ratios(raw)
    if by_stadium:
        games = pd.read_parquet(parks.OUT_PATH)
        game_stadium = games.loc[games["status"] == "resuelto"].set_index("game_anon_id")["stadium_id"]
        raw = raw.merge(pd.read_parquet(cf_v2.RAW_PATH, columns=["PitchUID", "game_anon_id"]), on="PitchUID", how="left")
        d_obs, m_obs, has = parks.pitch_ratios(raw, game_stadium, ratios, parks.load_stadium_ratios())
        logger.info("Modelo 3a: %.1f%% de los pitches con ratio de su estadio anónimo", 100 * has.mean())
        one = np.ones(len(raw))
        proj = trajectory.project_with_ratios(raw, d_obs, m_obs, one, one)
        proj["altitude_category"] = NEUTRAL_CATEGORY
    else:
        proj = trajectory.project_to_altitude(raw, NEUTRAL_CATEGORY, ratios)
    proj = trajectory.recompute_fastball_diffs(proj)
    _, coefs = stuff_plus_v2.build_v2_features(stuff_plus_v2.load_data())
    feats, _ = stuff_plus_v2.build_v2_features(proj, location_coefs=coefs)
    feats = feats.set_index("PitchUID").loc[chain_df["PitchUID"]].reset_index()
    return feats


def generate_oof(refresh: bool = False, params: dict | None = None, out_path: Path = OOF_PATH,
                 return_probs: bool = False):
    """E[Δ] OOF observado y proyectado a nivel del mar.

    `params`: hiperparámetros por sub-modelo (por defecto chain.submodel_params). Con
    `return_probs` devuelve también las probabilidades OOF observadas de cada sub-modelo
    (para evaluar log-loss/calibración en src/models/tuning.py).
    """
    if out_path.exists() and not refresh and not return_probs:
        logger.info("Usando OOF neutral en cache: %s", out_path.name)
        return pd.read_parquet(out_path)

    df = chain.load_data()
    scenarios = {"obs": df, "neutral": neutral_features(df), "neutral_estadio": neutral_features(df, by_stadium=True)}
    feats = chain.FEATURE_SETS["pitching"]
    cols = feats["numeric"] + feats["categorical"]
    masks = chain.universe_masks(df)
    folds = list(GroupKFold(n_splits=chain.N_FOLDS).split(df, groups=df["pitcher_anon_id"]))

    term = chain.terminal_values(df)
    _, deltas = chain.count_values(df, term)
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    delta_matrix = deltas.to_numpy()[[pos[bs] for bs in zip(df["Balls"], df["Strikes"])]]

    preds = {scen: {} for scen in scenarios}
    for name, universe, target, n_classes in chain.SUBMODELS:
        t0 = time.time()
        width = n_classes if n_classes > 2 else 1
        outs = {scen: np.full((len(df), width), np.nan) for scen in scenarios}
        for tr, ho in folds:
            tr = tr[masks[universe][tr]]
            model = lgb.LGBMClassifier(verbosity=-1, **chain.submodel_params(name, params))
            model.fit(df[cols].iloc[tr], df[target].iloc[tr].astype(int), categorical_feature=feats["categorical"])
            for scen, X in scenarios.items():
                p = model.predict_proba(X[cols].iloc[ho])
                outs[scen][ho] = p if n_classes > 2 else p[:, 1:2]
        for scen in scenarios:
            preds[scen][name] = outs[scen]
        logger.info("  %s: %.0fs", name, time.time() - t0)

    out = df[["PitchUID", "pitcher_anon_id", "year", "AutoPitchType", "Balls", "Strikes", "event"]].copy()
    out["AutoPitchType"] = out["AutoPitchType"].astype("object")
    alt = pd.read_parquet(cf_v2.RAW_PATH, columns=["PitchUID", "altitude_category", "game_anon_id"])
    out = out.merge(alt, on="PitchUID", how="left")
    games = pd.read_parquet(parks.OUT_PATH)
    out["stadium_id"] = out["game_anon_id"].map(games.loc[games["status"] == "resuelto"].set_index("game_anon_id")["stadium_id"])
    for scen in scenarios:
        out[f"xrv_{scen}"] = chain.expected_run_value(chain.event_probabilities(preds[scen]), delta_matrix)
    out.to_parquet(out_path, index=False)
    if return_probs:
        labels = {name: (df[target].to_numpy(), masks[universe]) for name, universe, target, _ in chain.SUBMODELS}
        return out, preds["obs"], labels
    return out


# ---------------------------------------------------------------------------
# 3-4. Shrinkage empírico-bayesiano y escala
# ---------------------------------------------------------------------------


def empirical_bayes(pitches: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, dict]:
    """Promedio por pitcher-temporada encogido hacia la media de su temporada."""
    g = pitches.groupby(["pitcher_anon_id", "year"])[value_col]
    agg = g.agg(n="size", mean="mean", var="var").reset_index()
    params = {}
    parts = []
    for year, sub in agg.groupby("year"):
        big = sub.loc[sub["n"] >= MIN_PITCHES_FOR_TAU]
        sigma2 = float(np.average(big["var"], weights=big["n"]))  # varianza pitch a pitch dentro de pitcher
        mu = float(np.average(big["mean"], weights=big["n"]))
        tau2 = float(big["mean"].var() - (sigma2 / big["n"]).mean())
        params[int(year)] = {"mu": mu, "tau": float(np.sqrt(tau2)), "sigma": float(np.sqrt(sigma2)),
                             "n_pitchers_for_tau": int(len(big))}
        se2 = sigma2 / sub["n"]
        shrink = se2 / (se2 + tau2)
        sub = sub.assign(
            shrink_factor=shrink,
            shrunk=mu + (1 - shrink) * (sub["mean"] - mu),
            post_sd=np.sqrt((1 - shrink) * se2),
        )
        tau = np.sqrt(tau2)
        sub["plus"] = 100 + 10 * (mu - sub["shrunk"]) / tau
        sub["plus_raw"] = 100 + 10 * (mu - sub["mean"]) / tau
        sub["plus_lo80"] = sub["plus"] - 10 * Z_80 * sub["post_sd"] / tau
        sub["plus_hi80"] = sub["plus"] + 10 * Z_80 * sub["post_sd"] / tau
        parts.append(sub)
    return pd.concat(parts, ignore_index=True), params


# ---------------------------------------------------------------------------
# Tabla oficial
# ---------------------------------------------------------------------------


def build_official_table(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    neutral, eb_params = empirical_bayes(oof, "xrv_neutral")
    neutral = neutral.rename(columns={
        "mean": "xrv_neutral_per_pitch", "plus": "pitching_plus", "plus_raw": "pitching_plus_sin_shrinkage",
        "plus_lo80": "pitching_plus_lo80", "plus_hi80": "pitching_plus_hi80", "n": "n_pitches",
    })[["pitcher_anon_id", "year", "n_pitches", "xrv_neutral_per_pitch", "shrink_factor",
        "pitching_plus", "pitching_plus_lo80", "pitching_plus_hi80", "pitching_plus_sin_shrinkage"]]

    # Misma construcción (shrinkage + escala τ) sobre E[Δ] observado, para comparar en la misma escala
    with_alt, _ = empirical_bayes(oof, "xrv_obs")
    with_alt = with_alt.rename(columns={"mean": "xrv_con_altitud_per_pitch", "plus": "pitching_plus_con_altitud"})[
        ["pitcher_anon_id", "year", "xrv_con_altitud_per_pitch", "pitching_plus_con_altitud"]
    ]

    mix = pd.crosstab([oof["pitcher_anon_id"], oof["year"]], oof["altitude_category"], normalize="index")
    mix = mix.reindex(columns=trajectory.ALTITUDE_CATEGORIES, fill_value=0.0)
    mix.columns = ["pct_no_altitude", "pct_medium_altitude", "pct_extreme_altitude"]
    mix = mix.reset_index()

    # Costo de altitud (contrafactual con los modelos finales, in-sample): Extreme − No Altitude del mismo pitch.
    # "antes del contacto" y "bola en juego" son la atribución de Shapley por eslabón (src/models/chain_attribution.py).
    cost = pd.read_parquet(cf_v2.PITCHES_OUT, columns=["pitcher_anon_id", "year", "delta", "sh_bip"] + cf_v2.PRE_CONTACT_COLUMNS)
    cost["antes_contacto"] = cost[cf_v2.PRE_CONTACT_COLUMNS].sum(axis=1)
    cost = cost.groupby(["pitcher_anon_id", "year"]).agg(
        costo_altitud_total_por_100=("delta", "mean"), costo_altitud_antes_contacto_por_100=("antes_contacto", "mean"),
        costo_altitud_bola_en_juego_por_100=("sh_bip", "mean"),
    ).mul(100).reset_index()

    table = (
        neutral.merge(with_alt, on=["pitcher_anon_id", "year"], how="left")
        .merge(mix, on=["pitcher_anon_id", "year"], how="left")
        .merge(cost, on=["pitcher_anon_id", "year"], how="left")
    )
    # Carreras por 100 pitches que el entorno le cargaba al pitcher (positivo = el entorno lo perjudicaba)
    table["ajuste_altitud_carreras_por_100"] = 100 * (table["xrv_con_altitud_per_pitch"] - table["xrv_neutral_per_pitch"])
    table["cambio_ranking"] = (
        table.groupby("year")["pitching_plus_con_altitud"].rank(ascending=False)
        - table.groupby("year")["pitching_plus"].rank(ascending=False)
    )
    table["percentil_temporada"] = table.groupby("year")["pitching_plus"].rank(pct=True)

    # Stuff+ v1: modelo PREDICTIVO de P(whiff) sobre todos los pitches (src/models/stuff_plus.py). Sus features incluyen
    # ángulos de salida y llegada (ubicación implícita), `altitude_level` y residuales de Magnus: NO es independiente de la
    # ubicación ni de la altitud. Promedio OOF por pitcher-temporada, escala 100 + 10·z por temporada (sobre todas las
    # pitcher-temporadas, sin shrinkage). Antes se llamaba `stuff_plus_v1_incluye_ubicacion`; se renombró porque el nombre
    # omitía la altitud (auditoría 2026-10-05, P1.4).
    stuff = pd.read_parquet(benchmark.STUFF_OOF_PITCHES_PATH, columns=["pitcher_anon_id", "year", "pred_is_whiff"])
    stuff = stuff.groupby(["pitcher_anon_id", "year"])["pred_is_whiff"].mean().reset_index()
    stuff["stuff_plus_v1"] = benchmark.scale_plus(stuff["pred_is_whiff"], stuff["year"])
    table = table.merge(stuff[["pitcher_anon_id", "year", "stuff_plus_v1"]], on=["pitcher_anon_id", "year"], how="left")
    return table, eb_params


# ---------------------------------------------------------------------------
# 5. Validación
# ---------------------------------------------------------------------------


def neutralization_check(table: pd.DataFrame) -> dict:
    big = table.loc[table["n_pitches"] >= MIN_PITCHES_FOR_TAU]
    out = {}
    for col in ["pitching_plus_con_altitud", "pitching_plus_sin_shrinkage", "pitching_plus"]:
        out[col] = {
            "corr_pct_extreme": float(big[col].corr(big["pct_extreme_altitude"])),
            "corr_pct_medium": float(big[col].corr(big["pct_medium_altitude"])),
        }
    big = big.assign(entorno=np.select(
        [big["pct_extreme_altitude"] >= 0.5, big["pct_no_altitude"] >= 0.5], ["mayoría Extreme", "mayoría No Altitude"],
        default="mixto",
    ))
    out["mean_by_environment"] = (
        big.groupby("entorno")[["pitching_plus_con_altitud", "pitching_plus_sin_shrinkage", "pitching_plus"]]
        .mean().round(2).to_dict("index")
    )
    out["n_by_environment"] = big["entorno"].value_counts().to_dict()
    return out


def run_benchmark(table: pd.DataFrame) -> dict:
    scores = table.rename(columns={
        "pitching_plus_con_altitud": "pp_v3_con_altitud",
        "pitching_plus_sin_shrinkage": "pp_v3_neutral",
        "pitching_plus": "pp_v3_neutral_eb",
    })
    names = ["pp_v3_con_altitud", "pp_v3_neutral", "pp_v3_neutral_eb"]
    bench_table, _ = benchmark.build_metric_table(extra_models={n: scores for n in names})
    model_metrics = {
        "pitching_plus_v1": "Pitching+ v1",
        "pp_v3_con_altitud": "Pitching+ v3 con altitud",
        "pp_v3_neutral": "Pitching+ v3 neutral",
        "pp_v3_neutral_eb": "Pitching+ v3 neutral + shrinkage (oficial)",
    }
    refs = {"pp_v3_neutral": "pp_v3_con_altitud", "pp_v3_neutral_eb": "pp_v3_con_altitud"}
    return {
        mp: benchmark.correlation_suite(benchmark.build_pairs(bench_table, mp), model_metrics=model_metrics, references=refs)
        for mp in benchmark.MIN_PITCHES_LEVELS
    }, model_metrics


def park_adjusted_test(
    table: pd.DataFrame,
    scores: tuple[str, ...] = ("pitching_plus_con_altitud", "pitching_plus_sin_shrinkage", "pitching_plus"),
    reference: str = "pitching_plus_con_altitud",
    adjust_by: str = "categoria",
) -> dict:
    """Prueba justa para un índice neutral: ¿predice las carreras del año siguiente AJUSTADAS POR ENTORNO?

    Las carreras observadas en t+1 incluyen el parque donde lanzó el pitcher, y
    la mayoría repite parque; un score que conserva el efecto de altitud se
    beneficia de esa persistencia. Aquí el resultado t+1 es el valor en
    carreras realizado de cada pitch menos el promedio de su categoría de
    altitud en esa temporada (carreras evitadas sobre lo esperado en ese entorno).
    """
    realized = pd.read_parquet(chain.OOF_PATH, columns=["PitchUID", "pitcher_anon_id", "year", "realized_delta"])
    alt = pd.read_parquet(cf_v2.RAW_PATH, columns=["PitchUID", "altitude_category", "game_anon_id"])
    realized = realized.merge(alt, on="PitchUID", how="inner")
    if adjust_by == "estadio":
        # Variante: promedio por temporada × estadio anónimo (si el juego no tiene estadio resuelto, por categoría)
        games = pd.read_parquet(parks.OUT_PATH)
        sid = realized["game_anon_id"].map(games.loc[games["status"] == "resuelto"].set_index("game_anon_id")["stadium_id"])
        realized["env"] = sid.fillna(realized["altitude_category"])
    else:
        realized["env"] = realized["altitude_category"]
    realized["adj"] = realized["realized_delta"] - realized.groupby(["year", "env"])["realized_delta"].transform("mean")
    outcome = (-100 * realized.groupby(["pitcher_anon_id", "year"])["adj"].mean()).rename("rv_ajustado").reset_index()

    scores = list(scores)
    t = table[["pitcher_anon_id", "year", "n_pitches"] + scores].merge(outcome, on=["pitcher_anon_id", "year"])
    results = {}
    for mp in benchmark.MIN_PITCHES_LEVELS:
        frames = []
        for y0, y1 in benchmark.SEASON_PAIRS:
            a = t.loc[(t["year"] == y0) & (t["n_pitches"] >= mp)]
            b = t.loc[(t["year"] == y1) & (t["n_pitches"] >= mp), ["pitcher_anon_id", "rv_ajustado"]]
            frames.append(a.merge(b, on="pitcher_anon_id", suffixes=("", "_t1")))
        cols = scores + ["rv_ajustado"]
        pairs = pd.concat(frames, ignore_index=True).dropna(subset=cols + ["rv_ajustado_t1"]).reset_index(drop=True)
        m = pairs[cols + ["rv_ajustado_t1"]].to_numpy(float)
        point = np.corrcoef(m, rowvar=False)[-1, :-1]
        boots = np.array([
            np.corrcoef(m[idx], rowvar=False)[-1, :-1]
            for idx in benchmark._bootstrap_indices(pairs["pitcher_anon_id"], benchmark.N_BOOT, benchmark.BOOT_SEED)
        ])
        ref = scores.index(reference)
        results[mp] = {
            "n_pairs": len(pairs),
            "r": {c: {"r": float(point[i]), "ci95": [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]}
                  for i, c in enumerate(cols)},
            "delta_vs_reference": {
                c: {"delta_r": float(point[i] - point[ref]),
                    "ci95": [float(np.percentile(boots[:, i] - boots[:, ref], 2.5)), float(np.percentile(boots[:, i] - boots[:, ref], 97.5))],
                    "p_model_better": float(np.mean(boots[:, i] > boots[:, ref]))}
                for i, c in enumerate(cols) if c not in (reference, "rv_ajustado")
            },
        }
    return results


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['r']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_delta(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] · P={d['p_model_better']:.0%}"


def format_report(table, eb_params, neutral_chk, bench, model_metrics, oof, adjusted) -> str:
    lines = [
        "# Pitching+ v3 neutral por altitud — índice oficial",
        "",
        "Generado por `src/models/pitching_plus_neutral.py`. Tabla oficial: `data/processed/pitching_plus_official.parquet`.",
        "",
        "## Qué es",
        "",
        "- Cada pitch se **proyecta a nivel del mar** (No Altitude) con el contrafactual de trayectoria completa "
        "(`reports/altitude_counterfactual_v2.md`) y se valúa en carreras con la cadena de Pitching+ v3, "
        "con predicciones out-of-fold por pitcher.",
        "- Se neutraliza el **efecto total** de la altitud (calidad del pitch + entorno de bola en juego), por decisión del usuario.",
        "- **Shrinkage empírico-bayesiano:** el promedio de cada pitcher-temporada se acerca a la media de la liga según su "
        "volumen; un pitcher con pocos pitches queda cerca de 100 con un intervalo ancho, en vez de quedar fuera del ranking.",
        "- **Escala:** 100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ.",
        "- **Incertidumbre:** intervalo de credibilidad empírico-bayesiano de 80%. Es la distribución posterior del modelo normal-normal con μ, τ y σ fijos en "
        "sus estimaciones; refleja cuántos pitches tiene el pitcher. NO incluye la incertidumbre de μ, τ y σ, de los modelos "
        "ML, de los ratios de densidad ni de los hiperparámetros, así que es más angosta que la incertidumbre total.",
        "",
        "## Parámetros del shrinkage por temporada (carreras/pitch)",
        "",
        "| temporada | μ (media liga) | τ (habilidad real) | σ (pitch a pitch) | pitches para encoger a la mitad | pitchers usados |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for y, p in eb_params.items():
        half = (p["sigma"] / p["tau"]) ** 2
        lines.append(f"| {y} | {p['mu']:+.5f} | {p['tau']:.5f} | {p['sigma']:.4f} | {half:,.0f} | {p['n_pitchers_for_tau']:,} |")
    lines += [
        "",
        "*Pitches para encoger a la mitad* = n con el que el score de un pitcher se acerca 50% a la media (σ²/τ²).",
        "",
        "## ¿Funciona la neutralización?",
        "",
        "Correlación del score con el % de pitches del pitcher en cada categoría (pitcher-temporadas con ≥100 pitches):",
        "",
        "| score | corr. con % Extreme | corr. con % Medium |",
        "|---|---:|---:|",
    ]
    labels = {
        "pitching_plus_con_altitud": "Pitching+ v3 con altitud (mismo shrinkage)",
        "pitching_plus_sin_shrinkage": "Pitching+ v3 neutral",
        "pitching_plus": "Pitching+ v3 neutral + shrinkage (oficial)",
    }
    for k, lab in labels.items():
        lines.append(f"| {lab} | {neutral_chk[k]['corr_pct_extreme']:+.3f} | {neutral_chk[k]['corr_pct_medium']:+.3f} |")
    lines += ["", "Promedio del score según dónde lanzó el pitcher la mayoría de sus pitches:", "",
              "| entorno | n | con altitud | neutral | neutral + shrinkage |", "|---|---:|---:|---:|---:|"]
    for env, vals in neutral_chk["mean_by_environment"].items():
        lines.append(
            f"| {env} | {neutral_chk['n_by_environment'][env]:,} | {vals['pitching_plus_con_altitud']:.2f} | "
            f"{vals['pitching_plus_sin_shrinkage']:.2f} | {vals['pitching_plus']:.2f} |"
        )

    for mp, res in bench.items():
        lines += [
            "",
            f"## Benchmark año-contra-año — mínimo {mp} pitches ({res['n_pairs']:,} pares)",
            "",
            "| métrica | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |",
            "|---|---|---|---|---|",
        ]
        for n, lab in model_metrics.items():
            pr = res["predictive"][n]
            lines.append(f"| {lab} | {_fmt(res['stability'][n])} | {_fmt(pr['obs_rv_prevented'])} | "
                         f"{_fmt(pr['obs_k_minus_bb'])} | {_fmt(pr['obs_whiff_rate'])} |")
        for b in ["obs_csw", "obs_k_minus_bb"]:
            pr = res["predictive"][b]
            lines.append(f"| *{benchmark.BASELINE_METRICS[b]}* | {_fmt(res['stability'][b])} | {_fmt(pr['obs_rv_prevented'])} | "
                         f"{_fmt(pr['obs_k_minus_bb'])} | {_fmt(pr['obs_whiff_rate'])} |")
        d = res["deltas_vs_reference"]
        lines += ["", "Delta contra Pitching+ v3 con altitud (bootstrap pareado):", "",
                  "| métrica | estabilidad | carreras evitadas t+1 |", "|---|---|---|"]
        for n in ["pp_v3_neutral", "pp_v3_neutral_eb"]:
            lines.append(f"| {model_metrics[n]} | {_fmt_delta(d[f'{n}|stability'])} | {_fmt_delta(d[f'{n}|obs_rv_prevented'])} |")

    lines += [
        "",
        "## Prueba secundaria (post-hoc): carreras del año siguiente ajustadas por entorno",
        "",
        "**Definida después de ver resultados.** La validación preespecificada (carreras observadas t+1, mín. 100) y la "
        "comparación obligatoria contra el ajuste trivial por temporada × categoría están en `reports/baselines.md`: el índice "
        "neutral no supera a ese ajuste trivial, así que esta tabla no demuestra que la física aporte valor predictivo.",
        "",
        "Las carreras observadas en t+1 incluyen el efecto del parque, y la mayoría de los pitchers repite parque: un score "
        "que conserva la altitud se beneficia de esa persistencia. Aquí el resultado t+1 son las carreras evitadas **sobre lo "
        "esperado en su categoría de altitud** (valor realizado de cada pitch menos el promedio de su categoría y temporada).",
        "",
        "| mín. pitches | pares | con altitud | neutral | neutral + shrinkage | Δ oficial vs con altitud | carreras ajustadas t (baseline) |",
        "|---|---:|---|---|---|---|---|",
    ]
    for mp, res in adjusted.items():
        r = res["r"]
        d = res["delta_vs_reference"]["pitching_plus"]
        lines.append(
            f"| {mp} | {res['n_pairs']} | {_fmt(r['pitching_plus_con_altitud'])} | {_fmt(r['pitching_plus_sin_shrinkage'])} | "
            f"{_fmt(r['pitching_plus'])} | {_fmt_delta(d)} | {_fmt(r['rv_ajustado'])} |"
        )

    ranked = table.loc[table["n_pitches"] >= 100].sort_values("pitching_plus", ascending=False)
    show = ["pitcher_anon_id", "year", "n_pitches", "pitching_plus", "pitching_plus_lo80", "pitching_plus_hi80",
            "pitching_plus_con_altitud", "pct_extreme_altitude", "costo_altitud_total_por_100"]
    lines += ["", "## Top 15 (≥100 pitches)", "", ranked[show].head(15).round(3).to_markdown(index=False), ""]

    movers = table.loc[table["n_pitches"] >= 300]
    lines += [
        "## Pitchers que más suben al neutralizar (≥300 pitches)",
        "",
        movers.sort_values("cambio_ranking", ascending=False)[
            ["pitcher_anon_id", "year", "n_pitches", "pct_extreme_altitude", "pitching_plus_con_altitud", "pitching_plus",
             "ajuste_altitud_carreras_por_100", "cambio_ranking"]
        ].head(10).round(2).to_markdown(index=False),
        "",
        "`cambio_ranking` = posiciones que sube en el ranking de su temporada (todos los pitchers) al neutralizar. "
        "`ajuste_altitud_carreras_por_100` = carreras por 100 pitches que el entorno le cargaba.",
        "",
        "## Pitchers con muestra chica: el shrinkage en acción",
        "",
        table.loc[table["n_pitches"] < 50].sort_values("pitching_plus_sin_shrinkage", ascending=False)[
            ["pitcher_anon_id", "year", "n_pitches", "pitching_plus_sin_shrinkage", "pitching_plus", "pitching_plus_lo80", "pitching_plus_hi80"]
        ].head(5).round(2).to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(refresh: bool = False) -> dict:
    oof = generate_oof(refresh=refresh)
    logger.info("E[Δ] medio observado %+.5f | neutral %+.5f", oof["xrv_obs"].mean(), oof["xrv_neutral"].mean())
    table, eb_params = build_official_table(oof)
    table.to_parquet(OFFICIAL_PATH, index=False)
    neutral_chk = neutralization_check(table)
    logger.info("Neutralización: %s", json.dumps({k: v for k, v in neutral_chk.items() if k.startswith("pitching")}, indent=1))
    bench, model_metrics = run_benchmark(table)
    for mp, res in bench.items():
        logger.info("min %d: carreras t+1 %s", mp, {n: round(res["predictive"][n]["obs_rv_prevented"]["r"], 3) for n in model_metrics})
    adjusted = park_adjusted_test(table)
    for mp, res in adjusted.items():
        logger.info("min %d (carreras ajustadas t+1): %s", mp, {k: round(v["r"], 3) for k, v in res["r"].items()})
    REPORT_PATH.write_text(format_report(table, eb_params, neutral_chk, bench, model_metrics, oof, adjusted), encoding="utf-8")
    payload = {"eb_params": eb_params, "neutralization": neutral_chk, "park_adjusted_test": {str(k): v for k, v in adjusted.items()},
               "benchmark": {str(k): v for k, v in bench.items()}}
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular predicciones OOF")
    run(refresh=parser.parse_args().refresh)


if __name__ == "__main__":
    main()
