"""Chequeo de out-of-distribution (OOD) para el contrafactual de altitud.

Pregunta a responder: cuando el contrafactual (reports/altitude_counterfactual.md)
recalcula InducedVertBreak/HorzBreak bajo una densidad de aire distinta
(manteniendo RelSpeed/SpinRate/SpinAxis/Extension y el residual de Magnus del
pitcher fijos, ver src/analysis/altitude_counterfactual.py), ¿esas combinaciones
contrafactuales de (InducedVertBreak_cf, HorzBreak_cf, RelSpeed, SpinRate) son
fisicamente plausibles para esa categoria de altitud, o el modelo esta
extrapolando fuera de lo que vio en entrenamiento?

Metodologia (las 3 pedidas por el usuario, sobre la MISMA muestra de 100k y
las MISMAS features contrafactuales que ya uso altitude_counterfactual.py,
regeneradas aqui con el mismo seed para reproducibilidad exacta):

1. Percentiles 1-99 por (AutoPitchType, altitude_category objetivo), calculados
   sobre el TRAIN de pitcher_holdout (el universo real que vio pitching_plus_v1.pkl,
   no todo el dataset) — OOD si CUALQUIERA de las 4 features cae fuera de [p1,p99].
2. Distancia de Mahalanobis (misma agrupacion, misma referencia train) contra
   chi2 al 99% con 4 grados de libertad.
3. Isolation Forest (mismo agrupamiento y referencia), contamination='auto'.

Solo se ajustan Mahalanobis/IsolationForest para grupos con n_train>=200 (celdas
mas chicas no tienen suficiente muestra para estimar covarianza 4x4 ni entrenar
un bosque razonable) — esos grupos se reportan aparte, solo con el chequeo de
percentiles.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

from src.analysis.altitude_counterfactual import (
    ALTITUDE_CATEGORIES,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    build_counterfactual_features,
    load_data,
    stratified_sample,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "altitude_counterfactual_ood_check.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FEATURES = ["InducedVertBreak", "HorzBreak", "RelSpeed", "SpinRate"]
MIN_N_FOR_MODEL = 200
IF_SEED = 42
MAHAL_ALPHA = 0.99  # umbral chi2 al 99%


# ---------------------------------------------------------------------------
# Referencia de entrenamiento: percentiles + Mahalanobis + Isolation Forest,
# por (AutoPitchType, altitude_category), sobre pitcher_holdout == "train"
# ---------------------------------------------------------------------------


def build_reference(train_df: pd.DataFrame) -> dict:
    ref = {}
    for (pt, cat), g in train_df.groupby(["AutoPitchType", "altitude_category"], observed=True):
        X = g[FEATURES].dropna().to_numpy(dtype=float)
        n = len(X)
        entry = {
            "n_train": n,
            "p1": np.percentile(X, 1, axis=0) if n > 0 else None,
            "p99": np.percentile(X, 99, axis=0) if n > 0 else None,
        }
        if n >= MIN_N_FOR_MODEL:
            cov = LedoitWolf().fit(X)
            entry["mean"] = cov.location_
            entry["precision"] = cov.get_precision()
            entry["mahal_threshold"] = stats.chi2.ppf(MAHAL_ALPHA, df=X.shape[1])

            iso = IsolationForest(contamination="auto", random_state=IF_SEED, n_estimators=200)
            iso.fit(X)
            entry["iso_forest"] = iso
        ref[(pt, cat)] = entry
    return ref


def score_ood(cf_df: pd.DataFrame, target_cat: str, reference: dict) -> pd.DataFrame:
    """Para cada fila de cf_df (ya con InducedVertBreak/HorzBreak contrafactuales
    bajo target_cat), evalua OOD contra la referencia de train de
    (AutoPitchType, target_cat)."""
    out = pd.DataFrame(index=cf_df.index)
    out["AutoPitchType"] = cf_df["AutoPitchType"].to_numpy()
    out["altitude_category_actual"] = cf_df["altitude_category"].to_numpy()
    out["n_train_ref"] = np.nan
    out["ood_percentile"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["mahalanobis_d2"] = np.nan
    out["ood_mahalanobis"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["ood_isoforest"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["has_model_ref"] = False

    for pt, g in cf_df.groupby("AutoPitchType", observed=True):
        entry = reference.get((pt, target_cat))
        idx = g.index
        if entry is None or entry["n_train"] == 0:
            out.loc[idx, "n_train_ref"] = 0
            continue

        X = g[FEATURES].to_numpy(dtype=float)
        out.loc[idx, "n_train_ref"] = entry["n_train"]

        p1, p99 = entry["p1"], entry["p99"]
        within = (X >= p1) & (X <= p99)
        out.loc[idx, "ood_percentile"] = ~within.all(axis=1)

        if "precision" in entry:
            out.loc[idx, "has_model_ref"] = True
            diff = X - entry["mean"]
            d2 = np.einsum("ij,jk,ik->i", diff, entry["precision"], diff)
            out.loc[idx, "mahalanobis_d2"] = d2
            out.loc[idx, "ood_mahalanobis"] = d2 > entry["mahal_threshold"]

            iso_pred = entry["iso_forest"].predict(X)  # -1 = anomalia, 1 = normal
            out.loc[idx, "ood_isoforest"] = iso_pred == -1

    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> dict:
    df_full = load_data()
    train_df = df_full.loc[df_full["pitcher_holdout"] == "train"].copy()
    logger.info("Train de pitcher_holdout (referencia OOD): %s pitches", f"{len(train_df):,}")

    reference = build_reference(train_df)
    n_groups_with_model = sum(1 for e in reference.values() if "precision" in e)
    logger.info(
        "Grupos (AutoPitchType x altitude_category) en train: %s | con n>=%s (Mahalanobis+IsoForest): %s",
        len(reference), MIN_N_FOR_MODEL, n_groups_with_model,
    )

    sample = stratified_sample(df_full, SAMPLE_SIZE, SAMPLE_SEED)
    logger.info("Muestra contrafactual (misma que altitude_counterfactual.py): %s pitches", f"{len(sample):,}")

    all_scores = {}
    for cat in ALTITUDE_CATEGORIES:
        cf_df = build_counterfactual_features(sample, cat)
        scored = score_ood(cf_df, cat, reference)
        scored["target_altitude_category"] = cat
        all_scores[cat] = scored
        logger.info("OOD evaluado para escenario: %s", cat)

    combined = pd.concat(all_scores.values(), keys=all_scores.keys(), names=["scenario", "orig_idx"]).reset_index(
        level=0, drop=True
    )

    # --- Calibracion: tasa "natural" de OOD sobre pitches REALES (no
    # contrafactuales) del TEST de pitcher_holdout, evaluados contra la MISMA
    # referencia de train, cada pitch contra su propia altitude_category real.
    # Esto separa "OOD por el metodo/por variacion pitcher-a-pitcher normal"
    # de "OOD extra introducido especificamente por el contrafactual".
    test_df = df_full.loc[df_full["pitcher_holdout"] == "test"].copy()
    baseline_parts = []
    for cat in ALTITUDE_CATEGORIES:
        g = test_df.loc[test_df["altitude_category"] == cat]
        if len(g) == 0:
            continue
        scored = score_ood(g, cat, reference)
        scored["target_altitude_category"] = cat
        baseline_parts.append(scored)
    baseline = pd.concat(baseline_parts)
    logger.info("Baseline de calibracion (pitches reales de test): %s filas", f"{len(baseline):,}")

    return {
        "reference": reference,
        "train_df": train_df,
        "sample": sample,
        "scores": combined,
        "baseline": baseline,
    }


# ---------------------------------------------------------------------------
# Agregacion / reporte
# ---------------------------------------------------------------------------


def summarize(scores: pd.DataFrame) -> dict:
    summary = {}

    # 1. % OOD global por escenario (metodo percentil = el pedido explicitamente)
    overall = (
        scores.groupby("target_altitude_category")
        .agg(
            n=("ood_percentile", "size"),
            pct_ood_percentile=("ood_percentile", lambda s: 100 * s.mean()),
            pct_with_model_ref=("has_model_ref", lambda s: 100 * s.mean()),
        )
    )
    mahal_pct = (
        scores.loc[scores["has_model_ref"]]
        .groupby("target_altitude_category")["ood_mahalanobis"]
        .apply(lambda s: 100 * s.mean())
    )
    iso_pct = (
        scores.loc[scores["has_model_ref"]]
        .groupby("target_altitude_category")["ood_isoforest"]
        .apply(lambda s: 100 * s.mean())
    )
    overall["pct_ood_mahalanobis"] = mahal_pct
    overall["pct_ood_isoforest"] = iso_pct
    summary["overall"] = overall.reindex(ALTITUDE_CATEGORIES)

    # 2. Desglose por AutoPitchType x escenario (foco en Curveball)
    by_pt = (
        scores.groupby(["target_altitude_category", "AutoPitchType"], observed=True)
        .agg(
            n=("ood_percentile", "size"),
            n_train_ref=("n_train_ref", "first"),
            pct_ood_percentile=("ood_percentile", lambda s: 100 * s.mean()),
        )
    )
    mahal_by_pt = (
        scores.loc[scores["has_model_ref"]]
        .groupby(["target_altitude_category", "AutoPitchType"], observed=True)["ood_mahalanobis"]
        .apply(lambda s: 100 * s.mean())
    )
    iso_by_pt = (
        scores.loc[scores["has_model_ref"]]
        .groupby(["target_altitude_category", "AutoPitchType"], observed=True)["ood_isoforest"]
        .apply(lambda s: 100 * s.mean())
    )
    by_pt["pct_ood_mahalanobis"] = mahal_by_pt
    by_pt["pct_ood_isoforest"] = iso_by_pt
    summary["by_pitchtype"] = by_pt.reset_index()

    return summary


def summarize_baseline(baseline: pd.DataFrame) -> pd.Series:
    """Tasa 'natural' de OOD (metodo percentil, el pedido explicitamente) sobre
    pitches REALES de pitchers de test contra la referencia de train — mide
    cuanto OOD se espera solo por variacion normal pitcher-a-pitcher (no por el
    contrafactual), para poder comparar contra la seccion 1."""
    return pd.Series(
        {
            "n": len(baseline),
            "pct_ood_percentile": 100 * baseline["ood_percentile"].mean(),
            "pct_ood_mahalanobis": 100 * baseline.loc[baseline["has_model_ref"], "ood_mahalanobis"].mean(),
            "pct_ood_isoforest": 100 * baseline.loc[baseline["has_model_ref"], "ood_isoforest"].mean(),
        }
    )


def format_report(summary: dict, baseline_summary: pd.Series, n_groups_ref: int, n_groups_model: int, min_n: int) -> str:
    overall = summary["overall"]
    by_pt = summary["by_pitchtype"]

    lines = [
        "# Chequeo de out-of-distribution (OOD) — contrafactual de altitud",
        "",
        "Verifica si el hallazgo de `reports/altitude_counterfactual.md` (is_whiff "
        "sube al mover el mismo pitch a otra altitud) es un artefacto de "
        "extrapolacion: para cada uno de los 100,000 pitches de la muestra "
        "contrafactual, se evalua si (InducedVertBreak_cf, HorzBreak_cf, RelSpeed, "
        "SpinRate) — la combinacion que se le da a `pitching_plus_v1.pkl` para "
        "predecir bajo cada escenario — cae dentro de lo que el modelo realmente "
        "vio en entrenamiento para esa combinacion (AutoPitchType, "
        "altitude_category objetivo).",
        "",
        "**Referencia usada: el TRAIN de `pitcher_holdout`** (503,112 pitches, no "
        "el dataset completo) — es el universo exacto que ajusto `pitching_plus_v1.pkl`, "
        "asi que es la pregunta correcta para 'extrapolacion del modelo', no una "
        "pregunta general de plausibilidad fisica.",
        "",
        f"3 metodos, agrupando por (AutoPitchType, altitude_category objetivo): "
        f"(1) percentiles 1-99 de train (OOD si CUALQUIERA de las 4 features cae "
        f"afuera); (2) distancia de Mahalanobis (covarianza LedoitWolf) contra "
        f"chi2 al 99% con 4 gl; (3) Isolation Forest (contamination='auto'). Los "
        f"metodos (2)-(3) solo se ajustan para grupos con n_train>={min_n} "
        f"({n_groups_model} de {n_groups_ref} grupos AutoPitchType x altitude_category "
        "califican) — grupos mas chicos solo tienen el chequeo de percentiles.",
        "",
        "## 1. % OOD por escenario contrafactual (los 3 metodos)",
        "",
        "| escenario simulado | n | % OOD (percentiles 1-99) | % OOD (Mahalanobis, solo grupos con ref) | "
        "% OOD (Isolation Forest, solo grupos con ref) | % filas con grupo de referencia solido (n_train>=" + str(min_n) + ") |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat in overall.index:
        row = overall.loc[cat]
        lines.append(
            f"| {cat} | {int(row['n']):,} | {row['pct_ood_percentile']:.2f}% | "
            f"{row['pct_ood_mahalanobis']:.2f}% | {row['pct_ood_isoforest']:.2f}% | "
            f"{row['pct_with_model_ref']:.1f}% |"
        )

    b = baseline_summary
    lines += [
        "",
        "### Calibracion: tasa 'natural' de OOD sin contrafactual",
        "",
        "Mismo chequeo, mismos 3 metodos, misma referencia de train — pero sobre "
        "pitches REALES (sin recalcular movimiento) de los pitchers de "
        "`pitcher_holdout` TEST (nunca vistos en train), cada uno contra su "
        "propia `altitude_category` real. Esto mide cuanto OOD aparece solo por "
        "variacion normal pitcher-a-pitcher, sin ningun cambio de altitud — el "
        "piso de comparacion para saber si el % de la seccion 1 es alto DE VERDAD "
        "o es del orden del ruido de metodo/generalizacion normal.",
        "",
        f"| n | % OOD percentiles | % OOD Mahalanobis | % OOD IsoForest |",
        "|---:|---:|---:|---:|",
        f"| {int(b['n']):,} | {b['pct_ood_percentile']:.2f}% | {b['pct_ood_mahalanobis']:.2f}% | {b['pct_ood_isoforest']:.2f}% |",
        "",
    ]

    lines += [
        "## 2. Desglose por AutoPitchType x escenario (solo tipos con n>=100 en la muestra)",
        "",
        "| escenario | AutoPitchType | n muestra | n_train referencia | % OOD percentiles | "
        "% OOD Mahalanobis | % OOD IsoForest |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    by_pt_reliable = by_pt.loc[by_pt["n"] >= 100].sort_values(["target_altitude_category", "pct_ood_percentile"])
    for _, r in by_pt_reliable.iterrows():
        mahal = f"{r['pct_ood_mahalanobis']:.2f}%" if pd.notna(r["pct_ood_mahalanobis"]) else "n/a (ref chica)"
        iso = f"{r['pct_ood_isoforest']:.2f}%" if pd.notna(r["pct_ood_isoforest"]) else "n/a (ref chica)"
        lines.append(
            f"| {r['target_altitude_category']} | {r['AutoPitchType']} | {int(r['n']):,} | "
            f"{int(r['n_train_ref']):,} | {r['pct_ood_percentile']:.2f}% | {mahal} | {iso} |"
        )

    return "\n".join(lines)


def main() -> None:
    result = run()
    summary = summarize(result["scores"])
    baseline_summary = summarize_baseline(result["baseline"])
    n_groups_ref = len(result["reference"])
    n_groups_model = sum(1 for e in result["reference"].values() if "precision" in e)

    report_text = format_report(summary, baseline_summary, n_groups_ref, n_groups_model, MIN_N_FOR_MODEL)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)
    logger.info("\n%s", report_text)

    result["scores"].to_parquet(ROOT / "data" / "processed" / "altitude_counterfactual_ood_scores.parquet")


if __name__ == "__main__":
    main()
