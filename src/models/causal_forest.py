"""Causal Forest (econml CausalForestDML) para el efecto heterogeneo de la
altitud sobre run_value.

DECISION DE TRATAMIENTO: `altitude_level` se trata como CONTINUO ORDINAL
(0/1/2), no binario Extreme-vs-No. Justificacion:
1. Comparabilidad directa con el modelo bayesiano jerarquico
   (`reports/bayesian_hierarchical.md`), que uso exactamente el mismo diseño
   (`altitude_level` como covariable lineal 0/1/2) — el efecto "por nivel" de
   ambos metodos es la misma cantidad, y multiplicar por 2 da el equivalente
   No->Extreme en ambos casos, permitiendo una comparacion de magnitud
   apples-to-apples, no solo de signo/ranking.
2. Preserva TODO el dato (no hay que descartar las filas de Medium Altitude,
   ~24% del dataset, como se necesitaria para un tratamiento binario
   Extreme-vs-No).
3. CausalForestDML soporta tratamiento continuo nativamente
   (`discrete_treatment=False`) via un modelo de nuisance de T (regresion, no
   clasificacion) — no hay que forzar una discretizacion.
Costo: se asume que el efecto es aproximadamente lineal en `altitude_level`
(un salto de No->Medio tiene, en promedio, la mitad del efecto de No->Extreme)
— supuesto ya usado sin objeciones en el modelo bayesiano y consistente con
que `altitude_level` se trato como ordinal (no one-hot) en TODOS los modelos
LightGBM del proyecto (Stuff+/Location+/Pitching+).

Outcome: `run_value` (mismo target de siempre).

Heterogeneidad (X): InducedVertBreak, HorzBreak, SpinRate, RelSpeed,
AutoPitchType (one-hot, restringido a los mismos 7 tipos principales que
`src/models/bayesian_hierarchical.py` para comparabilidad), velo_diff_vs_fb.

Controles (W, para el doble ML, NO heterogeneidad): PitcherThrows (one-hot),
BatterSide (one-hot), Balls, Strikes, year.

Universo/exclusiones: identico a `data/processed/splits/splits.parquet`
(pitcher_holdout != excluded) + run_value no nulo + velo_diff_vs_fb no nulo
(pitcher-year sin fastball identificable, ver reports/arsenal_summary.md) +
PitcherThrows/BatterSide no nulos + AutoPitchType en los 7 tipos principales.

Muestra: ~180,000 pitches (mismo orden de magnitud que el modelo bayesiano),
estratificada por (AutoPitchType, altitude_category) para preservar la
composicion real del universo valido.
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[2]
ARSENAL_PATH = ROOT / "data" / "processed" / "pitches_with_arsenal.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits" / "splits.parquet"
COUNTERFACTUAL_SCORES_PATH = ROOT / "data" / "processed" / "altitude_counterfactual_scores.parquet"
MODEL_PATH = ROOT / "src" / "models" / "artifacts" / "causal_forest.pkl"
REPORT_PATH = ROOT / "reports" / "causal_forest.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PITCH_TYPES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
CLUSTER_MAP = {
    "Four-Seam": "fastballs",
    "Sinker": "fastballs",
    "Cutter": "fastballs",
    "Slider": "breaking",
    "Curveball": "breaking",
    "Changeup": "offspeed",
    "Splitter": "offspeed",
}

HETEROGENEITY_NUMERIC = ["InducedVertBreak", "HorzBreak", "SpinRate", "RelSpeed", "velo_diff_vs_fb"]
CONTROL_NUMERIC = ["Balls", "Strikes", "year"]

TARGET_SAMPLE_SIZE = 180_000
SAMPLE_SEED = 42

N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 50
MAX_SAMPLES = 0.45
FOREST_SEED = 42
CV_FOLDS = 3

CI_ALPHA = 0.06  # IC 94%, misma convencion que reports/bayesian_hierarchical.md

IVB_BIN_EDGES = [-np.inf, 0, 5, 10, 15, np.inf]
IVB_BIN_LABELS = ["<0 in", "0-5 in", "5-10 in", "10-15 in", ">15 in"]


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------


def load_valid_data() -> pd.DataFrame:
    cols = [
        "PitchUID", "pitcher_anon_id", "AutoPitchType", "altitude_category", "run_value",
        "InducedVertBreak", "HorzBreak", "SpinRate", "RelSpeed", "velo_diff_vs_fb",
        "PitcherThrows", "BatterSide", "Balls", "Strikes", "year",
    ]
    df = pd.read_parquet(ARSENAL_PATH, columns=cols)
    splits = pd.read_parquet(SPLITS_PATH, columns=["PitchUID", "pitcher_holdout"])
    df = df.merge(splits, on="PitchUID", how="inner")
    df = df.loc[df["pitcher_holdout"] != "excluded"].copy()

    df = df.loc[df["AutoPitchType"].isin(PITCH_TYPES)]
    df = df.dropna(subset=["run_value", "velo_diff_vs_fb", "PitcherThrows", "BatterSide"])

    altitude_level_map = {"No Altitude": 0, "Medium Altitude": 1, "Extreme Altitude": 2}
    df["altitude_level"] = df["altitude_category"].map(altitude_level_map).astype(float)

    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    frac = min(1.0, target_n / len(df))
    sampled = df.groupby(["AutoPitchType", "altitude_category"], observed=True, group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed), include_groups=True
    )
    return sampled.sort_values("PitchUID").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Diseño (X, W, T, Y)
# ---------------------------------------------------------------------------


def build_design_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list, list]:
    pitch_dummies = pd.get_dummies(df["AutoPitchType"].astype(str), prefix="pt")[
        [f"pt_{pt}" for pt in PITCH_TYPES]
    ]
    X = pd.concat([df[HETEROGENEITY_NUMERIC].reset_index(drop=True), pitch_dummies.reset_index(drop=True)], axis=1)
    x_cols = list(X.columns)

    throws_dummies = pd.get_dummies(df["PitcherThrows"], prefix="throws", drop_first=True)
    batter_dummies = pd.get_dummies(df["BatterSide"], prefix="batter", drop_first=True)
    W = pd.concat(
        [
            df[CONTROL_NUMERIC].reset_index(drop=True),
            throws_dummies.reset_index(drop=True),
            batter_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    w_cols = list(W.columns)

    T = df["altitude_level"].to_numpy(dtype=float)
    Y = df["run_value"].to_numpy(dtype=float)

    return X.astype(float), W.astype(float), T, Y, x_cols, w_cols


# ---------------------------------------------------------------------------
# 3. Ajuste del Causal Forest
# ---------------------------------------------------------------------------


def fit_causal_forest(X: pd.DataFrame, W: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> CausalForestDML:
    model_y = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, verbosity=-1, random_state=FOREST_SEED)
    model_t = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, verbosity=-1, random_state=FOREST_SEED)

    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_samples=MAX_SAMPLES,
        cv=CV_FOLDS,
        honest=True,
        inference=True,
        n_jobs=-1,
        random_state=FOREST_SEED,
    )
    est.fit(Y, T, X=X.to_numpy(), W=W.to_numpy())
    return est


# ---------------------------------------------------------------------------
# 4. Agregacion del CATE
# ---------------------------------------------------------------------------


def compute_cate(est: CausalForestDML, X: pd.DataFrame) -> np.ndarray:
    return est.effect(X.to_numpy(), T0=0, T1=1)


def aggregate_by_pitchtype(est: CausalForestDML, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pt in PITCH_TYPES:
        mask = (df["AutoPitchType"] == pt).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        X_sub = X.loc[mask].to_numpy()
        ate = est.ate(X_sub, T0=0, T1=1)
        lo, hi = est.ate_interval(X_sub, T0=0, T1=1, alpha=CI_ALPHA)
        rows.append(
            {
                "AutoPitchType": pt,
                "cluster": CLUSTER_MAP[pt],
                "n": n,
                "cate_per_level": float(ate),
                "ci_lo_per_level": float(lo),
                "ci_hi_per_level": float(hi),
                "cate_no_to_extreme": float(ate) * 2,
                "ci_lo_no_to_extreme": float(lo) * 2,
                "ci_hi_no_to_extreme": float(hi) * 2,
            }
        )
    return pd.DataFrame(rows).sort_values("cate_per_level", ascending=False).reset_index(drop=True)


def aggregate_by_ivb_bin(df: pd.DataFrame, cate_per_pitch: np.ndarray) -> pd.DataFrame:
    bins = pd.cut(df["InducedVertBreak"], bins=IVB_BIN_EDGES, labels=IVB_BIN_LABELS)
    tmp = pd.DataFrame({"ivb_bin": bins, "cate_per_level": cate_per_pitch})
    agg = tmp.groupby("ivb_bin", observed=True)["cate_per_level"].agg(["mean", "std", "count"]).reindex(IVB_BIN_LABELS)
    agg["cate_no_to_extreme"] = agg["mean"] * 2
    return agg


# ---------------------------------------------------------------------------
# 5. Comparacion contra los otros 3 metodos
# ---------------------------------------------------------------------------


def load_other_methods_rankings() -> pd.DataFrame:
    # --- contrafactual: delta_pitching_plus por AutoPitchType ---
    cf = pd.read_parquet(COUNTERFACTUAL_SCORES_PATH, columns=["AutoPitchType", "delta_pitching_plus"])
    cf_by_type = cf.groupby("AutoPitchType", observed=True)["delta_pitching_plus"].mean()

    # --- z-test estratificado: diff % relativo de is_whiff por AutoPitchType ---
    # (valores ya calculados en reports/altitude_whiff_stratified.md, reproducidos
    # aqui via el mismo query minimo, sin re-parsear el .md)
    targets = pd.read_parquet(
        ROOT / "data" / "processed" / "pitches_with_targets.parquet",
        columns=["is_whiff", "altitude_category", "AutoPitchType"],
    ).dropna(subset=["is_whiff", "altitude_category"])
    ztest_rows = {}
    for pt in PITCH_TYPES:
        sub = targets.loc[targets["AutoPitchType"] == pt]
        rate_ext = sub.loc[sub["altitude_category"] == "Extreme Altitude", "is_whiff"].mean()
        rate_no = sub.loc[sub["altitude_category"] == "No Altitude", "is_whiff"].mean()
        ztest_rows[pt] = 100 * (rate_ext - rate_no) / rate_no

    out = pd.DataFrame(
        {
            "delta_pitching_plus_contrafactual": cf_by_type,
            "diff_pct_whiff_ztest": pd.Series(ztest_rows),
        }
    )
    return out.reindex(PITCH_TYPES)


def build_ranking_table(cf_by_pt: pd.DataFrame, causal_by_pt: pd.DataFrame) -> pd.DataFrame:
    table = cf_by_pt.copy()
    table["cate_causal_forest"] = causal_by_pt.set_index("AutoPitchType")["cate_no_to_extreme"]
    table["cluster"] = causal_by_pt.set_index("AutoPitchType")["cluster"]

    # bayesiano: solo a nivel de cluster (no de AutoPitchType) — se repite el
    # valor del cluster para cada tipo, marcado explicitamente en el reporte
    bayes_cluster_delta = {
        "fastballs": 0.01817,
        "breaking": 0.01839,
        "offspeed": 0.01688,
    }  # de reports/bayesian_hierarchical.md seccion 3 (mean x2, No->Extreme)
    table["delta_run_value_bayesiano_cluster"] = table["cluster"].map(bayes_cluster_delta)

    # rankings (1 = mas afectado/peor para el pitcher, N = menos afectado/mejor)
    # convencion de signo: en contrafactual y causal forest, MAS NEGATIVO/MAS
    # POSITIVO respectivamente indica peor para el pitcher segun la metrica
    # (Pitching+ mas negativo = peor; run_value mas positivo = mas carreras
    # permitidas = peor) — se invierte el signo de delta_pitching_plus para
    # que "mayor valor = peor" en las 4 columnas antes de rankear.
    table["rank_contrafactual"] = (-table["delta_pitching_plus_contrafactual"]).rank(ascending=False, method="min")
    table["rank_ztest_whiff"] = (-table["diff_pct_whiff_ztest"]).rank(ascending=False, method="min")
    table["rank_causal_forest"] = table["cate_causal_forest"].rank(ascending=False, method="min")
    table["rank_bayesiano_cluster"] = table["delta_run_value_bayesiano_cluster"].rank(ascending=False, method="min")

    return table


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def format_report(
    n_valid: int,
    n_sample: int,
    x_cols: list,
    w_cols: list,
    fit_seconds: float,
    global_ate: float,
    global_ci: tuple,
    causal_by_pt: pd.DataFrame,
    ivb_agg: pd.DataFrame,
    ranking_table: pd.DataFrame,
) -> str:
    lines = [
        "# Causal Forest (econml CausalForestDML) — efecto heterogeneo de altitud sobre run_value",
        "",
        "## 0. Decisiones de diseño",
        "",
        "**Tratamiento: `altitude_level` continuo ordinal (0/1/2), no binario.** "
        "Mismo diseño que `reports/bayesian_hierarchical.md` — permite comparar "
        "magnitudes directamente (no solo signo/ranking) multiplicando por 2 "
        "para obtener el equivalente No->Extreme, y preserva el 100% del "
        "dataset (no hay que descartar Medium Altitude). Ver docstring de "
        "este script para la justificacion completa.",
        "",
        f"**Heterogeneidad (X):** {', '.join(HETEROGENEITY_NUMERIC)} + AutoPitchType (one-hot, "
        f"{len(PITCH_TYPES)} tipos) — {len(x_cols)} columnas totales.",
        f"**Controles (W, doble ML):** {', '.join(CONTROL_NUMERIC)} + PitcherThrows + BatterSide "
        f"(one-hot) — {len(w_cols)} columnas totales.",
        "",
        "## 1. Datos y muestra",
        "",
        f"- Pitches validos (mismo filtro `pitcher_holdout != excluded`, mas "
        f"run_value/velo_diff_vs_fb/PitcherThrows/BatterSide no nulos, mas "
        f"AutoPitchType restringido a los {len(PITCH_TYPES)} tipos principales "
        f"— mismos que `reports/bayesian_hierarchical.md`): **{n_valid:,}**",
        f"- Muestra usada para el forest: **{n_sample:,}** pitches "
        f"({100*n_sample/n_valid:.1f}% del universo valido, estratificada por "
        "AutoPitchType x altitude_category)",
        "",
        "## 2. Ajuste del modelo",
        "",
        f"`CausalForestDML(model_y=LGBMRegressor, model_t=LGBMRegressor, "
        f"discrete_treatment=False, n_estimators={N_ESTIMATORS}, "
        f"min_samples_leaf={MIN_SAMPLES_LEAF}, max_samples={MAX_SAMPLES}, cv={CV_FOLDS}, "
        f"honest=True)`. Tiempo de ajuste: {fit_seconds/60:.1f} min.",
        "",
        f"**ATE global (efecto promedio, por nivel de altitude_level):** "
        f"{global_ate:+.5f} runs/pitch, IC {100*(1-CI_ALPHA):.0f}% "
        f"[{global_ci[0]:+.5f}, {global_ci[1]:+.5f}]. Equivalente No->Extreme "
        f"(x2): {2*global_ate:+.5f} [{2*global_ci[0]:+.5f}, {2*global_ci[1]:+.5f}].",
        "",
        "## 3. CATE agregado por AutoPitchType",
        "",
        "`cate_no_to_extreme` = efecto causal condicional promedio de mover ese "
        "tipo de pitch de No Altitude a Extreme Altitude (runs adicionales "
        "esperados por pitch). IC calculado con la inferencia asintotica nativa "
        f"de CausalForestDML (`ate_interval`, alpha={CI_ALPHA} -> "
        f"{100*(1-CI_ALPHA):.0f}%).",
        "",
        "| AutoPitchType | n | CATE por nivel | CATE No->Extreme | IC 94% (No->Extreme) |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in causal_by_pt.iterrows():
        lines.append(
            f"| {r['AutoPitchType']} | {int(r['n']):,} | {r['cate_per_level']:+.5f} | "
            f"{r['cate_no_to_extreme']:+.5f} | [{r['ci_lo_no_to_extreme']:+.5f}, {r['ci_hi_no_to_extreme']:+.5f}] |"
        )

    lines += [
        "",
        "## 4. CATE agregado por rango de InducedVertBreak",
        "",
        "| rango IVB | n | CATE por nivel (media) | sd | CATE No->Extreme |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, r in ivb_agg.iterrows():
        if pd.isna(r["mean"]):
            continue
        lines.append(f"| {label} | {int(r['count']):,} | {r['mean']:+.5f} | {r['std']:.5f} | {r['cate_no_to_extreme']:+.5f} |")

    lines += [
        "",
        "## 5. Comparacion de heterogeneidad contra los otros 3 metodos",
        "",
        "**Ranking por metodo** (1 = mas afectado por altitud / peor para el "
        "pitcher, "
        f"{len(PITCH_TYPES)} = menos afectado / mejor). Convencion de signo "
        "unificada antes de rankear: para contrafactual y z-test, mas negativo "
        "= peor; para el causal forest y el bayesiano, mas positivo (mas "
        "runs permitidos) = peor.",
        "",
        "**Nota sobre el bayesiano:** ese modelo se ajusto a nivel de CLUSTER "
        "(fastballs/breaking/offspeed), no de AutoPitchType individual — su "
        "columna aqui repite el valor del cluster para cada tipo que pertenece "
        "a el (marcado explicitamente), no es una estimacion pitch-type-especifica "
        "independiente como las otras 3.",
        "",
        "| AutoPitchType | cluster | rank contrafactual | rank z-test (whiff) | "
        "rank causal forest | rank bayesiano (cluster) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pt in ranking_table.index:
        r = ranking_table.loc[pt]
        lines.append(
            f"| {pt} | {r['cluster']} | {int(r['rank_contrafactual'])} | "
            f"{int(r['rank_ztest_whiff'])} | {int(r['rank_causal_forest'])} | "
            f"{int(r['rank_bayesiano_cluster'])} |"
        )

    # --- verificacion explicita: Sinker peor, Curveball/Splitter mejor? ---
    worst_cf = ranking_table["rank_contrafactual"].idxmin()
    worst_zt = ranking_table["rank_ztest_whiff"].idxmin()
    worst_cft = ranking_table["rank_causal_forest"].idxmin()
    best_cf = ranking_table.loc[["Curveball", "Splitter"], "rank_contrafactual"].mean()
    best_zt = ranking_table.loc[["Curveball", "Splitter"], "rank_ztest_whiff"].mean()
    best_cft = ranking_table.loc[["Curveball", "Splitter"], "rank_causal_forest"].mean()
    avg_rank_worst = ranking_table.loc[["Curveball", "Splitter"]].mean(numeric_only=True)

    corr_cf_cft = ranking_table["rank_contrafactual"].corr(ranking_table["rank_causal_forest"], method="spearman")
    corr_zt_cft = ranking_table["rank_ztest_whiff"].corr(ranking_table["rank_causal_forest"], method="spearman")
    corr_cf_zt = ranking_table["rank_contrafactual"].corr(ranking_table["rank_ztest_whiff"], method="spearman")

    lines += [
        "",
        "### Lectura",
        "",
        f"- Tipo de pitch mas afectado (rank 1) segun contrafactual: **{worst_cf}**; "
        f"segun z-test de whiff: **{worst_zt}**; segun causal forest: **{worst_cft}**.",
        f"- Rank promedio de Curveball+Splitter (los 2 tipos que el contrafactual "
        f"identifico como los que MENOS pierden): contrafactual={best_cf:.1f}, "
        f"z-test={best_zt:.1f}, causal forest={best_cft:.1f} (de {len(PITCH_TYPES)} "
        "tipos — mas cerca de 7 = mas consistente con 'los que menos pierden').",
        f"- Correlacion de Spearman entre rankings: contrafactual vs causal "
        f"forest = {corr_cf_cft:+.3f}; z-test vs causal forest = {corr_zt_cft:+.3f}; "
        f"contrafactual vs z-test = {corr_cf_zt:+.3f}.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    df_full = load_valid_data()
    logger.info("Pitches validos: %s | pitchers: %s", f"{len(df_full):,}", f"{df_full['pitcher_anon_id'].nunique():,}")

    sample = stratified_sample(df_full, TARGET_SAMPLE_SIZE, SAMPLE_SEED)
    logger.info("Muestra: %s pitches", f"{len(sample):,}")

    X, W, T, Y, x_cols, w_cols = build_design_matrices(sample)
    logger.info("X: %s columnas | W: %s columnas", len(x_cols), len(w_cols))

    logger.info("Ajustando CausalForestDML...")
    t_fit0 = time.time()
    est = fit_causal_forest(X, W, T, Y)
    fit_seconds = time.time() - t_fit0
    logger.info("Ajuste completo en %.1f min", fit_seconds / 60)

    global_ate = float(est.ate(X.to_numpy(), T0=0, T1=1))
    global_ci = tuple(float(v) for v in est.ate_interval(X.to_numpy(), T0=0, T1=1, alpha=CI_ALPHA))
    logger.info("ATE global (por nivel): %.5f, IC94%%=%s", global_ate, global_ci)

    cate_per_pitch = compute_cate(est, X)

    causal_by_pt = aggregate_by_pitchtype(est, sample, X)
    logger.info("CATE por AutoPitchType calculado")

    ivb_agg = aggregate_by_ivb_bin(sample, cate_per_pitch)

    cf_and_ztest = load_other_methods_rankings()
    ranking_table = build_ranking_table(cf_and_ztest, causal_by_pt)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "estimator": est,
                "x_cols": x_cols,
                "w_cols": w_cols,
                "pitch_types": PITCH_TYPES,
                "sample_size": len(sample),
                "seed": SAMPLE_SEED,
            },
            f,
        )
    logger.info("Modelo guardado en %s", MODEL_PATH)

    report_text = format_report(
        n_valid=len(df_full),
        n_sample=len(sample),
        x_cols=x_cols,
        w_cols=w_cols,
        fit_seconds=fit_seconds,
        global_ate=global_ate,
        global_ci=global_ci,
        causal_by_pt=causal_by_pt,
        ivb_agg=ivb_agg,
        ranking_table=ranking_table,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Reporte guardado en %s", REPORT_PATH)
    logger.info("Tiempo total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
