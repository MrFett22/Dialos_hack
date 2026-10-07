"""Pitching+ v1 (LightGBM) — modelo combinado: TODAS las features de Stuff+ +
Location+ + arsenal, para los 8 targets ya validados en Semana 2.

Misma estructura que src/models/stuff_plus.py (un LGBMClassifier/LGBMRegressor
por target, mismos hiperparametros por defecto, sin tuning), pero con el
feature set completo:

- Stuff+ (fisica pura + altitud + residual de Magnus): ver
  src.models.stuff_plus.NUMERIC_FEATURES/CATEGORICAL_FEATURES
- Location+ (ubicacion + conteo): PlateLocHeight, PlateLocSide, Balls, Strikes
- Arsenal (comparacion contra el propio repertorio del pitcher): velo/ivb/hb_diff_vs_fb,
  tunnel_similarity, pitch_usage_pct, prev_pitch_type

Targets (los 8 de Semana 2): is_whiff, is_chase, is_called_strike,
is_weak_contact, is_groundball, is_hard_contact, run_value, is_out_in_pa.

Datos: data/processed/pitches_with_arsenal.parquet (targets + arsenal +
ubicacion/conteo + fisica base + altitude_level) unido por PitchUID con
data/processed/pitches_with_physics.parquet (SOLO para magnus_residual_vert/horz,
que no viven en el parquet de arsenal) y con data/processed/splits/splits.parquet.

Validacion: se entrena y evalua CADA target en los 3 esquemas de holdout
(pitcher_holdout, season_holdout, altitude_holdout) para ver que tan sensible
es cada target al tipo de generalizacion exigida. El modelo que se GUARDA en
el artefacto es el entrenado con pitcher_holdout (el esquema mas estricto,
mismo criterio que Stuff+/Location+ v1).

Ademas se compara contra los sub-modelos individuales ya entrenados (Stuff+
para is_whiff, Location+ para is_called_strike/is_chase) sobre el mismo test
de pitcher_holdout, para ver si el modelo combinado mejora sobre tener el
arsenal completo de features o si son redundantes.
"""

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.stuff_plus import (
    CATEGORICAL_FEATURES as STUFF_CATEGORICAL_FEATURES,
)
from src.models.stuff_plus import (
    FEATURE_COLUMNS as STUFF_FEATURE_COLUMNS,
)
from src.models.stuff_plus import (
    LGB_PARAMS_COMMON,
    MODEL_PATH as STUFF_MODEL_PATH,
    SEED,
    evaluate_classification,
    evaluate_regression,
)
from src.models.stuff_plus import (
    NUMERIC_FEATURES as STUFF_NUMERIC_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
ARSENAL_PATH = ROOT / "data" / "processed" / "pitches_with_arsenal.parquet"
PHYSICS_PATH = ROOT / "data" / "processed" / "pitches_with_physics.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits" / "splits.parquet"
LOCATION_MODEL_PATH = ROOT / "src" / "models" / "artifacts" / "location_plus_v1.pkl"
MODEL_PATH = ROOT / "src" / "models" / "artifacts" / "pitching_plus_v1.pkl"
REPORT_PATH = ROOT / "reports" / "pitching_plus_baseline.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LOCATION_NUMERIC_FEATURES = ["PlateLocHeight", "PlateLocSide", "Balls", "Strikes"]
ARSENAL_NUMERIC_FEATURES = [
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "tunnel_similarity",
    "pitch_usage_pct",
]
ARSENAL_CATEGORICAL_FEATURES = ["prev_pitch_type"]

NUMERIC_FEATURES = STUFF_NUMERIC_FEATURES + LOCATION_NUMERIC_FEATURES + ARSENAL_NUMERIC_FEATURES
CATEGORICAL_FEATURES = STUFF_CATEGORICAL_FEATURES + ARSENAL_CATEGORICAL_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BINARY_TARGETS = [
    "is_whiff",
    "is_chase",
    "is_called_strike",
    "is_weak_contact",
    "is_groundball",
    "is_hard_contact",
    "is_out_in_pa",
]
REGRESSION_TARGETS = ["run_value"]
ALL_TARGETS = BINARY_TARGETS + REGRESSION_TARGETS

HOLDOUT_SCHEMES = ["pitcher_holdout", "season_holdout", "altitude_holdout"]
OFFICIAL_SCHEME = "pitcher_holdout"

# targets con un sub-modelo individual ya entrenado, contra el que se compara
SUBMODEL_COMPARISON = {
    "is_whiff": "Stuff+ v1",
    "is_called_strike": "Location+ v1",
    "is_chase": "Location+ v1",
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ARSENAL_PATH)
    physics_extra = pd.read_parquet(
        PHYSICS_PATH, columns=["PitchUID", "magnus_residual_vert", "magnus_residual_horz"]
    )
    splits = pd.read_parquet(SPLITS_PATH)

    df = df.merge(physics_extra, on="PitchUID", how="inner").merge(splits, on="PitchUID", how="inner")
    df = df.loc[df["pitcher_holdout"] != "excluded"].copy()

    df["EffectiveVelo"] = df["EffectiveVelo"].astype(float)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object").fillna("Unknown").astype("category")

    return df


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    sub = df.dropna(subset=[target])
    return sub[FEATURE_COLUMNS], sub[target]


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def train_model(X_train: pd.DataFrame, y_train: pd.Series, is_binary: bool):
    if is_binary:
        model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **LGB_PARAMS_COMMON)
    else:
        model = lgb.LGBMRegressor(objective="regression", verbosity=-1, **LGB_PARAMS_COMMON)
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    return model


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, is_binary: bool) -> dict:
    if is_binary:
        pred = model.predict_proba(X)[:, 1]
        return evaluate_classification(y, pred)
    pred = model.predict(X)
    return evaluate_regression(y, pred)


# ---------------------------------------------------------------------------
# Comparacion contra sub-modelos individuales (pitcher_holdout test)
# ---------------------------------------------------------------------------


def evaluate_stuff_plus_submodel(df_test: pd.DataFrame) -> dict:
    with open(STUFF_MODEL_PATH, "rb") as f:
        art = pickle.load(f)
    model = art["is_whiff_model"]
    sub = df_test.dropna(subset=["is_whiff"])
    X = sub[STUFF_FEATURE_COLUMNS]
    pred = model.predict_proba(X)[:, 1]
    return evaluate_classification(sub["is_whiff"], pred)


def evaluate_location_plus_submodel(df_test: pd.DataFrame, target: str) -> dict:
    with open(LOCATION_MODEL_PATH, "rb") as f:
        art = pickle.load(f)
    model = art["models"][target]
    batter_categories = art["batter_categories"]
    pitch_categories = art["pitch_categories"]

    sub = df_test.dropna(subset=["BatterSide", target]).copy()
    sub = sub[sub["BatterSide"].astype(str).isin(batter_categories)]
    batter_code = pd.Categorical(sub["BatterSide"].astype(str), categories=batter_categories).codes
    pitch_code = pd.Categorical(sub["AutoPitchType"].astype(str), categories=pitch_categories).codes

    X = np.column_stack(
        [
            sub["PlateLocSide"].to_numpy(dtype=float),
            sub["PlateLocHeight"].to_numpy(dtype=float),
            sub["Balls"].to_numpy(dtype=float),
            sub["Strikes"].to_numpy(dtype=float),
            batter_code,
            pitch_code,
        ]
    )
    pred = model.predict_proba(X)
    return evaluate_classification(sub[target], pred)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> tuple[dict, dict]:
    df = load_data()
    logger.info("Filas disponibles (pitcher_holdout != excluded): %s", f"{len(df):,}")

    results: dict[str, dict[str, dict]] = {t: {} for t in ALL_TARGETS}
    official_models: dict[str, object] = {}

    for target in ALL_TARGETS:
        is_binary = target in BINARY_TARGETS
        for scheme in HOLDOUT_SCHEMES:
            X_all, y_all = prepare_xy(df, target)
            scheme_col = df.loc[X_all.index, scheme]
            train_idx = X_all.index[scheme_col == "train"]
            test_idx = X_all.index[scheme_col == "test"]

            X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
            X_test, y_test = X_all.loc[test_idx], y_all.loc[test_idx]

            model = train_model(X_train, y_train, is_binary)
            metrics_test = evaluate_model(model, X_test, y_test, is_binary)
            metrics_train = evaluate_model(model, X_train, y_train, is_binary)

            results[target][scheme] = {
                "test": metrics_test,
                "train": metrics_train,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            if scheme == OFFICIAL_SCHEME:
                official_models[target] = model

            logger.info("Entrenado: %s / %s (n_train=%s, n_test=%s)", target, scheme, f"{len(train_idx):,}", f"{len(test_idx):,}")

    # --- comparacion contra sub-modelos individuales, sobre pitcher_holdout test ---
    test_df_official = df.loc[df[OFFICIAL_SCHEME] == "test"]
    submodel_comparison = {}
    submodel_comparison["is_whiff"] = evaluate_stuff_plus_submodel(test_df_official)
    submodel_comparison["is_called_strike"] = evaluate_location_plus_submodel(test_df_official, "is_called_strike")
    submodel_comparison["is_chase"] = evaluate_location_plus_submodel(test_df_official, "is_chase")

    report = {
        "n_total_valid": len(df),
        "results": results,
        "submodel_comparison": submodel_comparison,
    }

    artifact = {
        "models": official_models,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_targets": BINARY_TARGETS,
        "regression_targets": REGRESSION_TARGETS,
        "official_scheme": OFFICIAL_SCHEME,
        "seed": SEED,
        "lgb_params": LGB_PARAMS_COMMON,
    }
    return artifact, report


def format_report(report: dict) -> str:
    results = report["results"]
    sc = report["submodel_comparison"]

    lines = [
        "# Pitching+ v1 — modelo combinado (LightGBM, Stuff+ + Location+ + arsenal)",
        "",
        f"Filas usadas (pitcher_holdout != excluded): {report['n_total_valid']:,}",
        "",
        f"Features ({len(FEATURE_COLUMNS)}): " + ", ".join(FEATURE_COLUMNS),
        "",
        "## Tabla comparativa: 8 targets x 3 esquemas de holdout (metricas de test)",
        "",
        "### Targets binarios (AUC / log-loss)",
        "",
        "| target | pitcher_holdout AUC | log-loss | season_holdout AUC | log-loss | altitude_holdout AUC | log-loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in BINARY_TARGETS:
        row = [target]
        for scheme in HOLDOUT_SCHEMES:
            m = results[target][scheme]["test"]
            row += [f"{m['auc']:.4f}", f"{m['log_loss']:.4f}"]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### run_value (RMSE / R²)",
        "",
        "| target | pitcher_holdout RMSE | R² | season_holdout RMSE | R² | altitude_holdout RMSE | R² |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in REGRESSION_TARGETS:
        row = [target]
        for scheme in HOLDOUT_SCHEMES:
            m = results[target][scheme]["test"]
            row += [f"{m['rmse']:.4f}", f"{m['r2']:.4f}"]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### n de train/test por esquema (referencia)",
        "",
        "| target | pitcher train/test | season train/test | altitude train/test |",
        "|---|---|---|---|",
    ]
    for target in ALL_TARGETS:
        cells = []
        for scheme in HOLDOUT_SCHEMES:
            r = results[target][scheme]
            cells.append(f"{r['n_train']:,}/{r['n_test']:,}")
        lines.append(f"| {target} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Sensibilidad por esquema — lectura",
        "",
    ]
    for target in BINARY_TARGETS:
        aucs = {s: results[target][s]["test"]["auc"] for s in HOLDOUT_SCHEMES}
        spread = max(aucs.values()) - min(aucs.values())
        worst_scheme = min(aucs, key=aucs.get)
        if spread >= 0.03:
            lines.append(
                f"- **{target}**: AUC varia {spread:.4f} entre esquemas (peor en `{worst_scheme}`, "
                f"AUC={aucs[worst_scheme]:.4f}) — sensible al tipo de holdout, revisar antes de "
                "generalizar a pitchers/temporadas/estadios no vistos."
            )
        else:
            lines.append(
                f"- {target}: AUC estable entre esquemas (spread={spread:.4f}) — generaliza de "
                "forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud)."
            )
    rv_r2 = {s: results["run_value"][s]["test"]["r2"] for s in HOLDOUT_SCHEMES}
    lines.append(
        f"- **run_value**: R² por esquema = {', '.join(f'{s}={v:.4f}' for s, v in rv_r2.items())} — "
        "comparar contra el baseline por count de Stuff+ v1 (R²=0.0205) en reports/stuff_plus_baseline.md; "
        "aqui con ubicacion+conteo+arsenal completos se espera una mejora real, a diferencia del "
        "Stuff+ puro (que quedo peor que ese baseline)."
    )

    lines += [
        "",
        "## Comparacion contra sub-modelos individuales (pitcher_holdout test)",
        "",
        "| target | Pitching+ (combinado) AUC | Sub-modelo individual | AUC individual | Δ AUC |",
        "|---|---:|---|---:|---:|",
    ]
    for target, submodel_name in SUBMODEL_COMPARISON.items():
        combined_auc = results[target][OFFICIAL_SCHEME]["test"]["auc"]
        individual_auc = sc[target]["auc"]
        delta = combined_auc - individual_auc
        lines.append(
            f"| {target} | {combined_auc:.4f} | {submodel_name} | {individual_auc:.4f} | {delta:+.4f} |"
        )

    lines += [
        "",
        "**Interpretacion:** un Δ AUC grande y positivo indica que agregar el resto del "
        "arsenal de features (fuera del dominio del sub-modelo individual) SI aporta señal "
        "incremental real — no son redundantes. Un Δ cercano a 0 indica que el sub-modelo "
        "individual ya capturaba toda la señal disponible en su propio dominio, y las "
        "features de los otros sub-modelos no agregan nada nuevo para ESE target especifico "
        "(coherente con la logica de separar Stuff+/Location+/Pitching+ en primer lugar: cada "
        "uno deberia ser mejor en su propio dominio de target).",
        "",
        f"Modelo guardado en `src/models/artifacts/pitching_plus_v1.pkl` — 8 modelos "
        f"(uno por target), todos entrenados sobre `{OFFICIAL_SCHEME}` (el esquema oficial, "
        "igual criterio que Stuff+/Location+ v1). Las metricas de season_holdout/altitude_holdout "
        "de arriba son solo de validacion/diagnostico, esos modelos no se guardan por separado.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    artifact, report = run()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Guardado modelo en %s", MODEL_PATH)

    report_text = format_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)

    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
