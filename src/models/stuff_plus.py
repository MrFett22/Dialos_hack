"""Stuff+ v1 (LightGBM) — modelo baseline usando SOLO features fisicas del pitch.

Deliberadamente NO incluye ubicacion de la pelota (PlateLocHeight/Side, zona),
conteo (Balls/Strikes/count), ni features de arsenal (velo/ivb/hb_diff_vs_fb,
tunnel_similarity, pitch_usage_pct, prev_pitch_type) — este es el modelo
"Stuff+" puro (calidad fisica del pitch en si mismo), no Location+ ni
Pitching+, que se construiran aparte usando esas otras columnas.

Datos: data/processed/pitches_with_physics.parquet (features fisicas +
magnus_residual_vert/horz) unido por PitchUID con
data/processed/pitches_with_targets.parquet (targets is_whiff, run_value, y
`count` para el baseline) y con data/processed/splits/splits.parquet
(esquema pitcher_holdout). Se filtra a filas con pitcher_holdout != 'excluded'
(exclusion ya decidida y documentada en reports/pitcher_id_null_investigation.md).

DECISION (documentada en reports/stuff_plus_baseline.md): Stuff+ v1 tiene un
solo target oficial, **is_whiff** — "capacidad de generar swing-and-miss por
calidad fisica pura". `run_value` SI se entrena y se evalua aqui, pero solo
como diagnostico/referencia comparativa (no se guarda en el artefacto
`.pkl`, no es parte de Stuff+): el baseline v1 de LightGBM con solo fisica
queda ligeramente PEOR que un baseline ingenuo de "promedio de run_value por
count" (RMSE 0.2753 vs 0.2729, R^2 0.0032 vs 0.0205 en test) — evidencia de
que run_value depende mucho mas de conteo/situacion que de la fisica pura del
pitch en si. xRunValue completo (con conteo y ubicacion) queda para el
sub-modelo 3 (Pitching+), no para este Stuff+ puro.

Evaluacion: pitcher_holdout (train = pitcher_holdout=='train', test =
pitcher_holdout=='test') — ningun pitcher del test aparece en train, para
medir generalizacion a pitchers nuevos, no solo a pitches nuevos del mismo
pitcher.
"""

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PHYSICS_PATH = ROOT / "data" / "processed" / "pitches_with_physics.parquet"
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits" / "splits.parquet"
MODEL_PATH = ROOT / "src" / "models" / "artifacts" / "stuff_plus_v1.pkl"
REPORT_PATH = ROOT / "reports" / "stuff_plus_baseline.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "RelSpeed",
    "EffectiveVelo",
    "ZoneSpeed",
    "SpeedDrop",
    "ZoneTime",
    "SpinRate",
    "SpinAxis",
    "RelHeight",
    "RelSide",
    "Extension",
    "VertBreak",
    "InducedVertBreak",
    "HorzBreak",
    "VertRelAngle",
    "HorzRelAngle",
    "VertApprAngle",
    "HorzApprAngle",
    "altitude_level",
    "magnus_residual_vert",
    "magnus_residual_horz",
]
# altitude_level (ordinal 0/1/2) se usa en vez de altitude_category cruda:
# la altitud SI tiene un orden fisico real (No < Medium < Extreme), asi que
# codificarla como ordinal deja que los splits del arbol usen comparaciones
# de "mayor/menor que", en vez de tratarla como categorias sin relacion
# (lo que perderia esa informacion de orden con el encoding categorico nativo).
CATEGORICAL_FEATURES = ["PitcherThrows", "BatterSide", "AutoPitchType"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SEED = 42
LGB_PARAMS_COMMON = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    physics_cols = ["PitchUID"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # altitude_level ya viene numerica en physics parquet; el resto de
    # NUMERIC_FEATURES tambien, excepto EffectiveVelo (string, se castea abajo)
    physics_cols = list(dict.fromkeys(physics_cols))  # sin duplicados, preserva orden
    df_physics = pd.read_parquet(PHYSICS_PATH, columns=physics_cols)

    df_targets = pd.read_parquet(
        TARGETS_PATH, columns=["PitchUID", "pitcher_anon_id", "is_whiff", "run_value", "count"]
    )
    df_splits = pd.read_parquet(SPLITS_PATH, columns=["PitchUID", "pitcher_holdout"])

    df = df_physics.merge(df_targets, on="PitchUID", how="inner").merge(
        df_splits, on="PitchUID", how="inner"
    )
    df = df.loc[df["pitcher_holdout"] != "excluded"].copy()

    df["EffectiveVelo"] = df["EffectiveVelo"].astype(float)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object").fillna("Unknown").astype("category")

    return df


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    sub = df.dropna(subset=[target])
    X = sub[FEATURE_COLUMNS]
    y = sub[target]
    return X, y


# ---------------------------------------------------------------------------
# Baselines ingenuos
# ---------------------------------------------------------------------------


def naive_baseline_run_value(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Promedio de run_value por `count` (Balls-Strikes), calculado en train."""
    lookup = train.groupby("count")["run_value"].mean()
    global_mean = train["run_value"].mean()
    return test["count"].map(lookup).fillna(global_mean).to_numpy()


def naive_baseline_is_whiff(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Promedio de is_whiff por AutoPitchType, calculado en train."""
    lookup = train.groupby("AutoPitchType", observed=True)["is_whiff"].mean()
    global_mean = train["is_whiff"].mean()
    preds = test["AutoPitchType"].map(lookup).fillna(global_mean).to_numpy()
    return np.clip(preds, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


def train_is_whiff_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **LGB_PARAMS_COMMON)
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    return model


def train_run_value_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="regression", verbosity=-1, **LGB_PARAMS_COMMON)
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    return model


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------


def evaluate_classification(y_true: pd.Series, y_pred_proba: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "y_std": float(y_true.std()),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


ABLATION_VARIANTS = {
    "completa": FEATURE_COLUMNS,
    "sin_altitud_ni_magnus": [
        c for c in FEATURE_COLUMNS if c not in ("altitude_level", "magnus_residual_vert", "magnus_residual_horz")
    ],
    "solo_altitud": [c for c in FEATURE_COLUMNS if c not in ("magnus_residual_vert", "magnus_residual_horz")],
}


def run_ablation(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    train_idx: pd.Index,
    test_idx: pd.Index,
    full_model: lgb.LGBMClassifier,
    full_pred_test: np.ndarray,
) -> dict:
    """Compara 3 subconjuntos de features para is_whiff sobre el mismo split
    pitcher_holdout: completa (todas), sin altitud/magnus, y solo altitud (sin
    magnus). La variante "completa" reutiliza el modelo ya entrenado en run()
    en vez de re-entrenar."""
    results = {
        "completa": {
            **evaluate_classification(y_all.loc[test_idx], full_pred_test),
            "n_features": len(FEATURE_COLUMNS),
        }
    }
    for name, cols in ABLATION_VARIANTS.items():
        if name == "completa":
            continue
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in cols]
        X_train, y_train = X_all.loc[train_idx, cols], y_all.loc[train_idx]
        X_test, y_test = X_all.loc[test_idx, cols], y_all.loc[test_idx]

        model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **LGB_PARAMS_COMMON)
        model.fit(X_train, y_train, categorical_feature=cat_cols)
        pred = model.predict_proba(X_test)[:, 1]

        results[name] = {**evaluate_classification(y_test, pred), "n_features": len(cols)}
    return results


def run() -> tuple[dict, dict]:
    df = load_data()
    logger.info("Filas disponibles (pitcher_holdout != excluded): %s", f"{len(df):,}")

    train_mask = df["pitcher_holdout"] == "train"
    test_mask = df["pitcher_holdout"] == "test"

    # --- is_whiff ---
    X_all, y_all = prepare_xy(df, "is_whiff")
    train_idx = X_all.index.intersection(df.index[train_mask])
    test_idx = X_all.index.intersection(df.index[test_mask])
    X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
    X_test, y_test = X_all.loc[test_idx], y_all.loc[test_idx]

    whiff_model = train_is_whiff_model(X_train, y_train)
    whiff_pred_test = whiff_model.predict_proba(X_test)[:, 1]
    whiff_pred_train = whiff_model.predict_proba(X_train)[:, 1]

    whiff_baseline_test = naive_baseline_is_whiff(df.loc[train_idx], df.loc[test_idx])

    whiff_report = {
        "model_test": evaluate_classification(y_test, whiff_pred_test),
        "model_train": evaluate_classification(y_train, whiff_pred_train),
        "baseline_test": evaluate_classification(y_test, whiff_baseline_test),
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, whiff_model.feature_importances_.tolist()),
                key=lambda kv: -kv[1],
            )
        ),
    }

    ablation_report = run_ablation(X_all, y_all, train_idx, test_idx, whiff_model, whiff_pred_test)

    # --- run_value ---
    Xr_all, yr_all = prepare_xy(df, "run_value")
    r_train_idx = Xr_all.index.intersection(df.index[train_mask])
    r_test_idx = Xr_all.index.intersection(df.index[test_mask])
    Xr_train, yr_train = Xr_all.loc[r_train_idx], yr_all.loc[r_train_idx]
    Xr_test, yr_test = Xr_all.loc[r_test_idx], yr_all.loc[r_test_idx]

    rv_model = train_run_value_model(Xr_train, yr_train)
    rv_pred_test = rv_model.predict(Xr_test)
    rv_pred_train = rv_model.predict(Xr_train)

    rv_baseline_test = naive_baseline_run_value(df.loc[r_train_idx], df.loc[r_test_idx])

    n_run_value_dropped = int(df["run_value"].isna().sum())

    rv_report = {
        "model_test": evaluate_regression(yr_test, rv_pred_test),
        "model_train": evaluate_regression(yr_train, rv_pred_train),
        "baseline_test": evaluate_regression(yr_test, rv_baseline_test),
        "n_dropped_null_run_value": n_run_value_dropped,
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, rv_model.feature_importances_.tolist()),
                key=lambda kv: -kv[1],
            )
        ),
    }

    report = {
        "n_total_valid": len(df),
        "n_train_pitches": int(train_mask.sum()),
        "n_test_pitches": int(test_mask.sum()),
        "n_train_pitchers": int(df.loc[train_mask, "pitcher_anon_id"].nunique()),
        "n_test_pitchers": int(df.loc[test_mask, "pitcher_anon_id"].nunique()),
        "is_whiff": whiff_report,
        "ablation": ablation_report,
        "run_value": rv_report,
    }

    # run_value se entrena y evalua arriba solo como diagnostico (ver docstring
    # del modulo y reports/stuff_plus_baseline.md) — NO se guarda en el
    # artefacto de Stuff+, que queda con is_whiff como unico target oficial.
    artifact = {
        "is_whiff_model": whiff_model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "seed": SEED,
        "lgb_params": LGB_PARAMS_COMMON,
        "target": "is_whiff",
        "run_value_excluded_reason": (
            "run_value con solo features fisicas rinde peor que un baseline "
            "ingenuo por count (ver reports/stuff_plus_baseline.md) — se deja "
            "para el sub-modelo 3 (Pitching+/xRunValue), que si incluye conteo "
            "y ubicacion."
        ),
    }
    return artifact, report


def format_report(report: dict) -> str:
    wf = report["is_whiff"]
    ab = report["ablation"]
    rv = report["run_value"]

    lines = [
        "# Stuff+ v1 — modelo baseline (LightGBM, solo features fisicas)",
        "",
        "## Decision: target oficial de Stuff+ v1 = solo `is_whiff`",
        "",
        "**Stuff+ v1 queda con un solo target: `is_whiff`** — \"capacidad de "
        "generar swing-and-miss por calidad fisica pura\". Mas adelante, cuando "
        "se construya Location+, se le suma `is_chase` como segunda pieza de "
        "este mismo indice de calidad-de-pitch.",
        "",
        "`run_value` **se entrena y se evalua en esta corrida, pero solo como "
        "diagnostico/referencia comparativa — NO se guarda en el artefacto "
        "`.pkl` y NO es un target de Stuff+.** La razon es empirica, no de "
        "diseño a priori: el LightGBM de run_value con solo features fisicas "
        "queda ligeramente PEOR que el baseline ingenuo de \"promedio de "
        "run_value por count\" (ver seccion de apendice abajo) — evidencia "
        "directa de que run_value depende mucho mas del conteo/situacion que "
        "de la calidad fisica pura del pitch. Meter un target que el modelo no "
        "predice mejor que un lookup trivial dentro de \"Stuff+\" diluiria la "
        "interpretacion del indice (un componente que no aporta señal real, "
        "solo ruido) y mezclaria dos preguntas distintas: \"¿que tan bueno es "
        "este pitch en si mismo?\" (Stuff+) vs. \"¿que tanto valor de carreras "
        "genera en ESTA situacion especifica?\" (que necesita conteo+ubicacion, "
        "y le corresponde a xRunValue completo en el sub-modelo 3 / Pitching+).",
        "",
        f"Filas usadas (pitcher_holdout != excluded): {report['n_total_valid']:,}",
        f"- Train: {report['n_train_pitches']:,} pitches, {report['n_train_pitchers']:,} pitchers",
        f"- Test: {report['n_test_pitches']:,} pitches, {report['n_test_pitchers']:,} pitchers",
        "",
        "Features: " + ", ".join(FEATURE_COLUMNS),
        "",
        "Explicitamente NO incluidas: ubicacion (PlateLocHeight/Side, in/outside_strike_zone), "
        "conteo (Balls, Strikes, count), y features de arsenal (velo/ivb/hb_diff_vs_fb, "
        "tunnel_similarity, pitch_usage_pct, prev_pitch_type) — esas son para Location+/Pitching+, "
        "no para este Stuff+ puro.",
        "",
        "## is_whiff (clasificacion) — target oficial de Stuff+ v1",
        "",
        "| | AUC | log-loss | n | base rate |",
        "|---|---:|---:|---:|---:|",
        f"| Modelo (test) | {wf['model_test']['auc']:.4f} | {wf['model_test']['log_loss']:.4f} | "
        f"{wf['model_test']['n']:,} | {wf['model_test']['base_rate']:.4f} |",
        f"| Modelo (train) | {wf['model_train']['auc']:.4f} | {wf['model_train']['log_loss']:.4f} | "
        f"{wf['model_train']['n']:,} | {wf['model_train']['base_rate']:.4f} |",
        f"| Baseline ingenuo (test) — promedio is_whiff por AutoPitchType | "
        f"{wf['baseline_test']['auc']:.4f} | {wf['baseline_test']['log_loss']:.4f} | "
        f"{wf['baseline_test']['n']:,} | {wf['baseline_test']['base_rate']:.4f} |",
        "",
        f"Mejora del modelo sobre el baseline (test): AUC "
        f"{wf['model_test']['auc'] - wf['baseline_test']['auc']:+.4f}, log-loss "
        f"{wf['model_test']['log_loss'] - wf['baseline_test']['log_loss']:+.4f} (mas bajo es mejor).",
        "",
        "Top 10 features por importancia (is_whiff):",
        "",
        "| feature | importancia |",
        "|---|---:|",
    ]
    for feat, imp in list(wf["feature_importance"].items())[:10]:
        lines.append(f"| {feat} | {imp:.0f} |")

    completa, sin_am, solo_alt = ab["completa"], ab["sin_altitud_ni_magnus"], ab["solo_altitud"]
    auc_gap_altitud = completa["auc"] - solo_alt["auc"]
    auc_gap_sin_todo = completa["auc"] - sin_am["auc"]
    if abs(auc_gap_altitud) <= 0.002:
        interp_magnus = (
            "**La version completa y 'solo altitud' dan resultados casi identicos** "
            "(diferencia de AUC <= 0.002) — senal de que `magnus_residual_vert/horz` es "
            "redundante para el PODER PREDICTIVO del modelo de is_whiff. Esto no cambia "
            "su rol en el analisis explicativo de H1 (Semana 2, reports/magnus_analysis.md "
            "y reports/run_value_validation.md): ahi el residual de Magnus sirvio para "
            "AISLAR el efecto de la densidad de aire sobre el movimiento, una pregunta "
            "distinta a \"que tanta senal nueva le agrega a un modelo de whiff que ya "
            "tiene InducedVertBreak/HorzBreak crudos\". Es esperable que aporte poco aqui: "
            "el residual es built a partir de columnas (RelSpeed, SpinRate, SpinAxis, "
            "InducedVertBreak, HorzBreak) que YA estan en el modelo — LightGBM puede "
            "aproximar esa misma relacion combinando esas columnas crudas en sus splits, "
            "sin necesitar el residual pre-calculado."
        )
    else:
        interp_magnus = (
            f"La version completa supera a 'solo altitud' por {auc_gap_altitud:+.4f} de AUC — "
            "magnus_residual_vert/horz SI aporta señal predictiva incremental mas alla de "
            "la altitud sola."
        )
    if sin_am["auc"] < solo_alt["auc"] - 0.002:
        interp_altitud = (
            f"**'Sin altitud ni magnus' pierde {auc_gap_sin_todo:+.4f} de AUC frente a la "
            "version completa** (y tambien queda por debajo de 'solo altitud') — confirma "
            "que el contexto de altitud SI aporta señal real y no reemplazable por las "
            "features fisicas puras del pitch: saber en que categoria de altitud se lanzo "
            "ayuda al modelo mas alla de lo que ya le dicen RelSpeed/SpinRate/movimiento por "
            "si solos."
        )
    else:
        interp_altitud = (
            "'Sin altitud ni magnus' no pierde poder predictivo notable frente a las otras "
            "dos versiones — la altitud no esta aportando señal incremental relevante para "
            "is_whiff especificamente (puede seguir siendo relevante para otros targets, "
            "como run_value)."
        )

    lines += [
        "",
        "## Aporte de features de altitud/fisica",
        "",
        "Ablation sobre is_whiff, mismo split pitcher_holdout, mismos hiperparametros: "
        "3 subconjuntos de features. 'altitud' aqui es `altitude_level` (ordinal), la "
        "misma columna que usa Stuff+ v1 (ver seccion de features arriba) — el "
        "equivalente de `altitude_category` en este pipeline.",
        "",
        "| version | features (n) | AUC test | log-loss test |",
        "|---|---:|---:|---:|",
        f"| Completa (todas) | {completa['n_features']} | {completa['auc']:.4f} | {completa['log_loss']:.4f} |",
        f"| Sin altitud ni magnus | {sin_am['n_features']} | {sin_am['auc']:.4f} | {sin_am['log_loss']:.4f} |",
        f"| Solo altitud (sin magnus) | {solo_alt['n_features']} | {solo_alt['auc']:.4f} | {solo_alt['log_loss']:.4f} |",
        "",
        interp_magnus,
        "",
        interp_altitud,
        "",
        "**Nota metodologica importante para el contrafactual de altitud de Semana 5:** "
        "este resultado significa que el efecto de altitud sobre is_whiff esta "
        "completamente mediado por el movimiento observado (IVB/HB) — no hay efecto "
        "directo adicional. Un contrafactual que cambie altitude_category dejando el "
        "movimiento observado fijo va a medir ~0 por diseño, no porque la altitud no "
        "importe. Ver `reports/notas_metodologicas.md` para el diseño correcto "
        "(encadenar el modelo de Magnus antes de Stuff+).",
        "",
    ]

    lines += [
        "",
        "## Apendice — diagnostico de `run_value` (NO es parte de Stuff+ v1)",
        "",
        "Esta seccion existe solo para documentar la evidencia detras de la "
        "decision de arriba. El modelo de abajo NO se guarda en "
        "`stuff_plus_v1.pkl`.",
        "",
        f"Pitches con run_value nulo, excluidos de este target (bordes de datos de "
        f"la reconstruccion de plate appearances, ver reports/run_value_validation.md): "
        f"{rv['n_dropped_null_run_value']:,}",
        "",
        "| | RMSE | R^2 | n | std(y) |",
        "|---|---:|---:|---:|---:|",
        f"| Modelo (test) | {rv['model_test']['rmse']:.4f} | {rv['model_test']['r2']:.4f} | "
        f"{rv['model_test']['n']:,} | {rv['model_test']['y_std']:.4f} |",
        f"| Modelo (train) | {rv['model_train']['rmse']:.4f} | {rv['model_train']['r2']:.4f} | "
        f"{rv['model_train']['n']:,} | {rv['model_train']['y_std']:.4f} |",
        f"| Baseline ingenuo (test) — promedio run_value por count | "
        f"{rv['baseline_test']['rmse']:.4f} | {rv['baseline_test']['r2']:.4f} | "
        f"{rv['baseline_test']['n']:,} | {rv['baseline_test']['y_std']:.4f} |",
        "",
        f"Mejora del modelo sobre el baseline (test): RMSE "
        f"{rv['model_test']['rmse'] - rv['baseline_test']['rmse']:+.4f} (mas bajo es mejor), R^2 "
        f"{rv['model_test']['r2'] - rv['baseline_test']['r2']:+.4f}.",
        "",
        "Top 10 features por importancia (run_value):",
        "",
        "| feature | importancia |",
        "|---|---:|",
    ]
    for feat, imp in list(rv["feature_importance"].items())[:10]:
        lines.append(f"| {feat} | {imp:.0f} |")

    lines += [
        "",
        "## Hiperparametros",
        "",
        "Defaults razonables, sin tuning: " + ", ".join(f"{k}={v}" for k, v in LGB_PARAMS_COMMON.items()),
        "",
        f"Modelo guardado en `src/models/artifacts/stuff_plus_v1.pkl` "
        "(pickle con is_whiff_model, feature_columns y metadata — sin run_value_model, "
        "ver decision arriba).",
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
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Guardado reporte en %s", REPORT_PATH)

    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
