"""Stuff+ v2 — features corregidas: mano del pitcher, ubicación escondida y redundancias.

Diagnóstico de v1 (ver reports/resultados_finales.md §7.2): Stuff+ v1 es más
estable año contra año que cualquier estadística observada, pero predice PEOR
el whiff% del año siguiente que el propio whiff% observado. Hipótesis: el
modelo se apoya en información que no es "stuff":

1. Mano del pitcher sin normalizar. En zurdos se invierte el signo de
   HorzBreak, RelSide, HorzRelAngle, HorzApprAngle y el residual horizontal de
   Magnus, y SpinAxis se refleja (≈150° vs ≈211° en rectas). El modelo gasta
   splits aprendiendo esa simetría.
2. Ubicación escondida. VertApprAngle tiene correlación 0.78 con
   PlateLocHeight y HorzApprAngle 0.58 con PlateLocSide: un pitch bajo llega
   más inclinado. Los ángulos de salida (VertRelAngle/HorzRelAngle) dicen hacia
   dónde apuntó el pitcher. Es decir, el Stuff+ "sin ubicación" sí la ve.
3. Redundancias: VertBreak (≈ IVB + gravedad), EffectiveVelo (≈ velocidad +
   extensión), ZoneSpeed/ZoneTime (≈ velocidad − SpeedDrop), y altitude_level
   + residuales de Magnus (ablation v1: no aportan AUC).
4. Falta el contexto del arsenal: un cambio de 88 mph vale distinto si la
   recta va a 97 o a 90. Los Stuff+ modernos usan diferencias contra la recta.

Se evalúa como escalera de ablation (cada variante agrega un cambio sobre la
anterior). **Candidata oficial declarada antes de ver resultados: v2d**, salvo
que el benchmark muestre que algún escalón empeora el poder predictivo.

    v2a_mano       = v1 + espejo de zurdos + indicador de mismo lado (platoon)
    v2b_ubicacion  = v2a + VAA/HAA ajustados por ubicación, sin ángulos de salida
    v2c_limpio     = v2b − redundancias − altitud/Magnus
    v2d_arsenal    = v2c + diferencias contra la recta del pitcher

Todas con los mismos hiperparámetros de v1 (el tuning es un paso posterior) y
predicciones out-of-fold (GroupKFold por pitcher, k=5).

Salidas:
- data/processed/stuff_plus_v2_oof_pitches.parquet
- data/processed/stuff_plus_v2_scores.parquet (pitcher-temporada, escala 100+10z)
- src/models/artifacts/stuff_plus_v2.pkl (modelo final v2d entrenado con todo)
- reports/stuff_plus_v2.md
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models import stuff_plus
from src.validation import benchmark

ROOT = Path(__file__).resolve().parents[2]
PHYSICS_PATH = ROOT / "data" / "processed" / "pitches_with_physics.parquet"
ARSENAL_PATH = ROOT / "data" / "processed" / "pitches_with_arsenal.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits" / "splits.parquet"
OOF_PATH = ROOT / "data" / "processed" / "stuff_plus_v2_oof_pitches.parquet"
SCORES_PATH = ROOT / "data" / "processed" / "stuff_plus_v2_scores.parquet"
MODEL_PATH = ROOT / "src" / "models" / "artifacts" / "stuff_plus_v2.pkl"
REPORT_PATH = ROOT / "reports" / "stuff_plus_v2.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_FOLDS = 5
LGB_PARAMS = stuff_plus.LGB_PARAMS_COMMON  # mismos hiperparámetros que v1, a propósito
OFFICIAL_VARIANT = "v2d_arsenal"

# Columnas cuyo signo se invierte para zurdos (quedan en el marco de referencia de un derecho)
MIRROR_SIGN_COLUMNS = [
    "HorzBreak", "RelSide", "HorzRelAngle", "HorzApprAngle", "magnus_residual_horz",
    "hb_diff_vs_fb", "PlateLocSide",
]

V1_NUMERIC = stuff_plus.NUMERIC_FEATURES
V2A_NUMERIC = V1_NUMERIC  # mismas columnas, pero ya espejadas
V2B_NUMERIC = [c for c in V2A_NUMERIC if c not in ("VertApprAngle", "HorzApprAngle", "VertRelAngle", "HorzRelAngle")] + [
    "vaa_loc_adj", "haa_loc_adj",
]
REDUNDANT = ["VertBreak", "EffectiveVelo", "ZoneSpeed", "ZoneTime", "altitude_level",
             "magnus_residual_vert", "magnus_residual_horz"]
V2C_NUMERIC = [c for c in V2B_NUMERIC if c not in REDUNDANT]
V2D_NUMERIC = V2C_NUMERIC + ["velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb"]

V2_CATEGORICAL = ["same_side", "AutoPitchType"]

VARIANTS = {
    "v2a_mano": {
        "numeric": V2A_NUMERIC, "categorical": V2_CATEGORICAL,
        "label": "v2a: espejo de zurdos + platoon",
    },
    "v2b_ubicacion": {
        "numeric": V2B_NUMERIC, "categorical": V2_CATEGORICAL,
        "label": "v2b: + VAA/HAA ajustados por ubicación, sin ángulos de salida",
    },
    "v2c_limpio": {
        "numeric": V2C_NUMERIC, "categorical": V2_CATEGORICAL,
        "label": "v2c: − redundancias, altitud y Magnus",
    },
    "v2d_arsenal": {
        "numeric": V2D_NUMERIC, "categorical": V2_CATEGORICAL,
        "label": "v2d: + diferencias contra la recta (candidata oficial)",
    },
}

# Diagnóstico (adelanto del Paso 3, NO candidatas de este paso): el mismo
# modelo entrenado solo con swings, es decir P(whiff | swing). Prueba si el
# problema de v2 es el target y no las features: con is_whiff sobre todos los
# pitches, la ubicación domina porque decide si el bateador tira.
SWING_DIAGNOSTICS = {
    "diag_v1_swing": {
        "numeric": V1_NUMERIC, "categorical": stuff_plus.CATEGORICAL_FEATURES, "swings_only": True,
        "label": "diagnóstico: features v1, P(whiff | swing)",
    },
    "diag_v2d_swing": {
        "numeric": V2D_NUMERIC, "categorical": V2_CATEGORICAL, "swings_only": True,
        "label": "diagnóstico: features v2d, P(whiff | swing)",
    },
}
ALL_MODELS = {**VARIANTS, **SWING_DIAGNOSTICS}


# ---------------------------------------------------------------------------
# Datos y features
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    physics_cols = list(dict.fromkeys(
        ["PitchUID", "pitcher_anon_id", "year", "is_whiff", "is_swing", "PlateLocHeight", "PlateLocSide"]
        + stuff_plus.NUMERIC_FEATURES + stuff_plus.CATEGORICAL_FEATURES
    ))
    df = pd.read_parquet(PHYSICS_PATH, columns=physics_cols)
    arsenal = pd.read_parquet(ARSENAL_PATH, columns=["PitchUID", "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb"])
    splits = pd.read_parquet(SPLITS_PATH, columns=["PitchUID", "pitcher_holdout"])

    df = df.merge(arsenal, on="PitchUID", how="left").merge(splits, on="PitchUID", how="inner")
    df = df.loc[df["pitcher_holdout"] != "excluded"].drop(columns="pitcher_holdout")
    df = df.dropna(subset=["is_whiff"]).reset_index(drop=True)
    df["EffectiveVelo"] = df["EffectiveVelo"].astype(float)
    df["is_whiff"] = df["is_whiff"].astype(int)
    return df


def pitcher_hand(df: pd.DataFrame) -> pd.Series:
    """Mano del pitcher; si viene Undefined/nula se infiere del lado de salida (RelSide < 0 = zurdo)."""
    hand = df["PitcherThrows"].astype("object")
    unknown = ~hand.isin(["Right", "Left"])
    inferred = np.where(df["RelSide"] < 0, "Left", "Right")
    return hand.where(~unknown, inferred)


def fit_location_adjustment(df: pd.DataFrame, angle: str, loc: str) -> dict:
    """Recta angle ~ a + b·loc por tipo de pitch. El residual es el ángulo 'a ubicación constante'."""
    coefs = {}
    for ptype, g in df.groupby("AutoPitchType", observed=True):
        g = g[[angle, loc]].dropna()
        if len(g) < 100:
            continue
        b, a = np.polyfit(g[loc], g[angle], 1)
        coefs[str(ptype)] = (float(a), float(b))
    sub = df[[angle, loc]].dropna()
    b, a = np.polyfit(sub[loc], sub[angle], 1)
    coefs["__global__"] = (float(a), float(b))
    return coefs


def apply_location_adjustment(df: pd.DataFrame, angle: str, loc: str, coefs: dict) -> pd.Series:
    ptype = df["AutoPitchType"].astype("object").astype(str)
    a = ptype.map(lambda p: coefs.get(p, coefs["__global__"])[0])
    b = ptype.map(lambda p: coefs.get(p, coefs["__global__"])[1])
    return df[angle] - (a + b * df[loc])


def build_v2_features(df: pd.DataFrame, location_coefs: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Agrega las columnas v2. Devuelve (df, coeficientes del ajuste por ubicación).

    Si `location_coefs` es None se ajustan sobre `df` (son 2 parámetros por tipo
    de pitch sobre ~600k pitches, así que ajustarlos con todo no filtra
    información relevante de ningún pitcher).
    """
    df = df.copy()
    hand = pitcher_hand(df)
    sign = np.where(hand == "Left", -1.0, 1.0)
    for col in MIRROR_SIGN_COLUMNS:
        df[col] = df[col] * sign
    df["SpinAxis"] = np.where(hand == "Left", 360.0 - df["SpinAxis"], df["SpinAxis"])

    bat = df["BatterSide"].astype("object")
    same = np.where(bat == hand, "same", "opposite")
    same = np.where(bat.isin(["Right", "Left", "Switch"]), same, "Unknown")
    df["same_side"] = pd.Categorical(same, categories=["same", "opposite", "Unknown"])

    if location_coefs is None:
        location_coefs = {
            "vaa": fit_location_adjustment(df, "VertApprAngle", "PlateLocHeight"),
            "haa": fit_location_adjustment(df, "HorzApprAngle", "PlateLocSide"),
        }
    df["vaa_loc_adj"] = apply_location_adjustment(df, "VertApprAngle", "PlateLocHeight", location_coefs["vaa"])
    df["haa_loc_adj"] = apply_location_adjustment(df, "HorzApprAngle", "PlateLocSide", location_coefs["haa"])

    df["AutoPitchType"] = df["AutoPitchType"].astype("object").fillna("Unknown").astype("category")
    for col in stuff_plus.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object").fillna("Unknown").astype("category")
    return df, location_coefs


# ---------------------------------------------------------------------------
# Out-of-fold
# ---------------------------------------------------------------------------


def oof_predictions(
    df: pd.DataFrame, numeric: list[str], categorical: list[str], train_mask: np.ndarray | None = None
) -> np.ndarray:
    """OOF por pitcher. Con `train_mask` se entrena solo con esas filas (p. ej. swings)
    pero se predice para TODOS los pitches del fold held-out."""
    X = df[numeric + categorical]
    y = df["is_whiff"]
    oof = np.full(len(df), np.nan)
    for tr, ho in GroupKFold(n_splits=N_FOLDS).split(df, groups=df["pitcher_anon_id"]):
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **LGB_PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr], categorical_feature=categorical)
        oof[ho] = model.predict_proba(X.iloc[ho])[:, 1]
    return oof


def generate_all_oof(refresh: bool = False) -> pd.DataFrame:
    if OOF_PATH.exists() and not refresh:
        logger.info("Usando OOF v2 en cache: %s", OOF_PATH.name)
        return pd.read_parquet(OOF_PATH)
    df, _ = build_v2_features(load_data())
    logger.info("Stuff+ v2: %s pitches, %s pitchers", f"{len(df):,}", f"{df['pitcher_anon_id'].nunique():,}")
    out = df[["PitchUID", "pitcher_anon_id", "year", "is_whiff", "is_swing"]].copy()
    swing_mask = (df["is_swing"] == 1).to_numpy()
    for name, spec in ALL_MODELS.items():
        t0 = time.time()
        mask = swing_mask if spec.get("swings_only") else None
        out[f"pred_{name}"] = oof_predictions(df, spec["numeric"], spec["categorical"], train_mask=mask)
        logger.info("  %s: %d features, %.0fs", name, len(spec["numeric"]) + len(spec["categorical"]), time.time() - t0)
    out.to_parquet(OOF_PATH, index=False)
    return out


def train_final_model(variant: str = OFFICIAL_VARIANT) -> dict:
    """Entrena la variante oficial con todos los pitches (para dashboard/inferencia) y la guarda."""
    df, location_coefs = build_v2_features(load_data())
    spec = VARIANTS[variant]
    model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **LGB_PARAMS)
    model.fit(df[spec["numeric"] + spec["categorical"]], df["is_whiff"], categorical_feature=spec["categorical"])
    artifact = {
        "is_whiff_model": model,
        "variant": variant,
        "numeric_features": spec["numeric"],
        "categorical_features": spec["categorical"],
        "location_coefs": location_coefs,
        "mirror_sign_columns": MIRROR_SIGN_COLUMNS,
        "metadata": {"n_train": len(df), "lgb_params": LGB_PARAMS},
    }
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)
    return artifact


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['r']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_delta(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] · P={d['p_model_better']:.0%}"


def format_report(results: dict, pitch_metrics: dict, importance: pd.Series, location_coefs: dict) -> str:
    names = ["stuff_plus_v1"] + list(ALL_MODELS)
    labels = {"stuff_plus_v1": "v1 (referencia)", **{k: v["label"] for k, v in ALL_MODELS.items()}}
    lines = [
        "# Stuff+ v2 — features corregidas (mano, ubicación escondida, redundancias, arsenal)",
        "",
        "Generado por `src/models/stuff_plus_v2.py`. Benchmark de `src/validation/benchmark.py` "
        "(pares 2024→2025 y 2025→2026, bootstrap pareado por pitcher, 2,000 réplicas). "
        f"Candidata oficial declarada antes de ver resultados: **{OFFICIAL_VARIANT}**.",
        "",
        "## Cambios de features (escalera de ablation)",
        "",
        "| variante | features | cambio |",
        "|---|---:|---|",
        f"| v1 | {len(stuff_plus.FEATURE_COLUMNS)} | referencia |",
    ]
    for name, spec in VARIANTS.items():
        lines.append(f"| {name} | {len(spec['numeric']) + len(spec['categorical'])} | {spec['label']} |")
    lines += [
        "",
        f"Features finales de {OFFICIAL_VARIANT}: " + ", ".join(VARIANTS[OFFICIAL_VARIANT]["numeric"] + VARIANTS[OFFICIAL_VARIANT]["categorical"]),
        "",
        "- **Espejo de zurdos:** se invierte el signo de " + ", ".join(f"`{c}`" for c in MIRROR_SIGN_COLUMNS)
        + "; `SpinAxis` → 360 − SpinAxis. Mano Undefined/nula se infiere de `RelSide`. "
        "`PitcherThrows`/`BatterSide` se reemplazan por `same_side` (mismo lado / lado contrario).",
        "- **Ajuste por ubicación:** `vaa_loc_adj` = VertApprAngle − (a + b·PlateLocHeight) y `haa_loc_adj` = "
        "HorzApprAngle − (a + b·PlateLocSide), con la recta ajustada por tipo de pitch. Es el ángulo de llegada "
        "\"a ubicación constante\": cuánto más plano o inclinado llega el pitch que uno promedio de su tipo en el mismo punto.",
        "",
        f"Pendiente global VAA ~ altura: b={location_coefs['vaa']['__global__'][1]:.3f} °/pie; "
        f"HAA ~ lado: b={location_coefs['haa']['__global__'][1]:.3f} °/pie.",
        "",
        "## Métricas pitch a pitch (OOF, `is_whiff`)",
        "",
        "| variante | AUC | log-loss | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in names:
        pm = pitch_metrics[n]
        lines.append(f"| {n} | {pm['auc']:.4f} | {pm['log_loss']:.4f} | {pm['brier']:.4f} | {pm['ece']:.4f} |")
    lines += [
        "",
        "Los diagnósticos `diag_*_swing` se evalúan solo sobre swings (su universo), así que su AUC no es comparable 1 a 1 con el resto.",
        "",
        "Bajar el AUC pitch a pitch es **esperado** al quitar información de ubicación: el v1 predecía mejor en parte porque veía dónde cruzaba el pitch. Lo que importa es el benchmark de abajo.",
        "",
    ]

    for mp, res in results.items():
        lines += [
            f"## Benchmark año-contra-año — mínimo {mp} pitches ({res['n_pairs']:,} pares, {res['n_pitchers']:,} pitchers)",
            "",
            "| variante | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |",
            "|---|---|---|---|---|",
        ]
        for n in names:
            pr = res["predictive"][n]
            lines.append(
                f"| {n} | {_fmt(res['stability'][n])} | {_fmt(pr['obs_rv_prevented'])} | "
                f"{_fmt(pr['obs_k_minus_bb'])} | {_fmt(pr['obs_whiff_rate'])} |"
            )
        for b in ["obs_whiff_rate", "obs_csw", "obs_k_minus_bb"]:
            pr = res["predictive"][b]
            lines.append(
                f"| *{benchmark.BASELINE_METRICS[b]}* | {_fmt(res['stability'][b])} | {_fmt(pr['obs_rv_prevented'])} | "
                f"{_fmt(pr['obs_k_minus_bb'])} | {_fmt(pr['obs_whiff_rate'])} |"
            )
        lines += [
            "",
            "**Delta contra v1 (bootstrap pareado; P = probabilidad de que la variante sea mejor):**",
            "",
            "| variante | estabilidad | carreras evitadas t+1 | whiff% t+1 |",
            "|---|---|---|---|",
        ]
        for n in ALL_MODELS:
            d = res["deltas_vs_reference"]
            lines.append(
                f"| {n} | {_fmt_delta(d[f'{n}|stability'])} | {_fmt_delta(d[f'{n}|obs_rv_prevented'])} | "
                f"{_fmt_delta(d[f'{n}|obs_whiff_rate'])} |"
            )
        lines += [
            "",
            "**Delta contra el mejor baseline observado:**",
            "",
            "| variante | carreras evitadas t+1 | whiff% t+1 |",
            "|---|---|---|",
        ]
        for n in names:
            d = res["deltas_vs_best_baseline"]
            rv, wh = d[f"{n}|obs_rv_prevented"], d[f"{n}|obs_whiff_rate"]
            lines.append(f"| {n} | {_fmt_delta(rv)} (vs {rv['vs']}) | {_fmt_delta(wh)} (vs {wh['vs']}) |")
        lines.append("")

    lines += [
        f"## Importancia de features — {OFFICIAL_VARIANT} (gain, modelo final)",
        "",
        "| feature | % del gain total |",
        "|---|---:|",
    ]
    for feat, val in importance.items():
        lines.append(f"| {feat} | {val:.1%} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(refresh: bool = False) -> dict:
    oof = generate_all_oof(refresh=refresh)

    pitch_metrics = {}
    scores = oof[["pitcher_anon_id", "year"]].drop_duplicates().copy()
    agg = oof.groupby(["pitcher_anon_id", "year"])[[f"pred_{n}" for n in ALL_MODELS]].mean().reset_index()
    swings = oof["is_swing"] == 1
    for n in ALL_MODELS:
        rows = swings if ALL_MODELS[n].get("swings_only") else slice(None)
        pitch_metrics[n] = benchmark.pitch_level_metrics(
            oof.loc[rows, "is_whiff"].to_numpy(), oof.loc[rows, f"pred_{n}"].to_numpy()
        )
        agg[n] = benchmark.scale_plus(agg[f"pred_{n}"], agg["year"])
    scores = agg[["pitcher_anon_id", "year"] + list(ALL_MODELS)]
    scores.to_parquet(SCORES_PATH, index=False)

    table, v1_pitch = benchmark.build_metric_table(extra_models={n: scores for n in ALL_MODELS})
    pitch_metrics["stuff_plus_v1"] = v1_pitch["stuff_plus_v1"]

    model_metrics = {"stuff_plus_v1": "Stuff+ v1", **{n: ALL_MODELS[n]["label"] for n in ALL_MODELS}}
    results = {}
    for mp in benchmark.MIN_PITCHES_LEVELS:
        pairs = benchmark.build_pairs(table, mp)
        results[mp] = benchmark.correlation_suite(
            pairs, model_metrics=model_metrics, references={n: "stuff_plus_v1" for n in ALL_MODELS}
        )
        logger.info(
            "min %d: pred. carreras t+1 v1=%.3f %s | pred. whiff t+1 v1=%.3f %s",
            mp, results[mp]["predictive"]["stuff_plus_v1"]["obs_rv_prevented"]["r"],
            " ".join(f"{n}={results[mp]['predictive'][n]['obs_rv_prevented']['r']:.3f}" for n in ALL_MODELS),
            results[mp]["predictive"]["stuff_plus_v1"]["obs_whiff_rate"]["r"],
            " ".join(f"{n}={results[mp]['predictive'][n]['obs_whiff_rate']['r']:.3f}" for n in ALL_MODELS),
        )

    artifact = train_final_model()
    model = artifact["is_whiff_model"]
    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"), index=model.booster_.feature_name()
    )
    importance = (importance / importance.sum()).sort_values(ascending=False)

    REPORT_PATH.write_text(
        format_report(results, pitch_metrics, importance, artifact["location_coefs"]), encoding="utf-8"
    )
    payload = {
        "version": "stuff_plus_v2",
        "results_by_min_pitches": {str(k): v for k, v in results.items()},
        "pitch_level": {k: {kk: vv for kk, vv in v.items() if kk != "calibration_table"} for k, v in pitch_metrics.items()},
    }
    benchmark.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (benchmark.BENCHMARK_DIR / "stuff_plus_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular predicciones OOF")
    args = parser.parse_args()
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
