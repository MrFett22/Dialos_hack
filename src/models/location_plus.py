"""Location+ v1 (GAM, pygam) — cuanto varia el resultado SOLO por ubicacion.

Pregunta distinta a la de Stuff+: dado (aproximadamente) el mismo pitch
fisico, ¿cuanto cambia el resultado SOLO por donde se ubico la pelota y en
que conteo? El ablation de Stuff+ (reports/stuff_plus_baseline.md) ya mostro
que `is_whiff` depende fuertemente de la fisica del pitch (velocidad, spin,
movimiento, angulos) y casi nada de la altitud una vez que esa fisica esta en
el modelo — aqui, deliberadamente, NO se usa ninguna feature fisica del
pitch (RelSpeed, SpinRate, movimiento, etc.), solo ubicacion + conteo +
lateralidad del bateador + tipo de pitch como factor. Si `is_whiff` sale con
poco poder predictivo en ESTE modelo, es la confirmacion simetrica del mismo
hallazgo (ver seccion de comparacion en el reporte).

Features: PlateLocSide, PlateLocHeight (tensor spline 2D, te()), Balls,
Strikes, BatterSide y AutoPitchType (terminos de factor, f()).

Targets oficiales de Location+: is_called_strike, is_chase.
Target de comparacion (no oficial): is_whiff — se entrena y reporta solo
para contrastar contra el hallazgo del ablation de Stuff+, no se guarda en
el artefacto principal salvo que quede claramente marcado como comparacion.

Datos: data/processed/pitches_with_targets.parquet unido por PitchUID con
data/processed/splits/splits.parquet, filtrando a pitcher_holdout != 'excluded'.
Evaluacion: pitcher_holdout (mismo esquema que Stuff+).
"""

import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LogisticGAM, f, te

from src.models.stuff_plus import evaluate_classification

ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits" / "splits.parquet"
MODEL_PATH = ROOT / "src" / "models" / "artifacts" / "location_plus_v1.pkl"
REPORT_PATH = ROOT / "reports" / "location_plus_baseline.md"
FIG_DIR = ROOT / "reports" / "figures" / "location_heatmaps"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OFFICIAL_TARGETS = ["is_called_strike", "is_chase"]
COMPARISON_TARGETS = ["is_whiff"]
ALL_TARGETS = OFFICIAL_TARGETS + COMPARISON_TARGETS

RAW_FEATURE_COLUMNS = ["PlateLocSide", "PlateLocHeight", "Balls", "Strikes", "BatterSide", "AutoPitchType"]
# orden usado dentro de la matriz X que ve pygam (indices 0-5, referenciados
# por te()/f() abajo)
X_COLUMN_ORDER = ["PlateLocSide", "PlateLocHeight", "Balls", "Strikes", "BatterSide_code", "AutoPitchType_code"]

GAM_TERMS = te(0, 1) + f(2) + f(3) + f(4) + f(5)

# grilla para los heatmaps: rango tipico de zona + un poco de margen de "chase",
# no el rango completo de PlateLocHeight/Side del dataset (que llega a valores
# de pitches muy desviados, ~14 pies, inutil para visualizar la zona)
GRID_SIDE = np.linspace(-2.0, 2.0, 60)
GRID_HEIGHT = np.linspace(0.0, 5.0, 60)
HEATMAP_PITCH_TYPE = "Four-Seam"  # tipo de pitch fijado para los heatmaps (el mas comun)
HEATMAP_COUNTS = [(0, 0), (3, 2)]
# rectangulo de referencia visual (zona oficial aproximada MLB, ancho de plato
# 17in + radio de pelota, altura tipica rodilla-a-letras) — no es la zona
# dinamica por bateador del dataset (in_strike_zone), solo una referencia.
STRIKE_ZONE_REF = {"side": (-0.83, 0.83), "height": (1.5, 3.5)}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, dict]:
    cols = ["PitchUID", "pitcher_anon_id", "count"] + RAW_FEATURE_COLUMNS + ALL_TARGETS
    df = pd.read_parquet(TARGETS_PATH, columns=cols)
    splits = pd.read_parquet(SPLITS_PATH, columns=["PitchUID", "pitcher_holdout"])
    df = df.merge(splits, on="PitchUID", how="inner")
    df = df.loc[df["pitcher_holdout"] != "excluded"].copy()

    n_before_batterside = len(df)
    df = df.dropna(subset=["BatterSide"])
    n_dropped_batterside = n_before_batterside - len(df)

    batter_categories = sorted(df["BatterSide"].unique().tolist())
    pitch_categories = sorted(df["AutoPitchType"].astype(str).unique().tolist())
    df["BatterSide_code"] = pd.Categorical(df["BatterSide"], categories=batter_categories).codes
    df["AutoPitchType_code"] = pd.Categorical(df["AutoPitchType"].astype(str), categories=pitch_categories).codes

    meta = {
        "n_dropped_null_batterside": n_dropped_batterside,
        "batter_categories": batter_categories,
        "pitch_categories": pitch_categories,
    }
    return df, meta


def to_X(df: pd.DataFrame) -> np.ndarray:
    return df[X_COLUMN_ORDER].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Baseline ingenuo
# ---------------------------------------------------------------------------


def naive_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Promedio del target por (count, BatterSide), calculado en train."""
    lookup = train.groupby(["count", "BatterSide"], observed=True)[target].mean()
    global_mean = train[target].mean()
    keys = list(zip(test["count"], test["BatterSide"]))
    preds = np.array([lookup.get(k, global_mean) for k in keys])
    return np.clip(preds, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def train_gam(X_train: np.ndarray, y_train: np.ndarray) -> LogisticGAM:
    gam = LogisticGAM(GAM_TERMS)
    gam.fit(X_train, y_train)
    return gam


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


def _predict_grid(gam: LogisticGAM, balls: int, strikes: int, batter_code: int, pitch_code: int) -> np.ndarray:
    side_grid, height_grid = np.meshgrid(GRID_SIDE, GRID_HEIGHT)
    n = side_grid.size
    X = np.column_stack(
        [
            side_grid.ravel(),
            height_grid.ravel(),
            np.full(n, balls),
            np.full(n, strikes),
            np.full(n, batter_code),
            np.full(n, pitch_code),
        ]
    )
    proba = gam.predict_proba(X)
    return proba.reshape(side_grid.shape)


def plot_heatmaps(gam: LogisticGAM, target: str, meta: dict) -> Path:
    pitch_code = meta["pitch_categories"].index(HEATMAP_PITCH_TYPE)
    batter_sides = meta["batter_categories"]

    fig, axes = plt.subplots(
        len(batter_sides), len(HEATMAP_COUNTS), figsize=(6 * len(HEATMAP_COUNTS), 5.5 * len(batter_sides))
    )
    axes = np.atleast_2d(axes)

    for i, batter_side in enumerate(batter_sides):
        batter_code = meta["batter_categories"].index(batter_side)
        for j, (balls, strikes) in enumerate(HEATMAP_COUNTS):
            ax = axes[i, j]
            grid = _predict_grid(gam, balls, strikes, batter_code, pitch_code)
            im = ax.pcolormesh(GRID_SIDE, GRID_HEIGHT, grid, cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto")
            rect = plt.Rectangle(
                (STRIKE_ZONE_REF["side"][0], STRIKE_ZONE_REF["height"][0]),
                STRIKE_ZONE_REF["side"][1] - STRIKE_ZONE_REF["side"][0],
                STRIKE_ZONE_REF["height"][1] - STRIKE_ZONE_REF["height"][0],
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.set_title(f"{target} | count={balls}-{strikes} | BatterSide={batter_side}")
            ax.set_xlabel("PlateLocSide (ft)")
            ax.set_ylabel("PlateLocHeight (ft)")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, label=f"P({target})")

    fig.suptitle(
        f"Location+ v1 — {target} predicho sobre la grilla PlateLocHeight x PlateLocSide "
        f"(AutoPitchType={HEATMAP_PITCH_TYPE}, rectangulo = zona de referencia aproximada)",
        y=1.0,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"{target}_heatmaps.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> tuple[dict, dict]:
    df, meta = load_data()
    logger.info("Filas disponibles (pitcher_holdout != excluded, BatterSide no nulo): %s", f"{len(df):,}")

    train_df = df.loc[df["pitcher_holdout"] == "train"]
    test_df = df.loc[df["pitcher_holdout"] == "test"]

    X_train = to_X(train_df)
    X_test = to_X(test_df)

    models: dict[str, LogisticGAM] = {}
    target_reports: dict[str, dict] = {}
    heatmap_paths: dict[str, str] = {}

    for target in ALL_TARGETS:
        y_train = train_df[target].to_numpy()
        y_test = test_df[target].to_numpy()

        gam = train_gam(X_train, y_train)
        models[target] = gam

        pred_test = gam.predict_proba(X_test)
        pred_train = gam.predict_proba(X_train)
        baseline_test = naive_baseline(train_df, test_df, target)

        target_reports[target] = {
            "official": target in OFFICIAL_TARGETS,
            "model_test": evaluate_classification(pd.Series(y_test), pred_test),
            "model_train": evaluate_classification(pd.Series(y_train), pred_train),
            "baseline_test": evaluate_classification(pd.Series(y_test), baseline_test),
        }

        heatmap_paths[target] = str(plot_heatmaps(gam, target, meta).relative_to(ROOT))
        logger.info("Entrenado y evaluado: %s", target)

    report = {
        "n_total_valid": len(df),
        "n_dropped_null_batterside": meta["n_dropped_null_batterside"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_pitchers": int(train_df["pitcher_anon_id"].nunique()),
        "n_test_pitchers": int(test_df["pitcher_anon_id"].nunique()),
        "batter_side_counts": df["BatterSide"].value_counts().to_dict(),
        "targets": target_reports,
        "heatmap_paths": heatmap_paths,
    }

    artifact = {
        "models": {t: models[t] for t in OFFICIAL_TARGETS},
        "comparison_models": {t: models[t] for t in COMPARISON_TARGETS},
        "feature_order": X_COLUMN_ORDER,
        "batter_categories": meta["batter_categories"],
        "pitch_categories": meta["pitch_categories"],
        "official_targets": OFFICIAL_TARGETS,
        "comparison_targets": COMPARISON_TARGETS,
    }
    return artifact, report


def format_report(report: dict) -> str:
    lines = [
        "# Location+ v1 — modelo baseline (GAM / pygam, solo ubicacion + conteo + lateralidad)",
        "",
        "Pregunta: dado (aproximadamente) el mismo pitch fisico, ¿cuanto cambia el "
        "resultado SOLO por donde se ubico y en que conteo? Deliberadamente NO se "
        "usa ninguna feature fisica del pitch (RelSpeed, SpinRate, movimiento, "
        "angulos) — esas son de Stuff+.",
        "",
        f"Features: {', '.join(RAW_FEATURE_COLUMNS)} — PlateLocSide/PlateLocHeight con "
        "tensor spline 2D (te()), el resto como terminos de factor (f()).",
        "",
        f"Filas usadas: {report['n_total_valid']:,} (de las cuales se excluyeron "
        f"{report['n_dropped_null_batterside']:,} por BatterSide nulo)",
        f"- Train: {report['n_train']:,} pitches, {report['n_train_pitchers']:,} pitchers",
        f"- Test: {report['n_test']:,} pitches, {report['n_test_pitchers']:,} pitchers",
        "",
    ]

    for target in ALL_TARGETS:
        tr = report["targets"][target]
        tag = "target oficial de Location+" if tr["official"] else "SOLO COMPARACION, no oficial"
        lines += [
            f"## {target} ({tag})",
            "",
            "| | AUC | log-loss | n | base rate |",
            "|---|---:|---:|---:|---:|",
            f"| Modelo (test) | {tr['model_test']['auc']:.4f} | {tr['model_test']['log_loss']:.4f} | "
            f"{tr['model_test']['n']:,} | {tr['model_test']['base_rate']:.4f} |",
            f"| Modelo (train) | {tr['model_train']['auc']:.4f} | {tr['model_train']['log_loss']:.4f} | "
            f"{tr['model_train']['n']:,} | {tr['model_train']['base_rate']:.4f} |",
            f"| Baseline ingenuo (test) — promedio por (count, BatterSide) | "
            f"{tr['baseline_test']['auc']:.4f} | {tr['baseline_test']['log_loss']:.4f} | "
            f"{tr['baseline_test']['n']:,} | {tr['baseline_test']['base_rate']:.4f} |",
            "",
            f"Mejora del modelo sobre el baseline (test): AUC "
            f"{tr['model_test']['auc'] - tr['baseline_test']['auc']:+.4f}, log-loss "
            f"{tr['model_test']['log_loss'] - tr['baseline_test']['log_loss']:+.4f} (mas bajo es mejor).",
            "",
            f"Heatmap: `{report['heatmap_paths'][target]}`",
            "",
        ]

    whiff = report["targets"]["is_whiff"]
    called = report["targets"]["is_called_strike"]
    chase = report["targets"]["is_chase"]
    lines += [
        "## Comparacion con el ablation de Stuff+",
        "",
        f"Location+ (solo ubicacion+conteo+lateralidad, SIN fisica del pitch) obtiene "
        f"AUC={whiff['model_test']['auc']:.4f} para `is_whiff` — comparar contra Stuff+ "
        f"(solo fisica, sin ubicacion/conteo), que obtuvo AUC=0.7297 en "
        "reports/stuff_plus_baseline.md. ",
    ]
    if whiff["model_test"]["auc"] < called["model_test"]["auc"] and whiff["model_test"]["auc"] < chase["model_test"]["auc"]:
        lines.append(
            "**Confirmacion simetrica del hallazgo del ablation de Stuff+:** aqui, con "
            "SOLO ubicacion/conteo/lateralidad (sin fisica), `is_whiff` es el target que "
            "peor predice el modelo de los tres — menos que is_called_strike y is_chase, "
            "que si son fundamentalmente de ubicacion/conteo. Confirma que is_whiff "
            "depende de la fisica del pitch (Stuff+), no de donde se ubico (Location+); "
            "is_called_strike/is_chase son al reves: dependen de ubicacion, no de fisica."
        )
    else:
        lines.append(
            "is_whiff no salio claramente peor que is_called_strike/is_chase en este "
            "modelo — revisar manualmente antes de citar como confirmacion del hallazgo "
            "de Stuff+."
        )
    lines.append("")

    lines += [
        "## Notas de los heatmaps",
        "",
        f"Grilla PlateLocSide en [{GRID_SIDE.min()},{GRID_SIDE.max()}] ft x PlateLocHeight en "
        f"[{GRID_HEIGHT.min()},{GRID_HEIGHT.max()}] ft (no el rango completo del dataset, que "
        "llega a pitches muy desviados e inutiles para visualizar la zona). AutoPitchType fijo "
        f"en '{HEATMAP_PITCH_TYPE}' (el mas comun). El rectangulo negro es una zona de "
        "referencia aproximada (17in de plato + radio de pelota, ~rodilla a letras) — NO es la "
        "zona dinamica por bateador (`in_strike_zone`) del dataset, solo una guia visual.",
        "",
        "Verificar visualmente en las figuras: la zona de is_called_strike deberia verse mas "
        "generosa/ancha en conteo de pitcher (0-0) que en conteo de bateador (3-2) si el "
        "modelo aprendio el sesgo real de los umpires; is_chase deberia concentrarse fuera "
        "del rectangulo de referencia, mas en 3-2 (el bateador se ve forzado a perseguir "
        "mas cerca de la zona) que en 0-0.",
        "",
        "`BatterSide` tiene 4 valores reales en el dato (no solo Left/Right), por eso los "
        "heatmaps salen con 4 filas: " + ", ".join(
            f"{k}={v:,}" for k, v in report["batter_side_counts"].items()
        ) + ". Los paneles de `Switch` y `Undefined` "
        f"({report['batter_side_counts'].get('Switch', 0):,} y "
        f"{report['batter_side_counts'].get('Undefined', 0):,} pitches respectivamente en el "
        "subconjunto usado para entrenar, <0.04% del dataset cada uno) son estimaciones del "
        "factor `f()` con muestra muy chica — tratar esos dos paneles especificos con cautela, "
        "no como una lectura confiable del "
        "sesgo real de umpire para esos grupos.",
        "",
        f"Modelo guardado en `src/models/artifacts/location_plus_v1.pkl` "
        "(is_called_strike y is_chase oficiales; is_whiff guardado aparte como comparison_models, "
        "no es parte de Location+).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    artifact, report = run()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(artifact, f_out)
    logger.info("Guardado modelo en %s", MODEL_PATH)

    report_text = format_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)

    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
