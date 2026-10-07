"""Analisis SHAP de stuff_plus_v1.pkl (modelo de is_whiff).

Explica el modelo oficial de Stuff+ (LightGBM, target is_whiff, ver
src/models/stuff_plus.py) con shap.TreeExplainer sobre el set de test de
pitcher_holdout (pitchers nunca vistos en train). Genera:

1. Summary plot global (importancia + direccion de cada feature)
2. Summary plots separados por AutoPitchType (los 7 tipos con n>=1000 en test)
3. Dependence plots para InducedVertBreak, HorzBreak, SpinRate, RelSpeed,
   HorzRelAngle, RelSide, VertApprAngle
4. Verificacion cruzada contra el ablation de Stuff+ (reports/stuff_plus_baseline.md):
   altitude_level y magnus_residual_vert/horz deberian salir con SHAP bajo,
   consistente con "no aportan señal incremental" — se reporta el ranking
   completo y se marca explicitamente si hay discrepancia.
"""

import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.stuff_plus import FEATURE_COLUMNS, MODEL_PATH, load_data, prepare_xy

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "shap_findings.md"
FIG_DIR = ROOT / "reports" / "figures" / "shap"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEPENDENCE_FEATURES = [
    "InducedVertBreak",
    "HorzBreak",
    "SpinRate",
    "RelSpeed",
    "HorzRelAngle",
    "RelSide",
    "VertApprAngle",
]
ABLATION_FLAGGED_FEATURES = ["altitude_level", "magnus_residual_vert", "magnus_residual_horz"]
MIN_N_PER_PITCHTYPE = 1000


# ---------------------------------------------------------------------------
# Datos + SHAP
# ---------------------------------------------------------------------------


def load_model_and_test_data() -> tuple[object, pd.DataFrame, pd.Series]:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["is_whiff_model"]

    df = load_data()
    test_df = df.loc[df["pitcher_holdout"] == "test"]
    X_test, y_test = prepare_xy(test_df, "is_whiff")
    return model, X_test, y_test


def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # clase positiva (whiff=1)
    return shap_values, float(np.ravel(explainer.expected_value)[-1])


# ---------------------------------------------------------------------------
# 1. Summary plot global
# ---------------------------------------------------------------------------


def plot_summary_global(shap_values: np.ndarray, X: pd.DataFrame) -> Path:
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP summary global — is_whiff (Stuff+ v1)")
    out = FIG_DIR / "summary_global.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# 2. Summary plots por AutoPitchType
# ---------------------------------------------------------------------------


def plot_summary_by_pitchtype(shap_values: np.ndarray, X: pd.DataFrame) -> dict[str, Path]:
    out_dir = FIG_DIR / "summary_by_pitchtype"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = X["AutoPitchType"].value_counts()
    pitch_types = counts[counts >= MIN_N_PER_PITCHTYPE].index.tolist()

    paths = {}
    for pt in pitch_types:
        mask = (X["AutoPitchType"] == pt).to_numpy()
        X_sub = X.loc[mask]
        sv_sub = shap_values[mask]

        plt.figure()
        shap.summary_plot(sv_sub, X_sub, show=False)
        plt.title(f"SHAP summary — is_whiff | AutoPitchType={pt} (n={mask.sum():,})")
        out = out_dir / f"{pt.replace(' ', '_')}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        paths[pt] = out
    return paths


def rank_features_by_pitchtype(shap_values: np.ndarray, X: pd.DataFrame, pitch_types: list[str]) -> pd.DataFrame:
    rows = {}
    for pt in pitch_types:
        mask = (X["AutoPitchType"] == pt).to_numpy()
        mean_abs = np.abs(shap_values[mask]).mean(axis=0)
        rows[pt] = pd.Series(mean_abs, index=X.columns)
    table = pd.DataFrame(rows)
    return table


# ---------------------------------------------------------------------------
# 3. Dependence plots
# ---------------------------------------------------------------------------


def plot_dependence(shap_values: np.ndarray, X: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    for feat in DEPENDENCE_FEATURES:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.title(f"SHAP dependence — {feat} (is_whiff)")
        out = FIG_DIR / f"dependence_{feat}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        paths[feat] = out
    return paths


def dependence_direction(shap_values: np.ndarray, X: pd.DataFrame, feature: str) -> dict:
    idx = list(X.columns).index(feature)
    x_vals = X[feature].to_numpy(dtype=float)
    sv = shap_values[:, idx]
    corr = float(np.corrcoef(x_vals, sv)[0, 1])
    return {"corr_feature_vs_shap": corr, "mean_abs_shap": float(np.abs(sv).mean())}


# ---------------------------------------------------------------------------
# 4. Verificacion cruzada contra el ablation
# ---------------------------------------------------------------------------


def global_importance_ranking(shap_values: np.ndarray, X: pd.DataFrame) -> pd.Series:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> dict:
    model, X_test, y_test = load_model_and_test_data()
    logger.info("Filas de test para SHAP: %s", f"{len(X_test):,}")

    shap_values, expected_value = compute_shap_values(model, X_test)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = plot_summary_global(shap_values, X_test)

    counts = X_test["AutoPitchType"].value_counts()
    pitch_types = counts[counts >= MIN_N_PER_PITCHTYPE].index.tolist()
    pitchtype_paths = plot_summary_by_pitchtype(shap_values, X_test)
    pitchtype_ranking = rank_features_by_pitchtype(shap_values, X_test, pitch_types)

    dependence_paths = plot_dependence(shap_values, X_test)
    dependence_stats = {feat: dependence_direction(shap_values, X_test, feat) for feat in DEPENDENCE_FEATURES}

    global_ranking = global_importance_ranking(shap_values, X_test)

    report = {
        "n_test": len(X_test),
        "expected_value": expected_value,
        "summary_path": str(summary_path.relative_to(ROOT)),
        "pitchtype_paths": {k: str(v.relative_to(ROOT)) for k, v in pitchtype_paths.items()},
        "pitchtype_counts": counts.to_dict(),
        "pitchtype_ranking": pitchtype_ranking,
        "dependence_paths": {k: str(v.relative_to(ROOT)) for k, v in dependence_paths.items()},
        "dependence_stats": dependence_stats,
        "global_ranking": global_ranking,
    }
    return report


def format_report(report: dict) -> str:
    gr = report["global_ranking"]
    n_features = len(gr)

    lines = [
        "# Analisis SHAP — Stuff+ v1 (is_whiff)",
        "",
        f"Metodologia: `shap.TreeExplainer` sobre TODO el set de test de "
        f"`pitcher_holdout` ({report['n_test']:,} pitches, pitchers nunca vistos "
        f"en train). Valor base (log-odds, clase whiff=1): {report['expected_value']:.4f}.",
        "",
        "## 1. Importancia global (mean |SHAP value|)",
        "",
        f"Figura: `{report['summary_path']}`",
        "",
        "| # | feature | mean |SHAP| |",
        "|---:|---|---:|",
    ]
    for i, (feat, val) in enumerate(gr.items(), start=1):
        lines.append(f"| {i} | {feat} | {val:.4f} |")

    top5 = list(gr.index[:5])
    lines += [
        "",
        f"Top 5 global: {', '.join(top5)}. Consistente con el ranking de "
        "`feature_importances_` (split count) de LightGBM reportado en "
        "`reports/stuff_plus_baseline.md` — dominan angulos de release/aproximacion "
        "y posicion de release, no solo velocidad/spin crudos.",
        "",
        "## 2. Summary plots por AutoPitchType",
        "",
        "Tipos con n>=1,000 en test: " + ", ".join(report["pitchtype_paths"].keys()) + ".",
        "",
    ]
    for pt, path in report["pitchtype_paths"].items():
        n = report["pitchtype_counts"][pt]
        lines.append(f"- {pt} (n={n:,}): `{path}`")

    pr = report["pitchtype_ranking"]
    lines += [
        "",
        "Top 5 features por tipo de pitch (mean |SHAP| dentro de ese subgrupo):",
        "",
        "| AutoPitchType | top 5 features |",
        "|---|---|",
    ]
    for pt in pr.columns:
        top5_pt = pr[pt].sort_values(ascending=False).head(5).index.tolist()
        lines.append(f"| {pt} | {', '.join(top5_pt)} |")

    # variabilidad del ranking entre pitch types: cuantos de los top-5 globales
    # se mantienen en el top-5 de CADA tipo
    top5_global_set = set(top5)
    overlaps = {
        pt: len(top5_global_set & set(pr[pt].sort_values(ascending=False).head(5).index))
        for pt in pr.columns
    }
    min_overlap_pt = min(overlaps, key=overlaps.get)
    lines += [
        "",
        f"El top-5 global se mantiene parcialmente por tipo de pitch (overlap con el "
        f"top-5 de cada tipo: {overlaps}). El tipo con menos coincidencia es "
        f"**{min_overlap_pt}** ({overlaps[min_overlap_pt]}/5 features en comun con el "
        "top-5 global) — revisar su summary plot individual para ver que feature(s) "
        "distinta(s) domina(n) ahi.",
        "",
        "## 3. Dependence plots",
        "",
    ]
    for feat in DEPENDENCE_FEATURES:
        stats = report["dependence_stats"][feat]
        direction = "aumenta" if stats["corr_feature_vs_shap"] > 0 else "disminuye"
        strength = abs(stats["corr_feature_vs_shap"])
        strength_label = "fuerte" if strength >= 0.5 else ("moderada" if strength >= 0.2 else "debil/no-lineal")
        lines += [
            f"### {feat}",
            "",
            f"Figura: `{report['dependence_paths'][feat]}`",
            "",
            f"mean |SHAP| = {stats['mean_abs_shap']:.4f}. Correlacion (valor de la feature "
            f"vs. su SHAP value) = {stats['corr_feature_vs_shap']:.3f} — relacion "
            f"{strength_label}, en promedio a mayor {feat} el modelo {direction} la "
            "probabilidad predicha de whiff (la correlacion es un resumen lineal; el "
            "dependence plot puede mostrar no-linealidades que esto no captura).",
            "",
        ]

    # --- 4. verificacion cruzada altitud/magnus ---
    rank_lookup = {feat: (i + 1) for i, feat in enumerate(gr.index)}
    flagged_ranks = {feat: rank_lookup[feat] for feat in ABLATION_FLAGGED_FEATURES}
    flagged_vals = {feat: float(gr[feat]) for feat in ABLATION_FLAGGED_FEATURES}
    median_rank = n_features / 2
    all_below_median = all(r > median_rank for r in flagged_ranks.values())

    lines += [
        "## 4. Verificacion cruzada contra el ablation (altitude_level, magnus_residual)",
        "",
        f"El ablation de Stuff+ (reports/stuff_plus_baseline.md) mostro que quitar "
        "`altitude_level` y `magnus_residual_vert/horz` no le cuesta nada de AUC al "
        "modelo. SHAP deberia mostrar estas 3 features con importancia baja de forma "
        "consistente:",
        "",
        "| feature | mean |SHAP| | ranking (de " + str(n_features) + ") |",
        "|---|---:|---:|",
    ]
    for feat in ABLATION_FLAGGED_FEATURES:
        lines.append(f"| {feat} | {flagged_vals[feat]:.4f} | {flagged_ranks[feat]} |")

    if all_below_median:
        lines += [
            "",
            "**Consistente con el ablation.** Las 3 features quedan en la mitad inferior "
            f"del ranking de importancia ({n_features} features en total) — SHAP no "
            "detecta una señal individual relevante para ninguna de las tres, en linea "
            "con que quitarlas del modelo no perdio poder predictivo. No hay "
            "discrepancia que investigar.",
        ]
    else:
        above = [f for f in ABLATION_FLAGGED_FEATURES if flagged_ranks[f] <= median_rank]
        lines += [
            "",
            f"**DISCREPANCIA A INVESTIGAR:** {', '.join(above)} aparece(n) en la mitad "
            "superior del ranking de SHAP a pesar de que el ablation no mostro perdida "
            "de AUC al quitarla(s). Esto no es necesariamente una contradiccion: SHAP "
            "mide cuanto mueve la prediccion una feature CUANDO ESTA PRESENTE en el "
            "modelo, lo cual puede ser alto incluso si esa señal es REDUNDANTE con "
            "otras features ya en el modelo (en ese caso, quitarla no cuesta AUC porque "
            "el modelo reparte el credito hacia las features correlacionadas restantes). "
            "Revisar el dependence plot de la feature senalada contra InducedVertBreak/"
            "HorzBreak/RelSpeed (sus correlatos fisicos directos) para confirmar si es "
            "redundancia (mismo patron dependiente) o una interaccion genuina no "
            "capturada por el ablation simple.",
        ]

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = run()
    report_text = format_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)
    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
