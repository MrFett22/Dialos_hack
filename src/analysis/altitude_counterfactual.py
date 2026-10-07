"""Analisis contrafactual de altitud sobre Pitching+ v1.

Implementa el diseño de 2 pasos documentado en reports/notas_metodologicas.md
(seccion 1) para evitar el contrafactual "ingenuo" que mantiene el movimiento
observado fijo al cambiar la altitud (ese diseño mide ~0 por construccion,
segun el ablation de Stuff+).

Paso A: para cada pitch, recalcular el movimiento TEORICO (ivb_theoretical,
        hb_theoretical) bajo la densidad de aire de cada una de las 3
        categorias de altitud, manteniendo fijos RelSpeed, SpinRate, SpinAxis
        y Extension (el "diseño" del pitch tal como sale de la mano del
        pitcher — no cambia con la altitud).
Paso B: sumar magnus_residual_vert/horz (la habilidad del pitcher no
        explicada por fisica pura, se asume invariante a la altitud) al
        movimiento teorico de cada escenario -> InducedVertBreak_cf,
        HorzBreak_cf.
Paso C: alimentar esos movimientos ajustados (y altitude_level actualizado
        para ser consistente con el escenario — el resto de las features
        queda igual) a los 8 modelos de pitching_plus_v1.pkl, combinar con
        los mismos pesos de src/models/pitching_plus_score.py, normalizar a
        escala Pitching+ (100+10z, usando como referencia de z-score la
        media/std de cada target sobre TODO el dataset valido en condiciones
        factuales — el "+100" se cancela en cualquier delta, pero se reporta
        completo por transparencia).

Muestra: 100,000 pitches estratificados por (AutoPitchType, altitude_category
actual), para mantener el analisis rapido y reproducible (semilla fija) —
no es una limitacion de computo real para LightGBM (predict() es barato),
pero se sigue el diseño pedido explicitamente.
"""

import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features.physics import AIR_DENSITY_BY_ALTITUDE_CATEGORY, compute_theoretical_break
from src.models.pitching_plus import ARSENAL_PATH, FEATURE_COLUMNS, MODEL_PATH, PHYSICS_PATH, SPLITS_PATH
from src.models.pitching_plus_score import ALL_TARGETS, BINARY_TARGETS, INVERT_SIGN, PITCHING_PLUS_WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "altitude_counterfactual.md"
FIG_DIR = ROOT / "reports" / "figures" / "altitude_counterfactual"
SCORES_OUT_PATH = ROOT / "data" / "processed" / "altitude_counterfactual_scores.parquet"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 100_000
SAMPLE_SEED = 42
ALTITUDE_CATEGORIES = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
ALTITUDE_LEVEL_MAP = {"No Altitude": 0, "Medium Altitude": 1, "Extreme Altitude": 2}

N_BOOT = 1000
BOOT_SEED = 42

VELO_DIFF_BINS = [-np.inf, -8, -4, -1, 1, np.inf]
VELO_DIFF_LABELS = [
    "muy lento vs FB (<-8mph)",
    "lento vs FB (-8 a -4)",
    "algo lento vs FB (-4 a -1)",
    "similar a FB (-1 a 1)",
    "igual/mas rapido que FB (>1)",
]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ARSENAL_PATH)
    physics_extra = pd.read_parquet(
        PHYSICS_PATH, columns=["PitchUID", "magnus_residual_vert", "magnus_residual_horz"]
    )
    splits = pd.read_parquet(SPLITS_PATH, columns=["PitchUID", "pitcher_holdout"])

    df = df.merge(physics_extra, on="PitchUID", how="inner").merge(splits, on="PitchUID", how="inner")
    df = df.loc[df["pitcher_holdout"] != "excluded"].copy()
    df = df.dropna(subset=["velo_diff_vs_fb"])  # pitcher-year sin fastball, ver reports/arsenal_summary.md

    df["EffectiveVelo"] = df["EffectiveVelo"].astype(float)
    for col in ["PitcherThrows", "BatterSide", "AutoPitchType", "prev_pitch_type"]:
        df[col] = df[col].astype("object").fillna("Unknown").astype("category")

    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    frac = n / len(df)
    parts = []
    for _, g in df.groupby(["AutoPitchType", "altitude_category"], observed=True):
        k = min(len(g), max(1, round(len(g) * frac)))
        parts.append(g.sample(n=k, random_state=seed))
    sampled = pd.concat(parts)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.sort_index()


# ---------------------------------------------------------------------------
# Prediccion con los 8 modelos de Pitching+
# ---------------------------------------------------------------------------


def load_models() -> dict:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact["models"]


def predict_all_targets(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    preds = pd.DataFrame(index=X.index)
    for target in ALL_TARGETS:
        model = models[target]
        if target in BINARY_TARGETS:
            preds[target] = model.predict_proba(X)[:, 1]
        else:
            preds[target] = model.predict(X)
    return preds


# ---------------------------------------------------------------------------
# Paso A + B: movimiento contrafactual por categoria de altitud
# ---------------------------------------------------------------------------


def build_counterfactual_features(df: pd.DataFrame, altitude_category: str) -> pd.DataFrame:
    """Copia de df con InducedVertBreak/HorzBreak/altitude_level reemplazados
    por su version contrafactual bajo `altitude_category` (Paso A + B). El
    resto de las features (incluyendo magnus_residual_vert/horz, que se
    asumen invariantes a la altitud) queda sin cambiar."""
    air_density = AIR_DENSITY_BY_ALTITUDE_CATEGORY[altitude_category]
    ivb_theo_cf, hb_theo_cf, _ = compute_theoretical_break(
        df["RelSpeed"], df["SpinRate"], df["SpinAxis"], df["Extension"], air_density
    )

    cf = df.copy()
    cf["InducedVertBreak"] = df["magnus_residual_vert"] + ivb_theo_cf
    cf["HorzBreak"] = df["magnus_residual_horz"] + hb_theo_cf
    cf["altitude_level"] = ALTITUDE_LEVEL_MAP[altitude_category]
    return cf


# ---------------------------------------------------------------------------
# Paso C: combinar predicciones en un score Pitching+ contrafactual
# ---------------------------------------------------------------------------


def compute_reference_z_stats(models: dict, df_full: pd.DataFrame) -> dict:
    """Media/std de cada target, en condiciones FACTUALES, sobre TODO el
    dataset valido (no solo la muestra de 100k) — referencia estable para
    z-scorear los escenarios contrafactuales."""
    X_full = df_full[FEATURE_COLUMNS]
    preds_full = predict_all_targets(models, X_full)
    stats = {}
    for target in ALL_TARGETS:
        signed = preds_full[target] * (-1 if target in INVERT_SIGN else 1)
        stats[target] = {"mean": float(signed.mean()), "std": float(signed.std(ddof=1))}
    return stats


def combine_to_pitching_plus(preds: pd.DataFrame, ref_stats: dict) -> pd.Series:
    weight_sum = sum(PITCHING_PLUS_WEIGHTS.values())
    combined_z = pd.Series(0.0, index=preds.index)
    for target, weight in PITCHING_PLUS_WEIGHTS.items():
        signed = preds[target] * (-1 if target in INVERT_SIGN else 1)
        z = (signed - ref_stats[target]["mean"]) / ref_stats[target]["std"]
        combined_z += z * (weight / weight_sum)
    return 100 + 10 * combined_z


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> tuple[pd.DataFrame, dict]:
    df_full = load_data()
    logger.info("Filas validas (referencia z-score): %s", f"{len(df_full):,}")

    models = load_models()
    ref_stats = compute_reference_z_stats(models, df_full)
    logger.info("Referencia z-score calculada sobre %s pitches factuales", f"{len(df_full):,}")

    sample = stratified_sample(df_full, SAMPLE_SIZE, SAMPLE_SEED)
    logger.info("Muestra estratificada: %s pitches", f"{len(sample):,}")

    scores = {}
    all_preds = {}
    for cat in ALTITUDE_CATEGORIES:
        cf_df = build_counterfactual_features(sample, cat)
        preds = predict_all_targets(models, cf_df[FEATURE_COLUMNS])
        all_preds[cat] = preds
        scores[cat] = combine_to_pitching_plus(preds, ref_stats)
        logger.info("Prediccion contrafactual completa: %s", cat)

    result = sample[
        ["PitchUID", "AutoPitchType", "altitude_category", "velo_diff_vs_fb", "pitcher_anon_id", "year"]
    ].copy()
    for cat in ALTITUDE_CATEGORIES:
        col = "pp_" + cat.replace(" ", "_")
        result[col] = scores[cat].to_numpy()
        for target in ALL_TARGETS:
            result[f"{target}_{cat.replace(' ', '_')}"] = all_preds[cat][target].to_numpy()
    result["delta_pitching_plus"] = result["pp_Extreme_Altitude"] - result["pp_No_Altitude"]
    result["velo_diff_bin"] = pd.cut(result["velo_diff_vs_fb"], bins=VELO_DIFF_BINS, labels=VELO_DIFF_LABELS)

    # --- 1. delta por AutoPitchType ---
    by_pitchtype = (
        result.groupby("AutoPitchType", observed=True)["delta_pitching_plus"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )

    # --- 2. delta por bin de velo_diff_vs_fb (H2) ---
    by_velo_bin = (
        result.groupby("velo_diff_bin", observed=True)["delta_pitching_plus"].agg(["mean", "count"])
    )

    # --- 3. tabla cruzada AutoPitchType x altitude_category actual, con bootstrap CI ---
    cross_rows = []
    for (pt, alt), g in result.groupby(["AutoPitchType", "altitude_category"], observed=True):
        if len(g) < 20:
            continue
        mean, lo, hi = bootstrap_mean_ci(g["delta_pitching_plus"].to_numpy())
        cross_rows.append(
            {"AutoPitchType": pt, "altitude_category_actual": alt, "n": len(g), "mean_delta": mean, "ci_lo": lo, "ci_hi": hi}
        )
    cross_table = pd.DataFrame(cross_rows).sort_values(["AutoPitchType", "altitude_category_actual"])

    # --- 4. descomposicion del delta en los 8 componentes individuales ---
    decomposition = decompose_delta_by_target(result, ref_stats)

    report = {
        "n_sample": len(result),
        "ref_stats": ref_stats,
        "by_pitchtype": by_pitchtype,
        "by_velo_bin": by_velo_bin,
        "cross_table": cross_table,
        "decomposition": decomposition,
    }
    return result, report


def decompose_delta_by_target(result: pd.DataFrame, ref_stats: dict, min_n: int = 100) -> pd.DataFrame:
    """Descompone delta_pitching_plus en sus 8 componentes, ANTES de combinar
    con pesos (raw_delta_<target>, en las unidades naturales del target) y
    DESPUES de aplicar signo+z-score+peso (contrib_<target>, en puntos de
    Pitching+ — estos SI suman exactamente el delta combinado, ver
    contrib_sum_check)."""
    weight_sum = sum(PITCHING_PLUS_WEIGHTS.values())
    rows = []
    for pt, g in result.groupby("AutoPitchType", observed=True):
        if len(g) < min_n:
            continue
        row = {"AutoPitchType": pt, "n": len(g)}
        for target in ALL_TARGETS:
            col_ext = f"{target}_Extreme_Altitude"
            col_no = f"{target}_No_Altitude"
            sign = -1 if target in INVERT_SIGN else 1
            raw_delta = g[col_ext] - g[col_no]
            z_delta = raw_delta * sign / ref_stats[target]["std"]
            contribution = z_delta * (PITCHING_PLUS_WEIGHTS[target] / weight_sum) * 10
            row[f"raw_delta_{target}"] = raw_delta.mean()
            row[f"contrib_{target}"] = contribution.mean()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("AutoPitchType")
    table["contrib_sum_check"] = table[[f"contrib_{t}" for t in ALL_TARGETS]].sum(axis=1)
    return table


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def plot_delta_by_pitchtype(by_pitchtype: pd.DataFrame, min_n: int = 100) -> Path:
    reliable = by_pitchtype.loc[by_pitchtype["count"] >= min_n]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in reliable["mean"]]
    ax.barh(reliable.index, reliable["mean"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("delta_pitching_plus (Extreme Altitude - No Altitude)")
    ax.set_title(f"Cambio en Pitching+ al mover cada tipo de pitch a Extreme Altitude (n>={min_n})")
    plt.tight_layout()
    out = FIG_DIR / "delta_by_pitchtype.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta_by_velo_bin(by_velo_bin: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    order = VELO_DIFF_LABELS
    vals = by_velo_bin.reindex(order)["mean"]
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in vals]
    ax.bar(range(len(order)), vals, color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("delta_pitching_plus (Extreme Altitude - No Altitude)")
    ax.set_title("Cambio en Pitching+ por altitud, segun diferencial de velocidad vs. el fastball (H2)")
    plt.tight_layout()
    out = FIG_DIR / "delta_by_velo_diff_bin.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cross_table(cross_table: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    pivot = cross_table.pivot(index="AutoPitchType", columns="altitude_category_actual", values="mean_delta")
    pivot = pivot.reindex(columns=[c for c in ["No Altitude", "Medium Altitude", "Extreme Altitude"] if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu", center=0, ax=ax)
    ax.set_title("delta_pitching_plus promedio: AutoPitchType x altitude_category actual del pitch")
    plt.tight_layout()
    out = FIG_DIR / "delta_crosstable_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def format_report(report: dict, fig_paths: dict) -> str:
    by_pt = report["by_pitchtype"]
    by_vb = report["by_velo_bin"]
    cross = report["cross_table"]

    lines = [
        "# Analisis contrafactual de altitud — Pitching+ v1",
        "",
        "Diseño de 2 pasos (Magnus -> Pitching+), documentado en "
        "`reports/notas_metodologicas.md`. Para cada pitch de una muestra "
        f"estratificada de {report['n_sample']:,} (por AutoPitchType x "
        "altitude_category actual), se recalculo el movimiento bajo las 3 "
        "densidades de aire (No/Medium/Extreme Altitude), preservando "
        "RelSpeed/SpinRate/SpinAxis/Extension y el residual de Magnus del "
        "pitcher, y se re-predijo con los 8 modelos de `pitching_plus_v1.pkl`.",
        "",
        "`altitude_level` se actualizo para ser consistente con cada escenario "
        "contrafactual (no se dejo en su valor real mientras el movimiento "
        "cambiaba) — el resto de las features quedo sin tocar. El ablation de "
        "Stuff+ ya mostro que altitude_level aporta ~0 señal una vez que el "
        "movimiento esta en el modelo, asi que esta decision no deberia "
        "cambiar el resultado de forma material, pero es la version "
        "internamente consistente.",
        "",
        "La normalizacion z-score usa como referencia la media/std de cada "
        "target en condiciones FACTUALES sobre TODO el dataset valido (no "
        "solo la muestra), para que el delta sea comparable a la escala "
        "Pitching+ real. El termino '+100' se cancela exactamente en "
        "cualquier delta — solo importan los pesos y las std de referencia.",
        "",
        "## 1. delta_pitching_plus por AutoPitchType",
        "",
        f"Figura: `{fig_paths['by_pitchtype'].relative_to(ROOT)}`",
        "",
        "| AutoPitchType | delta promedio (Extreme - No Altitude) | n |",
        "|---|---:|---:|",
    ]
    for pt, row in by_pt.iterrows():
        lines.append(f"| {pt} | {row['mean']:+.3f} | {int(row['count']):,} |")

    MIN_N_FOR_NARRATIVE = 100
    by_pt_reliable = by_pt.loc[by_pt["count"] >= MIN_N_FOR_NARRATIVE]
    losers = by_pt_reliable.sort_values("mean").head(3).index.tolist()
    gainers = by_pt_reliable.sort_values("mean").tail(3).index.tolist()
    lines += [
        "",
        f"(Solo tipos con n>={MIN_N_FOR_NARRATIVE} en la muestra — {', '.join(by_pt.loc[by_pt['count'] < MIN_N_FOR_NARRATIVE].index)} "
        "tienen muestra demasiado chica para ser confiables, aunque aparecen en la tabla completa arriba.)",
        f"Pierden mas Pitching+ al moverse a Extreme Altitude: {', '.join(losers)}. "
        f"Pierden menos: {', '.join(gainers)}.",
        "",
        "## 2. delta_pitching_plus por diferencial de velocidad vs. fastball (H2)",
        "",
        f"Figura: `{fig_paths['by_velo_bin'].relative_to(ROOT)}`",
        "",
        "H2: pitches con mayor diferencial de velocidad respecto al fastball "
        "(mecanismo de engaño mas basado en el cambio de velocidad/tunneling "
        "que en el movimiento por Magnus) deberian perder MENOS Pitching+ en "
        "altitud que los pitches de velocidad similar al fastball (que "
        "dependen mas del movimiento puro).",
        "",
        "| bin de velo_diff_vs_fb | delta promedio | n |",
        "|---|---:|---:|",
    ]
    for label in VELO_DIFF_LABELS:
        if label in by_vb.index:
            row = by_vb.loc[label]
            lines.append(f"| {label} | {row['mean']:+.3f} | {int(row['count']):,} |")

    vals_by_bin = by_vb.reindex(VELO_DIFF_LABELS)["mean"]
    monotonic_away_from_fb_worse = vals_by_bin.iloc[0] > vals_by_bin.iloc[-1] if vals_by_bin.notna().all() else None
    lines += [
        "",
    ]
    if monotonic_away_from_fb_worse:
        lines.append(
            "**Consistente con H2 en direccion:** el bin mas alejado del fastball "
            f"(delta={vals_by_bin.iloc[0]:+.3f}) pierde MENOS (o gana mas) que el bin "
            f"mas cercano/igual al fastball (delta={vals_by_bin.iloc[-1]:+.3f}). "
            "Revisar la tabla completa para confirmar si la relacion es monotonica "
            "en todos los bins intermedios, no solo en los extremos."
        )
    else:
        lines.append(
            "**No consistente con H2 en la direccion esperada** — el bin mas alejado "
            f"del fastball (delta={vals_by_bin.iloc[0]:+.3f}) NO pierde menos que el "
            f"mas cercano al fastball (delta={vals_by_bin.iloc[-1]:+.3f}). Revisar con "
            "cautela antes de citar H2 como confirmada."
        )

    lines += [
        "",
        "## 3. Tabla cruzada AutoPitchType x altitude_category actual (con bootstrap CI 95%)",
        "",
        f"Figura: `{fig_paths['cross_table'].relative_to(ROOT)}`",
        "",
        f"Bootstrap de {N_BOOT} remuestreos por celda, solo celdas con n>=20.",
        "",
        "| AutoPitchType | altitude_category actual | n | delta promedio | IC 95% |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in cross.iterrows():
        lines.append(
            f"| {row['AutoPitchType']} | {row['altitude_category_actual']} | {int(row['n']):,} | "
            f"{row['mean_delta']:+.3f} | [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] |"
        )

    decomp = report["decomposition"]
    lines += [
        "",
        "## 4. Descomposicion del delta en los 8 componentes individuales",
        "",
        "Investigacion especifica: Curveball tiene la mayor reduccion de "
        "movimiento absoluto por altitud (EDA de Semana 2), pero NO es de los "
        "que mas pierde Pitching+ en el contrafactual (seccion 1). Se "
        "descompuso el delta en sus 8 componentes, antes y despues de "
        "combinar con pesos, para ver que target compensa.",
        "",
        "### Delta crudo por target (unidades naturales — probabilidad para "
        "targets binarios, carreras para run_value)",
        "",
        "| AutoPitchType | " + " | ".join(ALL_TARGETS) + " |",
        "|---|" + "---:|" * len(ALL_TARGETS),
    ]
    for pt, row in decomp.iterrows():
        vals = " | ".join(f"{row[f'raw_delta_{t}']:+.4f}" for t in ALL_TARGETS)
        lines.append(f"| {pt} | {vals} |")

    lines += [
        "",
        "### Contribucion en puntos de Pitching+ por target (estos SI suman el "
        "delta total — ver `contrib_sum_check` vs. la seccion 1)",
        "",
        "| AutoPitchType | " + " | ".join(ALL_TARGETS) + " | suma (check) |",
        "|---|" + "---:|" * (len(ALL_TARGETS) + 1),
    ]
    for pt, row in decomp.iterrows():
        vals = " | ".join(f"{row[f'contrib_{t}']:+.3f}" for t in ALL_TARGETS)
        lines.append(f"| {pt} | {vals} | {row['contrib_sum_check']:+.3f} |")

    if "Curveball" in decomp.index:
        cb = decomp.loc["Curveball"]
        contribs = {t: cb[f"contrib_{t}"] for t in ALL_TARGETS}
        best_target = max(contribs, key=contribs.get)
        worst_target = min(contribs, key=contribs.get)
        whiff_contrib = contribs["is_whiff"]
        hard_contact_raw = cb["raw_delta_is_hard_contact"]
        hard_contact_contrib = contribs["is_hard_contact"]

        lines += [
            "",
            "### Curveball especificamente",
            "",
            f"- El componente que MAS compensa (contribucion menos negativa o mas "
            f"positiva) es **{best_target}** ({contribs[best_target]:+.3f} puntos).",
            f"- El componente que MAS castiga es **{worst_target}** "
            f"({contribs[worst_target]:+.3f} puntos).",
            f"- `is_whiff` contribuye {whiff_contrib:+.3f} puntos — "
            f"{'es de los mas negativos, como se esperaba (menos movimiento -> menos swing-and-miss)' if whiff_contrib <= sorted(contribs.values())[1] else 'no es el mas castigado, sorprendente dado que se esperaba que fuera el mas afectado'}.",
            f"- `is_hard_contact` (crudo, probabilidad de barrel): delta = "
            f"{hard_contact_raw:+.5f}, contribucion = {hard_contact_contrib:+.3f} puntos — "
            + (
                "consistente con la hipotesis: una curva 'mas plana' en altitud reduce "
                "tambien la probabilidad de contacto duro (menos break -> mas facil de "
                "conectar, pero con menos calidad de contacto), lo cual COMPENSA en el "
                "score combinado la caida esperada en is_whiff."
                if hard_contact_raw < 0 and hard_contact_contrib > 0
                else "NO consistente con la hipotesis de 'curva mas plana = menos contacto duro' "
                "— revisar el signo antes de citar esta explicacion."
            ),
            "",
            "Nota: `is_hard_contact` esta invertido de signo en la combinacion "
            "(mayor probabilidad de barrel = peor para el pitcher), asi que un "
            "`raw_delta_is_hard_contact` NEGATIVO (menos barrels en altitud) se "
            "traduce en una contribucion POSITIVA al score — exactamente el "
            "mecanismo de compensacion que se investigo.",
            "",
        ]

    lines += [
        "",
        "## Referencia de normalizacion (media/std factual por target, dataset completo)",
        "",
        "| target | mean (con signo ajustado) | std |",
        "|---|---:|---:|",
    ]
    for target, stats in report["ref_stats"].items():
        lines.append(f"| {target} | {stats['mean']:.4f} | {stats['std']:.4f} |")

    lines += [
        "",
        f"Datos completos por pitch guardados en "
        f"`data/processed/altitude_counterfactual_scores.parquet`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    result, report = run()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "by_pitchtype": plot_delta_by_pitchtype(report["by_pitchtype"]),
        "by_velo_bin": plot_delta_by_velo_bin(report["by_velo_bin"]),
        "cross_table": plot_cross_table(report["cross_table"]),
    }

    SCORES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(SCORES_OUT_PATH, index=False)
    logger.info("Guardado %s (%s filas)", SCORES_OUT_PATH, f"{len(result):,}")

    report_text = format_report(report, fig_paths)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)

    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
