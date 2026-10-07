"""Splits de train/test para los sub-modelos Stuff+, Location+ y Pitching+.

Parte de data/processed/pitches_with_physics.parquet y genera 3 esquemas de
holdout, guardados en data/processed/splits/splits.parquet (una fila por
PitchUID, con una columna train/test/excluded por esquema) para poder unir
por PitchUID a cualquier tabla de features:

1. pitcher_holdout   : 20% de pitchers (no de pitches) a test, estratificado
                        por altitude_category modal del pitcher y por cuartil
                        de volumen de pitches (para no sesgar el holdout hacia
                        pitchers de muestra chica o grande).
2. season_holdout     : year=2026 a test, resto a train. Se documenta la
                        cobertura parcial de 2026 causada por el problema de
                        anonimizacion (ver reports/pitcher_id_null_investigation.md).
3. altitude_holdout   : Extreme Altitude a test, resto a train (reemplaza el
                        park_holdout original — no hay variable de estadio).

En los tres esquemas se excluyen primero las filas con `pitcher_anon_id` nulo
(3,008 pitches, 0.47%) y/o `altitude_category` nulo (3,019 pitches, 0.48%;
2,703 de ellas con pitcher valido — un problema DISTINTO, no la misma causa
raiz, ver reports/pitcher_id_null_investigation.md seccion "altitude_category
nulo"). Ambas exclusiones se aplican a los 3 sub-modelos por igual (no solo a
los que dependen directamente de esa columna) para mantener el mismo universo
de pitches comparable entre Stuff+/Location+/Pitching+ — Stuff+ necesita
altitude_category como feature (paso 3.2), asi que aunque pitcher_holdout y
season_holdout no la necesiten para construirse, se excluye ahi tambien por
consistencia.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "pitches_with_physics.parquet"
OUTPUT_DIR = ROOT / "data" / "processed" / "splits"
DEFAULT_REPORT = ROOT / "reports" / "splits_summary.md"

SEED = 42
PITCHER_TEST_FRACTION = 0.20
PARTIAL_2026_RATIO_THRESHOLD = 0.20  # ver check_2026_partial_coverage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Carga + exclusion de pitcher_anon_id nulo y/o altitude_category nulo
# ---------------------------------------------------------------------------


def load_and_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_total = len(df)
    null_pitcher_mask = df["pitcher_anon_id"].isna()
    null_altitude_mask = df["altitude_category"].isna()
    null_both_mask = null_pitcher_mask & null_altitude_mask
    exclude_mask = null_pitcher_mask | null_altitude_mask

    df_valid = df.loc[~exclude_mask].copy()
    report = {
        "n_total_input": n_total,
        "n_excluded_null_pitcher": int(null_pitcher_mask.sum()),
        "n_excluded_null_altitude": int(null_altitude_mask.sum()),
        "n_excluded_null_both": int(null_both_mask.sum()),
        "n_excluded_total": int(exclude_mask.sum()),
        "n_valid": len(df_valid),
        "n_unique_pitchers_valid": int(df_valid["pitcher_anon_id"].nunique()),
    }
    return df_valid, report


# ---------------------------------------------------------------------------
# 2. pitcher_holdout: 20% de pitchers, estratificado
# ---------------------------------------------------------------------------


def _pitcher_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("pitcher_anon_id").agg(n_pitches=("PitchUID", "size"))
    modal_altitude = df.groupby("pitcher_anon_id")["altitude_category"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"
    )
    stats["altitude_mode"] = modal_altitude
    stats["pitch_count_quartile"] = pd.qcut(
        stats["n_pitches"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    stats["stratum"] = stats["altitude_mode"].astype(str) + "|" + stats["pitch_count_quartile"].astype(str)
    return stats


def build_pitcher_holdout(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.Series, dict]:
    stats = _pitcher_level_stats(df)
    rng = np.random.default_rng(seed)

    test_pitchers: set = set()
    for _, group in stats.groupby("stratum"):
        pitchers = group.index.to_numpy()
        n_test = round(len(pitchers) * PITCHER_TEST_FRACTION)
        if n_test > 0:
            chosen = rng.choice(pitchers, size=n_test, replace=False)
            test_pitchers.update(chosen)

    assignment = pd.Series("train", index=stats.index, name="pitcher_holdout")
    assignment.loc[sorted(test_pitchers)] = "test"

    stats["split"] = assignment
    balance_altitude = (
        pd.crosstab(stats["altitude_mode"], stats["split"], normalize="columns") * 100
    ).round(2)
    balance_quartile = (
        pd.crosstab(stats["pitch_count_quartile"], stats["split"], normalize="columns") * 100
    ).round(2)

    report = {
        "n_pitchers_total": len(stats),
        "n_pitchers_test": int((assignment == "test").sum()),
        "n_pitchers_train": int((assignment == "train").sum()),
        "balance_altitude_mode_pct": balance_altitude,
        "balance_pitch_count_quartile_pct": balance_quartile,
        "quartile_edges": stats["n_pitches"].quantile([0.25, 0.5, 0.75]).to_dict(),
    }
    return assignment, report


# ---------------------------------------------------------------------------
# 3. season_holdout: year=2026 a test
# ---------------------------------------------------------------------------


def check_2026_partial_coverage(df: pd.DataFrame) -> dict:
    """Pitchers con datos validos en 2026 Y en 2024/2025, cuyo volumen 2026 es
    anomalamente bajo respecto a su promedio historico — posible sintoma del
    problema de anonimizacion (ver reports/pitcher_id_null_investigation.md),
    no una fragmentacion que podamos corregir (un pitch nulo no se puede
    reasignar a un pitcher especifico a posteriori).
    """
    counts = df.groupby(["pitcher_anon_id", "year"]).size().unstack(fill_value=0)
    has_2026 = counts.get(2026, pd.Series(dtype=int)) > 0
    has_hist = (counts.get(2024, pd.Series(dtype=int)) > 0) | (counts.get(2025, pd.Series(dtype=int)) > 0)
    both = counts.loc[has_2026 & has_hist].copy()

    n_hist_years = (counts.get(2024, 0) > 0).astype(int) + (counts.get(2025, 0) > 0).astype(int)
    hist_avg = (counts.get(2024, 0) + counts.get(2025, 0)) / n_hist_years.replace(0, 1)
    ratio = counts.get(2026, 0) / hist_avg.replace(0, np.nan)
    ratio_both = ratio.loc[both.index]

    flagged = ratio_both[ratio_both < PARTIAL_2026_RATIO_THRESHOLD]

    return {
        "n_pitchers_2026": int(has_2026.sum()),
        "n_pitchers_2026_only": int((has_2026 & ~has_hist).sum()),
        "n_pitchers_2026_with_history": int(len(both)),
        "n_pitchers_flagged_partial": int(len(flagged)),
        "threshold": PARTIAL_2026_RATIO_THRESHOLD,
        "flagged_pitchers": flagged.round(3).sort_values().to_dict(),
    }


def build_season_holdout(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    assignment = pd.Series(
        np.where(df["year"] == 2026, "test", "train"), index=df.index, name="season_holdout"
    )
    coverage_report = check_2026_partial_coverage(df)
    report = {
        "n_pitches_train": int((assignment == "train").sum()),
        "n_pitches_test": int((assignment == "test").sum()),
        "n_pitchers_train": int(df.loc[assignment == "train", "pitcher_anon_id"].nunique()),
        "n_pitchers_test": int(df.loc[assignment == "test", "pitcher_anon_id"].nunique()),
        "coverage_2026": coverage_report,
    }
    return assignment, report


# ---------------------------------------------------------------------------
# 4. altitude_holdout: Extreme Altitude a test
# ---------------------------------------------------------------------------


def build_altitude_holdout(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    conditions = [
        df["altitude_category"] == "Extreme Altitude",
        df["altitude_category"].isin(["No Altitude", "Medium Altitude"]),
    ]
    choices = ["test", "train"]
    assignment = pd.Series(
        np.select(conditions, choices, default="excluded"), index=df.index, name="altitude_holdout"
    )
    report = {
        "n_pitches_train": int((assignment == "train").sum()),
        "n_pitches_test": int((assignment == "test").sum()),
        "n_pitches_excluded_null_altitude": int((assignment == "excluded").sum()),
        "n_pitchers_train": int(df.loc[assignment == "train", "pitcher_anon_id"].nunique()),
        "n_pitchers_test": int(df.loc[assignment == "test", "pitcher_anon_id"].nunique()),
    }
    return assignment, report


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_splits(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_valid, filter_report = load_and_filter(raw_df)
    report: dict = {"filter": filter_report}

    pitcher_assignment, pitcher_report = build_pitcher_holdout(df_valid)
    report["pitcher_holdout"] = pitcher_report

    season_assignment, season_report = build_season_holdout(df_valid)
    report["season_holdout"] = season_report

    altitude_assignment, altitude_report = build_altitude_holdout(df_valid)
    report["altitude_holdout"] = altitude_report

    out = pd.DataFrame({"PitchUID": raw_df["PitchUID"]})
    out = out.set_index(raw_df.index)

    out["pitcher_holdout"] = "excluded"
    out.loc[df_valid.index, "pitcher_holdout"] = df_valid["pitcher_anon_id"].map(pitcher_assignment).to_numpy()

    out["season_holdout"] = "excluded"
    out.loc[df_valid.index, "season_holdout"] = season_assignment.to_numpy()

    out["altitude_holdout"] = "excluded"
    out.loc[df_valid.index, "altitude_holdout"] = altitude_assignment.to_numpy()

    return out.reset_index(drop=True), report


def format_report(report: dict) -> str:
    f = report["filter"]
    ph = report["pitcher_holdout"]
    sh = report["season_holdout"]
    ah = report["altitude_holdout"]
    cov = sh["coverage_2026"]

    lines = [
        "# Reporte de splits — data/processed/splits/splits.parquet",
        "",
        "## Filtro base (aplicado a los 3 esquemas por igual)",
        "",
        f"- Pitches totales en pitches_with_physics.parquet: {f['n_total_input']:,}",
        f"- Excluidos por `pitcher_anon_id` nulo: {f['n_excluded_null_pitcher']:,} "
        f"({100*f['n_excluded_null_pitcher']/f['n_total_input']:.2f}%)",
        f"- Excluidos por `altitude_category` nulo: {f['n_excluded_null_altitude']:,} "
        f"({100*f['n_excluded_null_altitude']/f['n_total_input']:.2f}%)",
        f"- (de los cuales {f['n_excluded_null_both']:,} tienen AMBOS nulos — se cuentan una sola vez)",
        f"- Excluidos en total (union de ambos): {f['n_excluded_total']:,} "
        f"({100*f['n_excluded_total']/f['n_total_input']:.2f}%) — ver "
        "reports/pitcher_id_null_investigation.md",
        f"- Pitches validos usados para los 3 splits: {f['n_valid']:,}",
        f"- Pitchers unicos validos: {f['n_unique_pitchers_valid']:,}",
        "",
        "## 1. pitcher_holdout",
        "",
        f"- Pitchers en train: {ph['n_pitchers_train']:,} | en test: {ph['n_pitchers_test']:,} "
        f"({100*ph['n_pitchers_test']/ph['n_pitchers_total']:.1f}% de {ph['n_pitchers_total']:,})",
        f"- Cuartiles de pitches-por-pitcher usados para estratificar: "
        f"p25={ph['quartile_edges'][0.25]:.0f}, p50={ph['quartile_edges'][0.5]:.0f}, "
        f"p75={ph['quartile_edges'][0.75]:.0f}",
        "",
        "Balance de altitude_category modal del pitcher (%), train vs test — deberia "
        "verse parecido si la estratificacion funciono:",
        "",
        ph["balance_altitude_mode_pct"].to_markdown(),
        "",
        "Balance de cuartil de volumen de pitches del pitcher (%), train vs test:",
        "",
        ph["balance_pitch_count_quartile_pct"].to_markdown(),
        "",
        "## 2. season_holdout (year=2026 a test)",
        "",
        f"- Pitches train: {sh['n_pitches_train']:,} | test: {sh['n_pitches_test']:,}",
        f"- Pitchers train: {sh['n_pitchers_train']:,} | test: {sh['n_pitchers_test']:,}",
        "",
        "### Cobertura parcial de 2026 por el problema de anonimizacion",
        "",
        f"- Pitchers con datos validos en 2026: {cov['n_pitchers_2026']:,}",
        f"- De esos, sin historial previo (solo aparecen en 2026): {cov['n_pitchers_2026_only']:,}",
        f"- Con historial en 2024 y/o 2025 tambien: {cov['n_pitchers_2026_with_history']:,}",
        f"- **Pitchers flageados con cobertura 2026 sospechosamente baja** "
        f"(volumen 2026 < {int(cov['threshold']*100)}% de su promedio historico "
        f"2024-2025): **{cov['n_pitchers_flagged_partial']}**",
        "",
        "Nota importante: un pitch con `pitcher_anon_id` nulo NO se puede reasignar "
        "a un pitcher especifico a posteriori — ya esta excluido de todos los "
        "splits (ver filtro base arriba). Lo que se reporta aqui es un sintoma "
        "indirecto: pitchers con historial 2024/2025 solido cuyo conteo 2026 cae "
        "muy por debajo de lo esperado, consistente con haber perdido parte de "
        "sus apariciones de 2026 en alguno de los 34 juegos con el problema de "
        "anonimizacion. **No se excluyen de season_holdout** (sus pitches validos "
        "restantes son datos reales, no ruido), pero su senal en 2026 debe "
        "tratarse con mas cautela. Como season_holdout parte estrictamente por "
        "`year`, todos los pitches validos de 2026 de un pitcher quedan enteros "
        "en test (no hay fragmentacion adicional dentro del split).",
        "",
        "Top 10 pitchers mas flageados (2026 vs. promedio historico):",
        "",
        "| pitcher_anon_id | ratio 2026/historico |",
        "|---|---:|",
    ]
    for pid, r in list(cov["flagged_pitchers"].items())[:10]:
        lines.append(f"| {pid} | {r:.3f} |")

    lines += [
        "",
        "## 3. altitude_holdout (Extreme Altitude a test)",
        "",
        f"- Pitches train (No Altitude + Medium Altitude): {ah['n_pitches_train']:,}",
        f"- Pitches test (Extreme Altitude): {ah['n_pitches_test']:,}",
        f"- Excluidos por altitude_category nulo: {ah['n_pitches_excluded_null_altitude']:,}",
        f"- Pitchers train: {ah['n_pitchers_train']:,} | test: {ah['n_pitchers_test']:,}",
        "",
        f"Nota: un pitcher puede aparecer en ambos lados de este split (lanza tanto en "
        "Extreme Altitude como en otros estadios) — es esperado, altitude_holdout "
        "particiona por PITCH, no por pitcher.",
        "",
        f"## Reproducibilidad",
        "",
        f"Seed fijo: `{SEED}`. Splits guardados en `data/processed/splits/splits.parquet` "
        "(una fila por PitchUID, columnas pitcher_holdout/season_holdout/altitude_holdout "
        "con valores train/test/excluded).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    raw_df = pd.read_parquet(args.input)
    logger.info("Cargado %s: %s filas", args.input.name, f"{len(raw_df):,}")

    splits_df, report = build_splits(raw_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "splits.parquet"
    splits_df.to_parquet(out_path, index=False)
    logger.info("Guardado %s (%s filas)", out_path, f"{len(splits_df):,}")

    report_text = format_report(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    logger.info("Guardado reporte en %s", args.report)


if __name__ == "__main__":
    main()
