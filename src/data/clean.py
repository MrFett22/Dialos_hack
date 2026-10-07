"""Limpieza del dataframe crudo de pitches hacia data/processed/pitches_clean.parquet."""

import argparse
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "raw" / "stuff_model_df.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pitches_clean.parquet"
DEFAULT_REPORT = ROOT / "reports" / "data_cleaning_report.md"

CRITICAL_STUFF_FEATURES = [
    "RelSpeed",
    "SpinRate",
    "InducedVertBreak",
    "HorzBreak",
    "Extension",
    "RelHeight",
    "RelSide",
]

ALTITUDE_LEVEL_MAP = {
    "No Altitude": 0,
    "Medium Altitude": 1,
    "Extreme Altitude": 2,
}

GROUPING_ID_COLUMNS = [
    "game_anon_id",
    "pitcher_anon_id",
    "batter_anon_id",
    "catcher_anon_id",
]

# columnas de las que otros pipelines rio abajo dependen para filtrar/agrupar/
# hacer join (ids) o como feature de contexto obligatoria (altitude_category) —
# un nulo aqui se propaga silenciosamente si no se chequea desde clean.py.
# altitude_category se agrego despues de que su nulo (2,703 filas con pitcher
# valido, ver reports/pitcher_id_null_investigation.md) se descubrio recien al
# construir src/validation/splits.py, no en la limpieza original.
KEY_DEPENDENCY_COLUMNS = GROUPING_ID_COLUMNS + ["altitude_category"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("Cargado %s: %s filas x %s columnas", path.name, f"{len(df):,}", df.shape[1])
    return df


def drop_missing_critical_features(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    per_column_nulls = {c: int(df[c].isna().sum()) for c in cols}
    missing_mask = df[cols].isna().any(axis=1)
    n_dropped = int(missing_mask.sum())
    report = {"per_column_nulls": per_column_nulls, "rows_dropped": n_dropped}
    return df.loc[~missing_mask].copy(), report


def standardize_count(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    known_balls_strikes = df["Balls"].notna() & df["Strikes"].notna()
    expected_count = pd.Series(index=df.index, dtype="object")
    expected_count[known_balls_strikes] = (
        df.loc[known_balls_strikes, "Balls"].astype(int).astype(str)
        + "-"
        + df.loc[known_balls_strikes, "Strikes"].astype(int).astype(str)
    )

    mismatch_mask = known_balls_strikes & (df["count"] != expected_count)
    n_mismatched = int(mismatch_mask.sum())
    n_missing_balls_strikes = int((~known_balls_strikes).sum())

    df.loc[mismatch_mask, "count"] = expected_count[mismatch_mask]

    report = {
        "rows_fixed": n_mismatched,
        "rows_with_missing_balls_or_strikes": n_missing_balls_strikes,
    }
    return df, report


def check_pitchuid_uniqueness(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_null_uid = int(df["PitchUID"].isna().sum())
    duplicated_mask = df["PitchUID"].notna() & df.duplicated(subset=["PitchUID"], keep=False)
    n_duplicated_rows = int(duplicated_mask.sum())

    # dentro de cada grupo con PitchUID repetido, las filas 100% identicas son
    # duplicacion pura (se elimina el sobrante, quedandose con la primera ocurrencia)
    exact_dup_mask = df.duplicated(keep="first") & duplicated_mask
    n_exact_duplicates_dropped = int(exact_dup_mask.sum())
    df = df.loc[~exact_dup_mask].copy()

    # lo que sigue compartiendo PitchUID despues de quitar duplicados exactos son
    # filas con el mismo id pero datos distintos: conflicto real, no se decide solo
    conflicting_mask = df["PitchUID"].notna() & df.duplicated(subset=["PitchUID"], keep=False)
    n_conflicting = int(conflicting_mask.sum())
    if n_conflicting > 0:
        logger.warning(
            "%s filas comparten PitchUID con datos distintos (no se eliminan automaticamente, "
            "requieren revision manual).",
            n_conflicting,
        )

    report = {
        "null_pitchuid": n_null_uid,
        "duplicated_pitchuid_rows": n_duplicated_rows,
        "exact_duplicate_rows_dropped": n_exact_duplicates_dropped,
        "conflicting_duplicate_rows_kept": n_conflicting,
    }
    return df, report


def validate_altitude_category(df: pd.DataFrame) -> dict:
    expected = set(ALTITUDE_LEVEL_MAP)
    observed = set(df["altitude_category"].dropna().unique())
    unexpected_values = sorted(observed - expected)

    n_unexpected = int(df["altitude_category"].isin(unexpected_values).sum()) if unexpected_values else 0
    n_null = int(df["altitude_category"].isna().sum())

    if unexpected_values:
        logger.warning("Valores inesperados en altitude_category: %s", unexpected_values)

    return {
        "unexpected_values": unexpected_values,
        "rows_with_unexpected_value": n_unexpected,
        "rows_with_null": n_null,
    }


def validate_key_dependency_columns(df: pd.DataFrame) -> dict:
    """Reporta (sin eliminar filas) nulos en columnas de las que otros
    pipelines dependen rio abajo: las de ID para agrupar/hacer join
    (game_anon_id, pitcher_anon_id, batter_anon_id, catcher_anon_id) y
    altitude_category (feature de contexto obligatoria para Stuff+, y
    dimension directa de altitude_holdout). No se eliminan filas aqui porque
    distintos sub-modelos pueden tolerar distintos nulos (ej. Location+ no
    depende de pitcher_anon_id) — la decision de exclusion se toma rio abajo,
    en cada pipeline de features/split/entrenamiento (ver
    reports/pitcher_id_null_investigation.md para la decision aplicada).

    NOTA: el nulo de altitude_category (2,703 filas con pitcher_anon_id VALIDO
    — un problema distinto al de pitcher_anon_id, no la misma causa raiz) se
    paso por alto en la version original de esta funcion y solo se detecto al
    construir src/validation/splits.py. Se agrega aqui explicitamente para que
    no se repita con otra columna en el futuro.
    """
    null_counts = {col: int(df[col].isna().sum()) for col in KEY_DEPENDENCY_COLUMNS}
    for col, n in null_counts.items():
        if n > 0:
            logger.warning(
                "%s tiene %s filas con nulo (%.2f%% del total) — no se elimina aqui, "
                "ver reports/data_cleaning_report.md",
                col,
                n,
                100 * n / len(df),
            )
    return null_counts


def add_altitude_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["altitude_level"] = df["altitude_category"].map(ALTITUDE_LEVEL_MAP).astype("Int8")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_input = len(df)
    report: dict = {"rows_input": n_input}

    df, missing_report = drop_missing_critical_features(df, CRITICAL_STUFF_FEATURES)
    report["missing_critical_features"] = missing_report

    df, count_report = standardize_count(df)
    report["count_standardization"] = count_report

    df, pitchuid_report = check_pitchuid_uniqueness(df)
    report["pitchuid"] = pitchuid_report

    report["altitude_category"] = validate_altitude_category(df)
    report["key_dependency_nulls"] = validate_key_dependency_columns(df)

    df = add_altitude_level(df)

    report["rows_output"] = len(df)
    report["rows_dropped_total"] = n_input - len(df)
    return df, report


def format_report(report: dict) -> str:
    mcf = report["missing_critical_features"]
    cnt = report["count_standardization"]
    uid = report["pitchuid"]
    alt = report["altitude_category"]

    lines = [
        "# Reporte de limpieza — pitches_clean.parquet",
        "",
        f"Filas de entrada: **{report['rows_input']:,}**  ",
        f"Filas de salida: **{report['rows_output']:,}**  ",
        f"Filas eliminadas en total: **{report['rows_dropped_total']:,}**",
        "",
        "## 1. Nulos en stuff_feature criticas",
        "",
        f"Filas eliminadas por tener al menos un nulo en "
        f"{', '.join(CRITICAL_STUFF_FEATURES)}: **{mcf['rows_dropped']:,}**",
        "",
        "Nulos por columna (antes de eliminar):",
        "",
    ]
    lines += [f"- `{col}`: {n:,}" for col, n in mcf["per_column_nulls"].items()]
    lines += [
        "",
        "## 2. Estandarizacion de `count` a partir de Balls/Strikes",
        "",
        f"- Filas con `count` inconsistente respecto a Balls-Strikes, corregidas: "
        f"**{cnt['rows_fixed']:,}**",
        f"- Filas con Balls o Strikes nulos (no se pudieron validar/corregir): "
        f"{cnt['rows_with_missing_balls_or_strikes']:,}",
        "",
        "## 3. Unicidad de PitchUID",
        "",
        f"- PitchUID nulos: {uid['null_pitchuid']:,}",
        f"- Filas involucradas en algun PitchUID duplicado: {uid['duplicated_pitchuid_rows']:,}",
        f"- De esas, duplicados exactos (fila identica) eliminados: "
        f"{uid['exact_duplicate_rows_dropped']:,}",
        f"- Duplicados con datos en conflicto (NO eliminados, requieren revision manual): "
        f"{uid['conflicting_duplicate_rows_kept']:,}",
        "",
        "## 4. Validacion de altitude_category",
        "",
        f"- Valores inesperados encontrados: {alt['unexpected_values'] or 'ninguno'}",
        f"- Filas con valor inesperado: {alt['rows_with_unexpected_value']:,}",
        f"- Filas con altitude_category nulo (quedan con altitude_level nulo, no se eliminan): "
        f"{alt['rows_with_null']:,}",
        "",
        "## 5. Columna ordinal `altitude_level`",
        "",
        "Creada a partir de altitude_category: "
        "`No Altitude` -> 0, `Medium Altitude` -> 1, `Extreme Altitude` -> 2, "
        "valores nulos o inesperados -> nulo (dtype `Int8`, usable como categorica u ordinal).",
        "",
        "## 6. Nulos en columnas clave (IDs de agrupacion + altitude_category)",
        "",
        "No se eliminan filas aqui — distintos sub-modelos/splits toleran distinto "
        "nivel de nulo en estas columnas (ej. Location+ no depende de "
        "pitcher_anon_id, pero Stuff+/Pitching+ si; altitude_category la necesitan "
        "los 3, ver Stuff+ paso 3.2). La decision de exclusion se documenta en "
        "`reports/pitcher_id_null_investigation.md` y se aplica rio abajo, en "
        "`src/validation/splits.py`. `altitude_category` se agrego a esta "
        "validacion despues de que su nulo (2,703 filas con pitcher valido, un "
        "problema DISTINTO al de pitcher_anon_id) se descubriera recien al "
        "construir los splits, no en la limpieza original — de ahi la necesidad "
        "de este chequeo explicito para no repetir el mismo descuido con otra "
        "columna.",
        "",
    ]
    kdn = report["key_dependency_nulls"]
    for col, n in kdn.items():
        lines.append(f"- `{col}`: {n:,} nulos ({100*n/report['rows_output']:.2f}% del dataset limpio)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    df_raw = load_raw(args.input)
    df_clean, report = clean(df_raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(args.output, index=False)
    logger.info("Guardado dataset limpio en %s (%s filas)", args.output, f"{len(df_clean):,}")

    report_text = format_report(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    logger.info("Guardado reporte de limpieza en %s", args.report)

    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
