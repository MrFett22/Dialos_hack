"""Construccion de columnas objetivo (targets) para el modelo Stuff+ LMB.

Toma data/processed/pitches_clean.parquet (salida de src/data/clean.py) y agrega,
a partir de las columnas `feature_role == target_only` del diccionario de datos
(data/raw/stuff_plus_data_dictionary.csv), las columnas objetivo pedidas:
is_whiff, is_chase, is_called_strike, is_weak_contact, is_groundball,
is_hard_contact (barrel), run_value y is_out_in_pa.

LIMITACION IMPORTANTE sobre run_value: el dataset no trae columnas de corredores
en base (no hay on_1b/on_2b/on_3b). Para construir la matriz de 24 estados
base-out reconstruimos el estado de corredores con una simulacion secuencial por
media-entrada, usando el orden cronologico real embebido en PitchUID (es un GUID
version 1 / time-based UUID, que trae timestamp) y heurísticas estandar de avance
de bases (ver `_resolve_plate_appearance`). No modela robos de base, bolas
pasadas, wild pitches, ni identifica con certeza a que corredor especifico se le
hizo el out en un fielder's choice. RunsScored y OutsOnPlay (columnas reales del
dataset) se usan como fuente de verdad para carreras y outs; solo la ocupacion de
bases es una aproximacion.
"""

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "pitches_clean.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
DEFAULT_REPORT = ROOT / "reports" / "targets_summary.md"
DATA_DICTIONARY = ROOT / "data" / "raw" / "stuff_plus_data_dictionary.csv"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NEW_TARGET_COLUMNS = [
    "is_whiff",
    "is_chase",
    "is_called_strike",
    "is_weak_contact",
    "is_groundball",
    "is_hard_contact",
    "run_value",
    "is_out_in_pa",
]

# anclas conocidas y publicamente citadas del clasificador de Barrel de Statcast:
# 98 mph -> rango de angulo [26, 30] grados; 116+ mph -> rango maximo [8, 50] grados.
# MLB no publica la formula exacta; esta es la aproximacion estandar por
# interpolacion lineal entre esos dos puntos, usada ampliamente en la comunidad
# sabermetrica. Ver docstring de `_is_barrel` para el detalle.
BARREL_MIN_EXIT_SPEED = 98.0
BARREL_MAX_EXIT_SPEED_ANCHOR = 116.0
BARREL_ANGLE_RANGE_AT_MIN = (26.0, 30.0)
BARREL_ANGLE_RANGE_AT_MAX = (8.0, 50.0)

WEAK_CONTACT_EXIT_SPEED_THRESHOLD = 85.0


# ---------------------------------------------------------------------------
# 1-3. Targets simples (existentes o derivacion directa de una sola columna)
# ---------------------------------------------------------------------------


def validate_is_whiff(df: pd.DataFrame) -> dict:
    """`is_whiff` ya existe en el dataset; valida que este bien poblada."""
    n_null = int(df["is_whiff"].isna().sum())
    not_swing = int(((df["is_whiff"] == 1) & (df["is_swing"] == 0)).sum())
    contact_and_whiff = int(((df["is_whiff"] == 1) & (df["is_contact"] == 1)).sum())
    matches_swinging_strike = int((df["is_whiff"] == df["is_swinging_strike"]).mean() * 100)
    return {
        "n_null": n_null,
        "n_whiff_1": int((df["is_whiff"] == 1).sum()),
        "inconsistent_not_a_swing": not_swing,
        "inconsistent_whiff_and_contact": contact_and_whiff,
        "pct_matches_is_swinging_strike": matches_swinging_strike,
    }


def validate_is_called_strike(df: pd.DataFrame) -> dict:
    """`is_called_strike` ya existe; valida contra PitchCall == 'StrikeCalled'."""
    n_null = int(df["is_called_strike"].isna().sum())
    expected = (df["PitchCall"] == "StrikeCalled").astype(int)
    mismatch = int((df["is_called_strike"].astype(int) != expected).sum())
    return {
        "n_null": n_null,
        "n_called_strike_1": int((df["is_called_strike"] == 1).sum()),
        "mismatch_vs_pitchcall": mismatch,
    }


def fix_is_hit_by_pitch(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reconstruye `is_hit_by_pitch` desde PitchCall.

    La columna original del dataset vale 0 en TODAS las filas, aunque hay
    ~2,276 pitches con PitchCall == 'HitByPitch' (play_result coincide en
    2,254; los 22 restantes vienen marcados además como base por bolas en
    KorBB/play_result). Verificado a mano: después de un golpeado, el
    siguiente pitch de la media entrada es de otro bateador en el 99.96% de
    los casos y empieza en 0-0 en el 99.3%. Con la columna en 0, el golpeado
    no cerraba el turno, no ponía al corredor en primera y su run_value
    quedaba en 0 (auditoría 2026-10-05, reports/auditoria_matematica.md C1).
    """
    df = df.copy()
    before = int(pd.to_numeric(df["is_hit_by_pitch"], errors="coerce").fillna(0).sum())
    df["is_hit_by_pitch"] = (df["PitchCall"] == "HitByPitch").astype(int)
    after = int(df["is_hit_by_pitch"].sum())
    return df, {
        "n_before": before,
        "n_after": after,
        "n_also_flagged_walk": int(((df["is_hit_by_pitch"] == 1) & (df["base_on_balls"] == 1)).sum()),
        "n_play_result_hbp": int((df["play_result"] == "HitByPitch").sum()),
    }


def add_is_chase(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_chase"] = df["swung_outside_strike_zone"].astype("Int8")
    return df


# ---------------------------------------------------------------------------
# 4-6. Targets de contacto (requieren ExitSpeed / Angle / hit_type)
# ---------------------------------------------------------------------------


def add_is_weak_contact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    in_play = df["is_ball_in_play"] == 1
    weak = in_play & (df["ExitSpeed"] < WEAK_CONTACT_EXIT_SPEED_THRESHOLD)
    result = pd.Series(0, index=df.index, dtype="Int8")
    result[weak] = 1
    # bola en juego pero sin ExitSpeed medido: resultado indefinido, no 0
    result[in_play & df["ExitSpeed"].isna()] = pd.NA
    df["is_weak_contact"] = result
    return df


def add_is_groundball(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_groundball"] = (df["hit_type"] == "GroundBall").astype("Int8")
    return df


def _barrel_angle_range(exit_speed: float) -> tuple[float, float]:
    if exit_speed >= BARREL_MAX_EXIT_SPEED_ANCHOR:
        return BARREL_ANGLE_RANGE_AT_MAX
    frac = (exit_speed - BARREL_MIN_EXIT_SPEED) / (
        BARREL_MAX_EXIT_SPEED_ANCHOR - BARREL_MIN_EXIT_SPEED
    )
    lo = BARREL_ANGLE_RANGE_AT_MIN[0] + frac * (
        BARREL_ANGLE_RANGE_AT_MAX[0] - BARREL_ANGLE_RANGE_AT_MIN[0]
    )
    hi = BARREL_ANGLE_RANGE_AT_MIN[1] + frac * (
        BARREL_ANGLE_RANGE_AT_MAX[1] - BARREL_ANGLE_RANGE_AT_MIN[1]
    )
    return lo, hi


def add_is_hard_contact(df: pd.DataFrame) -> pd.DataFrame:
    """Barrel: ExitSpeed >= 98 mph y Angle dentro del rango permitido para esa
    velocidad. El rango de angulo es [26, 30] grados a 98 mph y se ensancha
    linealmente hasta [8, 50] grados a 116 mph o mas (ver BARREL_* arriba).
    """
    df = df.copy()
    in_play = df["is_ball_in_play"] == 1
    has_measurement = df["ExitSpeed"].notna() & df["Angle"].notna()

    exit_speed = df["ExitSpeed"].fillna(-np.inf)
    lo = pd.Series(np.nan, index=df.index)
    hi = pd.Series(np.nan, index=df.index)
    eligible = has_measurement & (exit_speed >= BARREL_MIN_EXIT_SPEED)
    ranges = exit_speed[eligible].map(_barrel_angle_range)
    lo[eligible] = ranges.map(lambda r: r[0])
    hi[eligible] = ranges.map(lambda r: r[1])

    is_barrel = eligible & (df["Angle"] >= lo) & (df["Angle"] <= hi)

    result = pd.Series(0, index=df.index, dtype="Int8")
    result[in_play & is_barrel] = 1
    result[in_play & ~has_measurement] = pd.NA
    df["is_hard_contact"] = result
    return df


# ---------------------------------------------------------------------------
# 7. run_value: reconstruccion de estados base-out + matriz RE24 + diferencia
# ---------------------------------------------------------------------------


def _add_uuid_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_uuid_time"] = df["PitchUID"].map(lambda s: uuid.UUID(s).time)
    return df


def _assign_plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena cronologicamente (via timestamp embebido en PitchUID) y asigna
    un id de media-entrada y un numero de plate appearance dentro de ella."""
    df = df.sort_values(["game_anon_id", "_uuid_time"]).copy()
    df["_half_inning_id"] = (
        df["game_anon_id"].astype(str)
        + "_"
        + df["Inning"].astype(int).astype(str)
        + "_"
        + df["Top/Bottom"].astype(str)
    )
    df["_pa_end"] = (
        (df["strikeout"] == 1)
        | (df["base_on_balls"] == 1)
        | (df["is_hit_by_pitch"] == 1)
        | (df["is_ball_in_play"] == 1)
    )
    shifted_end = df.groupby("_half_inning_id")["_pa_end"].shift(fill_value=False)
    df["_pa_num"] = shifted_end.groupby(df["_half_inning_id"]).cumsum().astype(int)
    return df


def _advance_on_hit(
    bases: tuple[bool, bool, bool], n_bases: int, runs_scored: int
) -> tuple[bool, bool, bool]:
    """Avanza corredores (y bateador) en un hit, reconciliado contra RunsScored real.

    Paso 1 (avance ingenuo): cada corredor y el bateador avanzan exactamente
    n_bases bases (single=1, double=2, triple=3, home_run=4). Esto puede anotar
    de mas o de menos corredores respecto a RunsScored real, porque no sabemos
    si un corredor se detuvo en una base intermedia o si tomo una base extra.

    Paso 2 (reconciliacion): si el avance ingenuo anota menos corredores que
    RunsScored real, se hace anotar a los que quedaron en base empezando por el
    MAS adelantado (3B, luego 2B, luego 1B) — regla de desempate explicita:
    no sabemos con certeza cual corredor especifico anoto, asumimos el que
    tenia menos distancia por recorrer. Si el avance ingenuo anota MAS
    corredores que RunsScored real, se "deshace" la anotacion del/los
    corredor(es) que partieron de la base MENOS adelantada (mas lejos de
    home) — regla de desempate simetrica, igual de arbitraria en la identidad
    exacta del corredor — y se les coloca en la base libre mas cercana a home
    (3B primero, luego 2B, luego 1B).
    """
    on1, on2, on3 = bases
    origins_most_advanced_first = [(2, on3), (1, on2), (0, on1)]

    occupied = [False, False, False]
    naive_scored_origins = []  # base_idx de origen, en orden mas-adelantado-primero
    for base_idx, occ in origins_most_advanced_first:
        if not occ:
            continue
        dest = (base_idx + 1) + n_bases
        if dest >= 4:
            naive_scored_origins.append(base_idx)
        else:
            occupied[dest - 1] = True

    batter_scores = n_bases >= 4
    if not batter_scores:
        occupied[n_bases - 1] = True

    naive_scored = len(naive_scored_origins) + int(batter_scores)
    diff = runs_scored - naive_scored

    if diff > 0:
        for base_idx in [2, 1, 0]:
            if diff <= 0:
                break
            if occupied[base_idx]:
                occupied[base_idx] = False
                diff -= 1
    elif diff < 0:
        # candidatos a "des-anotar", del menos avanzado al mas avanzado; el
        # bateador (si conto) es el candidato menos natural, se deshace al final
        undo_candidates = list(reversed(naive_scored_origins))
        if batter_scores:
            undo_candidates.append(-1)
        for _ in range(min(-diff, len(undo_candidates))):
            undo_candidates.pop(0)
            for base_idx in [2, 1, 0]:
                if not occupied[base_idx]:
                    occupied[base_idx] = True
                    break

    return tuple(occupied)


def _force_advance_walk(bases: tuple[bool, bool, bool]) -> tuple[bool, bool, bool]:
    on1, on2, on3 = bases
    if not on1:
        return (True, on2, on3)
    if not on2:
        return (True, True, on3)
    return (True, True, True)


def _generic_out_hold(bases: tuple[bool, bool, bool], runs_scored: int) -> tuple[bool, bool, bool]:
    """Bola en juego que no fue hit (out, fielder's choice, sacrificio, error).

    Simplificacion: los corredores se quedan en su base por default; si el
    play produjo carreras (ej. sacrifice fly), se remueven del diamante
    empezando por el mas adelantado (3B, luego 2B, luego 1B) hasta agotar
    RunsScored. No identifica con certeza a que corredor especifico se le
    hizo el out en un fielder's choice.
    """
    b = list(bases)
    remaining = runs_scored
    for i in [2, 1, 0]:
        if remaining <= 0:
            break
        if b[i]:
            b[i] = False
            remaining -= 1
    return tuple(b)


def _resolve_plate_appearance(
    bases: tuple[bool, bool, bool], row: pd.Series
) -> tuple[tuple[bool, bool, bool], int, int]:
    """Devuelve (bases_despues, outs_en_la_jugada, runs_anotadas) para el
    ultimo pitch de una plate appearance, usando RunsScored/OutsOnPlay reales
    como fuente de verdad y una heuristica de avance para la ocupacion de bases.
    """
    runs = int(row["RunsScored"])
    outs_on_play = int(row["OutsOnPlay"])

    if row["strikeout"] == 1:
        new_bases = bases
    elif row["base_on_balls"] == 1 or row["is_hit_by_pitch"] == 1:
        new_bases = _force_advance_walk(bases)
    elif row["single"] == 1:
        new_bases = _advance_on_hit(bases, 1, runs)
    elif row["double"] == 1:
        new_bases = _advance_on_hit(bases, 2, runs)
    elif row["triple"] == 1:
        new_bases = _advance_on_hit(bases, 3, runs)
    elif row["home_run"] == 1:
        new_bases = _advance_on_hit(bases, 4, runs)
    else:
        new_bases = _generic_out_hold(bases, runs)

    return new_bases, outs_on_play, runs


def _simulate_base_out_states(df: pd.DataFrame) -> pd.DataFrame:
    """Simula, PA por PA y media-entrada por media-entrada, el estado
    base-out antes y despues de cada plate appearance. Devuelve una tabla a
    nivel (half_inning, pa_num) con el estado antes/despues y las carreras.
    """
    pa_terminal = df.loc[df["_pa_end"]].sort_values(["_half_inning_id", "_uuid_time"])

    records = []
    current_half_inning = None
    bases = (False, False, False)
    outs = 0
    for _, row in pa_terminal.iterrows():
        if row["_half_inning_id"] != current_half_inning:
            current_half_inning = row["_half_inning_id"]
            bases = (False, False, False)
            outs = 0

        outs_before = min(int(row["Outs"]), 2)  # 8 filas con Outs==3 (dato invalido), se acotan
        state_before = (*bases, outs_before)

        new_bases, outs_on_play, runs = _resolve_plate_appearance(bases, row)
        outs_after = min(outs_before + outs_on_play, 3)
        state_after = (*new_bases, outs_after) if outs_after < 3 else (False, False, False, 3)

        records.append(
            {
                "_half_inning_id": row["_half_inning_id"],
                "_pa_num": row["_pa_num"],
                "state_before": state_before,
                "state_after": state_after,
                "runs_scored": runs,
            }
        )

        bases, outs = new_bases, outs_after

    return pd.DataFrame.from_records(records)


def _build_run_expectancy_table(pa_states: pd.DataFrame) -> dict:
    """Matriz de 24 estados base-out: RE(estado) = promedio de carreras
    anotadas desde ese estado hasta el final de la media-entrada."""
    pa_states = pa_states.sort_values(["_half_inning_id", "_pa_num"]).copy()
    reversed_cumsum = (
        pa_states.iloc[::-1]
        .groupby("_half_inning_id")["runs_scored"]
        .cumsum()
        .iloc[::-1]
    )
    pa_states["runs_remaining_from_here"] = reversed_cumsum

    re_table = (
        pa_states.groupby("state_before")["runs_remaining_from_here"].mean().to_dict()
    )
    for on1 in (False, True):
        for on2 in (False, True):
            for on3 in (False, True):
                re_table[(on1, on2, on3, 3)] = 0.0  # 3 outs: no quedan carreras por anotar
    return re_table


def add_run_value(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = _add_uuid_time(df)
    df = _assign_plate_appearances(df)

    pa_states = _simulate_base_out_states(df)
    re_table = _build_run_expectancy_table(pa_states)

    pa_states["re_before"] = pa_states["state_before"].map(re_table)
    pa_states["re_after"] = pa_states["state_after"].map(re_table)
    pa_states["run_value"] = (
        pa_states["re_after"] + pa_states["runs_scored"] - pa_states["re_before"]
    )

    # estado "antes" es el mismo para todos los pitches de una PA (antes de que
    # el ultimo pitch resuelva la jugada); para pitches no-terminales run_value=0
    # porque el estado base-out no cambia hasta el ultimo pitch de la PA.
    pa_state_before = pa_states.set_index(["_half_inning_id", "_pa_num"])["state_before"]
    re_before_by_pa = pa_states.set_index(["_half_inning_id", "_pa_num"])["re_before"]
    run_value_terminal = pa_states.set_index(["_half_inning_id", "_pa_num"])["run_value"]

    df = df.set_index(["_half_inning_id", "_pa_num"], drop=False)
    df["state_before"] = pa_state_before.reindex(df.index)
    df["_re_before"] = re_before_by_pa.reindex(df.index)
    # 0 por default (el estado base-out no cambia dentro de una PA ya resuelta);
    # NaN donde la PA no se pudo resolver (borde de datos, ver _re_before nulo)
    df["run_value"] = np.where(df["_re_before"].isna(), np.nan, 0.0)
    df.loc[df["_pa_end"], "run_value"] = run_value_terminal.reindex(
        df.loc[df["_pa_end"]].index
    ).to_numpy()
    df = df.reset_index(drop=True)

    n_states_missing_re = int(df["_re_before"].isna().sum())

    report = {
        "n_plate_appearances": len(pa_states),
        "n_half_innings": df["_half_inning_id"].nunique(),
        "re_table": {str(k): round(v, 3) for k, v in sorted(re_table.items())},
        "n_pitches_without_re_state": n_states_missing_re,
    }
    return df, report


# ---------------------------------------------------------------------------
# 8. is_out_in_pa
# ---------------------------------------------------------------------------


def add_is_out_in_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega, a nivel de plate appearance, si esta termino en out del
    bateador (strikeout o bola en juego que no fue hit). Se transmite (broadcast)
    a todos los pitches de esa PA. NOTA: por limitacion de datos no distingue
    embasarse por error de embasarse por fielder's choice (ambos is_hit==0);
    ambos quedan marcados como "out" de forma aproximada.
    """
    df = df.copy()
    is_out = ((df["strikeout"] == 1) | ((df["is_ball_in_play"] == 1) & (df["is_hit"] == 0))).astype(
        "Int8"
    )
    is_out_terminal = is_out.where(df["_pa_end"])

    pa_key = ["_half_inning_id", "_pa_num"]
    pa_outcome = (
        pd.DataFrame({**{k: df[k] for k in pa_key}, "is_out_in_pa": is_out_terminal})
        .dropna(subset=["is_out_in_pa"])
        .drop_duplicates(subset=pa_key)
        .set_index(pa_key)["is_out_in_pa"]
    )

    df = df.set_index(pa_key, drop=False)
    df["is_out_in_pa"] = pa_outcome.reindex(df.index).astype("Int8")
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Verificacion de fuga de informacion features <-> targets
# ---------------------------------------------------------------------------


def check_feature_target_leakage(df: pd.DataFrame) -> dict:
    data_dict = pd.read_csv(DATA_DICTIONARY)
    stuff_feature_cols = data_dict.loc[
        data_dict["feature_role"] == "stuff_feature", "column_name"
    ].tolist()
    target_only_cols = data_dict.loc[
        data_dict["feature_role"] == "target_only", "column_name"
    ].tolist()

    overlap_dict = sorted(set(stuff_feature_cols) & set(target_only_cols))
    overlap_new_targets = sorted(set(stuff_feature_cols) & set(NEW_TARGET_COLUMNS))

    known_outcome_cols = set(target_only_cols) | {
        "PlateLocHeight",
        "PlateLocSide",
        "in_strike_zone",
        "outside_strike_zone",
    }
    stuff_feature_cols_using_outcome = sorted(set(stuff_feature_cols) & known_outcome_cols)

    numeric_stuff_features = [c for c in stuff_feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    binary_targets = ["is_whiff", "is_chase", "is_called_strike", "is_weak_contact", "is_groundball", "is_hard_contact", "is_out_in_pa"]
    binary_targets = [c for c in binary_targets if c in df.columns]

    suspicious_correlations = []
    for target in binary_targets:
        target_numeric = pd.to_numeric(df[target], errors="coerce")
        if target_numeric.notna().sum() < 2 or target_numeric.nunique() < 2:
            continue
        for feat in numeric_stuff_features:
            corr = df[feat].corr(target_numeric)
            if pd.notna(corr) and abs(corr) >= 0.3:
                suspicious_correlations.append(
                    {"feature": feat, "target": target, "corr": round(float(corr), 3)}
                )
    suspicious_correlations.sort(key=lambda r: -abs(r["corr"]))

    return {
        "stuff_feature_cols_overlap_target_only": overlap_dict,
        "stuff_feature_cols_overlap_new_targets": overlap_new_targets,
        "stuff_feature_cols_using_outcome_data": stuff_feature_cols_using_outcome,
        "suspicious_correlations_ge_0_30": suspicious_correlations,
    }


# ---------------------------------------------------------------------------
# Distribuciones y reporte
# ---------------------------------------------------------------------------


def distribution_by_altitude(df: pd.DataFrame) -> pd.DataFrame:
    binary_targets = [
        "is_whiff",
        "is_chase",
        "is_called_strike",
        "is_weak_contact",
        "is_groundball",
        "is_hard_contact",
        "is_out_in_pa",
    ]
    rows = []
    for target in binary_targets:
        prop = df.groupby("altitude_category", dropna=False)[target].apply(
            lambda s: pd.to_numeric(s, errors="coerce").mean()
        )
        rows.append(prop.rename(target))
    table = pd.concat(rows, axis=1).T
    return table.round(4)


def run_value_by_altitude(df: pd.DataFrame) -> pd.DataFrame:
    terminal = df.loc[df["_pa_end"]]
    return (
        terminal.groupby("altitude_category", dropna=False)["run_value"]
        .agg(["mean", "std", "count"])
        .round(4)
    )


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {}

    report["is_whiff_validation"] = validate_is_whiff(df)
    report["is_called_strike_validation"] = validate_is_called_strike(df)
    df, report["is_hit_by_pitch_fix"] = fix_is_hit_by_pitch(df)

    df = add_is_chase(df)
    df = add_is_weak_contact(df)
    df = add_is_groundball(df)
    df = add_is_hard_contact(df)
    df, run_value_report = add_run_value(df)
    report["run_value"] = run_value_report
    df = add_is_out_in_pa(df)

    report["leakage_check"] = check_feature_target_leakage(df)
    report["distribution_by_altitude"] = distribution_by_altitude(df)
    report["run_value_by_altitude"] = run_value_by_altitude(df)

    df = df.drop(columns=["_uuid_time", "_half_inning_id", "_pa_num", "_pa_end", "state_before", "_re_before"])
    return df, report


def format_report(report: dict) -> str:
    wf = report["is_whiff_validation"]
    cs = report["is_called_strike_validation"]
    rv = report["run_value"]
    leak = report["leakage_check"]

    lines = [
        "# Reporte de targets — pitches_with_targets.parquet",
        "",
        "## 1. Validacion de `is_whiff` (columna existente)",
        "",
        f"- Nulos: {wf['n_null']:,}",
        f"- Filas con is_whiff==1: {wf['n_whiff_1']:,}",
        f"- Coincide con is_swinging_strike en {wf['pct_matches_is_swinging_strike']}% de las filas",
        f"- Inconsistencias is_whiff==1 pero is_swing==0: {wf['inconsistent_not_a_swing']:,}",
        f"- **Inconsistencias is_whiff==1 y is_contact==1 al mismo tiempo (contradiccion logica): "
        f"{wf['inconsistent_whiff_and_contact']:,}** — revisar con el organizador antes de usarla sin filtrar.",
        "",
        "## 3. Validacion de `is_called_strike` (columna existente)",
        "",
        f"- Nulos: {cs['n_null']:,}",
        f"- Filas con is_called_strike==1: {cs['n_called_strike_1']:,}",
        f"- Discrepancias vs PitchCall=='StrikeCalled': {cs['mismatch_vs_pitchcall']:,}",
        "",
        "## 3b. `is_hit_by_pitch` (reconstruida desde PitchCall)",
        "",
        f"- La columna original vale 1 en {report['is_hit_by_pitch_fix']['n_before']:,} filas (está vacía en el dataset).",
        f"- Reconstruida como PitchCall == 'HitByPitch': {report['is_hit_by_pitch_fix']['n_after']:,} golpeados "
        f"({report['is_hit_by_pitch_fix']['n_play_result_hbp']:,} con play_result == 'HitByPitch'; "
        f"{report['is_hit_by_pitch_fix']['n_also_flagged_walk']:,} marcados además como base por bolas).",
        "- Cierra el turno y pone al bateador en primera (avance forzado, igual que la base por bolas).",
        "",
        "## 2, 4, 5, 6. Targets derivados",
        "",
        "- `is_chase` = swung_outside_strike_zone (copia directa).",
        f"- `is_weak_contact`: 1 si ExitSpeed < {WEAK_CONTACT_EXIT_SPEED_THRESHOLD} mph e "
        "is_ball_in_play==1; nulo si is_ball_in_play==1 pero ExitSpeed no se midio.",
        "- `is_groundball`: 1 si hit_type == 'GroundBall'.",
        f"- `is_hard_contact` (barrel): ExitSpeed >= {BARREL_MIN_EXIT_SPEED} mph con Angle dentro del "
        f"rango [{BARREL_ANGLE_RANGE_AT_MIN[0]},{BARREL_ANGLE_RANGE_AT_MIN[1]}] grados a "
        f"{BARREL_MIN_EXIT_SPEED} mph, ensanchando linealmente hasta "
        f"[{BARREL_ANGLE_RANGE_AT_MAX[0]},{BARREL_ANGLE_RANGE_AT_MAX[1]}] grados a "
        f"{BARREL_MAX_EXIT_SPEED_ANCHOR}+ mph (aproximacion estandar del Barrel de Statcast; "
        "MLB no publica la formula exacta).",
        "",
        "## 7. `run_value` (matriz de 24 estados base-out)",
        "",
        f"- Plate appearances reconstruidas: {rv['n_plate_appearances']:,} en "
        f"{rv['n_half_innings']:,} medias-entradas.",
        f"- Pitches sin estado RE valido (bordes de datos / Outs invalido): "
        f"{rv['n_pitches_without_re_state']:,}",
        "- Metodologia: se ordenan los pitches cronologicamente usando el timestamp "
        "embebido en PitchUID (UUID version 1), se agrupan en plate appearances y se "
        "simula la ocupacion de bases con heuristicas de avance estandar (ver docstring "
        "del modulo). RunsScored y OutsOnPlay reales se usan como fuente de verdad; solo "
        "la ocupacion de bases es aproximada.",
        "- run_value es 0 para pitches que no terminan la plate appearance (el estado "
        "base-out no cambia hasta el ultimo pitch).",
        "",
        "Matriz de run expectancy (estado = corredor en 1B, 2B, 3B, outs):",
        "",
        "| on1 | on2 | on3 | outs | RE |",
        "|---|---|---|---|---|",
    ]
    for state_str, re_val in rv["re_table"].items():
        state = eval(state_str)  # tupla (bool,bool,bool,int) serializada como string
        lines.append(f"| {int(state[0])} | {int(state[1])} | {int(state[2])} | {state[3]} | {re_val} |")

    lines += [
        "",
        "## 8. `is_out_in_pa`",
        "",
        "1 si la plate appearance termino en strikeout o en bola en juego que no fue hit "
        "(is_ball_in_play==1 e is_hit==0); 0 si termino en hit, base por bolas o hit by pitch. "
        "Se transmite (broadcast) a todos los pitches de la PA. **Limitacion:** no distingue "
        "embasarse por error de embasarse por fielder's choice; ambos casos quedan como 'out'.",
        "",
        "## Verificacion de fuga de informacion (features vs targets)",
        "",
        f"- Columnas `stuff_feature` que tambien son `target_only` en el diccionario: "
        f"{leak['stuff_feature_cols_overlap_target_only'] or 'ninguna'}",
        f"- Columnas `stuff_feature` que coinciden con los nuevos targets creados aqui: "
        f"{leak['stuff_feature_cols_overlap_new_targets'] or 'ninguna'}",
        f"- Columnas `stuff_feature` que en realidad son datos de resultado/ubicacion "
        f"(PlateLoc*, in/outside_strike_zone, target_only): "
        f"{leak['stuff_feature_cols_using_outcome_data'] or 'ninguna'}",
        "",
        "Correlaciones |r| >= 0.30 entre una stuff_feature numerica y un target binario "
        "(a revisar manualmente, no implican fuga por si solas):",
        "",
    ]
    if leak["suspicious_correlations_ge_0_30"]:
        lines.append("| feature | target | corr |")
        lines.append("|---|---|---|")
        for row in leak["suspicious_correlations_ge_0_30"]:
            lines.append(f"| {row['feature']} | {row['target']} | {row['corr']} |")
    else:
        lines.append("Ninguna encontrada.")

    lines += [
        "",
        "## Distribucion (proporcion de 1s) por altitude_category",
        "",
        report["distribution_by_altitude"].to_markdown(),
        "",
        "## `run_value` (solo pitches terminales de PA) por altitude_category",
        "",
        report["run_value_by_altitude"].to_markdown(),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    logger.info("Cargado %s: %s filas", args.input.name, f"{len(df):,}")

    df_out, report = build_targets(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    logger.info("Guardado %s (%s filas, %s columnas)", args.output, f"{len(df_out):,}", df_out.shape[1])

    report_text = format_report(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    logger.info("Guardado reporte en %s", args.report)


if __name__ == "__main__":
    main()
