"""Funciones de puntuación (defaults por tipo de pitch, ratios de densidad, conteos) que usa la página (web/engine.py).

Antes servían al dashboard v3 (archivado en historial/dashboard/app.py). Sin dependencias de Streamlit.

Versión v3 (2026-10-03): usa el índice oficial Pitching+ v3 y el
contrafactual de trayectoria completa. Reemplaza al mecanismo v1 (Magnus
teórico + residual, que solo cambiaba IVB/HB).

Pitch hipotético → 3 categorías de altitud:
1. El usuario da velocidad, IVB, HB, spin, extensión, altura de salida, tipo
   de pitch y manos, y la categoría donde se OBSERVÓ ese movimiento.
2. `src.features.trajectory.synthesize_pitch` construye la trayectoria que
   reproduce exactamente ese movimiento en esa categoría (drag típico del
   tipo) y la vuela en las otras dos escalando drag y Magnus con los ratios
   medidos en los datos. Salen todas las features de trayectoria coherentes
   (velocidad en zona, tiempo de vuelo, VAA, HAA, IVB, HB, quiebre total).
3. Se valúa con los modelos finales de Pitching+ v3 en carreras por pitch,
   con el pitch al centro de la zona y neutral al conteo: se predice en los
   12 conteos (cada uno con sus valores en carreras) y se promedia con la
   frecuencia real de cada conteo en la liga.

Se separa de app.py para poder usarlo en pruebas (tests/test_dashboard_scoring.py).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import altitude_counterfactual_v2 as cf_v2  # noqa: E402
from src.features import trajectory  # noqa: E402
from src.features.physics import SPIN_AXIS_TO_BREAK_OFFSET_DEG  # noqa: E402
from src.models import chain_attribution, run_value_chain, stuff_plus_v2  # noqa: E402
from src.reports import scouting_report_v2  # noqa: E402

OFFICIAL_PATH = ROOT / "data" / "processed" / "pitching_plus_official.parquet"
RATIOS_PATH = ROOT / "data" / "processed" / "altitude_density_ratios.parquet"
EFFECTS_PATH = ROOT / "reports" / "benchmark" / "altitude_counterfactual_v2.json"
DEFAULTS_CACHE = ROOT / "data" / "processed" / "dashboard_pitch_defaults.parquet"
COUNT_SHARES_CACHE = ROOT / "data" / "processed" / "dashboard_count_shares.parquet"

ALTITUDE_CATEGORIES = trajectory.ALTITUDE_CATEGORIES
ALTITUDE_LABELS_ES = {"No Altitude": "Nivel del mar", "Medium Altitude": "Altitud media", "Extreme Altitude": "Altitud extrema"}
PITCH_TYPES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
FASTBALL_TYPES = {"Four-Seam", "Sinker"}
PITCHER_THROWS_OPTIONS = ["Right", "Left"]
BATTER_SIDE_OPTIONS = ["Right", "Left"]
STARTER_PITCHES, RELIEVER_PITCHES = 2750, 900

# Supuestos de lo que el formulario no pide (visibles en la app)
TARGET_PLATE_HEIGHT = 2.5
TARGET_PLATE_SIDE = 0.0
DEFAULT_COLUMNS = ["RelSide", "k_drag", "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
                   "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight"]


# ---------------------------------------------------------------------------
# Carga de recursos (app.py las envuelve con @st.cache_resource)
# ---------------------------------------------------------------------------


def load_artifact() -> dict:
    with open(run_value_chain.FINAL_MODEL_PATHS["pitching"], "rb") as fh:
        return pickle.load(fh)


def load_ratios() -> pd.DataFrame:
    return pd.read_parquet(RATIOS_PATH)


def load_official() -> pd.DataFrame:
    return pd.read_parquet(OFFICIAL_PATH)


def load_altitude_effects() -> dict:
    return json.loads(EFFECTS_PATH.read_text(encoding="utf-8"))


def load_scouting_resources() -> dict:
    return scouting_report_v2.load_resources()


def compute_pitch_defaults(refresh: bool = False) -> pd.DataFrame:
    """Medianas por (tipo, mano, categoría): punto de salida lateral, drag típico, diferencias vs recta y el pitch típico."""
    if DEFAULTS_CACHE.exists() and not refresh:
        return pd.read_parquet(DEFAULTS_CACHE)
    df = cf_v2.load_data()
    comp = trajectory.decompose_acceleration(df)
    df["k_drag"] = comp["drag"] / comp["speed"] ** 2
    df["hand"] = stuff_plus_v2.pitcher_hand(df)
    df["AutoPitchType"] = df["AutoPitchType"].astype("object")
    out = (
        df.loc[df["AutoPitchType"].isin(PITCH_TYPES)]
        .groupby(["AutoPitchType", "hand", "altitude_category"])[DEFAULT_COLUMNS].median()
        .reset_index()
    )
    out.to_parquet(DEFAULTS_CACHE, index=False)
    shares = df.groupby(["Balls", "Strikes"]).size().rename("share").reset_index()
    shares["share"] /= shares["share"].sum()
    shares.to_parquet(COUNT_SHARES_CACHE, index=False)
    return out


def load_count_shares() -> pd.DataFrame:
    if not COUNT_SHARES_CACHE.exists():
        compute_pitch_defaults(refresh=True)
    return pd.read_parquet(COUNT_SHARES_CACHE)


def _defaults_for(defaults: pd.DataFrame, pitch_type: str, hand: str, category: str) -> pd.Series:
    row = defaults.loc[(defaults["AutoPitchType"] == pitch_type) & (defaults["hand"] == hand) & (defaults["altitude_category"] == category)]
    return row.iloc[0]


# ---------------------------------------------------------------------------
# Pitch hipotético en las 3 categorías
# ---------------------------------------------------------------------------


def estimate_spin_axis(ivb: float, hb: float) -> float:
    """SpinAxis a partir de la dirección del movimiento (offset calibrado en src/features/physics.py)."""
    return (np.degrees(np.arctan2(hb, ivb)) % 360 - SPIN_AXIS_TO_BREAK_OFFSET_DEG) % 360


def _feature_rows(inputs: dict, d: pd.Series, traj: pd.DataFrame) -> pd.DataFrame:
    """Filas con todas las columnas que espera build_v2_features (las no usadas por v2d van en NaN)."""
    pt = inputs["AutoPitchType"]
    is_fb = pt in FASTBALL_TYPES
    rows = traj.copy()
    rows["RelSpeed"] = inputs["RelSpeed"]
    rows["SpinRate"] = inputs["SpinRate"]
    rows["SpinAxis"] = estimate_spin_axis(inputs["InducedVertBreak"], inputs["HorzBreak"])
    rows["RelHeight"] = inputs["RelHeight"]
    rows["RelSide"] = d["RelSide"]
    rows["Extension"] = inputs["Extension"]
    rows["velo_diff_vs_fb"] = 0.0 if is_fb else d["velo_diff_vs_fb"]
    # Diferencias de movimiento vs recta: la recta del pitcher también cambia con la altitud; para un pitch que no es recta
    # se usa la diferencia típica del tipo (aprox.), para una recta es 0.
    rows["ivb_diff_vs_fb"] = 0.0 if is_fb else d["ivb_diff_vs_fb"]
    rows["hb_diff_vs_fb"] = 0.0 if is_fb else d["hb_diff_vs_fb"]
    for col in ["EffectiveVelo", "VertRelAngle", "HorzRelAngle", "altitude_level", "magnus_residual_vert", "magnus_residual_horz"]:
        rows[col] = np.nan
    rows["PitcherThrows"] = inputs["PitcherThrows"]
    rows["BatterSide"] = inputs["BatterSide"]
    rows["AutoPitchType"] = pt
    return rows


def _score(inputs: dict, reference_category: str, artifact: dict, ratios: pd.DataFrame, defaults: pd.DataFrame) -> pd.DataFrame:
    d = _defaults_for(defaults, inputs["AutoPitchType"], inputs["PitcherThrows"], reference_category)
    traj = trajectory.synthesize_pitch(
        rel_speed_mph=inputs["RelSpeed"], ivb_in=inputs["InducedVertBreak"], hb_in=inputs["HorzBreak"],
        extension_ft=inputs["Extension"], rel_height_ft=inputs["RelHeight"], rel_side_ft=float(d["RelSide"]),
        drag_k=float(d["k_drag"]), target_side_ft=TARGET_PLATE_SIDE, target_height_ft=TARGET_PLATE_HEIGHT,
        reference_category=reference_category, pitch_type=inputs["AutoPitchType"], ratios=ratios,
    )
    rows = _feature_rows(inputs, d, traj)
    shares = load_count_shares()
    # Cada categoría × 12 conteos; luego promedio ponderado por la frecuencia del conteo
    grid = rows.merge(shares, how="cross")
    feats, _ = stuff_plus_v2.build_v2_features(grid, location_coefs=artifact["location_coefs"])
    pred = run_value_chain.predict_expected_run_value(artifact, feats, count_neutral=False)
    value_cols = ["xrv", "whiff"] + [f"comp_{g}" for g in run_value_chain.COMPONENT_GROUPS]
    weighted = pred[value_cols].mul(grid["share"].to_numpy(), axis=0)
    weighted["altitude_category"] = grid["altitude_category"].to_numpy()
    agg = weighted.groupby("altitude_category", sort=False)[value_cols].sum()
    out = traj[["altitude_category", "InducedVertBreak", "HorzBreak", "ZoneSpeed", "VertApprAngle", "ZoneTime"]].copy()
    agg = agg.loc[out["altitude_category"]]
    out["runs_per_100"] = 100 * agg["xrv"].to_numpy()
    for g in run_value_chain.COMPONENT_GROUPS:
        out[f"comp_{g}_per_100"] = 100 * agg[f"comp_{g}"].to_numpy()
    out["p_whiff"] = agg["whiff"].to_numpy()
    # Atribución de Shapley del cambio contra No Altitude (misma definición que el contrafactual oficial):
    # "antes del contacto" = swing + take + whiff + foul; "bola en juego" = resultado del contacto (+ parque).
    subs = chain_attribution.submodel_probabilities(artifact, feats)
    D = cf_v2.count_delta_matrix(artifact, grid)
    cats = grid["altitude_category"].to_numpy()
    w = grid["share"].to_numpy()
    base = np.flatnonzero(cats == "No Altitude")
    for col in ("antes_contacto_vs_no_per_100", "bola_en_juego_vs_no_per_100"):
        out[col] = 0.0
    for i, cat in enumerate(out["altitude_category"]):
        idx = np.flatnonzero(cats == cat)
        sh = chain_attribution.shapley({k: v[base] for k, v in subs.items()}, {k: v[idx] for k, v in subs.items()}, D[idx])
        out.loc[out.index[i], "antes_contacto_vs_no_per_100"] = 100 * float((sh[chain_attribution.PRE_CONTACT].sum(axis=1) * w[idx]).sum())
        out.loc[out.index[i], "bola_en_juego_vs_no_per_100"] = 100 * float((sh["bip"] * w[idx]).sum())
    return out


def compute_altitude_scenarios(
    inputs: dict, reference_category: str, artifact: dict, ratios: pd.DataFrame, defaults: pd.DataFrame
) -> pd.DataFrame:
    """Valor del pitch (carreras/100, más bajo = mejor) en las 3 categorías y contra el pitch típico de su tipo."""
    pitch = _score(inputs, reference_category, artifact, ratios, defaults)
    d = _defaults_for(defaults, inputs["AutoPitchType"], inputs["PitcherThrows"], reference_category)
    typical_inputs = {**inputs, **{k: float(d[k]) for k in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight"]}}
    typical = _score(typical_inputs, reference_category, artifact, ratios, defaults)
    pitch["typical_runs_per_100"] = typical["runs_per_100"].to_numpy()
    # Positivo = el pitch evita más carreras que el típico de su tipo en el mismo entorno
    pitch["runs_saved_vs_typical_per_100"] = pitch["typical_runs_per_100"] - pitch["runs_per_100"]
    return pitch


def altitude_note(pitch_type: str, effects: dict) -> str:
    """Contexto del tipo de pitch a partir de reports/altitude_counterfactual_v2.md."""
    total = effects["effect_runs_per_pitch"].get(pitch_type)
    pre = effects["antes_del_contacto_runs_per_pitch"].get(pitch_type)
    if total is None or pre is None:
        return ""
    ranking = sorted(effects["antes_del_contacto_runs_per_pitch"].items(), key=lambda kv: -kv[1]["point"])
    pos = [k for k, _ in ranking].index(pitch_type) + 1
    q = pre["point"] * 100
    return (
        f"En la liga, un **{pitch_type}** promedio permite **{q:+.2f} carreras por cada 100 pitches** más en altitud extrema que "
        f"a nivel del mar antes del contacto (menos whiffs, más pelotas en juego; ≈{q * STARTER_PITCHES / 100:+.1f} por temporada "
        f"de abridor); es el #{pos} más afectado de {len(ranking)} tipos. Con el resultado de la bola en juego, que mezcla calidad "
        f"del contacto y parque, el efecto total es {total['point'] * 100:+.2f} carreras/100 (`reports/altitude_counterfactual_v2.md`). "
        "Altitud extrema = los estadios anónimos de la categoría Extreme del dataset."
    )


def load_scatter_sample(resources: dict, official: pd.DataFrame, n: int = 4000, seed: int = 42) -> pd.DataFrame:
    p = resources["pitches"]
    merged = p.merge(official[["pitcher_anon_id", "year", "pitching_plus", "n_pitches"]], on=["pitcher_anon_id", "year"])
    merged = merged.loc[(merged["n_pitches"] >= 100) & merged["AutoPitchType"].isin(PITCH_TYPES)]
    sample = merged.sample(n=min(n, len(merged)), random_state=seed)
    return sample[["InducedVertBreak", "HorzBreak", "AutoPitchType", "pitching_plus"]].reset_index(drop=True)
