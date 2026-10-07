"""Contrafactual de altitud coherente: re-simula la trayectoria completa del pitch.

El contrafactual v1 (src/analysis/altitude_counterfactual.py) solo cambiaba
InducedVertBreak y HorzBreak, dejando fijos el ángulo de llegada (la feature
más importante del modelo), la velocidad en zona, el tiempo de vuelo y el
quiebre total. Eso producía pitches físicamente incoherentes.

Aquí se usa la trayectoria completa de Trackman (modelo de 9 parámetros:
posición, velocidad y aceleración constantes en y=50 pies) y se cambia la
aceleración del aire:

1. La aceleración sin gravedad se descompone en DRAG (componente a lo largo
   de la velocidad) y MAGNUS (componente perpendicular).
2. Cada componente se escala por su ratio de densidad efectiva entre la
   categoría observada y la categoría destino.
3. Se reconstruye el punto y la velocidad de salida (con la aceleración
   observada) y se vuelve a lanzar el pitch con la aceleración contrafactual.
4. Supuesto de comando: el pitcher apunta al MISMO punto del plato (corrige
   la dirección de salida para llegar a la ubicación observada). Así Location
   no cambia y todo el efecto pasa por "stuff". El modo "same_release"
   (misma dirección de salida, la ubicación se mueve) queda disponible.
5. Las features se recalculan como DELTAS sobre lo observado (feature_cf =
   feature_obs + modelo_cf − modelo_obs), así que proyectar a la misma
   categoría devuelve exactamente lo observado.

Ratios de densidad: NO se suponen elevaciones. Se MIDEN con el propio dataset
comparando al mismo pitcher, mismo tipo de pitch y misma temporada entre
categorías de altitud: drag/v² y Magnus/v² son proporcionales a la densidad
del aire. Resultado (ver reports/altitude_counterfactual_v2.md): el Magnus
baja bastante más que el drag, algo que la densidad sola no explica.

Coordenadas Trackman (validadas contra las columnas del dataset, r ≥ 0.997):
x = −PlateLocSide (vista del catcher), y = distancia al plato, z = altura.
HorzBreak = −½·ax·T²·12, HorzApprAngle = −atan(vx/−vy).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRAVITY_FT_S2 = 32.174
Y_FRONT_OF_PLATE = 17 / 12  # pies
FT_S_TO_MPH = 3600 / 5280
MOUND_TO_PLATE_FT = 60.5
REFERENCE_CATEGORY = "No Altitude"
ALTITUDE_CATEGORIES = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
MIN_PITCHES_PER_CELL = 15
MIN_GROUPS_PER_PITCH_TYPE = 50

TRAJECTORY_COLUMNS = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"]
# Features que el contrafactual recalcula (el resto queda igual)
UPDATED_FEATURES = [
    "ZoneSpeed", "ZoneTime", "SpeedDrop", "VertApprAngle", "HorzApprAngle",
    "InducedVertBreak", "HorzBreak", "VertBreak",
]


# ---------------------------------------------------------------------------
# Descomposición drag / Magnus
# ---------------------------------------------------------------------------


def decompose_acceleration(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Aceleración sin gravedad → componente drag (paralela a v) y Magnus (perpendicular)."""
    v = df[["vx0", "vy0", "vz0"]].to_numpy(dtype=float).copy()
    a = df[["ax0", "ay0", "az0"]].to_numpy(dtype=float).copy()
    a[:, 2] += GRAVITY_FT_S2
    speed = np.linalg.norm(v, axis=1)
    v_hat = v / speed[:, None]
    along = (a * v_hat).sum(axis=1)
    drag_vec = along[:, None] * v_hat
    magnus_vec = a - drag_vec
    return {"speed": speed, "drag_vec": drag_vec, "magnus_vec": magnus_vec, "drag": -along,
            "magnus": np.linalg.norm(magnus_vec, axis=1)}


def estimate_density_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios drag y Magnus de cada categoría contra No Altitude, medidos dentro de pitcher.

    Para cada (pitcher, temporada, tipo de pitch) con ≥15 pitches en ambas
    categorías se compara la mediana de drag/v² y Magnus/v². El ratio de la
    categoría es la mediana de esos ratios. Se calcula por tipo de pitch (si
    hay ≥50 grupos) y global; devuelve una fila por (categoría, tipo).
    """
    comp = decompose_acceleration(df)
    d = df[["pitcher_anon_id", "year", "AutoPitchType", "altitude_category"]].copy()
    d["AutoPitchType"] = d["AutoPitchType"].astype("object")
    d["k_drag"] = comp["drag"] / comp["speed"] ** 2
    d["k_magnus"] = comp["magnus"] / comp["speed"] ** 2
    cells = (
        d.groupby(["pitcher_anon_id", "year", "AutoPitchType", "altitude_category"], observed=True)
        .agg(n=("k_drag", "size"), k_drag=("k_drag", "median"), k_magnus=("k_magnus", "median"))
        .reset_index()
    )
    cells = cells.loc[cells["n"] >= MIN_PITCHES_PER_CELL]
    ref = cells.loc[cells["altitude_category"] == REFERENCE_CATEGORY]
    rows = []
    for cat in ALTITUDE_CATEGORIES:
        if cat == REFERENCE_CATEGORY:
            rows.append({"altitude_category": cat, "AutoPitchType": "__all__", "drag_ratio": 1.0, "magnus_ratio": 1.0, "n_groups": None})
            continue
        m = cells.loc[cells["altitude_category"] == cat].merge(
            ref, on=["pitcher_anon_id", "year", "AutoPitchType"], suffixes=("", "_ref")
        )
        m["r_drag"] = m["k_drag"] / m["k_drag_ref"]
        m["r_magnus"] = m["k_magnus"] / m["k_magnus_ref"]
        rows.append({"altitude_category": cat, "AutoPitchType": "__all__", "drag_ratio": float(m["r_drag"].median()),
                     "magnus_ratio": float(m["r_magnus"].median()), "n_groups": len(m)})
        for ptype, g in m.groupby("AutoPitchType"):
            if len(g) >= MIN_GROUPS_PER_PITCH_TYPE:
                rows.append({"altitude_category": cat, "AutoPitchType": ptype, "drag_ratio": float(g["r_drag"].median()),
                             "magnus_ratio": float(g["r_magnus"].median()), "n_groups": len(g)})
    return pd.DataFrame(rows)


def lookup_ratios(ratios: pd.DataFrame, categories: pd.Series, pitch_types: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Ratio (drag, Magnus) por pitch, usando el de su tipo si existe y el global si no."""
    by_key = {(r.altitude_category, r.AutoPitchType): (r.drag_ratio, r.magnus_ratio) for r in ratios.itertuples()}
    keys = zip(categories.astype("object"), pitch_types.astype("object"))
    pairs = np.array([by_key.get(k, by_key[(k[0], "__all__")]) for k in keys], dtype=float)
    if not np.isfinite(pairs).all() or (pairs <= 0).any():
        # Sin esta guardia, un ratio NaN (categoría sin grupos suficientes) hacía que la trayectoria se marcara
        # "no simulable" y el pitch conservara sus features observadas: una proyección que no proyecta, sin aviso.
        bad = sorted({k for k, (d, m) in by_key.items() if not (np.isfinite(d) and np.isfinite(m) and d > 0 and m > 0)})
        raise ValueError(f"Ratios de densidad no válidos para {bad}: faltan grupos dentro de pitcher para estimarlos")
    return pairs[:, 0], pairs[:, 1]


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------


def _release_state(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Posición y velocidad en el punto de salida (y = 60.5 − Extension), con la aceleración observada."""
    ext = df["Extension"].astype(float).fillna(df["Extension"].astype(float).median()).to_numpy()
    y_rel = MOUND_TO_PLATE_FT - ext
    a = 0.5 * df["ay0"].to_numpy(float)
    b = df["vy0"].to_numpy(float)
    c = df["y0"].to_numpy(float) - y_rel
    with np.errstate(invalid="ignore"):
        disc = np.sqrt(b * b - 4 * a * c)
    roots = np.stack([(-b + disc) / (2 * a), (-b - disc) / (2 * a)], axis=1)
    # la raíz negativa más cercana a 0 (antes de pasar por y=50)
    roots = np.where(roots < 0, roots, -np.inf)
    t_rel = roots.max(axis=1)
    t_rel = np.where(np.isfinite(t_rel), t_rel, np.nan)
    p0 = df[["x0", "y0", "z0"]].to_numpy(float)
    v0 = df[["vx0", "vy0", "vz0"]].to_numpy(float)
    a0 = df[["ax0", "ay0", "az0"]].to_numpy(float)
    pos = p0 + v0 * t_rel[:, None] + 0.5 * a0 * t_rel[:, None] ** 2
    vel = v0 + a0 * t_rel[:, None]
    return pos, vel


def _fly(pos_r: np.ndarray, vel_r: np.ndarray, acc: np.ndarray, target_xz: np.ndarray | None) -> dict[str, np.ndarray]:
    """Vuelo desde la salida hasta el frente del plato con aceleración constante.

    Si `target_xz` no es None, se ajustan vx y vz de salida para llegar a ese
    punto (supuesto de comando); vy de salida se conserva.
    """
    a = 0.5 * acc[:, 1]
    b = vel_r[:, 1]
    c = pos_r[:, 1] - Y_FRONT_OF_PLATE
    with np.errstate(invalid="ignore"):
        disc = np.sqrt(b * b - 4 * a * c)
    r1, r2 = (-b + disc) / (2 * a), (-b - disc) / (2 * a)
    T = np.where((r1 > 0) & ((r1 < r2) | (r2 <= 0)), r1, r2)
    vel = vel_r.copy()
    if target_xz is not None:
        vel[:, 0] = (target_xz[:, 0] - pos_r[:, 0] - 0.5 * acc[:, 0] * T**2) / T
        vel[:, 2] = (target_xz[:, 1] - pos_r[:, 2] - 0.5 * acc[:, 2] * T**2) / T
    v_plate = vel + acc * T[:, None]
    plate = pos_r + vel * T[:, None] + 0.5 * acc * T[:, None] ** 2
    return {
        "ZoneTime": T,
        "ZoneSpeed": np.linalg.norm(v_plate, axis=1) * FT_S_TO_MPH,
        "RelSpeedSim": np.linalg.norm(vel, axis=1) * FT_S_TO_MPH,
        "VertApprAngle": np.degrees(np.arctan2(v_plate[:, 2], -v_plate[:, 1])),
        "HorzApprAngle": -np.degrees(np.arctan2(v_plate[:, 0], -v_plate[:, 1])),
        "InducedVertBreak": 0.5 * (acc[:, 2] + GRAVITY_FT_S2) * T**2 * 12,
        "HorzBreak": -0.5 * acc[:, 0] * T**2 * 12,
        "VertBreak": 0.5 * acc[:, 2] * T**2 * 12,
        "PlateLocSide": -plate[:, 0],
        "PlateLocHeight": plate[:, 2],
    }


def project_to_altitude(
    df: pd.DataFrame, target_category: str, ratios: pd.DataFrame, mode: str = "same_target"
) -> pd.DataFrame:
    """Devuelve una copia de `df` con las features de trayectoria re-simuladas en `target_category`.

    Requiere las columnas de TRAJECTORY_COLUMNS, Extension, altitude_category,
    AutoPitchType, PlateLocSide/PlateLocHeight y las de UPDATED_FEATURES.
    Los ratios (drag, Magnus) de origen y destino salen de la tabla por categoría × tipo de pitch.
    """
    drag_obs, mag_obs = lookup_ratios(ratios, df["altitude_category"], df["AutoPitchType"])
    drag_tgt, mag_tgt = lookup_ratios(ratios, pd.Series(target_category, index=df.index), df["AutoPitchType"])
    out = project_with_ratios(df, drag_obs, mag_obs, drag_tgt, mag_tgt, mode)
    out["altitude_category"] = target_category
    return out


def project_with_ratios(
    df: pd.DataFrame, drag_obs: np.ndarray, mag_obs: np.ndarray, drag_tgt: np.ndarray, mag_tgt: np.ndarray,
    mode: str = "same_target",
) -> pd.DataFrame:
    """Núcleo del contrafactual con ratios POR PITCH (permite ratios por estadio anónimo).

    Cada componente de la aceleración sin gravedad se escala por ratio_destino / ratio_origen:
    a' = D·(ρ_D,dest/ρ_D,orig) + M·(ρ_M,dest/ρ_M,orig) − g ẑ. Después se vuelve a lanzar desde el mismo punto
    de salida (al mismo objetivo con `same_target`) y las features se aplican como deltas sobre lo observado.
    """
    if mode not in ("same_target", "same_release"):
        raise ValueError(mode)
    for arr in (drag_obs, mag_obs, drag_tgt, mag_tgt):
        if not np.isfinite(arr).all() or (np.asarray(arr) <= 0).any():
            raise ValueError("Ratios de densidad no válidos (NaN o ≤ 0): la proyección no se puede hacer")
    out = df.copy()
    comp = decompose_acceleration(df)
    acc_obs = df[["ax0", "ay0", "az0"]].to_numpy(float)
    acc_cf = (
        comp["drag_vec"] * (np.asarray(drag_tgt) / np.asarray(drag_obs))[:, None]
        + comp["magnus_vec"] * (np.asarray(mag_tgt) / np.asarray(mag_obs))[:, None]
    )
    acc_cf[:, 2] -= GRAVITY_FT_S2

    pos_r, vel_r = _release_state(df)
    target = None
    if mode == "same_target":
        target = np.stack([-df["PlateLocSide"].to_numpy(float), df["PlateLocHeight"].to_numpy(float)], axis=1)
    sim_obs = _fly(pos_r, vel_r, acc_obs, target)
    sim_cf = _fly(pos_r, vel_r, acc_cf, target)

    # Trayectorias que no se pueden simular (p. ej. Extension físicamente imposible,
    # >10 pies, en ~20 pitches) conservan sus features observadas.
    sim_cols = ["ZoneSpeed", "ZoneTime", "VertApprAngle", "HorzApprAngle", "InducedVertBreak", "HorzBreak", "VertBreak"]
    if mode == "same_release":
        sim_cols += ["PlateLocSide", "PlateLocHeight"]
    valid = np.all([np.isfinite(sim_cf[c]) & np.isfinite(sim_obs[c]) for c in sim_cols], axis=0)
    if (~valid).any():
        logger.info("Trayectorias no simulables (se conservan las features observadas): %d", int((~valid).sum()))
    for col in sim_cols:
        out[col] = df[col].astype(float) + np.where(valid, sim_cf[col] - sim_obs[col], 0.0)
    out["SpeedDrop"] = df["RelSpeed"].astype(float) - out["ZoneSpeed"]
    return out


def synthesize_pitch(
    rel_speed_mph: float, ivb_in: float, hb_in: float, extension_ft: float, rel_height_ft: float, rel_side_ft: float,
    drag_k: float, target_side_ft: float, target_height_ft: float, reference_category: str, pitch_type: str,
    ratios: pd.DataFrame, return_states: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Construye la trayectoria de un pitch hipotético y la vuela en las 3 categorías de altitud.

    Con `return_states=True` devuelve además {categoría: {"pos": salida, "acc": aceleración, "target": objetivo}}
    para dibujar la trayectoria completa (la página web la usa).

    Para el dashboard: el usuario da velocidad, IVB y HB *observados en
    `reference_category`*; aquí se elige la aceleración que reproduce
    exactamente ese movimiento en esa categoría (drag típico del tipo de pitch,
    `drag_k` = drag/v² en ft⁻¹), y después se escalan drag y Magnus con los
    ratios medidos para obtener el mismo pitch en las otras categorías. Todas
    se lanzan desde el mismo punto de salida al mismo objetivo en el plato.
    Devuelve una fila por categoría con las features de trayectoria.
    """
    v0 = rel_speed_mph / FT_S_TO_MPH
    pos_r = np.array([[-rel_side_ft, MOUND_TO_PLATE_FT - extension_ft, rel_height_ft]])
    target = np.array([[-target_side_ft, target_height_ft]])
    drag_acc = drag_k * v0**2

    # Aceleración en la categoría de referencia: Magnus fijado por el IVB/HB pedidos, drag por el tipo de pitch.
    # T depende de ay y ay del drag; dos iteraciones bastan para converger (cambios < 1e-6 s).
    T = (pos_r[0, 1] - Y_FRONT_OF_PLATE) / v0
    for _ in range(3):
        ax = -2 * (hb_in / 12) / T**2
        az = 2 * (ivb_in / 12) / T**2 - GRAVITY_FT_S2
        ay = drag_acc  # el pitch viaja hacia −y; el drag lo frena (+y)
        acc_ref = np.array([[ax, ay, az]])
        vel_r = np.array([[0.0, -np.sqrt(max(v0**2, 1e-9)), 0.0]])
        T = _fly(pos_r, vel_r, acc_ref, target)["ZoneTime"][0]

    sim_ref = _fly(pos_r, vel_r, acc_ref, target)
    vel_dir = np.array([[(target[0, 0] - pos_r[0, 0]) / T, -v0, (target[0, 1] - pos_r[0, 2]) / T]])
    a_ng = acc_ref.copy()
    a_ng[:, 2] += GRAVITY_FT_S2
    v_hat = vel_dir / np.linalg.norm(vel_dir)
    drag_vec = (a_ng * v_hat).sum(axis=1)[:, None] * v_hat
    magnus_vec = a_ng - drag_vec

    ref_drag, ref_mag = lookup_ratios(ratios, pd.Series([reference_category]), pd.Series([pitch_type]))
    rows = []
    states = {}
    for cat in ALTITUDE_CATEGORIES:
        c_drag, c_mag = lookup_ratios(ratios, pd.Series([cat]), pd.Series([pitch_type]))
        acc = drag_vec * (c_drag / ref_drag)[:, None] + magnus_vec * (c_mag / ref_mag)[:, None]
        acc[:, 2] -= GRAVITY_FT_S2
        sim = sim_ref if cat == reference_category else _fly(pos_r, vel_r, acc, target)
        states[cat] = {"pos": pos_r[0].copy(), "acc": (acc_ref if cat == reference_category else acc)[0].copy(),
                       "vy0": float(vel_r[0, 1]), "target": target[0].copy()}
        rows.append({
            "altitude_category": cat,
            **{k: float(sim[k][0]) for k in ["ZoneSpeed", "ZoneTime", "VertApprAngle", "HorzApprAngle",
                                              "InducedVertBreak", "HorzBreak", "VertBreak", "PlateLocSide", "PlateLocHeight"]},
        })
    out = pd.DataFrame(rows)
    out["SpeedDrop"] = rel_speed_mph - out["ZoneSpeed"]
    return (out, states) if return_states else out


def recompute_fastball_diffs(df: pd.DataFrame, fastball_types: frozenset = frozenset({"Four-Seam", "Sinker"})) -> pd.DataFrame:
    """Recalcula ivb/hb_diff_vs_fb con el movimiento ya proyectado (misma definición que src/features/arsenal.py).

    Solo para pitcher-temporadas que ya tenían baseline de recta (las demás quedan nulas, igual que en arsenal.py).
    """
    out = df.copy()
    is_fb = out["AutoPitchType"].astype("object").isin(fastball_types)
    fb = out.loc[is_fb].groupby(["pitcher_anon_id", "year"])[["InducedVertBreak", "HorzBreak"]].mean()
    fb.columns = ["fb_ivb", "fb_hb"]
    merged = out[["pitcher_anon_id", "year"]].merge(fb, left_on=["pitcher_anon_id", "year"], right_index=True, how="left")
    had_baseline = out["ivb_diff_vs_fb"].notna()
    out["ivb_diff_vs_fb"] = np.where(had_baseline, out["InducedVertBreak"] - merged["fb_ivb"].to_numpy(), np.nan)
    out["hb_diff_vs_fb"] = np.where(had_baseline, out["HorzBreak"] - merged["fb_hb"].to_numpy(), np.nan)
    return out
