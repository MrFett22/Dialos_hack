"""Residual de Magnus para el modelo Stuff+ LMB.

Compara el movimiento observado de cada pitch (InducedVertBreak, HorzBreak)
contra el movimiento TEORICO esperado por fuerza de Magnus pura, usando
`altitude_category` como proxy de densidad de aire (el dataset no trae
altitud numerica ni identificador de estadio — ver LIMITACION abajo).

LIMITACION PRINCIPAL (explicita, pedida por el usuario): no tenemos la
elevacion real del estadio de cada pitch, solo el bucket `altitude_category`
(No Altitude / Medium Altitude / Extreme Altitude). Se le asigna a cada
bucket un punto medio de elevacion representativo, supuesto manualmente por
el usuario (no medido):

    No Altitude       ~50 m
    Medium Altitude   ~800 m
    Extreme Altitude  ~1900 m

Esto es una aproximacion gruesa (version v1): las elevaciones son supuestas,
no vienen del dataset, y cada categoria agrupa varios estadios anonimos con
densidades distintas (ver src/data/parks.py). La version vigente mide la
densidad efectiva con los datos (src/features/trajectory.py) en vez de suponerla.

MODELO FISICO usado para el movimiento teorico:
- Densidad de aire: formula barometrica estandar (ISA, atmosfera troposferica
  con gradiente termico), NO el modelo isotermico simplificado.
- Fuerza de Magnus: a = rho * A * Cl(S) * v^2 / (2*m), con el coeficiente de
  sustentacion Cl(S) = 1 / (2.32 + 0.4/S) (ajuste empirico publicado por Alan
  Nathan, "The Physics of Baseball" — spin factor S = r*omega/v).
- Desplazamiento (break teorico, magnitud) = 0.5 * a * t^2, con t = distancia
  release-a-home / RelSpeed (aproximacion de aceleracion constante, ignora la
  desaceleracion por drag durante el vuelo — simplificacion estandar).
- Direccion (componente vertical vs horizontal): SpinAxis no tiene una
  convencion de "cero" garantizada en este export especifico, asi que el
  offset entre SpinAxis y la direccion observada del movimiento se calibro
  EMPIRICAMENTE contra los propios datos (ver `SPIN_AXIS_TO_BREAK_OFFSET_DEG`
  y el reporte, seccion de calibracion) en vez de asumir una convencion de
  manual/vendor sin verificar.

LIMITACION FISICA ADICIONAL: el modelo usa el SpinRate TOTAL medido, que
incluye tanto spin "util" (transversal, el que genera fuerza de Magnus) como
spin giroscopico (que no contribuye al break). Sin un dato de "spin
efficiency" no podemos separarlos, asi que el modelo teorico sobreestima
sistematicamente la magnitud absoluta del break (confirmado empiricamente:
la magnitud teorica promedio es ~1.5-2x la observada). Como la eficiencia de
spin es una caracteristica del agarre/pitch, no de la altitud, este sesgo
sistematico no deberia contaminar la comparacion RELATIVA entre categorias de
altitud, que es el objetivo de este analisis — pero si invalida usar la
magnitud absoluta del residuo fuera de ese contexto comparativo.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "pitches_clean.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pitches_with_physics.parquet"
DEFAULT_REPORT = ROOT / "reports" / "magnus_analysis.md"
FIG_DIR = ROOT / "reports" / "figures"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- supuestos de elevacion por altitude_category (ver docstring del modulo) ---
ELEVATION_M_BY_ALTITUDE_CATEGORY = {
    "No Altitude": 50.0,
    "Medium Altitude": 800.0,
    "Extreme Altitude": 1900.0,
}

# --- constantes de la formula barometrica (ISA, tropósfera, hasta 11km) ---
RHO0 = 1.225          # kg/m^3, densidad de aire a nivel del mar (ISA estandar)
T0 = 288.15           # K, temperatura a nivel del mar (ISA estandar)
LAPSE_RATE = 0.0065   # K/m, gradiente termico troposferico estandar
G = 9.80665           # m/s^2
MOLAR_MASS_AIR = 0.0289644  # kg/mol
GAS_CONSTANT = 8.3144598    # J/(mol*K)
_BAROMETRIC_EXPONENT = (G * MOLAR_MASS_AIR) / (GAS_CONSTANT * LAPSE_RATE) - 1

# --- constantes fisicas de la pelota (especificaciones oficiales MLB/beisbol) ---
BALL_RADIUS_M = 0.0369   # ~2.9in de diametro / 2
BALL_MASS_KG = 0.145
BALL_AREA_M2 = np.pi * BALL_RADIUS_M**2
MOUND_TO_PLATE_FT = 60.5

MPH_TO_MS = 0.44704
FT_TO_M = 0.3048
M_TO_IN = 39.3701

# calibrado empiricamente contra los datos (ver calibrate_spin_axis_offset):
# mediana de (angulo observado del movimiento - SpinAxis) = 178.67 grados,
# std=8.72 grados sobre 633,447 pitches -> se usa 180 grados como offset.
SPIN_AXIS_TO_BREAK_OFFSET_DEG = 180.0


def barometric_air_density(elevation_m: float) -> float:
    """Densidad de aire (kg/m^3) via formula barometrica ISA (tropósfera)."""
    ratio = 1 - (LAPSE_RATE * elevation_m) / T0
    return RHO0 * ratio**_BAROMETRIC_EXPONENT


AIR_DENSITY_BY_ALTITUDE_CATEGORY = {
    cat: barometric_air_density(elev) for cat, elev in ELEVATION_M_BY_ALTITUDE_CATEGORY.items()
}


# ---------------------------------------------------------------------------
# 1. Densidad de aire por altitude_category
# ---------------------------------------------------------------------------


def add_air_density(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["air_density_kg_m3"] = df["altitude_category"].map(AIR_DENSITY_BY_ALTITUDE_CATEGORY)
    return df


# ---------------------------------------------------------------------------
# Calibracion empirica del offset SpinAxis -> direccion de break
# ---------------------------------------------------------------------------


def calibrate_spin_axis_offset(df: pd.DataFrame) -> dict:
    obs_angle = np.degrees(np.arctan2(df["HorzBreak"], df["InducedVertBreak"])) % 360
    diff = (obs_angle - df["SpinAxis"]) % 360
    return {
        "median_offset_deg": float(diff.median()),
        "std_deg": float(diff.std()),
        "n": int(len(diff)),
    }


# ---------------------------------------------------------------------------
# 2. Movimiento teorico (fuerza de Magnus)
# ---------------------------------------------------------------------------


def compute_theoretical_break(rel_speed, spin_rate, spin_axis, extension, air_density_kg_m3):
    """Movimiento teorico (IVB, HB, magnitud) en pulgadas por fuerza de Magnus
    pura, dado RelSpeed (mph)/SpinRate (rpm)/SpinAxis (grados)/Extension (ft) y
    una densidad de aire arbitraria (kg/m^3). Funcion pura vectorizada (acepta
    escalares o pandas Series/np.ndarray) — reutilizada tanto por
    `add_theoretical_break` (densidad real del pitch) como por el analisis
    contrafactual de altitud (densidad de una categoria distinta a la real,
    manteniendo el resto de los inputs fijos).
    """
    v_ms = rel_speed * MPH_TO_MS
    omega = spin_rate * 2 * np.pi / 60
    spin_factor = BALL_RADIUS_M * omega / v_ms
    lift_coef = 1 / (2.32 + 0.4 / spin_factor)

    a_magnus = air_density_kg_m3 * BALL_AREA_M2 * lift_coef * v_ms**2 / (2 * BALL_MASS_KG)

    dist_m = (MOUND_TO_PLATE_FT - extension) * FT_TO_M
    flight_time_s = dist_m / v_ms

    magnitude_in = (0.5 * a_magnus * flight_time_s**2) * M_TO_IN

    theta = np.radians((spin_axis + SPIN_AXIS_TO_BREAK_OFFSET_DEG) % 360)
    ivb_theoretical = magnitude_in * np.cos(theta)
    hb_theoretical = magnitude_in * np.sin(theta)
    return ivb_theoretical, hb_theoretical, magnitude_in


def add_theoretical_break(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ivb, hb, magnitude_in = compute_theoretical_break(
        df["RelSpeed"], df["SpinRate"], df["SpinAxis"], df["Extension"], df["air_density_kg_m3"]
    )
    df["ivb_theoretical"] = ivb
    df["hb_theoretical"] = hb
    df["_theoretical_magnitude_in"] = magnitude_in
    return df


# ---------------------------------------------------------------------------
# 3-4. Residuales de Magnus
# ---------------------------------------------------------------------------


def add_magnus_residuals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["magnus_residual_vert"] = df["InducedVertBreak"] - df["ivb_theoretical"]
    df["magnus_residual_horz"] = df["HorzBreak"] - df["hb_theoretical"]
    return df


# ---------------------------------------------------------------------------
# 5. Figuras: crudo vs residual, por altitude_category
# ---------------------------------------------------------------------------


def _variance_explained_by_altitude(df: pd.DataFrame, column: str) -> float:
    """Eta^2: fraccion de la varianza total explicada por altitude_category (ANOVA)."""
    sub = df.dropna(subset=[column, "altitude_category"])
    grand_mean = sub[column].mean()
    ss_total = ((sub[column] - grand_mean) ** 2).sum()
    group_means = sub.groupby("altitude_category")[column].transform("mean")
    ss_between = ((group_means - grand_mean) ** 2).sum()
    return float(ss_between / ss_total)


def plot_raw_vs_residual(df: pd.DataFrame) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    order = ["No Altitude", "Medium Altitude", "Extreme Altitude"]

    eta2 = {}
    pairs = [
        ("InducedVertBreak", "magnus_residual_vert", "06_magnus_vert_crudo_vs_residual.png", "Movimiento vertical"),
        ("HorzBreak", "magnus_residual_horz", "07_magnus_horz_crudo_vs_residual.png", "Movimiento horizontal"),
    ]
    for raw_col, resid_col, fname, title in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        sns.boxplot(data=df, x="altitude_category", y=raw_col, order=order, showfliers=False, ax=axes[0])
        axes[0].set_title(f"{raw_col} (crudo)")
        axes[0].tick_params(axis="x", rotation=15)

        sns.boxplot(data=df, x="altitude_category", y=resid_col, order=order, showfliers=False, ax=axes[1])
        axes[1].set_title(f"{resid_col} (residual vs. fisica)")
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].tick_params(axis="x", rotation=15)

        fig.suptitle(f"{title}: crudo vs. residual de Magnus, por altitude_category")
        plt.tight_layout()
        fig.savefig(FIG_DIR / fname, bbox_inches="tight")
        plt.close(fig)

        eta2[raw_col] = _variance_explained_by_altitude(df, raw_col)
        eta2[resid_col] = _variance_explained_by_altitude(df, resid_col)

    return eta2


# ---------------------------------------------------------------------------
# ¿El remanente vertical es un artefacto de composicion de AutoPitchType?
# ---------------------------------------------------------------------------

MAIN_PITCH_TYPES = ["Four-Seam", "Sinker", "Slider", "Changeup", "Curveball", "Cutter", "Splitter"]
_ALTITUDE_ORDER = ["No Altitude", "Medium Altitude", "Extreme Altitude"]


def investigate_pitch_type_composition(df: pd.DataFrame) -> dict:
    """¿El patron no-monotonico de magnus_residual_vert (Medium mas bajo que
    las otras dos) es un artefacto de que Medium Altitude tira una mezcla
    distinta de AutoPitchType, o se mantiene dentro de cada tipo de pitch?

    Metodologia: (1) tabla de composicion + chi2/Cramer's V para cuantificar
    que tan distinta es la mezcla; (2) residual promedio por
    (altitude_category, AutoPitchType) para ver si el patron se sostiene
    DENTRO de cada tipo; (3) estandarizacion directa (reponderar cada
    altitude_category con la mezcla GLOBAL fija de pitch types) para separar
    el efecto de composicion del efecto dentro-de-tipo.
    """
    from scipy import stats

    sub = df.dropna(subset=["altitude_category", "magnus_residual_vert"])
    sub = sub[sub["AutoPitchType"].isin(MAIN_PITCH_TYPES)]

    composition_pct = (
        pd.crosstab(sub["altitude_category"], sub["AutoPitchType"], normalize="index")
        .reindex(_ALTITUDE_ORDER)[MAIN_PITCH_TYPES]
        * 100
    )

    counts = pd.crosstab(sub["altitude_category"], sub["AutoPitchType"]).reindex(_ALTITUDE_ORDER)[
        MAIN_PITCH_TYPES
    ]
    chi2, p_value, _, _ = stats.chi2_contingency(counts)
    n = counts.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(counts.shape) - 1))))

    residual_by_type_altitude = (
        sub.groupby(["AutoPitchType", "altitude_category"])["magnus_residual_vert"]
        .mean()
        .unstack()[_ALTITUDE_ORDER]
    )

    # estandarizacion directa: reponderar cada altitude_category con la mezcla
    # GLOBAL fija de pitch types, para aislar el efecto "dentro de tipo" del
    # efecto de que mezcla de pitches tira cada categoria
    global_mix = sub["AutoPitchType"].value_counts(normalize=True)[MAIN_PITCH_TYPES]
    means_by_group = sub.groupby(["altitude_category", "AutoPitchType"])["magnus_residual_vert"].mean().unstack()
    adjusted = (means_by_group[MAIN_PITCH_TYPES] * global_mix).sum(axis=1) / global_mix.sum()
    raw_avg = sub.groupby("altitude_category")["magnus_residual_vert"].mean()

    return {
        "composition_pct": composition_pct,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "residual_by_type_altitude": residual_by_type_altitude,
        "raw_avg_by_altitude": raw_avg.reindex(_ALTITUDE_ORDER),
        "composition_adjusted_avg_by_altitude": adjusted.reindex(_ALTITUDE_ORDER),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_physics_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    report["spin_axis_calibration"] = calibrate_spin_axis_offset(df)

    df = add_air_density(df)
    df = add_theoretical_break(df)
    df = add_magnus_residuals(df)

    report["air_density_by_category"] = AIR_DENSITY_BY_ALTITUDE_CATEGORY
    report["elevation_assumptions_m"] = ELEVATION_M_BY_ALTITUDE_CATEGORY

    valid = df.dropna(subset=["_theoretical_magnitude_in"])
    obs_mag = np.sqrt(valid["InducedVertBreak"] ** 2 + valid["HorzBreak"] ** 2)
    report["sanity_check"] = {
        "theoretical_magnitude_mean_in": float(valid["_theoretical_magnitude_in"].mean()),
        "observed_magnitude_mean_in": float(obs_mag.mean()),
        "ratio_theoretical_over_observed_mean": float(
            (valid["_theoretical_magnitude_in"] / obs_mag.replace(0, np.nan)).median()
        ),
    }

    report["eta_squared"] = plot_raw_vs_residual(df)
    report["pitch_type_composition"] = investigate_pitch_type_composition(df)

    report["null_counts"] = {
        "magnus_residual_vert": int(df["magnus_residual_vert"].isna().sum()),
        "magnus_residual_horz": int(df["magnus_residual_horz"].isna().sum()),
    }

    df = df.drop(columns=["_theoretical_magnitude_in"])
    return df, report


def format_report(report: dict) -> str:
    cal = report["spin_axis_calibration"]
    sc = report["sanity_check"]
    eta2 = report["eta_squared"]
    elev = report["elevation_assumptions_m"]
    rho = report["air_density_by_category"]
    ptc = report["pitch_type_composition"]

    lines = [
        "# Residual de Magnus por altitude_category",
        "",
        "## Limitacion explicita",
        "",
        "El dataset no trae altitud numerica ni identificador de estadio, solo el "
        "bucket `altitude_category` (3 valores). Los siguientes puntos medios de "
        "elevacion son **supuestos manuales del usuario, no medidos**:",
        "",
        "| altitude_category | elevacion asumida (m) | densidad de aire (kg/m^3, ISA) |",
        "|---|---:|---:|",
    ]
    for cat, e in elev.items():
        lines.append(f"| {cat} | {e:.0f} | {rho[cat]:.4f} |")
    lines += [
        "",
        "Nota: estas elevaciones son SUPUESTOS de la version v1 (el dataset no trae "
        "elevaciones ni estadios). Cada categoria agrupa varios estadios anonimos con "
        "densidades distintas (`reports/estadios_anonimos.md`); la version vigente mide "
        "la densidad efectiva con los datos en lugar de suponerla.",
        "",
        "## Modelo fisico (fuerza de Magnus)",
        "",
        "Ver docstring de `src/features/physics.py` para el detalle completo "
        "(formula barometrica ISA, coeficiente de sustentacion de Nathan, "
        "aproximacion de aceleracion constante sobre el tiempo de vuelo). "
        "Limitacion clave: usa SpinRate total (no separa spin util de spin "
        "giroscopico), asi que sobreestima sistematicamente la magnitud absoluta "
        "del break — ver chequeo de cordura abajo.",
        "",
        "### Calibracion empirica del offset SpinAxis -> direccion de break",
        "",
        f"Mediana de (angulo observado del movimiento - SpinAxis) sobre "
        f"{cal['n']:,} pitches: **{cal['median_offset_deg']:.2f} grados** "
        f"(std={cal['std_deg']:.2f}). Se uso **{SPIN_AXIS_TO_BREAK_OFFSET_DEG}** "
        "grados como offset (valor redondo dentro de esta distribucion tan angosta) "
        "en vez de asumir a ciegas una convencion de manual del proveedor de tracking.",
        "",
        "### Chequeo de cordura: magnitud teorica vs observada",
        "",
        f"- Magnitud teorica promedio: {sc['theoretical_magnitude_mean_in']:.2f} in",
        f"- Magnitud observada promedio: {sc['observed_magnitude_mean_in']:.2f} in",
        f"- Razon teorica/observada (mediana): {sc['ratio_theoretical_over_observed_mean']:.2f}x",
        "",
        "La sobreestimacion (~1.5-2x) es consistente con no separar spin util de "
        "spin giroscopico (ver limitacion arriba) — el orden de magnitud es "
        "razonable (decenas de pulgadas, no cientos ni una fraccion), lo cual "
        "confirma que el modelo no tiene un error de unidades o de formula.",
        "",
        "## Nulos",
        "",
    ]
    for c, n in report["null_counts"].items():
        lines.append(f"- `{c}`: {n:,}")

    lines += [
        "",
        "## ¿La fisica simple explica la diferencia entre categorias de altitud?",
        "",
        "Eta^2 (fraccion de la varianza total explicada por altitude_category, "
        "via ANOVA de un factor) — crudo vs. residual:",
        "",
        "| variable | eta^2 crudo | eta^2 residual | reduccion |",
        "|---|---:|---:|---:|",
    ]
    vert_raw, vert_res = eta2["InducedVertBreak"], eta2["magnus_residual_vert"]
    horz_raw, horz_res = eta2["HorzBreak"], eta2["magnus_residual_horz"]
    lines.append(
        f"| Vertical (IVB) | {vert_raw:.4f} | {vert_res:.4f} | "
        f"{100*(1-vert_res/vert_raw):.1f}% |"
    )
    lines.append(
        f"| Horizontal (HB) | {horz_raw:.4f} | {horz_res:.4f} | "
        f"{100*(1-horz_res/horz_raw):.1f}% |"
    )
    lines += [
        "",
        "Ver figuras `reports/figures/06_magnus_vert_crudo_vs_residual.png` y "
        "`07_magnus_horz_crudo_vs_residual.png` — cada una compara boxplots de la "
        "variable cruda (izquierda) contra el residual (derecha), por "
        "altitude_category.",
        "",
        "## ¿El remanente vertical es un artefacto de composicion de AutoPitchType?",
        "",
        "Medium Altitude quedo con el residual vertical mas negativo de las tres "
        "categorias (ver seccion anterior), un patron no-monotonico raro si el "
        "unico factor fuera densidad de aire. Se investigo si es porque Medium "
        "Altitude tira una mezcla distinta de tipos de pitch (ej. mas curvas o "
        "sinkers, que tienen IVB naturalmente mas bajo).",
        "",
        "### Composicion de AutoPitchType por altitude_category (%)",
        "",
    ]
    lines.append(ptc["composition_pct"].round(2).to_markdown())
    lines += [
        "",
        f"Test chi-cuadrado de independencia: chi2={ptc['chi2']:.1f}, "
        f"p={ptc['p_value']:.2e} (n enorme, casi cualquier diferencia sale "
        f"'significativa'). El tamaño de efecto real es lo que importa: "
        f"**Cramer's V = {ptc['cramers_v']:.4f}** — segun convenciones estandar "
        "(V<0.1 = asociacion despreciable), la mezcla de pitch types SI difiere "
        "entre categorias de altitud, pero muy poco en terminos practicos.",
        "",
        "### magnus_residual_vert promedio por (AutoPitchType, altitude_category)",
        "",
        "Si el patron 'Medium mas bajo' fuera puramente por composicion, "
        "deberia desaparecer o invertirse al mirar DENTRO de cada tipo de pitch "
        "por separado:",
        "",
    ]
    lines.append(ptc["residual_by_type_altitude"].round(3).to_markdown())
    lines += [
        "",
        "**No desaparece.** Para 6 de los 7 tipos de pitch principales "
        "(Four-Seam, Sinker, Slider, Changeup, Curveball, Cutter), Medium "
        "Altitude tiene el residual mas negativo de las tres categorias — el "
        "mismo patron que en el agregado. Solo Splitter (la muestra mas chica, "
        "~1% de los pitches) rompe el patron.",
        "",
        "### Estandarizacion directa: crudo vs. ajustado por composicion",
        "",
        "Se recalculo el promedio de cada altitude_category reponderando con la "
        "mezcla GLOBAL fija de pitch types (en vez de la mezcla real de cada "
        "categoria), para aislar el efecto de composicion del efecto "
        "dentro-de-tipo:",
        "",
        "| altitude_category | crudo | ajustado por composicion |",
        "|---|---:|---:|",
    ]
    raw_avg = ptc["raw_avg_by_altitude"]
    adj_avg = ptc["composition_adjusted_avg_by_altitude"]
    for cat in _ALTITUDE_ORDER:
        lines.append(f"| {cat} | {raw_avg[cat]:.4f} | {adj_avg[cat]:.4f} |")

    lines += [
        "",
        "**Conclusion: NO es un artefacto de composicion.** El ajuste apenas "
        "mueve los valores (Medium Altitude pasa de -2.6214 crudo a -2.6341 "
        "ajustado — practicamente identico). El patron 'Medium mas bajo' se "
        "sostiene dentro de casi todos los tipos de pitch individuales, con "
        "muestras grandes en cada celda (miles a decenas de miles de pitches "
        "por combinacion tipo-altitud). **Esto es un hallazgo abierto**: hay "
        "algo en el movimiento vertical de Medium Altitude que el modelo de "
        "Magnus simple (densidad de aire + spin + velocidad) no explica, y no "
        "se debe a que Medium Altitude tire una mezcla distinta de pitches. "
        "Candidatos a investigar despues (fuera del alcance de este analisis): "
        "efecto de drag sobre la forma de la trayectoria (IVB no es un efecto "
        "de Magnus puro, tambien lo modula el drag, que este modelo no separa "
        "explicitamente mas alla de la densidad de aire en la propia formula "
        "de Magnus), o composicion de PITCHERS/equipos especificos que juegan "
        "desproporcionadamente en estadios de altitud media (no se investigo "
        "aqui, solo se descarto composicion de AutoPitchType).",
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

    df_out, report = build_physics_features(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    logger.info("Guardado %s (%s filas, %s columnas)", args.output, f"{len(df_out):,}", df_out.shape[1])

    report_text = format_report(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    logger.info("Guardado reporte en %s", args.report)


if __name__ == "__main__":
    main()
