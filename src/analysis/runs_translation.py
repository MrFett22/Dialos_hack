"""Traduce los deltas de Pitching+ por altitud (reports/altitude_counterfactual.md)
a un estimado de impacto en carreras por temporada, por rol (abridor/relevista).

Metodologia:
1. Relacion Pitching+ <-> run_value: regresion lineal ponderada (por n_pitches)
   de `run_value` (promedio OOF por pitcher-temporada, en
   data/processed/pitching_plus_scores_oof.parquet — carreras esperadas por
   pitch, prediccion del modelo, mayor = peor para el pitcher) sobre
   `pitching_plus` (mismo archivo), restringido a pitcher-temporadas con
   n_pitches>=100 (mismo filtro de `src/models/pitching_plus_score.py`, para
   no dejar que pitcher-temporadas de muestra minuscula distorsionen la
   pendiente). La pendiente da runs esperados por pitch por cada punto de
   Pitching+.
2. delta_pitching_plus por AutoPitchType: recuperado de
   data/processed/altitude_counterfactual_scores.parquet (el mismo dato ya
   usado para reports/altitude_counterfactual.md seccion 1 — no se recalcula
   el contrafactual, solo se re-agrega).
3. delta_runs_per_pitch = pendiente * delta_pitching_plus. Se multiplica por 2
   volumenes tipicos de pitches/temporada (abridor: 2,500-3,000; relevista:
   800-1,000) para traducir a carreras por temporada.

Signo: run_value mas alto = mas carreras permitidas = peor para el pitcher.
delta_runs_per_pitch positivo = el pitcher permite MAS carreras en altitud
(pierde valor); negativo = permite MENOS carreras (gana valor) — este es el
caso de Curveball.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.altitude_counterfactual import (
    compute_reference_z_stats,
    decompose_delta_by_target,
    load_data as load_full_valid_data,
    load_models,
)
from src.models.pitching_plus_score import MIN_PITCHES_FOR_RANKING

ROOT = Path(__file__).resolve().parents[2]
SCORES_OOF_PATH = ROOT / "data" / "processed" / "pitching_plus_scores_oof.parquet"
COUNTERFACTUAL_SCORES_PATH = ROOT / "data" / "processed" / "altitude_counterfactual_scores.parquet"
REPORT_PATH = ROOT / "reports" / "runs_translation.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MIN_N_PITCHTYPE = 100  # mismo umbral de confiabilidad usado en altitude_counterfactual.py

PITCH_VOLUME_BY_ROLE = {
    "Abridor": {"low": 2500, "high": 3000, "mid": 2750},
    "Relevista": {"low": 800, "high": 1000, "mid": 900},
}


# ---------------------------------------------------------------------------
# 1. Pendiente Pitching+ -> run_value (regresion lineal ponderada por n_pitches)
# ---------------------------------------------------------------------------


def weighted_linreg(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    cov = np.sum(w * (x - xbar) * (y - ybar))
    var_x = np.sum(w * (x - xbar) ** 2)
    slope = cov / var_x
    intercept = ybar - slope * xbar

    y_pred = intercept + slope * x
    ss_res = np.sum(w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    r2 = 1 - ss_res / ss_tot

    corr = cov / np.sqrt(var_x * np.sum(w * (y - ybar) ** 2))

    return {"slope": slope, "intercept": intercept, "r2": r2, "weighted_corr": corr, "n": len(x)}


def fit_pitching_plus_to_run_value() -> dict:
    df = pd.read_parquet(SCORES_OOF_PATH, columns=["n_pitches", "run_value", "pitching_plus"])
    df_all = df.copy()
    df = df.loc[df["n_pitches"] >= MIN_PITCHES_FOR_RANKING]

    fit = weighted_linreg(
        df["pitching_plus"].to_numpy(dtype=float),
        df["run_value"].to_numpy(dtype=float),
        df["n_pitches"].to_numpy(dtype=float),
    )
    fit["n_total_pitcher_seasons"] = len(df_all)
    fit["n_used_pitcher_seasons"] = len(df)
    fit["min_n_pitches_filter"] = MIN_PITCHES_FOR_RANKING

    # correlacion simple (no ponderada) como referencia/sanity check
    fit["unweighted_corr"] = float(df["pitching_plus"].corr(df["run_value"]))
    return fit


# ---------------------------------------------------------------------------
# 2. delta_pitching_plus por AutoPitchType (re-agregacion del contrafactual)
# ---------------------------------------------------------------------------


def load_delta_by_pitchtype() -> pd.DataFrame:
    df = pd.read_parquet(
        COUNTERFACTUAL_SCORES_PATH, columns=["AutoPitchType", "delta_pitching_plus"]
    )
    agg = df.groupby("AutoPitchType", observed=True)["delta_pitching_plus"].agg(["mean", "count"])
    agg = agg.rename(columns={"mean": "delta_pitching_plus", "count": "n"})
    return agg.loc[agg["n"] >= MIN_N_PITCHTYPE].sort_values("delta_pitching_plus")


# ---------------------------------------------------------------------------
# 2b. Descomposicion de Curveball: el mecanismo de compensacion via is_whiff
# ---------------------------------------------------------------------------


def get_curveball_whiff_compensation() -> dict:
    """IMPORTANTE: el delta_pitching_plus TOTAL de Curveball es NEGATIVO
    (-0.829, ver reports/altitude_counterfactual.md seccion 1) — un pitcher
    curveball-heavy SIGUE perdiendo Pitching+/runs en altitud en el agregado,
    solo que menos que Sinker/Changeup/Cutter. La 'direccion opuesta' real
    esta confinada al componente is_whiff (y en menor medida is_weak_contact)
    DENTRO de la descomposicion de 8 targets (reports/altitude_counterfactual.md
    seccion 4) — ese componente especifico SI es positivo, pero queda
    sobrecompensado por is_out_in_pa/is_called_strike. Esta funcion recupera
    esa contribucion aislada (en puntos de Pitching+) reutilizando
    `decompose_delta_by_target`, sin volver a correr el contrafactual (solo
    recarga los modelos + el dataset completo para recalcular ref_stats, que
    es lo unico que decompose_delta_by_target necesita ademas del parquet ya
    generado)."""
    df_full = load_full_valid_data()
    models = load_models()
    ref_stats = compute_reference_z_stats(models, df_full)

    scores = pd.read_parquet(COUNTERFACTUAL_SCORES_PATH)
    decomp = decompose_delta_by_target(scores, ref_stats, min_n=MIN_N_PITCHTYPE)

    if "Curveball" not in decomp.index:
        raise RuntimeError("Curveball no tiene suficiente muestra en la descomposicion.")
    return decomp.loc["Curveball"].to_dict()


# ---------------------------------------------------------------------------
# 3. Traduccion a carreras por temporada, por rol
# ---------------------------------------------------------------------------


def build_runs_table(by_pt: pd.DataFrame, slope: float) -> pd.DataFrame:
    rows = []
    for pt, r in by_pt.iterrows():
        delta_pp = r["delta_pitching_plus"]
        delta_runs_per_pitch = slope * delta_pp
        row = {"AutoPitchType": pt, "n_muestra": int(r["n"]), "delta_pitching_plus": delta_pp}
        for role, vol in PITCH_VOLUME_BY_ROLE.items():
            row[f"runs_{role}_low"] = delta_runs_per_pitch * vol["low"]
            row[f"runs_{role}_mid"] = delta_runs_per_pitch * vol["mid"]
            row[f"runs_{role}_high"] = delta_runs_per_pitch * vol["high"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("AutoPitchType")


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def format_report(fit: dict, runs_table: pd.DataFrame, curveball_decomp: dict) -> str:
    lines = [
        "# Traduccion de Pitching+ a carreras por temporada — impacto de altitud",
        "",
        "Convierte los deltas de Pitching+ por altitud de "
        "`reports/altitude_counterfactual.md` (seccion 1) a un estimado de "
        "carreras por temporada, para que el hallazgo sea legible sin tener "
        "que interpretar puntos de un indice abstracto. Pensado para usarse "
        "directamente en el entregable de Recomendaciones Operativas.",
        "",
        "## 1. Pendiente Pitching+ -> run_value",
        "",
        "Regresion lineal ponderada por `n_pitches` de `run_value` (carreras "
        "esperadas por pitch, prediccion OOF del modelo — mayor = peor para "
        "el pitcher) sobre `pitching_plus`, usando las "
        f"{fit['n_used_pitcher_seasons']:,} pitcher-temporadas con "
        f"n_pitches>={fit['min_n_pitches_filter']} (de {fit['n_total_pitcher_seasons']:,} "
        "totales — mismo filtro de tamaño minimo que "
        "`reports/pitching_plus_score_oof.md`).",
        "",
        "| estadistico | valor |",
        "|---|---:|",
        f"| pendiente (runs por pitch, por punto de Pitching+) | {fit['slope']:.6f} |",
        f"| intercepto | {fit['intercept']:.6f} |",
        f"| R² (ponderado) | {fit['r2']:.4f} |",
        f"| correlacion ponderada | {fit['weighted_corr']:.4f} |",
        f"| correlacion simple (sin ponderar, referencia) | {fit['unweighted_corr']:.4f} |",
        "",
        f"**Lectura de la pendiente:** cada punto de Pitching+ perdido equivale a "
        f"~{abs(fit['slope']):.5f} carreras esperadas ADICIONALES por pitch "
        "(pendiente negativa: mas Pitching+ -> menos run_value/menos carreras "
        "permitidas, como se espera de un indice donde mas alto es mejor).",
        "",
        "**Nota sobre R²:** run_value es solo 1 de los 8 componentes del score "
        "combinado (peso 20%, ver `reports/pitching_plus_score.md`) — un R² "
        "notablemente menor a 1 es esperado y no invalida la pendiente como "
        "conversion de escala; refleja que Pitching+ tambien captura comando/"
        "engaño y calidad de contacto que no se traducen 1-a-1 en run_value "
        "pitch a pitch, sino en agregado.",
        "",
        "**Nota sobre rango de validez:** los deltas de Pitching+ por altitud "
        "(seccion 2) son pequeños (entre -1.3 y +0.1 puntos) frente al rango "
        "real de pitcher-temporadas usado para ajustar la pendiente (std=6.32, "
        "ver `reports/pitching_plus_score_oof.md`) — la extrapolacion lineal "
        "es local, sobre una porcion muy chica del rango donde la relacion fue "
        "estimada, no hace falta asumir linealidad global.",
        "",
        "## 2. Impacto en carreras por temporada, por tipo de pitch y rol",
        "",
        "Runs por temporada = pendiente x delta_pitching_plus x volumen de "
        "pitches del rol. Volumen tipico: **abridor 2,500-3,000 pitches/temporada** "
        "(punto medio 2,750), **relevista 800-1,000** (punto medio 900). "
        "Convencion de signo: **positivo = mas carreras permitidas (el "
        "pitcher PIERDE valor en altitud)**; **negativo = menos carreras "
        "permitidas (el pitcher GANA valor en altitud)**.",
        "",
        "| AutoPitchType | delta Pitching+ | Abridor (2500-3000) | Relevista (800-1000) |",
        "|---|---:|---|---|",
    ]
    for pt, r in runs_table.iterrows():
        starter_str = f"{r['runs_Abridor_low']:+.2f} a {r['runs_Abridor_high']:+.2f} (≈{r['runs_Abridor_mid']:+.2f})"
        reliever_str = f"{r['runs_Relevista_low']:+.2f} a {r['runs_Relevista_high']:+.2f} (≈{r['runs_Relevista_mid']:+.2f})"
        lines.append(f"| {pt} | {r['delta_pitching_plus']:+.3f} | {starter_str} | {reliever_str} |")

    # --- lectura en lenguaje llano ---
    losers = runs_table.sort_values("delta_pitching_plus")  # los 7 tipos son negativos en el agregado

    lines += ["", "## 3. Lectura en lenguaje llano", ""]

    worst = losers.iloc[0]
    lines.append(
        f"- Un **abridor {losers.index[0].lower()}-heavy** (repertorio dominado por "
        f"{losers.index[0]}) que se muda a jugar en altitud extrema "
        f"pierde aproximadamente **{abs(worst['runs_Abridor_mid']):.1f} carreras por "
        f"temporada** (rango {abs(worst['runs_Abridor_low']):.1f}-{abs(worst['runs_Abridor_high']):.1f}, "
        f"segun volumen de pitches) frente a jugar a nivel del mar — el mayor impacto "
        "de los tipos de pitch analizados."
    )
    lines.append(
        f"- El mismo pitch type, en un **relevista**, representa "
        f"~{abs(worst['runs_Relevista_mid']):.1f} carreras por temporada "
        f"({abs(worst['runs_Relevista_low']):.1f}-{abs(worst['runs_Relevista_high']):.1f}) — "
        "menor en magnitud absoluta solo porque lanza muchos menos pitches en la "
        "temporada, no porque el efecto por pitch sea distinto."
    )

    lines += [
        "",
        "### Caso especial — Curveball: correccion de encuadre",
        "",
        "**El delta_pitching_plus TOTAL de Curveball tambien es NEGATIVO "
        f"({runs_table.loc['Curveball', 'delta_pitching_plus']:+.3f}, ver "
        "`reports/altitude_counterfactual.md` seccion 1) — un abridor "
        "curveball-heavy tambien PIERDE runs en el agregado al jugar en "
        "altitud, no los gana.** Lo que la seccion 1 de ese reporte muestra es "
        "que Curveball pierde MENOS que Sinker/Changeup/Cutter "
        f"({abs(runs_table.loc['Curveball', 'runs_Abridor_mid']):.1f} carreras/temporada "
        f"para un abridor, vs. {abs(worst['runs_Abridor_mid']):.1f} de {losers.index[0]}), "
        "no que gane valor.",
        "",
        "**La direccion opuesta real esta confinada a UN componente dentro de "
        "la descomposicion de 8 targets** (`reports/altitude_counterfactual.md` "
        "seccion 4): `is_whiff` (y, en menor medida, `is_weak_contact`) SI "
        "mejoran para Curveball en altitud — pero ese beneficio queda "
        "sobre-compensado por perdidas en `is_out_in_pa` e `is_called_strike`, "
        "que pesan mas en el score combinado. Traduciendo SOLO ese componente "
        "aislado a carreras (mismo metodo de la seccion 1, aplicado a "
        f"`contrib_is_whiff`={curveball_decomp['contrib_is_whiff']:+.3f} puntos "
        "de Pitching+ en vez de al delta total):",
        "",
    ]
    whiff_contrib = curveball_decomp["contrib_is_whiff"]
    whiff_runs_starter_mid = fit["slope"] * whiff_contrib * PITCH_VOLUME_BY_ROLE["Abridor"]["mid"]
    whiff_runs_starter_low = fit["slope"] * whiff_contrib * PITCH_VOLUME_BY_ROLE["Abridor"]["low"]
    whiff_runs_starter_high = fit["slope"] * whiff_contrib * PITCH_VOLUME_BY_ROLE["Abridor"]["high"]
    lines += [
        f"- El mecanismo especifico de 'mas swing-and-miss con la curva en altitud' "
        f"vale, AISLADO, aproximadamente **{abs(whiff_runs_starter_mid):.2f} carreras "
        f"ganadas por temporada** para un abridor "
        f"({abs(whiff_runs_starter_low):.2f} a {abs(whiff_runs_starter_high):.2f}) — "
        "pero es un efecto pequeño que NO revierte el resultado neto (que sigue "
        f"siendo una perdida de ~{abs(runs_table.loc['Curveball', 'runs_Abridor_mid']):.1f} "
        "carreras/temporada, no una ganancia).",
        f"- Recordar la nota de cautela de `reports/altitude_whiff_stratified.md`: "
        "el patron de is_whiff en Curveball tiene p=0.036 (vs. p<0.001 del efecto "
        "agregado) — es un patron real confirmado por dos metodos independientes, "
        "pero con mayor incertidumbre relativa que el resto de las cifras de este "
        "reporte. No usar esta cifra especifica con la misma confianza que las "
        "demas de la tabla de la seccion 2.",
        "",
        "**Para el entregable de Recomendaciones Operativas: no presentar a "
        "Curveball como un tipo de pitch que 'gana' valor en altitud.** Todos los "
        "7 tipos de pitch analizados pierden Pitching+/runs en altitud extrema; "
        "Curveball, Four-Seam y Splitter son los que MENOS pierden, no los que "
        "ganan.",
    ]

    lines += [
        "",
        "**Limitacion explicita:** estas cifras asumen que el 100% de los pitches "
        "del pitcher son del tipo indicado (el escenario 'heavy' de cada fila) — "
        "un pitcher real mezcla varios tipos, asi que el impacto real sobre un "
        "pitcher especifico es un promedio ponderado por su propio "
        "`pitch_usage_pct` (ver `reports/arsenal_summary.md`), no directamente "
        "uno de estos valores. Estas cifras son el efecto marginal por tipo de "
        "pitch, utiles para comparar arquetipos de repertorio, no una prediccion "
        "puntual para un pitcher individual sin ponderar por su arsenal real.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    fit = fit_pitching_plus_to_run_value()
    logger.info(
        "Pendiente Pitching+ -> run_value: %.6f (R2=%.4f, n=%s)",
        fit["slope"], fit["r2"], f"{fit['n_used_pitcher_seasons']:,}",
    )

    by_pt = load_delta_by_pitchtype()
    logger.info("Tipos de pitch con n>=%d: %s", MIN_N_PITCHTYPE, ", ".join(by_pt.index))

    runs_table = build_runs_table(by_pt, fit["slope"])

    logger.info("Recuperando descomposicion de Curveball (recarga modelos + dataset completo)...")
    curveball_decomp = get_curveball_whiff_compensation()
    logger.info("contrib_is_whiff (Curveball): %+.4f", curveball_decomp["contrib_is_whiff"])

    report_text = format_report(fit, runs_table, curveball_decomp)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)
    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
