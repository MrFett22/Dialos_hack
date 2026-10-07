"""z-test de is_whiff por altitude_category, CONTROLANDO por AutoPitchType.

Contexto: el chequeo de OOD (reports/altitude_counterfactual_ood_check.md)
descarta que el resultado contrafactual (is_whiff sube al mover un pitch a
Extreme Altitude, reports/altitude_counterfactual.md) sea extrapolacion del
modelo — el % de filas fuera de distribucion en el contrafactual es
practicamente identico al % 'natural' que ya existe al evaluar pitchers de
test nunca vistos en train (~8-14% vs ~8.5% de referencia, segun el metodo).

Esto deja una pregunta observacional pendiente: el z-test agregado de
reports/run_value_validation.md (Extreme vs No Altitude, is_whiff -7.7%
relativo, z=-9.52) NO controla por AutoPitchType. Si Extreme Altitude tira
una mezcla de tipos de pitch distinta a la de No Altitude (ej. mas de un tipo
con whiff rate naturalmente bajo), parte de esa brecha podria ser composicion,
no un efecto de altitud dentro de cada tipo. Este script recalcula el z-test
DENTRO de cada AutoPitchType por separado, mas una descomposicion tipo
Oaxaca-Blinder (mismo metodo ya usado en reports/magnus_analysis.md para el
residual de Magnus) para separar el efecto de composicion del efecto
dentro-de-tipo.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
REPORT_PATH = ROOT / "reports" / "altitude_whiff_stratified.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAIN_PITCH_TYPES = ["Four-Seam", "Slider", "Sinker", "Changeup", "Curveball", "Cutter", "Splitter"]
COMPARISON = ("Extreme Altitude", "No Altitude")
MIN_N_PER_GROUP = 200


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    """z-test de dos proporciones (pooled). Devuelve (p1-p2, z, p_value)."""
    from scipy import stats

    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, np.nan, np.nan
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p1 - p2, z, p_value


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(TARGETS_PATH, columns=["is_whiff", "altitude_category", "AutoPitchType"])
    df = df.dropna(subset=["is_whiff", "altitude_category"])
    return df


def by_pitchtype_ztest(df: pd.DataFrame) -> pd.DataFrame:
    cat_hi, cat_lo = COMPARISON
    rows = []
    for pt in MAIN_PITCH_TYPES:
        sub = df.loc[df["AutoPitchType"] == pt]
        g_hi = sub.loc[sub["altitude_category"] == cat_hi, "is_whiff"]
        g_lo = sub.loc[sub["altitude_category"] == cat_lo, "is_whiff"]
        n_hi, n_lo = len(g_hi), len(g_lo)
        if n_hi < MIN_N_PER_GROUP or n_lo < MIN_N_PER_GROUP:
            continue
        x_hi, x_lo = int(g_hi.sum()), int(g_lo.sum())
        diff, z, p = two_proportion_ztest(x_hi, n_hi, x_lo, n_lo)
        rows.append(
            {
                "AutoPitchType": pt,
                f"n_{cat_hi.replace(' ', '_')}": n_hi,
                f"rate_{cat_hi.replace(' ', '_')}": x_hi / n_hi,
                f"n_{cat_lo.replace(' ', '_')}": n_lo,
                f"rate_{cat_lo.replace(' ', '_')}": x_lo / n_lo,
                "diff": diff,
                "diff_pct_relative": 100 * diff / (x_lo / n_lo),
                "z": z,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def cochran_mantel_haenszel(df: pd.DataFrame) -> dict:
    """CMH test: pooled odds-ratio / z a traves de los estratos de
    AutoPitchType, controlando por composicion (equivalente estratificado del
    z-test agregado)."""
    cat_hi, cat_lo = COMPARISON
    # OR pooled (Mantel-Haenszel):
    num, den = 0.0, 0.0
    for pt in MAIN_PITCH_TYPES:
        sub = df.loc[df["AutoPitchType"] == pt]
        g_hi = sub.loc[sub["altitude_category"] == cat_hi, "is_whiff"]
        g_lo = sub.loc[sub["altitude_category"] == cat_lo, "is_whiff"]
        n_hi, n_lo = len(g_hi), len(g_lo)
        if n_hi < MIN_N_PER_GROUP or n_lo < MIN_N_PER_GROUP:
            continue
        a, c = int(g_hi.sum()), int(g_lo.sum())
        b, d = n_hi - a, n_lo - c
        n = n_hi + n_lo
        num += (a * d) / n
        den += (b * c) / n

    or_pooled = num / den if den > 0 else np.nan

    # z de Mantel-Haenszel (version chi2, para significancia, no para el OR)
    sum_a, sum_ea, sum_var = 0.0, 0.0, 0.0
    for pt in MAIN_PITCH_TYPES:
        sub = df.loc[df["AutoPitchType"] == pt]
        g_hi = sub.loc[sub["altitude_category"] == cat_hi, "is_whiff"]
        g_lo = sub.loc[sub["altitude_category"] == cat_lo, "is_whiff"]
        n_hi, n_lo = len(g_hi), len(g_lo)
        if n_hi < MIN_N_PER_GROUP or n_lo < MIN_N_PER_GROUP:
            continue
        a, c = int(g_hi.sum()), int(g_lo.sum())
        n1, n2, n = n_hi, n_lo, n_hi + n_lo
        m1 = a + c  # total eventos
        ea = n1 * m1 / n  # esperado bajo independencia
        va = n1 * n2 * m1 * (n - m1) / (n**2 * (n - 1))
        sum_a += a
        sum_ea += ea
        sum_var += va

    from scipy import stats

    chi2_mh = (abs(sum_a - sum_ea) - 0.5) ** 2 / sum_var if sum_var > 0 else np.nan
    p_mh = 1 - stats.chi2.cdf(chi2_mh, df=1) if not np.isnan(chi2_mh) else np.nan

    return {"or_pooled": or_pooled, "chi2_mh": chi2_mh, "p_value_mh": p_mh, "sum_a": sum_a, "sum_ea": sum_ea}


def composition_adjusted_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Mismo metodo que reports/magnus_analysis.md (investigate_pitch_type_composition):
    reponderar cada altitude_category con la mezcla GLOBAL fija de AutoPitchType,
    para separar el efecto de composicion del efecto dentro-de-tipo."""
    sub = df[df["AutoPitchType"].isin(MAIN_PITCH_TYPES)]
    global_mix = sub["AutoPitchType"].value_counts(normalize=True)[MAIN_PITCH_TYPES]
    rate_by_group = sub.groupby(["altitude_category", "AutoPitchType"])["is_whiff"].mean().unstack()[MAIN_PITCH_TYPES]
    raw_avg = sub.groupby("altitude_category")["is_whiff"].mean()
    adjusted = (rate_by_group * global_mix).sum(axis=1) / global_mix.sum()

    counts = pd.crosstab(sub["altitude_category"], sub["AutoPitchType"], normalize="index")[MAIN_PITCH_TYPES] * 100

    from scipy import stats as scipy_stats

    counts_n = pd.crosstab(sub["altitude_category"], sub["AutoPitchType"])[MAIN_PITCH_TYPES]
    chi2, p_value, _, _ = scipy_stats.chi2_contingency(counts_n)
    n = counts_n.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(counts_n.shape) - 1))))

    order = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
    return {
        "composition_pct": counts.reindex(order),
        "raw_avg": raw_avg.reindex(order),
        "adjusted_avg": adjusted.reindex(order),
        "chi2": chi2,
        "p_value": p_value,
        "cramers_v": cramers_v,
    }


def format_report(by_pt: pd.DataFrame, cmh: dict, comp: dict) -> str:
    cat_hi, cat_lo = COMPARISON
    hi_col, lo_col = cat_hi.replace(" ", "_"), cat_lo.replace(" ", "_")

    lines = [
        "# z-test de is_whiff por altitud, controlando por AutoPitchType",
        "",
        "El chequeo de OOD (`reports/altitude_counterfactual_ood_check.md`) "
        "descarto que el contrafactual (is_whiff sube al mover un pitch a "
        "Extreme Altitude) sea un artefacto de extrapolacion: el % de "
        "combinaciones fuera de distribucion es del mismo orden que el que ya "
        "existe naturalmente al evaluar pitchers de test nunca vistos en "
        "train. Este reporte cierra el otro extremo de la pregunta: si el "
        "z-test observacional agregado de `reports/run_value_validation.md` "
        "(is_whiff -7.7% relativo Extreme vs No Altitude, z=-9.52) sobrevive "
        "al controlar por `AutoPitchType`, o si es composicion de tipos de "
        "pitch.",
        "",
        f"## 1. z-test DENTRO de cada AutoPitchType ({cat_hi} vs {cat_lo})",
        "",
        f"| AutoPitchType | n {cat_hi} | rate {cat_hi} | n {cat_lo} | rate {cat_lo} | "
        "diff | diff % relativo | z | p-value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in by_pt.iterrows():
        lines.append(
            f"| {r['AutoPitchType']} | {int(r[f'n_{hi_col}']):,} | {r[f'rate_{hi_col}']:.4f} | "
            f"{int(r[f'n_{lo_col}']):,} | {r[f'rate_{lo_col}']:.4f} | {r['diff']:+.4f} | "
            f"{r['diff_pct_relative']:+.2f}% | {r['z']:.2f} | {r['p_value']:.2e} |"
        )

    n_sig_same_dir = int(((by_pt["z"] < 0) & (by_pt["p_value"] < 0.05)).sum())
    n_total = len(by_pt)
    lines += [
        "",
        f"**{n_sig_same_dir} de {n_total}** tipos de pitch principales muestran la "
        f"MISMA direccion (is_whiff mas bajo en {cat_hi}) con p<0.05 dentro del "
        "propio tipo — si el efecto agregado fuera puramente composicion, "
        "esperariamos que la mayoria de estos z-tests individuales NO fueran "
        "significativos o cambiaran de signo.",
        "",
        "## 2. Test de Cochran-Mantel-Haenszel (pooled, controla por AutoPitchType)",
        "",
        f"- Odds ratio pooled ({cat_hi} vs {cat_lo}): **{cmh['or_pooled']:.4f}** "
        "(OR<1 = menos whiffs en la categoria de mayor altitud, dentro de cada "
        "estrato de tipo de pitch)",
        f"- Chi2 de Mantel-Haenszel (con correccion de continuidad): {cmh['chi2_mh']:.2f}, "
        f"p-value: {cmh['p_value_mh']:.2e}",
        f"- Eventos observados (whiffs en {cat_hi}, sumado sobre estratos): {cmh['sum_a']:,.0f} "
        f"vs. esperados bajo independencia: {cmh['sum_ea']:,.1f}",
        "",
        "## 3. Estandarizacion directa (reponderar por mezcla GLOBAL de AutoPitchType)",
        "",
        "Mismo metodo que `reports/magnus_analysis.md` para el residual de "
        "Magnus: recalcular is_whiff promedio de cada altitude_category "
        "reponderando con la mezcla GLOBAL fija de tipos de pitch, para "
        "aislar el efecto de composicion del efecto dentro-de-tipo.",
        "",
        "### Composicion de AutoPitchType por altitude_category (%)",
        "",
        comp["composition_pct"].round(2).to_markdown(),
        "",
        f"Test chi-cuadrado de independencia: chi2={comp['chi2']:.1f}, "
        f"p={comp['p_value']:.2e}. **Cramer's V = {comp['cramers_v']:.4f}** "
        "(V<0.1 = asociacion despreciable segun convenciones estandar).",
        "",
        "### is_whiff promedio: crudo vs. ajustado por composicion",
        "",
        "| altitude_category | crudo | ajustado por composicion |",
        "|---|---:|---:|",
    ]
    for cat in comp["raw_avg"].index:
        lines.append(f"| {cat} | {comp['raw_avg'][cat]:.4f} | {comp['adjusted_avg'][cat]:.4f} |")

    raw_gap = comp["raw_avg"][cat_lo] - comp["raw_avg"][cat_hi]
    adj_gap = comp["adjusted_avg"][cat_lo] - comp["adjusted_avg"][cat_hi]
    pct_explained_by_composition = 100 * (1 - adj_gap / raw_gap) if raw_gap != 0 else np.nan

    lines += [
        "",
        f"Brecha cruda ({cat_lo} - {cat_hi}): {raw_gap:+.4f}. Brecha ajustada por "
        f"composicion: {adj_gap:+.4f}. Composicion de AutoPitchType explica "
        f"~{pct_explained_by_composition:.1f}% de la brecha cruda "
        "(si este numero es chico, el efecto es genuinamente dentro-de-tipo, "
        "no un artefacto de que Extreme Altitude tira una mezcla distinta de "
        "pitches).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    df = load_data()
    logger.info("Filas validas: %s", f"{len(df):,}")

    by_pt = by_pitchtype_ztest(df)
    cmh = cochran_mantel_haenszel(df)
    comp = composition_adjusted_rates(df)

    report_text = format_report(by_pt, cmh, comp)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)
    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
