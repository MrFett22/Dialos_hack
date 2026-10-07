"""Calidad neutral vs. sensibilidad a la altitud extrema, por pitcher.

Dos conceptos distintos (P1.9 de la auditoría):

- CALIDAD NEUTRAL: Pitching+ oficial (`pitching_plus`), el pitcher proyectado a
  No Altitude, con shrinkage.
- SENSIBILIDAD A LA ALTITUD EXTREMA: cuánto peor le va al pitcher en Extreme
  que en No Altitude. Se miden dos versiones, que NO son lo mismo:

  1. Implicada por el modelo (S_mod). Promedio sobre los pitches del pitcher de
     E[Δ | proyectado a Extreme] − E[Δ | proyectado a No Altitude], en carreras
     por 100 pitches (contrafactual, `altitude_counterfactual_v2_pitches`).
     Todos los pitchers reciben los MISMOS ratios de densidad por tipo de pitch,
     así que S_mod depende de qué lanza (mezcla y características de sus
     pitches), no de cómo reacciona él en particular a la altitud. Se reporta
     qué tanto de S_mod explica la sola mezcla de pitches.
  2. Empírica (S_emp). Con los pitches que el pitcher realmente lanzó en Extreme
     y en No Altitude (todas sus temporadas):
         S_emp = 100·[ media_E(ỹ) − media_N(ỹ) ],  ỹ = Δ realizado − media(Δ realizado | temporada, categoría)
     Es su diferencia E − N relativa a la diferencia E − N de toda la liga.
     Positivo = le va relativamente peor en Extreme. Es ruidosa: se reporta su
     error estándar, su confiabilidad (correlación entre mitades de juegos
     pares/impares, corregida por Spearman–Brown) y una versión con shrinkage
     hacia 0.

Pregunta de validación: ¿S_mod predice S_emp? Si no, la "resistencia a la
altitud" del modelo describe el tipo de pitches, no una habilidad del pitcher
demostrada en los datos.

Salidas: data/processed/sensibilidad_altitud.parquet, reports/sensibilidad_altitud.md.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import altitude_counterfactual_v2 as cf
from src.models import pitching_plus_neutral as pn
from src.models import run_value_chain as chain

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "sensibilidad_altitud.parquet"
REPORT_PATH = ROOT / "reports" / "sensibilidad_altitud.md"
MIN_EACH = 150  # pitches mínimos en Extreme y en No Altitude para la versión empírica
MAIN_TYPES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def model_sensitivity() -> pd.DataFrame:
    """S_mod por pitcher (todas sus temporadas) y qué parte explica la mezcla de pitches."""
    p = pd.read_parquet(cf.PITCHES_OUT, columns=["pitcher_anon_id", "year", "AutoPitchType", "delta"])
    p["AutoPitchType"] = p["AutoPitchType"].astype(object)
    league = p["delta"].mean()
    g = p.groupby("pitcher_anon_id").agg(n_total=("delta", "size"), s_mod=("delta", "mean"), sd=("delta", "std"))
    g["s_mod_rel"] = 100 * (g["s_mod"] - league)
    g["s_mod"] *= 100
    g["s_mod_se"] = 100 * g["sd"] / np.sqrt(g["n_total"])
    mix = pd.crosstab(p["pitcher_anon_id"], p["AutoPitchType"].where(p["AutoPitchType"].isin(MAIN_TYPES), "Otro"), normalize="index")
    big = g[g["n_total"] >= 300].join(mix)
    X = np.column_stack([np.ones(len(big))] + [big[c] for c in mix.columns[:-1]])
    beta = np.linalg.lstsq(X, big["s_mod"].to_numpy(), rcond=None)[0]
    resid = big["s_mod"].to_numpy() - X @ beta
    r2 = 1 - resid.var() / big["s_mod"].var()
    return g.drop(columns="sd").reset_index(), float(r2)


def empirical_sensitivity() -> tuple[pd.DataFrame, dict]:
    real = pd.read_parquet(chain.OOF_PATH, columns=["PitchUID", "pitcher_anon_id", "year", "realized_delta"])
    alt = pd.read_parquet(cf.RAW_PATH, columns=["PitchUID", "altitude_category", "game_anon_id"])
    d = real.merge(alt, on="PitchUID")
    d["adj"] = d["realized_delta"] - d.groupby(["year", "altitude_category"])["realized_delta"].transform("mean")
    d = d[d["altitude_category"].isin(["No Altitude", "Extreme Altitude"])]
    d["half"] = pd.factorize(d["game_anon_id"])[0] % 2

    def stats(x: pd.DataFrame) -> pd.DataFrame:
        s = x.groupby(["pitcher_anon_id", "altitude_category"])["adj"].agg(["mean", "var", "size"]).unstack("altitude_category")
        s = s.dropna()
        out = pd.DataFrame({
            "n_extreme": s[("size", "Extreme Altitude")], "n_no": s[("size", "No Altitude")],
            "s_emp": 100 * (s[("mean", "Extreme Altitude")] - s[("mean", "No Altitude")]),
            "s_emp_se": 100 * np.sqrt(s[("var", "Extreme Altitude")] / s[("size", "Extreme Altitude")]
                                      + s[("var", "No Altitude")] / s[("size", "No Altitude")]),
        })
        return out

    full = stats(d)
    full = full[(full["n_extreme"] >= MIN_EACH) & (full["n_no"] >= MIN_EACH)]
    h0, h1 = stats(d[d["half"] == 0]), stats(d[d["half"] == 1])
    halves = h0[["s_emp"]].join(h1[["s_emp"]], lsuffix="_0", rsuffix="_1", how="inner").join(full[[]], how="inner")
    r_half = float(halves["s_emp_0"].corr(halves["s_emp_1"]))
    reliability = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
    tau2 = float(full["s_emp"].var() - (full["s_emp_se"] ** 2).mean())
    shrink = (full["s_emp_se"] ** 2) / (full["s_emp_se"] ** 2 + max(tau2, 0.0)) if tau2 > 0 else pd.Series(1.0, index=full.index)
    full["s_emp_shrunk"] = (1 - shrink) * full["s_emp"]
    info = {"n_pitchers": len(full), "r_mitades": r_half, "confiabilidad_spearman_brown": reliability,
            "sd_observada": float(full["s_emp"].std()), "se_medio": float(np.sqrt((full["s_emp_se"] ** 2).mean())),
            "tau_s_emp": float(np.sqrt(tau2)) if tau2 > 0 else 0.0}
    return full.reset_index(), info


def run() -> dict:
    s_mod, r2_mix = model_sensitivity()
    s_emp, info = empirical_sensitivity()
    off = pd.read_parquet(pn.OFFICIAL_PATH)
    quality = off.groupby("pitcher_anon_id").apply(
        lambda g: pd.Series({"pitching_plus_neutral": np.average(g["pitching_plus"], weights=g["n_pitches"]),
                             "n_pitches": g["n_pitches"].sum(), "temporadas": g["year"].nunique()}), include_groups=False).reset_index()
    t = quality.merge(s_mod, on="pitcher_anon_id", how="left").merge(s_emp, on="pitcher_anon_id", how="left")
    t.to_parquet(OUT_PATH, index=False)

    both = t.dropna(subset=["s_emp", "s_mod_rel"])
    r_mod_emp = float(both["s_mod_rel"].corr(both["s_emp"]))
    r_mod_emp_s = float(both["s_mod_rel"].corr(both["s_emp"], method="spearman"))
    big = t[t["n_pitches"] >= 300].dropna(subset=["s_mod_rel"])
    r_quality_smod = float(big["pitching_plus_neutral"].corr(big["s_mod_rel"]))
    qx, qy = big["pitching_plus_neutral"].median(), big["s_mod_rel"].median()
    quad = pd.crosstab(np.where(big["pitching_plus_neutral"] >= qx, "calidad alta", "calidad baja"),
                       np.where(big["s_mod_rel"] <= qy, "sensibilidad baja", "sensibilidad alta"))

    L = [
        "# Calidad neutral vs. sensibilidad a la altitud extrema",
        "",
        "Generado por `src/analysis/sensitivity.py`. Definiciones en el docstring. Positivo en sensibilidad = peor en Extreme.",
        "",
        "## 1. Sensibilidad implicada por el modelo (S_mod)",
        "",
        f"- Promedio de la liga: {100 * pd.read_parquet(cf.PITCHES_OUT, columns=['delta'])['delta'].mean():+.2f} carreras/100 pitches "
        "(Extreme − No Altitude, mismo pitch proyectado a ambas).",
        f"- Dispersión entre pitchers con ≥300 pitches: sd = {big['s_mod_rel'].std():.2f} carreras/100.",
        f"- **R² de S_mod explicado solo por la mezcla de tipos de pitch: {r2_mix:.2f}.** Lo que queda depende de las "
        "características de sus pitches (velocidad, movimiento, ubicación), no de una reacción individual a la altitud: todos los "
        "pitchers reciben los mismos ratios de densidad.",
        "",
        "## 2. Sensibilidad empírica (S_emp)",
        "",
        f"- Pitchers con ≥{MIN_EACH} pitches en Extreme y en No Altitude: {info['n_pitchers']}.",
        f"- sd observada de S_emp: {info['sd_observada']:.2f} carreras/100; error estándar típico: {info['se_medio']:.2f}.",
        f"- Correlación entre mitades (juegos pares vs. impares): {info['r_mitades']:+.3f} → confiabilidad (Spearman–Brown) "
        f"**{info['confiabilidad_spearman_brown']:+.3f}**.",
        f"- sd \"real\" estimada entre pitchers (τ, momentos): {info['tau_s_emp']:.2f} carreras/100.",
        "",
        "## 3. ¿El modelo predice la sensibilidad observada?",
        "",
        f"- corr(S_mod relativo, S_emp) = **{r_mod_emp:+.3f}** (Spearman {r_mod_emp_s:+.3f}; {len(both)} pitchers).",
        "",
        "## 4. Calidad neutral vs. sensibilidad del modelo (pitchers con ≥300 pitches)",
        "",
        f"- corr(Pitching+ neutral, S_mod relativo) = {r_quality_smod:+.3f}.",
        "- Cuadrantes (cortes en las medianas):",
        "",
        quad.to_markdown(),
        "",
        "## 5. Lectura",
        "",
        "- La sensibilidad del modelo es un **perfil de pitches**: dice qué tipo de pitches pierde más valor cuando el aire es "
        "menos denso. No es una habilidad individual de adaptación medida en los datos.",
        "- La sensibilidad empírica es la que podría revelar una habilidad individual, pero su confiabilidad indica cuánto de "
        "ella es señal. Con confiabilidad cercana a 0, rankear pitchers por \"resistencia a la altitud\" observada no es defendible.",
        "- El cuadrante \"calidad alta + sensibilidad baja\" se puede mostrar, pero debe leerse como \"buen pitcher con un "
        "repertorio que el modelo espera que pierda poco en altura\", no como \"pitcher demostradamente resistente\".",
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    logger.info("\n".join(L))
    return {"r2_mix": r2_mix, "empirical": info, "r_mod_emp": r_mod_emp, "r_quality_smod": r_quality_smod}


if __name__ == "__main__":
    run()
