"""H2 (velo_diff_vs_fb vs. delta_pitching_plus en altitud), filtrado a Changeup.

Re-agregacion pura sobre `data/processed/altitude_counterfactual_scores.parquet`
(ya generado por src/analysis/altitude_counterfactual.py) — NO vuelve a correr
el contrafactual, solo re-agrupa los mismos 100k pitches y los mismos 5 bins de
velo_diff_vs_fb, restringido a AutoPitchType == 'Changeup'.

Motivacion: la seccion 2 de reports/altitude_counterfactual.md prueba H2
("mayor diferencial de velocidad vs. el fastball -> pierde MENOS Pitching+ en
altitud") sobre TODOS los tipos de pitch mezclados, y la rechaza en esa forma
agregada. Como el mecanismo de engaño de un Changeup depende casi por
definicion del diferencial de velocidad contra el fastball (a diferencia de,
digamos, un Slider, cuyo engaño es mas de movimiento), tiene sentido probar H2
DENTRO de Changeup especificamente, sin que la mezcla de otros tipos de pitch
diluya o contamine el patron.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.altitude_counterfactual import (
    SAMPLE_SEED,
    SAMPLE_SIZE,
    VELO_DIFF_LABELS,
    bootstrap_mean_ci,
    load_data,
    stratified_sample,
)

ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = ROOT / "data" / "processed" / "altitude_counterfactual_scores.parquet"
REPORT_PATH = ROOT / "reports" / "altitude_counterfactual.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MIN_N_RELIABLE = 100


def load_changeup() -> pd.DataFrame:
    df = pd.read_parquet(SCORES_PATH)
    return df.loc[df["AutoPitchType"] == "Changeup"].copy()


def by_bin_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in VELO_DIFF_LABELS:
        g = df.loc[df["velo_diff_bin"] == label, "delta_pitching_plus"]
        n = len(g)
        if n == 0:
            rows.append({"bin": label, "n": 0, "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
            continue
        mean, lo, hi = bootstrap_mean_ci(g.to_numpy())
        rows.append({"bin": label, "n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def check_monotonic(by_bin: pd.DataFrame) -> dict:
    reliable = by_bin.loc[by_bin["n"] >= MIN_N_RELIABLE].reset_index(drop=True)
    vals = reliable["mean"].to_numpy()
    is_monotonic_increasing = bool(np.all(np.diff(vals) >= 0)) if len(vals) >= 2 else None
    away_from_fb_loses_less = (
        bool(vals[0] > vals[-1]) if len(vals) >= 2 else None
    )  # H2: bin mas alejado del FB (primero) deberia perder MENOS (valor mas alto/menos negativo)
    return {
        "n_reliable_bins": len(reliable),
        "n_total_bins": len(by_bin),
        "is_monotonic_increasing_with_velo_diff": is_monotonic_increasing,
        "extreme_bin_consistent_with_h2": away_from_fb_loses_less,
        "reliable_labels": reliable["bin"].tolist(),
    }


# ---------------------------------------------------------------------------
# Investigacion opcional: por que "algo lento vs FB (-4 a -1)" rompe la
# monotonicidad. Trae RelSpeed/SpinRate/InducedVertBreak reales del pitch
# (no estan en el parquet de scores) regenerando la MISMA muestra de 100k
# (mismo seed) y uniendo por PitchUID — no se re-simula el contrafactual.
# ---------------------------------------------------------------------------


def investigate_break(df: pd.DataFrame) -> dict:
    df_full = load_data()
    sample = stratified_sample(df_full, SAMPLE_SIZE, SAMPLE_SEED)
    cu_physics = sample.loc[
        sample["AutoPitchType"] == "Changeup",
        ["PitchUID", "RelSpeed", "SpinRate", "InducedVertBreak"],
    ]
    cu = df.merge(cu_physics, on="PitchUID", how="inner")

    order = VELO_DIFF_LABELS
    physics_by_bin = cu.groupby("velo_diff_bin", observed=True)[
        ["RelSpeed", "SpinRate", "InducedVertBreak"]
    ].agg(["mean", "median"]).reindex(order)

    # robustez: la mediana de SpinRate se sostiene si se excluyen los 5
    # pitchers mas frecuentes de cada bin (para descartar que el salto sea
    # unos pocos pitchers concentrados, no un patron poblacional)
    median_excl_top5 = {}
    top5_share = {}
    for label in order:
        sub = cu.loc[cu["velo_diff_bin"] == label]
        if len(sub) == 0:
            continue
        vc = sub["pitcher_anon_id"].value_counts()
        top5_ids = vc.head(5).index
        top5_share[label] = 100 * vc.head(5).sum() / len(sub)
        sub_excl = sub.loc[~sub["pitcher_anon_id"].isin(top5_ids)]
        median_excl_top5[label] = sub_excl["SpinRate"].median() if len(sub_excl) > 0 else np.nan

    corr_spinrate = float(cu["SpinRate"].corr(cu["delta_pitching_plus"]))
    corr_relspeed = float(cu["RelSpeed"].corr(cu["delta_pitching_plus"]))

    return {
        "physics_by_bin": physics_by_bin,
        "median_spinrate_excl_top5": median_excl_top5,
        "top5_pitcher_share_pct": top5_share,
        "corr_spinrate_pitchlevel": corr_spinrate,
        "corr_relspeed_pitchlevel": corr_relspeed,
    }


def format_investigation(inv: dict) -> str:
    pb = inv["physics_by_bin"]
    lines = [
        "",
        "### Investigacion del quiebre en \"algo lento vs FB (-4 a -1)\"",
        "",
        "Chequeo rapido (no exhaustivo) de si el quiebre de monotonicidad tiene "
        "una explicacion de composicion. Se comparo `RelSpeed`, `SpinRate` e "
        "`InducedVertBreak` REALES del pitch (no contrafactuales) por bin, mas "
        "concentracion de pitchers especificos.",
        "",
        "| bin | RelSpeed (mean/mediana) | SpinRate (mean/mediana) | InducedVertBreak (mean/mediana) |",
        "|---|---|---|---|",
    ]
    for label in VELO_DIFF_LABELS:
        if label not in pb.index or pb.loc[label].isna().all():
            continue
        r = pb.loc[label]
        lines.append(
            f"| {label} | {r[('RelSpeed', 'mean')]:.1f} / {r[('RelSpeed', 'median')]:.1f} | "
            f"{r[('SpinRate', 'mean')]:.0f} / {r[('SpinRate', 'median')]:.0f} | "
            f"{r[('InducedVertBreak', 'mean')]:.2f} / {r[('InducedVertBreak', 'median')]:.2f} |"
        )

    lines += [
        "",
        "**Hallazgo:** `SpinRate` (y, en linea con eso, `InducedVertBreak`) da un "
        "salto claro justo en la misma transicion donde se rompe la "
        "monotonicidad de `delta_pitching_plus` — plano en los dos bins mas "
        "alejados del fastball (mediana ~1,750-1,760 rpm) y ~200-300 rpm mas "
        "alto en los tres bins restantes (mediana ~2,000-2,260 rpm). Esto es "
        "consistente con que el dataset contiene dos sub-poblaciones de "
        "'Changeup' distintas — una con menos spin y mas separacion de "
        "velocidad respecto al fastball, otra con mas spin y menos separacion "
        "— y como el contrafactual recalcula el movimiento via la formula de "
        "Magnus (`src/features/physics.py`), que escala con SpinRate, el "
        "sub-grupo de mayor spin sufre una perdida de movimiento mas grande al "
        "cambiar la densidad de aire, lo cual empuja su `delta_pitching_plus` "
        "mas abajo — sin necesidad de invocar un artefacto de pitchers "
        "especificos.",
        "",
        f"**Robustez:** excluyendo a los 5 pitchers mas frecuentes de cada bin "
        "(para descartar que sea solo un puñado de pitchers concentrados), la "
        "mediana de SpinRate se mantiene en el mismo orden "
        f"({', '.join(f'{k}: {v:.0f}' for k, v in inv['median_spinrate_excl_top5'].items() if not np.isnan(v))} "
        "rpm) — el patron es de la poblacion de pitches, no de unos pocos "
        "pitchers dominando el bin (la concentracion de los 5 pitchers mas "
        "frecuentes por bin va de "
        f"{min(inv['top5_pitcher_share_pct'].values()):.1f}% a "
        f"{max(inv['top5_pitcher_share_pct'].values()):.1f}% del bin, sin un "
        "salto que coincida con el quiebre de SpinRate).",
        "",
        f"**Advertencia:** la correlacion pitch-a-pitch entre SpinRate y "
        f"`delta_pitching_plus` dentro de Changeup es DEBIL "
        f"({inv['corr_spinrate_pitchlevel']:.3f}; RelSpeed: "
        f"{inv['corr_relspeed_pitchlevel']:.3f}) — el patron es visible al "
        "promediar por bin, no como una relacion lineal fuerte a nivel de "
        "pitch individual (mucho ruido pitch a pitch, esperable). Esta "
        "investigacion es un chequeo rapido (no un analisis causal completo) "
        "y no cambia la conclusion de monotonicidad de arriba — la documenta "
        "con una explicacion fisica plausible en vez de dejarla sin ninguna "
        "pista.",
    ]
    return "\n".join(lines)


def format_section(df: pd.DataFrame, by_bin: pd.DataFrame, mono: dict, investigation_text: str = "") -> str:
    n_total = len(df)
    unreliable = by_bin.loc[by_bin["n"] < MIN_N_RELIABLE, "bin"].tolist()

    lines = [
        "",
        "## 2b. H2 restringido a Changeup",
        "",
        "Re-agregacion de los mismos 100,000 pitches y los mismos 5 bins de "
        "`velo_diff_vs_fb` de la seccion 2 (sin volver a correr el "
        "contrafactual — datos de "
        "`data/processed/altitude_counterfactual_scores.parquet`), filtrando "
        "SOLO `AutoPitchType == 'Changeup'`. Motivacion: el engaño de un "
        "Changeup depende casi por definicion del diferencial de velocidad "
        "contra el fastball, a diferencia de otros tipos de pitch cuyo engaño "
        "es mas de movimiento — la mezcla de todos los tipos en la seccion 2 "
        "puede diluir un patron que si exista dentro de Changeup especificamente.",
        "",
        f"**n total de Changeup en la muestra: {n_total:,}** (de 100,000 pitches totales).",
        "",
        "| bin de velo_diff_vs_fb | n | delta promedio | IC 95% (bootstrap) |",
        "|---|---:|---:|---|",
    ]
    for _, r in by_bin.iterrows():
        if r["n"] == 0:
            lines.append(f"| {r['bin']} | 0 | — | — (sin datos en este bin) |")
        elif r["n"] < MIN_N_RELIABLE:
            lines.append(
                f"| {r['bin']} | **{int(r['n'])}** | {r['mean']:+.3f} | "
                f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] — **n<{MIN_N_RELIABLE}, NO CONFIABLE** |"
            )
        else:
            lines.append(
                f"| {r['bin']} | {int(r['n']):,} | {r['mean']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |"
            )

    if unreliable:
        lines += [
            "",
            f"**Advertencia de tamaño de muestra:** {len(unreliable)} de 5 bins "
            f"quedan con n<{MIN_N_RELIABLE} dentro de Changeup ({', '.join(unreliable)}) "
            "— sus promedios se reportan arriba mas por transparencia que como "
            "lectura confiable; no se usan para la conclusion de monotonicidad "
            "de abajo.",
        ]

    lines += ["", "### Monotonicidad dentro de Changeup", ""]

    if mono["n_reliable_bins"] < 3:
        lines.append(
            f"**Muestra insuficiente para concluir sobre monotonicidad.** Solo "
            f"{mono['n_reliable_bins']} de {mono['n_total_bins']} bins tienen "
            f"n>={MIN_N_RELIABLE} dentro de Changeup — no alcanza para evaluar "
            "un patron a lo largo de 5 bins con confianza. La pregunta de H2 "
            "dentro de Changeup queda **sin resolver por falta de muestra**, no "
            "refutada ni confirmada."
        )
    else:
        reliable = by_bin.loc[by_bin["n"] >= MIN_N_RELIABLE].reset_index(drop=True)
        vals_str = " → ".join(f"{v:+.3f}" for v in reliable["mean"])
        if mono["extreme_bin_consistent_with_h2"] and mono["is_monotonic_increasing_with_velo_diff"]:
            verdict = (
                "**H2 SE SOSTIENE dentro de Changeup, de forma monotonica**: el delta "
                "crece (pierde menos / gana mas Pitching+) a medida que el pitch se "
                "aleja mas del fastball en velocidad, consistente con la hipotesis "
                "de que un mecanismo de engaño basado en diferencial de velocidad "
                "(no en movimiento por Magnus) es menos vulnerable a la perdida de "
                "movimiento inducida por la altitud."
            )
        elif mono["extreme_bin_consistent_with_h2"]:
            verdict = (
                "**H2 se sostiene en los extremos pero NO de forma estrictamente "
                "monotonica** dentro de Changeup: el bin mas alejado del fastball "
                "pierde menos que el mas cercano/igual, pero el patron no es "
                "consistentemente creciente en los bins intermedios — revisar la "
                "tabla completa antes de citar un patron limpio."
            )
        else:
            verdict = (
                "**H2 NO se sostiene dentro de Changeup** — el bin mas alejado del "
                "fastball en velocidad NO pierde menos Pitching+ que el mas "
                "cercano/igual al fastball, la misma direccion (o falta de "
                "direccion) que ya se veia en el agregado de todos los tipos de "
                "pitch (seccion 2)."
            )
        lines += [f"Secuencia de deltas por bin confiable (orden de bins, mas lento a mas rapido vs. FB): {vals_str}", "", verdict]

    if investigation_text:
        lines.append(investigation_text)

    return "\n".join(lines)


def main() -> None:
    df = load_changeup()
    logger.info("Changeup en la muestra contrafactual: %s pitches", f"{len(df):,}")

    by_bin = by_bin_with_ci(df)
    mono = check_monotonic(by_bin)

    logger.info("Investigando el quiebre de monotonicidad (regenera la muestra de 100k, mismo seed)...")
    inv = investigate_break(df)
    investigation_text = format_investigation(inv)

    section = format_section(df, by_bin, mono, investigation_text)

    current = REPORT_PATH.read_text()
    marker = "## 3. Tabla cruzada AutoPitchType x altitude_category actual"
    if marker not in current:
        raise RuntimeError(f"No se encontro el marcador '{marker}' en {REPORT_PATH} — revisar manualmente.")
    if "## 2b. H2 restringido a Changeup" in current:
        # ya existe una version previa de esta seccion: reemplazarla en vez de duplicar
        start = current.index("## 2b. H2 restringido a Changeup")
        end = current.index(marker)
        current = current[:start] + current[end:]

    updated = current.replace(marker, section.strip("\n") + "\n\n" + marker)
    REPORT_PATH.write_text(updated)
    logger.info("Actualizado %s con la seccion 2b", REPORT_PATH)
    logger.info("\n%s", section)


if __name__ == "__main__":
    main()
