"""Fichas de scouting v2 — sobre el índice oficial Pitching+ v3 neutral por altitud.

Reemplaza a src/reports/scouting_report.py (v1). Cada ficha usa solo
resultados ya calculados por el pipeline oficial:

- data/processed/pitching_plus_official.parquet (Pitching+ neutral con
  shrinkage e intervalo de 80%, score con altitud, mezcla de altitudes,
  costo de altitud personal);
- data/processed/pitching_plus_neutral_oof_pitches.parquet (E[Δ carreras]
  por pitch, observado y proyectado a nivel del mar, out-of-fold);
- data/processed/altitude_counterfactual_v2_pitches.parquet (costo de la
  altitud por pitch: E[Δ] en Extreme − E[Δ] en No Altitude, separado en
  calidad del pitch y bola en juego).

Secciones: resumen por temporada, arsenal (carreras evitadas por tipo
contra el promedio de liga del mismo tipo, a nivel del mar), costo de la
altitud por tipo de pitch y recomendaciones por reglas fijas (sin LLM).

Selección: los 5 mejores Pitching+ y los 5 con mayor costo de altitud en
su temporada más reciente con ≥300 pitches.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import altitude_counterfactual_v2 as cf_v2

ROOT =Path(__file__).resolve().parents[2]
OFFICIAL_PATH = ROOT / "data" / "processed" / "pitching_plus_official.parquet"
NEUTRAL_OOF_PATH = ROOT / "data" / "processed" / "pitching_plus_neutral_oof_pitches.parquet"
ALTITUDE_PITCHES_PATH = ROOT / "data" / "processed" / "altitude_counterfactual_v2_pitches.parquet"
PHYSICS_PATH = ROOT / "data" / "processed" / "pitches_with_physics.parquet"
STUFF_V4_PITCHES_PATH = ROOT / "data" / "processed" / "stuff_plus_v4_oof_pitches.parquet"
OUT_DIR = ROOT / "reports" / "scouting"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MIN_PITCHES_SELECTION = 300
MIN_PITCHES_ARSENAL_ROW = 15
N_PER_LIST = 5
WIDE_INTERVAL_POINTS = 12.0  # intervalo de credibilidad de 80% más ancho que esto → advertencia de muestra


def load_resources() -> dict:
    official = pd.read_parquet(OFFICIAL_PATH)
    neutral = pd.read_parquet(NEUTRAL_OOF_PATH, columns=["PitchUID", "pitcher_anon_id", "year", "AutoPitchType", "xrv_neutral"])
    alt = pd.read_parquet(ALTITUDE_PITCHES_PATH, columns=["PitchUID", "delta"] + cf_v2.PRE_CONTACT_COLUMNS)
    alt["antes_contacto"] = alt[cf_v2.PRE_CONTACT_COLUMNS].sum(axis=1)
    phys = pd.read_parquet(PHYSICS_PATH, columns=["PitchUID", "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate"])
    stuff_v4 = pd.read_parquet(STUFF_V4_PITCHES_PATH, columns=["PitchUID", "p_whiff_swing_neutral"])
    pitches = (
        neutral.merge(alt, on="PitchUID", how="left").merge(phys, on="PitchUID", how="left")
        .merge(stuff_v4, on="PitchUID", how="left")
    )
    league_whiff_by_type = pitches.groupby("AutoPitchType")["p_whiff_swing_neutral"].mean()
    league_by_type = pitches.groupby("AutoPitchType")["xrv_neutral"].mean()
    league_cost_by_type = pitches.groupby("AutoPitchType")["delta"].mean()
    big = official.loc[official["n_pitches"] >= 100]
    cost_q = big["costo_altitud_total_por_100"].quantile([0.25, 0.75]).to_dict()
    return {"official": official, "pitches": pitches, "league_by_type": league_by_type,
            "league_cost_by_type": league_cost_by_type, "league_whiff_by_type": league_whiff_by_type,
            "cost_quartiles": cost_q}


def select_pitchers(official: pd.DataFrame) -> dict[str, list[str]]:
    eligible = official.loc[official["n_pitches"] >= MIN_PITCHES_SELECTION]
    latest = eligible.sort_values("year").groupby("pitcher_anon_id").tail(1)
    top = latest.sort_values("pitching_plus", ascending=False).head(N_PER_LIST)["pitcher_anon_id"].tolist()
    risk = (
        latest.loc[~latest["pitcher_anon_id"].isin(top)]
        .sort_values("costo_altitud_total_por_100", ascending=False)
        .head(N_PER_LIST)["pitcher_anon_id"].tolist()
    )
    return {"top_pitching_plus": top, "top_altitude_risk": risk}


def arsenal_table(pid: str, year: int, res: dict) -> pd.DataFrame:
    p = res["pitches"]
    sub = p.loc[(p["pitcher_anon_id"] == pid) & (p["year"] == year)]
    g = sub.groupby("AutoPitchType")
    t = g.agg(
        n=("xrv_neutral", "size"), velo=("RelSpeed", "mean"), ivb=("InducedVertBreak", "mean"), hb=("HorzBreak", "mean"),
        spin=("SpinRate", "mean"), xrv=("xrv_neutral", "mean"), costo_total=("delta", "mean"), costo_antes_contacto=("antes_contacto", "mean"),
        whiff_swing=("p_whiff_swing_neutral", "mean"),
    )
    t = t.loc[t["n"] >= MIN_PITCHES_ARSENAL_ROW].copy()
    t["uso_pct"] = 100 * t["n"] / len(sub)
    # Stuff+ v4 usado SOLO dentro de cada tipo de pitch (§7.9): P(whiff | swing) esperado por el "stuff"
    # del pitch, a nivel del mar, contra el promedio de la liga para ese mismo tipo, en puntos porcentuales.
    t["whiff_swing_vs_tipo_pp"] = 100 * (t["whiff_swing"] - res["league_whiff_by_type"].reindex(t.index))
    t["carreras_evitadas_vs_liga_por_100"] = -100 * (t["xrv"] - res["league_by_type"].reindex(t.index))
    # Valor en altitud extrema vs. un pitch promedio del mismo tipo también lanzado en altitud extrema:
    # su valor a nivel del mar menos lo que pierde DE MÁS (o de menos) que el promedio de su tipo.
    t["carreras_evitadas_vs_liga_en_altitud_por_100"] = (
        t["carreras_evitadas_vs_liga_por_100"] - 100 * (t["costo_total"] - res["league_cost_by_type"].reindex(t.index))
    )
    t["costo_total"] *= 100
    t["costo_antes_contacto"] *= 100
    return t.sort_values("n", ascending=False)


def recommendations(row: pd.Series, arsenal: pd.DataFrame, res: dict) -> list[str]:
    recs = []
    pct = row["percentil_temporada"] * 100
    width = row["pitching_plus_hi80"] - row["pitching_plus_lo80"]
    recs.append(
        f"**Nivel:** Pitching+ {row['pitching_plus']:.1f} (percentil {pct:.0f} de su temporada), "
        f"intervalo de credibilidad de 80% [{row['pitching_plus_lo80']:.1f}, {row['pitching_plus_hi80']:.1f}] "
        "(solo refleja el tamaño de muestra; no la incertidumbre de los modelos)."
    )
    if width > WIDE_INTERVAL_POINTS:
        recs.append(f"**Muestra:** intervalo de {width:.0f} puntos — tomar el nivel con cautela; {int(row['n_pitches'])} pitches en la temporada.")

    q25, q75 = res["cost_quartiles"][0.25], res["cost_quartiles"][0.75]
    cost = row["costo_altitud_total_por_100"]
    if cost >= q75:
        level = "ALTO (cuartil superior de la liga)"
    elif cost <= q25:
        level = "BAJO (cuartil inferior de la liga)"
    else:
        level = "intermedio"
    pre = row["costo_altitud_antes_contacto_por_100"]
    recs.append(
        f"**Riesgo de altitud {level} (según el modelo):** en altitud extrema su arsenal permite {cost:+.2f} carreras más por cada "
        f"100 pitches que a nivel del mar; de eso, {pre:+.2f} ocurre antes del contacto (menos whiffs, más pelotas en juego) y el "
        f"resto es el resultado de la bola en juego, que mezcla calidad del contacto y parque. La parte antes del contacto equivale "
        f"a ≈{pre * 27.5:+.1f} carreras por temporada de abridor (≈{pre * 9:+.1f} de relevista). Es una sensibilidad implicada por "
        "el modelo (depende de qué lanza), no una resistencia a la altitud medida en sus propios juegos "
        "(`reports/sensibilidad_altitud.md`)."
    )
    if len(arsenal):
        worst = arsenal["costo_antes_contacto"].idxmax()
        recs.append(
            f"**Pitch que más pierde antes del contacto en altitud:** {worst} ({arsenal.loc[worst, 'costo_antes_contacto']:+.2f} carreras/100)."
        )
        top_sea = arsenal["carreras_evitadas_vs_liga_por_100"].idxmax()
        top_alt = arsenal["carreras_evitadas_vs_liga_en_altitud_por_100"].idxmax()
        recs.append(
            f"**Mejor pitch a nivel del mar:** {top_sea} "
            f"({arsenal.loc[top_sea, 'carreras_evitadas_vs_liga_por_100']:+.2f} carreras/100 vs. el promedio de liga del mismo tipo)."
        )
        if top_alt != top_sea:
            recs.append(
                f"**En altitud extrema el mejor pitch cambia a {top_alt}** "
                f"({arsenal.loc[top_alt, 'carreras_evitadas_vs_liga_en_altitud_por_100']:+.2f} vs. "
                f"{arsenal.loc[top_sea, 'carreras_evitadas_vs_liga_en_altitud_por_100']:+.2f} de {top_sea}, contra el promedio "
                f"de su tipo también en altitud): considerar darle más uso en casa."
            )
        else:
            recs.append(f"**En altitud extrema {top_sea} sigue siendo su mejor pitch** frente al promedio de su tipo en el mismo entorno.")
    if row["pct_extreme_altitude"] >= 0.5:
        recs.append(
            f"**Contexto:** lanzó {row['pct_extreme_altitude']:.0%} de sus pitches en altitud extrema; el score con altitud "
            f"({row['pitching_plus_con_altitud']:.1f}) lo subestima — el neutral corrige ese entorno."
        )
    return recs


def format_report(pid: str, res: dict) -> str:
    off = res["official"]
    seasons = off.loc[off["pitcher_anon_id"] == pid].sort_values("year")
    ref = seasons.loc[seasons["n_pitches"] >= MIN_PITCHES_SELECTION].iloc[-1] if (seasons["n_pitches"] >= MIN_PITCHES_SELECTION).any() else seasons.iloc[-1]
    year = int(ref["year"])
    arsenal = arsenal_table(pid, year, res)

    cols = ["year", "n_pitches", "pitching_plus", "pitching_plus_lo80", "pitching_plus_hi80", "percentil_temporada",
            "pitching_plus_con_altitud", "stuff_plus_v1", "pct_no_altitude", "pct_medium_altitude", "pct_extreme_altitude",
            "costo_altitud_total_por_100", "costo_altitud_antes_contacto_por_100"]
    summary = seasons[cols].copy()
    summary["percentil_temporada"] = (100 * summary["percentil_temporada"]).round(0)
    for c in ["pct_no_altitude", "pct_medium_altitude", "pct_extreme_altitude"]:
        summary[c] = (100 * summary[c]).round(0)

    ars = arsenal[["n", "uso_pct", "velo", "ivb", "hb", "spin", "whiff_swing_vs_tipo_pp", "carreras_evitadas_vs_liga_por_100",
                   "carreras_evitadas_vs_liga_en_altitud_por_100", "costo_antes_contacto", "costo_total"]].rename(columns={
        "whiff_swing_vs_tipo_pp": "stuff: whiff/swing vs su tipo (pp)",
        "uso_pct": "uso %", "velo": "velo (mph)", "ivb": "IVB (pulg)", "hb": "HB (pulg)", "spin": "spin (rpm)",
        "carreras_evitadas_vs_liga_por_100": "carreras evitadas/100 vs liga (nivel del mar)",
        "carreras_evitadas_vs_liga_en_altitud_por_100": "carreras evitadas/100 vs liga (altitud extrema)",
        "costo_antes_contacto": "costo altitud: antes del contacto/100", "costo_total": "costo altitud: total/100",
    })
    lines = [
        f"# Ficha de scouting — {pid}",
        "",
        "Índice oficial: **Pitching+ v3 neutral por altitud** (100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). Altitud extrema = varios estadios de la "
        "categoría Extreme del dataset (estadios anónimos; el dataset no identifica estadios reales). Generado por "
        "`src/reports/scouting_report_v2.py`.",
        "",
        "## 1. Resumen por temporada",
        "",
        summary.round(2).to_markdown(index=False),
        "",
        "`pitching_plus` = neutral por altitud con shrinkage (oficial) · `lo80`/`hi80` = intervalo de credibilidad empírico-bayesiano de 80% · "
        "`pitching_plus_con_altitud` = mismo índice sin quitar el entorno · "
        "`stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado) · `pct_*` = % de sus pitches en cada categoría · "
        "`costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar.",
        "",
        f"## 2. Arsenal ({year})",
        "",
        ars.round(2).to_markdown(),
        "",
        "*Stuff: whiff/swing vs su tipo*: puntos porcentuales de probabilidad de swing-and-miss que el modelo espera por las "
        "características físicas del pitch (sin ubicación, a nivel del mar), contra el promedio de la liga para ese mismo tipo. "
        "Solo es comparable entre pitches del mismo tipo. "
        "*Carreras evitadas/100 vs liga*: cuántas carreras evita cada 100 pitches de ese tipo contra el promedio de la liga "
        "del mismo tipo en el mismo entorno (nivel del mar o altitud extrema). *Costo de altitud*: carreras extra cada 100 pitches en altitud extrema; "
        "*antes del contacto* = swing, take, whiff y foul, incluida la mayor probabilidad de contacto (atribución de Shapley); "
        "*total* = incluye el resultado de la bola en juego (que mezcla calidad del contacto y parque).",
        "",
        f"## 3. Recomendaciones ({year})",
        "",
    ]
    lines += [f"- {r}" for r in recommendations(ref, arsenal, res)]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    res = load_resources()
    selection = select_pitchers(res["official"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # La selección cambia al re-entrenar: se borran las fichas v2 anteriores (las v1 viven en OUT_DIR/v1)
    for old in OUT_DIR.glob("pitcher_*_top_*.md"):
        old.unlink()
    index_lines = [
        "# Fichas de scouting (v2)",
        "",
        "Sobre el índice oficial Pitching+ v3 neutral por altitud (`reports/pitching_plus_neutral.md`). "
        f"Selección: temporada más reciente con ≥{MIN_PITCHES_SELECTION} pitches. Las fichas v1 quedan en `reports/scouting/v1/`.",
        "",
    ]
    for group, pids in selection.items():
        title = "Mejores Pitching+" if group == "top_pitching_plus" else "Mayor costo de altitud"
        index_lines += [f"## {title}", ""]
        for pid in pids:
            fname = f"{pid}_{group}.md"
            (OUT_DIR / fname).write_text(format_report(pid, res), encoding="utf-8")
            row = res["official"].loc[(res["official"]["pitcher_anon_id"] == pid) & (res["official"]["n_pitches"] >= MIN_PITCHES_SELECTION)].iloc[-1]
            index_lines.append(
                f"- [{pid}]({fname}) — {int(row['year'])}: Pitching+ {row['pitching_plus']:.1f}, "
                f"costo de altitud {row['costo_altitud_total_por_100']:+.2f} carreras/100 pitches"
            )
            logger.info("Ficha: %s", fname)
        index_lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(index_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
