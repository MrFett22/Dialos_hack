"""Comparación antes/después de la corrección del golpeado (HBP), auditoría 2026-10-05.

Compara las salidas del pipeline de antes de la corrección (copiadas en
`--before`, un directorio con los archivos originales) contra las actuales:
número de HBP, run_value, valores terminales, V(b,s), Δ, Pitching+ y los
pitchers más afectados. Se corre una vez; queda en el repositorio como
registro reproducible de la corrección.

Salida: reports/hbp_antes_despues.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "hbp_antes_despues.md"


def _meta(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    m = json.loads(path.read_text(encoding="utf-8"))
    v = pd.DataFrame(m["count_values"]).set_index(["Balls", "Strikes"])["V"]
    d = pd.DataFrame(m["deltas"]).set_index(["Balls", "Strikes"])
    return m["terminal_values"], v, d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", type=Path, required=True)
    before = parser.parse_args().before

    cols = ["PitchUID", "PitchCall", "KorBB", "run_value", "is_hit_by_pitch"]
    old = pd.read_parquet(before / "pitches_with_targets.parquet", columns=cols)
    new = pd.read_parquet(PROC / "pitches_with_targets.parquet", columns=cols)
    m = old.merge(new, on="PitchUID", suffixes=("_antes", "_despues"))
    hbp = m["PitchCall_despues"] == "HitByPitch"
    walk = (m["KorBB_despues"] == "Walk") & ~hbp

    term_o, v_o, d_o = _meta(before / "run_value_chain_oof_pitches.meta.json")
    term_n, v_n, d_n = _meta(PROC / "run_value_chain_oof_pitches.meta.json")

    off_o = pd.read_parquet(before / "pitching_plus_official.parquet")
    off_n = pd.read_parquet(PROC / "pitching_plus_official.parquet")
    k = ["pitcher_anon_id", "year"]
    pp = off_o[k + ["n_pitches", "pitching_plus", "pitching_plus_con_altitud"]].merge(
        off_n[k + ["pitching_plus", "pitching_plus_con_altitud"]], on=k, suffixes=("_antes", "_despues"))
    pp["cambio"] = pp["pitching_plus_despues"] - pp["pitching_plus_antes"]
    pp["rank_antes"] = pp.groupby("year")["pitching_plus_antes"].rank(ascending=False)
    pp["rank_despues"] = pp.groupby("year")["pitching_plus_despues"].rank(ascending=False)
    big = pp[pp["n_pitches"] >= 100]

    # tasa de HBP por pitcher-temporada (para explicar quién se mueve)
    tg = pd.read_parquet(PROC / "pitches_with_targets.parquet", columns=["pitcher_anon_id", "year", "PitchCall"])
    rate = tg.assign(h=(tg["PitchCall"] == "HitByPitch")).groupby(k)["h"].mean().mul(100).rename("hbp_por_100").reset_index()
    big = big.merge(rate, on=k, how="left")

    L = [
        "# Corrección del golpeado (HBP): antes vs. después",
        "",
        "Generado por `src/validation/hbp_before_after.py`. *Antes* = pipeline del commit `db1eed6`; *después* = pipeline "
        "regenerado desde cero con `is_hit_by_pitch` reconstruida desde PitchCall (`src/features/targets.py`).",
        "",
        "## 1. Target",
        "",
        "| | antes | después |",
        "|---|---:|---:|",
        f"| pitches con is_hit_by_pitch = 1 | {int(m['is_hit_by_pitch_antes'].sum()):,} | {int(m['is_hit_by_pitch_despues'].sum()):,} |",
        f"| run_value medio del golpeado | {m.loc[hbp, 'run_value_antes'].mean():+.4f} | {m.loc[hbp, 'run_value_despues'].mean():+.4f} |",
        f"| golpeados con run_value = 0 | {int((m.loc[hbp, 'run_value_antes'] == 0).sum()):,} | {int((m.loc[hbp, 'run_value_despues'] == 0).sum()):,} |",
        f"| run_value medio de la base por bolas | {m.loc[walk, 'run_value_antes'].mean():+.4f} | {m.loc[walk, 'run_value_despues'].mean():+.4f} |",
        f"| pitches con run_value distinto | — | {int((m['run_value_antes'].fillna(-9) != m['run_value_despues'].fillna(-9)).sum()):,} |",
        f"| pitches sin run_value (turno no resoluble) | {int(m['run_value_antes'].isna().sum()):,} | {int(m['run_value_despues'].isna().sum()):,} |",
        "",
        "El run_value cambia también fuera de los golpeados: el golpeado ahora cierra el turno y pone un corredor en primera, "
        "así que cambian los estados base-out del resto de la media entrada y la propia matriz RE24.",
        "",
        "## 2. Valores terminales (carreras)",
        "",
        "| evento | antes | después | cambio |",
        "|---|---:|---:|---:|",
    ]
    for e in term_n:
        L.append(f"| {e} | {term_o[e]:+.4f} | {term_n[e]:+.4f} | {term_n[e] - term_o[e]:+.4f} |")
    L += ["", "## 3. V(b, s): carreras esperadas del turno desde cada conteo", "", "| conteo | antes | después | cambio |", "|---|---:|---:|---:|"]
    for bs in v_n.index:
        L.append(f"| {bs[0]}-{bs[1]} | {v_o[bs]:+.4f} | {v_n[bs]:+.4f} | {v_n[bs] - v_o[bs]:+.4f} |")
    L += ["", "## 4. Δ(golpeado, conteo) y conteos afectados", "", "| conteo | Δ(hbp) antes | Δ(hbp) después | Δ(bola) después |", "|---|---:|---:|---:|"]
    for bs in d_n.index:
        L.append(f"| {bs[0]}-{bs[1]} | {d_o.loc[bs, 'hbp']:+.4f} | {d_n.loc[bs, 'hbp']:+.4f} | {d_n.loc[bs, 'ball']:+.4f} |")
    neg_o, neg_n = int((d_o["hbp"] < 0).sum()), int((d_n["hbp"] < 0).sum())
    max_other = float((d_n.drop(columns="hbp") - d_o.drop(columns="hbp")).abs().max().max())
    L += [
        "",
        f"Conteos con Δ(hbp) < 0 (el golpeado \"favorece\" al pitcher): **{neg_o} antes → {neg_n} después**. "
        f"Cambio máximo en el Δ de los demás eventos: {max_other:.4f} carreras.",
        "",
        "## 5. Pitching+ oficial",
        "",
        f"- Pitcher-temporadas comparadas: {len(pp):,} ({len(big):,} con ≥100 pitches).",
        f"- Correlación antes/después (≥100): Pearson {big['pitching_plus_antes'].corr(big['pitching_plus_despues']):.4f}, "
        f"Spearman {big['pitching_plus_antes'].corr(big['pitching_plus_despues'], method='spearman'):.4f}.",
        f"- Cambio en puntos (≥100): media {big['cambio'].mean():+.2f}, sd {big['cambio'].std():.2f}, "
        f"|cambio| máximo {big['cambio'].abs().max():.2f}; {int((big['cambio'].abs() >= 2).sum())} pitcher-temporadas se mueven ≥2 puntos.",
        f"- Correlación entre el cambio y la tasa de golpeados del pitcher: {big['cambio'].corr(big['hbp_por_100']):+.3f} "
        "(los que más golpean bajan: antes el golpeado se les contaba casi como un strike).",
        "",
        "### Pitchers más afectados (≥100 pitches)",
        "",
    ]
    show = ["pitcher_anon_id", "year", "n_pitches", "hbp_por_100", "pitching_plus_antes", "pitching_plus_despues", "cambio", "rank_antes", "rank_despues"]
    top = big.reindex(big["cambio"].abs().sort_values(ascending=False).index).head(12)
    L += [top[show].round(2).to_markdown(index=False), ""]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
