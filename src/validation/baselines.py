"""Baselines obligatorios del ajuste ambiental y validación preespecificada.

Pregunta: ¿qué valor ADICIONAL aporta la modelación física (contrafactual)
más allá de una normalización ambiental simple? No se asume que deba ganar.

Todos los scores usan las mismas predicciones OOF (`pitching_plus_neutral_oof_pitches`),
el mismo shrinkage empírico-bayesiano y la misma escala 100 + 10·(μ − θ̂)/τ.
Solo cambia cómo se trata el entorno:

| modelo | trato del entorno |
|---|---|
| M0  sin ajuste           | E[Δ] con las features observadas |
| M1  ajuste simple        | E[Δ] observado − su promedio en temporada × categoría de altitud |
| M1b ajuste simple estadio| E[Δ] observado − su promedio en temporada × estadio anónimo (si el juego no tiene estadio resuelto, × categoría) |
| M2  contrafactual físico | cada pitch proyectado a No Altitude con ratios de densidad por categoría × tipo (índice oficial) |
| M3a físico por estadio   | igual, con el ratio de densidad del estadio anónimo del juego |
| M3b físico + estadio     | M2 − su promedio en temporada × estadio anónimo |

Validaciones:
1. PREESPECIFICADA (bitácora §7.2, fijada antes de ver mejoras): carreras evitadas
   observadas (RE24) por 100 pitches en t+1; mínimo 100 pitches en ambas temporadas;
   éxito = Δr > 0 contra Pitching+ v1 con P(Δr > 0) ≥ 0.90 (bootstrap pareado por
   pitcher, 2,000 réplicas). Se conserva sin cambios.
2. SECUNDARIA (post-hoc): carreras t+1 ajustadas por entorno, en dos variantes
   (resultado menos su promedio por temporada × categoría, o por temporada ×
   estadio anónimo). Cada variante favorece por construcción al ajuste que usa
   la misma partición; por eso se reportan las dos y se interpreta con cuidado.
3. NEUTRALIZACIÓN: correlación del score con el % de pitches en Extreme y con la
   "exposición física" del pitcher (promedio del ratio de drag de los estadios
   donde lanzó).

Salidas: reports/baselines.md, reports/benchmark/baselines.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import parks
from src.models import pitching_plus_neutral as pn
from src.validation import benchmark

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "baselines.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "baselines.json"
P_THRESHOLD = 0.90

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LABELS = {
    "pitching_plus_v1": "Pitching+ v1 (referencia preespecificada)",
    "m0": "M0 · sin ajuste ambiental",
    "m1": "M1 · ajuste simple temporada × categoría",
    "m1b": "M1b · ajuste simple temporada × estadio anónimo",
    "m2": "M2 · contrafactual físico (oficial)",
    "m3a": "M3a · contrafactual físico con densidad por estadio",
    "m3b": "M3b · contrafactual físico + ajuste por estadio",
}
SCORES = [k for k in LABELS if k != "pitching_plus_v1"]
REFERENCES = ["m0", "m1", "m1b"]


def environment_key(oof: pd.DataFrame) -> pd.Series:
    """Estadio anónimo resuelto o, si falta, la categoría de altitud."""
    return oof["stadium_id"].fillna(oof["altitude_category"])


def build_scores(oof: pd.DataFrame) -> pd.DataFrame:
    """Una fila por pitcher-temporada con todos los scores comparados."""
    oof = oof.copy()
    env = environment_key(oof) if "stadium_id" in oof else oof["altitude_category"]
    oof["xrv_m1"] = oof["xrv_obs"] - oof.groupby(["year", "altitude_category"])["xrv_obs"].transform("mean")
    oof["xrv_m1b"] = oof["xrv_obs"] - oof.groupby([oof["year"], env])["xrv_obs"].transform("mean")
    oof["xrv_m3b"] = oof["xrv_neutral"] - oof.groupby([oof["year"], env])["xrv_neutral"].transform("mean")
    cols = {"m0": "xrv_obs", "m1": "xrv_m1", "m1b": "xrv_m1b", "m2": "xrv_neutral", "m3b": "xrv_m3b"}
    if "xrv_neutral_estadio" in oof:
        cols["m3a"] = "xrv_neutral_estadio"
    out = None
    for name, col in cols.items():
        eb, _ = pn.empirical_bayes(oof, col)
        eb = eb[["pitcher_anon_id", "year", "n", "plus"]].rename(columns={"plus": name})
        out = eb if out is None else out.merge(eb.drop(columns="n"), on=["pitcher_anon_id", "year"])
    out = out.rename(columns={"n": "n_pitches"})
    mix = pd.crosstab([oof["pitcher_anon_id"], oof["year"]], oof["altitude_category"], normalize="index")
    mix = mix.reindex(columns=["No Altitude", "Medium Altitude", "Extreme Altitude"], fill_value=0.0)
    mix.columns = ["pct_no", "pct_medium", "pct_extreme"]
    out = out.merge(mix.reset_index(), on=["pitcher_anon_id", "year"])
    if "stadium_id" in oof:
        st = parks.load_stadium_ratios()["drag_ratio"]
        drag = oof["stadium_id"].map(st)
        exp = oof.assign(drag=drag).dropna(subset=["drag"]).groupby(["pitcher_anon_id", "year"])["drag"].mean().rename("exposicion_drag")
        out = out.merge(exp.reset_index(), on=["pitcher_anon_id", "year"], how="left")
    return out


def _delta_row(res: dict, model: str, ref: str, outcome: str) -> dict:
    d = res["deltas_vs_reference"][f"{model}|{outcome}"]
    return {"modelo": model, "referencia": ref, "delta_r": d["delta_r"], "ci95": d["ci95"], "p": d["p_model_better"],
            "cumple": bool(d["delta_r"] > 0 and d["p_model_better"] >= P_THRESHOLD)}


def prespecified(scores: pd.DataFrame, names: list[str]) -> dict:
    """Validación §7.2: carreras observadas t+1, contra Pitching+ v1 (y contra M0/M1), P ≥ 0.90."""
    table, _ = benchmark.build_metric_table(extra_models={s: scores for s in names})
    metrics = {k: LABELS[k] for k in ["pitching_plus_v1"] + names}
    out = {}
    for mp in benchmark.MIN_PITCHES_LEVELS:
        pairs = benchmark.build_pairs(table, mp)
        res = {ref: benchmark.correlation_suite(pairs, model_metrics=metrics, references={s: ref for s in names if s != ref})
               for ref in ["pitching_plus_v1"] + [r for r in REFERENCES if r in names]}
        base = res["pitching_plus_v1"]
        out[mp] = {
            "n_pairs": base["n_pairs"],
            "r_pred": {k: base["predictive"][k]["obs_rv_prevented"] for k in metrics},
            "stability": {k: base["stability"][k] for k in metrics},
            "deltas": [_delta_row(res[ref], s, ref, "obs_rv_prevented") for ref in res for s in names if s != ref],
        }
    return out


def secondary(scores: pd.DataFrame, names: list[str], adjust_by: str) -> dict:
    """Validación secundaria post-hoc: carreras t+1 ajustadas por entorno (categoría o estadio)."""
    out = {}
    for ref in [r for r in REFERENCES if r in names]:
        res = pn.park_adjusted_test(scores, scores=tuple(names), reference=ref, adjust_by=adjust_by)
        for mp, r in res.items():
            o = out.setdefault(mp, {"n_pairs": r["n_pairs"], "r": r["r"], "deltas": []})
            for s, d in r["delta_vs_reference"].items():
                o["deltas"].append({"modelo": s, "referencia": ref, "delta_r": d["delta_r"], "ci95": d["ci95"], "p": d["p_model_better"],
                                    "cumple": bool(d["delta_r"] > 0 and d["p_model_better"] >= P_THRESHOLD)})
    return out


def neutralization(scores: pd.DataFrame, names: list[str]) -> dict:
    big = scores[scores["n_pitches"] >= 100]
    out = {}
    for s in names:
        out[s] = {"pct_extreme": float(big[s].corr(big["pct_extreme"])),
                  **{f"pct_extreme_{int(y)}": float(g[s].corr(g["pct_extreme"])) for y, g in big.groupby("year")}}
        if "exposicion_drag" in big:
            out[s]["exposicion_drag"] = float(big[s].corr(big["exposicion_drag"]))
    return out


def _f(d: dict) -> str:
    return f"{d['r']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def _fd(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] · P={d['p']:.0%} · {'✅' if d['cumple'] else '❌'}"


def format_report(names, pre, sec_cat, sec_st, neu) -> str:
    L = [
        "# Baselines obligatorios del ajuste ambiental",
        "",
        "Generado por `src/validation/baselines.py`. Todos los scores usan las mismas predicciones OOF, el mismo shrinkage y la "
        "misma escala; solo cambia cómo se trata el entorno (tabla en el docstring del módulo). Criterio de éxito en todas las "
        "comparaciones: Δr > 0 y P(Δr > 0) ≥ 0.90 (bootstrap pareado por pitcher, 2,000 réplicas).",
        "",
        "## 1. Validación PREESPECIFICADA (§7.2: carreras observadas t+1, mín. 100, contra Pitching+ v1)",
        "",
    ]
    for mp, r in pre.items():
        tag = "**criterio oficial**" if mp == 100 else "sensibilidad (no preespecificada)"
        L += [f"### Mínimo {mp} pitches ({r['n_pairs']} pares) — {tag}", "", "| score | estabilidad | carreras observadas t+1 |", "|---|---|---|"]
        for k in ["pitching_plus_v1"] + names:
            L.append(f"| {LABELS[k]} | {_f(r['stability'][k])} | {_f(r['r_pred'][k])} |")
        L += ["", "| modelo | referencia | Δr [IC 95%] · P · cumple |", "|---|---|---|"]
        for d in r["deltas"]:
            L.append(f"| {LABELS[d['modelo']]} | {LABELS[d['referencia']]} | {_fd(d)} |")
        L.append("")
    for title, sec in [("temporada × categoría", sec_cat), ("temporada × estadio anónimo", sec_st)]:
        L += [f"## 2. Validación SECUNDARIA (post-hoc): carreras t+1 ajustadas por {title}", "",
              "Resultado t+1 = valor realizado de cada pitch menos el promedio de su " + title + ". Favorece por construcción al "
              "ajuste simple que usa la misma partición.", ""]
        for mp, r in sec.items():
            L += [f"### Mínimo {mp} pitches ({r['n_pairs']} pares)", "", "| score | r con carreras ajustadas t+1 |", "|---|---|"]
            for k in names + ["rv_ajustado"]:
                L.append(f"| {LABELS.get(k, 'carreras ajustadas en t (baseline observado)')} | {_f(r['r'][k])} |")
            L += ["", "| modelo | referencia | Δr [IC 95%] · P · cumple |", "|---|---|---|"]
            for d in r["deltas"]:
                L.append(f"| {LABELS[d['modelo']]} | {LABELS[d['referencia']]} | {_fd(d)} |")
            L.append("")
    years = sorted(k.split("_")[-1] for k in next(iter(neu.values())) if k.startswith("pct_extreme_"))
    L += ["## 3. Neutralización (pitcher-temporadas con ≥100 pitches)", "",
          "Correlación del score con el % de pitches en Extreme (total y por temporada) y con la exposición física (promedio del "
          "ratio de drag de los estadios anónimos donde lanzó; < 1 = más altitud).", "",
          "| score | % Extreme | " + " | ".join(years) + " | exposición física |", "|---|---:|" + "---:|" * len(years) + "---:|"]
    for s, v in neu.items():
        L.append(f"| {LABELS[s]} | {v['pct_extreme']:+.3f} | " + " | ".join(f"{v[f'pct_extreme_{y}']:+.3f}" for y in years)
                 + f" | {v.get('exposicion_drag', float('nan')):+.3f} |")
    L.append("")
    return "\n".join(L)


def run() -> dict:
    oof = pd.read_parquet(pn.OOF_PATH)
    scores = build_scores(oof)
    names = [s for s in SCORES if s in scores]
    pre = prespecified(scores, names)
    sec_cat = secondary(scores, names, "categoria")
    sec_st = secondary(scores, names, "estadio")
    neu = neutralization(scores, names)
    REPORT_PATH.write_text(format_report(names, pre, sec_cat, sec_st, neu), encoding="utf-8")
    payload = {"prespecified": {str(k): v for k, v in pre.items()},
               "secondary_posthoc_categoria": {str(k): v for k, v in sec_cat.items()},
               "secondary_posthoc_estadio": {str(k): v for k, v in sec_st.items()}, "neutralization": neu}
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


if __name__ == "__main__":
    run()
