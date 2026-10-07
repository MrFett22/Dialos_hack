"""Stuff+ v4 — P(whiff | swing) con features "puras", neutral por altitud y con shrinkage.

Antecedentes (reports/resultados_finales.md §7.3-7.4):
- Stuff+ v1 (is_whiff sobre todos los pitches) era en buena parte ubicación
  escondida en VAA y ángulos de salida.
- Quitar la ubicación con el target is_whiff no funciona: la ubicación decide
  si el bateador tira. La cadena completa sin ubicación tampoco (P(swing) y
  el resultado del take son casi aleatorios sin ubicación).
- El diagnóstico P(whiff | swing) con features v2d fue el mejor Stuff+ "puro".

Diseño (declarado ANTES de correr):
1. Modelo: P(whiff | swing), entrenado solo con swings, features de v2d
   (espejo de zurdos, VAA/HAA ajustados por ubicación, sin redundancias, con
   diferencias contra la recta; sin ubicación ni conteo). Hiperparámetros v1.
2. Se predice para TODOS los pitches ("si el bateador le tirara, ¿qué tan
   probable es que falle?"), out-of-fold por pitcher, sobre los pitches
   PROYECTADOS a nivel del mar (mismo contrafactual que el Pitching+ oficial).
3. Promedio por pitcher-temporada con shrinkage empírico-bayesiano y escala
   100 + 10·z (z en unidades de habilidad real τ), intervalo de 80%.

Criterio de aceptación (declarado antes de correr, coherente con §7.2 y §7.7):
- Principal: correlación con las carreras del año siguiente AJUSTADAS POR
  ENTORNO, mín. 100 pitches, delta pareado contra Stuff+ v1 con P ≥ 90%.
- Específico: correlación con el whiff% del año siguiente.

Salidas: data/processed/stuff_plus_v4_scores.parquet, reports/stuff_plus_v4.md,
reports/benchmark/stuff_plus_v4.json.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models import pitching_plus_neutral as neutral_mod
from src.models import run_value_chain as chain
from src.validation import benchmark

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "data" / "processed" / "stuff_plus_v4_oof_pitches.parquet"
SCORES_PATH = ROOT / "data" / "processed" / "stuff_plus_v4_scores.parquet"
REPORT_PATH = ROOT / "reports" / "stuff_plus_v4.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "stuff_plus_v4.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FEATURES = chain.FEATURE_SETS["stuff"]


def generate_oof(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    if OOF_PATH.exists() and not refresh:
        logger.info("Usando OOF en cache: %s", OOF_PATH.name)
        return pd.read_parquet(OOF_PATH), json.loads(OOF_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))

    df = chain.load_data()
    neutral = neutral_mod.neutral_features(df)
    cols = FEATURES["numeric"] + FEATURES["categorical"]
    swings = chain.universe_masks(df)["swing"]
    y = df["y_whiff"]
    p_obs = np.full(len(df), np.nan)
    p_neu = np.full(len(df), np.nan)
    t0 = time.time()
    for tr, ho in GroupKFold(n_splits=chain.N_FOLDS).split(df, groups=df["pitcher_anon_id"]):
        tr = tr[swings[tr]]
        model = lgb.LGBMClassifier(verbosity=-1, **chain.LGB_PARAMS)
        model.fit(df[cols].iloc[tr], y.iloc[tr].astype(int), categorical_feature=FEATURES["categorical"])
        p_obs[ho] = model.predict_proba(df[cols].iloc[ho])[:, 1]
        p_neu[ho] = model.predict_proba(neutral[cols].iloc[ho])[:, 1]
    logger.info("OOF P(whiff | swing): %.0fs", time.time() - t0)

    out = df[["PitchUID", "pitcher_anon_id", "year", "AutoPitchType", "event_group", "y_whiff"]].copy()
    out["AutoPitchType"] = out["AutoPitchType"].astype("object")
    out["p_whiff_swing_obs"] = p_obs
    out["p_whiff_swing_neutral"] = p_neu
    out.to_parquet(OOF_PATH, index=False)

    ys = y[swings].astype(int).to_numpy()
    meta = {
        "auc_swings": float(roc_auc_score(ys, p_obs[swings])),
        "log_loss_swings": float(log_loss(ys, p_obs[swings])),
        "base_rate_swings": float(ys.mean()),
        "n_swings": int(swings.sum()),
    }
    OOF_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta


def build_scores(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # empirical_bayes trata "menor = mejor"; aquí más whiff es mejor, así que se usa el negativo
    oof = oof.assign(neg_neutral=-oof["p_whiff_swing_neutral"], neg_obs=-oof["p_whiff_swing_obs"])
    neu, params = neutral_mod.empirical_bayes(oof, "neg_neutral")
    obs, _ = neutral_mod.empirical_bayes(oof, "neg_obs")
    scores = neu.rename(columns={
        "n": "n_pitches", "plus": "stuff_plus", "plus_raw": "stuff_plus_sin_shrinkage",
        "plus_lo80": "stuff_plus_lo80", "plus_hi80": "stuff_plus_hi80",
    })[["pitcher_anon_id", "year", "n_pitches", "stuff_plus", "stuff_plus_lo80", "stuff_plus_hi80", "stuff_plus_sin_shrinkage"]]
    scores["p_whiff_swing_neutral"] = -neu["mean"]
    scores = scores.merge(
        obs.rename(columns={"plus": "stuff_plus_con_altitud"})[["pitcher_anon_id", "year", "stuff_plus_con_altitud"]],
        on=["pitcher_anon_id", "year"],
    )
    params = {y: {**p, "mu": -p["mu"]} for y, p in params.items()}  # μ de vuelta en P(whiff | swing)
    return scores, params


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['r']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_delta(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] · P={d['p_model_better']:.0%}"


def format_report(meta, params, bench, model_metrics, adjusted, by_type, top, scores) -> str:
    lines = [
        "# Stuff+ v4 — P(whiff | swing), neutral por altitud, con shrinkage",
        "",
        "Generado por `src/models/stuff_plus_v4.py`. Diseño y criterio de aceptación declarados en el docstring antes de correr.",
        "",
        "## Modelo",
        "",
        f"- P(whiff | swing) OOF sobre {meta['n_swings']:,} swings: AUC={meta['auc_swings']:.4f}, "
        f"log-loss={meta['log_loss_swings']:.4f}, tasa base={meta['base_rate_swings']:.4f}.",
        "- Features (sin ubicación ni conteo): " + ", ".join(FEATURES["numeric"] + FEATURES["categorical"]),
        "- Score: promedio de P(whiff | swing) de todos los pitches del pitcher, proyectados a nivel del mar; shrinkage "
        "empírico-bayesiano; 10 puntos = 1 desviación estándar de habilidad real.",
        "",
        "| temporada | μ P(whiff\\|swing) | τ | pitches para encoger a la mitad |",
        "|---|---:|---:|---:|",
    ]
    for y, p in params.items():
        lines.append(f"| {y} | {p['mu']:.4f} | {p['tau']:.4f} | {(p['sigma'] / p['tau']) ** 2:,.0f} |")
    lines += ["", "## Criterio principal: carreras del año siguiente ajustadas por entorno", "",
              "| mín. pitches | pares | Stuff+ v1 | v4 con altitud | v4 neutral | **v4 oficial (neutral + shrinkage)** | Δ v4 oficial vs v1 |",
              "|---|---:|---|---|---|---|---|"]
    for mp, res in adjusted.items():
        r = res["r"]
        lines.append(
            f"| {mp} | {res['n_pairs']} | {_fmt(r['stuff_plus_v1'])} | {_fmt(r['stuff_plus_con_altitud'])} | "
            f"{_fmt(r['stuff_plus_sin_shrinkage'])} | {_fmt(r['stuff_plus'])} | {_fmt_delta(res['delta_vs_reference']['stuff_plus'])} |"
        )
    for mp, res in bench.items():
        lines += [
            "",
            f"## Benchmark año-contra-año — mínimo {mp} pitches ({res['n_pairs']:,} pares)",
            "",
            "| métrica | estabilidad | carreras observadas t+1 | K% t+1 | whiff% t+1 |",
            "|---|---|---|---|---|",
        ]
        for n, lab in model_metrics.items():
            pr = res["predictive"][n]
            lines.append(f"| {lab} | {_fmt(res['stability'][n])} | {_fmt(pr['obs_rv_prevented'])} | {_fmt(pr['obs_k_pct'])} | {_fmt(pr['obs_whiff_rate'])} |")
        for b in ["obs_whiff_rate", "obs_k_pct", "obs_csw"]:
            pr = res["predictive"][b]
            lines.append(f"| *{benchmark.BASELINE_METRICS[b]}* | {_fmt(res['stability'][b])} | {_fmt(pr['obs_rv_prevented'])} | {_fmt(pr['obs_k_pct'])} | {_fmt(pr['obs_whiff_rate'])} |")
        d = res["deltas_vs_reference"]
        lines += ["", "Delta contra Stuff+ v1 (bootstrap pareado):", "", "| métrica | estabilidad | carreras observadas t+1 | whiff% t+1 |", "|---|---|---|---|"]
        for n in [k for k in model_metrics if k != "stuff_plus_v1"]:
            lines.append(f"| {model_metrics[n]} | {_fmt_delta(d[f'{n}|stability'])} | {_fmt_delta(d[f'{n}|obs_rv_prevented'])} | {_fmt_delta(d[f'{n}|obs_whiff_rate'])} |")
        b = res["deltas_vs_best_baseline"]
        lines += ["", "Contra el mejor baseline observado:", "", "| métrica | whiff% t+1 | carreras observadas t+1 |", "|---|---|---|"]
        for n, lab in model_metrics.items():
            w, rv = b[f"{n}|obs_whiff_rate"], b[f"{n}|obs_rv_prevented"]
            lines.append(f"| {lab} | {_fmt_delta(w)} (vs {w['vs']}) | {_fmt_delta(rv)} (vs {rv['vs']}) |")
    lines += [
        "",
        "## P(whiff | swing) promedio por tipo de pitch (a nivel del mar)",
        "",
        by_type.round(4).to_markdown(),
        "",
        "## Top 10 Stuff+ v4 (≥300 pitches)",
        "",
        top.round(2).to_markdown(index=False),
        "",
        f"Distribución (todas las pitcher-temporadas): media={scores['stuff_plus'].mean():.2f}, std={scores['stuff_plus'].std():.2f}, "
        f"min={scores['stuff_plus'].min():.2f}, max={scores['stuff_plus'].max():.2f}.",
        "",
    ]
    return "\n".join(lines)


def run(refresh: bool = False) -> dict:
    oof, meta = generate_oof(refresh=refresh)
    logger.info("AUC sobre swings: %.4f", meta["auc_swings"])
    scores, params = build_scores(oof)
    scores.to_parquet(SCORES_PATH, index=False)

    names = {"stuff_plus_con_altitud": "Stuff+ v4 con altitud", "stuff_plus_sin_shrinkage": "Stuff+ v4 neutral",
             "stuff_plus": "Stuff+ v4 neutral + shrinkage"}
    table, _ = benchmark.build_metric_table(extra_models={n: scores for n in names})
    model_metrics = {"stuff_plus_v1": "Stuff+ v1", **names}
    refs = {n: "stuff_plus_v1" for n in names}
    bench = {
        mp: benchmark.correlation_suite(benchmark.build_pairs(table, mp), model_metrics=model_metrics, references=refs)
        for mp in benchmark.MIN_PITCHES_LEVELS
    }
    adj_table = scores.merge(table[["pitcher_anon_id", "year", "stuff_plus_v1"]], on=["pitcher_anon_id", "year"], how="left")
    adjusted = neutral_mod.park_adjusted_test(
        adj_table, scores=("stuff_plus_v1", "stuff_plus_con_altitud", "stuff_plus_sin_shrinkage", "stuff_plus"),
        reference="stuff_plus_v1",
    )
    for mp in benchmark.MIN_PITCHES_LEVELS:
        logger.info(
            "min %d | carreras ajustadas t+1: %s | whiff%% t+1: %s", mp,
            {k: round(v["r"], 3) for k, v in adjusted[mp]["r"].items()},
            {n: round(bench[mp]["predictive"][n]["obs_whiff_rate"]["r"], 3) for n in model_metrics},
        )

    by_type = oof.groupby("AutoPitchType")["p_whiff_swing_neutral"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    top = scores.loc[scores["n_pitches"] >= 300].sort_values("stuff_plus", ascending=False).head(10)
    REPORT_PATH.write_text(format_report(meta, params, bench, model_metrics, adjusted, by_type, top, scores), encoding="utf-8")
    payload = {"meta": meta, "eb_params": params, "park_adjusted": {str(k): v for k, v in adjusted.items()},
               "benchmark": {str(k): v for k, v in bench.items()}}
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular predicciones OOF")
    run(refresh=parser.parse_args().refresh)


if __name__ == "__main__":
    main()
