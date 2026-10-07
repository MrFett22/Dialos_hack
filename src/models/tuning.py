"""Ajuste de hiperparámetros de los 5 sub-modelos de Pitching+ v3 (índice oficial).

Diseño (declarado ANTES de correr):
- Objetivo: log-loss pitch a pitch en pitchers de validación (regla de
  puntuación propia). El benchmark año-contra-año NO entra en el ajuste, para
  no sobreajustar la métrica de evaluación.
- Partición: GroupShuffleSplit por pitcher (80% ajuste / 20% validación,
  seed 7, distinta a la de los folds OOF).
- Búsqueda: Optuna (TPE), 25 ensayos por sub-modelo. El primer ensayo es la
  configuración base (para que el resultado nunca sea peor que la base en
  validación). learning_rate = 0.1 con early stopping (50 rondas) durante la
  búsqueda; el modelo final usa learning_rate = 0.05 y el doble de árboles.
- Espacio: num_leaves [15, 255], min_child_samples [20, 3000], reg_lambda
  [1e-3, 50], reg_alpha [1e-3, 10], colsample_bytree [0.4, 1], subsample [0.5, 1].

Criterio de adopción (declarado antes de correr):
1. El log-loss OOF (5 folds por pitcher, mismo esquema del índice oficial)
   mejora en los 5 sub-modelos.
2. La calibración (ECE) de los sub-modelos binarios no empeora en más de 0.002.
3. El índice oficial ajustado no empeora en el benchmark de carreras
   ajustadas por entorno: P(peor) < 90% con mín. 100 y con mín. 300 pitches.

Salidas: src/models/artifacts/pitching_plus_v3_tuned_params_candidate.json,
reports/tuning.md, reports/benchmark/tuning.json. Con `--activate` el
candidato se copia a pitching_plus_v3_tuned_params.json y pasa a ser el que
usan los modelos oficiales.
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

from src.models import pitching_plus_neutral as neutral_mod
from src.models import run_value_chain as chain
from src.validation import benchmark

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATH = ROOT / "src" / "models" / "artifacts" / "pitching_plus_v3_tuned_params_candidate.json"
TUNED_OOF_PATH = ROOT / "data" / "processed" / "pitching_plus_neutral_oof_pitches_tuned.parquet"
BASE_OOF_PATH = ROOT / "data" / "processed" / "pitching_plus_neutral_oof_pitches_base.parquet"
REPORT_PATH = ROOT / "reports" / "tuning.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "tuning.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 25
SEARCH_LR = 0.1
FINAL_LR = 0.05
EARLY_STOP = 50
MAX_TREES = 3000
MAX_TUNE_ROWS = 300_000
SPLIT_SEED = 7
ECE_TOLERANCE = 0.002
BASE_TRIAL = {"num_leaves": 31, "min_child_samples": 20, "reg_lambda": 1e-3, "reg_alpha": 1e-3,
              "colsample_bytree": 0.8, "subsample": 0.8}


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------


def tune_submodel(df: pd.DataFrame, name: str, universe: np.ndarray, target: str, n_classes: int) -> dict:
    feats = chain.FEATURE_SETS["pitching"]
    cols = feats["numeric"] + feats["categorical"]
    sub = df.loc[universe]
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_SEED).split(sub, groups=sub["pitcher_anon_id"]))
    tr, va = sub.iloc[tr_idx], sub.iloc[va_idx]
    if len(tr) > MAX_TUNE_ROWS:
        tr = tr.sample(MAX_TUNE_ROWS, random_state=SPLIT_SEED)
    X_tr, y_tr = tr[cols], tr[target].astype(int)
    X_va, y_va = va[cols], va[target].astype(int)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 3000, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        model = lgb.LGBMClassifier(
            n_estimators=MAX_TREES, learning_rate=SEARCH_LR, subsample_freq=1, random_state=42, n_jobs=-1, verbosity=-1, **params
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=feats["categorical"],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        trial.set_user_attr("best_iter", int(model.best_iteration_ or MAX_TREES))
        p = model.predict_proba(X_va)
        return float(log_loss(y_va, p if n_classes > 2 else p[:, 1], labels=list(range(n_classes)) if n_classes > 2 else None))

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SPLIT_SEED))
    study.enqueue_trial(BASE_TRIAL)
    study.optimize(objective, n_trials=N_TRIALS)
    base_value = study.trials[0].value
    best = study.best_trial
    final = {
        **best.params,
        "n_estimators": int(2 * best.user_attrs["best_iter"]),
        "learning_rate": FINAL_LR,
        "subsample_freq": 1,
        "random_state": 42,
        "n_jobs": -1,
    }
    logger.info("  %s: log-loss validación base(lr=0.1)=%.5f → mejor=%.5f | %d árboles | %.0fs",
                name, base_value, best.value, final["n_estimators"], time.time() - t0)
    return {"params": final, "valid_logloss_base_trial": base_value, "valid_logloss_best": best.value,
            "n_train_rows": len(tr), "n_valid_rows": len(va)}


# ---------------------------------------------------------------------------
# Evaluación OOF
# ---------------------------------------------------------------------------


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"q": q, "y": y, "p": p}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((d["n"] * (d["y"] - d["p"]).abs()).sum() / d["n"].sum())


def submodel_metrics(probs: dict, labels: dict) -> dict:
    out = {}
    for name, _, _, n_classes in chain.SUBMODELS:
        y, mask = labels[name]
        y = y[mask].astype(int)
        p = probs[name][mask]
        if n_classes > 2:
            out[name] = {"log_loss": float(log_loss(y, p, labels=list(range(n_classes)))), "ece": None}
        else:
            out[name] = {"log_loss": float(log_loss(y, p[:, 0])), "ece": ece(y, p[:, 0])}
    return out


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['r']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_delta(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] · P(mejor)={d['p_model_better']:.0%}"


def run(n_trials: int = N_TRIALS) -> dict:
    global N_TRIALS
    N_TRIALS = n_trials
    df = chain.load_data()
    masks = chain.universe_masks(df)

    logger.info("Búsqueda de hiperparámetros (%d ensayos por sub-modelo)", N_TRIALS)
    search = {name: tune_submodel(df, name, masks[universe], target, n_classes)
              for name, universe, target, n_classes in chain.SUBMODELS}
    candidate = {name: s["params"] for name, s in search.items()}
    CANDIDATE_PATH.write_text(json.dumps(candidate, indent=2), encoding="utf-8")

    base_params = {name: chain.LGB_PARAMS for name, *_ in chain.SUBMODELS}
    logger.info("OOF con parámetros base...")
    oof_base, probs_base, labels = neutral_mod.generate_oof(refresh=True, params=base_params, out_path=BASE_OOF_PATH, return_probs=True)
    logger.info("OOF con parámetros ajustados...")
    oof_tuned, probs_tuned, _ = neutral_mod.generate_oof(refresh=True, params=candidate, out_path=TUNED_OOF_PATH, return_probs=True)

    m_base, m_tuned = submodel_metrics(probs_base, labels), submodel_metrics(probs_tuned, labels)

    t_base, _ = neutral_mod.build_official_table(oof_base)
    t_tuned, eb_tuned = neutral_mod.build_official_table(oof_tuned)
    both = t_base[["pitcher_anon_id", "year", "n_pitches", "pitching_plus"]].rename(columns={"pitching_plus": "pp_base"}).merge(
        t_tuned[["pitcher_anon_id", "year", "pitching_plus"]].rename(columns={"pitching_plus": "pp_tuned"}), on=["pitcher_anon_id", "year"]
    )
    adjusted = neutral_mod.park_adjusted_test(both, scores=("pp_base", "pp_tuned"), reference="pp_base")
    bench_table, _ = benchmark.build_metric_table(extra_models={"pp_base": both, "pp_tuned": both})
    bench = {
        mp: benchmark.correlation_suite(benchmark.build_pairs(bench_table, mp),
                                        model_metrics={"pp_base": "oficial base", "pp_tuned": "oficial ajustado"},
                                        references={"pp_tuned": "pp_base"})
        for mp in benchmark.MIN_PITCHES_LEVELS
    }
    rank_corr = float(both["pp_base"].corr(both["pp_tuned"], method="spearman"))

    crit1 = all(m_tuned[n]["log_loss"] < m_base[n]["log_loss"] for n in m_base)
    crit2 = all(m_tuned[n]["ece"] is None or m_tuned[n]["ece"] <= m_base[n]["ece"] + ECE_TOLERANCE for n in m_base)
    crit3 = all(adjusted[mp]["delta_vs_reference"]["pp_tuned"]["p_model_better"] > 0.10 for mp in adjusted)
    accepted = crit1 and crit2 and crit3
    logger.info("Criterios: log-loss=%s calibración=%s benchmark=%s → %s", crit1, crit2, crit3, "ADOPTAR" if accepted else "NO adoptar")

    lines = [
        "# Ajuste de hiperparámetros — Pitching+ v3",
        "",
        "Generado por `src/models/tuning.py`. Diseño y criterio de adopción declarados en el docstring antes de correr.",
        "",
        "## Búsqueda (validación por pitcher, learning_rate 0.1)",
        "",
        "| sub-modelo | filas ajuste | filas validación | log-loss config. base | log-loss mejor | parámetros elegidos |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, s in search.items():
        p = s["params"]
        desc = (f"hojas={p['num_leaves']}, min_hijos={p['min_child_samples']}, λ={p['reg_lambda']:.3g}, α={p['reg_alpha']:.3g}, "
                f"col={p['colsample_bytree']:.2f}, sub={p['subsample']:.2f}, árboles={p['n_estimators']}")
        lines.append(f"| {name} | {s['n_train_rows']:,} | {s['n_valid_rows']:,} | {s['valid_logloss_base_trial']:.5f} | {s['valid_logloss_best']:.5f} | {desc} |")
    lines += [
        "",
        "## Criterio 1 y 2: log-loss y calibración out-of-fold (5 folds por pitcher)",
        "",
        "| sub-modelo | log-loss base | log-loss ajustado | Δ | ECE base | ECE ajustado |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n in m_base:
        b, t = m_base[n], m_tuned[n]
        e_b = f"{b['ece']:.4f}" if b["ece"] is not None else "—"
        e_t = f"{t['ece']:.4f}" if t["ece"] is not None else "—"
        lines.append(f"| {n} | {b['log_loss']:.5f} | {t['log_loss']:.5f} | {t['log_loss'] - b['log_loss']:+.5f} | {e_b} | {e_t} |")
    lines += [
        "",
        "## Criterio 3: benchmark del índice oficial (neutral + shrinkage)",
        "",
        "| mín. pitches | carreras ajustadas t+1: base | ajustado | Δ |",
        "|---|---|---|---|",
    ]
    for mp, res in adjusted.items():
        lines.append(f"| {mp} | {_fmt(res['r']['pp_base'])} | {_fmt(res['r']['pp_tuned'])} | {_fmt_delta(res['delta_vs_reference']['pp_tuned'])} |")
    lines += ["", "| mín. pitches | estabilidad base | estabilidad ajustado | carreras observadas t+1: base | ajustado |", "|---|---|---|---|---|"]
    for mp, res in bench.items():
        lines.append(f"| {mp} | {_fmt(res['stability']['pp_base'])} | {_fmt(res['stability']['pp_tuned'])} | "
                     f"{_fmt(res['predictive']['pp_base']['obs_rv_prevented'])} | {_fmt(res['predictive']['pp_tuned']['obs_rv_prevented'])} |")
    lines += [
        "",
        f"Correlación de rangos (Spearman) entre el índice base y el ajustado: **{rank_corr:.3f}**.",
        "",
        "## Decisión según el criterio",
        "",
        f"- Log-loss OOF mejora en los 5 sub-modelos: **{'sí' if crit1 else 'no'}**",
        f"- Calibración no empeora (tolerancia {ECE_TOLERANCE}): **{'sí' if crit2 else 'no'}**",
        f"- Benchmark no empeora (P(peor) < 90%): **{'sí' if crit3 else 'no'}**",
        f"- **{'Se adopta' if accepted else 'No se adopta'}.**",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    payload = {"search": search, "oof_base": m_base, "oof_tuned": m_tuned, "park_adjusted": {str(k): v for k, v in adjusted.items()},
               "benchmark": {str(k): v for k, v in bench.items()}, "rank_corr": rank_corr,
               "criteria": {"logloss": crit1, "calibration": crit2, "benchmark": crit3, "accepted": accepted},
               "eb_params_tuned": eb_tuned}
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


def activate() -> None:
    """Hace oficial el candidato: los modelos finales y el índice neutral pasan a usar estos parámetros."""
    shutil.copyfile(CANDIDATE_PATH, chain.TUNED_PARAMS_PATH)
    logger.info("Parámetros activados: %s", chain.TUNED_PARAMS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--activate", action="store_true", help="copiar el candidato como parámetros oficiales")
    args = parser.parse_args()
    if args.activate:
        activate()
        return
    run(n_trials=args.trials)


if __name__ == "__main__":
    main()
