"""Benchmark año-contra-año: la vara para medir si un cambio de modelo mejora algo.

Un indice tipo "+" (Stuff+, Pitching+) no se juzga por su AUC pitch a pitch,
sino por dos propiedades a nivel pitcher-temporada:

1. Estabilidad   : corr(metrica en t, misma metrica en t+1) para el mismo
                   pitcher. Una metrica de "habilidad" debe repetirse de un
                   año al siguiente; si no se repite, esta midiendo ruido.
2. Poder predictivo: corr(metrica en t, resultado OBSERVADO en t+1). La
                   pregunta que le importa a un cuerpo tecnico: ¿el numero de
                   este año me dice como le va a ir el año que entra?

Ambas se comparan contra baselines "tradicionales" calculados con resultados
observados en t (whiff%, K%, K-BB%, CSW%, run value observado). Si un modelo
no predice t+1 mejor que el K-BB% observado de t, no esta aportando nada.

Pares de temporadas: 2024->2025 y 2025->2026 (2026 es parcial). Se exige un
minimo de pitches en AMBAS temporadas del par. Los intervalos de confianza
salen de un bootstrap por pitcher (se remuestrean pitchers, no filas, porque
un mismo pitcher puede aparecer en los dos pares). Las diferencias entre
metricas se calculan con bootstrap PAREADO (mismo remuestreo para ambas), que
es lo que permite decir si una version nueva es mejor que la anterior.

Este modulo tambien construye por primera vez un score Stuff+ v1 en escala
100+10z a nivel pitcher-temporada, a partir de predicciones out-of-fold del
modelo de is_whiff con exactamente las mismas features e hiperparametros de
src/models/stuff_plus.py (el v1 original solo existia como probabilidad por
pitch y con predicciones in-sample para los pitchers de train).

Salidas:
- data/processed/stuff_plus_v1_oof_pitches.parquet  (prediccion OOF por pitch)
- reports/benchmark/<version>.json                  (resultados para comparar)
- reports/benchmark_baseline_v1.md                  (reporte legible)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models import stuff_plus

ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits" / "splits.parquet"
PITCHING_PLUS_OOF_PATH = ROOT / "data" / "processed" / "pitching_plus_scores_oof.parquet"
STUFF_OOF_PITCHES_PATH = ROOT / "data" / "processed" / "stuff_plus_v1_oof_pitches.parquet"
BENCHMARK_DIR = ROOT / "reports" / "benchmark"
REPORT_PATH = ROOT / "reports" / "benchmark_baseline_v1.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_FOLDS = 5  # mismo esquema que src/models/pitching_plus_oof.py
SEASON_PAIRS = [(2024, 2025), (2025, 2026)]
MIN_PITCHES_LEVELS = [100, 300]  # 100 = convencion del proyecto; 300 = muestra "confiable"
N_BOOT = 2000
BOOT_SEED = 42

# Metricas a evaluar. Todas orientadas "mas alto = mejor para el pitcher".
MODEL_METRICS = {
    # Modelo predictivo de whiff: sus features incluyen ubicación implícita (VAA, ángulos de salida) y la categoría
    # de altitud (`altitude_level`); auditoría 2026-10-05, P1.4.
    "stuff_plus_v1": "Stuff+ v1 (predictivo; con ubicación implícita y altitud)",
    "pitching_plus_v1": "Pitching+ v1 (OOF, oficial)",
}
BASELINE_METRICS = {
    "obs_whiff_rate": "Whiff% observado (whiffs/pitch)",
    "obs_csw": "CSW% observado (strikes cantados + whiffs)/pitch",
    "obs_k_pct": "K% observado (ponches/PA)",
    "obs_k_minus_bb": "K-BB% observado",
    "obs_rv_prevented": "Run value observado (carreras evitadas/100 pitches)",
}
# Resultados observados en t+1 contra los que se mide el poder predictivo.
OUTCOMES = {
    "obs_rv_prevented": "Carreras evitadas/100 pitches (t+1)",
    "obs_k_minus_bb": "K-BB% (t+1)",
    "obs_k_pct": "K% (t+1)",
    "obs_whiff_rate": "Whiff% (t+1)",
}
PRIMARY_OUTCOME = "obs_rv_prevented"


# ---------------------------------------------------------------------------
# Resultados observados por pitcher-temporada
# ---------------------------------------------------------------------------


def load_pitches() -> pd.DataFrame:
    cols = [
        "PitchUID", "pitcher_anon_id", "year", "PitchCall", "KorBB",
        "is_whiff", "is_called_strike", "run_value",
    ]
    df = pd.read_parquet(TARGETS_PATH, columns=cols)
    splits = pd.read_parquet(SPLITS_PATH, columns=["PitchUID", "pitcher_holdout"])
    df = df.merge(splits, on="PitchUID", how="inner")
    return df.loc[df["pitcher_holdout"] != "excluded"].drop(columns="pitcher_holdout")


def observed_by_pitcher_season(df: pd.DataFrame) -> pd.DataFrame:
    """Estadisticas observadas (no modeladas) por pitcher-temporada.

    PA terminal = ponche, base por bolas, bola en juego o golpeado. Se usa
    PitchCall para el golpeado porque la columna is_hit_by_pitch viene en 0
    en todo el dataset (2,244 pitches con PitchCall == 'HitByPitch').
    """
    d = df.assign(
        is_k=(df["KorBB"] == "Strikeout").astype(int),
        is_bb=(df["KorBB"] == "Walk").astype(int),
        is_pa_end=(
            df["KorBB"].isin(["Strikeout", "Walk"]) | df["PitchCall"].isin(["InPlay", "HitByPitch"])
        ).astype(int),
        csw=(df["is_whiff"].astype(float) + df["is_called_strike"].astype(float)),
    )
    agg = d.groupby(["pitcher_anon_id", "year"]).agg(
        n_pitches=("year", "size"),
        n_pa=("is_pa_end", "sum"),
        n_k=("is_k", "sum"),
        n_bb=("is_bb", "sum"),
        obs_whiff_rate=("is_whiff", "mean"),
        obs_csw=("csw", "mean"),
        rv_sum=("run_value", "sum"),
        n_rv=("run_value", "count"),
    ).reset_index()

    pa = agg["n_pa"].where(agg["n_pa"] > 0)
    agg["obs_k_pct"] = agg["n_k"] / pa
    agg["obs_k_minus_bb"] = (agg["n_k"] - agg["n_bb"]) / pa
    # run_value positivo = carreras permitidas -> se invierte el signo
    agg["obs_rv_prevented"] = -100 * agg["rv_sum"] / agg["n_rv"].where(agg["n_rv"] > 0)
    return agg.drop(columns=["n_k", "n_bb", "rv_sum", "n_rv"])


# ---------------------------------------------------------------------------
# Stuff+ v1 out-of-fold
# ---------------------------------------------------------------------------


def stuff_plus_v1_oof_pitches(refresh: bool = False) -> pd.DataFrame:
    """Prediccion OOF de is_whiff con el modelo Stuff+ v1 (mismas features y params)."""
    if STUFF_OOF_PITCHES_PATH.exists() and not refresh:
        logger.info("Usando predicciones OOF de Stuff+ v1 en cache: %s", STUFF_OOF_PITCHES_PATH.name)
        return pd.read_parquet(STUFF_OOF_PITCHES_PATH)

    df = stuff_plus.load_data()
    years = pd.read_parquet(TARGETS_PATH, columns=["PitchUID", "year"])
    df = df.merge(years, on="PitchUID", how="inner").reset_index(drop=True)
    df = df.dropna(subset=["is_whiff"]).reset_index(drop=True)
    logger.info("Stuff+ v1 OOF: %s pitches, %s pitchers", f"{len(df):,}", f"{df['pitcher_anon_id'].nunique():,}")

    X = df[stuff_plus.FEATURE_COLUMNS]
    y = df["is_whiff"].astype(int)
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS)
    t0 = time.time()
    for fold_i, (tr, ho) in enumerate(gkf.split(df, groups=df["pitcher_anon_id"]), start=1):
        model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **stuff_plus.LGB_PARAMS_COMMON)
        model.fit(X.iloc[tr], y.iloc[tr], categorical_feature=stuff_plus.CATEGORICAL_FEATURES)
        oof[ho] = model.predict_proba(X.iloc[ho])[:, 1]
        logger.info("  fold %d/%d listo (%.0fs)", fold_i, N_FOLDS, time.time() - t0)

    out = df[["PitchUID", "pitcher_anon_id", "year", "is_whiff"]].copy()
    out["is_whiff"] = y.to_numpy()
    out["pred_is_whiff"] = oof
    out.to_parquet(STUFF_OOF_PITCHES_PATH, index=False)
    return out


def scale_plus(values: pd.Series, year: pd.Series) -> pd.Series:
    """100 + 10*z, con z calculado dentro de cada temporada (misma convencion que Pitching+)."""
    z = values.groupby(year).transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    return 100 + 10 * z


def pitch_level_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Discriminacion (AUC) + calibracion (log-loss, Brier, ECE por deciles)."""
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    cal = (
        pd.DataFrame({"bin": bins, "p": p, "y": y})
        .groupby("bin")
        .agg(pred_mean=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
    )
    ece = float((cal["n"] * (cal["pred_mean"] - cal["obs_rate"]).abs()).sum() / cal["n"].sum())
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece,
        "base_rate": float(np.mean(y)),
        "calibration_table": cal.reset_index().to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Tabla de metricas por pitcher-temporada
# ---------------------------------------------------------------------------


def build_metric_table(
    refresh: bool = False, extra_models: dict[str, pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Tabla pitcher-temporada con baselines observados + scores de modelos.

    `extra_models`: {nombre: DataFrame con pitcher_anon_id, year y una columna
    `nombre`} para evaluar versiones nuevas junto a las v1.
    """
    obs = observed_by_pitcher_season(load_pitches())

    stuff_pitches = stuff_plus_v1_oof_pitches(refresh=refresh)
    pitch_metrics = {
        "stuff_plus_v1": pitch_level_metrics(
            stuff_pitches["is_whiff"].to_numpy(), stuff_pitches["pred_is_whiff"].to_numpy()
        )
    }
    stuff = stuff_pitches.groupby(["pitcher_anon_id", "year"])["pred_is_whiff"].mean().reset_index()
    stuff["stuff_plus_v1"] = scale_plus(stuff["pred_is_whiff"], stuff["year"])

    pplus = pd.read_parquet(PITCHING_PLUS_OOF_PATH, columns=["pitcher_anon_id", "year", "pitching_plus"])
    pplus = pplus.rename(columns={"pitching_plus": "pitching_plus_v1"})

    table = (
        obs.merge(stuff[["pitcher_anon_id", "year", "stuff_plus_v1"]], on=["pitcher_anon_id", "year"], how="left")
        .merge(pplus, on=["pitcher_anon_id", "year"], how="left")
    )
    for name, scores in (extra_models or {}).items():
        table = table.merge(scores[["pitcher_anon_id", "year", name]], on=["pitcher_anon_id", "year"], how="left")
    return table, pitch_metrics


def build_pairs(table: pd.DataFrame, min_pitches: int) -> pd.DataFrame:
    """Une cada pitcher-temporada t con la t+1 del mismo pitcher (sufijos _t y _t1)."""
    frames = []
    for y0, y1 in SEASON_PAIRS:
        a = table.loc[(table["year"] == y0) & (table["n_pitches"] >= min_pitches)]
        b = table.loc[(table["year"] == y1) & (table["n_pitches"] >= min_pitches)]
        pair = a.merge(b, on="pitcher_anon_id", suffixes=("_t", "_t1"))
        pair["pair"] = f"{y0}->{y1}"
        frames.append(pair)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Correlaciones con bootstrap por pitcher
# ---------------------------------------------------------------------------


def _bootstrap_indices(pitchers: pd.Series, n_boot: int, seed: int) -> list[np.ndarray]:
    """Remuestreo por pitcher: cada replica toma pitchers con reemplazo y TODAS sus filas."""
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(pitchers)
    rows_by_pitcher = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    reps = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(uniques), size=len(uniques))
        reps.append(np.concatenate([rows_by_pitcher[k] for k in chosen]))
    return reps


def _corr_matrix(m: np.ndarray) -> np.ndarray:
    return np.corrcoef(m, rowvar=False)


def correlation_suite(
    pairs: pd.DataFrame,
    model_metrics: dict[str, str] = MODEL_METRICS,
    references: dict[str, str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Estabilidad y poder predictivo de cada metrica, con IC 95% y deltas pareados.

    `references`: {modelo_nuevo: modelo_referencia} para calcular, con el mismo
    bootstrap pareado, si la version nueva supera a la anterior.
    """
    metrics = list(model_metrics) + list(BASELINE_METRICS)
    cols_t = [f"{m}_t" for m in metrics]
    cols_stab = [f"{m}_t1" for m in metrics]
    cols_out = [f"{o}_t1" for o in OUTCOMES]
    all_cols = list(dict.fromkeys(cols_t + cols_stab + cols_out))
    pos = {c: i for i, c in enumerate(all_cols)}

    data = pairs[all_cols].astype(float)
    keep = data.notna().all(axis=1).to_numpy()
    m = data.to_numpy()[keep]
    pitchers = pairs.loc[keep, "pitcher_anon_id"].reset_index(drop=True)

    def extract(c: np.ndarray) -> dict:
        res = {"stability": {}, "predictive": {}}
        for mt in metrics:
            res["stability"][mt] = c[pos[f"{mt}_t"], pos[f"{mt}_t1"]]
            res["predictive"][mt] = {o: c[pos[f"{mt}_t"], pos[f"{o}_t1"]] for o in OUTCOMES}
        return res

    point = extract(_corr_matrix(m))
    boots = [extract(_corr_matrix(m[idx])) for idx in _bootstrap_indices(pitchers, n_boot, seed)]

    def ci(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    out = {"n_pairs": int(keep.sum()), "n_pitchers": int(pitchers.nunique()), "stability": {}, "predictive": {}}
    for mt in metrics:
        out["stability"][mt] = {
            "r": float(point["stability"][mt]),
            "ci95": ci([b["stability"][mt] for b in boots]),
        }
        out["predictive"][mt] = {
            o: {
                "r": float(point["predictive"][mt][o]),
                "ci95": ci([b["predictive"][mt][o] for b in boots]),
            }
            for o in OUTCOMES
        }

    # Deltas pareados: cada modelo vs. el mejor baseline observado (por punto estimado)
    deltas = {}
    for o in OUTCOMES:
        best_base = max(BASELINE_METRICS, key=lambda b: point["predictive"][b][o])
        for mt in model_metrics:
            diffs = [b["predictive"][mt][o] - b["predictive"][best_base][o] for b in boots]
            deltas[f"{mt}|{o}"] = {
                "vs": best_base,
                "delta_r": float(point["predictive"][mt][o] - point["predictive"][best_base][o]),
                "ci95": ci(diffs),
                "p_model_better": float(np.mean(np.array(diffs) > 0)),
            }
    out["deltas_vs_best_baseline"] = deltas

    vs_ref = {}
    for mt, ref in (references or {}).items():
        stab = [b["stability"][mt] - b["stability"][ref] for b in boots]
        vs_ref[f"{mt}|stability"] = {
            "vs": ref,
            "delta_r": float(point["stability"][mt] - point["stability"][ref]),
            "ci95": ci(stab),
            "p_model_better": float(np.mean(np.array(stab) > 0)),
        }
        for o in OUTCOMES:
            diffs = [b["predictive"][mt][o] - b["predictive"][ref][o] for b in boots]
            vs_ref[f"{mt}|{o}"] = {
                "vs": ref,
                "delta_r": float(point["predictive"][mt][o] - point["predictive"][ref][o]),
                "ci95": ci(diffs),
                "p_model_better": float(np.mean(np.array(diffs) > 0)),
            }
    out["deltas_vs_reference"] = vs_ref
    return out


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def _fmt_r(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['r']:.3f} [{lo:.3f}, {hi:.3f}]"


def format_report(results: dict, pitch_metrics: dict, table: pd.DataFrame) -> str:
    labels = {**MODEL_METRICS, **BASELINE_METRICS}
    metrics = list(MODEL_METRICS) + list(BASELINE_METRICS)
    lines = [
        "# Benchmark año-contra-año — línea base v1",
        "",
        "Referencia contra la cual se medirá cada mejora futura de los modelos "
        "(features, targets, contrafactual, tuning). Generado por `src/validation/benchmark.py`; "
        f"resultados crudos en `reports/benchmark/baseline_v1.json`.",
        "",
        "## Qué se mide",
        "",
        "- **Estabilidad:** correlación de la métrica de un pitcher en la temporada *t* con la misma métrica en *t+1*. "
        "Una métrica de habilidad debe repetirse; si no se repite, mide ruido.",
        "- **Poder predictivo:** correlación de la métrica en *t* con resultados **observados** en *t+1*. "
        f"Resultado principal: `{PRIMARY_OUTCOME}` (carreras evitadas por cada 100 pitches, a partir del run value RE24 observado).",
        "- **Baselines:** estadísticas tradicionales observadas en *t* (whiff%, CSW%, K%, K-BB%, run value). "
        "Un modelo que no supera al K-BB% observado del año anterior no está aportando información nueva.",
        f"- Pares de temporadas: {', '.join(f'{a}→{b}' for a, b in SEASON_PAIRS)} (2026 es parcial). "
        f"Correlación de Pearson; IC 95% por bootstrap de {N_BOOT:,} réplicas remuestreando **pitchers** (seed={BOOT_SEED}).",
        "- Stuff+ v1 se escala aquí por primera vez a `100 + 10·z` por temporada, con predicciones out-of-fold "
        "(GroupKFold por pitcher, k=5) del mismo modelo de `is_whiff` de `src/models/stuff_plus.py`.",
        "",
        "## Cobertura de pares",
        "",
    ]
    for mp, res in results.items():
        lines.append(f"- Mínimo {mp} pitches en ambas temporadas: **{res['n_pairs']:,} pares** de {res['n_pitchers']:,} pitchers.")
    lines += [
        "",
        "Solo ~280 pitchers aparecen en dos temporadas consecutivas (de ~600 por temporada) — la muestra de pares es chica, "
        "y eso se refleja en intervalos de confianza anchos.",
        "",
    ]

    for mp, res in results.items():
        lines += [f"## Resultados — mínimo {mp} pitches por temporada", "", "### Estabilidad (r de t con t+1)", ""]
        lines += ["| métrica | tipo | r [IC 95%] |", "|---|---|---|"]
        for mt in metrics:
            kind = "modelo" if mt in MODEL_METRICS else "baseline observado"
            lines.append(f"| {labels[mt]} | {kind} | {_fmt_r(res['stability'][mt])} |")
        lines += ["", "### Poder predictivo (r de la métrica en t con el resultado observado en t+1)", ""]
        header = "| métrica | " + " | ".join(OUTCOMES.values()) + " |"
        lines += [header, "|---|" + "---|" * len(OUTCOMES)]
        for mt in metrics:
            row = " | ".join(_fmt_r(res["predictive"][mt][o]) for o in OUTCOMES)
            lines.append(f"| {labels[mt]} | {row} |")
        lines += ["", "### Modelos vs. el mejor baseline observado (delta de r, bootstrap pareado)", ""]
        lines += ["| modelo | resultado t+1 | mejor baseline | Δr [IC 95%] | P(modelo mejor) |", "|---|---|---|---|---|"]
        for key, d in res["deltas_vs_best_baseline"].items():
            mt, o = key.split("|")
            lo, hi = d["ci95"]
            lines.append(
                f"| {labels[mt]} | {OUTCOMES[o]} | {labels[d['vs']]} | {d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {d['p_model_better']:.0%} |"
            )
        lines.append("")

    pm = pitch_metrics["stuff_plus_v1"]
    lines += [
        "## Métricas pitch a pitch — Stuff+ v1 (OOF, `is_whiff`)",
        "",
        "| AUC | log-loss | Brier | ECE (calibración) | tasa base |",
        "|---:|---:|---:|---:|---:|",
        f"| {pm['auc']:.4f} | {pm['log_loss']:.4f} | {pm['brier']:.4f} | {pm['ece']:.4f} | {pm['base_rate']:.4f} |",
        "",
        "ECE = error de calibración esperado (promedio ponderado de |probabilidad predicha − tasa observada| por decil). "
        "Importa porque Pitching+ y Stuff+ promedian probabilidades: si están descalibradas, el promedio se sesga.",
        "",
        "| decil | predicho | observado | n |",
        "|---:|---:|---:|---:|",
    ]
    for row in pm["calibration_table"]:
        lines.append(f"| {int(row['bin']) + 1} | {row['pred_mean']:.4f} | {row['obs_rate']:.4f} | {row['n']:,} |")

    stuff_dist = table["stuff_plus_v1"].describe()
    lines += [
        "",
        "## Distribución del Stuff+ v1 (pitcher-temporada, escala 100+10z)",
        "",
        f"n={int(stuff_dist['count']):,} · media={stuff_dist['mean']:.2f} · std={stuff_dist['std']:.2f} · "
        f"min={stuff_dist['min']:.2f} · max={stuff_dist['max']:.2f}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(version: str = "baseline_v1", refresh: bool = False) -> dict:
    table, pitch_metrics = build_metric_table(refresh=refresh)
    results = {}
    for mp in MIN_PITCHES_LEVELS:
        pairs = build_pairs(table, mp)
        results[mp] = correlation_suite(pairs)
        logger.info(
            "min %d pitches: %d pares | estabilidad Stuff+ %.3f, Pitching+ %.3f | pred. RV t+1: Stuff+ %.3f, Pitching+ %.3f",
            mp, results[mp]["n_pairs"],
            results[mp]["stability"]["stuff_plus_v1"]["r"], results[mp]["stability"]["pitching_plus_v1"]["r"],
            results[mp]["predictive"]["stuff_plus_v1"][PRIMARY_OUTCOME]["r"],
            results[mp]["predictive"]["pitching_plus_v1"][PRIMARY_OUTCOME]["r"],
        )

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "season_pairs": SEASON_PAIRS,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "results_by_min_pitches": {str(k): v for k, v in results.items()},
        "pitch_level": {k: {kk: vv for kk, vv in v.items() if kk != "calibration_table"} for k, v in pitch_metrics.items()},
    }
    (BENCHMARK_DIR / f"{version}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(format_report(results, pitch_metrics, table), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular las predicciones OOF de Stuff+ v1")
    args = parser.parse_args()
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
