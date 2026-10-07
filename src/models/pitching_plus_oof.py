"""Predicciones out-of-fold (K-fold, k=5) para los 8 modelos de Pitching+.

Resuelve la limitacion documentada en reports/pitching_plus_score.md: las
predicciones usadas para el score compuesto se generaban con modelos
entrenados sobre pitcher_holdout, asi que los pitchers que estuvieron en TRAIN
tenian predicciones ligeramente optimistas (el modelo vio sus propios pitches
al ajustarse) frente a los de TEST, que eran genuinamente out-of-sample.

Aqui se hace K-fold agrupado POR PITCHER (`GroupKFold`, no K-fold plano): cada
fold deja un ~20% de los PITCHERS completamente afuera del entrenamiento de
ese fold, y se predicen sus pitches con el modelo que nunca los vio. Se
agrupa por pitcher (no solo por pitch) a proposito — si se hiciera K-fold
plano, un pitcher podria tener parte de sus pitches en el fold de prediccion
y el resto en el train de ese mismo fold, y el modelo seguiria pudiendo
ajustarse a patrones de ESE pitcher especifico a traves de sus otros pitches.
Agrupando por pitcher, CADA pitcher (y todos sus pitches) queda completamente
afuera del entrenamiento del fold que lo predice — la misma logica de
`pitcher_holdout`, aplicada 5 veces con distintos grupos held-out en vez de
una sola vez.

Con esto, TODOS los pitchers (no solo los que cayeron en el test de
pitcher_holdout) tienen predicciones genuinamente out-of-sample, y el score
de Pitching+ deja de tener el sesgo train-vs-test documentado en la version
anterior.

Reutiliza la misma metodologia de agregacion/z-score/pesos de
src/models/pitching_plus_score.py — solo cambia el ORIGEN de las
predicciones (OOF en vez de un unico modelo train/test).
"""

import logging
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models.pitching_plus import (
    BINARY_TARGETS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LGB_PARAMS_COMMON,
    REGRESSION_TARGETS,
    load_data,
)
from src.models.pitching_plus_score import (
    MIN_PITCHES_FOR_RANKING,
    PITCHING_PLUS_WEIGHTS,
    aggregate_by_pitcher_season,
    build_composite_score,
)

ROOT = Path(__file__).resolve().parents[2]
SCORES_OOF_PATH = ROOT / "data" / "processed" / "pitching_plus_scores_oof.parquet"
SCORES_INSAMPLE_PATH = ROOT / "data" / "processed" / "pitching_plus_scores.parquet"
REPORT_PATH = ROOT / "reports" / "pitching_plus_score_oof.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ALL_TARGETS = BINARY_TARGETS + REGRESSION_TARGETS
N_FOLDS = 5
OOF_SEED = 42


# ---------------------------------------------------------------------------
# Predicciones out-of-fold
# ---------------------------------------------------------------------------


def generate_oof_predictions(df: pd.DataFrame) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=N_FOLDS)
    X_all = df[FEATURE_COLUMNS]

    oof = pd.DataFrame(index=df.index, columns=ALL_TARGETS, dtype=float)

    for fold_i, (train_pos, held_pos) in enumerate(gkf.split(df, groups=df["pitcher_anon_id"]), start=1):
        train_idx = df.index[train_pos]
        held_idx = df.index[held_pos]
        n_pitchers_held = df.loc[held_idx, "pitcher_anon_id"].nunique()
        logger.info(
            "Fold %d/%d: train=%s pitches, held-out=%s pitches (%s pitchers nunca vistos en este fold)",
            fold_i, N_FOLDS, f"{len(train_idx):,}", f"{len(held_idx):,}", f"{n_pitchers_held:,}",
        )

        for target in ALL_TARGETS:
            is_binary = target in BINARY_TARGETS
            train_sub = df.loc[train_idx].dropna(subset=[target])
            X_train = train_sub[FEATURE_COLUMNS]
            y_train = train_sub[target]

            if is_binary:
                model = lgb.LGBMClassifier(objective="binary", verbosity=-1, **LGB_PARAMS_COMMON)
            else:
                model = lgb.LGBMRegressor(objective="regression", verbosity=-1, **LGB_PARAMS_COMMON)
            model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

            X_held = X_all.loc[held_idx]
            if is_binary:
                oof.loc[held_idx, target] = model.predict_proba(X_held)[:, 1]
            else:
                oof.loc[held_idx, target] = model.predict(X_held)

    return oof


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> tuple[pd.DataFrame, dict]:
    df = load_data()
    logger.info("Filas validas: %s | pitchers: %s", f"{len(df):,}", f"{df['pitcher_anon_id'].nunique():,}")

    t0 = time.time()
    oof_preds = generate_oof_predictions(df)
    logger.info("Predicciones OOF generadas en %.1fs", time.time() - t0)

    n_null_oof = int(oof_preds.isna().sum().sum())
    logger.info("Nulos en oof_preds (esperado 0 — se predice cada fila held-out sin importar si su target real era nulo): %d", n_null_oof)

    agg = aggregate_by_pitcher_season(df, oof_preds)
    logger.info("Pitcher-temporadas: %s", f"{len(agg):,}")

    scored, z_stats = build_composite_score(agg)

    dist = scored["pitching_plus"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])

    ranked = scored.loc[scored["n_pitches"] >= MIN_PITCHES_FOR_RANKING].sort_values(
        "pitching_plus", ascending=False
    )
    top10 = ranked.head(10)
    bottom10 = ranked.tail(10).sort_values("pitching_plus")

    # --- comparacion contra la version in-sample anterior ---
    old_scored = pd.read_parquet(SCORES_INSAMPLE_PATH)
    merged = scored.merge(
        old_scored[["pitcher_anon_id", "year", "pitching_plus"]],
        on=["pitcher_anon_id", "year"],
        suffixes=("_oof", "_insample"),
    )
    merged["delta"] = merged["pitching_plus_oof"] - merged["pitching_plus_insample"]
    corr = merged["pitching_plus_oof"].corr(merged["pitching_plus_insample"])

    old_ranked = old_scored.loc[old_scored["n_pitches"] >= MIN_PITCHES_FOR_RANKING].sort_values(
        "pitching_plus", ascending=False
    )
    old_top10_ids = set(zip(old_ranked.head(10)["pitcher_anon_id"], old_ranked.head(10)["year"]))
    old_bottom10_ids = set(
        zip(old_ranked.tail(10).sort_values("pitching_plus")["pitcher_anon_id"], old_ranked.tail(10)["year"])
    )
    new_top10_ids = set(zip(top10["pitcher_anon_id"], top10["year"]))
    new_bottom10_ids = set(zip(bottom10["pitcher_anon_id"], bottom10["year"]))

    report = {
        "n_pitcher_seasons": len(scored),
        "n_pitcher_seasons_ranked": len(ranked),
        "distribution": dist,
        "top10": top10,
        "bottom10": bottom10,
        "comparison": {
            "corr_oof_vs_insample": float(corr),
            "mean_abs_delta": float(merged["delta"].abs().mean()),
            "max_abs_delta": float(merged["delta"].abs().max()),
            "top10_overlap": len(old_top10_ids & new_top10_ids),
            "bottom10_overlap": len(old_bottom10_ids & new_bottom10_ids),
            "merged": merged,
        },
    }
    return scored, report


def format_report(report: dict) -> str:
    dist = report["distribution"]
    top10 = report["top10"]
    bottom10 = report["bottom10"]
    comp = report["comparison"]

    z_cols = [f"z_{t}" for t in ALL_TARGETS]
    display_cols = ["pitcher_anon_id", "year", "n_pitches", "pitching_plus"] + z_cols

    lines = [
        "# Pitching+ v1 — score compuesto, predicciones OUT-OF-FOLD (version oficial)",
        "",
        "**Esta es ahora la version oficial del score de Pitching+ para todos los "
        "entregables de aqui en adelante**, reemplazando a "
        "`data/processed/pitching_plus_scores.parquet` (in-sample, se conserva solo "
        "como referencia historica/auditoria, no usar para analisis nuevos).",
        "",
        f"Metodologia identica a `reports/pitching_plus_score.md` (mismos pesos, mismo "
        f"signo invertido en run_value/is_hard_contact, mismo z-score por temporada, "
        f"misma formula 100+10z) — el UNICO cambio es el origen de las predicciones: "
        f"K-fold agrupado por pitcher (k={N_FOLDS}, seed={OOF_SEED}) en vez de un solo "
        "modelo train/test. Cada pitcher queda completamente afuera del entrenamiento "
        "del fold que predice sus pitches — ningun pitcher se predice con un modelo "
        "que haya visto (aunque sea parcialmente) sus propios pitches.",
        "",
        "## Distribucion del Pitching+ (OOF)",
        "",
        f"n pitcher-temporadas: {report['n_pitcher_seasons']:,}",
        "",
        "| estadistico | valor |",
        "|---|---:|",
    ]
    for stat in ["mean", "std", "min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max"]:
        if stat in dist.index:
            lines.append(f"| {stat} | {dist[stat]:.2f} |")

    lines += [
        "",
        "## Comparacion contra la version in-sample anterior",
        "",
        f"- Correlacion entre score in-sample y score OOF (las {report['n_pitcher_seasons']:,} "
        f"pitcher-temporadas): **{comp['corr_oof_vs_insample']:.4f}**",
        f"- Cambio absoluto promedio en el score (|OOF - in-sample|): {comp['mean_abs_delta']:.2f} puntos",
        f"- Cambio absoluto maximo: {comp['max_abs_delta']:.2f} puntos",
        f"- Overlap del top-10: **{comp['top10_overlap']}/10** pitcher-temporadas se mantienen "
        "en el top-10 en ambas versiones",
        f"- Overlap del bottom-10: **{comp['bottom10_overlap']}/10** pitcher-temporadas se "
        "mantienen en el bottom-10 en ambas versiones",
        "",
    ]

    if comp["corr_oof_vs_insample"] >= 0.9 and comp["top10_overlap"] >= 7:
        lines.append(
            "**El ranking es robusto:** la correlacion es alta y el top-10/bottom-10 se "
            "mantiene mayormente igual — el sesgo in-sample que preocupaba no cambio "
            "sustancialmente el resultado, aunque sigue siendo la version metodologicamente "
            "correcta a usar de ahora en adelante."
        )
    else:
        lines.append(
            "**El ranking SI cambia de forma notable con OOF** — confirma que el sesgo "
            "in-sample (pitchers de train con predicciones optimistas) tenia un efecto real "
            "en el ranking anterior. Usar esta version OOF es lo correcto, no un detalle menor."
        )

    lines += [
        "",
        "### Pitcher-temporadas con mayor cambio (|delta| top 10)",
        "",
    ]
    biggest_movers = comp["merged"].reindex(
        comp["merged"]["delta"].abs().sort_values(ascending=False).index
    ).head(10)
    lines.append(
        biggest_movers[["pitcher_anon_id", "year", "pitching_plus_insample", "pitching_plus_oof", "delta"]]
        .round(2)
        .to_markdown(index=False)
    )

    lines += [
        "",
        f"## Ranking OOF — top 10 y bottom 10 (filtro: n_pitches >= {MIN_PITCHES_FOR_RANKING})",
        "",
        "### Top 10 (mejor Pitching+, OOF)",
        "",
    ]
    lines.append(top10[display_cols].round(3).to_markdown(index=False))
    lines += [
        "",
        "### Bottom 10 (peor Pitching+, OOF)",
        "",
    ]
    lines.append(bottom10[display_cols].round(3).to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    scored, report = run()

    SCORES_OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORES_OOF_PATH, index=False)
    logger.info("Guardado %s (%s filas) — VERSION OFICIAL", SCORES_OOF_PATH, f"{len(scored):,}")

    report_text = format_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Guardado reporte en %s", REPORT_PATH)

    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
