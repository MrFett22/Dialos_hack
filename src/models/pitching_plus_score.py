"""Pitching+ v1 — score compuesto por pitcher-temporada, combinando los 8 targets.

Construye un indice unico "Pitching+" por (pitcher_anon_id, year) a partir de
las PREDICCIONES (no los outcomes observados crudos) de los 8 modelos de
src/models/pitching_plus.py, agregadas a nivel pitcher-temporada. Se usan
predicciones del modelo (no promedios de outcomes 0/1 observados) porque ese
es justamente el punto de un indice tipo "+": suavizar el ruido pitch a pitch
con la expectativa modelada, igual que Stuff+/Pitching+ reales de MLB.

Metodologia:
1. Prediccion por pitch con los 8 modelos de pitching_plus_v1.pkl, sobre TODOS
   los pitches validos (train+test de pitcher_holdout combinados — ver
   limitacion en el reporte sobre in-sample vs out-of-sample).
2. Agregacion: promedio de cada prediccion por (pitcher_anon_id, year).
3. Ajuste de signo: run_value e is_hard_contact se invierten (mayor valor =
   peor para el pitcher), el resto se deja igual (mayor valor = mejor).
4. z-score de cada target POR TEMPORADA (year) — no pooled entre años, para
   no confundir diferencias de nivel entre temporadas con diferencias de
   habilidad.
5. Combinacion ponderada de los 8 z-scores (pesos normalizados a sumar 1,
   ver PITCHING_PLUS_WEIGHTS y su justificacion en el reporte).
6. Pitching+ = 100 + 10 * z_combinado.

Guarda la tabla pitcher-temporada en
data/processed/pitching_plus_scores.parquet y el reporte en
reports/pitching_plus_score.md.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.pitching_plus import BINARY_TARGETS, FEATURE_COLUMNS, MODEL_PATH, REGRESSION_TARGETS, load_data

ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = ROOT / "data" / "processed" / "pitching_plus_scores.parquet"
REPORT_PATH = ROOT / "reports" / "pitching_plus_score.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ALL_TARGETS = BINARY_TARGETS + REGRESSION_TARGETS

# targets donde un valor mas ALTO es peor para el pitcher -> se invierte el
# signo del z-score antes de combinar, para que en todos los componentes
# "mas alto siempre sea mejor" antes de ponderar
INVERT_SIGN = {"run_value", "is_hard_contact"}

# Pesos de combinacion (documentados/justificados en el reporte). Suman 1.00
# antes de aplicarse; se normalizan explicitamente por si se editan a mano.
PITCHING_PLUS_WEIGHTS = {
    # mayor peso: metricas mas directamente ligadas a prevencion de carreras,
    # y las que mejor explicaron el hallazgo de altitud de Semana 2 (el efecto
    # de altitud en run_value estaba ~82% mediado por menos outs, no por mas
    # contacto duro)
    "run_value": 0.20,
    "is_out_in_pa": 0.20,
    # peso medio: comando y engaño (generar swing-and-miss, robarse strikes,
    # generar persecucion fuera de zona) — el "como" detras de los outs
    "is_whiff": 0.12,
    "is_chase": 0.12,
    "is_called_strike": 0.12,
    # menor peso: calidad de contacto. is_hard_contact no mostro patron claro
    # por altitud (Semana 2/3) y el efecto dominante de altitud fue via tasa
    # de outs, no via calidad de contacto — coherente con darle menos peso
    # relativo aqui tambien
    "is_weak_contact": 0.08,
    "is_groundball": 0.08,
    "is_hard_contact": 0.08,
}
assert abs(sum(PITCHING_PLUS_WEIGHTS.values()) - 1.0) < 1e-9, "los pesos deben sumar 1.0"

MIN_PITCHES_FOR_RANKING = 100  # filtro de tamano minimo para el chequeo de cordura (no shrinkage bayesiano)


# ---------------------------------------------------------------------------
# 1-2. Predicciones por pitch + agregacion a pitcher-temporada
# ---------------------------------------------------------------------------


def predict_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    models = artifact["models"]

    X = df[FEATURE_COLUMNS]
    preds = pd.DataFrame(index=df.index)
    for target in ALL_TARGETS:
        model = models[target]
        if target in BINARY_TARGETS:
            preds[target] = model.predict_proba(X)[:, 1]
        else:
            preds[target] = model.predict(X)
    return preds


def aggregate_by_pitcher_season(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([df[["pitcher_anon_id", "year"]], preds], axis=1)
    agg = combined.groupby(["pitcher_anon_id", "year"]).agg(
        n_pitches=("year", "size"), **{t: (t, "mean") for t in ALL_TARGETS}
    )
    return agg.reset_index()


# ---------------------------------------------------------------------------
# 3-6. Ajuste de signo, z-score por temporada, combinacion, escala 100+10z
# ---------------------------------------------------------------------------


def build_composite_score(agg: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    agg = agg.copy()
    z_stats: dict[str, dict] = {}

    for target in ALL_TARGETS:
        signed = agg[target] * (-1 if target in INVERT_SIGN else 1)
        z_col = f"z_{target}"
        agg[z_col] = signed.groupby(agg["year"]).transform(lambda s: (s - s.mean()) / s.std(ddof=1))
        z_stats[target] = (
            agg.groupby("year")[target]
            .agg(["mean", "std"])
            .rename(columns={"mean": "raw_mean", "std": "raw_std"})
            .to_dict("index")
        )

    weight_sum = sum(PITCHING_PLUS_WEIGHTS.values())
    agg["combined_z"] = sum(
        agg[f"z_{t}"] * (w / weight_sum) for t, w in PITCHING_PLUS_WEIGHTS.items()
    )
    agg["pitching_plus"] = 100 + 10 * agg["combined_z"]

    return agg, z_stats


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> tuple[pd.DataFrame, dict]:
    df = load_data()
    logger.info("Filas validas: %s", f"{len(df):,}")

    preds = predict_all_targets(df)
    agg = aggregate_by_pitcher_season(df, preds)
    logger.info("Pitcher-temporadas: %s", f"{len(agg):,}")

    scored, z_stats = build_composite_score(agg)

    dist = scored["pitching_plus"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])

    ranked = scored.loc[scored["n_pitches"] >= MIN_PITCHES_FOR_RANKING].sort_values(
        "pitching_plus", ascending=False
    )
    top10 = ranked.head(10)
    bottom10 = ranked.tail(10).sort_values("pitching_plus")

    report = {
        "n_pitcher_seasons": len(scored),
        "n_pitcher_seasons_ranked": len(ranked),
        "distribution": dist,
        "top10": top10,
        "bottom10": bottom10,
        "z_stats": z_stats,
    }
    return scored, report


def format_report(report: dict) -> str:
    dist = report["distribution"]
    top10 = report["top10"]
    bottom10 = report["bottom10"]

    z_cols = [f"z_{t}" for t in ALL_TARGETS]
    display_cols = ["pitcher_anon_id", "year", "n_pitches", "pitching_plus"] + z_cols

    lines = [
        "# Pitching+ v1 — score compuesto por pitcher-temporada",
        "",
        "## Pesos de combinacion (documentados)",
        "",
        "| target | peso | nivel | justificacion |",
        "|---|---:|---|---|",
        f"| run_value | {PITCHING_PLUS_WEIGHTS['run_value']:.2f} | alto | metrica mas directa de prevencion de carreras |",
        f"| is_out_in_pa | {PITCHING_PLUS_WEIGHTS['is_out_in_pa']:.2f} | alto | ~82% del efecto de altitud en run_value (Semana 2) estaba mediado por tasa de outs, no por contacto |",
        f"| is_whiff | {PITCHING_PLUS_WEIGHTS['is_whiff']:.2f} | medio | comando/engaño — capacidad de generar swing-and-miss |",
        f"| is_chase | {PITCHING_PLUS_WEIGHTS['is_chase']:.2f} | medio | comando/engaño — capacidad de generar persecucion fuera de zona |",
        f"| is_called_strike | {PITCHING_PLUS_WEIGHTS['is_called_strike']:.2f} | medio | comando/engaño — capacidad de robarse strikes cantados |",
        f"| is_weak_contact | {PITCHING_PLUS_WEIGHTS['is_weak_contact']:.2f} | bajo | calidad de contacto — señal secundaria |",
        f"| is_groundball | {PITCHING_PLUS_WEIGHTS['is_groundball']:.2f} | bajo | calidad de contacto — señal secundaria |",
        f"| is_hard_contact | {PITCHING_PLUS_WEIGHTS['is_hard_contact']:.2f} | bajo | is_hard_contact NO mostro patron claro por altitud (Semana 2/3); el efecto dominante de altitud es via outs, no via contacto duro — coherente con pesarlo menos aqui tambien |",
        "",
        f"Suma de pesos: {sum(PITCHING_PLUS_WEIGHTS.values()):.2f}.",
        "",
        "## Metodologia",
        "",
        "1. Se usan las PREDICCIONES de los 8 modelos de "
        "`src/models/pitching_plus.py` (no los outcomes 0/1 observados crudos), "
        "promediadas por (pitcher_anon_id, year) — asi se suaviza el ruido "
        "pitch a pitch con la expectativa modelada, el mismo principio detras "
        "de cualquier indice tipo '+'.",
        "2. `run_value` e `is_hard_contact` se invierten de signo antes de "
        "combinar (mayor valor = peor para el pitcher en esos dos; en el "
        "resto, mayor valor = mejor).",
        "3. Cada target se estandariza (z-score) DENTRO de cada `year` por "
        "separado, no agrupando las 3 temporadas — para no confundir "
        "diferencias de nivel entre temporadas (ej. 2026 parcial) con "
        "diferencias reales de habilidad.",
        "4. Los 8 z-scores se combinan con los pesos de la tabla de arriba "
        "(normalizados a sumar 1).",
        "5. `Pitching+ = 100 + 10 * z_combinado`.",
        "",
        "**Limitacion explicita:** las predicciones se generan sobre TODOS los "
        "pitches validos (train+test de pitcher_holdout combinados), no solo "
        "sobre out-of-fold. Los pitchers que estuvieron en el TRAIN del modelo "
        "pueden tener predicciones ligeramente optimistas (el modelo vio sus "
        "propios pitches al ajustarse) frente a los pitchers de TEST, que son "
        "genuinamente out-of-sample. Para una version de produccion, lo "
        "correcto es generar predicciones out-of-fold para todos los "
        "pitchers (K-fold o similar) antes de construir el ranking — queda "
        "pendiente, no se implemento en este v1.",
        "",
        "## Distribucion del Pitching+ final",
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
        f"## Ranking — top 10 y bottom 10 (filtro: n_pitches >= {MIN_PITCHES_FOR_RANKING} en la temporada)",
        "",
        f"De {report['n_pitcher_seasons']:,} pitcher-temporadas, {report['n_pitcher_seasons_ranked']:,} "
        f"cumplen el filtro de tamaño minimo. El filtro es una regla simple de corte, NO shrinkage "
        "bayesiano completo (pendiente de fases futuras) — pitcher-temporadas con muestra chica "
        "quedan fuera del ranking en vez de regularizadas hacia la media.",
        "",
        "### Top 10 (mejor Pitching+)",
        "",
    ]
    lines.append(top10[display_cols].round(3).to_markdown(index=False))
    lines += [
        "",
        "### Bottom 10 (peor Pitching+)",
        "",
    ]
    lines.append(bottom10[display_cols].round(3).to_markdown(index=False))

    lines += [
        "",
        "## Chequeo de cordura",
        "",
        "Los pitchers estan anonimizados (`pitcher_anon_id`), asi que este chequeo es de "
        "**consistencia interna de los componentes**, no de reputacion real de jugadores: "
        "¿el pitcher #1 domina en la mayoria de los z-components (no solo en uno, "
        "arrastrado por un peso alto), y el ultimo lugar es consistentemente malo, "
        "no solo un outlier en una sola metrica con signo mal invertido?",
        "",
    ]
    top1 = top10.iloc[0]
    bottom1 = bottom10.iloc[0]
    top1_positive = sum(1 for t in ALL_TARGETS if top1[f"z_{t}"] > 0)
    bottom1_negative = sum(1 for t in ALL_TARGETS if bottom1[f"z_{t}"] < 0)
    lines += [
        f"- **#1 ({top1['pitcher_anon_id']}, {int(top1['year'])}, Pitching+={top1['pitching_plus']:.1f})**: "
        f"{top1_positive}/8 componentes con z positivo (favorable) — "
        f"{'consistente en todos los frentes, no solo en las metricas de mayor peso' if top1_positive >= 6 else 'dominado por pocos componentes especificos, revisar cuales'}.",
        f"- **Ultimo lugar ({bottom1['pitcher_anon_id']}, {int(bottom1['year'])}, Pitching+={bottom1['pitching_plus']:.1f})**: "
        f"{bottom1_negative}/8 componentes con z negativo (desfavorable) — "
        f"{'consistentemente malo, no un outlier de una sola metrica' if bottom1_negative >= 6 else 'concentrado en pocos componentes, revisar signo/calculo de esos'}.",
        "",
        "Revisar las tablas de arriba: si algun pitcher del top-10 tiene un `z_run_value` o "
        "`z_is_hard_contact` MUY positivo mientras el resto de sus componentes son negativos, "
        "es señal de un posible error de signo (recordar: ambos se invierten antes de z-scorear) "
        "y debe investigarse antes de usar el score para el analisis contrafactual.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    scored, report = run()

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORES_PATH, index=False)
    logger.info("Guardado %s (%s filas)", SCORES_PATH, f"{len(scored):,}")

    report_text = format_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Guardado reporte en %s", REPORT_PATH)

    logger.info("\n%s", report_text)


if __name__ == "__main__":
    main()
