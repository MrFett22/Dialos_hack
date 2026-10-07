"""Stuff+ v3 y Pitching+ v3 — cadena de probabilidades condicionales valuada en carreras.

Reemplaza los 8 targets binarios independientes de v1 (y sus pesos elegidos a
mano) por el enfoque estándar de Stuff+/PitchingBot: descomponer cada pitch en
eventos mutuamente excluyentes, predecir la probabilidad de cada uno con
modelos CONDICIONALES y multiplicar por el valor en carreras de cada evento.

    pitch ─┬─ take  (1 − P(swing)) ─┬─ bola
           │                        ├─ strike cantado
           │                        └─ golpeado
           └─ swing (P(swing)) ─────┬─ whiff                     P(whiff | swing)
                                    └─ contacto ─┬─ foul           P(foul | contacto)
                                                 └─ en juego ─┬─ out / 1B / 2B / 3B / HR
                                                              P(resultado | en juego)

Ventajas frente a v1:
- Cada modelo se entrena en su universo correcto (whiff sobre swings, calidad
  de contacto sobre bolas en juego): ya no se mezcla "¿tiró?" con "¿falló?".
- Las probabilidades suman 1 por construcción.
- Los pesos no se inventan: los dan las carreras. El valor de cada evento es
  el cambio en carreras esperadas que produce, según el conteo.
- Desaparece la etiqueta ruidosa `is_out_in_pa` (resultado del turno copiado
  a todos sus pitches) y la regresión directa de `run_value` (R²=0.04).

Valores por evento (linear weights por conteo):
- Valores terminales = promedio observado del run_value (RE24) en los pitches
  que terminan el turno con ese evento: base por bolas, ponche, golpeado y
  cada resultado de bola en juego.
- V(conteo) = carreras esperadas del turno desde ese conteo, resuelto por
  recursión hacia atrás desde los conteos terminales con las frecuencias
  empíricas de cada evento por conteo (el foul con 2 strikes es un auto-lazo).
- Δ(evento, conteo) = V(estado siguiente) − V(conteo actual).
  Positivo = carreras a favor del bateador (malo para el pitcher).

Dos índices con la misma cadena:
- Stuff+ v3   : features "puras" de v2d (sin ubicación ni conteo) y valores
                Δ promediados sobre conteos (neutral al conteo).
- Pitching+ v3: features v2d + ubicación + conteo, con Δ en el conteo real.

Variante DIPS (post-hoc, agregada tras ver el benchmark): igual, pero el
resultado de la bola en juego se toma como el promedio de liga (solo cuenta
la probabilidad de que haya contacto en juego, no su calidad). Motivo: en la
versión completa ~2/3 de la varianza entre pitchers de Stuff+ v3 viene del
componente de bola en juego, que es muy estable pero casi no predice el año
siguiente (teoría DIPS: el pitcher controla poco el resultado del contacto).

Score = 100 + 10·z del promedio de −E[Δ] por pitcher-temporada (z por año).
Predicciones out-of-fold (GroupKFold por pitcher, k=5), mismos hiperparámetros
que v1 (el tuning es un paso posterior).

Salidas:
- data/processed/run_value_chain_oof_pitches.parquet
- data/processed/run_value_chain_scores.parquet
- reports/run_value_chain.md, reports/benchmark/run_value_chain.json
"""

import argparse
import json
import pickle
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models import stuff_plus, stuff_plus_v2
from src.validation import benchmark

ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
OOF_PATH = ROOT / "data" / "processed" / "run_value_chain_oof_pitches.parquet"
SCORES_PATH = ROOT / "data" / "processed" / "run_value_chain_scores.parquet"
FINAL_MODEL_PATHS = {
    "pitching": ROOT / "src" / "models" / "artifacts" / "pitching_plus_v3.pkl",
    "stuff": ROOT / "src" / "models" / "artifacts" / "stuff_plus_v3.pkl",
}
REPORT_PATH = ROOT / "reports" / "run_value_chain.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_FOLDS = 5
LGB_PARAMS = stuff_plus.LGB_PARAMS_COMMON
# Hiperparámetros ajustados por sub-modelo (src/models/tuning.py). Si el archivo existe, los modelos
# OFICIALES de Pitching+ (train_final_models y el índice neutral) los usan; si no, LGB_PARAMS.
TUNED_PARAMS_PATH = ROOT / "src" / "models" / "artifacts" / "pitching_plus_v3_tuned_params.json"


def submodel_params(name: str, override: dict | None = None) -> dict:
    """Hiperparámetros de un sub-modelo de Pitching+: `override` > ajustados (si existen) > base."""
    if override is not None:
        return override[name]
    if TUNED_PARAMS_PATH.exists():
        return json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))[name]
    return LGB_PARAMS

# Taxonomía de eventos a partir de PitchCall (no de is_swing: hay ~1,400 filas
# con is_swing=1 en bolas/strikes cantados, inconsistentes con PitchCall)
TAKE_CALLS = {
    "BallCalled": "ball", "BallinDirt": "ball", "BallIntentional": "ball", "AutomaticBall": "ball",
    "StrikeCalled": "called_strike", "AutomaticStrike": "called_strike",
    "HitByPitch": "hbp",
}
SWING_CALLS = {
    "StrikeSwinging": "whiff",
    "FoulBall": "foul", "FoulBallNotFieldable": "foul", "FoulBallFieldable": "foul",
    "InPlay": "in_play",
}
TAKE_CLASSES = ["ball", "called_strike", "hbp"]
BIP_CLASSES = ["out", "single", "double", "triple", "home_run"]
EVENTS = TAKE_CLASSES + ["whiff", "foul"] + BIP_CLASSES

STUFF_NUMERIC = stuff_plus_v2.V2D_NUMERIC
STUFF_CATEGORICAL = stuff_plus_v2.V2_CATEGORICAL
FEATURE_SETS = {
    "stuff": {"numeric": STUFF_NUMERIC, "categorical": STUFF_CATEGORICAL},
    "pitching": {
        "numeric": STUFF_NUMERIC + ["PlateLocHeight", "PlateLocSide", "Balls", "Strikes"],
        "categorical": STUFF_CATEGORICAL,
    },
}
SCORE_COLUMNS = {
    "stuff_plus_v3": "xrv_stuff",
    "stuff_plus_v3_dips": "xrv_stuff_dips",
    "pitching_plus_v3": "xrv_pitching",
    "pitching_plus_v3_dips": "xrv_pitching_dips",
}
# Sub-modelos: (nombre, universo de entrenamiento, columna objetivo, clases)
SUBMODELS = [
    ("swing", "all", "y_swing", 2),
    ("take", "take", "y_take", len(TAKE_CLASSES)),
    ("whiff", "swing", "y_whiff", 2),
    ("foul", "contact", "y_foul", 2),
    ("bip", "in_play", "y_bip", len(BIP_CLASSES)),
]


# ---------------------------------------------------------------------------
# Datos y etiquetas
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df, _ = stuff_plus_v2.build_v2_features(stuff_plus_v2.load_data())
    extra = pd.read_parquet(
        TARGETS_PATH,
        columns=["PitchUID", "PitchCall", "KorBB", "Balls", "Strikes", "run_value",
                 "single", "double", "triple", "home_run"],
    )
    df = df.merge(extra, on="PitchUID", how="inner")

    calls = df["PitchCall"].astype("object")
    df["event_group"] = np.select(
        [calls.isin(list(TAKE_CALLS)), calls.isin(list(SWING_CALLS))], ["take", "swing"], default="unknown"
    )
    n_unknown = int((df["event_group"] == "unknown").sum())
    valid_count = df["Balls"].between(0, 3) & df["Strikes"].between(0, 2)
    df = df.loc[(df["event_group"] != "unknown") & valid_count].reset_index(drop=True)
    logger.info("Pitches sin evento clasificable (PitchCall Undefined) o conteo inválido excluidos: %s", f"{n_unknown:,}")

    calls = df["PitchCall"].astype("object")
    swing_event = calls.map(SWING_CALLS)
    bip = np.select(
        [df["home_run"] == 1, df["triple"] == 1, df["double"] == 1, df["single"] == 1],
        ["home_run", "triple", "double", "single"], default="out",
    )
    df["event"] = np.where(
        df["event_group"] == "take", calls.map(TAKE_CALLS),
        np.where(swing_event == "in_play", bip, swing_event),
    )
    df["Balls"] = df["Balls"].astype(int)
    df["Strikes"] = df["Strikes"].astype(int)

    df["y_swing"] = (df["event_group"] == "swing").astype(int)
    df["y_take"] = df["event"].map({c: i for i, c in enumerate(TAKE_CLASSES)})
    df["y_whiff"] = np.where(df["event_group"] == "swing", (df["event"] == "whiff").astype(int), np.nan)
    contact = (df["event_group"] == "swing") & (df["event"] != "whiff")
    df["y_foul"] = np.where(contact, (df["event"] == "foul").astype(int), np.nan)
    df["y_bip"] = df["event"].map({c: i for i, c in enumerate(BIP_CLASSES)})
    return df


def universe_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(df), dtype=bool),
        "take": (df["event_group"] == "take").to_numpy(),
        "swing": (df["event_group"] == "swing").to_numpy(),
        "contact": df["y_foul"].notna().to_numpy(),
        "in_play": df["y_bip"].notna().to_numpy(),
    }


# ---------------------------------------------------------------------------
# Valores en carreras por evento y conteo
# ---------------------------------------------------------------------------


def terminal_values(df: pd.DataFrame) -> dict[str, float]:
    """Promedio de run_value (RE24) en los pitches que terminan el turno con cada evento."""
    rv = df["run_value"]
    vals = {
        "walk": float(rv[df["KorBB"] == "Walk"].mean()),
        "strikeout": float(rv[df["KorBB"] == "Strikeout"].mean()),
        "hbp": float(rv[df["event"] == "hbp"].mean()),
    }
    for c in BIP_CLASSES:
        vals[c] = float(rv[df["event"] == c].mean())
    return vals


def count_values(df: pd.DataFrame, term: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V(conteo) por recursión hacia atrás y Δ(evento, conteo).

    Devuelve (tabla V por conteo con frecuencias, tabla Δ con un renglón por conteo
    y una columna por evento).
    """
    freq = pd.crosstab([df["Balls"], df["Strikes"]], df["event"], normalize="index").reindex(columns=EVENTS, fill_value=0.0)
    V: dict[tuple[int, int], float] = {}

    def next_value(b: int, s: int, ev: str) -> float:
        if ev == "ball":
            return term["walk"] if b == 3 else V[(b + 1, s)]
        if ev in ("called_strike", "whiff"):
            return term["strikeout"] if s == 2 else V[(b, s + 1)]
        if ev == "foul":
            return V[(b, s + 1)] if s < 2 else np.nan  # auto-lazo, se resuelve abajo
        return term[ev]  # hbp y resultados en juego son terminales

    for total in range(5, -1, -1):
        for b in range(3, -1, -1):
            s = total - b
            if not 0 <= s <= 2:
                continue
            p = freq.loc[(b, s)]
            if s == 2:
                non_foul = [e for e in EVENTS if e != "foul"]
                V[(b, s)] = sum(p[e] * next_value(b, s, e) for e in non_foul) / (1 - p["foul"])
            else:
                V[(b, s)] = sum(p[e] * next_value(b, s, e) for e in EVENTS)

    delta_rows = {}
    for (b, s), v in V.items():
        row = {}
        for e in EVENTS:
            nv = next_value(b, s, e)
            row[e] = 0.0 if (e == "foul" and s == 2) else nv - v
        delta_rows[(b, s)] = row
    deltas = pd.DataFrame.from_dict(delta_rows, orient="index")[EVENTS]
    deltas.index = pd.MultiIndex.from_tuples(deltas.index, names=["Balls", "Strikes"])
    deltas = deltas.sort_index()

    vtab = pd.DataFrame({"V": pd.Series(V)}).sort_index()
    vtab.index.names = ["Balls", "Strikes"]
    vtab["share_of_pitches"] = df.groupby(["Balls", "Strikes"]).size() / len(df)
    return vtab, deltas


# ---------------------------------------------------------------------------
# Modelos out-of-fold
# ---------------------------------------------------------------------------


def oof_submodel(
    df: pd.DataFrame, folds: list, features: dict, universe: np.ndarray, target: str, n_classes: int
) -> np.ndarray:
    """Entrena en el universo del sub-modelo (dentro de cada fold) y predice TODOS los pitches held-out."""
    cols = features["numeric"] + features["categorical"]
    X = df[cols]
    y = df[target]
    width = n_classes if n_classes > 2 else 1
    out = np.full((len(df), width), np.nan)
    for tr, ho in folds:
        tr = tr[universe[tr]]
        model = lgb.LGBMClassifier(verbosity=-1, **LGB_PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr].astype(int), categorical_feature=features["categorical"])
        if len(model.classes_) != n_classes:
            raise ValueError(f"{target}: el fold no tiene todas las clases ({model.classes_})")
        proba = model.predict_proba(X.iloc[ho])
        out[ho] = proba if n_classes > 2 else proba[:, 1:2]
    return out


def event_probabilities(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Combina los sub-modelos en P(evento) incondicional; cada fila suma 1."""
    ps = preds["swing"][:, 0]
    pw = preds["whiff"][:, 0]
    pf = preds["foul"][:, 0]
    probs = {}
    for i, c in enumerate(TAKE_CLASSES):
        probs[c] = (1 - ps) * preds["take"][:, i]
    probs["whiff"] = ps * pw
    probs["foul"] = ps * (1 - pw) * pf
    for i, c in enumerate(BIP_CLASSES):
        probs[c] = ps * (1 - pw) * (1 - pf) * preds["bip"][:, i]
    return pd.DataFrame(probs)[EVENTS]


def expected_run_value(probs: pd.DataFrame, delta_matrix: np.ndarray) -> np.ndarray:
    """E[Δ] = Σ_evento P(evento)·Δ(evento). `delta_matrix` tiene una fila por pitch."""
    return (probs.to_numpy() * delta_matrix).sum(axis=1)


COMPONENT_GROUPS = {"take": TAKE_CLASSES, "whiff": ["whiff"], "foul": ["foul"], "bip": BIP_CLASSES}


def delta_matrix_for(oof: pd.DataFrame, meta: dict, set_name: str) -> np.ndarray:
    """Δ por pitch: neutral al conteo para Stuff+, conteo real para Pitching+."""
    if set_name == "stuff":
        nd = pd.Series(meta["neutral_delta"])[EVENTS].to_numpy()
        return np.tile(nd, (len(oof), 1))
    deltas = pd.DataFrame(meta["deltas"]).set_index(["Balls", "Strikes"])[EVENTS]
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    return deltas.to_numpy()[[pos[bs] for bs in zip(oof["Balls"], oof["Strikes"])]]


def add_components_and_dips(oof: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Contribución de cada grupo de eventos a E[Δ] y la variante DIPS (bola en juego = promedio de liga)."""
    oof = oof.copy()
    league = pd.Series(meta["league_bip_distribution"])[BIP_CLASSES].to_numpy()
    bip_idx = [EVENTS.index(e) for e in BIP_CLASSES]
    for set_name in FEATURE_SETS:
        D = delta_matrix_for(oof, meta, set_name)
        C = oof[[f"p_{set_name}_{e}" for e in EVENTS]].to_numpy() * D
        for g, evs in COMPONENT_GROUPS.items():
            oof[f"comp_{set_name}_{g}"] = C[:, [EVENTS.index(e) for e in evs]].sum(axis=1)
        p_in_play = oof[[f"p_{set_name}_{e}" for e in BIP_CLASSES]].to_numpy().sum(axis=1)
        bip_league_value = (D[:, bip_idx] * league).sum(axis=1)
        oof[f"xrv_{set_name}_dips"] = (
            oof[f"comp_{set_name}_take"] + oof[f"comp_{set_name}_whiff"] + oof[f"comp_{set_name}_foul"]
            + p_in_play * bip_league_value
        )
    return oof


def generate_oof(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    if OOF_PATH.exists() and not refresh:
        logger.info("Usando OOF en cache: %s", OOF_PATH.name)
        cached = pd.read_parquet(OOF_PATH)
        meta = json.loads((OOF_PATH.with_suffix(".meta.json")).read_text(encoding="utf-8"))
        return cached, meta

    df = load_data()
    logger.info("Cadena de run value: %s pitches, %s pitchers", f"{len(df):,}", f"{df['pitcher_anon_id'].nunique():,}")
    term = terminal_values(df)
    vtab, deltas = count_values(df, term)
    weights = vtab["share_of_pitches"]
    neutral_delta = (deltas.mul(weights, axis=0).sum() / weights.sum())[EVENTS]

    masks = universe_masks(df)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["pitcher_anon_id"]))
    out = df[["PitchUID", "pitcher_anon_id", "year", "event", "Balls", "Strikes", "run_value"]].copy()
    submodel_metrics: dict[str, dict] = {}

    count_pos = {bs: i for i, bs in enumerate(deltas.index)}
    pitch_deltas = deltas.to_numpy()[[count_pos[bs] for bs in zip(df["Balls"], df["Strikes"])]]
    neutral_matrix = np.tile(neutral_delta.to_numpy(), (len(df), 1))
    out["realized_delta"] = pitch_deltas[np.arange(len(df)), df["event"].map({e: i for i, e in enumerate(EVENTS)}).to_numpy()]

    for set_name, feats in FEATURE_SETS.items():
        preds = {}
        for name, universe, target, n_classes in SUBMODELS:
            t0 = time.time()
            preds[name] = oof_submodel(df, folds, feats, masks[universe], target, n_classes)
            m = masks[universe]
            y = df.loc[m, target].astype(int).to_numpy()
            p = preds[name][m]
            if n_classes == 2:
                submodel_metrics[f"{set_name}|{name}"] = {
                    "n": int(m.sum()), "auc": float(roc_auc_score(y, p[:, 0])), "log_loss": float(log_loss(y, p[:, 0])),
                    "base_rate": float(y.mean()),
                }
            else:
                submodel_metrics[f"{set_name}|{name}"] = {
                    "n": int(m.sum()), "auc": None, "log_loss": float(log_loss(y, p, labels=list(range(n_classes)))),
                    "base_rate": None,
                }
            logger.info("  %s/%s: %.0fs", set_name, name, time.time() - t0)

        probs = event_probabilities(preds)
        matrix = neutral_matrix if set_name == "stuff" else pitch_deltas
        out[f"xrv_{set_name}"] = expected_run_value(probs, matrix)
        for e in EVENTS:
            out[f"p_{set_name}_{e}"] = probs[e].to_numpy()

    out.to_parquet(OOF_PATH, index=False)
    meta = {
        "terminal_values": term,
        "count_values": vtab.reset_index().to_dict("records"),
        "deltas": deltas.reset_index().to_dict("records"),
        "neutral_delta": neutral_delta.to_dict(),
        "league_bip_distribution": df.loc[masks["in_play"], "event"].value_counts(normalize=True).reindex(BIP_CLASSES).to_dict(),
        "submodel_metrics": submodel_metrics,
    }
    OOF_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

MODEL_METRICS = {
    "stuff_plus_v1": "Stuff+ v1",
    "stuff_plus_v3": "Stuff+ v3",
    "stuff_plus_v3_dips": "Stuff+ v3 DIPS",
    "pitching_plus_v1": "Pitching+ v1",
    "pitching_plus_v3": "Pitching+ v3",
    "pitching_plus_v3_dips": "Pitching+ v3 DIPS",
}
REFERENCES = {
    "stuff_plus_v3": "stuff_plus_v1",
    "stuff_plus_v3_dips": "stuff_plus_v1",
    "pitching_plus_v3": "pitching_plus_v1",
    "pitching_plus_v3_dips": "pitching_plus_v1",
}


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['r']:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_delta(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta_r']:+.3f} [{lo:+.3f}, {hi:+.3f}] · P={d['p_model_better']:.0%}"


def format_report(
    results: dict, meta: dict, checks: dict, scores: pd.DataFrame,
    by_pair: dict, components: dict, variance_share: dict,
) -> str:
    term = meta["terminal_values"]
    lines = [
        "# Stuff+ v3 / Pitching+ v3 — cadena de probabilidades valuada en carreras",
        "",
        "Generado por `src/models/run_value_chain.py`. Benchmark de `src/validation/benchmark.py` "
        "(pares 2024→2025 y 2025→2026, bootstrap pareado por pitcher, 2,000 réplicas).",
        "",
        "## Estructura",
        "",
        "Cada pitch se descompone en eventos excluyentes: take → {bola, strike cantado, golpeado}; "
        "swing → {whiff, foul, en juego → {out, 1B, 2B, 3B, HR}}. Cinco sub-modelos LightGBM condicionales "
        "(swing; resultado del take; whiff | swing; foul | contacto; resultado | en juego), out-of-fold por pitcher.",
        "",
        "- **Stuff+ v3:** features de v2d (sin ubicación ni conteo); valores Δ promediados sobre conteos.",
        "- **Pitching+ v3:** features de v2d + `PlateLocHeight`, `PlateLocSide` (espejado por mano) + `Balls`, `Strikes`; Δ en el conteo real.",
        "- **Score:** `100 + 10·z` de −E[Δ] promedio por pitcher-temporada (z dentro de cada año). Más alto = mejor.",
        "",
        "## Valores en carreras",
        "",
        "Valores terminales (promedio de run_value RE24 en el pitch que termina el turno):",
        "",
        "| evento | carreras |",
        "|---|---:|",
    ]
    for k in ["walk", "hbp", "strikeout"] + BIP_CLASSES:
        lines.append(f"| {k} | {term[k]:+.3f} |")
    lines += ["", "V(conteo) = carreras esperadas del turno desde ese conteo:", "", "| conteo | V | % de pitches |", "|---|---:|---:|"]
    for r in meta["count_values"]:
        lines.append(f"| {r['Balls']}-{r['Strikes']} | {r['V']:+.3f} | {r['share_of_pitches']:.1%} |")
    lines += ["", "Δ(evento) promediado sobre conteos (valores que usa Stuff+ v3):", "", "| evento | Δ carreras |", "|---|---:|"]
    for e, v in meta["neutral_delta"].items():
        lines.append(f"| {e} | {v:+.4f} |")
    lines += [
        "",
        f"Chequeo de coherencia: V(0-0) = {next(r['V'] for r in meta['count_values'] if r['Balls'] == 0 and r['Strikes'] == 0):+.4f} "
        f"vs. run_value promedio por turno observado = {checks['mean_pa_rv']:+.4f}.",
        "",
        "## Sub-modelos (OOF, evaluados en su propio universo)",
        "",
        "| índice | sub-modelo | n | AUC | log-loss | tasa base |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, m in meta["submodel_metrics"].items():
        s, n = key.split("|")
        auc = f"{m['auc']:.4f}" if m["auc"] is not None else "multiclase"
        base = f"{m['base_rate']:.4f}" if m["base_rate"] is not None else "—"
        lines.append(f"| {s} | {n} | {m['n']:,} | {auc} | {m['log_loss']:.4f} | {base} |")
    lines += [
        "",
        "Comparación directa con v1: `whiff | swing` reemplaza a `is_whiff` sobre todos los pitches; "
        "`resultado | en juego` reemplaza a `is_weak_contact`/`is_groundball`/`is_hard_contact`/`is_out_in_pa`.",
        "",
        "## Calibración del valor esperado",
        "",
        f"- Media de E[Δ] predicho (Pitching+ v3) = {checks['mean_xrv_pitching']:+.5f} vs. Δ realizado = {checks['mean_realized']:+.5f} carreras/pitch.",
        f"- Correlación por pitcher-temporada (mín. 100 pitches, misma temporada) entre E[Δ] predicho y Δ realizado: "
        f"Pitching+ v3 r={checks['same_season_r_pitching']:.3f}, Stuff+ v3 r={checks['same_season_r_stuff']:.3f}.",
        "",
    ]

    lines += [
        "## Diagnóstico: ¿de dónde sale la diferencia entre pitchers?",
        "",
        "Contribución de cada grupo de eventos a la varianza de E[Δ] entre pitcher-temporadas (mín. 100 pitches):",
        "",
        "| índice | take (bola/strike cantado) | whiff | foul | bola en juego |",
        "|---|---:|---:|---:|---:|",
    ]
    for set_name, share in variance_share.items():
        lines.append(f"| {set_name} | " + " | ".join(f"{share[g]:.0%}" for g in COMPONENT_GROUPS) + " |")
    lines += [
        "",
        "Cada componente como índice por separado (estabilidad y correlación con resultados observados en t+1):",
        "",
    ]
    for mp, comps in components.items():
        lines += [f"Mínimo {mp} pitches:", "", "| componente | estabilidad | carreras t+1 | K-BB% t+1 |", "|---|---:|---:|---:|"]
        for c, v in comps.items():
            lines.append(f"| {c.removeprefix('comp_')} | {v['stability']:.3f} | {v['runs_t1']:.3f} | {v['kbb_t1']:.3f} |")
        lines.append("")
    lines += [
        "El componente de bola en juego de Stuff+ es muy estable pero casi no predice el año siguiente: "
        "el modelo aprende patrones de contacto persistentes (p. ej. por tipo de pitch) que no se traducen en "
        "prevención de carreras. Motiva la variante **DIPS** (resultado en juego = promedio de liga), "
        "agregada **después** de ver estos resultados y por eso validada también en cada par de temporadas por separado.",
        "",
    ]

    names = list(MODEL_METRICS)
    for mp, res in results.items():
        lines += [
            f"## Benchmark año-contra-año — mínimo {mp} pitches ({res['n_pairs']:,} pares, {res['n_pitchers']:,} pitchers)",
            "",
            "| métrica | estabilidad | carreras evitadas t+1 | K-BB% t+1 | K% t+1 | whiff% t+1 |",
            "|---|---|---|---|---|---|",
        ]
        for n in names + list(benchmark.BASELINE_METRICS):
            pr = res["predictive"][n]
            label = n if n in names else f"*{benchmark.BASELINE_METRICS[n]}*"
            lines.append(
                f"| {label} | {_fmt(res['stability'][n])} | {_fmt(pr['obs_rv_prevented'])} | {_fmt(pr['obs_k_minus_bb'])} | "
                f"{_fmt(pr['obs_k_pct'])} | {_fmt(pr['obs_whiff_rate'])} |"
            )
        d = res["deltas_vs_reference"]
        lines += [
            "",
            "**v3 contra v1 (bootstrap pareado):**",
            "",
            "| comparación | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |",
            "|---|---|---|---|---|",
        ]
        for n in REFERENCES:
            lines.append(
                f"| {n} vs {d[f'{n}|stability']['vs']} | {_fmt_delta(d[f'{n}|stability'])} | {_fmt_delta(d[f'{n}|obs_rv_prevented'])} | "
                f"{_fmt_delta(d[f'{n}|obs_k_minus_bb'])} | {_fmt_delta(d[f'{n}|obs_whiff_rate'])} |"
            )
        b = res["deltas_vs_best_baseline"]
        lines += ["", "**Contra el mejor baseline observado:**", "", "| modelo | carreras evitadas t+1 | K-BB% t+1 |", "|---|---|---|"]
        for n in names:
            rv, kbb = b[f"{n}|obs_rv_prevented"], b[f"{n}|obs_k_minus_bb"]
            lines.append(f"| {n} | {_fmt_delta(rv)} (vs {rv['vs']}) | {_fmt_delta(kbb)} (vs {kbb['vs']}) |")
        lines.append("")

    lines += [
        "## Validación por par de temporadas (carreras evitadas t+1, delta contra v1)",
        "",
        "| mín. pitches | par | n | " + " | ".join(REFERENCES) + " |",
        "|---|---|---:|" + "---|" * len(REFERENCES),
    ]
    for mp, pairs_res in by_pair.items():
        for pair, res in pairs_res.items():
            cells = []
            for n in REFERENCES:
                d = res["deltas_vs_reference"][f"{n}|obs_rv_prevented"]
                cells.append(f"{res['predictive'][n]['obs_rv_prevented']['r']:.3f} ({d['delta_r']:+.3f}, P={d['p_model_better']:.0%})")
            lines.append(f"| {mp} | {pair} | {res['n_pairs']} | " + " | ".join(cells) + " |")
    lines.append("")

    for col in SCORE_COLUMNS:
        dist = scores[col].describe()
        lines.append(
            f"- Distribución {col}: n={int(dist['count']):,} · media={dist['mean']:.2f} · std={dist['std']:.2f} · "
            f"min={dist['min']:.2f} · max={dist['max']:.2f}"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(refresh: bool = False) -> dict:
    oof, meta = generate_oof(refresh=refresh)
    oof = add_components_and_dips(oof, meta)

    comp_cols = [c for c in oof.columns if c.startswith("comp_")]
    agg = oof.groupby(["pitcher_anon_id", "year"]).agg(
        n=("year", "size"), realized=("realized_delta", "mean"),
        **{col: (col, "mean") for col in list(SCORE_COLUMNS.values()) + comp_cols},
    ).reset_index()
    for name, col in SCORE_COLUMNS.items():
        agg[name] = benchmark.scale_plus(-agg[col], agg["year"])
    for c in comp_cols:
        agg[f"{c}_plus"] = benchmark.scale_plus(-agg[c], agg["year"])
    scores = agg[["pitcher_anon_id", "year", "n"] + list(SCORE_COLUMNS)]
    scores.to_parquet(SCORES_PATH, index=False)

    big = agg[agg["n"] >= 100]
    pa_end = oof["run_value"].notna() & (
        oof["event"].isin(BIP_CLASSES + ["hbp"])
        | ((oof["event"] == "ball") & (oof["Balls"] == 3))
        | (oof["event"].isin(["called_strike", "whiff"]) & (oof["Strikes"] == 2))
    )
    checks = {
        "mean_xrv_pitching": float(oof["xrv_pitching"].mean()),
        "mean_realized": float(oof["realized_delta"].mean()),
        "same_season_r_pitching": float(big["xrv_pitching"].corr(big["realized"])),
        "same_season_r_stuff": float(big["xrv_stuff"].corr(big["realized"])),
        "mean_pa_rv": float(oof.loc[pa_end, "run_value"].sum() / pa_end.sum()),
    }

    extra = {**{n: scores for n in SCORE_COLUMNS}, **{f"{c}_plus": agg for c in comp_cols}}
    table, _ = benchmark.build_metric_table(extra_models=extra)
    results, by_pair, components = {}, {}, {}
    for mp in benchmark.MIN_PITCHES_LEVELS:
        pairs = benchmark.build_pairs(table, mp)
        results[mp] = benchmark.correlation_suite(pairs, model_metrics=MODEL_METRICS, references=REFERENCES)
        # La variante DIPS es post-hoc: se valida además en cada par de temporadas por separado
        by_pair[mp] = {
            pair: benchmark.correlation_suite(sub, model_metrics=MODEL_METRICS, references=REFERENCES, n_boot=500)
            for pair, sub in pairs.groupby("pair")
        }
        components[mp] = {
            c: {
                "stability": float(pairs[f"{c}_plus_t"].corr(pairs[f"{c}_plus_t1"])),
                "runs_t1": float(pairs[f"{c}_plus_t"].corr(pairs["obs_rv_prevented_t1"])),
                "kbb_t1": float(pairs[f"{c}_plus_t"].corr(pairs["obs_k_minus_bb_t1"])),
            }
            for c in comp_cols
        }
        pr = results[mp]["predictive"]
        logger.info(
            "min %d: carreras t+1 — " + " | ".join(f"{n} %.3f" for n in MODEL_METRICS),
            mp, *(pr[n]["obs_rv_prevented"]["r"] for n in MODEL_METRICS),
        )

    big = agg[agg["n"] >= 100]
    variance_share = {}
    for set_name in FEATURE_SETS:
        comp = big[[f"comp_{set_name}_{g}" for g in COMPONENT_GROUPS]]
        tot = comp.sum(axis=1)
        variance_share[set_name] = {
            g: float(np.cov(comp[f"comp_{set_name}_{g}"], tot)[0, 1] / tot.var()) for g in COMPONENT_GROUPS
        }

    REPORT_PATH.write_text(
        format_report(results, meta, checks, scores, by_pair, components, variance_share), encoding="utf-8"
    )
    payload = {
        "version": "run_value_chain_v3",
        "checks": checks,
        "variance_share": variance_share,
        "components": {str(k): v for k, v in components.items()},
        "by_pair": {str(k): v for k, v in by_pair.items()},
        "results_by_min_pitches": {str(k): v for k, v in results.items()},
        "submodel_metrics": meta["submodel_metrics"],
    }
    (benchmark.BENCHMARK_DIR / "run_value_chain.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


# ---------------------------------------------------------------------------
# Modelos finales (entrenados con todos los pitches) para inferencia
# ---------------------------------------------------------------------------


def train_final_models(set_name: str = "pitching") -> dict:
    """Entrena los 5 sub-modelos con todos los pitches y guarda el artefacto.

    Pitching+ v3 es el índice oficial desde 2026-10-03 (ver
    reports/resultados_finales.md §7.4); estos modelos son los que usan el
    contrafactual de altitud y el dashboard. Los valores Δ por conteo se
    guardan junto con los modelos para que la inferencia sea autocontenida.
    """
    df = load_data()
    term = terminal_values(df)
    vtab, deltas = count_values(df, term)
    weights = vtab["share_of_pitches"]
    neutral_delta = (deltas.mul(weights, axis=0).sum() / weights.sum())[EVENTS]
    masks = universe_masks(df)
    feats = FEATURE_SETS[set_name]
    cols = feats["numeric"] + feats["categorical"]

    models = {}
    for name, universe, target, n_classes in SUBMODELS:
        sub = df.loc[masks[universe]]
        params = submodel_params(name) if set_name == "pitching" else LGB_PARAMS
        model = lgb.LGBMClassifier(verbosity=-1, **params)
        model.fit(sub[cols], sub[target].astype(int), categorical_feature=feats["categorical"])
        models[name] = model
        logger.info("  modelo final %s/%s entrenado (%s filas)", set_name, name, f"{len(sub):,}")

    _, location_coefs = stuff_plus_v2.build_v2_features(stuff_plus_v2.load_data())
    artifact = {
        "set_name": set_name,
        "models": models,
        "numeric_features": feats["numeric"],
        "categorical_features": feats["categorical"],
        "events": EVENTS,
        "deltas_by_count": deltas,
        "neutral_delta": neutral_delta,
        "terminal_values": term,
        "location_coefs": location_coefs,
        "metadata": {"n_train": len(df), "lgb_params": {n: (submodel_params(n) if set_name == "pitching" else LGB_PARAMS) for n, *_ in SUBMODELS}},
    }
    with open(FINAL_MODEL_PATHS[set_name], "wb") as fh:
        pickle.dump(artifact, fh)
    return artifact


def predict_expected_run_value(artifact: dict, df: pd.DataFrame, count_neutral: bool = False) -> pd.DataFrame:
    """P(evento) y E[Δ] por pitch con un artefacto final. `df` debe traer las features v2."""
    X = df[artifact["numeric_features"] + artifact["categorical_features"]]
    preds = {}
    for name, _, _, n_classes in SUBMODELS:
        proba = artifact["models"][name].predict_proba(X)
        preds[name] = proba if n_classes > 2 else proba[:, 1:2]
    probs = event_probabilities(preds)
    if count_neutral:
        matrix = np.tile(artifact["neutral_delta"][EVENTS].to_numpy(), (len(df), 1))
    else:
        deltas = artifact["deltas_by_count"]
        pos = {bs: i for i, bs in enumerate(deltas.index)}
        matrix = deltas.to_numpy()[[pos[(int(b), int(s))] for b, s in zip(df["Balls"], df["Strikes"])]]
    out = probs.copy()
    out["xrv"] = expected_run_value(probs, matrix)
    for g, evs in COMPONENT_GROUPS.items():
        idx = [EVENTS.index(e) for e in evs]
        out[f"comp_{g}"] = (probs.to_numpy() * matrix)[:, idx].sum(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular predicciones OOF")
    parser.add_argument("--train-final", action="store_true", help="entrenar y guardar los modelos finales de Pitching+ v3")
    args = parser.parse_args()
    if args.train_final:
        train_final_models("pitching")
        return
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
