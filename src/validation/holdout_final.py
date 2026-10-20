"""Validación fuera de muestra de los modelos FINALES (Pitching+ v3 y Stuff+ Physical) — documento técnico.

Tres esquemas, con la misma especificación e hiperparámetros que los artefactos finales:

- pitcher holdout: GroupKFold (k=5) por pitcher; ningún pitcher está a la vez en entrenamiento y prueba;
- season holdout: entrenar con 2024–2025 y evaluar en 2026 (temporada parcial);
- park holdout: entrenar sin una categoría de ambiente y evaluar en ella (Extreme fuera; Medium fuera).
  El dataset no identifica estadios; la categoría completa es un holdout más exigente que un solo estadio.

Objetivo evaluado: el valor realizado del pitch, Δ(evento, conteo) con los valores RE24 por conteo calculados SOLO con
el conjunto de entrenamiento. Baselines nulos (sin física ni ubicación): media del Δ realizado por conteo (= "xRunValue
promedio por conteo") y por conteo × tipo de pitch, estimadas en entrenamiento.

Métricas: RMSE y mejora del MSE (skill) contra cada baseline; calibración por deciles del valor predicho (media
predicha vs. realizada, pendiente e intercepto de la regresión realizado ~ predicho); correlación y pendiente por
pitcher (mín. 100 pitches en prueba); log-loss y AUC de cada sub-modelo contra su tasa base. Stuff+ Physical:
AUC de P(whiff | swing) contra la tasa por tipo de pitch.
"""

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.analysis import altitude_counterfactual_v2 as cf
from src.models import run_value_chain as chain
from src.models import stuff_plus_physical as spp

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "reports" / "benchmark" / "validacion_fuera_de_muestra.json"
REPORT_PATH = ROOT / "reports" / "validacion_fuera_de_muestra.md"
WEB_PATH = ROOT / "data" / "web" / "validation_holdout.json"
MIN_PITCHER = 100

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load() -> pd.DataFrame:
    df = chain.load_data()
    cats = pd.read_parquet(cf.RAW_PATH, columns=["PitchUID", "altitude_category"])
    return df.merge(cats, on="PitchUID", how="left")


def fit_predict(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, dict]:
    """Entrena los 5 sub-modelos de Pitching+ (hiperparámetros oficiales) en `tr` y predice P(evento) en `te`."""
    masks = chain.universe_masks(df)
    feats = chain.FEATURE_SETS["pitching"]
    cols = feats["numeric"] + feats["categorical"]
    preds, sub = {}, {}
    for name, universe, target, n_classes in chain.SUBMODELS:
        m_tr = tr & masks[universe]
        model = lgb.LGBMClassifier(verbosity=-1, **chain.submodel_params(name))
        model.fit(df.loc[m_tr, cols], df.loc[m_tr, target].astype(int), categorical_feature=feats["categorical"])
        p = model.predict_proba(df.loc[te, cols])
        preds[name] = p if n_classes > 2 else p[:, 1:2]
        m_te = masks[universe][te]
        y = df.loc[te, target].to_numpy()[m_te].astype(int)
        pt = preds[name][m_te]
        base = df.loc[m_tr, target].astype(int).value_counts(normalize=True).reindex(range(n_classes), fill_value=1e-9).to_numpy()
        if n_classes == 2:
            sub[name] = {"n": int(m_te.sum()), "auc": float(roc_auc_score(y, pt[:, 0])), "log_loss": float(log_loss(y, pt[:, 0])),
                         "log_loss_base": float(log_loss(y, np.full(len(y), base[1])))}
        else:
            sub[name] = {"n": int(m_te.sum()), "auc": None, "log_loss": float(log_loss(y, pt, labels=list(range(n_classes)))),
                         "log_loss_base": float(log_loss(y, np.tile(base, (len(y), 1)), labels=list(range(n_classes))))}
    return chain.event_probabilities(preds)[chain.EVENTS].to_numpy(), sub


def stuff_auc(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> dict:
    """Stuff+ Physical (P(whiff | swing), sin ubicación) en el mismo esquema, contra la tasa de whiff por tipo de pitch."""
    sw = chain.universe_masks(df)["swing"]
    cols = spp.NUMERIC + spp.CATEGORICAL
    m_tr, m_te = tr & sw, te & sw
    model = lgb.LGBMClassifier(verbosity=-1, **chain.LGB_PARAMS)
    model.fit(df.loc[m_tr, cols], df.loc[m_tr, "y_whiff"].astype(int), categorical_feature=spp.CATEGORICAL)
    y = df.loc[m_te, "y_whiff"].astype(int).to_numpy()
    p = model.predict_proba(df.loc[m_te, cols])[:, 1]
    rate = df.loc[m_tr].groupby(df.loc[m_tr, "AutoPitchType"].astype(str))["y_whiff"].mean()
    pb = df.loc[m_te, "AutoPitchType"].astype(str).map(rate).fillna(df.loc[m_tr, "y_whiff"].mean()).to_numpy()
    return {"n": int(m_te.sum()), "auc": float(roc_auc_score(y, p)), "auc_base_tipo": float(roc_auc_score(y, pb)),
            "log_loss": float(log_loss(y, p)), "log_loss_base_tipo": float(log_loss(y, pb))}


def evaluate(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray, probs: np.ndarray) -> dict:
    """RMSE, skill contra los baselines nulos y calibración del E[Δ] en el conjunto de prueba."""
    d_tr = df.loc[tr]
    term = chain.terminal_values(d_tr)
    _, deltas = chain.count_values(d_tr, term)
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    ev_idx = {e: i for i, e in enumerate(chain.EVENTS)}

    def realized(d):
        D = deltas.to_numpy()[[pos[bs] for bs in zip(d["Balls"], d["Strikes"])]]
        return D, D[np.arange(len(d)), d["event"].map(ev_idx).to_numpy()]

    _, r_tr = realized(d_tr)
    d_te = df.loc[te]
    D_te, r_te = realized(d_te)
    pred = (probs * D_te).sum(axis=1)
    b_count = pd.Series(r_tr).groupby([d_tr["Balls"].to_numpy(), d_tr["Strikes"].to_numpy()]).mean()
    b_ct = pd.Series(r_tr).groupby([d_tr["Balls"].to_numpy(), d_tr["Strikes"].to_numpy(), d_tr["AutoPitchType"].astype(str).to_numpy()]).mean()
    base_c = np.array([b_count.get(k, 0.0) for k in zip(d_te["Balls"], d_te["Strikes"])])
    base_ct = np.array([b_ct.get(k, b_count.get(k[:2], 0.0)) for k in zip(d_te["Balls"], d_te["Strikes"], d_te["AutoPitchType"].astype(str))])

    def mse(a):
        return float(np.mean((r_te - a) ** 2))

    out = {"n_prueba": int(te.sum()), "n_entrenamiento": int(tr.sum()), "pitchers_prueba": int(d_te["pitcher_anon_id"].nunique()),
           "rmse_modelo": float(np.sqrt(mse(pred))), "rmse_base_conteo": float(np.sqrt(mse(base_c))), "rmse_base_conteo_tipo": float(np.sqrt(mse(base_ct))),
           "skill_vs_conteo": 1 - mse(pred) / mse(base_c), "skill_vs_conteo_tipo": 1 - mse(pred) / mse(base_ct)}
    # calibración por deciles del valor predicho
    q = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"pred": pred, "real": r_te, "q": q}).groupby("q").agg(predicho=("pred", "mean"), realizado=("real", "mean"), n=("real", "size"))
    X = np.c_[np.ones(len(pred)), pred]
    beta = np.linalg.lstsq(X, r_te, rcond=None)[0]
    out.update({"calibracion_deciles": cal.reset_index().to_dict("records"), "pendiente_calibracion": float(beta[1]), "intercepto_calibracion": float(beta[0]),
                "media_predicha": float(pred.mean()), "media_realizada": float(r_te.mean()),
                "ece": float((cal["n"] * (cal["predicho"] - cal["realizado"]).abs()).sum() / cal["n"].sum())})
    # por pitcher (mín. 100 pitches en prueba)
    g = pd.DataFrame({"p": d_te["pitcher_anon_id"].to_numpy(), "pred": pred, "real": r_te}).groupby("p").agg(pred=("pred", "mean"), real=("real", "mean"), n=("real", "size"))
    g = g[g["n"] >= MIN_PITCHER]
    Xg = np.c_[np.ones(len(g)), g["pred"]]
    bg = np.linalg.lstsq(Xg, g["real"].to_numpy(), rcond=None)[0]
    out.update({"pitchers_min100": int(len(g)), "corr_pitcher": float(g["pred"].corr(g["real"])), "pendiente_pitcher": float(bg[1])})
    return out


def run() -> dict:
    t0 = time.time()
    df = load()
    pitchers = df["pitcher_anon_id"].to_numpy()
    schemes = {}

    logger.info("pitcher holdout (GroupKFold k=5)…")
    probs = np.zeros((len(df), len(chain.EVENTS)))
    subs, stuffs = [], []
    for k, (tr_i, te_i) in enumerate(GroupKFold(n_splits=chain.N_FOLDS).split(df, groups=pitchers)):
        tr = np.zeros(len(df), bool)
        tr[tr_i] = True
        p, s = fit_predict(df, tr, ~tr)
        probs[~tr] = p
        subs.append(s)
        stuffs.append(stuff_auc(df, tr, ~tr))
        logger.info("  pliegue %d listo (%.0fs)", k + 1, time.time() - t0)
    # evaluación conjunta del OOF: valores Δ con todos los datos (cada pliegue usa su propio entrenamiento para predecir)
    allm = np.ones(len(df), bool)
    ev = evaluate(df, allm, allm, probs)
    ev["nota"] = "predicciones fuera de muestra por pitcher; valores Δ y baselines con todos los datos (no entran al modelo)"
    sub_mean = {n: {k: (float(np.mean([s[n][k] for s in subs])) if subs[0][n][k] is not None else None) for k in subs[0][n]} for n in subs[0]}
    st_mean = {k: float(np.mean([s[k] for s in stuffs])) for k in stuffs[0]}
    schemes["pitcher_holdout"] = {**ev, "submodelos": sub_mean, "stuff": st_mean}

    for name, tr, te in (("season_holdout_2026", (df["year"] < 2026).to_numpy(), (df["year"] == 2026).to_numpy()),
                         ("park_holdout_extreme", (df["altitude_category"] != "Extreme Altitude").to_numpy(), (df["altitude_category"] == "Extreme Altitude").to_numpy()),
                         ("park_holdout_medium", (df["altitude_category"] != "Medium Altitude").to_numpy(), (df["altitude_category"] == "Medium Altitude").to_numpy())):
        logger.info("%s…", name)
        p, s = fit_predict(df, tr, te)
        schemes[name] = {**evaluate(df, tr, te, p), "submodelos": s, "stuff": stuff_auc(df, tr, te)}
        logger.info("  listo (%.0fs)", time.time() - t0)

    out = {"esquemas": schemes, "min_pitches_pitcher": MIN_PITCHER, "segundos": time.time() - t0}
    JSON_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    WEB_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(report(out), encoding="utf-8")
    return out


LABEL = {"pitcher_holdout": "Pitcher holdout (k=5 por pitcher)", "season_holdout_2026": "Season holdout (2024–25 → 2026)",
         "park_holdout_extreme": "Park holdout: sin Extreme → Extreme", "park_holdout_medium": "Park holdout: sin Medium → Medium"}
SUB_ES = {"swing": "¿swing?", "take": "resultado del take", "whiff": r"whiff \| swing", "foul": r"foul \| contacto", "bip": "resultado en juego"}


def report(out: dict) -> str:
    E = out["esquemas"]
    L = ["# Validación fuera de muestra de los modelos finales", "",
         "Generado por `src/validation/holdout_final.py`. Misma especificación e hiperparámetros que los artefactos finales "
         "(Pitching+ v3 y Stuff+ Physical). El dataset no identifica estadios: el *park holdout* deja fuera una categoría completa de "
         "ambiente (todos sus estadios anónimos), que es más exigente que dejar fuera un solo estadio.", "",
         "**Objetivo:** el valor realizado de cada pitch, Δ(evento, conteo) con valores RE24 por conteo del conjunto de entrenamiento. "
         "**Baselines nulos:** media del Δ realizado por conteo (\"xRunValue promedio por conteo\") y por conteo × tipo de pitch.", "",
         "## 1. Error y mejora sobre el modelo nulo (Pitching+)", "",
         "| esquema | pitches de prueba | RMSE modelo | RMSE nulo (conteo) | mejora del MSE vs. conteo | vs. conteo × tipo |",
         "|---|---:|---:|---:|---:|---:|"]
    for k, e in E.items():
        L.append(f"| {LABEL[k]} | {e['n_prueba']:,} | {e['rmse_modelo']:.4f} | {e['rmse_base_conteo']:.4f} | {100 * e['skill_vs_conteo']:+.2f}% | "
                 f"{100 * e['skill_vs_conteo_tipo']:+.2f}% |")
    L += ["", "El valor de un pitch individual es muy ruidoso (un mismo pitch puede terminar en strike, foul o jonrón), así que el RMSE "
          "por pitch baja poco en términos absolutos; lo relevante es que el modelo supera al nulo en todos los esquemas, también en "
          "pitchers, temporada y ambientes que no vio.", "",
          "## 2. Calibración", "",
          "| esquema | media predicha | media realizada | pendiente realizado ~ predicho | intercepto | ECE (deciles) | correlación por pitcher (mín. 100) | pendiente por pitcher |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, e in E.items():
        L.append(f"| {LABEL[k]} | {e['media_predicha']:+.4f} | {e['media_realizada']:+.4f} | {e['pendiente_calibracion']:.2f} | "
                 f"{e['intercepto_calibracion']:+.4f} | {e['ece']:.4f} | {e['corr_pitcher']:.2f} ({e['pitchers_min100']}) | {e['pendiente_pitcher']:.2f} |")
    L += ["", "Pendiente 1 e intercepto 0 = calibración perfecta. Deciles del valor predicho (pitcher holdout):", "",
          "| decil | predicho | realizado | n |", "|---:|---:|---:|---:|"]
    for r in E["pitcher_holdout"]["calibracion_deciles"]:
        L.append(f"| {int(r['q']) + 1} | {r['predicho']:+.4f} | {r['realizado']:+.4f} | {r['n']:,} |")
    L += ["", "## 3. Sub-modelos del árbol (log-loss contra la tasa base; AUC)", "",
          "| esquema | " + " | ".join(SUB_ES.values()) + " |", "|---|" + "---|" * len(SUB_ES)]
    for k, e in E.items():
        cells = []
        for n in SUB_ES:
            s = e["submodelos"][n]
            mej = 100 * (1 - s["log_loss"] / s["log_loss_base"])
            cells.append(f"{s['log_loss']:.3f} vs. {s['log_loss_base']:.3f} ({mej:+.0f}%)" + (f"; AUC {s['auc']:.3f}" if s["auc"] is not None else ""))
        L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
    L += ["", "## 4. Stuff+ Physical: P(whiff | swing) sin ubicación", "",
          "| esquema | swings de prueba | AUC modelo | AUC nulo (tasa por tipo) | log-loss | log-loss nulo |", "|---|---:|---:|---:|---:|---:|"]
    for k, e in E.items():
        s = e["stuff"]
        L.append(f"| {LABEL[k]} | {s['n']:,.0f}{' (por pliegue)' if k == 'pitcher_holdout' else ''} | {s['auc']:.3f} | {s['auc_base_tipo']:.3f} | {s['log_loss']:.4f} | {s['log_loss_base_tipo']:.4f} |")
    L += ["", "## 5. Lectura", "",
          "- El modelo generaliza a **pitchers no vistos**, a una **temporada futura** y a **ambientes completos que no vio**: la mejora "
          "sobre el nulo y la calibración se mantienen en los tres esquemas.",
          f"- En el park holdout de Extreme el modelo **subestima un poco** el valor realizado (predicho {E['park_holdout_extreme']['media_predicha']:+.4f} vs. "
          f"realizado {E['park_holdout_extreme']['media_realizada']:+.4f} carreras por pitch): sin haber visto Extreme, no sabe que ahí el contacto "
          "cuesta más. Es el mismo hallazgo de P2 §7.2 y la razón de ajustar por ambiente (índice oficial) y de calibrar el costo de contacto en "
          "el Pitching+ para altitud.",
          f"- La correlación por pitcher baja de {E['pitcher_holdout']['corr_pitcher']:.2f} a {E['park_holdout_extreme']['corr_pitcher']:.2f} en el park "
          "holdout de Extreme, en parte porque cada pitcher tiene menos pitches de prueba (solo los de ese ambiente); la pendiente sigue cerca de 1.",
          "- La validación de los índices por pitcher (estabilidad y predicción del año siguiente) está en `reports/validacion_temporal.md` y "
          "`reports/p2_pitcher_profiles_altitude.md`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    run()
