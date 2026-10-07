"""Stuff+ Physical — calidad física del pitch sin crédito a la ubicación ni al ambiente.

Decisiones declaradas ANTES de entrenar (P2, 2026-10-06):

1. QUÉ SE MIDE. Stuff+ Physical responde: "con estas características físicas
   (velocidad, giro, movimiento, punto de salida, extensión, ángulo de llegada a
   ubicación constante), si el bateador le tira, ¿qué tan probable es que falle?".
2. TARGET: P(whiff | swing). Mide la capacidad de hacer fallar al bateador UNA VEZ
   QUE DECIDE TIRARLE. No mide la capacidad de inducir el swing: esa depende sobre
   todo de la ubicación y del conteo (Location+). Candidatos descartados:
   - whiff por pitch: mezcla "¿tira?" (ubicación) con "¿falla?" (física); es el
     target de Stuff+ v1 y la razón de que v1 sea en buena parte ubicación.
   - evitar contacto = 1 − P(contacto | swing) = P(whiff | swing): es el mismo target.
   - valor esperado del pitch sin ubicación (cadena completa, Stuff+ v3): sin
     ubicación, P(swing) y el resultado del take son casi ruido, y el componente
     de bola en juego es estable pero casi no predice el año siguiente (v3 DIPS,
     `reports/run_value_chain.md`). Se evalúa el daño de contacto por separado en
     el perfil del pitcher (P2), no dentro de Stuff+.
3. FEATURES: tabla `FEATURE_AUDIT` (física / ubicación / altitud / leakage / uso).
   Respecto de Stuff+ v4 cambian dos cosas: se quita `SpeedDrop` (casi todo es
   drag, es decir ambiente) y `same_side` (mano del bateador) se usa para
   entrenar pero se MARGINALIZA al predecir: cada pitch se evalúa contra la mezcla
   de lados de la liga, para no premiar ni castigar al pitcher por los bateadores
   que le tocaron.
4. OBSERVADO vs NEUTRALIZADO. El movimiento (IVB, HB) cambia con la densidad del
   aire: es parte real de lo que el bateador ve ("stuff observado"), pero si se
   compara a pitchers que lanzaron en ambientes distintos, castiga al que lanzó
   en altitud ("stuff neutralizado"). Se calculan tres puntajes con el MISMO modelo:
   - `stuff_plus_physical_observado`: features tal como se midieron (sin ajuste).
   - `stuff_plus_physical` (OFICIAL): lo anterior con el ajuste simple por
     temporada × estadio anónimo (`p2_common.env_adjust`). Se elige como oficial
     porque en P0 la normalización simple predijo igual o mejor que la física.
   - `stuff_plus_physical_neutral`: features proyectadas a No Altitude con el
     contrafactual físico (por categoría, el del Pitching+ M2). Solo comparación.
   Regla fijada antes de ver resultados: si oficial y neutral correlacionan
   ≥ 0.98 entre pitcher-temporadas, se reporta una sola métrica.
5. ESCALA: shrinkage empírico-bayesiano por temporada (`pitching_plus_neutral.empirical_bayes`)
   y 100 + 10·(θ̂ − μ)/τ: 100 = pitch promedio de la liga (pitchers con ≥ 100
   pitches), 10 puntos = 1 desviación estándar de habilidad real. Intervalo creíble de 80%.
   Hiperparámetros LightGBM: los base de v1/v4 (sin tuning, para comparar en igualdad).

Salidas: data/processed/stuff_plus_physical_oof.parquet, data/processed/stuff_plus_physical_scores.parquet,
reports/stuff_plus_physical.md, reports/benchmark/stuff_plus_physical.json.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.analysis import p2_common as pc
from src.models import pitching_plus_neutral as neutral_mod
from src.models import run_value_chain as chain
from src.models import stuff_plus_v4
from src.validation import benchmark

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "data" / "processed" / "stuff_plus_physical_oof.parquet"
SCORES_PATH = ROOT / "data" / "processed" / "stuff_plus_physical_scores.parquet"
REPORT_PATH = ROOT / "reports" / "stuff_plus_physical.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "stuff_plus_physical.json"
FINAL_PATH = ROOT / "src" / "models" / "artifacts" / "stuff_plus_physical.pkl"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NUMERIC = ["RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension", "InducedVertBreak", "HorzBreak",
           "vaa_loc_adj", "haa_loc_adj", "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb"]
CATEGORICAL = ["AutoPitchType", "same_side"]
SIDES = ["same", "opposite"]

# (variable, física, contiene ubicación, contiene altitud, posible leakage, usar, motivo)
FEATURE_AUDIT = [
    ("RelSpeed", "sí", "no", "mínima (se mide en la salida)", "no", "sí", "velocidad de salida del brazo"),
    ("EffectiveVelo", "sí", "no", "mínima", "no", "no", "redundante: velocidad + extensión"),
    ("ZoneSpeed", "sí", "no", "sí (drag)", "altitud", "no", "velocidad al llegar = salida − pérdida por drag"),
    ("SpeedDrop", "parcial", "no", "sí, casi toda (drag)", "altitud", "no", "RelSpeed − ZoneSpeed: mide sobre todo el aire (quitada respecto de v4)"),
    ("ZoneTime", "parcial", "no", "sí (drag)", "altitud", "no", "redundante con velocidad, extensión y drag"),
    ("SpinRate", "sí", "no", "mínima (se mide en la salida)", "no", "sí", "giro"),
    ("SpinAxis", "sí", "no", "no", "no", "sí", "eje de giro (espejado para zurdos)"),
    ("Tilt", "sí", "no", "no", "no", "no", "el mismo eje en formato de reloj (redundante)"),
    ("HorzBreak", "sí", "no", "sí (Magnus)", "altitud parcial", "sí", "movimiento horizontal; el ambiente se trata con el ajuste por estadio"),
    ("InducedVertBreak", "sí", "no", "sí (Magnus)", "altitud parcial", "sí", "movimiento vertical inducido; ídem"),
    ("VertBreak", "sí", "no", "sí", "altitud", "no", "incluye gravedad; redundante con IVB + velocidad"),
    ("pfxx / pfxz", "sí", "no", "sí", "altitud", "no", "las mismas componentes de movimiento (redundantes)"),
    ("RelHeight", "sí", "no", "no", "no", "sí", "altura de salida"),
    ("RelSide", "sí", "no", "no", "no", "sí", "lado de salida (espejado)"),
    ("Extension", "sí", "no", "no", "no", "sí", "extensión"),
    ("VertRelAngle / HorzRelAngle", "parcial", "sí (a dónde apunta)", "no", "ubicación", "no", "el ángulo de salida decide en buena parte dónde cruza"),
    ("VertApprAngle / HorzApprAngle", "parcial", "sí (fuerte)", "parcial", "ubicación", "no (crudos)", "un pitch bajo llega más inclinado por estar bajo"),
    ("vaa_loc_adj / haa_loc_adj", "sí", "residualizada", "parcial", "baja", "sí", "ángulo de llegada menos lo que explica la ubicación (recta por tipo de pitch)"),
    ("x0…az0 (9 parámetros)", "sí", "vx0, vz0 sí", "ax, az sí", "ubicación y altitud", "no", "redundantes con salida y movimiento; las velocidades iniciales codifican el destino"),
    ("magnus_residual_vert / _horz", "derivada", "no", "sí (supone elevaciones)", "altitud", "no", "construida con la categoría de altitud"),
    ("altitude_level / altitude_category", "no", "no", "sí", "ambiente", "no", "es el ambiente"),
    ("PlateLocHeight / PlateLocSide", "no", "sí", "no", "ubicación", "no", "es la ubicación (Location+)"),
    ("Balls / Strikes / count", "no", "no", "no", "contexto", "no", "es el conteo (Location+ / Pitching+)"),
    ("velo_diff_vs_fb / ivb_diff_vs_fb / hb_diff_vs_fb", "sí (arsenal)", "no", "parcial (se cancela en parte)", "baja", "sí", "contraste con la recta del pitcher"),
    ("tunnel_similarity", "derivada", "sí (usa VAA/HAA crudos)", "parcial", "ubicación", "no", "usa ángulos de llegada sin ajustar"),
    ("pitch_usage_pct / prev_pitch_type", "no", "no", "no", "uso y secuencia", "no", "decisiones de uso, no física"),
    ("AutoPitchType", "sí (clasificación)", "no", "no", "no", "sí", "tipo de pitch"),
    ("same_side (mano bateador vs pitcher)", "no", "no", "no", "contexto del enfrentamiento", "solo para entrenar", "se marginaliza al predecir (mezcla de lados de la liga)"),
]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def _marginal_predict(model, X: pd.DataFrame, w_same: float) -> np.ndarray:
    """P(whiff | swing) promediada sobre la mezcla de lados de la liga."""
    out = np.zeros(len(X))
    for side, w in (("same", w_same), ("opposite", 1 - w_same)):
        Xs = X.copy()
        Xs["same_side"] = pd.Categorical([side] * len(X), categories=X["same_side"].cat.categories)
        out += w * model.predict_proba(Xs)[:, 1]
    return out


def generate_oof(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    if OOF_PATH.exists() and not refresh:
        logger.info("Usando OOF en cache: %s", OOF_PATH.name)
        return pd.read_parquet(OOF_PATH), json.loads(OOF_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    df = chain.load_data()
    neutral = neutral_mod.neutral_features(df)
    cols = NUMERIC + CATEGORICAL
    swings = chain.universe_masks(df)["swing"]
    known = df["same_side"].isin(SIDES).to_numpy()
    w_same = float((df.loc[swings & known, "same_side"] == "same").mean())
    y = df["y_whiff"]
    p_fit, p_obs, p_neu = (np.full(len(df), np.nan) for _ in range(3))
    t0 = time.time()
    for tr, ho in GroupKFold(n_splits=chain.N_FOLDS).split(df, groups=df["pitcher_anon_id"]):
        tr = tr[swings[tr]]
        model = lgb.LGBMClassifier(verbosity=-1, **chain.LGB_PARAMS)
        model.fit(df[cols].iloc[tr], y.iloc[tr].astype(int), categorical_feature=CATEGORICAL)
        p_fit[ho] = model.predict_proba(df[cols].iloc[ho])[:, 1]
        p_obs[ho] = _marginal_predict(model, df[cols].iloc[ho], w_same)
        p_neu[ho] = _marginal_predict(model, neutral[cols].iloc[ho], w_same)
    logger.info("OOF Stuff+ Physical: %.0fs", time.time() - t0)

    out = df[["PitchUID", "pitcher_anon_id", "year", "event_group", "y_whiff"]].copy()
    out["AutoPitchType"] = df["AutoPitchType"].astype("object")
    out["p_whiff_swing_fit"] = p_fit
    out["p_whiff_swing_obs"] = p_obs
    out["p_whiff_swing_neutral"] = p_neu
    out.to_parquet(OOF_PATH, index=False)
    ys = y[swings].astype(int).to_numpy()
    meta = {"n_swings": int(swings.sum()), "w_same_side": w_same, "base_rate": float(ys.mean()),
            "auc_modelo": float(roc_auc_score(ys, p_fit[swings])), "log_loss_modelo": float(log_loss(ys, p_fit[swings])),
            "auc_marginal": float(roc_auc_score(ys, p_obs[swings])), "log_loss_marginal": float(log_loss(ys, p_obs[swings]))}
    OOF_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta


def train_final() -> dict:
    """Modelo final con todos los swings (para la página web). Misma especificación que el OOF."""
    df = chain.load_data()
    swings = chain.universe_masks(df)["swing"]
    known = df["same_side"].isin(SIDES).to_numpy()
    model = lgb.LGBMClassifier(verbosity=-1, **chain.LGB_PARAMS)
    model.fit(df.loc[swings, NUMERIC + CATEGORICAL], df.loc[swings, "y_whiff"].astype(int), categorical_feature=CATEGORICAL)
    _, coefs = spp_location_coefs()
    artifact = {"model": model, "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
                "w_same_side": float((df.loc[swings & known, "same_side"] == "same").mean()), "location_coefs": coefs,
                "metadata": {"n_train": int(swings.sum()), "lgb_params": chain.LGB_PARAMS}}
    with open(FINAL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)
    return artifact


def spp_location_coefs():
    from src.models import stuff_plus_v2
    return stuff_plus_v2.build_v2_features(stuff_plus_v2.load_data())


def marginal_predict(artifact: dict, feats: pd.DataFrame) -> np.ndarray:
    """P(whiff | swing) con el lado del bateador marginalizado (para el modelo final)."""
    return _marginal_predict(artifact["model"], feats[artifact["numeric_features"] + artifact["categorical_features"]], artifact["w_same_side"])


def add_environment(oof: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega entorno y la versión con ajuste simple por temporada × estadio anónimo."""
    oof = oof.merge(frame[["PitchUID", "env", "altitude_category", "PlateLocSide", "PlateLocHeight"]], on="PitchUID", how="left")
    oof["p_whiff_swing_adj"] = pc.env_adjust(oof, "p_whiff_swing_obs")
    return oof


def build_scores(oof: pd.DataFrame, pooled: bool = False) -> pd.DataFrame:
    """Puntajes por pitcher-temporada (o por pitcher con todas sus temporadas si `pooled`)."""
    d = oof.assign(year=0) if pooled else oof
    out = None
    for name, col in [("stuff_plus_physical", "p_whiff_swing_adj"), ("stuff_plus_physical_observado", "p_whiff_swing_obs"),
                      ("stuff_plus_physical_neutral", "p_whiff_swing_neutral")]:
        eb, _ = neutral_mod.empirical_bayes(d.assign(neg=-d[col]), "neg")  # empirical_bayes: menor = mejor
        eb = eb.rename(columns={"n": "n_pitches", "plus": name, "plus_lo80": f"{name}_lo80", "plus_hi80": f"{name}_hi80",
                                "plus_raw": f"{name}_sin_shrinkage"})
        eb[f"{name}_p_whiff_swing"] = -eb["mean"]
        keep = ["pitcher_anon_id", "year", "n_pitches", name, f"{name}_lo80", f"{name}_hi80", f"{name}_sin_shrinkage",
                f"{name}_p_whiff_swing"]
        if name != "stuff_plus_physical":
            keep = ["pitcher_anon_id", "year", name, f"{name}_sin_shrinkage"]
        else:
            keep.append("shrink_factor")
        out = eb[keep] if out is None else out.merge(eb[keep], on=["pitcher_anon_id", "year"])
    return out.drop(columns="year") if pooled else out


# ---------------------------------------------------------------------------
# Comparación v1 / v4 / Physical
# ---------------------------------------------------------------------------


def _location_share(pred: pd.Series, d: pd.DataFrame) -> float:
    """Fracción de la varianza de la predicción explicada por la ubicación DENTRO de cada tipo de pitch (η² incremental)."""
    loc = d[["PlateLocSide", "PlateLocHeight"]].clip(lower=[-2.5, -0.5], upper=[2.5, 5.0], axis=1)
    cell = (np.floor(loc["PlateLocSide"] / 0.5).astype("Int64").astype(str) + "_"
            + np.floor(loc["PlateLocHeight"] / 0.5).astype("Int64").astype(str))
    ok = pred.notna() & d["PlateLocSide"].notna() & d["PlateLocHeight"].notna()
    p, t, c = pred[ok], d.loc[ok, "AutoPitchType"].astype(str), cell[ok]
    tot = p.var()
    r2_type = 1 - (p - p.groupby(t).transform("mean")).var() / tot
    r2_loc = 1 - (p - p.groupby([t, c]).transform("mean")).var() / tot
    return float(r2_loc - r2_type)


def _altitude_shift(pred: pd.Series, d: pd.DataFrame) -> dict:
    """Diferencia media de la predicción Extreme − No y Medium − No dentro de pitcher × tipo × temporada (pp)."""
    g = d["pitcher_anon_id"].astype(str) + "|" + d["AutoPitchType"].astype(str) + "|" + d["year"].astype(str)
    sub = pd.DataFrame({"p": pred, "g": g, "cat": d["altitude_category"]}).dropna()
    sub["p_dm"] = sub["p"] - sub.groupby("g")["p"].transform("mean")
    out = {}
    for cat in ["Medium Altitude", "Extreme Altitude"]:
        x = (sub["cat"] == cat).astype(float)
        x_dm = x - x.groupby(sub["g"]).transform("mean")
        out[pc.CAT_SHORT[cat]] = float(100 * (x_dm * sub["p_dm"]).sum() / (x_dm ** 2).sum())
    return out


def _within_type_auc(y: pd.Series, preds: dict[str, pd.Series], types: pd.Series) -> pd.DataFrame:
    rows = []
    for t in pc.MAIN_TYPES:
        m = types == t
        if m.sum() < 1000:
            continue
        row = {"tipo": t, "swings": int(m.sum()), "whiff/swing observado": float(y[m].mean())}
        for name, p in preds.items():
            row[name] = float(roc_auc_score(y[m], p[m]))
        rows.append(row)
    return pd.DataFrame(rows)


def outcomes_by_pitcher_season(frame: pd.DataFrame) -> pd.DataFrame:
    """Resultados observados con ajuste simple por entorno, para validar t → t+1."""
    f = frame.copy()
    f["whiff_adj"] = pc.env_adjust(f, "is_whiff", f["is_swing"] == 1)
    f["rv_adj"] = pc.env_adjust(f, "run_value")
    pa = f["is_pa_end"] == 1
    f["k_adj"] = pc.env_adjust(f, "is_k", pa)
    g = f.groupby(["pitcher_anon_id", "year"])
    return pd.DataFrame({
        "n_pitches": g.size(), "n_swings": g["is_swing"].sum(), "n_pa": g["is_pa_end"].sum(),
        "whiff_swing_t": g["whiff_adj"].mean(), "k_pct_t": g["k_adj"].mean(),
        "runs_prevented_100": -100 * g["rv_adj"].mean(),
    }).reset_index()


def compare(oof: pd.DataFrame, frame: pd.DataFrame, scores: pd.DataFrame) -> dict:
    v1 = pd.read_parquet(benchmark.STUFF_OOF_PITCHES_PATH, columns=["PitchUID", "pred_is_whiff"])
    v4 = pd.read_parquet(stuff_plus_v4.OOF_PATH, columns=["PitchUID", "p_whiff_swing_obs"]).rename(columns={"p_whiff_swing_obs": "p_v4"})
    d = oof.merge(v1, on="PitchUID", how="left").merge(v4, on="PitchUID", how="left")
    sw = (d["event_group"] == "swing") & d["pred_is_whiff"].notna()
    y = d.loc[sw, "y_whiff"].astype(int)
    pitch = {
        "Stuff+ v1 (P(whiff) por pitch)": {"auc_swings": float(roc_auc_score(y, d.loc[sw, "pred_is_whiff"])), "log_loss": None},
        "Stuff+ v4": {"auc_swings": float(roc_auc_score(y, d.loc[sw, "p_v4"])), "log_loss": float(log_loss(y, d.loc[sw, "p_v4"]))},
        "Stuff+ Physical (modelo)": {"auc_swings": float(roc_auc_score(y, d.loc[sw, "p_whiff_swing_fit"])),
                                     "log_loss": float(log_loss(y, d.loc[sw, "p_whiff_swing_fit"]))},
        "Stuff+ Physical (puntaje, lados marginalizados)": {"auc_swings": float(roc_auc_score(y, d.loc[sw, "p_whiff_swing_obs"])),
                                                            "log_loss": float(log_loss(y, d.loc[sw, "p_whiff_swing_obs"]))},
    }
    pitch["Stuff+ v1 (P(whiff) por pitch)"]["nota"] = "target distinto: whiff sobre todos los pitches; su log-loss no es comparable"

    preds = {"Stuff+ v1": d["pred_is_whiff"], "Stuff+ v4": d["p_v4"], "Stuff+ Physical": d["p_whiff_swing_obs"]}
    location = {k: _location_share(v, d) for k, v in preds.items()}
    altitude = {k: _altitude_shift(v, d) for k, v in preds.items()}
    altitude["Stuff+ Physical ajustado por estadio"] = _altitude_shift(d["p_whiff_swing_adj"], d)
    altitude["Stuff+ Physical neutral"] = _altitude_shift(d["p_whiff_swing_neutral"], d)
    by_type = _within_type_auc(y, {k: v[sw] for k, v in preds.items()}, d.loc[sw, "AutoPitchType"])
    mean_by_type = d.groupby("AutoPitchType")[["p_whiff_swing_obs", "p_whiff_swing_neutral"]].mean().reindex(pc.MAIN_TYPES)

    # pitcher-temporada: v1 (escala de benchmark), v4 oficial, Physical (3 versiones)
    s1 = v1.merge(oof[["PitchUID", "pitcher_anon_id", "year"]], on="PitchUID")
    s1 = s1.groupby(["pitcher_anon_id", "year"])["pred_is_whiff"].mean().reset_index()
    s1["stuff_plus_v1"] = benchmark.scale_plus(s1["pred_is_whiff"], s1["year"])
    s4 = pd.read_parquet(stuff_plus_v4.SCORES_PATH, columns=["pitcher_anon_id", "year", "stuff_plus"]).rename(columns={"stuff_plus": "stuff_plus_v4"})
    obs = outcomes_by_pitcher_season(frame)
    zone = frame.assign(in_zone=(frame["PlateLocSide"].abs() <= 0.83) & frame["PlateLocHeight"].between(1.5, 3.5))
    zone = zone.groupby(["pitcher_anon_id", "year"])["in_zone"].mean().rename("zona_pct").reset_index()
    mix = pd.crosstab([frame["pitcher_anon_id"], frame["year"]], frame["altitude_category"], normalize="index")
    mix = mix.reindex(columns=pc.CATS, fill_value=0.0)["Extreme Altitude"].rename("pct_extreme").reset_index()
    t = (obs.merge(s1[["pitcher_anon_id", "year", "stuff_plus_v1"]], on=["pitcher_anon_id", "year"], how="left")
         .merge(s4, on=["pitcher_anon_id", "year"], how="left")
         .merge(scores.drop(columns=["n_pitches"]), on=["pitcher_anon_id", "year"], how="left")
         .merge(zone, on=["pitcher_anon_id", "year"], how="left").merge(mix, on=["pitcher_anon_id", "year"], how="left"))
    names = {"stuff_plus_v1": "Stuff+ v1 benchmark", "stuff_plus_v4": "Stuff+ v4",
             "stuff_plus_physical_observado": "Physical observado", "stuff_plus_physical": "Physical (oficial)",
             "stuff_plus_physical_neutral": "Physical neutral"}
    yoy = {}
    for mn in [100, 300]:
        cols = list(names) + ["whiff_swing_t", "k_pct_t", "runs_prevented_100"]
        pairs = pc.season_pairs(t, cols, "n_pitches", mn)
        res = {}
        for n in names:
            res[n] = {"estabilidad": pc.boot_corr(pairs, f"{n}_t", f"{n}_t1"),
                      "whiff_swing_t1": pc.boot_corr(pairs, f"{n}_t", "whiff_swing_t_t1"),
                      "k_pct_t1": pc.boot_corr(pairs, f"{n}_t", "k_pct_t_t1"),
                      "carreras_t1": pc.boot_corr(pairs, f"{n}_t", "runs_prevented_100_t1")}
        for n in ["stuff_plus_physical", "stuff_plus_physical_neutral", "stuff_plus_physical_observado"]:
            res[n]["delta_vs_v4_whiff"] = pc.boot_corr(pairs, f"{n}_t", "whiff_swing_t_t1", ref="stuff_plus_v4_t")
            res[n]["delta_vs_v1_whiff"] = pc.boot_corr(pairs, f"{n}_t", "whiff_swing_t_t1", ref="stuff_plus_v1_t")
            res[n]["delta_vs_v1_carreras"] = pc.boot_corr(pairs, f"{n}_t", "runs_prevented_100_t1", ref="stuff_plus_v1_t")
        yoy[mn] = {"n_pares": int(len(pairs)), "res": res}
    big = t[t["n_pitches"] >= 100]
    pitcher_level = {n: {"r_zona_pct": float(big[n].corr(big["zona_pct"])), "r_pct_extreme": float(big[n].corr(big["pct_extreme"]))}
                     for n in names}
    r_official_neutral = float(big["stuff_plus_physical"].corr(big["stuff_plus_physical_neutral"]))
    r_official_observed = float(big["stuff_plus_physical"].corr(big["stuff_plus_physical_observado"]))
    r_v4 = float(big["stuff_plus_physical"].corr(big["stuff_plus_v4"]))
    r_v1 = float(big["stuff_plus_physical"].corr(big["stuff_plus_v1"]))
    return {"pitch": pitch, "ubicacion_eta2": location, "altitud_pp": altitude, "auc_por_tipo": by_type,
            "media_por_tipo": mean_by_type, "yoy": yoy, "pitcher": pitcher_level, "names": names,
            "r_oficial_neutral": r_official_neutral, "r_oficial_observado": r_official_observed,
            "r_oficial_v4": r_v4, "r_oficial_v1": r_v1, "tabla": t}


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def format_report(meta: dict, cmp: dict, scores: pd.DataFrame) -> str:
    names = cmp["names"]
    L = [
        "# Stuff+ Physical — calidad física del pitch sin crédito a ubicación ni ambiente",
        "",
        "Generado por `src/models/stuff_plus_physical.py`. Las decisiones (target, features, versión oficial y regla para "
        "reportar una o dos métricas) están en el docstring y se fijaron **antes** de entrenar.",
        "",
        "## 1. Auditoría de features",
        "",
        "| Variable | Física | Contiene ubicación | Contiene altitud | Posible leakage | Usar | Motivo |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in FEATURE_AUDIT:
        L.append("| " + " | ".join(row) + " |")
    L += [
        "",
        "Features finales: " + ", ".join(f"`{c}`" for c in NUMERIC + CATEGORICAL) + ". Zurdos espejados (marco de un derecho).",
        "",
        "## 2. Target",
        "",
        "| candidato | qué mide | problema | decisión |",
        "|---|---|---|---|",
        "| P(whiff \\| swing) | capacidad de hacer fallar al bateador **una vez que decide tirar** | no mide si induce el swing | **elegido** |",
        "| whiff por pitch | lo anterior **más** la capacidad de inducir el swing | el swing depende sobre todo de ubicación y conteo: mezcla Stuff con Location | descartado (es el target de v1) |",
        "| evitar contacto | 1 − P(contacto \\| swing) | es el mismo target que P(whiff \\| swing) | equivalente |",
        "| valor esperado sin ubicación | todo el árbol en carreras | sin ubicación, swing y take son casi ruido; el componente de contacto es estable pero no predice t+1 (v3 DIPS) | descartado; el daño de contacto va en el perfil |",
        "",
        "## 3. Modelo",
        "",
        f"- OOF por pitcher (GroupKFold k=5) sobre {meta['n_swings']:,} swings. Tasa base {meta['base_rate']:.4f}.",
        f"- AUC del modelo (con el lado real del bateador): {meta['auc_modelo']:.4f}; log-loss {meta['log_loss_modelo']:.4f}.",
        f"- AUC del puntaje (lados marginalizados con la mezcla de la liga, {meta['w_same_side']:.1%} mismo lado): "
        f"{meta['auc_marginal']:.4f}; log-loss {meta['log_loss_marginal']:.4f}.",
        "",
        "## 4. Comparación pitch a pitch (universo: swings)",
        "",
        "| modelo | AUC whiff \\| swing | log-loss | nota |",
        "|---|---:|---:|---|",
    ]
    for k, v in cmp["pitch"].items():
        ll = f"{v['log_loss']:.4f}" if v["log_loss"] is not None else "—"
        L.append(f"| {k} | {v['auc_swings']:.4f} | {ll} | {v.get('nota', '')} |")
    L += [
        "",
        "Stuff+ v1 se entrenó para whiff por pitch (AUC 0.7315 en su universo, todos los pitches); aquí se evalúa también "
        "sobre swings para ponerlo en la misma pregunta.",
        "",
        "## 5. Sensibilidad a la ubicación y al ambiente",
        "",
        "*Ubicación:* fracción de la varianza de la predicción pitch a pitch que explica la celda de ubicación (0.5 × 0.5 pies) "
        "**dentro** de cada tipo de pitch. *Ambiente:* diferencia media de la predicción para el mismo pitcher, tipo de pitch y "
        "temporada (puntos porcentuales de whiff).",
        "",
        "| modelo | varianza explicada por ubicación | Medium − No (pp) | Extreme − No (pp) |",
        "|---|---:|---:|---:|",
    ]
    for k in ["Stuff+ v1", "Stuff+ v4", "Stuff+ Physical"]:
        a = cmp["altitud_pp"][k]
        L.append(f"| {k} | {cmp['ubicacion_eta2'][k]:.1%} | {a['Medium']:+.2f} | {a['Extreme']:+.2f} |")
    for k in ["Stuff+ Physical ajustado por estadio", "Stuff+ Physical neutral"]:
        a = cmp["altitud_pp"][k]
        L.append(f"| {k} | — | {a['Medium']:+.2f} | {a['Extreme']:+.2f} |")
    L += [
        "",
        "Por pitcher-temporada (≥ 100 pitches):",
        "",
        "| puntaje | r con % de pitches en zona | r con % de pitches en Extreme |",
        "|---|---:|---:|",
    ]
    for n, lab in names.items():
        p = cmp["pitcher"][n]
        L.append(f"| {lab} | {p['r_zona_pct']:+.3f} | {p['r_pct_extreme']:+.3f} |")
    L += [
        "",
        "## 6. Estabilidad y predicción del año siguiente",
        "",
        "Resultados t+1 con ajuste simple por entorno. IC 95% por bootstrap de pitchers (1,000 réplicas).",
        "",
    ]
    for mn, block in cmp["yoy"].items():
        L += [f"**Mínimo {mn} pitches ({block['n_pares']} pares)**", "",
              "| puntaje | estabilidad | whiff/swing t+1 | K% t+1 | carreras evitadas t+1 |", "|---|---|---|---|---|"]
        for n, lab in names.items():
            r = block["res"][n]
            L.append(f"| {lab} | {pc.fmt_ci(r['estabilidad'])} | {pc.fmt_ci(r['whiff_swing_t1'])} | {pc.fmt_ci(r['k_pct_t1'])} | "
                     f"{pc.fmt_ci(r['carreras_t1'])} |")
        L += ["", "| puntaje | Δr whiff/swing t+1 vs v4 | Δr whiff/swing t+1 vs v1 | Δr carreras t+1 vs v1 |", "|---|---|---|---|"]
        for n in ["stuff_plus_physical", "stuff_plus_physical_neutral", "stuff_plus_physical_observado"]:
            r = block["res"][n]
            L.append(f"| {names[n]} | {pc.fmt_ci(r['delta_vs_v4_whiff'], 'delta_r')} (P={r['delta_vs_v4_whiff']['p_mejor']:.0%}) | "
                     f"{pc.fmt_ci(r['delta_vs_v1_whiff'], 'delta_r')} (P={r['delta_vs_v1_whiff']['p_mejor']:.0%}) | "
                     f"{pc.fmt_ci(r['delta_vs_v1_carreras'], 'delta_r')} (P={r['delta_vs_v1_carreras']['p_mejor']:.0%}) |")
        L.append("")
    L += [
        "## 7. Por tipo de pitch",
        "",
        "AUC de whiff | swing dentro de cada tipo (swings):",
        "",
        cmp["auc_por_tipo"].round(4).to_markdown(index=False),
        "",
        "P(whiff | swing) promedio del puntaje por tipo (observado y proyectado a No Altitude):",
        "",
        cmp["media_por_tipo"].round(4).to_markdown(),
        "",
        "## 8. ¿Una o dos métricas?",
        "",
        f"- r(oficial, neutral) = **{cmp['r_oficial_neutral']:.3f}**; r(oficial, observado sin ajuste) = {cmp['r_oficial_observado']:.3f}; "
        f"r(oficial, v4) = {cmp['r_oficial_v4']:.3f}; r(oficial, v1) = {cmp['r_oficial_v1']:.3f} (pitcher-temporadas con ≥ 100 pitches).",
    ]
    if cmp["r_oficial_neutral"] >= 0.98:
        L.append("- Según la regla fijada antes de ver resultados (≥ 0.98), **se reporta una sola métrica**: `stuff_plus_physical` "
                 "(ajuste simple por estadio). La versión neutral (contrafactual físico) da prácticamente el mismo orden y queda "
                 "solo como verificación.")
    else:
        L.append("- La correlación queda por debajo de 0.98: las dos versiones ordenan distinto a algunos pitchers, así que se "
                 "conservan ambas. `stuff_plus_physical` (ajuste simple) es la oficial; `stuff_plus_physical_neutral` es la "
                 "contrafactual y se usa para explicar, no para rankear.")
    L += [
        "",
        f"Distribución del oficial (pitcher-temporadas): media {scores['stuff_plus_physical'].mean():.1f}, "
        f"sd {scores['stuff_plus_physical'].std():.1f}.",
        "",
    ]
    return "\n".join(L)


def run(refresh: bool = False) -> dict:
    oof, meta = generate_oof(refresh=refresh)
    frame = pc.load_pitch_frame()
    oof = add_environment(oof, frame)
    scores = build_scores(oof)
    pooled = build_scores(oof, pooled=True)
    scores.to_parquet(SCORES_PATH, index=False)
    pooled.to_parquet(SCORES_PATH.with_name("stuff_plus_physical_pooled.parquet"), index=False)
    oof[["PitchUID", "pitcher_anon_id", "year", "p_whiff_swing_obs", "p_whiff_swing_adj", "p_whiff_swing_neutral"]].to_parquet(
        OOF_PATH.with_name("stuff_plus_physical_pitches.parquet"), index=False)
    cmp = compare(oof, frame, scores)
    REPORT_PATH.write_text(format_report(meta, cmp, scores), encoding="utf-8")
    payload = {"meta": meta, "pitch": cmp["pitch"], "ubicacion_eta2": cmp["ubicacion_eta2"], "altitud_pp": cmp["altitud_pp"],
               "pitcher": cmp["pitcher"], "yoy": {str(k): v for k, v in cmp["yoy"].items()},
               "r_oficial_neutral": cmp["r_oficial_neutral"], "r_oficial_observado": cmp["r_oficial_observado"],
               "r_oficial_v4": cmp["r_oficial_v4"], "r_oficial_v1": cmp["r_oficial_v1"],
               "auc_por_tipo": cmp["auc_por_tipo"].to_dict("records")}
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    train_final()
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular predicciones OOF")
    run(refresh=parser.parse_args().refresh)


if __name__ == "__main__":
    main()
