"""Tablas de los entregables operativos del reto (recomendaciones, mapa de movimiento, arsenales, perfil de prospecto).

No cambia ningún índice oficial. Todo sale de los pitches reales y de los outputs de P2:

1. Perfiles por forma de pitch ("pitch shape"): cada pitcher × tipo se clasifica por su movimiento característico
   medido en No Altitude (tercil dentro del tipo), así la forma no depende del aire en el que se lanzó. Para cada forma se
   estima, con efectos fijos (mismo pitcher; y mismo bateador para resultados), el cambio en Extreme contra No Altitude de:
   Stuff+ observado (lo que vio el bateador), IVB, |HB|, whiff/swing, contacto duro y jonrones por bola en juego, run value
   por bola en juego y valor realizado por 100 pitches. Es descriptivo y ajustado por contexto, no causal.
2. Mapa de movimiento: cada pitcher × tipo en el plano HB–IVB (medido en No Altitude), con su Stuff+ del tipo; los 10% más y
   menos efectivos se marcan, y si lanzó en Extreme se dibuja hacia dónde se movió su pitch.
3. Arsenales: combinación de tipos con uso ≥ 10%; Stuff+ de cada pitcher calculado solo con sus pitches en Extreme (escala de
   pitcher), AEPP y K% por arsenal.
4. Prospecto / agente libre: percentiles de referencia de los pitchers elegibles y los parámetros de la calculadora de AEPP.
5. Revisión de `OutsOnPlay`: turnos cuyo siguiente turno de la misma media entrada empieza con más outs de los registrados
   (sobre todo dobles matanzas no registradas) y su sesgo aproximado en el run value por bola en juego de cada ambiente.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import altitude_hypotheses as hyp
from src.analysis import p2_common as pc
from src.features import targets as tg
from src.models import stuff_plus_physical as spp

ROOT = Path(__file__).resolve().parents[2]
WEB = pc.WEB_DIR
SEED = 2026
MIN_SHAPE_NO = 30          # pitches del pitcher × tipo en No Altitude para definir su forma
MIN_MAP = 50               # pitches del pitcher × tipo para el mapa de movimiento
MIN_MAP_EXT = 30           # pitches en Extreme para dibujar el desplazamiento
MIN_EXT_STUFF = 100        # pitches en Extreme para el Stuff+ del pitcher en Extreme
MIN_ARSENAL = 8            # pitchers por arsenal para reportarlo
USE_MIN = 0.10             # uso mínimo de un tipo para contarlo en el arsenal
# movimiento característico de cada tipo: (columna, signo, nombre del eje)
SHAPE = {"Four-Seam": ("InducedVertBreak", 1, "ride (IVB)"), "Sinker": ("hb_arm", 1, "corrida al brazo (HB)"),
         "Cutter": ("hb_arm", -1, "corte (HB al guante)"), "Slider": ("hb_arm", -1, "barrido (HB al guante)"),
         "Curveball": ("InducedVertBreak", -1, "caída (−IVB)"), "Changeup": ("hb_arm", 1, "fade (HB al brazo)"),
         "Splitter": ("InducedVertBreak", -1, "caída (−IVB)")}
TERCILES = ["bajo", "medio", "alto"]
FE_OUT = ["pitcher_anon_id", "batter_anon_id", "year", "count_id"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_frame() -> pd.DataFrame:
    f = pc.load_pitch_frame()
    st = pd.read_parquet(spp.OOF_PATH.with_name("stuff_plus_physical_pitches.parquet"), columns=["PitchUID", "p_whiff_swing_obs", "p_whiff_swing_adj"])
    f = f.merge(st, on="PitchUID", how="left")
    f["AutoPitchType"] = f["AutoPitchType"].astype("object")
    sign = np.where(f["PitcherThrows"] == "Left", -1.0, 1.0)
    f["hb_arm"] = f["HorzBreak"] * sign
    f["hb_abs"] = f["hb_arm"].abs()
    f["count_id"] = f["Balls"].astype(str) + "-" + f["Strikes"].astype(str)
    f["ptg"] = f["pitcher_anon_id"].astype(str) + "|" + f["AutoPitchType"] + "|" + f["year"].astype(str)
    scales = json.loads((WEB / "simulator_scales.json").read_text(encoding="utf-8"))["stuff"]
    f["stuff_obs_plus"] = 100 + 10 * (f["p_whiff_swing_obs"] - scales["media_lanzamiento"]) / scales["sd_lanzamiento"]
    f["valor_100"] = -100 * f["realized_delta"]  # + = carreras evitadas
    ok = f["altitude_category"].isin(pc.CATS) & f["AutoPitchType"].isin(pc.MAIN_TYPES) & f["batter_anon_id"].notna()
    return f[ok].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Formas de pitch
# ---------------------------------------------------------------------------


def shapes(f: pd.DataFrame) -> pd.Series:
    """Tercil de forma de cada pitcher × tipo según su movimiento característico en No Altitude."""
    no = f[f["altitude_category"] == "No Altitude"]
    out = {}
    for t, (col, sgn, _) in SHAPE.items():
        g = no[no["AutoPitchType"] == t].groupby("pitcher_anon_id")[col].agg(["mean", "size"])
        g = g[g["size"] >= MIN_SHAPE_NO]
        q = pd.qcut(sgn * g["mean"], 3, labels=TERCILES)
        for p, lab in q.items():
            out[(p, t)] = lab
    return pd.Series(out, name="forma")


def shape_effects(f: pd.DataFrame, forma: pd.Series) -> pd.DataFrame:
    key = pd.MultiIndex.from_arrays([f["pitcher_anon_id"], f["AutoPitchType"]])
    f = f.assign(forma=forma.reindex(key).to_numpy())
    rows = []
    for t in pc.MAIN_TYPES:
        for lab in TERCILES:
            x = f[(f["AutoPitchType"] == t) & (f["forma"] == lab)]
            if x["altitude_category"].eq("Extreme Altitude").sum() < 200:
                continue
            sw, bp = x[x["is_swing"] == 1], x[x["is_bip"] == 1]
            no = x[x["altitude_category"] == "No Altitude"]
            row = {"tipo": t, "forma": lab, "eje": SHAPE[t][2], "pitcher_tipos": int(x["pitcher_anon_id"].nunique()), "pitches": int(len(x)),
                   "pitches_extreme": int((x["altitude_category"] == "Extreme Altitude").sum()),
                   "ivb_no": float(no["InducedVertBreak"].mean()), "hb_arm_no": float(no["hb_arm"].mean()), "velo_no": float(no["RelSpeed"].mean()),
                   "stuff_no": float(no["stuff_obs_plus"].mean()), "hard_no": float(bp.loc[bp["altitude_category"] == "No Altitude", "is_hard"].mean()),
                   "hr_bip_no": float(bp.loc[bp["altitude_category"] == "No Altitude", "is_hr"].mean())}
            specs = [("stuff", x, "stuff_obs_plus", ["ptg"]), ("ivb", x, "InducedVertBreak", ["ptg"]), ("hb_abs", x, "hb_abs", ["ptg"]),
                     ("whiff_swing", sw.assign(v=sw["is_whiff"].astype(float)), "v", FE_OUT),
                     ("hard", bp, "is_hard", FE_OUT), ("hr_bip", bp, "is_hr", FE_OUT), ("rv_bip", bp, "run_value", FE_OUT),
                     ("valor_100", x, "valor_100", FE_OUT)]
            for name, data, y, fe in specs:
                r = hyp.fe_estimate(data.reset_index(drop=True), y, fe)
                row[f"{name}_E"], row[f"{name}_E_se"] = r["beta_extreme"], r["se_extreme"]
                row[f"{name}_M"], row[f"{name}_M_se"] = r["beta_medium"], r["se_medium"]
            rows.append(row)
            logger.info("  %s · %s listo", t, lab)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Mapa de movimiento
# ---------------------------------------------------------------------------


def movement_map(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby(["pitcher_anon_id", "AutoPitchType"])
    tot = g.size().rename("n")
    no = f[f["altitude_category"] == "No Altitude"].groupby(["pitcher_anon_id", "AutoPitchType"]).agg(
        n_no=("PitchUID", "size"), ivb=("InducedVertBreak", "mean"), hb_arm=("hb_arm", "mean"), velo=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"))
    ex = f[f["altitude_category"] == "Extreme Altitude"].groupby(["pitcher_anon_id", "AutoPitchType"]).agg(
        n_ext=("PitchUID", "size"), ivb_ext=("InducedVertBreak", "mean"), hb_arm_ext=("hb_arm", "mean"))
    st = g.agg(p_adj=("p_whiff_swing_adj", "mean"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"), valor_100=("valor_100", "mean"))
    m = pd.concat([tot, no, ex, st], axis=1).reset_index()
    m = m[(m["n"] >= MIN_MAP) & (m["n_no"].fillna(0) >= MIN_SHAPE_NO)].copy()
    m["whiff_swing"] = m["whiffs"] / m["swings"].where(m["swings"] > 0)
    ref = m.groupby("AutoPitchType")["p_adj"].agg(["mean", "std"])
    m["stuff_plus_tipo"] = 100 + 10 * (m["p_adj"] - m["AutoPitchType"].map(ref["mean"])) / m["AutoPitchType"].map(ref["std"])
    q = m.groupby("AutoPitchType")["stuff_plus_tipo"]
    m["grupo"] = np.where(m["stuff_plus_tipo"] >= q.transform(lambda s: s.quantile(0.9)), "10% más efectivo",
                          np.where(m["stuff_plus_tipo"] <= q.transform(lambda s: s.quantile(0.1)), "10% menos efectivo", "resto"))
    m.loc[m["n_ext"].fillna(0) < MIN_MAP_EXT, ["ivb_ext", "hb_arm_ext"]] = np.nan
    return m.drop(columns=["whiffs", "p_adj"])


# ---------------------------------------------------------------------------
# 3. Arsenales
# ---------------------------------------------------------------------------


def arsenals(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prof = pd.read_csv(WEB / "pitcher_altitude_profiles.csv")
    scales = json.loads((WEB / "simulator_scales.json").read_text(encoding="utf-8"))["stuff"]["escala_pitcher"]
    use = pd.crosstab(f["pitcher_anon_id"], f["AutoPitchType"], normalize="index")
    order = [t for t in pc.MAIN_TYPES if t in use.columns]
    ars = use[order].apply(lambda r: " + ".join(t for t in order if r[t] >= USE_MIN), axis=1).rename("arsenal")
    ext = f[f["altitude_category"] == "Extreme Altitude"].groupby("pitcher_anon_id")["p_whiff_swing_adj"].agg(["mean", "size"])
    ext = ext[ext["size"] >= MIN_EXT_STUFF]
    stuff_ext = (100 + 10 * (ext["mean"] - scales["mu"]) / scales["tau"]).rename("stuff_plus_extreme")
    t = prof[["pitcher_anon_id", "aepp", "k_pct", "whiff_swing", "stuff_plus_physical", "location_plus"]].merge(ars, left_on="pitcher_anon_id", right_index=True)
    t = t.merge(stuff_ext, left_on="pitcher_anon_id", right_index=True, how="left")
    t["n_tipos"] = t["arsenal"].str.count(r"\+") + 1
    rng = np.random.default_rng(SEED)
    rows = []
    for a, g in t.groupby("arsenal"):
        gx = g.dropna(subset=["stuff_plus_extreme"])
        if len(gx) < MIN_ARSENAL:
            continue
        v = gx["stuff_plus_extreme"].to_numpy()
        boot = rng.choice(v, (2000, len(v)), replace=True).mean(1)
        rows.append({"arsenal": a, "n_tipos": int(g["n_tipos"].iloc[0]), "pitchers": int(len(g)), "pitchers_con_extreme": int(len(gx)),
                     "stuff_plus_extreme": float(v.mean()), "stuff_extreme_lo": float(np.percentile(boot, 5)),
                     "stuff_extreme_hi": float(np.percentile(boot, 95)), "stuff_plus_general": float(g["stuff_plus_physical"].mean()),
                     "aepp": float(g["aepp"].mean()), "k_pct": float(g["k_pct"].mean()), "whiff_swing": float(g["whiff_swing"].mean()),
                     "location_plus": float(g["location_plus"].mean())})
    tab = pd.DataFrame(rows).sort_values("stuff_plus_extreme", ascending=False).reset_index(drop=True)
    return tab, t


# ---------------------------------------------------------------------------
# 4. Prospecto / agente libre
# ---------------------------------------------------------------------------


def prospect_reference() -> dict:
    """μ de la liga como en P2 (`pitcher_profiles.league_means`: media global, que el ajuste por entorno conserva)."""
    f = pc.load_pitch_frame()
    prof = pd.read_csv(WEB / "pitcher_altitude_profiles.csv")
    lw = pd.read_csv(WEB / "linear_weights_by_environment.csv").set_index("ambiente")
    pa = f[f["is_pa_end"] == 1]
    mu = {"k_pct": float(pa["is_k"].mean()), "bb_pct": float(pa["is_bb"].mean()), "hbp_pct": float(pa["is_hbp"].mean()),
          "rv_bip": float(f.loc[f["is_bip"] == 1, "run_value"].mean())}
    w = lw.loc["Altitud (Medium + Extreme)"]
    metrics = {"whiff_swing": "Whiff/Swing", "k_pct": "K%", "bb_pct": "BB%", "bip_pct": "BIP%", "stuff_plus_physical": "Stuff+ Physical",
               "location_plus": "Location+", "pitching_plus": "Pitching+", "aepp": "AEPP"}
    pct = {m: {"etiqueta": lab, **{f"p{q}": float(prof[m].quantile(q / 100)) for q in (10, 25, 50, 75, 90)},
               "mayor_es_mejor": m not in ("bb_pct", "bip_pct")} for m, lab in metrics.items()}
    return {"percentiles": pct, "media_liga": mu,
            "pesos_altitud": {k: float(w[k]) for k in ("v_K", "v_BB", "v_HBP", "v_BIP")},
            "aepp_elegibles": sorted(prof["aepp"].round(3).tolist()), "n_elegibles": int(len(prof))}


def outs_on_play_check() -> dict:
    """Outs que `OutsOnPlay` no registra y su sesgo aproximado en el run value por bola en juego.

    Un turno tiene un out faltante si el siguiente turno de la misma media entrada empieza con más outs que
    Outs + OutsOnPlay (no se puede revisar el último turno de cada media entrada). El costo de cada out faltante se
    aproxima con la expectativa de carreras por número de outs (promediada sobre las bases): RE(o) − RE(o + 1).
    """
    cols = ["PitchUID", "game_anon_id", "Inning", "Top/Bottom", "Outs", "OutsOnPlay", "RunsScored", "altitude_category",
            "strikeout", "base_on_balls", "is_hit_by_pitch", "is_ball_in_play", "hit_type"]
    df = tg._assign_plate_appearances(tg._add_uuid_time(pd.read_parquet(pc.ROOT / "data" / "processed" / "pitches_with_targets.parquet", columns=cols)))
    key = ["_half_inning_id", "_pa_num"]
    ini = df.groupby(key)["Outs"].first().rename("outs_ini")
    end = df[df["_pa_end"]].groupby(key)[["OutsOnPlay", "RunsScored", "altitude_category", "is_ball_in_play", "hit_type"]].last()
    pa = end.join(ini, how="inner").reset_index().sort_values(key)
    pa["outs_sig"] = pa.groupby("_half_inning_id")["outs_ini"].shift(-1)
    after = (pa["outs_ini"] + pa["OutsOnPlay"]).clip(upper=3)
    pa["falta"] = pa["outs_sig"] - after >= 1
    pa["resto"] = pa.iloc[::-1].groupby("_half_inning_id")["RunsScored"].cumsum().iloc[::-1]
    re_o = {**pa.groupby("outs_ini")["resto"].mean().reindex([0, 1, 2]).to_dict(), 3: 0.0}
    pa["costo"] = np.where(pa["falta"] & (after < 3), after.map(re_o) - (after + 1).clip(upper=3).map(re_o), 0.0)
    pa = pa[pa["outs_sig"].notna() & pa["altitude_category"].isin(pc.CATS)]
    bip = pa[pa["is_ball_in_play"] == 1]
    out = {"turnos_revisados": int(len(pa)), "turnos_con_out_faltante": int(pa["falta"].sum()),
           "pct_turnos": pa.groupby("altitude_category")["falta"].mean().to_dict(),
           "pct_rodados": pa[pa["hit_type"] == "GroundBall"].groupby("altitude_category")["falta"].mean().to_dict(),
           "sesgo_rv_bip": bip.groupby("altitude_category")["costo"].mean().to_dict(),
           "re_por_outs": {str(k): float(v) for k, v in re_o.items()}}
    out["sesgo_extreme_menos_no"] = out["sesgo_rv_bip"]["Extreme Altitude"] - out["sesgo_rv_bip"]["No Altitude"]
    return out


def run() -> dict:
    f = load_frame()
    logger.info("Formas de pitch…")
    forma = shapes(f)
    eff = shape_effects(f, forma)
    pc.write_web_table(eff, "pitch_shape_altitude_effects")
    logger.info("Mapa de movimiento…")
    mm = movement_map(f)
    pc.write_web_table(mm, "movement_map")
    logger.info("Arsenales…")
    ars, per_pitcher = arsenals(f)
    pc.write_web_table(ars, "arsenal_altitude")
    ref = prospect_reference()
    ref["arsenal_corr_stuff_extreme_general"] = float(per_pitcher["stuff_plus_extreme"].corr(per_pitcher["stuff_plus_physical"]))
    ref["arsenal_stuff_por_n_tipos"] = per_pitcher.groupby("n_tipos")["stuff_plus_extreme"].agg(["mean", "size"]).reset_index().to_dict("records")
    (WEB / "prospect_reference.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Revisión de OutsOnPlay…")
    oop = outs_on_play_check()
    (WEB / "outs_on_play_check.json").write_text(json.dumps(oop, indent=2, ensure_ascii=False), encoding="utf-8")
    from src.reports import web_data
    web_data.write_readme()
    return {"formas": eff, "mapa": mm, "arsenales": ars, "referencia": ref, "outs_on_play": oop}


if __name__ == "__main__":
    run()
