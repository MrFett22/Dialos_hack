"""P2 — Perfil favorable para ambientes de altitud (cálculos).

Pregunta central: ¿qué características de un pitcher parecen especialmente
favorables en ambientes donde hacer contacto se vuelve más costoso y generar
whiffs/ponches se vuelve más difícil? El reporte se arma en `p2_report.py`.

Decisiones declaradas ANTES de calcular (2026-10-06):

1. MÉTRICAS POR PITCHER (`METRICS`): cada una en su universo (swing, pitch,
   turno o bola en juego), con ajuste simple por temporada × estadio anónimo
   (`p2_common.env_adjust`) y shrinkage normal-normal. El ponche se mide por
   TURNO (pitch terminal), nunca copiado a los pitches.
2. PESOS DEL PERFIL (sin pesos manuales). Las carreras de un turno se
   descomponen exactamente por su resultado:
       R = Σ_o r_o · v_o,   o ∈ {K, BB, HBP, BIP},   Σ_o r_o = 1
   con v_o = valor RE24 promedio del resultado o en ese ambiente (pesos lineales
   medidos en los datos) y v_BIP del pitcher = v_BIP,ambiente + d_i (d_i = su daño
   por bola en juego relativo a la liga). Respecto de un pitcher promedio:
       R_i − R̄ = Σ_{o≠BIP} (r_o,i − r̄_o)·(v_o − v_BIP) + r_BIP,i · d_i
   El puntaje AEPP (Altitude Environment Pitcher Profile) = −100·(R_i − R̄) con
   los pesos de los ambientes de altitud (Medium + Extreme): carreras evitadas
   por 100 turnos respecto de un pitcher promedio en esos ambientes. Los pesos
   salen de la contabilidad de carreras del propio ambiente; ninguno se elige a
   mano. Se calcula también con los pesos de No Altitude para ver cuánto cambia.
3. STUFF+ Y LOCATION+ EN EL PERFIL. Entran como covariables de la media a priori
   del shrinkage (Fay–Herriot) de K% y BB% (Stuff+ Physical, Location+) y del daño
   por bola en juego (Stuff+ Physical, GB%). Regla fijada antes: se adoptan para
   un componente solo si mejoran la correlación con el mismo componente en t+1
   (mín. 100 pitches) con P ≥ 0.80 en el bootstrap pareado; si no, shrinkage simple.
4. INCERTIDUMBRE: 2,000 extracciones de los posteriores normales de cada
   componente (independientes) → intervalo de 80% del puntaje y del lugar en el
   ranking. No incluye la incertidumbre de los pesos lineales (se reporta aparte).
5. ELEGIBLES PARA EL RANKING: ≥ 300 pitches y ≥ 75 turnos (todas las temporadas).
6. La SENSIBILIDAD individual (confiabilidad ≈ 0.11) no entra en el perfil.
7. INTERACCIONES (pregunta 12): diseño cruzado por mitades de juegos (la
   característica se mide en una mitad y las carreras se miden en la otra),
   pendiente por ambiente con bootstrap de pitchers. Es asociación, no causalidad.
8. ARQUETIPOS: k-means sobre componentes estandarizados; los nombres de
   arquetipo solo se usan si la silueta ≥ 0.25 y el ARI de estabilidad ≥ 0.6.

Salida: data/processed/p2_profiles_results.pkl (lo consume `p2_report.py`).
"""

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import altitude_hypotheses as hyp
from src.analysis import p2_common as pc
from src.analysis import sensitivity as sens
from src.data import parks
from src.models import location_plus_v2 as locp
from src.models import pitching_plus_neutral as pn
from src.models import run_value_chain as chain
from src.models import stuff_plus_physical as spp

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "data" / "processed" / "p2_profiles_results.pkl"
STAGE_DIR = ROOT / "data" / "processed" / "p2_stages"
HYP_JSON = ROOT / "reports" / "benchmark" / "hipotesis_altitud.json"
MIN_PITCHES_RANK = 300
MIN_PA_RANK = 75
MIN_ALT_PA_T1 = 40
N_DRAWS = 2000
N_BOOT = 300
SEED = 42
ALT_CATS = ["Medium Altitude", "Extreme Altitude"]
OUTCOMES = ["K", "BB", "HBP", "BIP"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# nombre: (columna por evento, universo, mayor = mejor para el pitcher, n mínimo para estimar τ, etiqueta)
METRICS = {
    "whiff_swing": ("is_whiff", "swing", True, 100, "Whiff/Swing"),
    "whiff_pitch": ("is_whiff", "pitch", True, 300, "Whiff/Pitch"),
    "k_pct": ("is_k", "pa", True, 75, "K% (por turno)"),
    "bb_pct": ("is_bb", "pa", False, 75, "BB% (por turno)"),
    "hbp_pct": ("is_hbp", "pa", False, 75, "HBP% (por turno)"),
    "bip_pct": ("is_bip", "pa", False, 75, "BIP% (por turno)"),
    "rv_bip": ("rv_bip", "bip", False, 50, "Run value por BIP"),
    "hard_pct": ("is_hard", "bip", False, 50, "Hard contact% (EV ≥ 95)"),
    "gb_pct": ("is_gb", "bip", True, 50, "Groundball%"),
    "hr_bip": ("is_hr", "bip", False, 50, "HR/BIP"),
}
COVARIATES = {"k_pct": ["stuff_plus_physical", "location_plus"], "bb_pct": ["location_plus", "stuff_plus_physical"],
              "rv_bip": ["stuff_plus_physical", "gb_pct"]}
PROFILE_DIMS = {  # para explicaciones: (mayor = mejor, frase fuerte, frase débil)
    "whiff_swing": (True, "genera muchos whiffs por swing", "genera pocos whiffs por swing"),
    "k_pct": (True, "poncha a muchos bateadores", "poncha poco"),
    "bip_pct": (False, "permite pocas bolas en juego", "permite muchas bolas en juego"),
    "rv_bip": (False, "el contacto que permite hace poco daño", "el contacto que permite hace mucho daño"),
    "hard_pct": (False, "permite poco contacto duro", "permite mucho contacto duro"),
    "gb_pct": (True, "induce muchos rodados", "induce pocos rodados"),
    "bb_pct": (False, "da pocas bases por bolas", "da muchas bases por bolas"),
    "stuff_plus_physical": (True, "Stuff físico fuerte", "Stuff físico débil"),
    "location_plus": (True, "buen Location+", "Location+ bajo"),
}
ARCH_FEATURES = ["whiff_swing", "bb_pct", "gb_pct", "hard_pct", "stuff_plus_physical", "location_plus"]


def universe(frame: pd.DataFrame, u: str) -> pd.Series:
    return {"swing": frame["is_swing"] == 1, "pitch": pd.Series(True, index=frame.index),
            "pa": frame["is_pa_end"] == 1, "bip": frame["is_bip"] == 1}[u]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pc.load_pitch_frame()
    for m, (col, u, *_rest) in METRICS.items():
        mask = universe(frame, u)
        frame[f"{m}_raw"] = frame[col].where(mask)
        frame[f"{m}_adj"] = pc.env_adjust(frame, col, mask)
    st = pd.read_parquet(spp.OOF_PATH.with_name("stuff_plus_physical_pitches.parquet"), columns=["PitchUID", "p_whiff_swing_adj"])
    lo = pd.read_parquet(locp.PITCHES_PATH, columns=["PitchUID", "loc_value"])
    xr = pd.read_parquet(pn.OOF_PATH, columns=["PitchUID", "xrv_obs"])
    frame = frame.merge(st, on="PitchUID", how="left").merge(lo, on="PitchUID", how="left").merge(xr, on="PitchUID", how="left")
    frame["xrv_m1b"] = frame["xrv_obs"] - frame.groupby(["year", "env"])["xrv_obs"].transform("mean")
    frame["pa_rv_adj"] = pc.env_adjust(frame, "run_value", frame["is_pa_end"] == 1)
    return frame, pc.pa_frame(frame)


# ---------------------------------------------------------------------------
# 1. Evidencia: consistencia por estadio-temporada
# ---------------------------------------------------------------------------


def stadium_season_consistency(frame: pd.DataFrame) -> dict:
    d = frame[frame["altitude_category"].isin(pc.CATS) & frame["batter_anon_id"].notna()].copy()
    d["count_id"] = d["Balls"].astype(str) + "-" + d["Strikes"].astype(str)
    swings = d[d["is_swing"] == 1].assign(whiff=lambda x: x["is_whiff"].astype(float), en_juego=lambda x: x["is_bip"].astype(float))
    pas = d[d["is_pa_end"] == 1].assign(ponche=lambda x: x["is_k"].astype(float))
    bip = d[d["is_bip"] == 1]
    fe_pitch = ["pitcher_anon_id", "batter_anon_id", "year", "count_id"]
    fe_pa = ["pitcher_anon_id", "batter_anon_id", "year"]
    specs = [("whiff por swing", swings, "whiff", fe_pitch, -1), ("bola en juego por swing", swings, "en_juego", fe_pitch, +1),
             ("ponche por turno", pas, "ponche", fe_pa, -1), ("run value RE24 por bola en juego", bip, "run_value", fe_pitch, +1)]
    games = d.drop_duplicates("game_anon_id").set_index("game_anon_id")
    out = {}
    for cat in ALT_CATS:
        g = games[(games["stadium_category"] == cat) & games["stadium_resuelto"].notna()]
        label = g["stadium_resuelto"] + " · " + g["year"].astype(str)
        n_games = label.value_counts()
        res = {}
        for name, data, y, fe, direction in specs:
            by = hyp.fe_by_park(data, y, fe, label)
            res[name] = {"en_direccion": int(sum(np.sign(v["beta"]) == direction for v in by.values())), "total": len(by),
                         "por_estadio_temporada": {k: {**v, "juegos": int(n_games[k])} for k, v in by.items()}}
            logger.info("  %s %s: %d/%d", pc.CAT_SHORT[cat], name, res[name]["en_direccion"], len(by))
        # G (valor de evitar contacto) por estadio-temporada
        sw = d[d["is_swing"] == 1]
        weights = sw.groupby(["Balls", "Strikes"]).size() / len(sw)
        dd = d.assign(park=d["game_anon_id"].map(label)).dropna(subset=["park"])
        res["G"] = {p: hyp._gap(x.reset_index(drop=True), weights)[0] for p, x in dd.groupby("park") if len(x) > 4000}
        out[cat] = res
    return out


def environment_rates(frame: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Tasas crudas por categoría (sin controles), para dar escala a los efectos."""
    rows = []
    for cat in pc.CATS:
        f, p = frame[frame["altitude_category"] == cat], pa[pa["altitude_category"] == cat]
        sw, bp = f[f["is_swing"] == 1], f[f["is_bip"] == 1]
        rows.append({"ambiente": pc.CAT_SHORT[cat], "pitches": len(f), "turnos": len(p), "whiff_swing": sw["is_whiff"].mean(),
                     "k_pa": p["is_k"].mean(), "bb_pa": p["is_bb"].mean(), "bip_pa": p["is_bip"].mean(),
                     "rv_bip": bp["run_value"].mean(), "hard_bip": bp["is_hard"].mean(), "gb_bip": bp["is_gb"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Daño esperado por bola en juego y valor de evitar contacto
# ---------------------------------------------------------------------------


def _game_boot(g: pd.DataFrame, fn, n_boot: int = 500, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = g.to_numpy(float)
    return np.array([fn(arr[rng.integers(0, len(arr), len(arr))].sum(0)) for _ in range(n_boot)])


def contact_damage(frame: pd.DataFrame, pa: pd.DataFrame) -> dict:
    """P(BIP) × E[RV | BIP] por ambiente (categoría y estadio anónimo), con IC por bootstrap de juegos."""
    f = frame.assign(rvb=frame["run_value"].where(frame["is_bip"] == 1, 0.0), dlt=frame["realized_delta"].where(frame["is_bip"] == 1, 0.0),
                     one=1.0)
    per_game = f.groupby(["game_anon_id"]).agg(n=("one", "sum"), bip=("is_bip", "sum"), rv=("rvb", "sum"), dl=("dlt", "sum"),
                                               pa=("is_pa_end", "sum"))
    meta = f.drop_duplicates("game_anon_id").set_index("game_anon_id")[["altitude_category", "stadium_resuelto", "stadium_category"]]
    per_game = per_game.join(meta)

    def stats(v):
        n, bip, rv, dl, npa = v
        return np.array([bip / n, rv / bip, rv / n, bip / npa, dl / bip, dl / n])

    names = ["p_bip_por_pitch", "rv_por_bip", "dano_bip_por_pitch", "p_bip_por_turno", "delta_por_bip", "dano_delta_por_pitch"]
    rows, boots = [], {}
    groups = [("categoria", pc.CAT_SHORT[c], per_game[per_game["altitude_category"] == c]) for c in pc.CATS]
    groups += [("estadio", s, g) for s, g in per_game.dropna(subset=["stadium_resuelto"]).groupby("stadium_resuelto")]
    for level, name, g in groups:
        cols = g[["n", "bip", "rv", "dl", "pa"]]
        point = stats(cols.sum().to_numpy(float))
        b = _game_boot(cols, stats)
        if level == "categoria":
            boots[name] = b
        row = {"nivel": level, "ambiente": name, "juegos": len(g), "pitches": int(cols["n"].sum()), "bip": int(cols["bip"].sum())}
        if level == "estadio":
            row["categoria"] = pc.CAT_SHORT[g["stadium_category"].iloc[0]]
        for i, nm in enumerate(names):
            row[nm] = point[i]
            row[f"{nm}_lo"], row[f"{nm}_hi"] = np.percentile(b[:, i], [2.5, 97.5])
        rows.append(row)
    table = pd.DataFrame(rows)
    cat = table[table["nivel"] == "categoria"].set_index("ambiente")
    decomp = {}
    for c in ["Medium", "Extreme"]:
        pN, rN, pE, rE = cat.loc["No Altitude", "p_bip_por_pitch"], cat.loc["No Altitude", "rv_por_bip"], cat.loc[c, "p_bip_por_pitch"], cat.loc[c, "rv_por_bip"]
        diff_b = boots[c][:, 2] - boots["No Altitude"][:, 2]
        decomp[c] = {"total": pE * rE - pN * rN, "por_probabilidad": (pE - pN) * rN, "por_dano": pN * (rE - rN),
                     "interaccion": (pE - pN) * (rE - rN), "ic95_total": [float(np.percentile(diff_b, 2.5)), float(np.percentile(diff_b, 97.5))]}
    return {"tabla": table, "descomposicion": decomp, "chain_check": chain_consistency(frame)}


def chain_consistency(frame: pd.DataFrame) -> dict:
    """Identidad del árbol y calibración por ambiente de P̂(BIP) y E[Δ | BIP] (Pitching+ v3, OOF)."""
    probs = pd.read_parquet(chain.OOF_PATH, columns=["PitchUID"] + [f"p_pitching_{e}" for e in chain.EVENTS])
    meta = json.loads(chain.OOF_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    deltas = pd.DataFrame(meta["deltas"]).set_index(["Balls", "Strikes"])[chain.EVENTS]
    d = frame[["PitchUID", "Balls", "Strikes", "altitude_category", "is_bip", "realized_delta"]].merge(probs, on="PitchUID")
    P = d[[f"p_pitching_{e}" for e in chain.EVENTS]].to_numpy()
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    D = deltas.to_numpy()[[pos[bs] for bs in zip(d["Balls"], d["Strikes"])]]
    bi = [chain.EVENTS.index(e) for e in chain.BIP_CLASSES]
    p_bip = P[:, bi].sum(1)
    comp_bip = (P[:, bi] * D[:, bi]).sum(1)
    e_delta_bip = comp_bip / p_bip
    # árbol: P(BIP) = P(swing)·(1 − P(whiff|swing))·(1 − P(foul|contacto))
    ps = P[:, [chain.EVENTS.index(e) for e in ["whiff", "foul"] + chain.BIP_CLASSES]].sum(1)
    pw = P[:, chain.EVENTS.index("whiff")] / ps
    pf = P[:, chain.EVENTS.index("foul")] / (ps * (1 - pw))
    tree = ps * (1 - pw) * (1 - pf)
    d = d.assign(p_bip=p_bip, e_delta_bip=e_delta_bip, comp_bip=comp_bip)
    rows = []
    for c in pc.CATS:
        x = d[d["altitude_category"] == c]
        rows.append({"ambiente": pc.CAT_SHORT[c], "p_bip_predicho": x["p_bip"].mean(), "p_bip_observado": x["is_bip"].mean(),
                     "delta_bip_predicho": np.average(x["e_delta_bip"], weights=x["p_bip"]),
                     "delta_bip_observado": x.loc[x["is_bip"] == 1, "realized_delta"].mean(),
                     "dano_predicho_100": 100 * x["comp_bip"].mean(),
                     "dano_observado_100": 100 * x["realized_delta"].where(x["is_bip"] == 1, 0).mean()})
    return {"max_abs_identidad": float(np.abs(comp_bip - p_bip * e_delta_bip).max()),
            "max_abs_arbol": float(np.abs(tree - p_bip).max()), "calibracion": pd.DataFrame(rows)}


def linear_weights(pa: pd.DataFrame) -> pd.DataFrame:
    """Valor RE24 promedio de cada resultado de turno por ambiente (positivo = carreras para el bateador)."""
    envs = {pc.CAT_SHORT[c]: pa["altitude_category"] == c for c in pc.CATS}
    envs["Altitud (Medium + Extreme)"] = pa["altitude_category"].isin(ALT_CATS)
    rows = []
    for name, m in envs.items():
        p = pa[m]
        row = {"ambiente": name, "turnos": len(p)}
        for o in OUTCOMES:
            row[f"v_{o}"] = p.loc[p["outcome"] == o, "pa_rv"].mean()
            row[f"r_{o}"] = (p["outcome"] == o).mean()
        b = p[p["outcome"] == "BIP"]
        row["v_GB"] = b.loc[b["is_gb"] == 1, "pa_rv"].mean()
        row["v_aire"] = b.loc[b["is_gb"] == 0, "pa_rv"].mean()
        row["v_duro"] = b.loc[b["is_hard"] == 1, "pa_rv"].mean()
        row["v_no_duro"] = b.loc[b["is_hard"] == 0, "pa_rv"].mean()
        row["K_menos_BIP"] = row["v_K"] - row["v_BIP"]
        row["BB_menos_BIP"] = row["v_BB"] - row["v_BIP"]
        rows.append(row)
    lw = pd.DataFrame(rows).set_index("ambiente")
    # IC por bootstrap de juegos para v_K − v_BIP
    for name, m in envs.items():
        p = pa[m].assign(k=lambda x: (x["outcome"] == "K").astype(float), b=lambda x: (x["outcome"] == "BIP").astype(float))
        p = p.assign(rk=p["pa_rv"] * p["k"], rb=p["pa_rv"] * p["b"])
        g = p.groupby("game_anon_id")[["k", "rk", "b", "rb"]].sum()
        bt = _game_boot(g, lambda v: np.array([v[1] / v[0] - v[3] / v[2]]), n_boot=300)[:, 0]
        lw.loc[name, "K_menos_BIP_lo"], lw.loc[name, "K_menos_BIP_hi"] = np.percentile(bt, [2.5, 97.5])
    return lw


# ---------------------------------------------------------------------------
# 3. Métricas por pitcher con shrinkage
# ---------------------------------------------------------------------------


def metric_table(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, dict]:
    """Una fila por unidad (pitcher o pitcher-temporada). Para cada métrica: n, crudo, ajustado, encogido, IC 80%, B."""
    base = frame.groupby(keys).agg(n_pitches=("PitchUID", "size"), n_pa=("is_pa_end", "sum"),
                                   n_swings=("is_swing", "sum"), n_bip=("is_bip", "sum")).reset_index()
    hyper = {}
    for m, (col, u, higher, min_n, _) in METRICS.items():
        st = pc.unit_stats(frame, f"{m}_adj", keys)
        raw = pc.unit_stats(frame, f"{m}_raw", keys)[keys + ["mean"]].rename(columns={"mean": f"{m}_crudo"})
        parts = []
        for yr, sub in (st.groupby("year") if "year" in keys else [(None, st)]):
            sub = sub.copy()
            eb = pc.eb_shrink(sub["mean"], sub["n"], pc.pooled_sigma2(sub, min_n), min_n)
            hyper[(m, yr)] = {k: eb[k] for k in ["mu", "tau", "sigma", "reliability_n_tipico"]}
            sub[f"{m}_ajustado"] = sub["mean"]
            sub[f"{m}"] = eb["shrunk"]
            sub[f"{m}_sd"] = eb["post_sd"]
            sub[f"{m}_B"] = eb["B"]
            sub[f"{m}_n"] = sub["n"]
            parts.append(sub)
        st = pd.concat(parts)[keys + [f"{m}_n", f"{m}_ajustado", m, f"{m}_sd", f"{m}_B"]]
        st[f"{m}_lo80"] = st[m] - pc.Z80 * st[f"{m}_sd"]
        st[f"{m}_hi80"] = st[m] + pc.Z80 * st[f"{m}_sd"]
        base = base.merge(raw, on=keys, how="left").merge(st, on=keys, how="left")
    base["contact_pct"] = 1 - base["whiff_swing"]
    return base, hyper


def scores_for(keys: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    """Stuff+ Physical, Location+ y Pitching+ (ajuste simple M1b) por unidad."""
    pooled = keys == ["pitcher_anon_id"]
    if pooled:
        st = pd.read_parquet(spp.SCORES_PATH.with_name("stuff_plus_physical_pooled.parquet"))
        lo = pd.read_parquet(locp.SCORES_PATH.with_name("location_plus_v2_pooled.parquet"))
    else:
        st = pd.read_parquet(spp.SCORES_PATH)
        lo = pd.read_parquet(locp.SCORES_PATH)
    st = st[keys + ["stuff_plus_physical", "stuff_plus_physical_lo80", "stuff_plus_physical_hi80", "stuff_plus_physical_neutral"]]
    lo = lo[keys + ["location_plus", "location_plus_lo80", "location_plus_hi80"]]
    d = frame.dropna(subset=["xrv_m1b"])
    eb, _ = pn.empirical_bayes(d.assign(year=0) if pooled else d, "xrv_m1b")
    eb = eb.rename(columns={"plus": "pitching_plus", "plus_lo80": "pitching_plus_lo80", "plus_hi80": "pitching_plus_hi80"})
    eb = eb[["pitcher_anon_id", "year", "pitching_plus", "pitching_plus_lo80", "pitching_plus_hi80"]]
    if pooled:
        eb = eb.drop(columns="year")
    return st.merge(lo, on=keys, how="outer").merge(eb, on=keys, how="outer")


def apply_covariates(t: pd.DataFrame, frame: pd.DataFrame, keys: list[str], adopt: dict[str, bool]) -> pd.DataFrame:
    """Re-encoge K%, BB% y daño por BIP con media a priori en función de covariables (Fay–Herriot) cuando se adoptan."""
    t = t.copy()
    for m, covs in COVARIATES.items():
        t[f"{m}_simple"] = t[m]
        if not adopt.get(m):
            continue
        st = pc.unit_stats(frame, f"{m}_adj", keys)
        st = st.merge(t[keys + covs], on=keys, how="left")
        parts = []
        for yr, sub in (st.groupby("year") if "year" in keys else [(None, st)]):
            sub = sub.copy()
            X = sub[covs].astype(float)
            X = ((X - X.mean()) / X.std()).fillna(0.0).to_numpy()
            fh = pc.eb_shrink_covariates(sub["mean"], sub["n"], pc.pooled_sigma2(sub, METRICS[m][3]), X, METRICS[m][3])
            sub[m], sub[f"{m}_sd"], sub[f"{m}_B"] = fh["shrunk"], fh["post_sd"], fh["B"]
            parts.append(sub[keys + [m, f"{m}_sd", f"{m}_B"]])
        new = pd.concat(parts)
        t = t.drop(columns=[m, f"{m}_sd", f"{m}_B"]).merge(new, on=keys, how="left")
        t[f"{m}_lo80"] = t[m] - pc.Z80 * t[f"{m}_sd"]
        t[f"{m}_hi80"] = t[m] + pc.Z80 * t[f"{m}_sd"]
    return t


def covariate_check(frame: pd.DataFrame, season: pd.DataFrame) -> dict:
    """¿Las covariables mejoran la predicción del mismo componente en t+1? (regla del docstring, punto 3)."""
    fh = apply_covariates(season, frame, ["pitcher_anon_id", "year"], {m: True for m in COVARIATES})
    out = {}
    for m in COVARIATES:
        t = season[["pitcher_anon_id", "year", "n_pitches", f"{m}_ajustado", m]].rename(columns={m: "simple"})
        t = t.merge(fh[["pitcher_anon_id", "year", m]].rename(columns={m: "fay_herriot"}), on=["pitcher_anon_id", "year"], how="left")
        res = {}
        for mn in [100, 300]:
            pairs = pc.season_pairs(t, ["simple", "fay_herriot", f"{m}_ajustado"], "n_pitches", mn)
            res[mn] = {"simple": pc.boot_corr(pairs, "simple_t", f"{m}_ajustado_t1"),
                       "fay_herriot": pc.boot_corr(pairs, "fay_herriot_t", f"{m}_ajustado_t1"),
                       "delta": pc.boot_corr(pairs, "fay_herriot_t", f"{m}_ajustado_t1", ref="simple_t")}
        res["adoptar"] = bool(res[100]["delta"]["delta_r"] > 0 and res[100]["delta"]["p_mejor"] >= 0.80)
        out[m] = res
    return out


# ---------------------------------------------------------------------------
# 4. Perfil (AEPP), incertidumbre y validación
# ---------------------------------------------------------------------------


def aepp(t: pd.DataFrame, lw: pd.Series, mu: dict, draws: dict | None = None) -> np.ndarray:
    """Carreras evitadas por 100 turnos frente a un pitcher promedio, con los pesos lineales `lw` (fila de `linear_weights`)."""
    get = (lambda m: draws[m]) if draws is not None else (lambda m: t[m].to_numpy())
    rk, rbb, rhbp = np.clip(get("k_pct"), 0, 1), np.clip(get("bb_pct"), 0, 1), np.clip(get("hbp_pct"), 0, 1)
    rbip = 1 - rk - rbb - rhbp
    d = get("rv_bip") - mu["rv_bip"]
    R = ((rk - mu["k_pct"]) * (lw["v_K"] - lw["v_BIP"]) + (rbb - mu["bb_pct"]) * (lw["v_BB"] - lw["v_BIP"])
         + (rhbp - mu["hbp_pct"]) * (lw["v_HBP"] - lw["v_BIP"]) + rbip * d)
    return -100 * R


def league_means(t: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """μ de cada componente: promedio de la liga en la escala ajustada (ponderado por su universo)."""
    return {m: float(frame[f"{m}_adj"].mean()) for m in ["k_pct", "bb_pct", "hbp_pct", "rv_bip"]}


def profile_scores(t: pd.DataFrame, frame: pd.DataFrame, lw: pd.DataFrame, eligible: pd.Series) -> pd.DataFrame:
    mu = league_means(t, frame)
    t = t.copy()
    t["aepp"] = aepp(t, lw.loc["Altitud (Medium + Extreme)"], mu)
    t["aepp_extreme"] = aepp(t, lw.loc["Extreme"], mu)
    t["aepp_no_altitude"] = aepp(t, lw.loc["No Altitude"], mu)
    t["bono_altitud"] = t["aepp"] - t["aepp_no_altitude"]
    rng = np.random.default_rng(SEED)
    e = t[eligible].reset_index(drop=True)
    draws = {m: rng.normal(e[m].to_numpy(), e[f"{m}_sd"].to_numpy(), size=(N_DRAWS, len(e))) for m in ["k_pct", "bb_pct", "hbp_pct", "rv_bip"]}
    A = aepp(e, lw.loc["Altitud (Medium + Extreme)"], mu, draws)
    ranks = (-A).argsort(1).argsort(1) + 1
    e["aepp_lo80"], e["aepp_hi80"] = np.percentile(A, [10, 90], axis=0)
    e["rank"] = (-e["aepp"]).rank(method="first").astype(int)
    e["rank_lo80"], e["rank_hi80"] = np.percentile(ranks, [10, 90], axis=0).round().astype(int)
    e["p_top25"] = (ranks <= 25).mean(0)
    B = aepp(e, lw.loc["No Altitude"], mu, draws)
    e["bono_altitud_lo80"], e["bono_altitud_hi80"] = np.percentile(A - B, [10, 90], axis=0)
    return t.merge(e[["pitcher_anon_id", "aepp_lo80", "aepp_hi80", "rank", "rank_lo80", "rank_hi80", "p_top25",
                      "bono_altitud_lo80", "bono_altitud_hi80"]], on="pitcher_anon_id", how="left")


def altitude_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Carreras evitadas por 100 turnos (ajustadas por entorno) en juegos de altitud y en todos, por pitcher-temporada."""
    p = frame[frame["is_pa_end"] == 1]
    alt = p[p["altitude_category"].isin(ALT_CATS)].groupby(["pitcher_anon_id", "year"])["pa_rv_adj"].agg(["mean", "size"])
    allp = p.groupby(["pitcher_anon_id", "year"])["pa_rv_adj"].mean()
    out = pd.DataFrame({"carreras_altitud_100pa": -100 * alt["mean"], "pa_altitud": alt["size"]})
    out = out.join((-100 * allp).rename("carreras_100pa"), how="outer").reset_index()
    out["pa_altitud"] = out["pa_altitud"].fillna(0)
    return out


def validate_profile(season: pd.DataFrame, frame: pd.DataFrame, lw: pd.DataFrame) -> dict:
    mu = league_means(season, frame)
    s = season.copy()
    s["aepp"] = aepp(s, lw.loc["Altitud (Medium + Extreme)"], mu)
    s["aepp_no_altitude"] = aepp(s, lw.loc["No Altitude"], mu)
    s["k_bb"] = s["k_pct"] - s["bb_pct"]
    s = s.merge(altitude_outcomes(frame), on=["pitcher_anon_id", "year"], how="left")
    cand = {"aepp": "AEPP (pesos de altitud)", "aepp_no_altitude": "mismo perfil con pesos de No Altitude",
            "pitching_plus": "Pitching+ (ajuste simple por estadio)", "stuff_plus_physical": "Stuff+ Physical",
            "location_plus": "Location+", "k_bb": "K−BB% (con shrinkage)", "carreras_100pa": "carreras evitadas del año (crudo ajustado)"}
    out = {"candidatos": cand}
    for mn in [100, 300]:
        pairs = pc.season_pairs(s, list(cand) + ["carreras_altitud_100pa", "pa_altitud"], "n_pitches", mn)
        pa_alt = pairs[pairs["pa_altitud_t1"] >= MIN_ALT_PA_T1]
        res = {"n_pares": len(pairs), "n_pares_altitud": len(pa_alt)}
        for c in cand:
            res[c] = {"estabilidad": pc.boot_corr(pairs, f"{c}_t", f"{c}_t1"),
                      "carreras_t1": pc.boot_corr(pairs, f"{c}_t", "carreras_100pa_t1"),
                      "carreras_altitud_t1": pc.boot_corr(pa_alt, f"{c}_t", "carreras_altitud_100pa_t1")}
            if c != "pitching_plus":
                res[c]["delta_altitud_vs_pitching_plus"] = pc.boot_corr(pa_alt, f"{c}_t", "carreras_altitud_100pa_t1", ref="pitching_plus_t")
        res["aepp"]["delta_altitud_vs_pesos_no"] = pc.boot_corr(pa_alt, "aepp_t", "carreras_altitud_100pa_t1", ref="aepp_no_altitude_t")
        out[mn] = res
    return out


def metric_stability(season: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Estabilidad año a año y relación con las carreras en altitud del año siguiente, por métrica (pregunta 9)."""
    s = season.merge(altitude_outcomes(frame), on=["pitcher_anon_id", "year"], how="left")
    cols = list(METRICS) + ["stuff_plus_physical", "location_plus", "pitching_plus"]
    pairs = pc.season_pairs(s, cols + [f"{m}_ajustado" for m in METRICS] + ["carreras_altitud_100pa", "pa_altitud"], "n_pitches", 100)
    alt = pairs[pairs["pa_altitud_t1"] >= MIN_ALT_PA_T1]
    rows = []
    for c in cols:
        lab = METRICS[c][4] if c in METRICS else {"stuff_plus_physical": "Stuff+ Physical", "location_plus": "Location+",
                                                   "pitching_plus": "Pitching+ (ajuste simple)"}[c]
        higher = METRICS[c][2] if c in METRICS else True
        st = pc.boot_corr(pairs, f"{c}_ajustado_t" if c in METRICS else f"{c}_t", f"{c}_ajustado_t1" if c in METRICS else f"{c}_t1", n_boot=500)
        pr = pc.boot_corr(alt, f"{c}_t", "carreras_altitud_100pa_t1", n_boot=500)
        rows.append({"metrica": c, "etiqueta": lab, "mayor_es_mejor": higher, "estabilidad_cruda": st["r"], "estabilidad_lo": st["ci95"][0],
                     "estabilidad_hi": st["ci95"][1], "r_carreras_altitud_t1": pr["r"], "r_lo": pr["ci95"][0], "r_hi": pr["ci95"][1],
                     "n_pares": st["n"], "n_pares_altitud": pr["n"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Interacciones con el ambiente (diseño cruzado por mitades)
# ---------------------------------------------------------------------------


def interactions(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["half"] = pd.factorize(f["game_anon_id"])[0] % 2
    char_cols = {m: f"{m}_adj" for m in ["whiff_swing", "k_pct", "bb_pct", "bip_pct", "rv_bip", "hard_pct", "gb_pct"]}
    char_cols.update({"stuff_plus_physical": "p_whiff_swing_adj", "location_plus": "loc_value"})
    orient = {m: (METRICS[m][2] if m in METRICS else True) for m in char_cols}
    orient["location_plus"] = False  # loc_value: menor = mejor
    X = f.groupby(["pitcher_anon_id", "half"]).agg(n=("PitchUID", "size"), **{m: (c, "mean") for m, c in char_cols.items()})
    X = X[X["n"] >= 150].drop(columns="n")
    X = (X - X.mean()) / X.std()
    for m, hi in orient.items():
        if not hi:
            X[m] = -X[m]  # todas orientadas: +1 sd = mejor para el pitcher
    X = X.reset_index().assign(half=lambda x: 1 - x["half"])  # la característica de la OTRA mitad
    pa = f[f["is_pa_end"] == 1][["pitcher_anon_id", "half", "year", "altitude_category", "run_value"]].rename(columns={"run_value": "pa_rv"})
    pa = pa[pa["altitude_category"].isin(pc.CATS) & pa["pa_rv"].notna()].merge(X, on=["pitcher_anon_id", "half"], how="inner").reset_index(drop=True)
    cell = pa["altitude_category"] + "|" + pa["year"].astype(str)
    Dm = pd.get_dummies(cell).to_numpy(float)
    cat_dum = np.column_stack([(pa["altitude_category"] == c).to_numpy(float) for c in pc.CATS])
    y = pa["pa_rv"].to_numpy(float)
    codes, uniq = pd.factorize(pa["pitcher_anon_id"])
    rows_by = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    boot_idx = [np.concatenate([rows_by[k] for k in rng.integers(0, len(uniq), len(uniq))]) for _ in range(N_BOOT)]
    out = []
    for m in char_cols:
        Z = np.column_stack([Dm, cat_dum * pa[m].to_numpy()[:, None]])

        def fit(idx):
            b = np.linalg.lstsq(Z[idx], y[idx], rcond=None)[0][-3:]
            return -100 * b  # carreras EVITADAS por 100 turnos por +1 sd

        point = fit(np.arange(len(y)))
        bs = np.array([fit(i) for i in boot_idx])
        row = {"caracteristica": m, "turnos": len(y)}
        for j, c in enumerate(pc.CATS):
            row[f"pend_{pc.CAT_SHORT[c]}"] = point[j]
            row[f"pend_{pc.CAT_SHORT[c]}_lo"], row[f"pend_{pc.CAT_SHORT[c]}_hi"] = np.percentile(bs[:, j], [2.5, 97.5])
        for j, c in [(1, "Medium"), (2, "Extreme")]:
            dif = bs[:, j] - bs[:, 0]
            row[f"dif_{c}"] = point[j] - point[0]
            row[f"dif_{c}_lo"], row[f"dif_{c}_hi"] = np.percentile(dif, [2.5, 97.5])
            row[f"p_mayor_{c}"] = float((dif > 0).mean())
        out.append(row)
        logger.info("  interacción %s: No %.2f, Ext %.2f (dif %+.2f, P=%.2f)", m, point[0], point[2], row["dif_Extreme"], row["p_mayor_Extreme"])
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 6. Arquetipos y componentes principales
# ---------------------------------------------------------------------------


def kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 20, n_iter: int = 300) -> tuple[np.ndarray, np.ndarray, float]:
    """k-means con inicialización k-means++ (numpy). Devuelve (etiquetas, centroides, inercia) de la mejor corrida."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C)
        for _ in range(n_iter):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (lab, C, inertia)
    return best


def silhouette(X: np.ndarray, lab: np.ndarray) -> float:
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        a = D[i, same].sum() / max(same.sum() - 1, 1)
        b = min(D[i, lab == j].mean() for j in np.unique(lab) if j != lab[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


ARCHETYPE_RULES = [  # (nombre, condición sobre z de los centroides)
    ("Whiff dominante", lambda z: z["whiff_swing"] > 0.5 and z["stuff_plus_physical"] > 0.3),
    ("Manejo de contacto / rodados", lambda z: z["gb_pct"] > 0.5 and z["hard_pct"] > 0.0),
    ("Stuff alto / comando débil", lambda z: z["stuff_plus_physical"] > 0.3 and z["bb_pct"] < -0.3),
    ("Comando / Stuff bajo", lambda z: z["bb_pct"] > 0.3 and z["stuff_plus_physical"] < -0.3),
]


def archetypes(t: pd.DataFrame, eligible: pd.Series) -> dict:
    e = t[eligible].reset_index(drop=True)
    Xraw = e[ARCH_FEATURES].astype(float)
    sign = np.array([1 if PROFILE_DIMS[c][0] else -1 for c in ARCH_FEATURES])  # orientadas: + = mejor para el pitcher
    X = ((Xraw - Xraw.mean()) / Xraw.std()).to_numpy() * sign
    # componentes principales
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pca = {"varianza_explicada": (S ** 2 / (S ** 2).sum()).tolist(),
           "cargas": pd.DataFrame(Vt[:3].T, index=ARCH_FEATURES, columns=["PC1", "PC2", "PC3"])}
    sil = {}
    fits = {}
    for k in range(2, 8):
        lab, C, _ = kmeans(X, k, SEED)
        fits[k] = (lab, C)
        sil[k] = silhouette(X, lab)
    k_best = max(sil, key=sil.get)
    lab, C = fits[k_best]
    rng = np.random.default_rng(SEED)
    aris = []
    for _ in range(30):
        idx = rng.integers(0, len(X), len(X))
        _, Cb, _ = kmeans(X[idx], k_best, int(rng.integers(1e9)), n_init=5)
        lab_b = ((X[:, None, :] - Cb[None]) ** 2).sum(-1).argmin(1)
        aris.append(parks.ari(pd.Series(lab), pd.Series(lab_b)))
    stab = float(np.median(aris))
    named = sil[k_best] >= 0.25 and stab >= 0.6
    cent = pd.DataFrame(C, columns=ARCH_FEATURES)
    labels = []
    for j, z in cent.iterrows():
        desc = ", ".join(f"{'+' if v > 0 else '−'}{c}" for c, v in z.items() if abs(v) >= 0.5) or "cerca del promedio en todo"
        name = next((n for n, rule in ARCHETYPE_RULES if rule(z)), None) if named else None
        labels.append(name or f"Perfil {j + 1}")
        cent.loc[j, "descripcion"] = desc
    cent["nombre"] = labels
    cent["n"] = np.bincount(lab, minlength=k_best)
    e["arquetipo"] = [labels[j] for j in lab]
    cent["aepp_medio"] = e.groupby("arquetipo")["aepp"].mean().reindex(labels).to_numpy()
    cent["bono_altitud_medio"] = e.groupby("arquetipo")["bono_altitud"].mean().reindex(labels).to_numpy()
    return {"pca": pca, "silueta": sil, "k": k_best, "ari_estabilidad": stab, "nombres_validos": named, "centroides": cent,
            "asignacion": e[["pitcher_anon_id", "arquetipo"]]}


# ---------------------------------------------------------------------------
# 7. Tipos de pitch y pitch mix
# ---------------------------------------------------------------------------


def pitch_type_effects(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame[frame["altitude_category"].isin(pc.CATS) & frame["batter_anon_id"].notna()].copy()
    d["count_id"] = d["Balls"].astype(str) + "-" + d["Strikes"].astype(str)
    hand = np.where(d["PitcherThrows"] == "Left", -1.0, 1.0)
    d["hb_abs"] = (d["HorzBreak"] * hand).abs()
    d["mov_total"] = np.sqrt(d["InducedVertBreak"] ** 2 + d["HorzBreak"] ** 2)
    d["ptg"] = d["pitcher_anon_id"].astype(str) + "|" + d["AutoPitchType"].astype(str) + "|" + d["year"].astype(str)
    fe = ["pitcher_anon_id", "batter_anon_id", "year", "count_id"]
    rows = []
    for t in pc.MAIN_TYPES:
        x = d[d["AutoPitchType"] == t]
        sw, bp = x[x["is_swing"] == 1], x[x["is_bip"] == 1]
        row = {"tipo": t, "pitches": len(x), "pct_extreme": float((x["altitude_category"] == "Extreme Altitude").mean()),
               "whiff_swing_no": float(sw.loc[sw["altitude_category"] == "No Altitude", "is_whiff"].mean()),
               "rv_bip_no": float(bp.loc[bp["altitude_category"] == "No Altitude", "run_value"].mean())}
        specs = [("whiff_swing", sw.assign(v=sw["is_whiff"].astype(float)), "v", fe),
                 ("bip_swing", sw.assign(v=sw["is_bip"].astype(float)), "v", fe),
                 ("rv_bip", bp.reset_index(drop=True), "run_value", fe),
                 ("delta_por_100", x.assign(v=100 * x["realized_delta"]), "v", fe),
                 ("ivb", x, "InducedVertBreak", ["ptg"]), ("hb_abs", x, "hb_abs", ["ptg"]),
                 ("mov_total", x, "mov_total", ["ptg"]), ("velocidad", x, "RelSpeed", ["ptg"])]
        for name, data, y, f in specs:
            r = hyp.fe_estimate(data.reset_index(drop=True), y, f)
            row[f"{name}_E"], row[f"{name}_E_se"] = r["beta_extreme"], r["se_extreme"]
            row[f"{name}_M"], row[f"{name}_M_se"] = r["beta_medium"], r["se_medium"]
            if name in ("ivb", "hb_abs", "mov_total"):
                base = data.loc[data["altitude_category"] == "No Altitude", y]
                row[f"{name}_no"] = float(base.abs().mean()) if name == "ivb" else float(base.mean())
        rows.append(row)
        logger.info("  tipo %s listo", t)
    return pd.DataFrame(rows)


def mix_robustness(frame: pd.DataFrame, effects: pd.DataFrame, sens_t: pd.DataFrame) -> dict:
    """Costo esperado por la mezcla de pitches (Σ participación × efecto del tipo en Extreme) vs. sensibilidad observada."""
    mix = pd.crosstab(frame["pitcher_anon_id"], frame["AutoPitchType"].where(frame["AutoPitchType"].isin(pc.MAIN_TYPES), "Otro"),
                      normalize="index")
    cost = effects.set_index("tipo")["delta_por_100_E"]
    mix_cost = (mix[pc.MAIN_TYPES] * cost).sum(axis=1) + mix.get("Otro", 0) * float(cost.mean())
    s = sens_t.set_index("pitcher_anon_id").join(mix_cost.rename("costo_mezcla"), how="inner").dropna(subset=["s_emp"])
    return {"r_costo_mezcla_s_emp": float(s["costo_mezcla"].corr(s["s_emp"])),
            "r_costo_mezcla_s_mod": float(s["costo_mezcla"].corr(s["s_mod_rel"])), "n": int(len(s)),
            "mezcla": mix, "costo_mezcla": mix_cost}


# ---------------------------------------------------------------------------
# 8. Sensibilidad (secundaria) y por qué su confiabilidad es baja
# ---------------------------------------------------------------------------


def sensitivity_tables(frame: pd.DataFrame, pa: pd.DataFrame) -> dict:
    t = pd.read_parquet(sens.OUT_PATH)
    _, info = sens.empirical_sensitivity()
    s = t.dropna(subset=["s_emp"]).copy()
    B = 1 - s["s_emp_shrunk"] / s["s_emp"].where(s["s_emp"] != 0)
    s["factor_shrinkage"] = B
    s["post_sd"] = np.sqrt((1 - B).clip(0, 1) * s["s_emp_se"] ** 2)
    s["s_emp_lo80"], s["s_emp_hi80"] = s["s_emp"] - pc.Z80 * s["s_emp_se"], s["s_emp"] + pc.Z80 * s["s_emp_se"]
    s["s_shrunk_lo80"], s["s_shrunk_hi80"] = s["s_emp_shrunk"] - pc.Z80 * s["post_sd"], s["s_emp_shrunk"] + pc.Z80 * s["post_sd"]
    s["rank_menos_sensible"] = s["s_emp_shrunk"].rank(method="first").astype(int)
    s["confiabilidad_global"] = info["confiabilidad_spearman_brown"]
    rel = info["confiabilidad_spearman_brown"]
    need = {r: float(np.median(s["n_extreme"]) * (r / (1 - r)) / (rel / (1 - rel))) for r in (0.5, 0.7)} if rel > 0 else {}

    # factores que podrían explicar la baja confiabilidad
    f = frame[frame["altitude_category"].isin(["No Altitude", "Extreme Altitude"])].copy()
    f["adj"] = f["realized_delta"] - f.groupby(["year", "altitude_category"])["realized_delta"].transform("mean")
    # a) estabilidad entre temporadas de S_emp
    g = f.groupby(["pitcher_anon_id", "year", "altitude_category"])["adj"].agg(["mean", "size"]).unstack("altitude_category")
    g = g[(g[("size", "Extreme Altitude")] >= 75) & (g[("size", "No Altitude")] >= 75)]
    sy = (100 * (g[("mean", "Extreme Altitude")] - g[("mean", "No Altitude")])).rename("s").reset_index()
    sp = pc.season_pairs(sy.assign(n=1), ["s"], "n", 1)
    r_yoy = pc.boot_corr(sp, "s_t", "s_t1", n_boot=500) if len(sp) > 10 else None
    # b) cambio de mezcla entre ambientes (distancia de variación total)
    mixes = pd.crosstab([f["pitcher_anon_id"], f["altitude_category"]], f["AutoPitchType"], normalize="index")
    diff = (mixes.xs("Extreme Altitude", level=1) - mixes.xs("No Altitude", level=1)).dropna(how="all")
    tv = (0.5 * diff.abs().sum(axis=1)).rename("cambio_mezcla")
    # c) calidad de los bateadores enfrentados
    bq = pa.groupby("batter_anon_id")["pa_rv_adj"].agg(["mean", "size"])
    bq = bq["mean"].where(bq["size"] >= 50)
    pq = pa.assign(bq=pa["batter_anon_id"].map(bq)).dropna(subset=["bq"])
    pq = pq[pq["altitude_category"].isin(["No Altitude", "Extreme Altitude"])]
    opp = pq.groupby(["pitcher_anon_id", "altitude_category"])["bq"].mean().unstack()
    opp = (100 * (opp["Extreme Altitude"] - opp["No Altitude"])).rename("rivales_E_menos_N")
    # d) estadios de Extreme donde lanzó (drag promedio de sus pitches en Extreme)
    drag = parks.load_stadium_ratios()["drag_ratio"]
    ex = f[f["altitude_category"] == "Extreme Altitude"].assign(drag=lambda x: x["stadium_resuelto"].map(drag))
    expo = ex.groupby("pitcher_anon_id")["drag"].mean().rename("drag_extreme")
    # e) rol y equipo local
    app = frame.groupby(["pitcher_anon_id", "game_anon_id"])["Inning"].min()
    starter = (app == 1).groupby(level=0).mean().rename("pct_aperturas")
    games = pd.read_parquet(parks.OUT_PATH)
    home_cat = games.dropna(subset=["stadium_category"]).groupby(["year", "home_team"])["stadium_category"].agg(lambda x: x.mode().iloc[0])
    pt = frame.dropna(subset=["pitcher_team"]).drop_duplicates(["pitcher_anon_id", "year", "game_anon_id"])
    pt = pt.assign(home_cat=[home_cat.get((y, tm), np.nan) for y, tm in zip(pt["year"], pt["pitcher_team"])])
    local_ext = (pt["home_cat"] == "Extreme Altitude").groupby(pt["pitcher_anon_id"]).mean().rename("pct_juegos_equipo_local_extreme")
    s = s.set_index("pitcher_anon_id").join([tv, opp, expo, starter, local_ext]).reset_index()
    factors = {}
    for c in ["n_extreme", "cambio_mezcla", "rivales_E_menos_N", "drag_extreme", "pct_aperturas", "pct_juegos_equipo_local_extreme"]:
        x = s[[c, "s_emp"]].dropna()
        factors[c] = {"r_con_s_emp": float(x[c].corr(x["s_emp"])), "n": len(x)}
    s["rol"] = np.where(s["pct_aperturas"] >= 0.5, "abridor", "relevista")
    by_role = s.groupby("rol")["s_emp"].agg(["mean", "std", "size"])
    s["equipo_local_extreme"] = s["pct_juegos_equipo_local_extreme"] >= 0.5
    by_home = s.groupby("equipo_local_extreme")["s_emp"].agg(["mean", "std", "size"])
    # f) selección: quiénes llegan a tener S_emp
    off = t.assign(tiene=t["s_emp"].notna())
    return {"tabla": s, "info": info, "pitches_necesarios": need, "r_entre_temporadas": r_yoy, "n_pares_temporadas": len(sp),
            "factores": factors, "por_rol": by_role, "por_equipo_local": by_home,
            "seleccion": off.groupby("tiene")[["pitching_plus_neutral", "n_pitches"]].mean(),
            "r_mod_emp": float(s["s_mod_rel"].corr(s["s_emp"])), "r_mod_emp_spearman": float(s["s_mod_rel"].corr(s["s_emp"], method="spearman")),
            "s_mod": t.dropna(subset=["s_mod_rel"])}


# ---------------------------------------------------------------------------
# 9. Explicaciones por pitcher
# ---------------------------------------------------------------------------


def explanations(e: pd.DataFrame) -> pd.Series:
    pct = {}
    for c, (higher, _, _) in PROFILE_DIMS.items():
        r = e[c].rank(pct=True)
        pct[c] = r if higher else 1 - r + 1 / len(e)
    pct = pd.DataFrame(pct)
    texts = []
    for i, row in e.iterrows():
        strong = [PROFILE_DIMS[c][1] for c in PROFILE_DIMS if pct.loc[i, c] >= 0.75]
        weak = [PROFILE_DIMS[c][2] for c in PROFILE_DIMS if pct.loc[i, c] <= 0.25]
        txt = ("Aparece alto porque: " + "; ".join(strong) + "." if strong else "Ninguna característica sobresale (todas entre los percentiles 25 y 75).")
        if weak:
            txt += " Puntos débiles: " + "; ".join(weak) + "."
        txt += " Su perfil presenta características que históricamente son más favorables en estos contextos"
        txt += "." if row["aepp"] > 0 else " solo en parte."
        if pd.notna(row.get("rank_lo80")) and row["rank_hi80"] - row["rank_lo80"] > 60:
            txt += f" Incertidumbre alta: su lugar podría estar entre #{int(row['rank_lo80'])} y #{int(row['rank_hi80'])}."
        texts.append(txt)
    return pd.Series(texts, index=e.index), pct


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def _stage(name: str, fn, *args):
    """Etapas lentas. Con la variable de entorno P2_STAGE_CACHE=1 reutiliza el resultado guardado (solo para desarrollo;
    la ejecución oficial del pipeline recalcula todo)."""
    path = STAGE_DIR / f"{name}.pkl"
    if os.environ.get("P2_STAGE_CACHE") == "1" and path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    out = fn(*args)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def compute() -> dict:
    frame, pa = load()
    logger.info("Marco: %s pitches, %s turnos", f"{len(frame):,}", f"{len(pa):,}")
    hyp_json = json.loads(HYP_JSON.read_text(encoding="utf-8"))
    logger.info("Consistencia por estadio-temporada…")
    consistency = _stage("consistencia", stadium_season_consistency, frame)
    env_rates = environment_rates(frame, pa)
    logger.info("Daño de contacto…")
    damage = _stage("dano", contact_damage, frame, pa)
    lw = linear_weights(pa)

    logger.info("Métricas por pitcher…")
    season, hyper_season = metric_table(frame, ["pitcher_anon_id", "year"])
    season = season.merge(scores_for(["pitcher_anon_id", "year"], frame), on=["pitcher_anon_id", "year"], how="left")
    cov = covariate_check(frame, season)
    adopt = {m: v["adoptar"] for m, v in cov.items()}
    season = apply_covariates(season, frame, ["pitcher_anon_id", "year"], adopt)
    pooled, hyper_pooled = metric_table(frame, ["pitcher_anon_id"])
    pooled = pooled.merge(scores_for(["pitcher_anon_id"], frame), on="pitcher_anon_id", how="left")
    pooled = apply_covariates(pooled, frame, ["pitcher_anon_id"], adopt)
    seasons = frame.groupby("pitcher_anon_id")["year"].agg(lambda x: ",".join(map(str, sorted(x.unique())))).rename("temporadas")
    pooled = pooled.merge(seasons.reset_index(), on="pitcher_anon_id")
    env_n = pd.crosstab(frame["pitcher_anon_id"], frame["altitude_category"]).reindex(columns=pc.CATS, fill_value=0)
    env_n.columns = ["pitches_no_altitude", "pitches_medium", "pitches_extreme"]
    pooled = pooled.merge(env_n.reset_index(), on="pitcher_anon_id")
    eligible = (pooled["n_pitches"] >= MIN_PITCHES_RANK) & (pooled["n_pa"] >= MIN_PA_RANK)
    pooled = profile_scores(pooled, frame, lw, eligible)

    logger.info("Validación del perfil…")
    validation = _stage("validacion", validate_profile, season, frame, lw)
    stability = _stage("estabilidad", metric_stability, season, frame)
    logger.info("Interacciones…")
    inter = _stage("interacciones", interactions, frame)
    logger.info("Arquetipos…")
    arch = archetypes(pooled, eligible)
    pooled = pooled.merge(arch["asignacion"], on="pitcher_anon_id", how="left")
    e = pooled[eligible.to_numpy()].copy()
    exp, pct = explanations(e)
    pooled.loc[e.index, "explicacion"] = exp
    for c in PROFILE_DIMS:
        pooled.loc[e.index, f"pct_{c}"] = pct[c]
    logger.info("Tipos de pitch…")
    effects = _stage("tipos", pitch_type_effects, frame)
    logger.info("Sensibilidad…")
    sens_res = sensitivity_tables(frame, pa)
    mix = mix_robustness(frame, effects, sens_res["tabla"])
    pooled = pooled.merge(mix["mezcla"].add_prefix("mix_").reset_index(), on="pitcher_anon_id", how="left")
    return {"hyp": hyp_json, "consistencia": consistency, "tasas_ambiente": env_rates, "dano": damage, "pesos": lw,
            "temporada": season, "hiper_temporada": hyper_season, "pitchers": pooled, "hiper": hyper_pooled, "elegibles": eligible,
            "covariables": cov, "validacion": validation, "estabilidad_metricas": stability, "interacciones": inter,
            "arquetipos": arch, "tipos": effects, "mezcla": {k: v for k, v in mix.items() if k not in ("mezcla",)},
            "sensibilidad": sens_res}


def run(refresh: bool = False) -> dict:
    if RESULTS_PATH.exists() and not refresh:
        with open(RESULTS_PATH, "rb") as fh:
            return pickle.load(fh)
    res = compute()
    with open(RESULTS_PATH, "wb") as fh:
        pickle.dump(res, fh)
    return res


if __name__ == "__main__":
    run(refresh=True)
