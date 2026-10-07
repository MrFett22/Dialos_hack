"""Location+ v2 — ¿qué tan favorable es la ubicación donde terminó el pitch, dado su tipo y su contexto?

Decisiones declaradas ANTES de entrenar (P2, 2026-10-06):

1. PREGUNTA. Dado el tipo de pitch, el conteo y el enfrentamiento de manos,
   ¿cuántas carreras esperadas vale haber terminado en ESTA ubicación en lugar
   de en una ubicación típica para ese mismo contexto?
2. TARGET: el valor esperado del pitch en carreras con la misma cadena
   condicional de Pitching+ (swing → take {bola, strike cantado, golpeado} /
   swing {whiff, foul, en juego → out/1B/2B/3B/HR}), valuada con Δ(evento, conteo)
   de la matriz RE24 por conteo. Se usa el árbol completo y no un solo evento
   (p. ej. strike cantado) porque la ubicación cambia a la vez si el bateador
   tira, si falla y qué tan fuerte conecta.
3. FEATURES (solo ubicación y contexto):
   - `loc_in`: PlateLocSide con signo relativo al bateador (positivo = hacia el
     cuerpo del bateador, "adentro"). El signo se verificó con los golpeados:
     a los derechos les pegan con PlateLocSide ≈ +2 pies y a los zurdos ≈ −2.
   - `PlateLocHeight`; `Balls`, `Strikes`; `AutoPitchType`; `same_side` (mismo
     lado o lado contrario); `bat_side` (derecho/zurdo).
   Los árboles capturan las interacciones tipo × ubicación, mano × ubicación y
   conteo × ubicación sin especificarlas a mano. No se incluye ninguna variable
   física (velocidad, movimiento): el tipo de pitch ya condiciona lo esencial y
   meter física mezclaría Location+ con Stuff+.
4. PUNTAJE POR PITCH: loc_value = E[Δ | ubicación, contexto] − promedio de
   E[Δ] en el mismo contexto (tipo × conteo × mismo lado × mano del bateador).
   Así un slider no recibe crédito por ser slider ni un 3-0 castigo por ser 3-0:
   solo cuenta dónde terminó.
5. ESCALA: igual que Pitching+ y Stuff+ Physical (shrinkage empírico-bayesiano por
   temporada, 100 + 10·(μ − θ̂)/τ; 10 puntos = 1 sd de habilidad real).
   OOF por pitcher (GroupKFold k=5), hiperparámetros base.
6. AMBIENTE: no entra como feature. Se verifica aparte si el valor de cada
   región de la zona cambia entre ambientes (efectos fijos de pitcher, bateador,
   temporada y conteo).

Salidas: data/processed/location_plus_v2_pitches.parquet, data/processed/location_plus_v2_scores.parquet,
data/processed/location_plus_v2_grid.parquet, reports/location_plus_v2.md, reports/figures/p2/location_*.png,
reports/benchmark/location_plus_v2.json.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.analysis import altitude_hypotheses as hyp
from src.analysis import p2_common as pc
from src.models import pitching_plus_neutral as neutral_mod
from src.models import run_value_chain as chain

ROOT = Path(__file__).resolve().parents[2]
PITCHES_PATH = ROOT / "data" / "processed" / "location_plus_v2_pitches.parquet"
SCORES_PATH = ROOT / "data" / "processed" / "location_plus_v2_scores.parquet"
GRID_PATH = ROOT / "data" / "processed" / "location_plus_v2_grid.parquet"
REPORT_PATH = ROOT / "reports" / "location_plus_v2.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "location_plus_v2.json"
FINAL_PATH = ROOT / "src" / "models" / "artifacts" / "location_plus_v2.pkl"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NUMERIC = ["loc_in", "PlateLocHeight", "Balls", "Strikes"]
CATEGORICAL = ["AutoPitchType", "same_side", "bat_side"]
FEATURES = {"numeric": NUMERIC, "categorical": CATEGORICAL}
CONTEXT = ["AutoPitchType", "Balls", "Strikes", "same_side", "bat_side"]
COUNT_GROUPS = {"pitcher adelante": lambda b, s: s > b, "conteo parejo": lambda b, s: s == b, "pitcher atrás": lambda b, s: b > s}
# Regiones aproximadas de la zona (pies). Zona: |x| ≤ 0.83 (17" + una pelota), altura 1.5–3.5.
GRID_X = np.round(np.arange(-2.0, 2.0001, 0.1), 2)
GRID_Z = np.round(np.arange(0.5, 4.5001, 0.1), 2)
MASK_CELL = 0.25
MASK_MIN = 20


def region(x: pd.Series, z: pd.Series) -> pd.Series:
    ax = x.abs()
    heart = (ax <= 0.56) & z.between(1.83, 3.17)
    shadow = (ax <= 1.11) & z.between(1.17, 3.83)
    chase = (ax <= 1.67) & z.between(0.5, 4.5)
    return pd.Series(np.select([heart, shadow, chase], ["corazón", "borde", "persecución"], default="desperdicio"), index=x.index)


def load_data() -> pd.DataFrame:
    df = chain.load_data()
    raw = pd.read_parquet(pc.ARSENAL_PATH, columns=["PitchUID", "PlateLocSide", "BatterSide"]).rename(
        columns={"PlateLocSide": "plate_side_raw", "BatterSide": "bat_raw"})
    df = df.merge(raw, on="PitchUID", how="left")
    bat = df["bat_raw"].astype("object")
    df["bat_side"] = pd.Categorical(np.where(bat.isin(["Right", "Left"]), bat, "Other"), categories=["Right", "Left", "Other"])
    df["loc_in"] = df["plate_side_raw"] * np.where(bat == "Left", -1.0, 1.0)
    return df


def _delta_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    term = chain.terminal_values(df)
    _, deltas = chain.count_values(df, term)
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    return deltas.to_numpy()[[pos[bs] for bs in zip(df["Balls"], df["Strikes"])]], deltas


def generate_oof(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    if PITCHES_PATH.exists() and not refresh:
        logger.info("Usando OOF en cache: %s", PITCHES_PATH.name)
        return pd.read_parquet(PITCHES_PATH), json.loads(PITCHES_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    df = load_data()
    masks = chain.universe_masks(df)
    folds = list(GroupKFold(n_splits=chain.N_FOLDS).split(df, groups=df["pitcher_anon_id"]))
    preds, metrics = {}, {}
    for name, universe, target, n_classes in chain.SUBMODELS:
        t0 = time.time()
        preds[name] = chain.oof_submodel(df, folds, FEATURES, masks[universe], target, n_classes)
        m = masks[universe]
        y = df.loc[m, target].astype(int).to_numpy()
        p = preds[name][m]
        metrics[name] = ({"n": int(m.sum()), "auc": float(roc_auc_score(y, p[:, 0])), "log_loss": float(log_loss(y, p[:, 0]))}
                         if n_classes == 2 else {"n": int(m.sum()), "auc": None,
                                                 "log_loss": float(log_loss(y, p, labels=list(range(n_classes))))})
        logger.info("  %s: %.0fs", name, time.time() - t0)
    probs = chain.event_probabilities(preds)
    D, _ = _delta_matrix(df)
    out = df[["PitchUID", "pitcher_anon_id", "year", "Balls", "Strikes", "loc_in", "PlateLocHeight"]].copy()
    for c in ["AutoPitchType", "same_side", "bat_side"]:
        out[c] = df[c].astype("object")
    out["xrv_loc"] = chain.expected_run_value(probs, D)
    out["p_sum"] = probs.sum(axis=1).to_numpy()
    out["loc_value"] = out["xrv_loc"] - out.groupby(CONTEXT)["xrv_loc"].transform("mean")
    out["region"] = region(out["loc_in"], out["PlateLocHeight"])
    out.to_parquet(PITCHES_PATH, index=False)
    meta = {"submodelos": metrics, "max_abs_p_sum_menos_1": float((out["p_sum"] - 1).abs().max())}
    PITCHES_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta


# ---------------------------------------------------------------------------
# Puntajes por pitcher
# ---------------------------------------------------------------------------


def build_scores(pitches: pd.DataFrame, pooled: bool = False) -> pd.DataFrame:
    d = pitches.assign(year=0) if pooled else pitches
    eb, _ = neutral_mod.empirical_bayes(d, "loc_value")
    eb = eb.rename(columns={"n": "n_pitches", "plus": "location_plus", "plus_lo80": "location_plus_lo80",
                            "plus_hi80": "location_plus_hi80", "plus_raw": "location_plus_sin_shrinkage", "mean": "loc_value_por_pitch"})
    cols = ["pitcher_anon_id", "year", "n_pitches", "loc_value_por_pitch", "shrink_factor", "location_plus",
            "location_plus_lo80", "location_plus_hi80", "location_plus_sin_shrinkage"]
    eb = eb[cols]
    return eb.drop(columns="year") if pooled else eb


# ---------------------------------------------------------------------------
# Mapas de calor (modelos finales)
# ---------------------------------------------------------------------------


def heatmap_grid(df: pd.DataFrame) -> pd.DataFrame:
    """loc_value predicho en una rejilla, por tipo × enfrentamiento × grupo de conteo (pitcher derecho)."""
    masks = chain.universe_masks(df)
    cols = NUMERIC + CATEGORICAL
    models = {}
    for name, universe, target, _ in chain.SUBMODELS:
        m = lgb.LGBMClassifier(verbosity=-1, **chain.LGB_PARAMS)
        m.fit(df.loc[masks[universe], cols], df.loc[masks[universe], target].astype(int), categorical_feature=CATEGORICAL)
        models[name] = m
    D, deltas = _delta_matrix(df)
    # referencia de cada contexto: E[Δ] promedio del MISMO modelo final sobre los pitches reales de ese contexto
    pr_all = {name: m.predict_proba(df[cols]) for name, m in models.items()}
    pr_all = {k: (v if v.shape[1] > 2 else v[:, 1:2]) for k, v in pr_all.items()}
    xrv_all = pd.Series(chain.expected_run_value(chain.event_probabilities(pr_all), D))
    ref = xrv_all.groupby([df["AutoPitchType"].astype(str), df["same_side"].astype(str), df["bat_side"].astype(str),
                           df["Balls"], df["Strikes"]]).mean()
    # modelos finales + referencia por contexto: lo que usa el simulador de la página web
    with open(FINAL_PATH, "wb") as fh:
        pickle.dump({"models": models, "numeric_features": NUMERIC, "categorical_features": CATEGORICAL, "deltas_by_count": deltas,
                     "reference_by_context": ref.to_dict(), "context": CONTEXT,
                     "categories": {c: list(df[c].cat.categories) for c in CATEGORICAL}}, fh)
    xx, zz = np.meshgrid(GRID_X, GRID_Z)
    grid = pd.DataFrame({"loc_in": xx.ravel(), "PlateLocHeight": zz.ravel()})
    count_freq = df.groupby(["AutoPitchType", "Balls", "Strikes"], observed=True).size()
    rows = []
    for ptype in pc.MAIN_TYPES:
        for matchup, bat in (("mismo lado", "Right"), ("lado contrario", "Left")):
            side = "same" if matchup == "mismo lado" else "opposite"
            ctx = (df["AutoPitchType"] == ptype) & (df["same_side"] == side) & (df["bat_side"] == bat)
            for b in range(4):
                for s in range(3):
                    X = grid.assign(Balls=b, Strikes=s)
                    for c, v in (("AutoPitchType", ptype), ("same_side", side), ("bat_side", bat)):
                        X[c] = pd.Categorical([v] * len(X), categories=df[c].cat.categories)
                    pr = {name: m.predict_proba(X[cols]) for name, m in models.items()}
                    pr = {k: (v if v.shape[1] > 2 else v[:, 1:2]) for k, v in pr.items()}
                    probs = chain.event_probabilities(pr)
                    xrv = (probs.to_numpy() * deltas.loc[(b, s)].to_numpy()).sum(axis=1)
                    # referencia: E[Δ] promedio de los pitches reales en ese contexto (misma definición que loc_value)
                    real = ctx & (df["Balls"] == b) & (df["Strikes"] == s)
                    rows.append(pd.DataFrame({"tipo": ptype, "enfrentamiento": matchup, "Balls": b, "Strikes": s,
                                              "loc_in": X["loc_in"], "PlateLocHeight": X["PlateLocHeight"], "xrv": xrv,
                                              "ref": ref.get((ptype, side, bat, b, s), np.nan),
                                              "peso_conteo": float(count_freq.get((ptype, b, s), 0)), "n_real": int(real.sum())}))
    g = pd.concat(rows, ignore_index=True)
    return g


def summarize_grid(g: pd.DataFrame, pitches: pd.DataFrame) -> pd.DataFrame:
    """Promedia la rejilla dentro de cada grupo de conteo ; `ref` (promedio del contexto, mismo modelo final) viene de `heatmap_grid`."""
    g = g.copy()
    g["valor"] = g["xrv"] - g["ref"]
    g["grupo_conteo"] = [next(k for k, f in COUNT_GROUPS.items() if f(b, s)) for b, s in zip(g["Balls"], g["Strikes"])]
    g = g.dropna(subset=["valor"])
    out = (g.assign(w=g["peso_conteo"], wv=g["valor"] * g["peso_conteo"])
           .groupby(["tipo", "enfrentamiento", "grupo_conteo", "loc_in", "PlateLocHeight"])[["w", "wv"]].sum().reset_index())
    out["carreras_por_100"] = 100 * out["wv"] / out["w"]
    # soporte: pitches reales del tipo × enfrentamiento × grupo de conteo en la celda de MASK_CELL pies
    p = pitches.copy()
    p["grupo_conteo"] = [next(k for k, f in COUNT_GROUPS.items() if f(b, s)) for b, s in zip(p["Balls"], p["Strikes"])]
    p["enfrentamiento"] = np.select([(p["same_side"] == "same") & (p["bat_side"] == "Right"),
                                     (p["same_side"] == "opposite") & (p["bat_side"] == "Left")], ["mismo lado", "lado contrario"], "otro")
    p = p[p["enfrentamiento"] != "otro"]
    p["cx"] = np.floor(p["loc_in"] / MASK_CELL)
    p["cz"] = np.floor(p["PlateLocHeight"] / MASK_CELL)
    sup = p.groupby(["AutoPitchType", "enfrentamiento", "grupo_conteo", "cx", "cz"]).size()
    out["n_celda"] = [int(sup.get((t, m, c, np.floor(x / MASK_CELL), np.floor(z / MASK_CELL)), 0)) for t, m, c, x, z in
                      zip(out["tipo"], out["enfrentamiento"], out["grupo_conteo"], out["loc_in"], out["PlateLocHeight"])]
    out["con_soporte"] = out["n_celda"] >= MASK_MIN
    return out.drop(columns=["w", "wv"])


def plot_heatmaps(grid: pd.DataFrame) -> list[Path]:
    pc.FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    vmax = float(np.nanpercentile(grid.loc[grid["con_soporte"], "carreras_por_100"].abs(), 98))
    for ptype in pc.MAIN_TYPES:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8.5), sharex=True, sharey=True)
        for i, matchup in enumerate(["mismo lado", "lado contrario"]):
            for j, cg in enumerate(COUNT_GROUPS):
                ax = axes[i, j]
                sub = grid[(grid["tipo"] == ptype) & (grid["enfrentamiento"] == matchup) & (grid["grupo_conteo"] == cg)]
                piv = sub.pivot(index="PlateLocHeight", columns="loc_in", values="carreras_por_100")
                sup = sub.pivot(index="PlateLocHeight", columns="loc_in", values="con_soporte").astype(bool)
                im = ax.imshow(piv.where(sup).to_numpy(), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               extent=[GRID_X[0] - 0.05, GRID_X[-1] + 0.05, GRID_Z[0] - 0.05, GRID_Z[-1] + 0.05], aspect="equal")
                ax.add_patch(plt.Rectangle((-0.83, 1.5), 1.66, 2.0, fill=False, lw=1.5, color="black"))
                ax.set_title(f"{matchup} · {cg}", fontsize=10)
                if i == 1:
                    ax.set_xlabel("afuera ←  → adentro (pies)")
                if j == 0:
                    ax.set_ylabel("altura (pies)")
        fig.colorbar(im, ax=axes, shrink=0.8, label="carreras por 100 pitches vs. ubicación típica\n(azul = más favorable al pitcher)")
        fig.suptitle(f"Location+ · {ptype} · pitcher derecho\nZonas que SUELEN ser más (azul) o menos (rojo) favorables. "
                     "Blanco = pocos pitches reales; no garantiza ningún resultado.", fontsize=11)
        path = pc.FIG_DIR / f"location_{ptype.lower().replace('-', '')}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Valor de las regiones por ambiente
# ---------------------------------------------------------------------------


def region_by_environment(pitches: pd.DataFrame) -> pd.DataFrame:
    """Δ realizado por región de la zona: efecto Medium/Extreme con efectos fijos de pitcher, bateador, temporada y conteo."""
    frame = pc.load_pitch_frame()[["PitchUID", "game_anon_id", "batter_anon_id", "altitude_category", "realized_delta"]]
    d = pitches[["PitchUID", "pitcher_anon_id", "year", "Balls", "Strikes", "region"]].merge(frame, on="PitchUID")
    d = d[d["altitude_category"].isin(pc.CATS) & d["batter_anon_id"].notna()].copy()
    d["count_id"] = d["Balls"].astype(str) + "-" + d["Strikes"].astype(str)
    rows = []
    for reg in ["corazón", "borde", "persecución", "desperdicio"]:
        sub = d[d["region"] == reg].reset_index(drop=True)
        fx = hyp.fe_estimate(sub, "realized_delta", ["pitcher_anon_id", "batter_anon_id", "year", "count_id"])
        rows.append({"region": reg, "n": fx["n"], "pct_pitches": len(sub) / len(d),
                     "delta_medio_no_altitude_por_100": 100 * fx["media_no_altitude"],
                     "medium_menos_no_por_100": 100 * fx["beta_medium"], "ee_medium": 100 * fx["se_medium"],
                     "extreme_menos_no_por_100": 100 * fx["beta_extreme"], "ee_extreme": 100 * fx["se_extreme"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validación y reporte
# ---------------------------------------------------------------------------


def validate(scores: pd.DataFrame) -> dict:
    from src.models import stuff_plus_physical as spp
    frame = pc.load_pitch_frame()
    obs = spp.outcomes_by_pitcher_season(frame)
    f = frame.copy()
    pa = f["is_pa_end"] == 1
    f["bb_adj"] = pc.env_adjust(f, "is_bb", pa)
    bb = f.groupby(["pitcher_anon_id", "year"])["bb_adj"].mean().rename("bb_pct_t").reset_index()
    stuff = pd.read_parquet(spp.SCORES_PATH, columns=["pitcher_anon_id", "year", "stuff_plus_physical"])
    t = obs.merge(bb, on=["pitcher_anon_id", "year"]).merge(scores.drop(columns="n_pitches"), on=["pitcher_anon_id", "year"]).merge(
        stuff, on=["pitcher_anon_id", "year"], how="left")
    out = {}
    for mn in [100, 300]:
        pairs = pc.season_pairs(t, ["location_plus", "bb_pct_t", "k_pct_t", "runs_prevented_100"], "n_pitches", mn)
        out[mn] = {"n_pares": len(pairs),
                   "estabilidad": pc.boot_corr(pairs, "location_plus_t", "location_plus_t1"),
                   "bb_t1": pc.boot_corr(pairs, "location_plus_t", "bb_pct_t_t1"),
                   "k_t1": pc.boot_corr(pairs, "location_plus_t", "k_pct_t_t1"),
                   "carreras_t1": pc.boot_corr(pairs, "location_plus_t", "runs_prevented_100_t1"),
                   "bb_propio_t1": pc.boot_corr(pairs, "bb_pct_t_t", "bb_pct_t_t1")}
    big = t[t["n_pitches"] >= 100]
    out["r_stuff_physical"] = float(big["location_plus"].corr(big["stuff_plus_physical"]))
    out["r_bb_misma_temporada"] = float(big["location_plus"].corr(big["bb_pct_t"]))
    return out


def format_report(meta: dict, val: dict, reg: pd.DataFrame, scores: pd.DataFrame, pitches: pd.DataFrame, figs: list[Path]) -> str:
    L = [
        "# Location+ v2 — valor de la ubicación dado el tipo de pitch y el contexto",
        "",
        "Generado por `src/models/location_plus_v2.py`. Diseño fijado en el docstring antes de entrenar.",
        "",
        "## 1. Qué mide",
        "",
        "Dado el tipo de pitch, el conteo y el enfrentamiento de manos, ¿cuántas carreras esperadas vale haber terminado en "
        "**esta** ubicación en lugar de en una ubicación típica para ese mismo contexto? Usa la misma cadena de probabilidades "
        "de Pitching+ con solo ubicación y contexto (sin velocidad ni movimiento).",
        "",
        "Features: `loc_in` (lado relativo al bateador; positivo = adentro, signo verificado con los golpeados), "
        "`PlateLocHeight`, `Balls`, `Strikes`, `AutoPitchType`, `same_side`, `bat_side`.",
        "",
        "## 2. Sub-modelos (OOF por pitcher)",
        "",
        "| sub-modelo | n | AUC | log-loss |",
        "|---|---:|---:|---:|",
    ]
    for k, m in meta["submodelos"].items():
        L.append(f"| {k} | {m['n']:,} | {m['auc']:.4f} | {m['log_loss']:.4f} |" if m["auc"] is not None else
                 f"| {k} | {m['n']:,} | multiclase | {m['log_loss']:.4f} |")
    L += [
        "",
        f"Las probabilidades de los 10 eventos suman 1 en cada pitch (máx. |Σp − 1| = {meta['max_abs_p_sum_menos_1']:.1e}).",
        "",
        "## 3. Validación por pitcher",
        "",
        "| mín. pitches | pares | estabilidad | BB% t+1 | K% t+1 | carreras evitadas t+1 | (BB% propio → BB% t+1) |",
        "|---|---:|---|---|---|---|---|",
    ]
    for mn in [100, 300]:
        v = val[mn]
        L.append(f"| {mn} | {v['n_pares']} | {pc.fmt_ci(v['estabilidad'])} | {pc.fmt_ci(v['bb_t1'])} | {pc.fmt_ci(v['k_t1'])} | "
                 f"{pc.fmt_ci(v['carreras_t1'])} | {pc.fmt_ci(v['bb_propio_t1'])} |")
    L += [
        "",
        f"- Location+ y BB% del mismo año: r = {val['r_bb_misma_temporada']:+.3f} (signo esperado: negativo, mejor ubicación = menos bases por bolas).",
        f"- Location+ y Stuff+ Physical: r = {val['r_stuff_physical']:+.3f} → miden cosas casi independientes.",
        "",
        "## 4. Valor por región de la zona y ambiente",
        "",
        "Δ realizado (carreras por 100 pitches, positivo = malo para el pitcher) con efectos fijos de pitcher, bateador, "
        "temporada y conteo. Regiones aproximadas: corazón, borde, persecución, desperdicio.",
        "",
        "| región | % de pitches | Δ medio en No Altitude | Medium − No [±1.96 EE] | Extreme − No [±1.96 EE] |",
        "|---|---:|---:|---|---|",
    ]
    for _, r in reg.iterrows():
        L.append(f"| {r['region']} | {r['pct_pitches']:.1%} | {r['delta_medio_no_altitude_por_100']:+.2f} | "
                 f"{r['medium_menos_no_por_100']:+.2f} [{r['medium_menos_no_por_100'] - 1.96 * r['ee_medium']:+.2f}, "
                 f"{r['medium_menos_no_por_100'] + 1.96 * r['ee_medium']:+.2f}] | "
                 f"{r['extreme_menos_no_por_100']:+.2f} [{r['extreme_menos_no_por_100'] - 1.96 * r['ee_extreme']:+.2f}, "
                 f"{r['extreme_menos_no_por_100'] + 1.96 * r['ee_extreme']:+.2f}] |")
    L += [
        "",
        "## 5. Mapas de calor",
        "",
        "Un archivo por tipo de pitch (pitcher derecho; filas = mismo lado / lado contrario; columnas = pitcher adelante, "
        "conteo parejo, pitcher atrás). Azul = ubicación que **suele** ser más favorable al pitcher que la típica de ese "
        f"contexto; rojo = menos favorable. Se ocultan las celdas con menos de {MASK_MIN} pitches reales en {MASK_CELL} × {MASK_CELL} pies. "
        "Un mapa describe promedios: **ninguna zona garantiza un out**.",
        "",
    ]
    L += [f"- `{p.relative_to(ROOT).as_posix()}`" for p in figs]
    L += [
        "",
        f"Distribución de Location+ (pitcher-temporadas): media {scores['location_plus'].mean():.1f}, sd {scores['location_plus'].std():.1f}.",
        "",
    ]
    return "\n".join(L)


def run(refresh: bool = False) -> dict:
    pitches, meta = generate_oof(refresh=refresh)
    scores = build_scores(pitches)
    pooled = build_scores(pitches, pooled=True)
    scores.to_parquet(SCORES_PATH, index=False)
    pooled.to_parquet(SCORES_PATH.with_name("location_plus_v2_pooled.parquet"), index=False)
    if refresh or not GRID_PATH.exists():
        df = load_data()
        grid = summarize_grid(heatmap_grid(df), pitches)
        grid.to_parquet(GRID_PATH, index=False)
    grid = pd.read_parquet(GRID_PATH)
    figs = plot_heatmaps(grid)
    reg = region_by_environment(pitches)
    reg.to_parquet(ROOT / "data" / "processed" / "location_plus_v2_regions.parquet", index=False)
    val = validate(scores)
    REPORT_PATH.write_text(format_report(meta, val, reg, scores, pitches, figs), encoding="utf-8")
    payload = {"meta": meta, "validacion": {str(k): v for k, v in val.items()}, "regiones": reg.to_dict("records")}
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular OOF y rejilla")
    run(refresh=parser.parse_args().refresh)


if __name__ == "__main__":
    main()
