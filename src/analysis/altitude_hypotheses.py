"""Hipótesis del analista del equipo: en altitud extrema conviene evitar el contacto.

Las hipótesis, variables, universos y criterios se fijaron en este docstring
ANTES de correr el análisis por primera vez (2026-10-05).

H1 — ¿La altitud cambia la probabilidad de contacto?
    Universo swings:   y = whiff | swing;  y = bola en juego | swing.
    Universo turnos:   y = ponche | turno terminado (K, BB, HBP o bola en juego).
H2 — ¿El contacto es más peligroso en altitud?
    Universo bolas en juego (InPlay con evento clasificable):
    y = run value RE24 del turno, hit, extra base (2B/3B/HR), HR,
    velocidad de salida (mph), velocidad de salida ≥ 95 mph, barrel.
H3 — ¿Evitar el contacto vale relativamente más en altitud?
    Para cada categoría c se recalculan V_c(b,s) y Δ_c(evento, conteo) con los
    datos de esa categoría (misma recursión que Pitching+ v3). La ventaja de
    un whiff sobre una bola en juego en el conteo (b,s) es
        G_c(b,s) = E_c[Δ_c(k) | bola en juego, (b,s)] − Δ_c(whiff, b, s),
    promediada sobre conteos con los pesos de los swings de toda la liga.
    Se prueba G_Extreme − G_NoAltitude > 0.

Estimación (H1 y H2): modelo lineal con efectos fijos de pitcher, bateador,
temporada y conteo (H1 swings y H2) o solo pitcher, bateador y temporada
(turnos):
    y = β_M·[Medium] + β_E·[Extreme] + α_pitcher + γ_bateador + δ_temporada (+ κ_conteo) + ε
β_E es la diferencia promedio Extreme − No Altitude para el MISMO pitcher contra
el MISMO bateador. Se estima con proyecciones alternadas (teorema de
Frisch–Waugh–Lovell) y errores estándar robustos por juego. También se
reporta la diferencia cruda (sin controles), para mostrar qué hacen los
controles.

H3 se infiere con bootstrap de juegos (se remuestrean juegos dentro de cada
categoría, 500 réplicas).

Criterio (fijado antes de correr): una hipótesis se considera RESPALDADA si
el IC 95% del efecto excluye 0 en la dirección de la hipótesis; CONTRADICHA si
lo excluye en la dirección opuesta; NO CONCLUYENTE en otro caso.

Limitaciones que se reportan junto con el resultado:
- "Extreme" agrupa varios estadios anónimos (src/data/parks.py); el efecto es
  el de esos estadios, no necesariamente "la altitud" en general.
- Con tan pocos estadios por categoría, la altitud se confunde con las demás
  características del parque (dimensiones, clima, calibración de Trackman).
  Los IC por juego son condicionales a ESOS estadios.
- El run value RE24 viene de una sola matriz de toda la liga.

Salidas: reports/hipotesis_altitud.md, reports/benchmark/hipotesis_altitud.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import run_value_chain as chain

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "stuff_model_df.parquet"
REPORT_PATH = ROOT / "reports" / "hipotesis_altitud.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "hipotesis_altitud.json"
PARKS_PATH = ROOT / "data" / "processed" / "anon_stadiums.parquet"
CATS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
N_BOOT = 500
SEED = 42
MAX_ITER = 200
TOL = 1e-10

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def extreme_parks() -> pd.Series:
    """juego → estadio anónimo (ANON_STADIUM_xx, estable entre temporadas), solo grupos resueltos de Extreme (src/data/parks.py)."""
    g = pd.read_parquet(PARKS_PATH)
    g = g[(g["status"] == "resuelto") & (g["stadium_category"] == "Extreme Altitude")]
    return g.set_index("game_anon_id")["stadium_id"]


def load() -> pd.DataFrame:
    df = chain.load_data()
    extra = pd.read_parquet(RAW_PATH, columns=["PitchUID", "game_anon_id", "batter_anon_id", "altitude_category", "ExitSpeed", "Angle"])
    hard = pd.read_parquet(chain.TARGETS_PATH, columns=["PitchUID", "is_hard_contact"])
    df = df.merge(extra, on="PitchUID").merge(hard, on="PitchUID", how="left")
    df = df.loc[df["altitude_category"].isin(CATS) & df["batter_anon_id"].notna()].reset_index(drop=True)
    df["count_id"] = df["Balls"].astype(str) + "-" + df["Strikes"].astype(str)
    df["ExitSpeed"] = pd.to_numeric(df["ExitSpeed"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Efectos fijos por proyecciones alternadas + EE robustos por juego
# ---------------------------------------------------------------------------


def _demean(M: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Resta iterativamente las medias de cada factor (método de proyecciones alternadas)."""
    M = M.copy()
    counts = [np.bincount(g) for g in groups]
    for _ in range(MAX_ITER):
        prev = M.copy()
        for g, c in zip(groups, counts):
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(g, weights=M[:, j], minlength=len(c)) / np.maximum(c, 1))[g]
        if np.max(np.abs(M - prev)) < TOL:
            break
    return M


def _fe_fit(d: pd.DataFrame, y: str, X: np.ndarray, fe: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """β y EE (agrupados por juego) de y sobre las columnas de X, con efectos fijos `fe`."""
    Y = d[y].astype(float).to_numpy()[:, None]
    groups = [pd.factorize(d[f])[0] for f in fe]
    Z = _demean(np.hstack([Y, X]), groups) if groups else np.hstack([Y, X]) - np.hstack([Y, X]).mean(0)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtX_inv = np.linalg.inv(Xt.T @ Xt)
    beta = XtX_inv @ Xt.T @ yt
    e = yt - Xt @ beta
    g = pd.factorize(d["game_anon_id"])[0]
    U = np.column_stack([np.bincount(g, weights=Xt[:, j] * e) for j in range(Xt.shape[1])])
    n_g = U.shape[0]
    V = XtX_inv @ (U.T @ U) @ XtX_inv * n_g / (n_g - 1)
    return beta, np.sqrt(np.diag(V)), n_g


def fe_estimate(d: pd.DataFrame, y: str, fe: list[str]) -> dict:
    """β de [Medium, Extreme] contra No Altitude con efectos fijos `fe`; EE agrupados por juego."""
    d = d.dropna(subset=[y]).reset_index(drop=True)
    X = np.column_stack([(d["altitude_category"] == "Medium Altitude"), (d["altitude_category"] == "Extreme Altitude")]).astype(float)
    beta, se, n_g = _fe_fit(d, y, X, fe)
    base = float(d.loc[d["altitude_category"] == "No Altitude", y].mean())
    return {"n": len(d), "n_games": n_g, "media_no_altitude": base,
            "beta_medium": float(beta[0]), "se_medium": float(se[0]),
            "beta_extreme": float(beta[1]), "se_extreme": float(se[1]),
            "ic95_extreme": [float(beta[1] - 1.96 * se[1]), float(beta[1] + 1.96 * se[1])]}


def fe_by_park(d: pd.DataFrame, y: str, fe: list[str], parks: pd.Series) -> dict:
    """β de cada estadio local reconstruido de Extreme contra No Altitude (Medium y clusters mixtos fuera)."""
    d = d.assign(park=d["game_anon_id"].map(parks))
    d = d[(d["altitude_category"] == "No Altitude") | d["park"].notna()].dropna(subset=[y]).reset_index(drop=True)
    names = sorted(d["park"].dropna().unique())
    X = np.column_stack([(d["park"] == p) for p in names]).astype(float)
    beta, se, _ = _fe_fit(d, y, X, fe)
    return {p: {"beta": float(b), "se": float(s), "n": int((d["park"] == p).sum())} for p, b, s in zip(names, beta, se)}


def verdict(ci: list[float], direction: int) -> str:
    lo, hi = ci
    if direction * lo > 0 and direction * hi > 0:
        return "RESPALDADA"
    if direction * lo < 0 and direction * hi < 0:
        return "CONTRADICHA"
    return "NO CONCLUYENTE"


# ---------------------------------------------------------------------------
# H1 y H2
# ---------------------------------------------------------------------------


def h1_h2(df: pd.DataFrame) -> list[dict]:
    swings = df[df["event_group"] == "swing"].assign(
        whiff=lambda x: (x["event"] == "whiff").astype(float),
        en_juego=lambda x: x["event"].isin(chain.BIP_CLASSES).astype(float))
    pa_end = df["event"].isin(chain.BIP_CLASSES + ["hbp"]) | ((df["event"] == "ball") & (df["Balls"] == 3)) | \
        (df["event"].isin(["called_strike", "whiff"]) & (df["Strikes"] == 2))
    pas = df[pa_end].assign(ponche=lambda x: (x["event"].isin(["called_strike", "whiff"]) & (x["Strikes"] == 2)).astype(float))
    bip = df[df["event"].isin(chain.BIP_CLASSES)].assign(
        hit=lambda x: (x["event"] != "out").astype(float),
        extra_base=lambda x: x["event"].isin(["double", "triple", "home_run"]).astype(float),
        hr=lambda x: (x["event"] == "home_run").astype(float),
        ev_95=lambda x: np.where(x["ExitSpeed"].notna(), (x["ExitSpeed"] >= 95).astype(float), np.nan),
        barrel=lambda x: pd.to_numeric(x["is_hard_contact"], errors="coerce"))
    fe_pitch = ["pitcher_anon_id", "batter_anon_id", "year", "count_id"]
    fe_pa = ["pitcher_anon_id", "batter_anon_id", "year"]
    specs = [
        ("H1", "whiff por swing", swings, "whiff", fe_pitch, -1, "menos whiffs en altitud ⇒ más contacto"),
        ("H1", "bola en juego por swing", swings, "en_juego", fe_pitch, +1, "más contacto en juego en altitud"),
        ("H1", "ponche por turno", pas, "ponche", fe_pa, -1, "menos ponches en altitud"),
        ("H2", "run value RE24 por bola en juego", bip, "run_value", fe_pitch, +1, "el contacto vale más carreras"),
        ("H2", "hit por bola en juego", bip, "hit", fe_pitch, +1, "más hits por contacto"),
        ("H2", "extra base por bola en juego", bip, "extra_base", fe_pitch, +1, "más extra bases por contacto"),
        ("H2", "HR por bola en juego", bip, "hr", fe_pitch, +1, "más HR por contacto"),
        ("H2", "velocidad de salida (mph)", bip, "ExitSpeed", fe_pitch, +1, "contacto más fuerte"),
        ("H2", "velocidad de salida ≥ 95 mph", bip, "ev_95", fe_pitch, +1, "más contacto duro"),
        ("H2", "barrel", bip, "barrel", fe_pitch, +1, "más barrels"),
    ]
    parks = extreme_parks()
    rows = []
    for h, label, data, y, fe, direction, meaning in specs:
        raw = fe_estimate(data, y, [])
        fx = fe_estimate(data, y, fe)
        by_park = fe_by_park(data, y, fe, parks)
        same_sign = sum(np.sign(v["beta"]) == direction for v in by_park.values())
        rows.append({"hipotesis": h, "variable": label, "direccion_hipotesis": meaning, "sentido": direction,
                     "crudo": raw, "efectos_fijos": fx, "veredicto": verdict(fx["ic95_extreme"], direction),
                     "por_estadio": by_park, "estadios_en_la_direccion": f"{same_sign}/{len(by_park)}"}) 
        logger.info("%s %-34s crudo E−N=%+.4f | FE β_E=%+.4f [%+.4f, %+.4f] → %s", h, label, raw["beta_extreme"],
                    fx["beta_extreme"], *fx["ic95_extreme"], rows[-1]["veredicto"])
    return rows


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------


def _gap(d: pd.DataFrame, weights: pd.Series) -> tuple[float, float, float]:
    """(G promedio, Δ(whiff) promedio, E[Δ | bola en juego] promedio) para los datos `d`."""
    term = chain.terminal_values(d)
    _, deltas = chain.count_values(d, term)
    bip = d[d["event"].isin(chain.BIP_CLASSES)]
    mix = pd.crosstab([bip["Balls"], bip["Strikes"]], bip["event"], normalize="index").reindex(columns=chain.BIP_CLASSES, fill_value=0.0)
    v_bip = (deltas[chain.BIP_CLASSES] * mix.reindex(deltas.index).fillna(0.0)).sum(axis=1)
    w = weights.reindex(deltas.index).fillna(0.0)
    return float(((v_bip - deltas["whiff"]) * w).sum()), float((deltas["whiff"] * w).sum()), float((v_bip * w).sum())


def h3(df: pd.DataFrame) -> dict:
    sw = df[df["event_group"] == "swing"]
    weights = sw.groupby(["Balls", "Strikes"]).size() / len(sw)
    point = {c: _gap(df[df["altitude_category"] == c], weights) for c in CATS}
    rng = np.random.default_rng(SEED)
    by_cat = {c: df[df["altitude_category"] == c] for c in CATS}
    games = {c: by_cat[c]["game_anon_id"].unique() for c in CATS}
    idx = {c: by_cat[c].groupby("game_anon_id").indices for c in CATS}
    boots = []
    for b in range(N_BOOT):
        rep = {}
        for c in CATS:
            pick = rng.choice(games[c], size=len(games[c]), replace=True)
            rows = np.concatenate([idx[c][g] for g in pick])
            rep[c] = _gap(by_cat[c].iloc[rows].reset_index(drop=True), weights)[0]
        boots.append(rep)
    boots = pd.DataFrame(boots)
    diff = boots["Extreme Altitude"] - boots["No Altitude"]
    diff_m = boots["Medium Altitude"] - boots["No Altitude"]
    pe = point["Extreme Altitude"][0] - point["No Altitude"][0]
    ci = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    out = {
        "por_categoria": {c: {"G": point[c][0], "delta_whiff": point[c][1], "delta_bip": point[c][2],
                              "ic95_G": [float(np.percentile(boots[c], 2.5)), float(np.percentile(boots[c], 97.5))]} for c in CATS},
        "extreme_menos_no": pe, "ic95": ci, "p_mayor": float((diff > 0).mean()),
        "medium_menos_no": point["Medium Altitude"][0] - point["No Altitude"][0],
        "ic95_medium": [float(np.percentile(diff_m, 2.5)), float(np.percentile(diff_m, 97.5))],
        "veredicto": verdict(ci, +1),
    }
    parks = extreme_parks()
    d_park = df.assign(park=df["game_anon_id"].map(parks)).dropna(subset=["park"])
    out["G_por_estadio"] = {p: _gap(g.reset_index(drop=True), weights)[0] for p, g in d_park.groupby("park") if len(g) > 5000}
    out["estadios_con_G_mayor_que_no_altitude"] = f"{sum(v > point['No Altitude'][0] for v in out['G_por_estadio'].values())}/{len(out['G_por_estadio'])}"
    logger.info("H3: G_E − G_N = %+.4f [%+.4f, %+.4f] → %s", pe, *ci, out["veredicto"])
    return out


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def format_report(rows: list[dict], h3r: dict, n: int) -> str:
    L = [
        "# Hipótesis del analista: ¿en altitud extrema conviene evitar el contacto?",
        "",
        f"Generado por `src/analysis/altitude_hypotheses.py` ({n:,} pitches). Hipótesis, variables y criterio fijados en el "
        "docstring del módulo **antes** de correr. *Respaldada* = el IC 95% excluye 0 en la dirección de la hipótesis; "
        "*contradicha* = lo excluye en la dirección contraria; *no concluyente* = incluye 0.",
        "",
        "**Cómo leer β_E.** Diferencia promedio Extreme − No Altitude comparando al **mismo pitcher contra el mismo bateador** "
        "(efectos fijos de pitcher, bateador, temporada y conteo). La columna *crudo* es la diferencia simple, sin controles.",
        "",
        "## H1 y H2",
        "",
        "| hipótesis | variable | media en No Altitude | crudo E − N | β_E con controles [IC 95%] | β_M (Medium) | veredicto |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for r in rows:
        fx, rw = r["efectos_fijos"], r["crudo"]
        lo, hi = fx["ic95_extreme"]
        L.append(f"| {r['hipotesis']} | {r['variable']} | {fx['media_no_altitude']:.4f} | {rw['beta_extreme']:+.4f} | "
                 f"{fx['beta_extreme']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {fx['beta_medium']:+.4f} | **{r['veredicto']}** |")
    L += [
        "",
        "### Consistencia entre estadios de Extreme",
        "",
        "Con ~4–5 estadios por temporada, los IC por juego no capturan la variación entre estadios. Aquí se estima β por "
        "cada estadio anónimo de Extreme (`reports/estadios_anonimos.md`; IDs estables entre temporadas) contra No Altitude, con los mismos controles. Si el efecto fuera de un solo estadio, la mayoría "
        "de las columnas tendría el signo contrario o cero.",
        "",
        "| variable | estadios en la dirección de la hipótesis | β por estadio (ordenados) |",
        "|---|---:|---|",
    ]
    for r in rows:
        betas = sorted(v["beta"] for v in r["por_estadio"].values())
        L.append(f"| {r['variable']} | {r['estadios_en_la_direccion']} | " + ", ".join(f"{b:+.3f}" for b in betas) + " |")
    L += ["", "Tamaños: " + "; ".join(f"{r['variable']}: n={r['efectos_fijos']['n']:,}, juegos={r['efectos_fijos']['n_games']:,}" for r in rows), ""]
    L += [
        "## H3: ¿cuánto más cuesta una bola en juego que un whiff?",
        "",
        "G = valor esperado (en carreras, para el bateador) de una bola en juego menos el de un whiff en el mismo conteo, "
        "con V(b,s) y Δ recalculados en cada categoría y promediados con la mezcla de conteos de los swings de toda la liga. "
        "IC 95% por bootstrap de juegos.",
        "",
        "| categoría | Δ(whiff) | E[Δ \\| bola en juego] | G [IC 95%] |",
        "|---|---:|---:|---|",
    ]
    for c, v in h3r["por_categoria"].items():
        L.append(f"| {c} | {v['delta_whiff']:+.4f} | {v['delta_bip']:+.4f} | {v['G']:+.4f} [{v['ic95_G'][0]:+.4f}, {v['ic95_G'][1]:+.4f}] |")
    lo, hi = h3r["ic95"]
    L += [
        "",
        f"**G_Extreme − G_NoAltitude = {h3r['extreme_menos_no']:+.4f} carreras [{lo:+.4f}, {hi:+.4f}] → {h3r['veredicto']}.** "
        f"(Medium − No: {h3r['medium_menos_no']:+.4f} [{h3r['ic95_medium'][0]:+.4f}, {h3r['ic95_medium'][1]:+.4f}].)",
        "",
        f"Por estadio anónimo de Extreme: G > G_NoAltitude en **{h3r['estadios_con_G_mayor_que_no_altitude']}** estadios "
        "(G por estadio: " + ", ".join(f"{v:+.3f}" for v in sorted(h3r["G_por_estadio"].values())) + ").",
        "",
        "## Limitaciones",
        "",
        "- *Extreme* agrupa varios estadios anónimos distintos (`reports/estadios_anonimos.md`): es el efecto promedio de "
        "esos estadios, no de un estadio en particular.",
        "- Con tan pocos estadios por categoría, la altitud no se puede separar de otras características del parque "
        "(dimensiones, clima, calibración de Trackman de la velocidad de salida). Los IC por juego son condicionales a esos estadios.",
        "- Es asociación con controles por pitcher y bateador, no un efecto causal aislado de la densidad del aire.",
        "",
    ]
    return "\n".join(L)


def run() -> dict:
    df = load()
    rows = h1_h2(df)
    h3r = h3(df)
    REPORT_PATH.write_text(format_report(rows, h3r, len(df)), encoding="utf-8")
    JSON_PATH.write_text(json.dumps({"h1_h2": rows, "h3": h3r}, indent=2, default=float), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return {"h1_h2": rows, "h3": h3r}


if __name__ == "__main__":
    run()
