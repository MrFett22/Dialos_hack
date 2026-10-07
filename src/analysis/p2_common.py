"""Utilidades comunes de P2 (perfiles de pitcher para ambientes de altitud).

Definiciones que comparten Stuff+ Physical, Location+ y los perfiles:

- UNIVERSO: los pitches de la cadena de run value (`run_value_chain_oof_pitches`,
  evento clasificable y conteo válido).
- ENTORNO (`env`): estadio anónimo del juego (ANON_STADIUM_xx, `src/data/parks.py`)
  si el juego está resuelto; si no, la categoría de altitud. Es la partición del
  ajuste simple M1b, el mejor en la validación temporal estricta de P0.
- AJUSTE SIMPLE POR ENTORNO de una variable y en su universo:
      y_adj = y − media(y | temporada, entorno) + media(y)
  Quita el promedio del entorno sin modelar física y también los cambios de
  nivel de toda la liga entre temporadas (p. ej. el hard contact% promedio sube
  de 0.215 a 0.261 entre 2024 y 2025): no son habilidad del pitcher y, si se
  dejan, invierten las correlaciones año a año de las métricas muy encogidas.
  Conserva la escala de y.
- TURNO (PA): unidad del ponche. Un turno = su pitch terminal (ponche, base por
  bolas, golpeado o bola en juego). `run_value` (RE24) solo es distinto de 0 en
  el pitch terminal (verificado: |rv| medio 1.6e-5 en no terminales), así que el
  valor del turno es el `run_value` de ese pitch. Nunca se copia el resultado
  del turno a sus pitches.
- SHRINKAGE: normal-normal empírico-bayesiano (`eb_shrink`), con o sin
  covariables en la media a priori (Fay–Herriot, `eb_shrink_covariates`).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import parks
from src.models import run_value_chain as chain

ROOT = Path(__file__).resolve().parents[2]
ARSENAL_PATH = ROOT / "data" / "processed" / "pitches_with_arsenal.parquet"
WEB_DIR = ROOT / "data" / "web"
FIG_DIR = ROOT / "reports" / "figures" / "p2"
CATS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
CAT_SHORT = {"No Altitude": "No Altitude", "Medium Altitude": "Medium", "Extreme Altitude": "Extreme"}
MAIN_TYPES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
SEASON_PAIRS = [(2024, 2025), (2025, 2026)]
HARD_EV = 95.0
Z80 = 1.2815515655446004


# ---------------------------------------------------------------------------
# Marco por pitch y por turno
# ---------------------------------------------------------------------------


def game_environment() -> pd.DataFrame:
    g = pd.read_parquet(parks.OUT_PATH)
    g["stadium_resuelto"] = g["stadium_id"].where(g["status"] == "resuelto")
    return g[["game_anon_id", "stadium_resuelto", "home_team", "away_team", "stadium_category"]]


def load_pitch_frame() -> pd.DataFrame:
    """Un renglón por pitch del universo de la cadena, con evento, entorno y variables de contacto."""
    oof = pd.read_parquet(chain.OOF_PATH, columns=["PitchUID", "pitcher_anon_id", "year", "event", "Balls", "Strikes",
                                                    "run_value", "realized_delta"])
    raw = pd.read_parquet(ARSENAL_PATH, columns=[
        "PitchUID", "game_anon_id", "batter_anon_id", "altitude_category", "AutoPitchType", "PitcherThrows", "BatterSide",
        "PlateLocSide", "PlateLocHeight", "ExitSpeed", "Angle", "hit_type", "Inning", "Top/Bottom",
        "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "KorBB"])
    d = oof.merge(raw, on="PitchUID", how="left")
    for c in ["AutoPitchType", "PitcherThrows", "BatterSide", "hit_type", "altitude_category", "Top/Bottom", "KorBB"]:
        d[c] = d[c].astype("object")
    env = game_environment()
    d = d.merge(env, on="game_anon_id", how="left")
    d["env"] = d["stadium_resuelto"].fillna(d["altitude_category"])
    d["ExitSpeed"] = pd.to_numeric(d["ExitSpeed"], errors="coerce")

    ev = d["event"]
    d["is_swing"] = ev.isin(["whiff", "foul"] + chain.BIP_CLASSES).astype(int)
    d["is_whiff"] = (ev == "whiff").astype(int)
    d["is_bip"] = ev.isin(chain.BIP_CLASSES).astype(int)
    d["is_k"] = (ev.isin(["called_strike", "whiff"]) & (d["Strikes"] == 2)).astype(int)
    d["is_bb"] = ((ev == "ball") & (d["Balls"] == 3)).astype(int)
    d["is_hbp"] = (ev == "hbp").astype(int)
    d["is_pa_end"] = ((d["is_k"] + d["is_bb"] + d["is_hbp"] + d["is_bip"]) > 0).astype(int)
    bip = d["is_bip"] == 1
    d["is_hard"] = np.where(bip & d["ExitSpeed"].notna(), (d["ExitSpeed"] >= HARD_EV).astype(float), np.nan)
    batted = bip & d["hit_type"].isin(["GroundBall", "FlyBall", "LineDrive", "Popup"])
    d["is_gb"] = np.where(batted, (d["hit_type"] == "GroundBall").astype(float), np.nan)
    d["is_hr"] = np.where(bip, (ev == "home_run").astype(float), np.nan)
    d["rv_bip"] = np.where(bip, d["run_value"], np.nan)
    # equipo del pitcher: en la parte alta lanza el local
    d["pitcher_team"] = np.where(d["Top/Bottom"] == "Top", d["home_team"], d["away_team"])
    return d


def pa_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por turno (su pitch terminal). `outcome` ∈ {K, BB, HBP, BIP}; `pa_rv` = run value RE24 del turno."""
    pa = d.loc[d["is_pa_end"] == 1].copy()
    pa["outcome"] = np.select([pa["is_k"] == 1, pa["is_bb"] == 1, pa["is_hbp"] == 1], ["K", "BB", "HBP"], default="BIP")
    pa["pa_rv"] = pa["run_value"]
    return pa


# ---------------------------------------------------------------------------
# Ajuste simple por entorno
# ---------------------------------------------------------------------------


def env_adjust(d: pd.DataFrame, col: str, mask: pd.Series | None = None) -> pd.Series:
    """y − media(y | temporada, entorno) + media(y), dentro del universo `mask` (NaN fuera)."""
    m = d[col].notna() if mask is None else (mask & d[col].notna())
    sub = d.loc[m, ["year", "env", col]]
    adj = sub[col] - sub.groupby(["year", "env"])[col].transform("mean") + sub[col].mean()
    out = pd.Series(np.nan, index=d.index)
    out[m] = adj
    return out


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------


def eb_shrink(mean: np.ndarray, n: np.ndarray, sigma2: float, min_n: int, mu: float | None = None) -> dict:
    """Normal-normal: θ_i ~ N(μ, τ²), media_i | θ_i ~ N(θ_i, σ²/n_i).

    μ y τ² por momentos con las unidades de n ≥ min_n (μ ponderado por n). Si τ² ≤ 0
    (sin varianza real detectable) todo se encoge a μ y `reliability` = 0.
    Devuelve shrunk, post_sd, B (factor de encogimiento) y los hiperparámetros.
    """
    mean, n = np.asarray(mean, float), np.asarray(n, float)
    big = n >= min_n
    if mu is None:
        mu = float(np.average(mean[big], weights=n[big]))
    se2 = sigma2 / n
    tau2 = float(np.var(mean[big], ddof=1) - np.mean(se2[big]))
    if tau2 <= 0:
        B = np.ones_like(mean)
        tau2 = 0.0
    else:
        B = se2 / (se2 + tau2)
    shrunk = mu + (1 - B) * (mean - mu)
    return {"shrunk": shrunk, "post_sd": np.sqrt((1 - B) * se2), "B": B, "mu": mu, "tau": float(np.sqrt(tau2)),
            "sigma": float(np.sqrt(sigma2)), "reliability_n_tipico": float(np.median(1 - B[big])) if big.any() else np.nan}


def eb_shrink_covariates(mean: np.ndarray, n: np.ndarray, sigma2: float, X: np.ndarray, min_n: int,
                         n_iter: int = 50) -> dict:
    """Fay–Herriot: θ_i ~ N(x_i'β, τ²), media_i | θ_i ~ N(θ_i, σ²/n_i).

    β por mínimos cuadrados ponderados con pesos 1/(τ² + se²_i) y τ² por momentos sobre los
    residuales, iterando. La media a priori de cada pitcher deja de ser la de la liga y pasa a
    ser la que predicen sus covariables (p. ej. su Stuff+): con poca muestra se le acerca a ella.
    """
    mean, n = np.asarray(mean, float), np.asarray(n, float)
    X1 = np.column_stack([np.ones(len(mean)), X])
    big = n >= min_n
    se2 = sigma2 / n
    tau2 = max(float(np.var(mean[big]) - np.mean(se2[big])), 1e-12)
    for _ in range(n_iter):
        w = 1 / (tau2 + se2[big])
        Xw = X1[big] * w[:, None]
        beta = np.linalg.solve(X1[big].T @ Xw, Xw.T @ mean[big])
        resid = mean[big] - X1[big] @ beta
        new = max(float(np.mean(resid ** 2) - np.mean(se2[big])), 1e-12)
        if abs(new - tau2) < 1e-14:
            tau2 = new
            break
        tau2 = new
    prior = X1 @ beta
    B = se2 / (se2 + tau2)
    return {"shrunk": prior + (1 - B) * (mean - prior), "post_sd": np.sqrt((1 - B) * se2), "B": B, "prior": prior,
            "beta": beta, "tau": float(np.sqrt(tau2)), "sigma": float(np.sqrt(sigma2))}


def unit_stats(d: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """n, media y varianza de `value` por unidad (sin NaN)."""
    g = d.dropna(subset=[value]).groupby(keys)[value]
    return g.agg(n="size", mean="mean", var="var").reset_index()


def pooled_sigma2(stats: pd.DataFrame, min_n: int) -> float:
    big = stats[(stats["n"] >= min_n) & stats["var"].notna()]
    return float(np.average(big["var"], weights=big["n"]))


def plus_scale(shrunk: np.ndarray, mu: float, tau: float, higher_is_better: bool) -> np.ndarray:
    """100 + 10·(θ̂ − μ)/τ (o con signo invertido si menor = mejor). 10 puntos = 1 sd de habilidad real."""
    sign = 1.0 if higher_is_better else -1.0
    return 100 + 10 * sign * (np.asarray(shrunk) - mu) / tau


# ---------------------------------------------------------------------------
# Validación año contra año
# ---------------------------------------------------------------------------


def season_pairs(table: pd.DataFrame, cols: list[str], min_col: str, min_n: float) -> pd.DataFrame:
    frames = []
    for y0, y1 in SEASON_PAIRS:
        a = table[(table["year"] == y0) & (table[min_col] >= min_n)]
        b = table[(table["year"] == y1) & (table[min_col] >= min_n)]
        p = a[["pitcher_anon_id"] + cols].merge(b[["pitcher_anon_id"] + cols], on="pitcher_anon_id", suffixes=("_t", "_t1"))
        p["par"] = f"{y0}->{y1}"
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def boot_corr(pairs: pd.DataFrame, x: str, y: str, n_boot: int = 1000, seed: int = 42, ref: str | None = None) -> dict:
    """r(x, y) con IC 95% por bootstrap de pitchers; con `ref`, Δr = r(x,y) − r(ref,y) pareado."""
    cols = [x, y] + ([ref] if ref else [])
    sub = pairs.dropna(subset=cols).reset_index(drop=True)
    codes, uniq = pd.factorize(sub["pitcher_anon_id"])
    rows = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    M = sub[cols].to_numpy(float)

    def stat(m):
        r = np.corrcoef(m[:, 0], m[:, 1])[0, 1]
        return r - np.corrcoef(m[:, 2], m[:, 1])[0, 1] if ref else r

    point = stat(M)
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(uniq), len(uniq))])
        reps.append(stat(M[idx]))
    reps = np.array(reps)
    out = {"r" if not ref else "delta_r": float(point), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
           "n": int(len(sub))}
    if ref:
        out["p_mejor"] = float((reps > 0).mean())
    return out


def fmt_ci(d: dict, key: str = "r", digits: int = 3) -> str:
    lo, hi = d["ci95"]
    return f"{d[key]:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def write_web_table(df: pd.DataFrame, name: str) -> Path:
    """Tablas para la futura página: CSV UTF-8, redondeadas, orden de columnas estable."""
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    path = WEB_DIR / f"{name}.csv"
    out = df.copy()
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(6)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path
