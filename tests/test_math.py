"""Pruebas matemáticas del pipeline (auditoría 2026-10-05, §13 del encargo).

Cada prueba verifica una propiedad que la metodología promete:
probabilidades que suman 1, la recursión del conteo (incluido el auto-lazo
del foul con 2 strikes), la cinemática y la descomposición drag/Magnus, el
contrafactual, el shrinkage, los signos y la escala del índice, y la
atribución de Shapley. Las que necesitan los datos procesados se saltan si
no existen.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import trajectory as tr
from src.models import chain_attribution as attr
from src.models import pitching_plus_neutral as pn
from src.models import run_value_chain as chain

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "stuff_model_df.parquet"
OFFICIAL = ROOT / "data" / "processed" / "pitching_plus_official.parquet"
CHAIN_OOF = ROOT / "data" / "processed" / "run_value_chain_oof_pitches.parquet"
RATIOS = ROOT / "data" / "processed" / "altitude_density_ratios.parquet"
needs = lambda p: pytest.mark.skipif(not p.exists(), reason=f"requiere {p.name}")  # noqa: E731


def _random_preds(n: int, rng) -> dict:
    return {
        "swing": rng.uniform(0, 1, (n, 1)), "take": rng.dirichlet(np.ones(3), n), "whiff": rng.uniform(0, 1, (n, 1)),
        "foul": rng.uniform(0, 1, (n, 1)), "bip": rng.dirichlet(np.ones(5), n),
    }


# --------------------------------------------------------------- probabilidades
def test_event_probabilities_sum_to_one_including_edges():
    rng = np.random.default_rng(0)
    p = _random_preds(5000, rng)
    for k in ("swing", "whiff", "foul"):  # casos extremos 0 y 1
        p[k][:10] = 0.0
        p[k][10:20] = 1.0
    P = chain.event_probabilities(p).to_numpy()
    assert np.abs(P.sum(1) - 1).max() < 1e-12
    assert P.min() >= 0


@needs(CHAIN_OOF)
def test_oof_probabilities_sum_to_one():
    oof = pd.read_parquet(CHAIN_OOF, columns=[f"p_pitching_{e}" for e in chain.EVENTS])
    s = oof.to_numpy().sum(1)
    assert not np.isnan(s).any()
    assert np.abs(s - 1).max() < 1e-9


# --------------------------------------------------------------- recursión del conteo
def _synthetic_counts(rng, n=60000) -> pd.DataFrame:
    rows = []
    for b in range(4):
        for s in range(3):
            ev = rng.choice(chain.EVENTS, size=n // 12, p=rng.dirichlet(np.ones(len(chain.EVENTS))))
            rows.append(pd.DataFrame({"Balls": b, "Strikes": s, "event": ev}))
    return pd.concat(rows, ignore_index=True)


def test_count_recursion_bellman_with_two_strike_foul_self_loop():
    rng = np.random.default_rng(1)
    df = _synthetic_counts(rng)
    term = {"walk": 0.35, "strikeout": -0.30, "hbp": 0.36, "out": -0.28, "single": 0.45, "double": 0.75, "triple": 1.0, "home_run": 1.4}
    vtab, deltas = chain.count_values(df, term)
    V = vtab["V"]
    freq = pd.crosstab([df["Balls"], df["Strikes"]], df["event"], normalize="index").reindex(columns=chain.EVENTS, fill_value=0)
    for (b, s), v in V.items():
        nxt = {"ball": term["walk"] if b == 3 else V[(b + 1, s)]}
        nxt.update({e: term["strikeout"] if s == 2 else V[(b, s + 1)] for e in ("called_strike", "whiff")})
        nxt["foul"] = V[(b, s)] if s == 2 else V[(b, s + 1)]  # con 2 strikes el foul NO cambia el conteo
        nxt.update({e: term[e] for e in ["hbp"] + chain.BIP_CLASSES})
        assert abs(sum(freq.loc[(b, s), e] * nxt[e] for e in chain.EVENTS) - v) < 1e-12
    assert np.allclose(deltas.loc[(slice(None), 2), "foul"], 0)


# --------------------------------------------------------------- física
@pytest.fixture(scope="module")
def raw_sample():
    if not RAW.exists():
        pytest.skip("requiere datos crudos")
    d = pd.read_parquet(RAW).sample(20000, random_state=3)
    for c in tr.TRAJECTORY_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=tr.TRAJECTORY_COLUMNS + ["PlateLocSide", "PlateLocHeight", "altitude_category", "Extension"]).reset_index(drop=True)


def test_kinematics_reproduce_plate_location(raw_sample):
    d = raw_sample
    p0, v0, a0 = (d[c].to_numpy(float) for c in (["x0", "y0", "z0"], ["vx0", "vy0", "vz0"], ["ax0", "ay0", "az0"]))
    A, B, C = 0.5 * a0[:, 1], v0[:, 1], p0[:, 1] - tr.Y_FRONT_OF_PLATE
    disc = np.sqrt(B * B - 4 * A * C)
    r1, r2 = (-B + disc) / (2 * A), (-B - disc) / (2 * A)
    T = np.where((r1 > 0) & ((r1 < r2) | (r2 <= 0)), r1, r2)
    plate = p0 + v0 * T[:, None] + 0.5 * a0 * T[:, None] ** 2
    assert np.median(np.abs(plate[:, 0] + d["PlateLocSide"])) < 0.01  # pies
    assert np.median(np.abs(plate[:, 2] - d["PlateLocHeight"])) < 0.01


def test_drag_magnus_decomposition_is_exact_and_orthogonal(raw_sample):
    c = tr.decompose_acceleration(raw_sample)
    a = raw_sample[["ax0", "ay0", "az0"]].to_numpy(float).copy()
    a[:, 2] += tr.GRAVITY_FT_S2
    v = raw_sample[["vx0", "vy0", "vz0"]].to_numpy(float)
    vh = v / np.linalg.norm(v, axis=1)[:, None]
    assert np.abs(c["drag_vec"] + c["magnus_vec"] - a).max() < 1e-10
    assert np.abs((c["magnus_vec"] * vh).sum(1)).max() < 1e-10
    assert (c["drag"] > 0).mean() > 0.999  # el drag frena: componente opuesta a la velocidad


def test_ratios_with_too_few_groups_fail_loudly(raw_sample):
    """Con una muestra chica no hay grupos para estimar ρ̂: la proyección debe fallar, no devolver lo observado."""
    ratios = tr.estimate_density_ratios(raw_sample.sample(3000, random_state=0))
    with pytest.raises(ValueError):
        tr.project_to_altitude(raw_sample.head(50), "No Altitude", ratios)


@needs(RATIOS)
def test_counterfactual_hits_same_target_without_nans_and_is_physical(raw_sample):
    ratios = pd.read_parquet(RATIOS)
    ext = raw_sample[raw_sample["altitude_category"] == "Extreme Altitude"].reset_index(drop=True)
    proj = tr.project_to_altitude(ext, "No Altitude", ratios)
    assert proj[tr.UPDATED_FEATURES].isna().sum().sum() == ext[tr.UPDATED_FEATURES].isna().sum().sum()
    assert np.allclose(proj["PlateLocSide"], ext["PlateLocSide"]) and np.allclose(proj["PlateLocHeight"], ext["PlateLocHeight"])
    dz = (proj["ZoneSpeed"] - ext["ZoneSpeed"]).to_numpy()
    assert np.nanmedian(dz) < 0 and np.nanmax(np.abs(dz)) < 5  # más aire → más frenado, cambio acotado (mph)
    assert np.nanmax(np.abs(proj["ZoneTime"] - ext["ZoneTime"])) < 0.05  # s


# --------------------------------------------------------------- shrinkage, signos y escala
def _eb_toy(rng):
    rows = []
    for p in range(400):
        theta = rng.normal(0, 0.006)
        n = int(rng.choice([10, 50, 200, 800, 2000]))
        rows.append(pd.DataFrame({"pitcher_anon_id": p, "year": 2025, "x": rng.normal(theta, 0.05, n)}))
    return pd.concat(rows, ignore_index=True)


def test_shrinkage_factor_bounds_and_limits():
    eb, par = pn.empirical_bayes(_eb_toy(np.random.default_rng(4)), "x")
    assert eb["shrink_factor"].between(0, 1).all()
    s = eb.sort_values("n")
    assert (s.groupby("n")["shrink_factor"].mean().diff().dropna() < 0).all()
    small, large = eb[eb["n"] == 10], eb[eb["n"] == 2000]
    mu = par[2025]["mu"]
    assert (small["shrunk"] - mu).abs().mean() < 0.3 * (small["mean"] - mu).abs().mean()  # n chico → cerca de μ
    assert np.corrcoef(large["shrunk"], large["mean"])[0, 1] > 0.999  # n grande → conserva su promedio


def test_score_sign_and_scale_definition():
    eb, par = pn.empirical_bayes(_eb_toy(np.random.default_rng(5)), "x")
    p = par[2025]
    assert np.allclose(eb["plus"], 100 + 10 * (p["mu"] - eb["shrunk"]) / p["tau"])
    assert np.corrcoef(eb["plus_raw"], eb["mean"])[0, 1] < -0.999  # menos carreras esperadas → score más alto


@needs(OFFICIAL)
def test_official_scale_matches_definition():
    off = pd.read_parquet(OFFICIAL)
    big = off[off["n_pitches"] >= 100]
    for _, g in big.groupby("year"):
        w = np.average(g["pitching_plus_sin_shrinkage"], weights=g["n_pitches"])
        assert abs(w - 100) < 0.5, "100 = media ponderada por pitches de los pitchers con ≥100"
        assert 6 < g["pitching_plus"].std() < 10.5, "sd < 10 por diseño: 10 puntos = 1 τ y θ̂ está encogido"


# --------------------------------------------------------------- atribución de Shapley
def test_shapley_is_efficient_and_null_link_gets_zero():
    rng = np.random.default_rng(6)
    pa, pb = _random_preds(2000, rng), _random_preds(2000, rng)
    pb["take"] = pa["take"]  # eslabón que no cambia
    D = rng.normal(0, 0.3, (2000, len(chain.EVENTS)))
    phi = attr.shapley(pa, pb, D)
    total = attr.expected_value(pb, D) - attr.expected_value(pa, D)
    assert np.abs(phi.sum(1).to_numpy() - total).max() < 1e-12
    assert np.abs(phi["take"]).max() < 1e-12
