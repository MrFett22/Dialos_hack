"""Pruebas de P2: escalas, probabilidades, universos, turno vs pitch, shrinkage, rankings y reproducibilidad."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import p2_common as pc
from src.analysis import pitcher_profiles as pp
from src.models import location_plus_v2 as locp
from src.models import stuff_plus_physical as spp

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "data" / "web"
needs = lambda p: pytest.mark.skipif(not p.exists(), reason=f"requiere {p.name}")  # noqa: E731


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------


def _sim(n_units=400, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(20, 2000, n_units)
    theta = rng.normal(0.25, 0.03, n_units)
    sigma2 = 0.25 * 0.75
    mean = theta + rng.normal(0, np.sqrt(sigma2 / n))
    return mean, n, sigma2, theta


def test_eb_shrink_moves_toward_mean_and_less_with_more_data():
    mean, n, sigma2, _ = _sim()
    eb = pc.eb_shrink(mean, n, sigma2, min_n=100)
    lo, hi = np.minimum(mean, eb["mu"]), np.maximum(mean, eb["mu"])
    assert ((eb["shrunk"] >= lo - 1e-12) & (eb["shrunk"] <= hi + 1e-12)).all()
    order = np.argsort(n)
    assert np.all(np.diff(eb["B"][order]) <= 1e-12)  # más muestra → menos encogimiento
    assert eb["tau"] == pytest.approx(0.03, rel=0.25)


def test_eb_shrink_without_real_variance_collapses_to_mean():
    rng = np.random.default_rng(1)
    n = np.full(300, 200)
    mean = 0.3 + rng.normal(0, np.sqrt(0.21 / 200), 300) * 0.5  # menos varianza que la de muestreo
    eb = pc.eb_shrink(mean, n, 0.21, min_n=100)
    assert eb["tau"] == 0.0 and np.allclose(eb["shrunk"], eb["mu"]) and np.allclose(eb["B"], 1.0)


def test_fay_herriot_uses_covariates():
    mean, n, sigma2, theta = _sim(seed=2)
    x = (theta - theta.mean()) / theta.std() + np.random.default_rng(3).normal(0, 0.3, len(theta))
    fh = pc.eb_shrink_covariates(mean, n, sigma2, x[:, None], min_n=100)
    plain = pc.eb_shrink(mean, n, sigma2, min_n=100)
    assert fh["beta"][1] > 0  # la covariable informativa recibe peso positivo
    assert np.mean((fh["shrunk"] - theta) ** 2) < np.mean((plain["shrunk"] - theta) ** 2)


def test_plus_scale_is_centered_and_oriented():
    s = np.array([0.20, 0.25, 0.30])
    assert pc.plus_scale(s, 0.25, 0.05, True).tolist() == pytest.approx([90, 100, 110])
    assert pc.plus_scale(s, 0.25, 0.05, False).tolist() == pytest.approx([110, 100, 90])


# ---------------------------------------------------------------------------
# Ajuste por entorno
# ---------------------------------------------------------------------------


def test_env_adjust_removes_environment_and_season_means():
    rng = np.random.default_rng(4)
    d = pd.DataFrame({"year": np.repeat([2024, 2025], 600), "env": np.tile(np.repeat(["A", "B", "C"], 200), 2)})
    d["y"] = rng.normal(0, 1, len(d)) + d["env"].map({"A": 0.0, "B": 1.0, "C": -2.0}) + (d["year"] == 2025) * 0.5
    d["mask"] = rng.random(len(d)) < 0.7
    adj = pc.env_adjust(d, "y", d["mask"])
    assert adj[~d["mask"]].isna().all()  # fuera del universo queda vacío
    sub = d[d["mask"]].assign(adj=adj)
    means = sub.groupby(["year", "env"])["adj"].mean()
    assert np.allclose(means, sub["y"].mean())  # todas las celdas temporada × entorno quedan en la media global


# ---------------------------------------------------------------------------
# Perfil: identidad de carreras, reproducibilidad, faltantes
# ---------------------------------------------------------------------------

LW = pd.Series({"v_K": -0.33, "v_BB": 0.38, "v_HBP": 0.40, "v_BIP": 0.05})
MU = {"k_pct": 0.20, "bb_pct": 0.10, "hbp_pct": 0.01, "rv_bip": 0.05}


def test_aepp_zero_for_average_and_rewards_strikeouts():
    t = pd.DataFrame({"k_pct": [0.20, 0.25, 0.20], "bb_pct": [0.10, 0.10, 0.10], "hbp_pct": [0.01] * 3,
                      "rv_bip": [0.05, 0.05, 0.08]})
    a = pp.aepp(t, LW, MU)
    assert a[0] == pytest.approx(0.0, abs=1e-12)
    assert a[1] > 0  # más ponches (y menos bolas en juego) → evita carreras
    assert a[2] < 0  # más daño por bola en juego → permite carreras
    # identidad exacta: 5 pp de K cambian el turno de BIP a K
    assert a[1] == pytest.approx(-100 * 0.05 * (LW["v_K"] - LW["v_BIP"]))


def test_aepp_weights_increase_value_of_strikeout_when_contact_costs_more():
    t = pd.DataFrame({"k_pct": [0.25], "bb_pct": [0.10], "hbp_pct": [0.01], "rv_bip": [0.05]})
    costly = LW.copy()
    costly["v_BIP"] += 0.07
    assert pp.aepp(t, costly, MU)[0] > pp.aepp(t, LW, MU)[0]


def test_kmeans_reproducible_and_silhouette_bounds():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal(3, 0.3, (60, 2))])
    a = pp.kmeans(X, 2, seed=7)
    b = pp.kmeans(X, 2, seed=7)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])
    s = pp.silhouette(X, a[0])
    assert 0.7 < s <= 1.0


def test_fay_herriot_handles_missing_covariates():
    mean, n, sigma2, _ = _sim(n_units=100, seed=6)
    X = np.random.default_rng(7).normal(size=(100, 1))
    X[:10] = 0.0  # faltantes imputados en la media (z = 0) por apply_covariates
    fh = pc.eb_shrink_covariates(mean, n, sigma2, X, min_n=100)
    assert np.isfinite(fh["shrunk"]).all() and np.isfinite(fh["post_sd"]).all()


# ---------------------------------------------------------------------------
# Salidas reales (si existen)
# ---------------------------------------------------------------------------


@needs(pp.chain.OOF_PATH)
def test_universes_and_pa_unit():
    f = pc.load_pitch_frame()
    assert f["PitchUID"].is_unique
    assert (f.loc[f["is_whiff"] == 1, "is_swing"] == 1).all()  # whiff ⊂ swing
    assert (f.loc[f["is_bip"] == 1, ["is_swing", "is_pa_end"]] == 1).all().all()  # BIP ⊂ swing ∩ fin de turno
    assert f.loc[f["is_bip"] == 0, "is_hard"].isna().all()  # contacto duro solo en bolas en juego
    pa = pc.pa_frame(f)
    assert len(pa) == f["is_pa_end"].sum() and pa["PitchUID"].is_unique  # un renglón por turno
    assert (pa[["is_k", "is_bb", "is_hbp", "is_bip"]].sum(axis=1) == 1).all()  # exactamente un resultado
    assert (pa.loc[pa["outcome"] == "K", "is_k"] == 1).all()
    # el ponche se cuenta una vez por turno, no en cada pitch del turno
    assert f["is_k"].sum() == (pa["outcome"] == "K").sum()
    # el run value del turno vive en el pitch terminal
    assert f.loc[f["is_pa_end"] == 0, "run_value"].abs().mean() < 1e-3


def _check_plus_scale(scores: pd.DataFrame, raw_col: str):
    for _, g in scores.groupby("year"):
        big = g[g["n_pitches"] >= 100]
        assert np.average(big[raw_col], weights=big["n_pitches"]) == pytest.approx(100.0, abs=1e-6)


@needs(spp.SCORES_PATH)
def test_stuff_plus_physical_scale():
    s = pd.read_parquet(spp.SCORES_PATH)
    _check_plus_scale(s, "stuff_plus_physical_sin_shrinkage")
    assert (s["stuff_plus_physical_lo80"] <= s["stuff_plus_physical"]).all()
    assert (s["stuff_plus_physical"] <= s["stuff_plus_physical_hi80"]).all()
    # el shrinkage acerca a 100 (nunca aleja)
    assert ((s["stuff_plus_physical"] - 100).abs() <= (s["stuff_plus_physical_sin_shrinkage"] - 100).abs() + 1e-9).all()


@needs(locp.SCORES_PATH)
def test_location_plus_scale_and_probabilities():
    s = pd.read_parquet(locp.SCORES_PATH)
    _check_plus_scale(s, "location_plus_sin_shrinkage")
    p = pd.read_parquet(locp.PITCHES_PATH, columns=["p_sum", "loc_value", "AutoPitchType", "Balls", "Strikes", "same_side", "bat_side"])
    assert (p["p_sum"] - 1).abs().max() < 1e-9  # los 10 eventos suman 1
    # loc_value está centrado dentro de cada contexto (no premia el tipo ni el conteo)
    assert p.groupby(locp.CONTEXT)["loc_value"].mean().abs().max() < 1e-9


@needs(WEB / "pitcher_rankings.csv")
def test_rankings_sample_size_uncertainty_and_missing():
    r = pd.read_csv(WEB / "pitcher_rankings.csv")
    assert (r["n_pitches"] >= pp.MIN_PITCHES_RANK).all() and (r["n_pa"] >= pp.MIN_PA_RANK).all()
    assert sorted(r["rank"]) == list(range(1, len(r) + 1))
    assert r["aepp"].is_monotonic_decreasing
    assert (r["aepp_lo80"] <= r["aepp_hi80"]).all() and (r["rank_lo80"] <= r["rank_hi80"]).all()
    key = ["pitcher_anon_id", "rank", "aepp", "aepp_lo80", "aepp_hi80", "explicacion"]
    assert r[key].notna().all().all()
    assert r["pitcher_anon_id"].is_unique


@needs(WEB / "pitcher_sensitivity.csv")
def test_sensitivity_table_carries_warning():
    s = pd.read_csv(WEB / "pitcher_sensitivity.csv")
    assert s["confiabilidad_global"].nunique() == 1 and s["confiabilidad_global"].iloc[0] < 0.3
    assert s["advertencia"].str.contains("exploratorio").all()
    assert (s["s_shrunk_lo80"] <= s["s_shrunk_hi80"]).all()


@needs(pp.RESULTS_PATH)
def test_profile_results_reproducible():
    """El perfil depende solo de semillas fijas: recalcular los intervalos da exactamente lo mismo."""
    res = pp.run()
    t = res["pitchers"]
    elig = res["elegibles"]
    again = pp.profile_scores(t.drop(columns=[c for c in t.columns if c.startswith(("aepp", "rank", "p_top25", "bono_altitud"))]),
                              pp.load()[0], res["pesos"], elig)
    assert np.allclose(again["aepp_lo80"].dropna(), t["aepp_lo80"].dropna())
    assert np.array_equal(again["rank"].dropna(), t["rank"].dropna())
