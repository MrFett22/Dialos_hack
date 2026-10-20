"""Pruebas de los entregables operativos y de la validación fuera de muestra del modelo final."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "data" / "web"
needs = pytest.mark.skipif(not (WEB / "pitch_shape_altitude_effects.csv").exists() or not (WEB / "validation_holdout.json").exists(),
                           reason="requiere los pasos holdout_final y deliverables del pipeline")


@needs
def test_holdout_validation_beats_null_and_is_calibrated():
    V = json.loads((WEB / "validation_holdout.json").read_text(encoding="utf-8"))["esquemas"]
    assert set(V) == {"pitcher_holdout", "season_holdout_2026", "park_holdout_extreme", "park_holdout_medium"}
    for k, e in V.items():
        assert e["rmse_modelo"] < e["rmse_base_conteo"], f"{k}: no supera al modelo nulo por conteo"
        assert e["skill_vs_conteo"] == pytest.approx(1 - (e["rmse_modelo"] / e["rmse_base_conteo"]) ** 2, abs=1e-9)
        assert 0.8 < e["pendiente_calibracion"] < 1.2
        assert e["stuff"]["auc"] > e["stuff"]["auc_base_tipo"]
        dec = pd.DataFrame(e["calibracion_deciles"])
        assert dec["predicho"].is_monotonic_increasing and len(dec) == 10


@needs
def test_aepp_calculator_reproduces_official_aepp():
    """La calculadora de prospectos usa la misma fórmula, pesos y medias que el AEPP del ranking."""
    R = json.loads((WEB / "prospect_reference.json").read_text(encoding="utf-8"))
    mu, w = R["media_liga"], R["pesos_altitud"]
    p = pd.read_csv(WEB / "pitcher_altitude_profiles.csv")
    rk, rbb, rh = p["k_pct"].clip(0, 1), p["bb_pct"].clip(0, 1), p["hbp_pct"].clip(0, 1)
    A = -100 * ((rk - mu["k_pct"]) * (w["v_K"] - w["v_BIP"]) + (rbb - mu["bb_pct"]) * (w["v_BB"] - w["v_BIP"])
                + (rh - mu["hbp_pct"]) * (w["v_HBP"] - w["v_BIP"]) + (1 - rk - rbb - rh) * (p["rv_bip"] - mu["rv_bip"]))
    assert np.abs(A - p["aepp"]).max() < 1e-3
    assert len(R["aepp_elegibles"]) == R["n_elegibles"] == len(p)


@needs
def test_shape_effects_are_complete_and_coherent():
    e = pd.read_csv(WEB / "pitch_shape_altitude_effects.csv")
    assert set(e["forma"]) <= {"bajo", "medio", "alto"}
    for c in ("stuff", "ivb", "hb_abs", "whiff_swing", "hard", "hr_bip", "rv_bip", "valor_100"):
        assert e[f"{c}_E"].notna().all() and (e[f"{c}_E_se"] > 0).all()
    # en altitud los pitches pierden movimiento: la media del cambio de |HB| es negativa en cada tipo con muestra
    assert (e.groupby("tipo")["hb_abs_E"].mean() < 0).all()


@needs
def test_movement_map_and_arsenals():
    m = pd.read_csv(WEB / "movement_map.csv")
    assert (m["n"] >= 50).all()
    for t, g in m.groupby("AutoPitchType"):
        top, bot = g[g["grupo"] == "10% más efectivo"], g[g["grupo"] == "10% menos efectivo"]
        assert len(top) and len(bot) and top["stuff_plus_tipo"].min() > bot["stuff_plus_tipo"].max()
    a = pd.read_csv(WEB / "arsenal_altitude.csv")
    assert (a["pitchers_con_extreme"] >= 8).all()
    assert ((a["stuff_extreme_lo"] <= a["stuff_plus_extreme"]) & (a["stuff_plus_extreme"] <= a["stuff_extreme_hi"])).all()
    assert a["stuff_plus_extreme"].is_monotonic_decreasing
