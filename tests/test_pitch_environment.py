"""Pruebas de la extensión: Pitching+ general vs. altitud y lanzamiento óptimo por ambiente."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "data" / "web"
needs = pytest.mark.skipif(not (WEB / "pitching_env_values.json").exists() or not (WEB / "optimal_pitch_profiles.csv").exists(),
                           reason="requiere el paso pitch_environment del pipeline")
PROFILE_COLS = ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight", "RelSide"]
DOMAIN_OF = {"RelSpeed": "RelSpeed", "InducedVertBreak": "InducedVertBreak", "HorzBreak": "hb_arm", "SpinRate": "SpinRate",
             "Extension": "Extension", "RelHeight": "RelHeight", "RelSide": "rel_side_arm"}


@pytest.fixture(scope="module")
def EV():
    return json.loads((WEB / "pitching_env_values.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def E():
    from web.engine import Engine
    return Engine()


# Parte B ---------------------------------------------------------------------
@needs
def test_general_values_are_the_official_ones(EV):
    """El Pitching+ "general" usa exactamente los Δ(evento, conteo) del Pitching+ oficial."""
    assert EV["verificacion"]["max_dif_deltas_general_vs_oficial"] < 1e-12
    assert EV["verificacion"]["max_dif_xrv_general_vs_oficial"] < 1e-12
    assert EV["conjuntos"]["general"]["shift_bip"] == 0.0


@needs
def test_altitude_values_have_no_manual_bonus(EV):
    """Los valores de altitud salen de lo observado: la calibración es Δ observado − predicho de las bolas en juego."""
    for k in ("no_altitude", "medium", "extreme"):
        c = EV["conjuntos"][k]
        assert c["shift_bip"] == pytest.approx(c["e_bip_observado"] - c["e_bip_predicho"])
    # coherencia con P2: el modelo subestima el daño por bola en juego en altitud (ajuste positivo) y no en No Altitude
    assert EV["conjuntos"]["extreme"]["shift_bip"] > 0 and EV["conjuntos"]["medium"]["shift_bip"] > 0
    # en un conteo de 2 strikes, el foul es auto-lazo (Δ = 0) en todos los ambientes
    for k in EV["conjuntos"]:
        assert EV["conjuntos"][k]["deltas"]["1-2"]["foul"] == 0.0


@needs
def test_engine_general_equals_official_pitching_plus(E):
    """En el simulador, E[Δ] general de un lanzamiento = el E[Δ] del Pitching+ oficial; Stuff+ y Location+ no cambian."""
    for pt, b, s in (("Slider", 0, 2), ("Four-Seam", 1, 0), ("Changeup", 3, 2)):
        x = {**E.median_inputs(pt), "balls": b, "strikes": s}
        r = E.simulate(x)
        g = E.general_vs_altitude(r)
        assert g["general"]["xrv"] == pytest.approx(r.xrv_per_100 / 100, abs=1e-10)
        assert np.isfinite(g["alt"]["pitching_plus"])
        assert g["general"]["dano_contacto"] == pytest.approx(g["general"]["e_delta_bip"] * g["p_bip"], rel=1e-9)
        # la descomposición por grupos explica el cambio relativo frente al típico del conteo
        rel = (g["alt"]["relativo_conteo"] - g["general"]["relativo_conteo"])
        assert -100 * rel == pytest.approx(g["cambio_relativo_100"], abs=1e-9)
        assert isinstance(g["explicacion"], str) and len(g["explicacion"]) > 20


# Parte C ---------------------------------------------------------------------
@needs
def test_optimal_profiles_inside_support():
    """1. Todo perfil óptimo y todo mejor-observado está dentro del rango real de los candidatos (son lanzamientos reales o su
    mediana), y los perfiles fuera de p1–p99 de los candidatos quedan marcados como "frontera" (se reportan, no se ocultan)."""
    meta = json.loads((WEB / "optimal_pitch_meta.json").read_text(encoding="utf-8"))["tipos"]
    P = pd.read_csv(WEB / "optimal_pitch_profiles.csv")
    for r in P.itertuples():
        rng = meta[r.tipo]["rango_candidatos_p1_p99_min_max"]
        flagged = set(str(r.frontera).split(";")) if isinstance(r.frontera, str) else set()
        for c in PROFILE_COLS:
            p1, p99, lo, hi = rng[c]
            for suf in ("_med", "_mejor"):
                v = getattr(r, c + suf)
                assert lo - 1e-6 <= v <= hi + 1e-6, f"{r.tipo} {r.objetivo} {c}{suf}={v} fuera del rango real [{lo}, {hi}]"
            assert (not p1 <= getattr(r, c + "_med") <= p99) == (c in flagged), f"{r.tipo} {r.objetivo} {c}: marca de frontera incorrecta"


@needs
def test_optimal_profiles_have_data_density_and_no_joint_extrapolation():
    """2–3. Hay densidad de datos cerca del perfil y la combinación no extrapola (Mahalanobis ≤ p99.9 de los reales del tipo)."""
    dom = json.loads((WEB / "simulator_domain.json").read_text(encoding="utf-8"))
    cols = dom["columnas"]
    P = pd.read_csv(WEB / "optimal_pitch_profiles.csv")
    assert (P["densidad_percentil"] < 99).all(), "algún perfil cae en una zona casi vacía"
    for r in P.itertuples():
        t = dom["tipos"][r.tipo]
        v = np.array([getattr(r, {v_: k for k, v_ in DOMAIN_OF.items()}[c] + "_med") for c in cols]) - np.array(t["media"])
        d = float(np.sqrt(v @ np.array(t["cov_inv"]) @ v))
        assert d <= t["mahalanobis_cuantiles"]["99.9"], f"{r.tipo} {r.objetivo} {r.contexto}: distancia {d:.2f}"


@needs
def test_optimum_stability_is_reported():
    """4. La estabilidad del óptimo se mide y se reporta: vecinos reales del perfil con dispersión finita."""
    P = pd.read_csv(WEB / "optimal_pitch_profiles.csv")
    assert P["mejor_estable"].isin([True, False]).all()
    assert np.isfinite(P["sd_vecinos_perfil"]).all()
    # el perfil (mediana del top 1%) es menos extremo que el mejor observado: no es un pico aislado
    pp = P[P["objetivo"] != "stuff"]
    assert (pp["puntaje_perfil_vecinos"] <= pp["puntaje_mejor_observado"] + 1e-9).all()


@needs
def test_comparison_classes_follow_the_declared_rule():
    C = pd.read_csv(WEB / "optimal_pitch_comparison.csv")
    expect = np.where(C["distancia_z"] <= C["ruido_q95"], "Poco", np.where(C["distancia_z"] <= 2 * C["ruido_q95"], "Moderado", "Mucho"))
    assert (C["clase"].to_numpy() == expect).all()
    assert C["spearman"].between(-1, 1).all() and C["jaccard_top1"].between(0, 1).all()


@needs
def test_location_grid_only_uses_supported_cells():
    meta = json.loads((WEB / "optimal_pitch_meta.json").read_text(encoding="utf-8"))
    H = pd.read_csv(WEB / "optimal_location_heatmap.csv")
    assert (H["n_celda"] >= meta["loc_min"]).all()
    L = pd.read_csv(WEB / "optimal_pitch_location.csv")
    for r in L.itertuples():
        h = H[(H["tipo"] == r.tipo) & (H["enfrentamiento"] == r.enfrentamiento)]
        if len(h):
            assert ((h["x"] - r.x).abs().lt(1e-6) & (h["z"] - r.z).abs().lt(1e-6)).any(), "óptimo en una celda sin soporte"


# Guía para presentadores --------------------------------------------------------
@needs
def test_presentation_guide_cites_current_numbers(EV):
    """Las cifras clave de la guía (escrita a mano) coinciden con los outputs actuales del pipeline."""
    g = (ROOT / "reports" / "guia_para_presentacion.md").read_text(encoding="utf-8")
    S = json.loads((WEB / "summary.json").read_text(encoding="utf-8"))
    C = pd.read_csv(WEB / "optimal_pitch_comparison.csv")

    def m(x, d=1):  # signo menos tipográfico, como en la guía
        return f"{x:+.{d}f}".replace("-", "−")

    ef = S["efectos"]
    expected = [
        f"{m(100 * ef['whiff_swing']['extreme'])} puntos porcentuales", f"{m(100 * ef['k_pa']['extreme'])} pp", f"{ef['rv_bip']['extreme']:+.3f}",
        f"{S['aepp']['rho_pesos_altitud_vs_no']:.3f}", f"{S['aepp']['ancho_mediano_rango_80']:.0f} posiciones",
        f"{S['sensibilidad']['confiabilidad']:.2f}", f"{ef['whiff_swing']['consistencia_extreme'][0]} de {ef['whiff_swing']['consistencia_extreme'][1]}",
        f"{-S['pesos']['k_menos_bip']['Extreme']:.3f} contra {-S['pesos']['k_menos_bip']['No Altitude']:.3f}",
        f"{100 * S['G_aumento_relativo_extreme']:.0f}%", f"{EV['comparacion_lanzamientos']['spearman_general_extreme']:.3f}",
        f"{S['stuff']['estabilidad']['stuff_plus_physical']:.2f}", f"{S['ahorro_rodado']['No Altitude']:.3f}", f"{S['ahorro_rodado']['Extreme']:.3f}",
    ]
    oo = json.loads((WEB / "outs_on_play_check.json").read_text(encoding="utf-8"))
    expected += [f"{100 * oo['pct_turnos'][c]:.1f}%" for c in ("No Altitude", "Extreme Altitude")]
    expected += [f"{100 * oo['pct_rodados'][c]:.1f}%" for c in ("No Altitude", "Extreme Altitude")]
    expected.append(f"{oo['sesgo_extreme_menos_no']:+.3f} de las")
    for alt in ("extreme", "extreme_fisica"):
        x = C[C["alterno"] == alt]
        expected.append(f"{int((x['clase'] == 'Poco').sum())} de {len(x)}")
    missing = [e for e in expected if e not in g]
    assert not missing, f"la guía no cita estas cifras actuales: {missing}"
