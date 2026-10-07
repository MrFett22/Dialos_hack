"""Pruebas del simulador y de la página (web/)."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "data" / "web"
needs = pytest.mark.skipif(not (WEB / "simulator_scales.json").exists() or not (ROOT / "src/models/artifacts/location_plus_v2.pkl").exists(),
                           reason="requiere los datos de la web y los modelos finales")


@pytest.fixture(scope="module")
def E():
    from web.engine import Engine
    return Engine()


def real_inputs(n: int = 150, seed: int = 0) -> list[dict]:
    from src.analysis import p2_common as pc
    d = pd.read_parquet(pc.ARSENAL_PATH, columns=["AutoPitchType", "PitcherThrows", "BatterSide", "RelSpeed", "InducedVertBreak", "HorzBreak",
                                                  "SpinRate", "Extension", "RelHeight", "RelSide", "PlateLocSide", "PlateLocHeight", "Balls",
                                                  "Strikes", "altitude_category"])
    d = d[d["AutoPitchType"].astype(str).isin(["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"])
          & d["PitcherThrows"].isin(["Right", "Left"]) & d["BatterSide"].isin(["Right", "Left"])
          & d["altitude_category"].isin(["No Altitude", "Medium Altitude", "Extreme Altitude"])
          & d["Balls"].between(0, 3) & d["Strikes"].between(0, 2)].dropna()
    d = d.sample(n, random_state=seed)
    out = []
    for r in d.itertuples():
        sign = 1.0 if r.PitcherThrows == "Right" else -1.0
        out.append({"pitch_type": str(r.AutoPitchType), "p_throws": r.PitcherThrows, "b_side": r.BatterSide, "velo": r.RelSpeed,
                    "ivb": r.InducedVertBreak, "hb_arm": r.HorzBreak * sign, "spin_rate": r.SpinRate, "extension": r.Extension,
                    "rel_height": r.RelHeight, "rel_side_arm": r.RelSide * sign, "spin_axis": None, "plate_x": r.PlateLocSide,
                    "plate_z": r.PlateLocHeight, "balls": int(r.Balls), "strikes": int(r.Strikes), "env": r.altitude_category})
    return out


# Test 1 ----------------------------------------------------------------------
@needs
def test_average_pitch_gives_average_score(E):
    res = [E.simulate(x) for x in real_inputs()]
    for k in ("pitching_plus", "stuff_plus", "location_plus"):
        m = np.mean([getattr(r, k) for r in res])
        assert abs(m - 100) < 5, f"{k}: media {m:.1f} en lanzamientos reales"
    for pt in ("Four-Seam", "Slider", "Changeup"):
        r = E.simulate(E.median_inputs(pt))
        assert 80 < r.pitching_plus < 120 and 70 < r.stuff_plus < 130


# Test 2 ----------------------------------------------------------------------
@needs
def test_out_of_range_inputs_warn(E):
    x = E.median_inputs("Four-Seam")
    assert E.simulate(x).warnings == []  # el lanzamiento mediano está dentro del dominio
    far = {**x, "velo": 110.0}
    w = E.simulate(far).warnings
    assert any("fuera del rango común" in s for s in w)
    combo = {**x, "ivb": E.type_domain("Four-Seam")["percentiles"]["InducedVertBreak"]["1"], "velo": E.type_domain("Four-Seam")["percentiles"]["RelSpeed"]["99"]}
    assert E.domain_check(combo)["mahalanobis"] > E.domain_check(x)["mahalanobis"]
    loc = {**E.median_inputs("Splitter"), "plate_x": 1.8, "plate_z": 4.4}  # esquina sin splitters reales
    assert E.domain_check(loc)["pitches_en_celda"] < 20
    assert any("pocos" in s for s in E.simulate(loc).warnings)


# Test 3 ----------------------------------------------------------------------
@needs
def test_location_changes_location_plus_not_stuff(E):
    x = E.median_inputs("Slider")
    a, b = E.simulate(x), E.simulate({**x, "plate_x": -0.9, "plate_z": 1.3})
    assert abs(a.location_plus - b.location_plus) > 1
    # Stuff+ Physical no usa la ubicación; solo el ángulo de llegada residualizado, que apenas cambia
    assert abs(a.stuff_plus - b.stuff_plus) < 0.25 * abs(a.location_plus - b.location_plus)


# Test 4 ----------------------------------------------------------------------
@needs
def test_physical_changes_move_stuff(E):
    x = E.median_inputs("Four-Seam")
    a = E.simulate(x)
    b = E.simulate({**x, "velo": x["velo"] + 3, "ivb": x["ivb"] + 3})
    assert abs(b.stuff_plus - a.stuff_plus) > 1
    assert b.location_plus == pytest.approx(a.location_plus)  # Location+ no ve la física


# Test 5 ----------------------------------------------------------------------
@needs
def test_probabilities_sum_to_one(E):
    for x in real_inputs(20, seed=1):
        r = E.simulate(x)
        assert sum(r.probs.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0 <= p <= 1 for p in r.probs.values())
        assert r.p_bip == pytest.approx(sum(r.probs[e] for e in ("out", "single", "double", "triple", "home_run")))
        for _, row in r.by_env.iterrows():
            assert sum(row[f"p_{e}"] for e in r.probs) == pytest.approx(1.0, abs=1e-9)


# Test 6 ----------------------------------------------------------------------
@needs
def test_pitching_plus_decreases_with_expected_runs(E):
    vals = np.linspace(-0.05, 0.05, 11)
    plus = [E._plus("pitching", v, "No Altitude") for v in vals]
    assert np.all(np.diff(plus) < 0)  # más carreras esperadas → menos Pitching+
    res = [E.simulate(x) for x in real_inputs(30, seed=2)]
    same_env = {}
    for r in res:
        same_env.setdefault(r.env, []).append((r.xrv_per_100, r.pitching_plus))
    for pairs in same_env.values():
        pairs.sort()
        assert all(p1[1] >= p2[1] - 1e-9 for p1, p2 in zip(pairs, pairs[1:]))


# Test 7 ----------------------------------------------------------------------
@needs
def test_missing_values_do_not_break(E):
    x = E.median_inputs("Changeup")
    for bad in ({**x, "velo": None}, {**x, "ivb": float("nan"), "spin_rate": None}, {"pitch_type": "Changeup"},
                {**x, "pitch_type": "Knuckleball"}, {**x, "env": None, "p_throws": None, "balls": 7}):
        r = E.simulate(bad)
        assert math.isfinite(r.pitching_plus) and math.isfinite(r.stuff_plus) and math.isfinite(r.location_plus)
    r = E.simulate({**x, "velo": None})
    assert "velo" in r.imputed and any("mediana" in w for w in r.warnings)


# Extras ----------------------------------------------------------------------
@needs
def test_improvements_stay_in_range(E):
    x = E.median_inputs("Slider")
    imp = E.improvements(x)
    assert len(imp) > 0 and imp["delta_pitching_plus"].is_monotonic_decreasing
    dom = E.type_domain("Slider")
    for name in imp["cambio"]:
        if name.startswith("velocidad"):
            delta = float(name.split()[-1])
            assert dom["percentiles"]["RelSpeed"]["5"] <= x["velo"] + delta <= dom["percentiles"]["RelSpeed"]["95"]


@needs
def test_summary_matches_sources():
    S = json.loads((WEB / "summary.json").read_text(encoding="utf-8"))
    hyp = json.loads((ROOT / "reports" / "benchmark" / "hipotesis_altitud.json").read_text(encoding="utf-8"))
    rows = {r["variable"]: r for r in hyp["h1_h2"]}
    assert S["efectos"]["whiff_swing"]["extreme"] == pytest.approx(rows["whiff por swing"]["efectos_fijos"]["beta_extreme"])
    assert S["efectos"]["k_pa"]["extreme"] == pytest.approx(rows["ponche por turno"]["efectos_fijos"]["beta_extreme"])
    assert S["efectos"]["rv_bip"]["extreme"] == pytest.approx(rows["run value RE24 por bola en juego"]["efectos_fijos"]["beta_extreme"])
    G = hyp["h3"]["por_categoria"]
    assert S["G_aumento_relativo_extreme"] == pytest.approx(G["Extreme Altitude"]["G"] / G["No Altitude"]["G"] - 1)
    meta = json.loads((ROOT / "data" / "processed" / "run_value_chain_oof_pitches.meta.json").read_text(encoding="utf-8"))
    assert S["hbp_run_value"] == pytest.approx(meta["terminal_values"]["hbp"]) and S["hbp_run_value"] > 0.3  # posterior al fix de HBP
    sens = pd.read_csv(WEB / "pitcher_sensitivity.csv")
    assert S["sensibilidad"]["n_pitchers"] == len(sens)
    assert S["sensibilidad"]["confiabilidad"] == pytest.approx(sens["confiabilidad_global"].iloc[0])
    prof = pd.read_csv(WEB / "pitcher_altitude_profiles.csv")
    assert S["aepp"]["n_elegibles"] == len(prof)


@needs
def test_page_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Pitching+" in m.value for m in at.markdown)
    # cambiar de tipo de pitch y de ambiente no rompe la página
    at.selectbox(key="sim_type").set_value("Curveball").run()
    at.select_slider(key="sim_env").set_value("Extreme Altitude").run()
    assert not at.exception
    # sección de tipos de pitch y simulador con un tipo de muestra pequeña
    at.selectbox(key="type_sel").set_value("Splitter").run()
    at.radio(key="type_metric").set_value("rv_bip").run()
    at.selectbox(key="sim_type").set_value("Splitter").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "no destino garantizado" in text or "poca muestra" in text
    assert "no el efecto causal de elegir un tipo de pitch" in text


# Ampliación (tipos de pitch, regiones, daño de contacto) -----------------------
@needs
def test_contact_damage_is_bip_branch_of_tree(E):
    """Daño esperado de contacto = P(BIP) × E[RV | BIP] = Σ_k∈BIP P(k)·Δ(k, conteo), la rama de bola en juego del árbol."""
    from src.models import run_value_chain as chain
    D = E.pit["deltas_by_count"]
    for x in real_inputs(15, seed=3):
        r = E.simulate(x)
        d = D.loc[(r.inputs["balls"], r.inputs["strikes"])]
        branch = sum(r.probs[k] * d[k] for k in chain.BIP_CLASSES)
        assert r.contact_damage == pytest.approx(branch, abs=1e-9)
        assert r.contact_damage == pytest.approx(r.p_bip * r.e_delta_bip, abs=1e-12)
        total = sum(r.probs[k] * d[k] for k in chain.EVENTS)
        assert 100 * total == pytest.approx(r.xrv_per_100, abs=1e-6)  # el resto del pitch es E[Δ] − daño de contacto
        assert r.p_whiff == pytest.approx(r.probs["whiff"])


@needs
def test_region_and_pitch_type_context(E):
    S = json.loads((WEB / "summary.json").read_text(encoding="utf-8"))
    assert E.region_of(0.0, 2.5) == "corazón" and E.region_of(2.0, 0.3) == "desperdicio"
    reg = pd.DataFrame(S["location"]["regiones"]).set_index("region")
    rc = E.region_context(0.0, 2.5)
    assert rc["extreme_menos_no"] == pytest.approx(reg.loc["corazón", "extreme_menos_no_por_100"])
    assert rc["orden_perdida"] == 1 and reg["extreme_menos_no_por_100"].idxmax() == "corazón"
    TP = S["tipos_pitch"]
    best = E.pitch_type_context(TP["menor_costo_extreme"])
    assert best["muestra_suficiente"] and best["orden_costo"] == 1
    for t in TP["muestra_pequena"]:
        ctx = E.pitch_type_context(t)
        assert not ctx["muestra_suficiente"] and ctx["orden_costo"] is None


@needs
def test_extended_summary_matches_sources():
    S = json.loads((WEB / "summary.json").read_text(encoding="utf-8"))
    lw = pd.read_csv(WEB / "linear_weights_by_environment.csv").set_index("ambiente")
    for a in ("No Altitude", "Medium", "Extreme"):
        assert S["ahorro_rodado"][a] == pytest.approx(lw.loc[a, "v_aire"] - lw.loc[a, "v_GB"], abs=1e-5)  # CSV con 6 decimales
        assert S["valor_bip"][a] == pytest.approx(lw.loc[a, "v_BIP"], abs=1e-5)
    assert S["ahorro_rodado"]["Extreme"] > S["ahorro_rodado"]["No Altitude"]
    st = json.loads((ROOT / "reports" / "benchmark" / "stuff_plus_physical.json").read_text(encoding="utf-8"))
    assert S["stuff_versiones"]["observado"]["r_pct_extreme"] == pytest.approx(st["pitcher"]["stuff_plus_physical_observado"]["r_pct_extreme"])
    assert S["sensibilidad_extra"]["techo_correlacion"] == pytest.approx(math.sqrt(S["sensibilidad"]["confiabilidad"]))
    sh = json.loads((ROOT / "reports" / "benchmark" / "altitude_counterfactual_v2.json").read_text(encoding="utf-8"))["validation"]
    ivb = next(r for r in sh if r["categoria"] == "Extreme Altitude" and r["feature"] == "InducedVertBreak")
    assert S["contrafactual_split_half"]["sobreprediccion_ivb"] == pytest.approx(ivb["cambio_predicho_mediana"] / ivb["cambio_observado_mediana"] - 1)
    t = pd.read_csv(WEB / "pitch_type_altitude_effects.csv")
    ok = t[t["pitches"] >= S["tipos_pitch"]["min_pitches_comparacion"]]
    assert S["tipos_pitch"]["menor_costo_extreme"] == ok.loc[ok["delta_por_100_E"].idxmin(), "tipo"]
    prof = pd.read_csv(WEB / "pitcher_altitude_profiles.csv")
    assert S["aepp"]["ancho_mediano_rango_80"] == pytest.approx((prof["rank_hi80"] - prof["rank_lo80"]).median())
    assert S["ranking_top"]["n_p_top25_mayor_50"] == int((prof["p_top25"] > 0.5).sum())
