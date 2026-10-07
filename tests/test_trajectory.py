"""Pruebas de la física de trayectoria, el shrinkage y el dashboard v3.

Requieren los artefactos del pipeline (data/processed/*, src/models/artifacts/pitching_plus_v3.pkl).
Correr con:  .venv\\Scripts\\python -m pytest tests
"""

import numpy as np
import pandas as pd
import pytest

from src.features import trajectory
from src.models import pitching_plus_neutral

RAW_COLUMNS = trajectory.TRAJECTORY_COLUMNS + [
    "PitchUID", "altitude_category", "AutoPitchType", "Extension", "RelSpeed", "PlateLocSide", "PlateLocHeight",
    "ZoneSpeed", "ZoneTime", "VertApprAngle", "HorzApprAngle", "InducedVertBreak", "HorzBreak", "VertBreak", "SpeedDrop",
]


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    df = pd.read_parquet(trajectory_raw_path(), columns=RAW_COLUMNS)
    df = df.loc[df["altitude_category"].isin(trajectory.ALTITUDE_CATEGORIES)].sample(5000, random_state=0)
    for c in RAW_COLUMNS:
        if c not in ("PitchUID", "altitude_category", "AutoPitchType"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def trajectory_raw_path():
    from src.analysis.altitude_counterfactual_v2 import RAW_PATH
    return RAW_PATH


@pytest.fixture(scope="module")
def ratios() -> pd.DataFrame:
    from src.analysis.altitude_counterfactual_v2 import RATIOS_PATH
    return pd.read_parquet(RATIOS_PATH)


def test_projection_to_own_category_is_identity(sample, ratios):
    for cat in trajectory.ALTITUDE_CATEGORIES:
        sub = sample.loc[sample["altitude_category"] == cat]
        proj = trajectory.project_to_altitude(sub, cat, ratios)
        diff = (proj[trajectory.UPDATED_FEATURES] - sub[trajectory.UPDATED_FEATURES]).abs().to_numpy()
        assert np.nanmax(diff) < 1e-4


def test_extreme_altitude_reduces_fastball_movement_and_raises_zone_speed(sample, ratios):
    fb = sample.loc[(sample["altitude_category"] == "No Altitude") & (sample["AutoPitchType"] == "Four-Seam")]
    proj = trajectory.project_to_altitude(fb, "Extreme Altitude", ratios)
    assert (proj["InducedVertBreak"] - fb["InducedVertBreak"]).median() < -1.0
    assert (proj["ZoneSpeed"] - fb["ZoneSpeed"]).median() > 0.5
    # supuesto de comando: la ubicación en el plato no cambia
    assert np.allclose(proj["PlateLocHeight"], fb["PlateLocHeight"])


def test_density_ratios_are_below_one(ratios):
    glob = ratios.loc[ratios["AutoPitchType"] == "__all__"].set_index("altitude_category")
    assert glob.loc["No Altitude", "drag_ratio"] == 1.0
    assert 0.7 < glob.loc["Extreme Altitude", "drag_ratio"] < glob.loc["Medium Altitude", "drag_ratio"] < 1.0
    assert glob.loc["Extreme Altitude", "magnus_ratio"] < glob.loc["Extreme Altitude", "drag_ratio"]


def test_synthesized_pitch_reproduces_inputs_in_reference_category(ratios):
    for ref in trajectory.ALTITUDE_CATEGORIES:
        out = trajectory.synthesize_pitch(
            rel_speed_mph=93.0, ivb_in=16.0, hb_in=8.0, extension_ft=6.2, rel_height_ft=5.8, rel_side_ft=1.6,
            drag_k=0.0015, target_side_ft=0.0, target_height_ft=2.5, reference_category=ref, pitch_type="Four-Seam",
            ratios=ratios,
        ).set_index("altitude_category")
        assert out.loc[ref, "InducedVertBreak"] == pytest.approx(16.0, abs=0.01)
        assert out.loc[ref, "HorzBreak"] == pytest.approx(8.0, abs=0.01)
        assert out.loc[ref, "PlateLocHeight"] == pytest.approx(2.5, abs=1e-6)
        assert out.loc["Extreme Altitude", "InducedVertBreak"] < out.loc["No Altitude", "InducedVertBreak"]


def test_empirical_bayes_shrinks_small_samples_more():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        n = int(rng.integers(5, 1500))
        talent = rng.normal(0, 0.006)
        rows.append(pd.DataFrame({"pitcher_anon_id": f"p{i}", "year": 2025, "x": rng.normal(talent, 0.05, n)}))
    df = pd.concat(rows, ignore_index=True)
    table, params = pitching_plus_neutral.empirical_bayes(df, "x")
    assert params[2025]["tau"] == pytest.approx(0.006, rel=0.35)
    small, big = table.loc[table["n"] < 30], table.loc[table["n"] > 1000]
    assert small["shrink_factor"].mean() > big["shrink_factor"].mean()
    assert (table["plus_lo80"] <= table["plus"]).all() and (table["plus"] <= table["plus_hi80"]).all()


def test_dashboard_left_and_right_handers_are_mirror_images():
    from dashboard import scoring
    art, ratios_, defaults = scoring.load_artifact(), scoring.load_ratios(), scoring.compute_pitch_defaults()
    base = dict(RelSpeed=93.0, InducedVertBreak=16.0, SpinRate=2300, Extension=6.2, RelHeight=5.8, AutoPitchType="Four-Seam")
    right = scoring.compute_altitude_scenarios(dict(base, HorzBreak=8.0, PitcherThrows="Right", BatterSide="Right"), "No Altitude", art, ratios_, defaults)
    left = scoring.compute_altitude_scenarios(dict(base, HorzBreak=-8.0, PitcherThrows="Left", BatterSide="Left"), "No Altitude", art, ratios_, defaults)
    assert np.allclose(right["runs_per_100"], left["runs_per_100"], atol=0.25)
