"""Pruebas de los targets y de los valores en carreras por evento.

Origen: la auditoría del 2026-10-05 encontró que `is_hit_by_pitch` venía en 0
en todo el dataset, así que el golpeado no cerraba el turno y valía ≈ 0
carreras. Estas pruebas fallan si un target de evento terminal queda vacío o
constante, o si el valor de un evento tiene el signo equivocado.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import targets

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
needs_data = pytest.mark.skipif(not TARGETS_PATH.exists(), reason="requiere data/processed/pitches_with_targets.parquet")


def test_fix_is_hit_by_pitch_uses_pitchcall():
    df = pd.DataFrame({
        "PitchCall": ["BallCalled", "HitByPitch", "InPlay", "HitByPitch"],
        "is_hit_by_pitch": [0, 0, 0, 0],
        "base_on_balls": [0, 0, 0, 1],
        "play_result": ["BallCalled", "HitByPitch", "Out", "Walk"],
    })
    out, rep = targets.fix_is_hit_by_pitch(df)
    assert out["is_hit_by_pitch"].tolist() == [0, 1, 0, 1]
    assert rep["n_before"] == 0 and rep["n_after"] == 2 and rep["n_also_flagged_walk"] == 1


@needs_data
def test_terminal_event_targets_not_empty():
    """Ningún indicador de evento terminal puede estar vacío o ser constante."""
    t = pd.read_parquet(TARGETS_PATH, columns=["PitchCall", "is_hit_by_pitch", "strikeout", "base_on_balls", "is_ball_in_play"])
    for col in ["is_hit_by_pitch", "strikeout", "base_on_balls", "is_ball_in_play"]:
        s = pd.to_numeric(t[col], errors="coerce")
        assert s.nunique(dropna=True) == 2, f"{col} es constante"
        assert s.sum() > 100, f"{col} casi vacío ({s.sum()})"
    assert (t["is_hit_by_pitch"] == (t["PitchCall"] == "HitByPitch").astype(int)).all()


@needs_data
def test_hbp_run_value_is_like_a_walk():
    """El golpeado pone al bateador en primera: su run value medio debe parecerse al de la base por bolas."""
    t = pd.read_parquet(TARGETS_PATH, columns=["PitchCall", "KorBB", "run_value"])
    hbp = t.loc[t["PitchCall"] == "HitByPitch", "run_value"].mean()
    walk = t.loc[(t["KorBB"] == "Walk") & (t["PitchCall"] != "HitByPitch"), "run_value"].mean()
    assert hbp > 0.2, f"run value medio del golpeado = {hbp:.3f}"
    assert abs(hbp - walk) < 0.1, f"golpeado {hbp:.3f} vs base por bolas {walk:.3f}"


@needs_data
def test_event_value_signs_all_counts():
    """Δ(evento, conteo): bola y golpeado > 0, strike < 0, out < 0, hits > 0, en los 12 conteos."""
    from src.models import run_value_chain as chain

    df = chain.load_data()
    _, deltas = chain.count_values(df, chain.terminal_values(df))
    assert len(deltas) == 12
    assert (deltas["ball"] > 0).all()
    assert (deltas["hbp"] > 0).all(), deltas["hbp"].to_dict()
    assert (deltas["hbp"] >= deltas["ball"] - 1e-12).all(), "un golpeado no puede valer menos que una bola"
    assert (deltas[["called_strike", "whiff"]] < 0).all().all()
    assert (deltas["out"] < 0).all()
    assert (deltas[["single", "double", "triple", "home_run"]] > 0).all().all()
    assert np.allclose(deltas.loc[(slice(None), 2), "foul"], 0.0)
