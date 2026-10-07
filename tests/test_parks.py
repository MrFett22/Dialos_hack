"""Pruebas de los grupos de estadio anónimos y de la proyección con ratios por estadio."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import parks
from src.features import trajectory as tr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "stuff_model_df.parquet"
needs = lambda p: pytest.mark.skipif(not p.exists(), reason=f"requiere {p.name}")  # noqa: E731


def test_ari_identity_and_relabeling():
    a = pd.Series([0, 0, 1, 1, 2, 2])
    assert parks.ari(a, a) == pytest.approx(1.0)
    assert parks.ari(a, a.map({0: 7, 1: 3, 2: 5})) == pytest.approx(1.0)  # los nombres de los grupos no importan
    assert parks.ari(a, pd.Series([0, 1, 0, 1, 0, 1])) < 0.1


@needs(parks.OUT_PATH)
def test_anonymous_ids_and_quality():
    g = pd.read_parquet(parks.OUT_PATH)
    assert g["stadium_id"].dropna().str.fullmatch(r"ANON_STADIUM_\d{2}").all()  # solo IDs anónimos
    res = g[g["status"] == "resuelto"]
    assert len(res) / len(g) > 0.95
    assert (g["stadium_id"] == g["away_stadium_id"]).mean() < 0.02  # local ≠ visitante
    # cada grupo resuelto conserva su categoría entre temporadas (la categoría no se usó para enlazar)
    cats = res.groupby("stadium_id")["stadium_category"].nunique()
    assert (cats == 1).all()


@needs(parks.RATIOS_PATH)
def test_stadium_ratios_are_physical():
    r = parks.load_stadium_ratios()
    assert r["drag_ratio"].between(0.6, 1.2).all() and r["magnus_ratio"].between(0.5, 1.3).all()
    ext = r[r["categoria"] == "Extreme Altitude"]["drag_ratio"]
    no = r[r["categoria"] == "No Altitude"]["drag_ratio"]
    assert ext.max() < no.min()  # todos los estadios de Extreme tienen menos drag que todos los de No Altitude


@needs(RAW)
def test_projection_with_equal_ratios_is_identity_and_fallback_works():
    d = pd.read_parquet(RAW).sample(3000, random_state=1)
    for c in tr.TRAJECTORY_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=tr.TRAJECTORY_COLUMNS + ["PlateLocSide", "PlateLocHeight", "Extension", "altitude_category"]).reset_index(drop=True)
    x = np.full(len(d), 0.85)
    same = tr.project_with_ratios(d, x, x, x, x)
    # SpeedDrop viene redondeado en los datos (≈1e-5), el resto es exacto
    assert (same[tr.UPDATED_FEATURES] - d[tr.UPDATED_FEATURES].astype(float)).abs().max().max() < 1e-4
    cat = pd.read_parquet(ROOT / "data" / "processed" / "altitude_density_ratios.parquet")
    empty = pd.DataFrame(columns=["drag_ratio", "magnus_ratio"])
    drag, mag, has = parks.pitch_ratios(d, pd.Series(dtype=object), cat, empty)
    cd, cm = tr.lookup_ratios(cat, d["altitude_category"], d["AutoPitchType"])
    assert not has.any() and np.allclose(drag, cd) and np.allclose(mag, cm)  # sin estadio → ratio de la categoría
