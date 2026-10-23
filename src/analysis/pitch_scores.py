"""Stuff+, Location+ y Pitching+ de cada lanzamiento real (entregable 1: score individualizado por lanzamiento).

Usa los modelos FINALES y la misma escala de lanzamiento que el simulador (`simulator_scales.json`): 100 = lanzamiento
promedio de la liga, 10 = 1 sd entre lanzamientos; Stuff+ y Pitching+ con el ajuste de la categoría de altitud en la que
se lanzó. Así, un lanzamiento real y uno del simulador se leen en la misma escala.

La tabla no incluye identificadores de lanzamiento, juego ni bateador: solo el pitcher anónimo, la temporada, el
contexto (ambiente, conteo, mano del bateador), las características físicas principales, el resultado y los tres scores.
"""

import json
import logging

import numpy as np
import pandas as pd

from src.analysis import p2_common as pc
from src.reports import web_data

WEB = pc.WEB_DIR
OUT_PATH = WEB / "pitch_scores.parquet"
RESULTADO = {"ball": "bola", "called_strike": "strike cantado", "hbp": "golpeado", "whiff": "swing fallado", "foul": "foul", "out": "out",
             "single": "sencillo", "double": "doble", "triple": "triple", "home_run": "jonrón"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def plus(values: np.ndarray, s: dict, env: pd.Series | None = None) -> np.ndarray:
    """Misma fórmula que `web/engine.py` (`Engine._plus`)."""
    v = values + (env.map(s["offset_por_ambiente"]).to_numpy() if env is not None else 0.0)
    z = (v - s["media_lanzamiento"]) / s["sd_lanzamiento"]
    return 100 + 10 * (z if s["mayor_es_mejor"] else -z)


def build(values: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> pd.DataFrame:
    df, ld = values if values is not None else web_data.final_pitch_values()
    sc = json.loads((WEB / "simulator_scales.json").read_text(encoding="utf-8"))
    s = df[["PitchUID", "altitude_category"]].copy()
    s["stuff_plus"] = plus(df["p_st"].to_numpy(), sc["stuff"], df["altitude_category"])
    s["pitching_plus"] = plus(df["xrv"].to_numpy(), sc["pitching"], df["altitude_category"])
    s = s.merge(pd.DataFrame({"PitchUID": ld["PitchUID"], "location_plus": plus(ld["loc_value"].to_numpy(), sc["location"])}),
                on="PitchUID", how="left")

    f = pc.load_pitch_frame()
    f = f[["PitchUID", "pitcher_anon_id", "year", "AutoPitchType", "PitcherThrows", "BatterSide", "Balls", "Strikes", "event",
           "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "PlateLocSide", "PlateLocHeight"]]
    t = f.merge(s.drop(columns="altitude_category"), on="PitchUID", how="inner").merge(
        df[["PitchUID", "altitude_category"]], on="PitchUID", how="left")
    t = t[t["altitude_category"].isin(pc.CATS) & t["AutoPitchType"].isin(pc.MAIN_TYPES)]
    t["hb_brazo"] = t["HorzBreak"] * web_data.hand_sign(t["PitcherThrows"])
    out = pd.DataFrame({
        "pitcher_anon_id": t["pitcher_anon_id"], "temporada": t["year"].astype(int), "tipo": t["AutoPitchType"].astype(str),
        "ambiente": t["altitude_category"].astype(str), "conteo": t["Balls"].astype(int).astype(str) + "-" + t["Strikes"].astype(int).astype(str),
        "bateador": t["BatterSide"].astype("object").map({"Right": "derecho", "Left": "zurdo"}), "velocidad": t["RelSpeed"].round(1), "ivb": t["InducedVertBreak"].round(1),
        "hb_brazo": t["hb_brazo"].round(1), "giro": t["SpinRate"].round(0), "lado_placa": t["PlateLocSide"].round(2),
        "altura_placa": t["PlateLocHeight"].round(2), "resultado": t["event"].astype(str).map(RESULTADO),
        "stuff_plus": t["stuff_plus"].round(1), "location_plus": t["location_plus"].round(1), "pitching_plus": t["pitching_plus"].round(1)})
    for c in ("pitcher_anon_id", "tipo", "ambiente", "conteo", "bateador", "resultado"):
        out[c] = out[c].astype("category")
    return out.sort_values(["pitcher_anon_id", "temporada"]).reset_index(drop=True)


def run() -> pd.DataFrame:
    logger.info("Scores por lanzamiento con los modelos finales…")
    out = build()
    out.to_parquet(OUT_PATH, index=False, compression="zstd")
    logger.info("%s lanzamientos en %s (%.1f MB)", f"{len(out):,}", OUT_PATH, OUT_PATH.stat().st_size / 1e6)
    web_data.write_readme()
    return out


if __name__ == "__main__":
    run()
