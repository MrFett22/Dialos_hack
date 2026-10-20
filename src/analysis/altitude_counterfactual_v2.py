"""Efecto de la altitud con el contrafactual de trayectoria completa y Pitching+ v3.

Reemplaza a src/analysis/altitude_counterfactual.py (v1), que:
- solo cambiaba IVB/HB (dejaba fijos VAA, velocidad en zona, tiempo de vuelo);
- suponía elevaciones (50 / 800 / 1,900 m) en vez de medirlas;
- expresaba el efecto en puntos de Pitching+ v1 y lo traducía a carreras con
  una pendiente circular.

Pasos:
1. Mide los ratios de densidad efectiva (drag y Magnus) dentro de pitcher,
   con validación split-half: se estiman con la mitad de los pitchers y se
   comprueba con la otra mitad que la re-simulación reproduce los cambios
   OBSERVADOS de IVB, HB, velocidad en zona, VAA y tiempo de vuelo entre
   categorías de altitud.
2. Chequeo de identidad: proyectar cada pitch a su propia categoría devuelve
   exactamente las features observadas.
3. Proyecta TODOS los pitches a No Altitude y a Extreme Altitude (mismo
   pitch, mismo objetivo en el plato) y predice E[Δ carreras] con los modelos
   finales de Pitching+ v3 (índice oficial). El efecto ya sale en carreras.
4. IC 95% por bootstrap de pitchers, por tipo de pitch y por componente.

Salidas: reports/altitude_counterfactual_v2.md,
data/processed/altitude_density_ratios.parquet,
data/processed/altitude_counterfactual_v2_pitches.parquet
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import trajectory
from src.models import chain_attribution as attr
from src.models import run_value_chain, stuff_plus_v2

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "stuff_model_df.parquet"
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
RATIOS_PATH = ROOT / "data" / "processed" / "altitude_density_ratios.parquet"
PITCHES_OUT = ROOT / "data" / "processed" / "altitude_counterfactual_v2_pitches.parquet"
REPORT_PATH = ROOT / "reports" / "altitude_counterfactual_v2.md"
CHAIN_OOF_PATH = ROOT / "data" / "processed" / "run_value_chain_oof_pitches.parquet"
# Atribución de Shapley por eslabón de la cadena (src/models/chain_attribution.py). "Antes del contacto" =
# swing + take + whiff + foul; "bip" (resultado de la bola en juego) mezcla calidad del contacto y parque.
SHAPLEY_COLUMNS = [f"sh_{k}" for k in attr.LINKS]
PRE_CONTACT_COLUMNS = [f"sh_{k}" for k in attr.PRE_CONTACT]
MECHANISM_FEATURES = ["InducedVertBreak", "HorzBreak", "ZoneSpeed", "VertApprAngle"]
JSON_PATH = ROOT / "reports" / "benchmark" / "altitude_counterfactual_v2.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_BOOT = 1000
BOOT_SEED = 42
PITCHES_PER_SEASON = {"abridor (2,750 pitches)": 2750, "relevista (900 pitches)": 900}
VALIDATION_FEATURES = ["InducedVertBreak", "HorzBreak", "ZoneSpeed", "VertApprAngle", "ZoneTime", "VertBreak"]
OLD_V1_DELTA_POINTS = {  # reports/altitude_counterfactual.md (Pitching+ v1, Extreme − No)
    "Sinker": -1.059, "Changeup": -0.951, "Cutter": -0.904, "Slider": -0.843,
    "Four-Seam": -0.839, "Curveball": -0.829, "Splitter": -0.817,
}
OLD_V1_RUNS_STARTER = {  # reports/runs_translation.md (punto medio abridor)
    "Sinker": 4.52, "Changeup": 4.06, "Cutter": 3.86, "Slider": 3.60,
    "Four-Seam": 3.58, "Curveball": 3.54, "Splitter": 3.49,
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Features crudas (sin espejar) + trayectoria + categoría de altitud + conteo."""
    base = stuff_plus_v2.load_data()
    raw = pd.read_parquet(RAW_PATH, columns=["PitchUID", "altitude_category"] + trajectory.TRAJECTORY_COLUMNS)
    for c in trajectory.TRAJECTORY_COLUMNS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    counts = pd.read_parquet(TARGETS_PATH, columns=["PitchUID", "Balls", "Strikes"])
    df = base.merge(raw, on="PitchUID", how="inner").merge(counts, on="PitchUID", how="inner")
    ok = (
        df["altitude_category"].isin(trajectory.ALTITUDE_CATEGORIES)
        & df[trajectory.TRAJECTORY_COLUMNS].notna().all(axis=1)
        & df["Balls"].between(0, 3) & df["Strikes"].between(0, 2)
        & df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()
    )
    logger.info("Pitches sin trayectoria/ubicación/conteo válido excluidos: %s", f"{int((~ok).sum()):,}")
    df = df.loc[ok].reset_index(drop=True)
    df["Balls"] = df["Balls"].astype(int)
    df["Strikes"] = df["Strikes"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 1. Validación split-half de la re-simulación
# ---------------------------------------------------------------------------


def split_half_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ratios con la mitad A de pitchers; con la mitad B se compara el cambio predicho vs el observado.

    Para cada grupo (pitcher, temporada, tipo) de la mitad B con ≥15 pitches en
    No Altitude y en la categoría destino: cambio observado = mediana(destino) −
    mediana(No Altitude); cambio predicho = mediana de (proyección − observado)
    de sus pitches en No Altitude proyectados al destino.
    """
    pitchers = df["pitcher_anon_id"].drop_duplicates().sort_values()
    half_a = set(pitchers.sample(frac=0.5, random_state=BOOT_SEED))
    in_a = df["pitcher_anon_id"].isin(half_a)
    ratios_a = trajectory.estimate_density_ratios(df.loc[in_a])
    b = df.loc[~in_a].copy()
    b["AutoPitchType"] = b["AutoPitchType"].astype("object")

    rows = []
    keys = ["pitcher_anon_id", "year", "AutoPitchType"]
    for cat in ["Medium Altitude", "Extreme Altitude"]:
        no = b.loc[b["altitude_category"] == trajectory.REFERENCE_CATEGORY]
        dest = b.loc[b["altitude_category"] == cat]
        n_no = no.groupby(keys).size()
        n_dest = dest.groupby(keys).size()
        valid = n_no[n_no >= trajectory.MIN_PITCHES_PER_CELL].index.intersection(
            n_dest[n_dest >= trajectory.MIN_PITCHES_PER_CELL].index
        )
        no = no.set_index(keys).loc[valid].reset_index()
        dest = dest.set_index(keys).loc[valid].reset_index()
        proj = trajectory.project_to_altitude(no, cat, ratios_a)
        obs_shift = dest.groupby(keys)[VALIDATION_FEATURES].median() - no.groupby(keys)[VALIDATION_FEATURES].median()
        pred_shift = (proj[VALIDATION_FEATURES] - no[VALIDATION_FEATURES]).assign(**{k: no[k] for k in keys}).groupby(keys).median()
        for feat in VALIDATION_FEATURES:
            o, p = obs_shift[feat], pred_shift.loc[obs_shift.index, feat]
            rows.append({
                "categoria": cat, "feature": feat, "n_grupos": len(o),
                "cambio_observado_mediana": float(o.median()), "cambio_predicho_mediana": float(p.median()),
                "corr_entre_grupos": float(np.corrcoef(o, p)[0, 1]),
            })
    return pd.DataFrame(rows), ratios_a


# ---------------------------------------------------------------------------
# 3-4. Efecto en carreras con Pitching+ v3
# ---------------------------------------------------------------------------


def predict_scenario(df: pd.DataFrame, target: str | None, ratios: pd.DataFrame, artifact: dict):
    """E[Δ] por pitch: observado (target=None) o proyectado a la categoría `target`.

    Devuelve (P(evento), E[Δ] y componentes; probabilidades de los 5 sub-modelos; features del escenario).
    """
    scen = df if target is None else trajectory.recompute_fastball_diffs(trajectory.project_to_altitude(df, target, ratios))
    feats, _ = stuff_plus_v2.build_v2_features(scen, location_coefs=artifact["location_coefs"])
    return run_value_chain.predict_expected_run_value(artifact, feats), attr.submodel_probabilities(artifact, feats), scen


def count_delta_matrix(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    """Δ(evento, conteo real) por pitch, con la tabla del artefacto."""
    deltas = artifact["deltas_by_count"]
    pos = {bs: i for i, bs in enumerate(deltas.index)}
    return deltas.to_numpy()[[pos[(int(b), int(s))] for b, s in zip(df["Balls"], df["Strikes"])]]


def bootstrap_by_pitcher(values: pd.DataFrame, group_col: str, value_cols: list[str]) -> dict:
    """Media por grupo con IC 95% remuestreando pitchers (suma/conteo por pitcher×grupo)."""
    grouped = values.groupby(["pitcher_anon_id", group_col])
    sums = grouped[value_cols].sum()
    cnts = grouped.size()
    groups = sums.index.get_level_values(1).unique()
    pitchers = sums.index.get_level_values(0).unique()
    # Arreglos pitcher × grupo (× valor) para remuestrear pitchers sin pandas en el ciclo
    S = np.stack([sums[c].unstack(fill_value=0.0).reindex(index=pitchers, columns=groups, fill_value=0.0).to_numpy()
                  for c in value_cols], axis=2)
    C = cnts.unstack(fill_value=0).reindex(index=pitchers, columns=groups, fill_value=0).to_numpy(dtype=float)
    point = pd.DataFrame(S.sum(0) / C.sum(0)[:, None], index=groups, columns=value_cols)
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty((N_BOOT, len(groups), len(value_cols)))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(pitchers), size=len(pitchers))
        boots[b] = S[idx].sum(0) / C[idx].sum(0)[:, None]
    lo = pd.DataFrame(np.percentile(boots, 2.5, axis=0), index=groups, columns=value_cols)
    hi = pd.DataFrame(np.percentile(boots, 97.5, axis=0), index=groups, columns=value_cols)
    return {"point": point, "lo": lo, "hi": hi}


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def format_report(ratios, ratios_a, validation, identity, effect, effect_all, comp, quality, points_per_run, n, gap, mech) -> str:
    glob = ratios.loc[ratios["AutoPitchType"] == "__all__"].set_index("altitude_category")
    lines = [
        "# Contrafactual de altitud v2 — trayectoria completa + Pitching+ v3",
        "",
        "Generado por `src/analysis/altitude_counterfactual_v2.py` (física en `src/features/trajectory.py`). "
        f"Reemplaza al contrafactual v1 (`historial/reports/altitude_counterfactual.md`). Pitches: {n:,}.",
        "",
        "## 1. Densidad del aire medida con los datos (no supuesta)",
        "",
        "Drag/v² y Magnus/v² son proporcionales a la densidad del aire. Comparando al **mismo pitcher, mismo tipo de pitch "
        "y misma temporada** entre categorías (≥15 pitches por celda), la mediana de los ratios estima la densidad efectiva "
        "de cada categoría relativa a No Altitude.",
        "",
        "| categoría | ratio drag | ratio Magnus | grupos | elevación equivalente (drag) | supuesto v1 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    assumed = {"Medium Altitude": 0.930, "Extreme Altitude": 0.834}
    for cat in ["Medium Altitude", "Extreme Altitude"]:
        r = glob.loc[cat]
        elev = 8434.5 * -np.log(r["drag_ratio"]) + 50
        lines.append(
            f"| {cat} | {r['drag_ratio']:.3f} | {r['magnus_ratio']:.3f} | {int(r['n_groups']):,} | ≈{elev:,.0f} m | {assumed[cat]:.3f} |"
        )
    lines += [
        "",
        "Por tipo de pitch:",
        "",
        "| categoría | tipo | ratio drag | ratio Magnus | grupos |",
        "|---|---|---:|---:|---:|",
    ]
    for r in ratios.loc[ratios["AutoPitchType"] != "__all__"].itertuples():
        lines.append(f"| {r.altitude_category} | {r.AutoPitchType} | {r.drag_ratio:.3f} | {r.magnus_ratio:.3f} | {int(r.n_groups):,} |")
    lines += [
        "",
        "**Lectura.** Si solo cambiara la densidad del aire, drag y Magnus bajarían en la misma proporción. "
        "El Magnus baja bastante más, sobre todo en Medium Altitude. Eso apunta a algo más que densidad: "
        "calibración de Trackman por estadio, clima o la pelota. Es consistente con el residual vertical anómalo de "
        "Medium Altitude que v1 dejó sin explicar (`reports/magnus_analysis.md`). "
        "Para el contrafactual se usan los ratios medidos de cada componente, que reproducen lo que el modelo realmente ve.",
        "",
        "## 2. Validación split-half de la re-simulación",
        "",
        "Ratios estimados con la mitad de los pitchers. Con la **otra mitad**, se proyectan sus pitches de No Altitude a la "
        "categoría destino y se compara el cambio predicho contra el cambio observado en esos mismos pitchers.",
        "",
        "| categoría | feature | grupos | cambio observado | cambio predicho | corr. entre grupos |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in validation.itertuples():
        lines.append(
            f"| {r.categoria} | {r.feature} | {r.n_grupos} | {r.cambio_observado_mediana:+.3f} | "
            f"{r.cambio_predicho_mediana:+.3f} | {r.corr_entre_grupos:.3f} |"
        )
    lines += [
        "",
        "El contrafactual v1 predecía cambio **cero** en ZoneSpeed, VAA, ZoneTime y VertBreak, porque no los recalculaba.",
        "",
        f"**Identidad:** proyectar cada pitch a su propia categoría cambia las features como máximo en {identity:.2e} "
        "(error de punto flotante).",
        "",
        "## 3. Brecha observada entre categorías vs. lo que atribuye el modelo",
        "",
        "Carreras por 100 pitches (positivo = más carreras permitidas). *Realizado* = valor en carreras del evento que "
        "ocurrió (misma escala Δ por conteo de Pitching+ v3). *Predicho* = E[Δ] de Pitching+ v3 con las features observadas.",
        "",
        "| categoría | realizado | predicho (obs.) | proyectado a No Altitude | proyectado a Extreme |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, r in gap.iterrows():
        lines.append(f"| {cat} | {r['realized']:+.3f} | {r['obs']:+.3f} | {r['no']:+.3f} | {r['ext']:+.3f} |")
    g_real = gap.loc["Extreme Altitude", "realized"] - gap.loc["No Altitude", "realized"]
    g_pred = gap.loc["Extreme Altitude", "obs"] - gap.loc["No Altitude", "obs"]
    lines += [
        "",
        f"Brecha Extreme − No: realizada **{g_real:+.2f}**, predicha por el modelo **{g_pred:+.2f}** carreras/100 pitches. "
        "Pitching+ v3 no tiene ninguna variable de altitud, pero **la detecta por la firma física de la trayectoria** "
        "(drag y Magnus más bajos) y le atribuye casi toda la diferencia entre parques, incluido el entorno ofensivo "
        "(vuelo de la pelota, dimensiones). Proyectar un pitch a otra categoría mueve esa firma, así que el contrafactual "
        "captura el **efecto total** del entorno de altitud, no solo la pérdida de calidad del pitch. La sección 5 los separa.",
        "",
        "## 4. Efecto total de la altitud en carreras (Pitching+ v3, Extreme − No Altitude)",
        "",
        "Cada pitch se proyecta a ambas categorías (mismo pitcher, misma salida, mismo objetivo en el plato) y se predice "
        "E[Δ carreras] con los modelos finales de Pitching+ v3. Positivo = más carreras permitidas. IC 95% por bootstrap "
        f"de pitchers ({N_BOOT:,} réplicas).",
        "",
        "| tipo de pitch | carreras/100 pitches [IC 95%] | Δ Pitching+ v3 | abridor/temporada | relevista/temporada | v1: Δ Pitching+ v1 | v1: abridor |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    e = effect
    order = e["point"]["delta"].sort_values(ascending=False).index
    for t in order:
        p, lo, hi = e["point"].loc[t, "delta"] * 100, e["lo"].loc[t, "delta"] * 100, e["hi"].loc[t, "delta"] * 100
        starter = p * PITCHES_PER_SEASON["abridor (2,750 pitches)"] / 100
        reliever = p * PITCHES_PER_SEASON["relevista (900 pitches)"] / 100
        pts = -p / 100 * points_per_run
        old_pts = OLD_V1_DELTA_POINTS.get(t)
        old_runs = OLD_V1_RUNS_STARTER.get(t)
        lines.append(
            f"| {t} | {p:+.3f} [{lo:+.3f}, {hi:+.3f}] | {pts:+.2f} | {starter:+.2f} | {reliever:+.2f} | "
            f"{old_pts if old_pts is not None else '—'} | {('+' + format(old_runs, '.2f')) if old_runs else '—'} |"
        )
    pa, la, ha = (effect_all[k] * 100 for k in ("point", "lo", "hi"))
    lines += [
        f"| **todos** | {pa:+.3f} [{la:+.3f}, {ha:+.3f}] | {-pa / 100 * points_per_run:+.2f} | "
        f"{pa * 27.5:+.2f} | {pa * 9:+.2f} | | |",
        "",
        f"Conversión a puntos de Pitching+ v3: 10 puntos = 1 desviación estándar del E[Δ] promedio por pitcher-temporada "
        f"→ {points_per_run:,.0f} puntos por carrera/pitch. A diferencia de v1, la conversión a carreras por temporada es "
        "directa (el score ya está en carreras), no una regresión circular.",
        "",
        "## 5. ¿Por dónde pasa el efecto? Atribución por eslabón de la cadena",
        "",
        "El cambio en E[Δ] se reparte entre los 5 eslabones de la cadena con valores de Shapley "
        "(`src/models/chain_attribution.py`): promedia, sobre todos los órdenes posibles, cuánto cambia E[Δ] al pasar cada "
        "eslabón de No Altitude a Extreme. Los componentes suman exactamente el efecto total. Es una **atribución contable "
        "entre dos predicciones del modelo, no una descomposición causal**.",
        "",
        "- **Antes del contacto** = swing + take + whiff + foul: lo que el pitch provoca hasta que la pelota queda en juego, "
        "incluido el aumento de la probabilidad de contacto. Es la parte atribuible al pitch con más confianza.",
        "- **Bola en juego** = cambio en el resultado del contacto (out/1B/2B/3B/HR). Mezcla calidad del contacto con el "
        "entorno del parque (la pelota vuela más en altitud); con este modelo no se pueden separar.",
        "",
        "La versión anterior de esta sección llamaba 'calidad del pitch' a take + whiff + foul por **grupo de eventos**. Eso "
        "omitía que menos whiffs implican más pelotas en juego: ese efecto, causado por el pitch, quedaba contado como 'bola "
        "en juego' (auditoría 2026-10-05, C3).",
        "",
        "| tipo de pitch | antes del contacto, carreras/100 [IC 95%] | abridor/temporada | bola en juego, carreras/100 | abridor/temporada |",
        "|---|---|---:|---:|---:|",
    ]
    for t in order:
        q, ql, qh = (quality[k].loc[t, "antes_del_contacto"] * 100 for k in ("point", "lo", "hi"))
        b = comp["point"].loc[t, "sh_bip"] * 100
        lines.append(f"| {t} | {q:+.3f} [{ql:+.3f}, {qh:+.3f}] | {q * 27.5:+.2f} | {b:+.3f} | {b * 27.5:+.2f} |")
    lines += [
        "",
        "Detalle por eslabón (carreras/100 pitches, Shapley):",
        "",
        "| tipo de pitch | " + " | ".join(attr.LINK_LABELS[k] for k in attr.LINKS) + " | total |",
        "|---|" + "---:|" * (len(attr.LINKS) + 1),
    ]
    for t in order:
        row = [comp["point"].loc[t, f"sh_{k}"] * 100 for k in attr.LINKS]
        lines.append(f"| {t} | " + " | ".join(f"{v:+.3f}" for v in row) + f" | {sum(row):+.3f} |")
    lines += [
        "",
        "Mecanismo (Extreme − No Altitude, promedio por pitch): cambio físico del pitch y cambio en las probabilidades de la "
        "cadena. Los cambios físicos son los del contrafactual (mismo pitch, mismo objetivo).",
        "",
        "| tipo de pitch | IVB (pulg.) | HB (pulg.) | ZoneSpeed (mph) | VAA (°) | P(swing) (pp) | P(whiff dado swing) (pp) | P(en juego) (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in order:
        m = mech.loc[t]
        lines.append(f"| {t} | {m['dfeat_InducedVertBreak']:+.2f} | {m['dfeat_HorzBreak']:+.2f} | {m['dfeat_ZoneSpeed']:+.2f} | "
                     f"{m['dfeat_VertApprAngle']:+.3f} | {100 * m['dp_swing']:+.2f} | {100 * m['dp_whiff_swing']:+.2f} | {100 * m['dp_en_juego']:+.2f} |")
    lines += [
        "",
        "Lo que **no** se puede separar con estos modelos: dentro de 'bola en juego', la parte que se debe a que el pitch "
        "provoca peor contacto de la que se debe a que la pelota vuela más en el parque.",
    ]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> dict:
    df = load_data()
    logger.info("Pitches: %s", f"{len(df):,}")

    ratios = trajectory.estimate_density_ratios(df)
    ratios.to_parquet(RATIOS_PATH, index=False)
    logger.info("Ratios globales:\n%s", ratios.loc[ratios["AutoPitchType"] == "__all__"].to_string(index=False))

    validation, ratios_a = split_half_validation(df)
    logger.info("Validación split-half:\n%s", validation.to_string(index=False))

    identity = 0.0
    for cat in trajectory.ALTITUDE_CATEGORIES:
        sub = df.loc[df["altitude_category"] == cat]
        same = trajectory.project_to_altitude(sub, cat, ratios)
        identity = max(identity, float((same[trajectory.UPDATED_FEATURES] - sub[trajectory.UPDATED_FEATURES]).abs().max().max()))
    logger.info("Identidad: max |Δ| = %.2e", identity)

    with open(run_value_chain.FINAL_MODEL_PATHS["pitching"], "rb") as fh:
        artifact = pickle.load(fh)
    preds, subs, scens = {}, {}, {}
    for name, target in [("obs", None), ("no", "No Altitude"), ("ext", "Extreme Altitude")]:
        preds[name], subs[name], scens[name] = predict_scenario(df, target, ratios, artifact)
        logger.info("Escenario %s: E[Δ] medio = %+.5f", name, preds[name]["xrv"].mean())

    out = df[["PitchUID", "pitcher_anon_id", "year", "AutoPitchType", "altitude_category"]].copy()
    out["AutoPitchType"] = out["AutoPitchType"].astype("object")
    out["xrv_obs"] = preds["obs"]["xrv"].to_numpy()
    out["xrv_no_altitude"] = preds["no"]["xrv"].to_numpy()
    out["xrv_extreme"] = preds["ext"]["xrv"].to_numpy()
    out["delta"] = out["xrv_extreme"] - out["xrv_no_altitude"]
    sh = attr.shapley(subs["no"], subs["ext"], count_delta_matrix(artifact, df))
    for k in attr.LINKS:
        out[f"sh_{k}"] = sh[k].to_numpy()
    logger.info("Shapley: max |Σφ − δ| = %.2e", float(np.abs(sh.sum(axis=1).to_numpy() - out["delta"].to_numpy()).max()))
    # Mecanismo: cambio físico del pitch y cambio en las probabilidades de la cadena (Extreme − No Altitude)
    for f in MECHANISM_FEATURES:
        out[f"dfeat_{f}"] = scens["ext"][f].to_numpy(float) - scens["no"][f].to_numpy(float)
    bip_e = preds["ext"][run_value_chain.BIP_CLASSES].sum(axis=1).to_numpy()
    bip_n = preds["no"][run_value_chain.BIP_CLASSES].sum(axis=1).to_numpy()
    out["dp_swing"] = subs["ext"]["swing"][:, 0] - subs["no"]["swing"][:, 0]
    out["dp_whiff_swing"] = subs["ext"]["whiff"][:, 0] - subs["no"]["whiff"][:, 0]
    out["dp_en_juego"] = bip_e - bip_n
    out.to_parquet(PITCHES_OUT, index=False)

    main_types = out["AutoPitchType"].isin(OLD_V1_DELTA_POINTS)
    effect = bootstrap_by_pitcher(out.loc[main_types], "AutoPitchType", ["delta"])
    comp = bootstrap_by_pitcher(out.loc[main_types], "AutoPitchType", SHAPLEY_COLUMNS)
    out["antes_del_contacto"] = out[PRE_CONTACT_COLUMNS].sum(axis=1)
    quality = bootstrap_by_pitcher(out.loc[main_types], "AutoPitchType", ["antes_del_contacto"])
    mech_cols = [f"dfeat_{f}" for f in MECHANISM_FEATURES] + ["dp_swing", "dp_whiff_swing", "dp_en_juego"]
    mech = out.loc[main_types].groupby("AutoPitchType")[mech_cols].mean()

    realized = pd.read_parquet(CHAIN_OOF_PATH, columns=["PitchUID", "realized_delta"])
    gap = (
        out.merge(realized, on="PitchUID", how="inner")
        .groupby("altitude_category")[["realized_delta", "xrv_obs", "xrv_no_altitude", "xrv_extreme"]].mean()
        .mul(100)
        .rename(columns={"realized_delta": "realized", "xrv_obs": "obs", "xrv_no_altitude": "no", "xrv_extreme": "ext"})
        .reindex(trajectory.ALTITUDE_CATEGORIES)
    )
    all_eff = bootstrap_by_pitcher(out.assign(all="todos"), "all", ["delta"])
    effect_all = {k: float(all_eff[k].loc["todos", "delta"]) for k in ("point", "lo", "hi")}

    season = out.groupby(["pitcher_anon_id", "year"]).agg(n=("xrv_obs", "size"), xrv=("xrv_obs", "mean"))
    points_per_run = 10 / float(season.loc[season["n"] >= 100, "xrv"].std())

    REPORT_PATH.write_text(
        format_report(ratios, ratios_a, validation, identity, effect, effect_all, comp, quality, points_per_run, len(df), gap, mech),
        encoding="utf-8",
    )
    payload = {
        "ratios": ratios.to_dict("records"),
        "validation": validation.to_dict("records"),
        "identity_max_abs": identity,
        "effect_runs_per_pitch": {t: {k: float(effect[k].loc[t, "delta"]) for k in ("point", "lo", "hi")} for t in effect["point"].index},
        "effect_all": effect_all,
        "antes_del_contacto_runs_per_pitch": {t: {k: float(quality[k].loc[t, "antes_del_contacto"]) for k in ("point", "lo", "hi")} for t in quality["point"].index},
        "shapley_runs_per_pitch": {t: {c: float(comp["point"].loc[t, c]) for c in SHAPLEY_COLUMNS} for t in comp["point"].index},
        "observed_gap_runs_per_100": gap.to_dict("index"),
        "points_per_run_per_pitch": points_per_run,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)
    return payload


if __name__ == "__main__":
    run()
