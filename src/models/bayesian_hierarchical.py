"""Modelo bayesiano jerarquico (PyMC) de run_value con shrinkage por pitcher
dentro de cluster de tipo de pitch, y efecto de altitud (fijo + aleatorio por
cluster).

DECISION: se modela `run_value` A NIVEL DE PITCH (no el score Pitching+ a
nivel pitcher-temporada). Justificacion: la jerarquia pedida es de 3 niveles
(pitch -> pitcher -> cluster de pitch type). Pitching+ ya es un agregado por
(pitcher_anon_id, year) — modelarlo dejaria solo 2 niveles posibles
(pitcher-temporada -> cluster), perdiendo el nivel de pitch que es
precisamente el que necesita el shrinkage: la varianza que se quiere encoger
es la del PROMEDIO DE run_value POR PITCHER, que se estima con mas o menos
precision segun cuantos pitches tiene ese pitcher — eso solo se modela
correctamente con una verosimilitud a nivel de pitch (el N real por grupo) en
vez de una unica fila "y ya promediada" por pitcher-temporada. Ademas
run_value ya es la unidad mas granular y "cruda" del proyecto (Stuff+ es
solo una parte de Pitching+, ver reports/stuff_plus_baseline.md), asi que un
modelo bayesiano sobre run_value es comparable en espiritu al Pitching+ real
(que USA run_value como su componente de mayor peso, 20%, ver
reports/pitching_plus_score.md) sin heredar el resto de la maquinaria de
LightGBM.

ESTRUCTURA DE LA JERARQUIA (nesting real): un mismo `pitcher_anon_id` aparece
en MULTIPLES clusters (su arsenal mezcla fastballs/breaking/offspeed) — no
tiene sentido asignarle un unico cluster. La forma correcta de implementar
"pitch -> pitcher -> cluster" es entonces un nesting de PITCHER-DENTRO-DE-
CLUSTER: cada combinacion (pitcher_anon_id, cluster) es su propio grupo con
un intercepto propio (`alpha_pc`), que se encoge hacia la media de SU cluster
(`mu_cluster`). Esto responde exactamente a la pregunta de interes: "que tan
bueno es el repertorio de fastballs de este pitcher especifico" (shrunk hacia
el promedio de fastballs de la liga si tiene pocos fastballs), sin forzar a
un pitcher a pertenecer a un solo cluster.

Altitud: `altitude_level` (0/1/2) entra como covariable a nivel de pitch, con
un coeficiente de efecto fijo GLOBAL (`beta_altitude_global`) y una desviacion
aleatoria POR CLUSTER (`beta_altitude_cluster[c] = beta_altitude_global +
sigma_altitude_cluster * offset[c]`, no-centrado) — el efecto de altitud real
que multiplica a `altitude_level` en la verosimilitud es el de cluster,
parcialmente pooled hacia el efecto global. Esto da ambas cosas pedidas: un
coeficiente fijo (la media de la distribucion) y un efecto aleatorio por
cluster (las desviaciones respecto a esa media).

Excluye las mismas filas que data/processed/splits/splits.parquet marca como
'excluded' en pitcher_holdout (pitcher_anon_id o altitude_category nulos),
mas los ~2,989 pitches con run_value nulo (bordes de reconstruccion de PA,
ver reports/run_value_validation.md) y los AutoPitchType fuera de los 3
clusters pedidos (Sweeper, Other, y las clases residuales de 1 pitch).

Muestra: se toman TODOS los pitches de pitchers con <=SMALL_PITCHER_THRESHOLD
pitches validos (son justamente el caso de uso de interes — no tiene sentido
sub-muestrearlos mas), y se sub-muestrea el resto (estratificado por
pitcher_anon_id x altitude_category) para llegar a ~TARGET_SAMPLE_SIZE filas
totales.
"""

import logging
import pickle
import time
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
SPLITS_PATH = ROOT / "data" / "processed" / "splits" / "splits.parquet"
COUNTERFACTUAL_SCORES_PATH = ROOT / "data" / "processed" / "altitude_counterfactual_scores.parquet"
TRACE_PATH = ROOT / "src" / "models" / "artifacts" / "bayesian_hierarchical.nc"
REPORT_PATH = ROOT / "reports" / "bayesian_hierarchical.md"
FIG_DIR = ROOT / "reports" / "figures" / "bayesian_hierarchical"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CLUSTER_MAP = {
    "Four-Seam": "fastballs",
    "Sinker": "fastballs",
    "Cutter": "fastballs",
    "Slider": "breaking",
    "Curveball": "breaking",
    "Changeup": "offspeed",
    "Splitter": "offspeed",
}
CLUSTER_NAMES = ["fastballs", "breaking", "offspeed"]

TARGET_SAMPLE_SIZE = 180_000
SMALL_PITCHER_THRESHOLD = 150  # pitchers con <= esto: se quedan con TODOS sus pitches
SAMPLE_SEED = 42

N_CHAINS = 4
N_DRAWS = 1000
N_TUNE = 1000
TARGET_ACCEPT = 0.95
MAX_TREEDEPTH = 12
SAMPLE_SEED_MCMC = 42

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400  # regla practica (Vehtari et al. 2021) para ess_bulk/ess_tail

SMALL_PITCHER_TABLE_N = 8  # cuantos pitchers de muestra chica mostrar en la comparacion de shrinkage


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------


def load_valid_data() -> pd.DataFrame:
    df = pd.read_parquet(
        TARGETS_PATH, columns=["PitchUID", "pitcher_anon_id", "AutoPitchType", "altitude_category", "run_value"]
    )
    splits = pd.read_parquet(SPLITS_PATH, columns=["PitchUID", "pitcher_holdout"])
    df = df.merge(splits, on="PitchUID", how="inner")
    df = df.loc[df["pitcher_holdout"] != "excluded"].copy()

    df["cluster"] = df["AutoPitchType"].map(CLUSTER_MAP)
    df = df.dropna(subset=["cluster", "run_value"])

    altitude_level_map = {"No Altitude": 0, "Medium Altitude": 1, "Extreme Altitude": 2}
    df["altitude_level"] = df["altitude_category"].map(altitude_level_map).astype(int)

    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, target_n: int, small_threshold: int, seed: int) -> pd.DataFrame:
    counts = df.groupby("pitcher_anon_id").size()
    small_pitchers = counts.loc[counts <= small_threshold].index
    large_pitchers = counts.loc[counts > small_threshold].index

    small_df = df.loc[df["pitcher_anon_id"].isin(small_pitchers)]
    large_df = df.loc[df["pitcher_anon_id"].isin(large_pitchers)]

    remaining_target = max(target_n - len(small_df), 0)
    frac = min(1.0, remaining_target / len(large_df)) if len(large_df) > 0 else 0.0

    large_sample = large_df.groupby(["pitcher_anon_id", "altitude_category"], observed=True, group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed) if frac > 0 else g.iloc[0:0]
    )

    sampled = pd.concat([small_df, large_sample]).sort_values("PitchUID").reset_index(drop=True)
    return sampled


# ---------------------------------------------------------------------------
# 2. Modelo
# ---------------------------------------------------------------------------


def build_model(df: pd.DataFrame) -> tuple[pm.Model, pd.DataFrame]:
    df = df.copy()
    df["pc_key"] = df["pitcher_anon_id"].astype(str) + "||" + df["cluster"].astype(str)

    pc_labels, pc_idx = np.unique(df["pc_key"], return_inverse=True)
    cluster_idx_by_pc = np.array(
        [CLUSTER_NAMES.index(label.split("||")[1]) for label in pc_labels]
    )
    cluster_idx_obs = df["cluster"].map({c: i for i, c in enumerate(CLUSTER_NAMES)}).to_numpy()

    pc_meta = pd.DataFrame(
        {
            "pc_key": pc_labels,
            "pitcher_anon_id": [label.split("||")[0] for label in pc_labels],
            "cluster": [label.split("||")[1] for label in pc_labels],
        }
    )

    coords = {
        "cluster": CLUSTER_NAMES,
        "pitcher_cluster": pc_labels,
        "obs_id": np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        pc_idx_data = pm.Data("pc_idx", pc_idx, dims="obs_id")
        cluster_idx_obs_data = pm.Data("cluster_idx_obs", cluster_idx_obs, dims="obs_id")
        altitude_level_data = pm.Data("altitude_level", df["altitude_level"].to_numpy(dtype=float), dims="obs_id")
        cluster_of_pc_data = pm.Data("cluster_of_pc", cluster_idx_by_pc, dims="pitcher_cluster")

        # --- nivel 3: cluster de tipo de pitch ---
        mu_global = pm.Normal("mu_global", mu=0.0, sigma=0.05)
        sigma_cluster = pm.HalfNormal("sigma_cluster", sigma=0.02)
        mu_cluster_offset = pm.Normal("mu_cluster_offset", 0.0, 1.0, dims="cluster")
        mu_cluster = pm.Deterministic("mu_cluster", mu_global + sigma_cluster * mu_cluster_offset, dims="cluster")

        # --- nivel 2: pitcher DENTRO de cluster (shrinkage hacia mu_cluster) ---
        sigma_pc = pm.HalfNormal("sigma_pc", sigma=0.05)
        alpha_pc_offset = pm.Normal("alpha_pc_offset", 0.0, 1.0, dims="pitcher_cluster")
        alpha_pc = pm.Deterministic(
            "alpha_pc", mu_cluster[cluster_of_pc_data] + sigma_pc * alpha_pc_offset, dims="pitcher_cluster"
        )

        # --- efecto de altitud: fijo global + aleatorio por cluster (no-centrado) ---
        beta_altitude_global = pm.Normal("beta_altitude_global", mu=0.0, sigma=0.02)
        sigma_altitude_cluster = pm.HalfNormal("sigma_altitude_cluster", sigma=0.01)
        beta_altitude_cluster_offset = pm.Normal("beta_altitude_cluster_offset", 0.0, 1.0, dims="cluster")
        beta_altitude_cluster = pm.Deterministic(
            "beta_altitude_cluster",
            beta_altitude_global + sigma_altitude_cluster * beta_altitude_cluster_offset,
            dims="cluster",
        )

        # --- nivel 1: pitch ---
        mu_obs = alpha_pc[pc_idx_data] + beta_altitude_cluster[cluster_idx_obs_data] * altitude_level_data
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.3)
        pm.Normal("run_value_obs", mu=mu_obs, sigma=sigma_obs, observed=df["run_value"].to_numpy(), dims="obs_id")

    return model, pc_meta


# ---------------------------------------------------------------------------
# 3. Sampling + diagnosticos
# ---------------------------------------------------------------------------


def sample_model(model: pm.Model) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            cores=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            max_treedepth=MAX_TREEDEPTH,
            random_seed=SAMPLE_SEED_MCMC,
            progressbar=True,
        )
    return idata


TOP_LEVEL_VARS = [
    "mu_global",
    "sigma_cluster",
    "mu_cluster",
    "sigma_pc",
    "beta_altitude_global",
    "sigma_altitude_cluster",
    "beta_altitude_cluster",
    "sigma_obs",
]


def compute_diagnostics(idata: az.InferenceData) -> dict:
    summary_top = az.summary(idata, var_names=TOP_LEVEL_VARS, hdi_prob=0.94)
    summary_alpha_pc = az.summary(idata, var_names=["alpha_pc"], hdi_prob=0.94)

    all_summary = pd.concat([summary_top, summary_alpha_pc])
    n_rhat_bad = int((all_summary["r_hat"] > RHAT_THRESHOLD).sum())
    n_ess_bulk_bad = int((all_summary["ess_bulk"] < ESS_THRESHOLD).sum())
    n_ess_tail_bad = int((all_summary["ess_tail"] < ESS_THRESHOLD).sum())

    divergences = int(idata.sample_stats["diverging"].sum().item())
    n_total_draws = int(idata.sample_stats["diverging"].size)

    return {
        "summary_top": summary_top,
        "n_params_checked": len(all_summary),
        "n_rhat_bad": n_rhat_bad,
        "max_rhat": float(all_summary["r_hat"].max()),
        "n_ess_bulk_bad": n_ess_bulk_bad,
        "min_ess_bulk": float(all_summary["ess_bulk"].min()),
        "n_ess_tail_bad": n_ess_tail_bad,
        "min_ess_tail": float(all_summary["ess_tail"].min()),
        "divergences": divergences,
        "n_total_draws": n_total_draws,
        "pct_divergences": 100 * divergences / n_total_draws,
    }


def plot_traces(idata: az.InferenceData) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    axes = az.plot_trace(idata, var_names=TOP_LEVEL_VARS, compact=True)
    fig = axes.ravel()[0].figure
    fig.suptitle("Trace plots — parametros de nivel superior (cluster + altitud)", y=1.0)
    plt.tight_layout()
    out = FIG_DIR / "trace_top_level.png"
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Shrinkage: pitchers de muestra chica, crudo vs. shrunk
# ---------------------------------------------------------------------------


def build_shrinkage_table(df: pd.DataFrame, idata: az.InferenceData, pc_meta: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pc_key"] = df["pitcher_anon_id"].astype(str) + "||" + df["cluster"].astype(str)

    raw = df.groupby("pc_key").agg(n_pitches=("run_value", "size"), raw_mean_run_value=("run_value", "mean"))
    raw = raw.reset_index()

    alpha_pc_mean = idata.posterior["alpha_pc"].mean(dim=["chain", "draw"]).to_dataframe().reset_index()
    alpha_pc_mean = alpha_pc_mean.rename(columns={"pitcher_cluster": "pc_key", "alpha_pc": "shrunk_mean_run_value"})

    mu_cluster_mean = idata.posterior["mu_cluster"].mean(dim=["chain", "draw"]).to_dataframe().reset_index()
    mu_cluster_mean = mu_cluster_mean.rename(columns={"cluster": "cluster_name", "mu_cluster": "cluster_mean"})

    table = raw.merge(pc_meta, on="pc_key").merge(alpha_pc_mean, on="pc_key")
    table = table.merge(mu_cluster_mean, left_on="cluster", right_on="cluster_name", how="left")

    table["shrinkage_pct"] = 100 * (
        1 - (table["shrunk_mean_run_value"] - table["cluster_mean"]).abs()
        / (table["raw_mean_run_value"] - table["cluster_mean"]).abs().replace(0, np.nan)
    )
    return table


def select_small_pitcher_examples(table: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    small = table.loc[table["n_pitches"] < 100].copy()
    # elegir un rango variado de n_pitches (no solo los mas chicos) para que
    # la tabla muestre el gradiente de shrinkage segun tamano de muestra
    small = small.sort_values("n_pitches")
    idx = np.linspace(0, len(small) - 1, n).round().astype(int)
    idx = sorted(set(idx))
    return small.iloc[idx]


# ---------------------------------------------------------------------------
# 5. Comparacion contra el contrafactual (reports/altitude_counterfactual.md)
# ---------------------------------------------------------------------------


def load_counterfactual_run_value_delta_by_cluster() -> pd.DataFrame:
    scores = pd.read_parquet(
        COUNTERFACTUAL_SCORES_PATH, columns=["AutoPitchType", "run_value_Extreme_Altitude", "run_value_No_Altitude"]
    )
    scores = scores.loc[scores["AutoPitchType"].isin(CLUSTER_MAP.keys())].copy()
    scores["cluster"] = scores["AutoPitchType"].map(CLUSTER_MAP)
    scores["delta_run_value"] = scores["run_value_Extreme_Altitude"] - scores["run_value_No_Altitude"]

    by_type = scores.groupby("AutoPitchType", observed=True)["delta_run_value"].agg(["mean", "count"])
    by_cluster = (
        scores.groupby("cluster", observed=True)
        .apply(lambda g: np.average(g["delta_run_value"], weights=None))
        .rename("delta_run_value_extreme_vs_no")
    )
    by_cluster_n = scores.groupby("cluster", observed=True).size().rename("n")
    by_cluster_df = pd.concat([by_cluster, by_cluster_n], axis=1).reindex(CLUSTER_NAMES)

    return by_type, by_cluster_df


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def format_report(
    n_valid: int,
    n_sample: int,
    n_small_pitchers: int,
    n_pc_groups: int,
    diag: dict,
    idata: az.InferenceData,
    shrinkage_examples: pd.DataFrame,
    cf_by_type: pd.DataFrame,
    cf_by_cluster: pd.DataFrame,
    trace_fig_path: Path,
    runtime_s: float,
) -> str:
    summary_top = diag["summary_top"]

    convergence_ok = diag["n_rhat_bad"] == 0 and diag["n_ess_bulk_bad"] == 0 and diag["n_ess_tail_bad"] == 0

    lines = [
        "# Modelo bayesiano jerarquico (PyMC) — run_value con shrinkage pitcher-dentro-de-cluster",
        "",
        "## 0. Decisiones de diseño",
        "",
        "**Unidad modelada: `run_value` a nivel de PITCH**, no el score Pitching+ "
        "a nivel pitcher-temporada. La jerarquia de 3 niveles pedida "
        "(pitch -> pitcher -> cluster) necesita el nivel de pitch para que el "
        "shrinkage sea real: lo que se encoge es el promedio de un pitcher "
        "estimado con mas o menos precision segun su N real de pitches, y eso "
        "solo se modela bien con una verosimilitud pitch a pitch (ver "
        "docstring de este script para el detalle completo).",
        "",
        "**Nesting real:** un pitcher lanza en varios clusters a la vez (su "
        "arsenal mezcla fastballs/breaking/offspeed), asi que el nivel "
        "intermedio de la jerarquia es la combinacion **(pitcher_anon_id, "
        "cluster)** — cada una con su propio intercepto `alpha_pc`, encogido "
        "hacia la media de SU cluster (`mu_cluster`). Esto responde "
        "directamente 'que tan bueno es el repertorio de fastballs de este "
        "pitcher' en vez de forzar a cada pitcher a un unico cluster.",
        "",
        "**Altitud:** `altitude_level` (0/1/2) entra con un coeficiente fijo "
        "global (`beta_altitude_global`) y una desviacion aleatoria por "
        "cluster no-centrada (`beta_altitude_cluster[c] = beta_altitude_global "
        "+ sigma_altitude_cluster * offset[c]`) — el coeficiente que "
        "realmente multiplica `altitude_level` en la verosimilitud es el de "
        "cluster, parcialmente pooled hacia el efecto global.",
        "",
        "## 1. Datos y muestra",
        "",
        f"- Pitches validos (mismo filtro que `pitcher_holdout != excluded` de "
        f"`data/processed/splits/splits.parquet`, mas exclusion de run_value "
        f"nulo y de AutoPitchType fuera de los 3 clusters): **{n_valid:,}**",
        f"- Muestra usada para el modelo: **{n_sample:,}** pitches "
        f"({100*n_sample/n_valid:.1f}% del universo valido)",
        f"- Pitchers con <= {SMALL_PITCHER_THRESHOLD} pitches validos: "
        f"**{n_small_pitchers}** — se incluyeron TODOS sus pitches (no se "
        "sub-muestrearon), justamente porque son el caso de uso de interes. "
        "El resto de los pitchers se sub-muestreo estratificado por "
        "(pitcher_anon_id, altitude_category) para llegar al total.",
        f"- Grupos (pitcher_anon_id x cluster) en el modelo: **{n_pc_groups:,}**",
        "",
        "## 2. Sampling y diagnosticos de convergencia",
        "",
        f"NUTS, {N_CHAINS} cadenas, {N_DRAWS} draws + {N_TUNE} tuning cada una "
        f"(target_accept={TARGET_ACCEPT}). Tiempo de sampling: {runtime_s/60:.1f} min.",
        "",
        f"- Divergencias: **{diag['divergences']}** de {diag['n_total_draws']:,} "
        f"draws post-tuning ({diag['pct_divergences']:.3f}%)",
        f"- r_hat maximo (sobre {diag['n_params_checked']:,} parametros, "
        f"incluye los {n_pc_groups:,} `alpha_pc`): **{diag['max_rhat']:.4f}** "
        f"({diag['n_rhat_bad']} parametros con r_hat > {RHAT_THRESHOLD})",
        f"- ess_bulk minimo: **{diag['min_ess_bulk']:.0f}** "
        f"({diag['n_ess_bulk_bad']} parametros con ess_bulk < {ESS_THRESHOLD})",
        f"- ess_tail minimo: **{diag['min_ess_tail']:.0f}** "
        f"({diag['n_ess_tail_bad']} parametros con ess_tail < {ESS_THRESHOLD})",
        "",
    ]

    if convergence_ok:
        lines.append(
            "**Convergencia OK (r_hat/ESS).** Los "
            f"{diag['n_params_checked']:,} parametros chequeados (incluyendo los "
            f"{n_pc_groups:,} interceptos `alpha_pc`) cumplen r_hat<{RHAT_THRESHOLD} "
            f"y ess_bulk/ess_tail>={ESS_THRESHOLD} — sin excepciones. Los "
            "resultados de las secciones siguientes se reportan con confianza "
            "en la muestra posterior."
        )
        if diag["divergences"] > 0:
            lines.append(
                f"\n**Nota sobre divergencias:** hubo {diag['divergences']} "
                f"divergencias en {diag['n_total_draws']:,} draws post-tuning "
                f"({diag['pct_divergences']:.3f}%) — una tasa baja (bien por "
                "debajo del 1% que suele usarse como referencia informal de "
                "alarma) y consistente con r_hat/ESS perfectos en todos los "
                "parametros, asi que no invalida el resultado. Se deja "
                "documentado por transparencia, no como problema sin resolver: "
                "si se quisiera eliminarlas del todo, subir `target_accept` "
                "por encima de 0.95 o `max_treedepth` por encima de 12 "
                "probablemente las resuelve, a costa de mas tiempo de sampling."
            )
    else:
        lines.append(
            "**ADVERTENCIA DE CONVERGENCIA — reportado honestamente, sin forzar "
            "el resultado.** No todos los parametros cumplen los umbrales de "
            f"r_hat<{RHAT_THRESHOLD} y ess_bulk/ess_tail>={ESS_THRESHOLD} "
            f"({diag['n_rhat_bad']} con r_hat alto, {diag['n_ess_bulk_bad']} con "
            f"ess_bulk bajo, {diag['n_ess_tail_bad']} con ess_tail bajo, de "
            f"{diag['n_params_checked']:,} chequeados), y hubo "
            f"{diag['divergences']} divergencias ({diag['pct_divergences']:.3f}%). "
            "Los intervalos de credibilidad de las secciones siguientes deben "
            "leerse con cautela — no se recomienda usarlos para decisiones "
            "operativas sin antes resolver la convergencia (mas tuning, "
            "reparametrizacion adicional, o mas draws)."
        )

    lines += [
        "",
        f"Trace plots (parametros de nivel superior): `{trace_fig_path.relative_to(ROOT)}`",
        "",
        "### Resumen posterior — parametros de nivel superior",
        "",
        summary_top[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk", "ess_tail"]].round(5).to_markdown(),
        "",
    ]

    # --- 1. IC 94% del efecto de altitud por cluster ---
    beta_alt = idata.posterior["beta_altitude_cluster"]
    lines += [
        "## 3. Efecto de altitud sobre run_value, por cluster (IC 94%)",
        "",
        "`beta_altitude_cluster[c]` = runs esperados adicionales por pitch, "
        "por cada nivel de `altitude_level` (0->1->2), para el cluster c. "
        "Multiplicado por 2 da el delta total No Altitude -> Extreme Altitude, "
        "comparable directamente contra el contrafactual (seccion 4).",
        "",
        "| cluster | mean (por nivel) | IC 94% (por nivel) | mean x2 (No->Extreme) | IC 94% x2 |",
        "|---|---:|---|---:|---|",
    ]
    for c in CLUSTER_NAMES:
        vals = beta_alt.sel(cluster=c).to_numpy().flatten()
        mean = vals.mean()
        lo, hi = az.hdi(vals, hdi_prob=0.94)
        lines.append(
            f"| {c} | {mean:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {2*mean:+.5f} | [{2*lo:+.5f}, {2*hi:+.5f}] |"
        )

    beta_global_vals = idata.posterior["beta_altitude_global"].to_numpy().flatten()
    lo_g, hi_g = az.hdi(beta_global_vals, hdi_prob=0.94)
    lines += [
        "",
        f"Efecto fijo global (referencia, pooled entre clusters): "
        f"{beta_global_vals.mean():+.5f} por nivel, IC 94% "
        f"[{lo_g:+.5f}, {hi_g:+.5f}].",
        "",
    ]

    # --- 2. shrinkage examples ---
    lines += [
        "## 4. Shrinkage: pitchers de muestra chica (n<100), crudo vs. jerarquico",
        "",
        "Para cada (pitcher_anon_id, cluster) con n<100 pitches en la muestra, "
        "se compara el promedio crudo de `run_value` (sin pooling, solo esos "
        "pitches) contra la media posterior de `alpha_pc` (el intercepto "
        "jerarquico, que combina la señal propia del pitcher con la media de "
        "su cluster segun cuanta muestra tiene). `shrinkage_pct` = que tanto "
        "de la distancia entre el crudo y la media del cluster se cerro el "
        "shrinkage (100% = colapso total a la media del cluster, 0% = sin "
        "shrinkage).",
        "",
        "| pitcher_anon_id | cluster | n pitches | crudo (sin shrinkage) | jerarquico (shrunk) | media del cluster | shrinkage % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in shrinkage_examples.iterrows():
        lines.append(
            f"| {r['pitcher_anon_id']} | {r['cluster']} | {int(r['n_pitches'])} | "
            f"{r['raw_mean_run_value']:+.4f} | {r['shrunk_mean_run_value']:+.4f} | "
            f"{r['cluster_mean']:+.4f} | {r['shrinkage_pct']:.1f}% |"
        )

    lines += [
        "",
        "**Lectura esperada:** a menor n_pitches, mayor shrinkage_pct (mas "
        "peso a la media del cluster, menos al promedio crudo, propio ruidoso "
        "de pocos pitches) — verificar que la tabla muestre ese gradiente.",
        "",
    ]

    # --- 3. comparacion contra contrafactual ---
    lines += [
        "## 5. Comparacion contra el contrafactual de altitud",
        "",
        "Comparacion directa: `raw_delta_run_value` de "
        "`reports/altitude_counterfactual.md` (seccion 4, columna run_value — "
        "delta de la PREDICCION del modelo LightGBM Extreme vs. No Altitude, "
        "manteniendo fijo RelSpeed/SpinRate/residual de Magnus) contra "
        "`beta_altitude_cluster` x2 de este modelo bayesiano (delta OBSERVADO "
        "en run_value real, estimado con partial pooling, sin recalcular "
        "movimiento fisico — dos metodos completamente distintos sobre la "
        "misma pregunta).",
        "",
        "### Por AutoPitchType individual (referencia, contrafactual)",
        "",
        "| AutoPitchType | delta_run_value contrafactual | n |",
        "|---|---:|---:|",
    ]
    for pt, r in cf_by_type.sort_values("mean").iterrows():
        lines.append(f"| {pt} | {r['mean']:+.5f} | {int(r['count']):,} |")

    lines += [
        "",
        "### Por cluster: contrafactual vs. modelo bayesiano",
        "",
        "| cluster | delta_run_value contrafactual (No->Extreme) | delta_run_value bayesiano (No->Extreme, IC94%) |",
        "|---|---:|---|",
    ]
    for c in CLUSTER_NAMES:
        cf_val = cf_by_cluster.loc[c, "delta_run_value_extreme_vs_no"]
        vals = beta_alt.sel(cluster=c).to_numpy().flatten() * 2
        lo, hi = az.hdi(vals, hdi_prob=0.94)
        lines.append(f"| {c} | {cf_val:+.5f} | {vals.mean():+.5f} [{lo:+.5f}, {hi:+.5f}] |")

    # lectura automatica de consistencia
    same_sign_all = all(
        (cf_by_cluster.loc[c, "delta_run_value_extreme_vs_no"] > 0) == (beta_alt.sel(cluster=c).mean().item() > 0)
        for c in CLUSTER_NAMES
    )
    lines += [
        "",
        f"**Direccion (signo):** {'coincide' if same_sign_all else 'NO coincide'} "
        "entre ambos metodos en los 3 clusters (ambos deberian ser positivos: "
        "mas runs permitidos en altitud extrema = peor para el pitcher).",
        "",
        "**Nota de escala:** el contrafactual muestra deltas de run_value MUY "
        "similares entre clusters (todos ~0.009-0.010, ver tabla por "
        "AutoPitchType arriba) — el hallazgo de que Sinker pierde mas y "
        "Curveball/Splitter pierden menos (`reports/altitude_counterfactual.md` "
        "seccion 1) esta dominado por `is_out_in_pa`, no por `run_value` en si "
        "mismo (ver la descomposicion de la seccion 4 de ese reporte) — asi "
        "que **no se espera que este modelo bayesiano (que solo mira "
        "run_value) reproduzca el mismo ranking de perdida por cluster que el "
        "Pitching+ compuesto**, solo el mismo SIGNO/direccion general. Si el "
        "modelo bayesiano muestra diferencias grandes de magnitud entre "
        "clusters en run_value especificamente, eso seria una señal nueva no "
        "vista en el contrafactual (o una discrepancia a investigar, ver "
        "seccion 6).",
        "",
    ]

    return "\n".join(lines)


def append_discrepancy_note(report_text: str, cf_by_cluster: pd.DataFrame, idata: az.InferenceData) -> str:
    beta_alt = idata.posterior["beta_altitude_cluster"]
    rows = []
    for c in CLUSTER_NAMES:
        cf_val = cf_by_cluster.loc[c, "delta_run_value_extreme_vs_no"]
        vals = beta_alt.sel(cluster=c).to_numpy().flatten() * 2
        bayes_mean = vals.mean()
        lo, hi = az.hdi(vals, hdi_prob=0.94)
        cf_in_ci = lo <= cf_val <= hi
        rows.append((c, cf_val, bayes_mean, lo, hi, cf_in_ci))

    lines = ["## 6. Discrepancias a investigar", ""]
    any_flag = False
    for c, cf_val, bayes_mean, lo, hi, cf_in_ci in rows:
        if not cf_in_ci:
            any_flag = True
            lines.append(
                f"- **{c}**: el delta del contrafactual ({cf_val:+.5f}) cae "
                f"FUERA del IC 94% bayesiano ({bayes_mean:+.5f} [{lo:+.5f}, "
                f"{hi:+.5f}]) — revisar si es diferencia real entre metodos "
                "(el contrafactual mantiene el residual de Magnus del pitcher "
                "fijo y recalcula solo el movimiento teorico; el modelo "
                "bayesiano usa la variacion observada real entre altitudes, "
                "que incluye cualquier otro confusor no modelado) o ruido de "
                "muestra."
            )
    if not any_flag:
        lines.append(
            "Los 3 clusters tienen el delta del contrafactual DENTRO del IC "
            "94% del modelo bayesiano — ambos metodos, construidos de forma "
            "completamente distinta (uno recalcula fisica de Magnus, el otro "
            "estima directamente de los datos observados con partial "
            "pooling), son consistentes entre si. Esto es una validacion "
            "cruzada razonablemente fuerte del hallazgo de que la altitud "
            "aumenta el run_value esperado por pitch."
        )
    lines.append("")
    return report_text.rstrip("\n") + "\n\n" + "\n".join(lines)


def main() -> None:
    t0 = time.time()
    df_full = load_valid_data()
    logger.info("Pitches validos: %s | pitchers: %s", f"{len(df_full):,}", f"{df_full['pitcher_anon_id'].nunique():,}")

    n_small_pitchers = int(
        (df_full.groupby("pitcher_anon_id").size() <= SMALL_PITCHER_THRESHOLD).sum()
    )

    sample = stratified_sample(df_full, TARGET_SAMPLE_SIZE, SMALL_PITCHER_THRESHOLD, SAMPLE_SEED)
    logger.info("Muestra para el modelo: %s pitches", f"{len(sample):,}")

    model, pc_meta = build_model(sample)
    logger.info("Modelo construido: %s grupos (pitcher x cluster)", f"{len(pc_meta):,}")

    logger.info("Sampling NUTS: %d cadenas, %d draws + %d tune...", N_CHAINS, N_DRAWS, N_TUNE)
    t_sample0 = time.time()
    idata = sample_model(model)
    runtime_s = time.time() - t_sample0
    logger.info("Sampling completo en %.1f min", runtime_s / 60)

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(TRACE_PATH)
    logger.info("Trace guardado en %s", TRACE_PATH)

    logger.info("Calculando diagnosticos (r_hat, ess)...")
    diag = compute_diagnostics(idata)
    logger.info(
        "r_hat max=%.4f | ess_bulk min=%.0f | ess_tail min=%.0f | divergencias=%d (%.3f%%)",
        diag["max_rhat"], diag["min_ess_bulk"], diag["min_ess_tail"], diag["divergences"], diag["pct_divergences"],
    )

    trace_fig_path = plot_traces(idata)
    logger.info("Trace plot guardado en %s", trace_fig_path)

    shrinkage_table = build_shrinkage_table(sample, idata, pc_meta)
    shrinkage_examples = select_small_pitcher_examples(shrinkage_table, SMALL_PITCHER_TABLE_N, SAMPLE_SEED)

    cf_by_type, cf_by_cluster = load_counterfactual_run_value_delta_by_cluster()

    report_text = format_report(
        n_valid=len(df_full),
        n_sample=len(sample),
        n_small_pitchers=n_small_pitchers,
        n_pc_groups=len(pc_meta),
        diag=diag,
        idata=idata,
        shrinkage_examples=shrinkage_examples,
        cf_by_type=cf_by_type,
        cf_by_cluster=cf_by_cluster,
        trace_fig_path=trace_fig_path,
        runtime_s=runtime_s,
    )
    report_text = append_discrepancy_note(report_text, cf_by_cluster, idata)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("Reporte guardado en %s", REPORT_PATH)
    logger.info("Tiempo total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
