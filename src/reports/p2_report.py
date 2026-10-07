"""Reporte P2: `reports/p2_pitcher_profiles_altitude.md`, figuras en `reports/figures/p2/` y tablas en `data/web/`.

Lee los resultados de `src/analysis/pitcher_profiles.py`, `src/models/stuff_plus_physical.py`
y `src/models/location_plus_v2.py`. No entrena nada.
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import p2_common as pc
from src.analysis import pitcher_profiles as pp
from src.models import location_plus_v2 as locp
from src.models import stuff_plus_physical as spp

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "p2_pitcher_profiles_altitude.md"
FIG = pc.FIG_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SENS_WARNING = ("Análisis exploratorio. Confiabilidad estimada ≈ {rel:.2f}: la mayor parte de la variación es ruido de muestra o de "
                "contexto. NO es evidencia de una capacidad estable del pitcher para resistir la altitud.")
CF_WARNING = ("Este modelo es explicativo y contrafactual; no mostró una ventaja predictiva frente a una normalización ambiental "
              "más simple.")


def _ci(v, lo, hi, d=3, pct=False):
    if pct:
        return f"{100 * v:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def _rel(path: Path) -> str:
    """Ruta relativa a `reports/` (donde vive el reporte), para que las imágenes se vean en el Markdown."""
    return path.relative_to(REPORT_PATH.parent).as_posix()


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def fig_contact_damage(dmg: dict) -> Path:
    t = dmg["tabla"]
    cat = t[t["nivel"] == "categoria"].set_index("ambiente").loc[["No Altitude", "Medium", "Extreme"]]
    st = t[t["nivel"] == "estadio"]
    colors = {"No Altitude": "#4c78a8", "Medium": "#f2a541", "Extreme": "#d1495b"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    panels = [("p_bip_por_pitch", "Probabilidad de permitir\nbola en juego (por pitch)", 100, "%"),
              ("rv_por_bip", "Daño esperado dado que hubo\nbola en juego (carreras RE24)", 1, ""),
              ("dano_bip_por_pitch", "Daño esperado de contacto\npor 100 pitches = P(BIP) × E[RV|BIP]", 100, "")]
    for ax, (c, title, k, unit) in zip(axes[:3], panels):
        y = cat[c] * k
        err = np.vstack([y - cat[f"{c}_lo"] * k, cat[f"{c}_hi"] * k - y])
        ax.bar(cat.index, y, yerr=err, color=[colors[i] for i in cat.index], capsize=5)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.3f}{unit}" if k == 1 else f"{v:.2f}{unit}", ha="center", va="bottom", fontsize=9)
    ax = axes[3]
    for c, g in st.groupby("categoria"):
        ax.scatter(100 * g["p_bip_por_pitch"], g["rv_por_bip"], s=g["juegos"] * 1.5, color=colors[c], alpha=0.75, label=c,
                   edgecolor="black", lw=0.4)
    ax.set_xlabel("P(bola en juego) por pitch (%)")
    ax.set_ylabel("carreras por bola en juego")
    ax.set_title("Cada punto = un estadio anónimo\n(tamaño = juegos)", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("¿Por qué evitar contacto vale más en altitud? Las dos piezas del daño de contacto (IC 95% por bootstrap de juegos)",
                 fontsize=11)
    fig.tight_layout()
    path = FIG / "contact_damage_by_environment.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_avoid_contact(res: dict) -> Path:
    h3 = res["hyp"]["h3"]["por_categoria"]
    cats = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = [h3[c]["G"] for c in cats]
    err = np.array([[h3[c]["G"] - h3[c]["ic95_G"][0] for c in cats], [h3[c]["ic95_G"][1] - h3[c]["G"] for c in cats]])
    ax.bar([pc.CAT_SHORT[c] for c in cats], y, yerr=err, capsize=5, color=["#4c78a8", "#f2a541", "#d1495b"], alpha=0.85)
    for j, cat in enumerate(["Medium Altitude", "Extreme Altitude"], start=1):
        g = list(res["consistencia"][cat]["G"].values())
        ax.scatter(np.full(len(g), j) + np.linspace(-0.25, 0.25, len(g)), g, color="black", s=12, zorder=3,
                   label="estadio-temporada" if j == 2 else None)
    ax.axhline(y[0], color="#4c78a8", ls="--", lw=1)
    ax.set_ylabel("carreras: bola en juego − whiff, mismo conteo")
    ax.set_title("Valor de convertir una bola en juego en un whiff (G)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "value_of_avoiding_contact.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_ranking(top: pd.DataFrame) -> Path:
    t = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.2))
    ax.errorbar(t["aepp"], range(len(t)), xerr=[t["aepp"] - t["aepp_lo80"], t["aepp_hi80"] - t["aepp"]], fmt="o", color="#d1495b",
                ecolor="gray", capsize=3)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"#{r} {p}" for r, p in zip(t["rank"], t["pitcher_anon_id"])], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("AEPP: carreras evitadas por 100 turnos vs. pitcher promedio en altitud (intervalo 80%)")
    ax.set_title("Pitchers con perfil más favorable para ambientes de altitud", fontsize=11)
    fig.tight_layout()
    path = FIG / "ranking_aepp_top.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_sensitivity(s: pd.DataFrame, r: float, rel: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.errorbar(s["s_mod_rel"], s["s_emp"], yerr=1.645 * s["s_emp_se"], fmt="o", ms=3, color="#888", ecolor="#ddd", alpha=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Sensibilidad según el modelo (carreras/100, relativa a la liga)")
    ax.set_ylabel("Sensibilidad observada (carreras/100)")
    ax.set_title(f"Modelo vs. observado: r = {r:+.2f}  ·  confiabilidad observada ≈ {rel:.2f}\n"
                 "HALLAZGO NEGATIVO: el modelo no identifica quién se adapta mejor", fontsize=10)
    fig.tight_layout()
    path = FIG / "sensitivity_model_vs_observed.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_interactions(inter: pd.DataFrame, labels: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(inter) + 1.2))
    cols = {"No Altitude": "#4c78a8", "Medium": "#f2a541", "Extreme": "#d1495b"}
    for i, (_, r) in enumerate(inter.iterrows()):
        for k, (c, col) in enumerate(cols.items()):
            v = r[f"pend_{c}"]
            ax.errorbar(v, i + (k - 1) * 0.22, xerr=[[v - r[f"pend_{c}_lo"]], [r[f"pend_{c}_hi"] - v]], fmt="o", color=col, capsize=2,
                        label=c if i == 0 else None)
    ax.set_yticks(range(len(inter)))
    ax.set_yticklabels([labels.get(m, m) for m in inter["caracteristica"]], fontsize=9)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("carreras evitadas por 100 turnos por +1 sd\n(característica medida en la otra mitad de los juegos)")
    ax.set_title("¿Cuánto vale cada característica en cada ambiente? (asociación, IC 95%)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "interactions_by_environment.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Tablas para la web
# ---------------------------------------------------------------------------


PROFILE_COLS = ["n_pitches", "n_pa", "n_swings", "n_bip", "temporadas", "pitches_no_altitude", "pitches_medium", "pitches_extreme",
                "stuff_plus_physical", "stuff_plus_physical_lo80", "stuff_plus_physical_hi80",
                "location_plus", "location_plus_lo80", "location_plus_hi80", "pitching_plus", "pitching_plus_lo80", "pitching_plus_hi80"]


def web_tables(res: dict) -> dict:
    t = res["pitchers"]
    elig = res["elegibles"].to_numpy()
    cols = ["pitcher_anon_id"] + PROFILE_COLS
    for m in pp.METRICS:
        cols += [f"{m}_crudo", f"{m}_ajustado", m, f"{m}_lo80", f"{m}_hi80", f"{m}_n", f"{m}_B"]
    cols += ["contact_pct", "aepp", "aepp_lo80", "aepp_hi80", "aepp_extreme", "aepp_no_altitude", "bono_altitud", "bono_altitud_lo80",
             "bono_altitud_hi80", "rank", "rank_lo80", "rank_hi80", "p_top25", "arquetipo", "explicacion"]
    cols += [c for c in t.columns if c.startswith(("pct_", "mix_"))]
    prof = t.loc[elig, [c for c in cols if c in t.columns]].sort_values("rank")
    prof.insert(1, "lectura", "Su perfil presenta características que históricamente son más favorables (o menos) en ambientes de altitud; "
                              "no es una predicción garantizada de rendimiento.")
    paths = {"pitcher_altitude_profiles": pc.write_web_table(prof, "pitcher_altitude_profiles")}
    rk = prof[["rank", "pitcher_anon_id", "aepp", "aepp_lo80", "aepp_hi80", "rank_lo80", "rank_hi80", "p_top25", "bono_altitud",
               "n_pitches", "n_pa", "pitches_extreme", "pitches_medium", "arquetipo", "explicacion"]]
    paths["pitcher_rankings"] = pc.write_web_table(rk, "pitcher_rankings")

    s = res["sensibilidad"]["tabla"]
    rel = res["sensibilidad"]["info"]["confiabilidad_spearman_brown"]
    sens = s[["pitcher_anon_id", "rank_menos_sensible", "s_emp", "s_emp_se", "s_emp_lo80", "s_emp_hi80", "s_emp_shrunk", "factor_shrinkage",
              "s_shrunk_lo80", "s_shrunk_hi80", "n_extreme", "n_no", "s_mod_rel", "s_mod_se", "confiabilidad_global"]].sort_values(
        "rank_menos_sensible").copy()
    sens["advertencia"] = SENS_WARNING.format(rel=rel)
    paths["pitcher_sensitivity"] = pc.write_web_table(sens, "pitcher_sensitivity")

    st = pd.read_parquet(spp.SCORES_PATH)
    paths["stuff_plus_physical"] = pc.write_web_table(st, "stuff_plus_physical")
    lo = pd.read_parquet(locp.SCORES_PATH)
    paths["location_plus"] = pc.write_web_table(lo, "location_plus")
    grid = pd.read_parquet(locp.GRID_PATH)
    paths["location_plus_heatmap"] = pc.write_web_table(grid[grid["con_soporte"]].drop(columns="con_soporte"), "location_plus_heatmap")
    paths["pitch_type_altitude_effects"] = pc.write_web_table(res["tipos"], "pitch_type_altitude_effects")
    paths["contact_damage_by_environment"] = pc.write_web_table(res["dano"]["tabla"], "contact_damage_by_environment")
    paths["linear_weights_by_environment"] = pc.write_web_table(res["pesos"].reset_index(), "linear_weights_by_environment")
    paths["characteristic_value_by_environment"] = pc.write_web_table(res["interacciones"], "characteristic_value_by_environment")
    readme = ["# Tablas para la página (P2)", "",
              "Generadas por `src/reports/p2_report.py`. Identificadores anónimos (`pitcher_anon_id`, `ANON_STADIUM_xx`). "
              "Definiciones en `reports/p2_pitcher_profiles_altitude.md`.", ""]
    for k, p in paths.items():
        readme.append(f"- `{p.name}`")
    readme += ["", f"`pitcher_sensitivity.csv`: {SENS_WARNING.format(rel=rel)}", "",
               f"Contrafactual físico (`stuff_plus_physical_neutral`, Pitching+ neutral): {CF_WARNING}", ""]
    (pc.WEB_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return paths


def run() -> None:
    from src.reports import p2_report_text

    res = pp.run()
    stuff = json.loads(spp.JSON_PATH.read_text(encoding="utf-8"))
    loc = json.loads(locp.JSON_PATH.read_text(encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)
    e = res["pitchers"][res["elegibles"].to_numpy()].sort_values("rank")
    sens = res["sensibilidad"]
    figs = {"damage": fig_contact_damage(res["dano"]), "avoid": fig_avoid_contact(res), "ranking": fig_ranking(e.head(25)),
            "sens": fig_sensitivity(sens["tabla"], sens["r_mod_emp"], sens["info"]["confiabilidad_spearman_brown"]),
            "inter": fig_interactions(res["interacciones"], p2_report_text.LAB_INT)}
    web = web_tables(res)
    REPORT_PATH.write_text(p2_report_text.build_report(res, stuff, loc, figs, web), encoding="utf-8")
    logger.info("Reporte: %s", REPORT_PATH)


if __name__ == "__main__":
    run()
