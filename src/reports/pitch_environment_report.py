"""Reportes de la extensión: `reports/pitching_plus_general_vs_altitude.md` y `reports/optimal_pitch_by_environment.md`.

Todas las cifras salen de `src/analysis/pitch_environment.py` (o de su caché `data/processed/pitch_environment_results.pkl`).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_B = ROOT / "reports" / "pitching_plus_general_vs_altitude.md"
REPORT_C = ROOT / "reports" / "optimal_pitch_by_environment.md"
SUMMARY = ROOT / "data" / "web" / "summary.json"
TYPES = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
EVENT_ES = {"walk": "base por bolas", "strikeout": "ponche", "hbp": "golpeado", "out": "out en juego", "single": "sencillo",
            "double": "doble", "triple": "triple", "home_run": "jonrón"}
VAR_ES = {"RelSpeed": "velocidad", "InducedVertBreak": "IVB", "HorzBreak": "HB", "SpinRate": "spin", "Extension": "extensión",
          "RelHeight": "altura de salida", "RelSide": "salida lateral"}
UNITS = {"RelSpeed": ("mph", 1), "InducedVertBreak": ("pulg.", 1), "HorzBreak": ("pulg.", 1), "SpinRate": ("rpm", 0),
         "Extension": ("pies", 2), "RelHeight": ("pies", 2), "RelSide": ("pies", 2)}


def _f(x, d=2, sign=False):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# Parte B
# ---------------------------------------------------------------------------


def report_b(out: dict) -> str:
    E = out["env_json"]
    C = E["conjuntos"]
    st = out["part_b_stats"]
    wc = out["within_count"]
    cor = wc["correlaciones"]
    tab = wc["tabla"]
    S = json.loads(SUMMARY.read_text(encoding="utf-8"))
    L = []
    add = L.extend
    add([
        "# Pitching+ general vs. Pitching+ para ambientes de altitud",
        "",
        "Generado por `src/analysis/pitch_environment.py` y `src/reports/pitch_environment_report.py`. No cambia ningún índice oficial:",
        "Pitching+, Stuff+ Physical, Location+, AEPP y el ranking quedan igual. Es una extensión para explicar qué cambia en el valor de un",
        "lanzamiento cuando cambia el costo de los resultados.",
        "",
        "## 1. Qué es \"normal\" (decisión del baseline)",
        "",
        "Se revisaron tres candidatos:",
        "",
        "| candidato | qué es | ¿sirve como baseline? |",
        "|---|---|---|",
        "| Pitching+ simple ajustado por entorno | el índice oficial por pitcher: E[Δ] menos la media de su temporada × estadio anónimo | no: el ajuste resta el nivel de cada ambiente, pero NO cambia lo que vale cada resultado; compararlo con otra cosa mezclaría el ajuste con el costo de los resultados |",
        "| valores RE24 de No Altitude | la recursión por conteo con solo los pitches de No Altitude | sirve como contraste extremo (se reporta), pero no es lo que usa la página |",
        "| **índice general de liga** | **la recursión por conteo con TODOS los pitches: exactamente los Δ(evento, conteo) del Pitching+ oficial** | **sí: es el Pitching+ que el usuario ya ve en el simulador** |",
        "",
        f"**General = valores de toda la liga.** Se verificó contra el artefacto final: diferencia máxima en Δ(evento, conteo) = "
        f"{E['verificacion']['max_dif_deltas_general_vs_oficial']:.1e}; en E[Δ] por pitch = {E['verificacion']['max_dif_xrv_general_vs_oficial']:.1e}.",
        "",
        "**Pitching+ para Extreme = las mismas probabilidades del modelo** (mismo pitch, misma física, misma ubicación, mismo conteo) "
        "valuadas con lo que vale cada resultado en Extreme:",
        "",
        "1. Valores terminales de Extreme: promedio del run value RE24 de cada resultado en los pitches de Extreme.",
        "2. Recursión por conteo de Extreme: V(conteo) y Δ(evento, conteo) con las frecuencias de transición observadas en Extreme.",
        "3. **Calibración del costo de contacto.** El modelo no ve el ambiente: su reparto out / sencillo / doble / triple / jonrón dada una "
        "bola en juego es el de la liga, y P2 §7.2 mostró que acierta P(BIP) pero **subestima el daño por bola en juego en altitud**. Por eso "
        "a la rama de bola en juego se le suma, en cada ambiente, la diferencia entre el Δ observado de sus bolas en juego y el que predice el "
        "modelo. Es un valor observado, no un peso elegido a mano.",
        "",
        "No hay ningún \"bono de altitud\" manual. Escala: 100 + 10·(media − E[Δ])/sd con media y sd de todos los lanzamientos reales de la liga "
        "valuados con cada juego de valores, sin desplazamiento por ambiente, para que la única diferencia sea el costo de los resultados.",
        "",
        "## 2. Lo que vale cada resultado",
        "",
        "| resultado (valor terminal, carreras) | General | No Altitude | Extreme |",
        "|---|---:|---:|---:|",
    ])
    for e, lab in EVENT_ES.items():
        add([f"| {lab} | {_f(C['general']['terminales'][e], 3, True)} | {_f(C['no_altitude']['terminales'][e], 3, True)} | "
             f"{_f(C['extreme']['terminales'][e], 3, True)} |"])
    add([
        "",
        "| V(conteo): carreras esperadas del resto del turno | General | No Altitude | Extreme |",
        "|---|---:|---:|---:|",
    ])
    for c in ("0-0", "0-2", "2-0", "3-2"):
        add([f"| {c} | {_f(C['general']['V'][c], 3, True)} | {_f(C['no_altitude']['V'][c], 3, True)} | {_f(C['extreme']['V'][c], 3, True)} |"])
    add([
        "",
        "| calibración del costo de contacto (Δ medio de una bola en juego) | predicho por el modelo | observado | ajuste |",
        "|---|---:|---:|---:|",
    ])
    for k, lab in (("general", "General (no se ajusta; control)"), ("no_altitude", "No Altitude"), ("medium", "Medium"), ("extreme", "Extreme")):
        add([f"| {lab} | {_f(C[k]['e_bip_predicho'], 4, True)} | {_f(C[k]['e_bip_observado'], 4, True)} | {_f(C[k]['shift_bip'], 4, True)} |"])
    add([
        "",
        "**Lectura.** En Extreme casi todo resultado vale un poco más para la ofensiva, y **el turno completo empieza siendo más caro** "
        f"(V(0-0) = {_f(C['extreme']['V']['0-0'], 3, True)} contra {_f(C['no_altitude']['V']['0-0'], 3, True)} en No Altitude). Eso es "
        "importante: un strike o una bola dejan al pitcher en un conteo que en Extreme también cuesta más.",
        "",
        "## 3. ¿Cambia el valor de un mismo lanzamiento?",
        "",
        f"Sobre {st['n_pitches']:,} lanzamientos reales:",
        "",
        f"- Correlación de rangos entre Pitching+ general y Pitching+ Extreme: **{st['spearman_general_extreme']:.3f}** "
        f"(No Altitude vs. Extreme: {st['spearman_no_extreme']:.3f}).",
        f"- Cambio absoluto mediano: **{st['cambio_abs_mediana']:.2f} puntos** de Pitching+ (p90: {st['cambio_abs_p90']:.2f}); "
        f"{100 * st['pct_cambio_mayor_1']:.0f}% cambia más de 1 punto y {100 * st['pct_cambio_mayor_3']:.1f}% más de 3.",
        "",
        "El orden de los lanzamientos casi no cambia. Es coherente con P2: el ranking de pitchers con pesos de altitud y sin ellos correlaciona "
        f"{S['aepp']['rho_pesos_altitud_vs_no']:.3f}.",
        "",
        "### 3.1 ¿Quién gana valor relativo? Comparando con el lanzamiento típico del MISMO conteo",
        "",
        "Comparar contra el promedio de toda la liga mezcla conteos (con dos strikes hay más contacto y los valores de cada conteo cambian con el "
        "ambiente). Por eso la ganancia se mide frente al lanzamiento típico del mismo conteo: ganancia = −100·[(E[Δ] − media del conteo) en "
        "Extreme − (lo mismo) en el baseline], en carreras por 100 pitches (positivo = el pitch gana valor relativo en Extreme).",
        "",
        "| fase | quintil de P(whiff) relativo | P(whiff) relativo | P(BIP) relativo | ganancia vs. General | ganancia vs. No Altitude |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for r in tab.itertuples():
        add([f"| {r.fase} | {r.quintil_whiff} | {_f(100 * r.p_whiff_rel, 1, True)} pp | {_f(100 * r.p_bip_rel, 1, True)} pp | "
             f"{_f(r.gana_vs_general, 2, True)} | {_f(r.gana_vs_no, 2, True)} |"])
    two, early = cor["dos strikes"], cor["0 o 1 strike"]
    add([
        "",
        f"- **Con dos strikes** (donde un whiff es ponche), la ganancia sigue al swing-and-miss: correlación con P(whiff) relativo "
        f"{two['whiff_vs_no']:+.2f} contra No Altitude ({two['whiff_vs_general']:+.2f} contra General).",
        f"- **Con 0 o 1 strike**, el swing-and-miss casi no mueve el valor relativo ({early['whiff_vs_no']:+.2f}); la ganancia se asocia más bien "
        f"con P(BIP) ({early['bip_vs_no']:+.2f}). Antes de dos strikes, un pitch que no termina en contacto lleva a otro conteo, y en Extreme "
        "el resto del turno también es más caro: evitar el contacto temprano no rinde más.",
        "",
        "**Conclusión.** El hallazgo de P1 (evitar una bola en juego vale ≈ "
        f"{100 * S['G_aumento_relativo_extreme']:.0f}% más por swing y cambiar una bola en juego por un ponche ≈ "
        f"{100 * S['pesos']['k_vs_bip_aumento_extreme']:.0f}% más por turno) se ve en el valor por lanzamiento **donde el whiff termina el turno**: "
        "con dos strikes. Lanzamiento por lanzamiento, el Pitching+ para Extreme es casi el mismo que el general; la diferencia está en el margen "
        "y premia el swing-and-miss en el momento en que produce ponches.",
        "",
        "### 3.2 Por tipo de pitch (promedios; Pitching+ en escala de lanzamiento)",
        "",
        "| tipo | lanzamientos | Pitching+ general | Pitching+ Extreme | cambio |",
        "|---|---:|---:|---:|---:|",
    ])
    tb = out["part_b_tabla"]
    for r in tb[tb["agrupacion"] == "tipo de pitch"].itertuples():
        add([f"| {r.grupo} | {r.n:,} | {r.pp_general:.2f} | {r.pp_extreme:.2f} | {r.cambio:+.2f} |"])
    add([
        "",
        "Los promedios por tipo casi no cambian: el costo de los resultados no favorece en bloque a un tipo de pitch. (Los efectos por tipo de la "
        "sección 6 de la página son otra cosa: cambios observados de whiff, movimiento y valor realizado entre ambientes.)",
        "",
        "## 4. Cómo se explica en la página",
        "",
        "Para el lanzamiento del simulador se muestra la tabla General / Extreme (Stuff+, Location+, P(whiff), P(BIP), E[RV | BIP], Pitching+) y una "
        "explicación automática a partir de la descomposición Σ (p − p̄)·(Δ_Extreme − Δ_General) por grupo de resultados (bola en juego, ponche / "
        "strike, base por bolas, golpeado), donde p̄ es la probabilidad media de la liga en el mismo conteo. Stuff+ y Location+ no cambian: "
        "es el mismo lanzamiento. Lo que cambia es lo que vale cada resultado.",
        "",
        "## 5. Lo que no decimos",
        "",
        "- No es un efecto causal de la altitud: son los valores observados en esos ambientes (P1/P2).",
        "- No reemplaza al Pitching+ oficial ni al AEPP: el AEPP mide al pitcher por turno y es la herramienta para elegir pitchers.",
        "- La calibración del costo de contacto es un promedio por ambiente: no distingue estadios dentro de Extreme.",
        "- Hereda las limitaciones del run value (por ejemplo, la reconstrucción de los outs de cada jugada).",
        "",
    ])
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Parte C
# ---------------------------------------------------------------------------


def _prof(P: pd.DataFrame, pt: str, obj: str, ctx: str) -> pd.Series:
    return P[(P["tipo"] == pt) & (P["objetivo"] == obj) & (P["contexto"] == ctx)].iloc[0]


def _cmp(Cm: pd.DataFrame, pt: str, alt: str, ctx: str) -> pd.Series:
    return Cm[(Cm["tipo"] == pt) & (Cm["alterno"] == alt) & (Cm["contexto"] == ctx)].iloc[0]


def _fmt_var(c, v):
    u, d = UNITS[c]
    return f"{v:.{d}f} {u}"


def global_answer(Cm: pd.DataFrame) -> dict:
    out = {}
    for alt in ("extreme", "extreme_fisica"):
        x = Cm[Cm["alterno"] == alt]
        out[alt] = {"clases": x["clase"].value_counts().to_dict(), "n": int(len(x)),
                    "spearman_min": float(x["spearman"].min()), "spearman_mediana": float(x["spearman"].median()),
                    "jaccard_mediana": float(x["jaccard_top1"].median()), "percentil_min": float(x["percentil_en_alt_del_mejor_general"].min()),
                    "moderado_o_mucho": x[x["clase"] != "Poco"][["tipo", "contexto", "clase", "variable_que_mas_cambia"]].to_dict("records")}
    return out


def reorder_text(Cm: pd.DataFrame, n: int = 3) -> str:
    x = Cm[(Cm["alterno"] == "extreme_fisica") & (Cm["contexto"] == "0-0")].sort_values("spearman").head(n)
    return ", ".join(f"{r.tipo} (correlación de rangos {r.spearman:.2f})" for r in x.itertuples())


def report_c(out: dict) -> str:
    P, Cm, Lc, meta, fx = out["profiles"], out["comparison"], out["location"], out["meta"], out["fixed"]
    G = global_answer(Cm)
    ge, gf = G["extreme"], G["extreme_fisica"]
    L = []
    add = L.extend
    add([
        "# ¿Es el mismo un buen lanzamiento en cualquier ambiente?",
        "",
        "Generado por `src/analysis/pitch_environment.py`. Extensión exploratoria: no cambia ningún índice oficial.",
        "",
        "## 1. Respuesta corta",
        "",
    ])
    poco_e = ge["clases"].get("Poco", 0)
    poco_f = gf["clases"].get("Poco", 0)
    add([
        f"- **Con el costo de los resultados de Extreme (capa validada), el lanzamiento óptimo prácticamente no cambia**: en {poco_e} de "
        f"{ge['n']} combinaciones tipo × conteo el cambio del perfil óptimo es menor que el ruido de muestreo; la correlación de rangos "
        f"entre candidatos es ≥ {ge['spearman_min']:.3f} y el mejor lanzamiento general siempre queda en el top "
        f"{100 - ge['percentil_min']:.2f}% en Extreme.",
        f"- **Si además se simula la física del aire de Extreme (capa aproximada)**, el orden de los candidatos se mueve más "
        f"(correlación de rangos mediana {gf['spearman_mediana']:.2f}; mínimo {gf['spearman_min']:.2f}), pero el perfil óptimo sigue dentro del "
        f"ruido en {poco_f} de {gf['n']} combinaciones.",
        "- **\"Mejor Stuff\" y \"mejor pitch\" no son lo mismo**: los lanzamientos con mayor Stuff+ Physical y los de mayor Pitching+ casi no se "
        "traslapan (sección 5). Esa diferencia es mucho más grande que la que produce el ambiente.",
        "",
        "Conclusión: **un gran lanzamiento sigue siendo un gran lanzamiento**. El ambiente cambia el margen, no la definición.",
        "",
        "## 2. Método",
        "",
        "- **Candidatos = lanzamientos reales** de cada tipo lanzados en No Altitude (muestra de hasta "
        f"{meta['pool_max']:,} por tipo, marco de un derecho). No hay optimización libre: el óptimo no puede ser 110 mph ni una combinación "
        "nunca observada, y la correlación entre velocidad, movimiento, spin y salida es la real.",
        "- **Objetivos:** Stuff+ Physical; Pitching+ general (valores de la liga); Pitching+ Extreme (mismo pitch, valores de Extreme, capa "
        "validada); Pitching+ Extreme con física aproximada (el mismo pitch re-simulado en el aire de Extreme con el contrafactual, que "
        "sobreestima ≈ 20% el cambio de IVB). Pitching+ en escala de lanzamiento (100 = promedio de la liga).",
        "- **Contextos (documentados, no elegidos a la vista del resultado):** conteo 0-0 (neutral), 0-2 (dos strikes) y 2-0 (bateador adelante); "
        "mismo lado y lado contrario ponderados por la mezcla real de cada tipo.",
        "- **Experimento 1 (física sola):** la ubicación se fija en la ubicación mediana real del tipo para cada enfrentamiento.",
        f"- **Experimento 2 (física + ubicación):** una rejilla de {meta['rejilla_pies']} pies, solo celdas con al menos {meta['loc_min']} "
        "lanzamientos reales del tipo en el mismo enfrentamiento.",
        f"- **El óptimo** se reporta de dos formas: el **mejor lanzamiento observado** y el **perfil del top 1%** (mediana y p10–p90), puntuado con "
        f"sus {meta['k_vecinos']} vecinos reales más cercanos (\"mejor plausible\"). Si el mejor observado está rodeado de vecinos que ni siquiera "
        "están en el top 10%, se marca como inestable.",
        f"- **Incertidumbre:** bootstrap por pitcher ({meta['n_boot']} réplicas) del perfil óptimo. **Distancia** entre óptimos = distancia en "
        f"desviaciones estándar del tipo en 7 variables. {meta['clases']}. No se reentrenaron los modelos en cada réplica (5 modelos × 200 "
        "réplicas no es razonable aquí): la incertidumbre del modelo no está incluida.",
        "",
        "## 3. Resultado global",
        "",
        "| tipo | conteo | costos de Extreme: correlación de rangos | top 1% compartido | cambio del perfil | física aprox.: correlación | top 1% compartido | cambio del perfil |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ])
    for pt in TYPES:
        for ctx in ("0-0", "0-2", "2-0"):
            a, b = _cmp(Cm, pt, "extreme", ctx), _cmp(Cm, pt, "extreme_fisica", ctx)
            add([f"| {pt} | {ctx} | {a.spearman:.3f} | {100 * a.jaccard_top1:.0f}% | {a.clase} ({a.distancia_z:.2f} vs. ruido {a.ruido_q95:.2f}) | "
                 f"{b.spearman:.2f} | {100 * b.jaccard_top1:.0f}% | {b.clase} ({b.distancia_z:.2f}) |"])
    mods = gf["moderado_o_mucho"]
    add([
        "",
        "\"Top 1% compartido\" = índice de Jaccard entre los conjuntos top 1% con cada objetivo. "
        + ("Casos con cambio mayor al ruido (física aproximada): " + "; ".join(f"{m['tipo']} {m['contexto']} ({m['clase']}, sobre todo "
                                                                              f"{VAR_ES[m['variable_que_mas_cambia']]})" for m in mods) + "."
           if mods else "Ningún caso supera el ruido con la física aproximada."),
        "",
        "## 4. Tipo por tipo (experimento 1, conteo 0-0)",
        "",
    ])
    for pt in TYPES:
        g, e, f, s = (_prof(P, pt, "general", "0-0"), _prof(P, pt, "extreme", "0-0"), _prof(P, pt, "extreme_fisica", "0-0"),
                      _prof(P, pt, "stuff", "todos"))
        a, b = _cmp(Cm, pt, "extreme", "0-0"), _cmp(Cm, pt, "extreme_fisica", "0-0")
        info = meta["tipos"][pt]
        add([f"### {pt}", "", f"{info['n_candidatos']:,} candidatos reales de {info['n_pitchers']} pitchers.", "",
             "| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |",
             "|---|---|---|---|---|"])
        for c in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension"]:
            add([f"| {VAR_ES[c]} | {_fmt_var(c, g[c + '_med'])} [{_fmt_var(c, g[c + '_p10'])}–{_fmt_var(c, g[c + '_p90'])}] | "
                 f"{_fmt_var(c, e[c + '_med'])} | {_fmt_var(c, f[c + '_med'])} | {_fmt_var(c, s[c + '_med'])} |"])
        add([f"| Pitching+ del perfil (vecinos reales) | {g.puntaje_perfil_vecinos:.1f} | {e.puntaje_perfil_vecinos:.1f} | {f.puntaje_perfil_vecinos:.1f} | "
             f"Stuff+ {s.puntaje_perfil_vecinos:.1f} |",
             f"| mejor lanzamiento observado | {g.puntaje_mejor_observado:.1f} ({'estable' if g.mejor_estable else 'inestable'}) | "
             f"{e.puntaje_mejor_observado:.1f} ({'estable' if e.mejor_estable else 'inestable'}) | {f.puntaje_mejor_observado:.1f} | "
             f"{s.puntaje_mejor_observado:.1f} ({'estable' if s.mejor_estable else 'inestable'}) |",
             "", f"- Costos de Extreme: cambio **{a.clase.lower()}** (distancia {a.distancia_z:.2f} contra ruido {a.ruido_q95:.2f}); el mejor "
             f"lanzamiento general queda en el percentil {a.percentil_en_alt_del_mejor_general:.2f} en Extreme.",
             f"- Física aproximada: cambio **{b.clase.lower()}** (distancia {b.distancia_z:.2f}); la variable que más se mueve es "
             f"**{VAR_ES[b.variable_que_mas_cambia]}** ({b['z_' + b.variable_que_mas_cambia]:+.2f} sd).",
             f"- Mejor Stuff vs. mejor pitch: correlación de rangos {a.stuff_vs_general_spearman:.2f}; top 1% compartido "
             f"{100 * a.jaccard_stuff_general:.0f}%."])
        fr = P[(P["tipo"] == pt) & (P["frontera"].fillna("").str.len() > 0)]
        if len(fr):
            add(["- **Frontera del soporte:** " + "; ".join(f"perfil {r.objetivo} ({r.contexto}) fuera de p1–p99 de los candidatos en "
                                                           + ", ".join(VAR_ES[v] for v in r.frontera.split(";")) for r in fr.itertuples())
                 + ". Sigue dentro del rango real, pero en el borde: leerlo con cautela."])
        add([""])
    add([
        "## 5. \"Mejor Stuff\" no es \"mejor pitch\"",
        "",
        "| tipo | correlación de rangos Stuff+ vs. Pitching+ general (0-0) | top 1% compartido | (0-2) | top 1% compartido |",
        "|---|---:|---:|---:|---:|",
    ])
    for pt in TYPES:
        a, b = _cmp(Cm, pt, "extreme", "0-0"), _cmp(Cm, pt, "extreme", "0-2")
        add([f"| {pt} | {a.stuff_vs_general_spearman:.2f} | {100 * a.jaccard_stuff_general:.0f}% | {b.stuff_vs_general_spearman:.2f} | "
             f"{100 * b.jaccard_stuff_general:.0f}% |"])
    add([
        "",
        "Stuff+ premia lo que hace fallar swings; Pitching+ premia el valor total, que en 0-0 incluye strikes cantados, bolas y contacto débil. "
        "Con dos strikes ambos se acercan, porque ahí el swing fallado es el resultado más valioso.",
        "",
        "## 6. Experimento 2: física + ubicación (conteo 0-0)",
        "",
        "| tipo | enfrentamiento | ubicación óptima General (lado, altura) | Extreme (costos) | Extreme (física aprox.) | distancia General–Extreme | correlación de rangos de combinaciones |",
        "|---|---|---|---|---|---:|---:|",
    ])
    for pt in TYPES:
        for same in ("same", "opposite"):
            x = Lc[(Lc["tipo"] == pt) & (Lc["enfrentamiento"] == same) & (Lc["contexto"] == "0-0")].set_index("objetivo")
            if x.empty:
                continue
            g, e, f = x.loc["general"], x.loc["extreme"], x.loc["extreme_fisica"]
            add([f"| {pt} | {'mismo lado' if same == 'same' else 'lado contrario'} | ({g.x:+.1f}, {g.z:.1f}) | ({e.x:+.1f}, {e.z:.1f}) | "
                 f"({f.x:+.1f}, {f.z:.1f}) | {g.distancia_ubicacion_pies:.2f} pies | {g.spearman_combos:.3f} |"])
    fx0 = Lc[(Lc["contexto"] == "0-0") & (Lc["objetivo"] == "general")].dropna(subset=["fijo_spearman_ubicaciones"])
    fixed_sp = pd.to_numeric(fx0["fijo_spearman_ubicaciones"], errors="coerce")
    same_cell = int((fx0["fijo_mejor_celda_general"] == fx0["fijo_mejor_celda_extreme"]).sum())
    same_cell_f = int((fx0["fijo_mejor_celda_general"] == fx0["fijo_mejor_celda_extreme_fisica"]).sum())
    moved = (Lc[(Lc["contexto"] == "0-0") & (Lc["objetivo"] == "general") & (Lc["distancia_ubicacion_pies"] > 0)])
    add([
        "",
        "Ubicación en pies, marco de un derecho (lado + = lado del brazo del pitcher; con el bateador del mismo lado, + es adentro).",
        "",
        f"- **Mejor combinación pitch + ubicación:** la celda óptima con costos de Extreme coincide con la general en "
        f"{int((Lc[(Lc['contexto'] == '0-0') & (Lc['objetivo'] == 'general')]['distancia_ubicacion_pies'] == 0).sum())} de "
        f"{len(Lc[(Lc['contexto'] == '0-0') & (Lc['objetivo'] == 'general')])} casos. Cuando se mueve ("
        + ", ".join(f"{r.tipo} {'mismo lado' if r.enfrentamiento == 'same' else 'lado contrario'}: {r.distancia_ubicacion_pies:.2f} pies"
                    for r in moved.itertuples())
        + f"), el orden de todas las combinaciones sigue correlacionando ≥ {Lc['spearman_combos'].min():.3f}: hay varias ubicaciones "
        "prácticamente empatadas y el máximo salta entre ellas.",
        (f"- **Lanzamiento fijo con muy buen Stuff+** (el real más cercano al perfil de mejor Stuff+ de cada tipo): el orden de las ubicaciones "
         f"por Pitching+ general y Pitching+ Extreme correlaciona al menos {fixed_sp.min():.3f}, y su mejor celda es la misma en {same_cell} de "
         f"{len(fx0)} casos con los costos de Extreme y en {same_cell_f} de {len(fx0)} con la física aproximada." if len(fx0) else ""),
        "",
        "## 7. Relación con los hallazgos de pitch mix (comparación posterior, no usada en la búsqueda)",
        "",
        "- La sección 6 de la página muestra que, en promedio, la recta pierde más whiff y el slider menos valor en Extreme. Eso describe cómo "
        "cambian los lanzamientos reales entre ambientes (incluye la física del aire y el vuelo de la pelota).",
        "- Este análisis pregunta otra cosa: **cuál sería la mejor versión de cada tipo**. Con los costos de Extreme el óptimo no cambia. Con la "
        f"física aproximada, los tipos que más reordenan sus candidatos en 0-0 son {reorder_text(Cm)}. Es coherente con la sección 6 (sinker y "
        "changeup son los que más movimiento horizontal pierden y la recta la que más IVB y whiff pierde), pero aun así el perfil óptimo de cada "
        "tipo queda cerca del general.",
        "",
        "## 8. Lo que no decimos",
        "",
        "- No es una receta: \"todo pitcher debe lanzar este slider\". El óptimo del modelo es una combinación favorable según los datos y no "
        "considera biomecánica, salud, tunneling, secuencias, repertorio ni fatiga.",
        "- Los perfiles de tipos con poca muestra (Splitter) y los de ruido alto (Cutter, Slider y Curveball tienen umbrales de ruido grandes) "
        "solo detectarían cambios grandes.",
        "- La capa física es aproximada (sobreestima el cambio de IVB) y no incluye el vuelo de la pelota bateada.",
        "- Los modelos se entrenaron con todos los datos; el bootstrap es de candidatos (pitchers), no de modelos.",
        "",
    ])
    return "\n".join(L)


def write(out: dict | None = None) -> None:
    if out is None:
        from src.analysis.pitch_environment import RESULTS_PATH
        with open(RESULTS_PATH, "rb") as fh:
            out = pickle.load(fh)
    REPORT_B.write_text(report_b(out), encoding="utf-8")
    REPORT_C.write_text(report_c(out), encoding="utf-8")


if __name__ == "__main__":
    write()
