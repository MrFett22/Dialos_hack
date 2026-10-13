"""Pitching+ para ambientes de altitud — página final del proyecto (Streamlit).

Para correrla:  .venv\\Scripts\\streamlit run web/app.py

Toda cifra viene de `data/web/` (generado por el pipeline: pasos `p2_report` y `web_data`) o de los modelos
finales (`web/engine.py`). No hay valores escritos a mano en los textos.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web import charts  # noqa: E402
from web.engine import CAT_LABEL, CATS, EVENT_LABELS, LABELS, LOC_MIN_CELL, PHYS_INPUTS, PITCH_TYPES, Engine  # noqa: E402

WEB = ROOT / "data" / "web"
st.set_page_config(page_title="Pitching+ en altitud", page_icon="⚾", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando modelos…")
def engine() -> Engine:
    return Engine()


@st.cache_data(show_spinner=False)
def load() -> dict:
    csv = lambda n: pd.read_csv(WEB / f"{n}.csv")  # noqa: E731
    return {"S": json.loads((WEB / "summary.json").read_text(encoding="utf-8")),
            "damage": csv("contact_damage_by_environment"), "lw": csv("linear_weights_by_environment"),
            "cv": csv("characteristic_value_by_environment"), "heat": csv("location_plus_heatmap"),
            "types": csv("pitch_type_altitude_effects"), "prof": csv("pitcher_altitude_profiles"),
            "mix": csv("pitcher_pitch_types"), "sens": csv("pitcher_sensitivity"), "stadiums": csv("anon_stadiums"),
            "stuff": csv("stuff_plus_physical"),
            "envv": json.loads((WEB / "pitching_env_values.json").read_text(encoding="utf-8")),
            "relgain": csv("pitching_relative_gain_by_whiff"), "opt_prof": csv("optimal_pitch_profiles"),
            "opt_cmp": csv("optimal_pitch_comparison"), "opt_loc": csv("optimal_pitch_location"),
            "opt_heat": csv("optimal_location_heatmap"), "opt_fixed": csv("optimal_location_fixed_pitch"),
            "opt_meta": json.loads((WEB / "optimal_pitch_meta.json").read_text(encoding="utf-8"))}


D = load()
S = D["S"]
E = engine()

BADGE = {"obs": ("Observado", "b-obs"), "asoc": ("Asociación", "b-asoc"), "pred": ("Predictivo", "b-pred"),
         "sim": ("Simulación del modelo", "b-sim"), "expl": ("Exploratorio", "b-expl"), "hip": ("Hipótesis", "b-hip"),
         "lim": ("Limitación", "b-lim")}


def badges(*kinds) -> str:
    return "".join(f'<span class="badge {BADGE[k][1]}">{BADGE[k][0]}</span>' for k in kinds)


def section(anchor: str, title: str, *kinds):
    st.markdown(f'<a name="{anchor}"></a>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if kinds:
        st.markdown(badges(*kinds), unsafe_allow_html=True)


def levels(simple: str, beisbol: str, tecnico: str):
    t1, t2, t3 = st.tabs(["Explicación simple", "En béisbol", "Técnico"])
    t1.markdown(simple)
    t2.markdown(beisbol)
    t3.markdown(tecnico)


def card(label: str, value: str, sub: str = "", hero: bool = False) -> str:
    return f'<div class="card{" hero-score" if hero else ""}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def stat(num: str, cap: str, cls: str = "") -> str:
    return f'<div class="stat"><div class="num {cls}">{num}</div><div class="cap">{cap}</div></div>'


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


EF = S["efectos"]
G_UP = S["G_aumento_relativo_extreme"]
K_UP = S["pesos"]["k_vs_bip_aumento_extreme"]
CONS = EF["whiff_swing"]["consistencia_extreme"]
AE = S["aepp"]
SE = S["sensibilidad"]
SX = S["sensibilidad_extra"]
VB = S["valor_bip"]
GB = S["ahorro_rodado"]
TP = S["tipos_pitch"]
STAB = {r["metrica"]: r for r in S["estabilidad_metricas"]}
REG_LABEL = {"corazón": "Corazón de la zona", "borde": "Borde", "persecución": "Zona de persecución", "desperdicio": "Desperdicio"}

# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------

NAV = [("hero", "Inicio"), ("evidencia", "1 · Evidencia"), ("insight", "2 · El contacto cuesta más"), ("perfil", "3 · Qué buscamos"),
       ("stuff", "4 · Stuff+ Physical"), ("location", "5 · Location+"), ("tipos", "6 · Tipos de lanzamiento"), ("pitching", "7 · Pitching+"),
       ("simulador", "8 · Construye tu lanzamiento"), ("generalvsalt", "9 · Pitching+ general vs. altitud"),
       ("optimo", "10 · ¿El mismo buen lanzamiento?"), ("ambientes", "11 · Compara ambientes"), ("ranking", "12 · Ranking de pitchers"),
       ("sensibilidad", "13 · Sensibilidad (exploratorio)"), ("fisica", "14 · Física y contrafactual"), ("metodo", "15 · Qué funcionó y qué no"),
       ("futuro", "16 · Qué mejoraríamos")]
with st.sidebar:
    st.markdown("### Pitching+ en altitud")
    st.markdown("\n".join(f"- [{t}](#{a})" for a, t in NAV))
    st.markdown("---")
    st.markdown("**Tipos de evidencia**")
    st.markdown("<br>".join(badges(k) for k in BADGE), unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Datos anónimos: estadios `ANON_STADIUM_xx`, pitchers `pitcher_xxxxx`. No se usa información externa.")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown('<a name="hero"></a>', unsafe_allow_html=True)
st.markdown(f"""
<div class="hero">
  <div class="kicker">Proyecto de baseball analytics · datos Trackman anónimos 2024–2026</div>
  <h1>Pitching+ para ambientes de altitud</h1>
  <div class="question">¿Qué cambia cuando lanzar en altura hace más caro permitir contacto?</div>
  <div class="chips">
    <span class="chip">Whiff por swing {100 * EF['whiff_swing']['extreme']:+.1f} pp en Extreme</span>
    <span class="chip">K por turno {100 * EF['k_pa']['extreme']:+.1f} pp</span>
    <span class="chip">Run value por bola en juego {EF['rv_bip']['extreme']:+.3f}</span>
    <span class="chip">Evitar contacto vale ≈ {pct(G_UP, 0)} más</span>
    <span class="chip">{CONS[0]}/{CONS[1]} estadio-temporadas</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="bigidea">En estadios de altitud no necesariamente necesitamos un tipo completamente nuevo de pitcher.
Necesitamos identificar qué pitchers poseen en mayor medida <b>las habilidades que se vuelven todavía más valiosas cuando permitir contacto cuesta más</b>:
<ul><li>generar whiffs;</li><li>generar ponches;</li><li>evitar bolas en juego;</li><li>mantener buen Stuff;</li><li>ejecutar buenas ubicaciones.</li></ul>
</div>""", unsafe_allow_html=True)
st.markdown("**Stuff+** nos dice qué tan bueno es el lanzamiento físicamente. **Location+**, qué tan bien fue ejecutado. "
            "**Pitching+**, cuánto vale realmente.")

# ---------------------------------------------------------------------------
# 1. Evidencia
# ---------------------------------------------------------------------------

section("evidencia", "1 · Los ambientes de altitud se juegan distinto", "obs", "asoc")
st.markdown("**Los ambientes de altitud media y extrema presentan patrones consistentes de menor generación de whiffs y ponches "
            "y mayor costo de permitir contacto.**")
c = st.columns(4)
c[0].markdown(stat(f"{100 * EF['whiff_swing']['extreme']:+.1f} pp", f"whiff por swing en Extreme (de {pct(EF['whiff_swing']['no_altitude'])})", "neg"), unsafe_allow_html=True)
c[1].markdown(stat(f"{100 * EF['k_pa']['extreme']:+.1f} pp", f"ponches por turno (de {pct(EF['k_pa']['no_altitude'])})", "neg"), unsafe_allow_html=True)
c[2].markdown(stat(f"{EF['rv_bip']['extreme']:+.3f}", f"carreras por bola en juego (de {EF['rv_bip']['no_altitude']:+.3f})", "pos"), unsafe_allow_html=True)
c[3].markdown(stat(f"+{pct(G_UP, 0)}", "valor de convertir una bola en juego en whiff", "pos"), unsafe_allow_html=True)

c = st.columns(4)
c[0].plotly_chart(charts.effects_bars(EF, "whiff_swing", "Whiff por swing"), width="stretch")
c[1].plotly_chart(charts.effects_bars(EF, "k_pa", "Ponches por turno"), width="stretch")
c[2].plotly_chart(charts.effects_bars(EF, "bip_swing", "Bolas en juego por swing"), width="stretch")
c[3].plotly_chart(charts.effects_bars(EF, "rv_bip", "Run value por bola en juego", scale=1, unit="carreras"), width="stretch")
st.caption("Diferencias contra No Altitude con efectos fijos de pitcher y bateador (además de temporada y conteo): el mismo pitcher "
           "contra el mismo bateador. Barras = IC 95% con errores agrupados por juego.")

st.plotly_chart(charts.consistency_strip(S["consistencia_detalle"], ["Medium Altitude", "Extreme Altitude"]), width="stretch")
cm = EF["whiff_swing"]["consistencia_medium"]
st.markdown(f"El patrón aparece en **{CONS[0]} de {CONS[1]} combinaciones estadio-temporada de Extreme** en las cuatro variables. "
            f"**Medium se parece bastante a Extreme**: el contacto se encarece casi igual (run value por BIP "
            f"{EF['rv_bip']['medium']:+.3f} vs. {EF['rv_bip']['extreme']:+.3f}), aunque el whiff baja menos "
            f"({100 * EF['whiff_swing']['medium']:+.1f} pp; {cm[0]}/{cm[1]} estadio-temporadas).")
c = st.columns(3)
for col, (a, lab) in zip(c, (("No Altitude", "No Altitude"), ("Medium", "Medium"), ("Extreme", "Extreme"))):
    col.markdown(stat(f"{VB[a]:+.3f}", f"carreras por bola en juego en {lab} (promedio crudo del turno)", "neg" if a == "No Altitude" else "pos"),
                 unsafe_allow_html=True)
st.markdown(f"Medium ({VB['Medium']:+.3f}) está mucho más cerca de Extreme ({VB['Extreme']:+.3f}) que de No Altitude ({VB['No Altitude']:+.3f}). "
            "Por eso hablamos de **ambientes de altitud** y no de un umbral mágico llamado \"Extreme\".")
st.markdown('<div class="warnbox"><b>Lo que NO decimos:</b> "la altitud extrema causa estos efectos". Cada categoría agrupa pocos '
            "estadios anónimos, y la altitud no se puede separar del resto de las características del parque (dimensiones, clima, "
            "calibración del Trackman). Que Medium se parezca a Extreme refuerza la cautela.</div>", unsafe_allow_html=True)
levels(
    "Comparamos a los mismos pitchers contra los mismos bateadores en estadios de distinta altitud. En los de altitud, los bateadores "
    "fallan menos swings, se ponchan menos y, cuando conectan, hacen más daño.",
    "Menos swing-and-miss y menos ponches significan más pelotas en juego, y cada pelota en juego vale más para la ofensiva. "
    "Para el pitcher, el contacto es el gran enemigo en esos parques.",
    "Regresión lineal con efectos fijos de pitcher, bateador, temporada y conteo (proyecciones alternadas, teorema de "
    "Frisch–Waugh–Lovell); errores agrupados por juego. Universos: swings (whiff, BIP), turnos (K) y bolas en juego (run value RE24). "
    "Consistencia: la misma regresión por cada estadio anónimo × temporada. Fuente: `reports/hipotesis_altitud.md` y "
    "`reports/p2_pitcher_profiles_altitude.md` §2.")

# ---------------------------------------------------------------------------
# 2. El contacto cuesta más
# ---------------------------------------------------------------------------

section("insight", "2 · El contacto cuesta más", "obs")
dm = S["dano"]
st.markdown(f'<div class="bigidea">El problema no es principalmente que haya muchas más bolas en juego. <b>El problema es que cada bola '
            f'en juego cuesta más.</b></div>', unsafe_allow_html=True)
c = st.columns(3)
c[0].markdown(stat(f"{dm['por_100_no']:.2f} → {dm['por_100_extreme']:.2f}", "carreras de daño de contacto por 100 pitches (No Altitude → Extreme)", "pos"), unsafe_allow_html=True)
c[1].markdown(stat(pct(dm["fraccion_por_dano_extreme"], 0), "del aumento: cada bola en juego hace más daño", "pos"), unsafe_allow_html=True)
c[2].markdown(stat(pct(dm["fraccion_por_probabilidad_extreme"], 0), "del aumento: hay más bolas en juego", "neg"), unsafe_allow_html=True)
dmg = D["damage"]
st.plotly_chart(charts.damage_panels(dmg[dmg["nivel"] == "categoria"]), width="stretch")
c = st.columns([1, 1.4])
c[0].plotly_chart(charts.decomposition_donut(dm["fraccion_por_dano_extreme"], dm["fraccion_por_probabilidad_extreme"], dm["fraccion_interaccion_extreme"]),
                  width="stretch")
c[1].plotly_chart(charts.stadium_scatter(dmg[dmg["nivel"] == "estadio"]), width="stretch")
levels(
    "Imagina que cada vez que el bateador conecta la pelota, el pitcher paga una multa. En altitud no le conectan mucho más seguido, "
    "pero la multa es mucho más cara.",
    f"Una pelota en juego en Extreme vale para la ofensiva {EF['rv_bip']['extreme']:+.3f} carreras más que en No Altitude: más hits y más extra bases. "
    "Por eso el ponche y el swing fallado protegen más.",
    "Daño esperado de contacto por pitch = P(BIP) × E[RV | BIP]. Descomposición del cambio: ΔP·E_N + P_N·ΔE + ΔP·ΔE. IC por bootstrap de "
    "juegos (500 réplicas). En el árbol de Pitching+ el componente de bola en juego es Σ_k P(k)·Δ(k) = P(BIP)·E[Δ | BIP] "
    "(identidad verificada numéricamente).")
st.caption("Advertencia de medición: la velocidad de salida y el contacto duro dependen de la calibración del Trackman de cada estadio y "
           "cambian entre temporadas (sección 15). El run value no depende de esa calibración.")

st.markdown("### Si el contacto ocurre, el tipo de contacto también importa")
c = st.columns([1, 1.25])
with c[0]:
    st.markdown(badges("hip") + " **Pirámide conceptual del resultado de un pitch** (explicación visual del riesgo, no una ecuación)",
                unsafe_allow_html=True)
    st.markdown("""<div class="pyramid">
<div class="lvl l1">Whiff / ponche<small>mejor escenario: no hay contacto</small></div><div class="arrow">▼</div>
<div class="lvl l2">Foul / strike favorable<small>el turno sigue, la cuenta mejora</small></div><div class="arrow">▼</div>
<div class="lvl l3">Bola en juego débil / rodado<small>contacto, pero poco peligroso</small></div><div class="arrow">▼</div>
<div class="lvl l4">Bola en juego peligrosa<small>línea, elevado con fuerza</small></div><div class="arrow">▼</div>
<div class="lvl l5">Jonrón / extra bases<small>peor escenario</small></div></div>""", unsafe_allow_html=True)
with c[1]:
    st.markdown(badges("obs") + " **Segunda línea de defensa: el rodado**", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown(stat(f"{GB['No Altitude']:.3f}", "carreras que ahorra un rodado en lugar de un batazo al aire · No Altitude", "neg"), unsafe_allow_html=True)
    b.markdown(stat(f"{GB['Extreme']:.3f}", "carreras que ahorra · Extreme", "pos"), unsafe_allow_html=True)
    st.markdown(f"En Extreme el rodado parece proteger relativamente más (+{pct(GB['Extreme'] / GB['No Altitude'] - 1, 0)}; Medium "
                f"{GB['Medium']:.3f}).\n\n"
                "- **Primera defensa:** evitar el contacto (whiff, ponche).\n"
                "- **Segunda defensa:** si hay contacto, favorecer contacto menos peligroso.\n\n"
                f"No convertimos el % de rodados en la métrica central: como habilidad del pitcher es más estable que el daño por bola en juego "
                f"(r año a año {STAB['gb_pct']['estabilidad_cruda']:.2f}), pero casi no se relaciona con las carreras evitadas en altitud del año "
                f"siguiente (r = {STAB['gb_pct']['r_carreras_altitud_t1']:+.2f}). Es una estrategia secundaria.")

# ---------------------------------------------------------------------------
# 3. Qué buscamos
# ---------------------------------------------------------------------------

section("perfil", "3 · Qué buscamos en un pitcher: swing-and-miss", "obs", "asoc")
st.markdown(f'<div class="bigidea">Perfil favorable: <b>muchos whiffs, muchos ponches, pocas bolas en juego.</b> Reemplazar una bola en '
            f'juego por un ponche vale ≈ <b>{pct(K_UP, 0)} más</b> en Extreme.</div>', unsafe_allow_html=True)
c = st.columns([1, 1.3])
c[0].plotly_chart(charts.k_vs_bip(D["lw"][D["lw"]["ambiente"] != "Altitud (Medium + Extreme)"]), width="stretch")
lab_cv = {"whiff_swing": "Whiff/Swing", "k_pct": "K%", "bb_pct": "pocas BB", "bip_pct": "pocas bolas en juego", "rv_bip": "poco daño por BIP",
          "hard_pct": "poco contacto duro", "gb_pct": "muchos rodados", "stuff_plus_physical": "Stuff+ Physical", "location_plus": "Location+"}
with c[1]:
    st.markdown(badges("asoc") + " **¿Cuánto vale cada característica en cada ambiente?**", unsafe_allow_html=True)
    st.plotly_chart(charts.characteristic_forest(D["cv"], lab_cv), width="stretch")
st.caption("Diseño cruzado: la característica del pitcher se mide en una mitad de sus juegos y las carreras en la otra. Son asociaciones con "
           "muchas comparaciones; se leen como patrón, no como pruebas individuales.")
st.markdown("Generar whiffs, ponches y el Stuff físico que los produce se asocian con más carreras evitadas en Extreme. En cambio, **el daño "
            "que permite un pitcher cuando hay contacto casi no se repite de un año a otro**: es sobre todo ruido y contexto, no una habilidad estable.")

st.markdown("### Dos formas de ser bueno… que no pesan igual")
c = st.columns(2)
c[0].markdown(f"""<div class="okbox"><b>A. Evitar el contacto</b> — bajar P(BIP)<ul>
<li>Whiff/Swing: r año a año <b>{STAB['whiff_swing']['estabilidad_cruda']:.2f}</b></li>
<li>K%: <b>{STAB['k_pct']['estabilidad_cruda']:.2f}</b> · BIP%: <b>{STAB['bip_pct']['estabilidad_cruda']:.2f}</b></li>
<li>Stuff+ Physical: <b>{STAB['stuff_plus_physical']['estabilidad_cruda']:.2f}</b> · Location+: <b>{STAB['location_plus']['estabilidad_cruda']:.2f}</b> ·
Pitching+: <b>{STAB['pitching_plus']['estabilidad_cruda']:.2f}</b></li></ul>Es una habilidad que se repite: se puede medir y usar para elegir pitchers.</div>""",
              unsafe_allow_html=True)
c[1].markdown(f"""<div class="warnbox"><b>B. Reducir el daño cuando hay contacto</b> — bajar E[RV | BIP]<ul>
<li>Run Value/BIP: r año a año <b>{STAB['rv_bip']['estabilidad_cruda']:+.2f}</b></li>
<li>Hard Contact%: <b>{STAB['hard_pct']['estabilidad_cruda']:.2f}</b> · HR/BIP: <b>{STAB['hr_bip']['estabilidad_cruda']:.2f}</b></li></ul>
En estos datos casi no se repite: puede reflejar ruido, defensa, estadio o muestra. No le damos el mismo peso interpretativo.</div>""",
              unsafe_allow_html=True)
hl = {m: "#1f3a63" for m in ("whiff_swing", "whiff_pitch", "k_pct", "bip_pct", "stuff_plus_physical", "location_plus", "pitching_plus")}
hl.update({m: "#e08a3c" for m in ("rv_bip", "hard_pct", "hr_bip")})
st.plotly_chart(charts.stability_bars(pd.DataFrame(S["estabilidad_metricas"]), hl), width="stretch")
st.markdown(f'<div class="bigidea">La capacidad de generar whiffs y ponches se repite mucho más de un año a otro que el daño específico que '
            f'ocurre cuando ponen la pelota en juego. <b>Por eso el perfil favorable se apoya en evitar contacto.</b></div>', unsafe_allow_html=True)
st.caption(f"Azul: evitar contacto y calidad del pitch. Naranja: control del daño de contacto. Pares de temporadas consecutivas del mismo pitcher "
           f"({STAB['whiff_swing']['n_pares']} pares; {' y '.join(S['temporadas']['pares_validacion'])}); métricas ajustadas por entorno, IC 95% por bootstrap.")

# ---------------------------------------------------------------------------
# 4. Stuff+ Physical
# ---------------------------------------------------------------------------

section("stuff", "4 · Stuff+ Physical", "pred")
SS = S["stuff"]
st.markdown("**Mide qué tan buenas son las características físicas del lanzamiento sin darle crédito directo por la ubicación ni por el ambiente.**")
c = st.columns(4)
c[0].markdown(stat(pct(SS["eta2_ubicacion_physical"]), "de su predicción la explica la ubicación"), unsafe_allow_html=True)
c[1].markdown(stat(pct(SS["eta2_ubicacion_v1"], 0), "en Stuff+ v1 (benchmark predictivo)"), unsafe_allow_html=True)
c[2].markdown(stat(f"{SS['estabilidad']['stuff_plus_physical']:.2f}", "estabilidad año a año"), unsafe_allow_html=True)
c[3].markdown(stat(f"{SS['delta_vs_v4_whiff']['delta_r']:+.3f}", f"Δr whiff/swing t+1 vs. v4 (P = {pct(SS['delta_vs_v4_whiff']['p_mejor'], 0)})"), unsafe_allow_html=True)
c = st.columns(2)
names = {"stuff_plus_v1": "v1 benchmark", "stuff_plus_v4": "v4", "stuff_plus_physical": "Physical"}
c[0].plotly_chart(charts.bars_simple(list(names.values()), [100 * SS[k] for k in ("eta2_ubicacion_v1", "eta2_ubicacion_v4", "eta2_ubicacion_physical")],
                                     "% de la predicción explicada por la ubicación (menos = más separado de Location+)", fmt="{:.1f}%",
                                     colors=["#8892a6", "#8892a6", "#d1495b"]), width="stretch")
c[1].plotly_chart(charts.bars_simple(list(names.values()), [SS["whiff_t1"][k] for k in names], "Correlación con el whiff/swing del año siguiente",
                                     fmt="{:.3f}", colors=["#8892a6", "#8892a6", "#d1495b"]), width="stretch")
st.markdown("Variables físicas que usa: velocidad, movimiento vertical inducido y horizontal, spin rate, eje de giro, extensión, altura y lado "
            "de salida, ángulos de llegada **a ubicación constante** y diferencias contra la recta del pitcher, más el tipo de pitch. El lado del "
            "bateador se promedia con la mezcla de la liga para no premiar ni castigar los enfrentamientos que le tocaron.")
with st.expander("Comparar neutral vs. observado (tres versiones del mismo modelo)"):
    V = S["stuff_versiones"]
    vt = pd.DataFrame([
        {"versión": "Observado", "qué hace con el ambiente": "nada: mide lo que el bateador realmente vio", "uso": "explicar lo que pasó"},
        {"versión": "Stuff+ Physical (oficial)", "qué hace con el ambiente": "ajuste simple por temporada × estadio anónimo", "uso": "ranking y scouting"},
        {"versión": "Neutral físico", "qué hace con el ambiente": "contrafactual físico a No Altitude", "uso": "explicar el mecanismo"}])
    for k, i in (("observado", 0), ("oficial", 1), ("neutral", 2)):
        vt.loc[i, "r con % de pitches en Extreme"] = V[k]["r_pct_extreme"]
        vt.loc[i, "estabilidad"] = V[k]["estabilidad"]
        vt.loc[i, "r con whiff/swing t+1"] = V[k]["whiff_t1"]
    st.dataframe(vt.style.format({"r con % de pitches en Extreme": "{:+.3f}", "estabilidad": "{:.3f}", "r con whiff/swing t+1": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.markdown(f"- El **observado** castiga a quien lanzó mucho en altitud (r = {V['observado']['r_pct_extreme']:+.2f}).\n"
                f"- La versión **oficial** casi no depende del ambiente ({V['oficial']['r_pct_extreme']:+.3f}) y es la que mejor predice el whiff "
                f"del año siguiente ({V['oficial']['whiff_t1']:.3f} vs. {V['neutral']['whiff_t1']:.3f} de la neutral): **es la que se usa en toda la página**.\n"
                f"- La **neutral física** es un poco más estable ({V['neutral']['estabilidad']:.3f}) y sirve para explicar el mecanismo (sección 14).")
levels(
    "Stuff+ responde: si el bateador le tira, ¿qué tan difícil es pegarle a este pitch solo por cómo se mueve y qué tan rápido va? 100 es el "
    "promedio y cada 10 puntos son una desviación estándar.",
    "Un pitch con mucha velocidad, movimiento poco común y buena extensión hace fallar más swings. Stuff+ lo mide sin importar si el pitcher "
    "lo puso en buen o mal lugar.",
    f"Target: P(whiff | swing), LightGBM con validación cruzada por pitcher (GroupKFold k=5). Sin SpeedDrop (casi todo es drag) y con los "
    f"ángulos de llegada residualizados contra la ubicación. Ajuste simple por temporada × estadio anónimo. Escala de pitcher: shrinkage "
    f"empírico-bayesiano, 100 + 10·(θ̂ − μ)/τ. AUC sobre swings: Physical {SS['auc']['Stuff+ Physical (puntaje, lados marginalizados)']:.3f}; "
    f"v1 {SS['auc']['Stuff+ v1 (P(whiff) por pitch)']:.3f} (porque v1 ve la ubicación). Detalle: `reports/stuff_plus_physical.md`.")

# ---------------------------------------------------------------------------
# 5. Location+
# ---------------------------------------------------------------------------

section("location", "5 · Location+", "pred")
SL = S["location"]
st.markdown("**Mide qué tan favorable fue la ubicación del lanzamiento dado el tipo de pitch, el conteo y el bateador.**")
c = st.columns(3)
c[0].markdown(stat(f"{SL['r_stuff']:+.2f}", "correlación con Stuff+ Physical: casi independientes"), unsafe_allow_html=True)
c[1].markdown(stat(f"{SL['bb_t1']:+.2f}", "correlación con el BB% del año siguiente"), unsafe_allow_html=True)
c[2].markdown(stat(f"{SL['bb_propio_t1']:+.2f}", "el propio BB% predice su BB% futuro peor"), unsafe_allow_html=True)
c = st.columns([1, 2.2])
with c[0]:
    ht = st.selectbox("Tipo de pitch", PITCH_TYPES, index=3, key="heat_type")
    hm = st.radio("Enfrentamiento (pitcher derecho)", ["mismo lado", "lado contrario"], key="heat_match")
    hc = st.radio("Conteo", ["pitcher adelante", "conteo parejo", "pitcher atrás"], key="heat_count")
    st.caption("Azul: zonas que **suelen** ser más favorables para este tipo de pitch que su ubicación típica. Rojo: menos favorables. "
               "Solo se muestran celdas con suficientes pitches reales. Ninguna zona garantiza un out.")
hg = D["heat"]
c[1].plotly_chart(charts.heatmap(hg[(hg["tipo"] == ht) & (hg["enfrentamiento"] == hm) & (hg["grupo_conteo"] == hc)],
                                 f"Location+ · {ht} · {hm} · {hc}"), width="stretch")
reg = pd.DataFrame(SL["regiones"])
reg["etiqueta"] = reg["region"].map(REG_LABEL)
top_reg = reg.loc[reg["extreme_menos_no_por_100"].idxmax()]
st.markdown(f'<div class="bigidea">{badges("asoc")} Los pitches que terminan en el <b>{top_reg["etiqueta"].lower()}</b> son los que más valor '
            f'pierden en estos ambientes ({top_reg["extreme_menos_no_por_100"]:+.2f} carreras por 100 pitches en Extreme).</div>', unsafe_allow_html=True)
c = st.columns([1.2, 1])
c[0].plotly_chart(charts.region_bars(reg), width="stretch")
with c[1]:
    rt = reg[["etiqueta", "pct_pitches", "delta_medio_no_altitude_por_100", "extreme_menos_no_por_100"]].rename(columns={
        "etiqueta": "región", "pct_pitches": "% de pitches", "delta_medio_no_altitude_por_100": "valor en No Altitude",
        "extreme_menos_no_por_100": "Extreme − No"})
    st.dataframe(rt.style.format({"% de pitches": "{:.0%}", "valor en No Altitude": "{:+.2f}", "Extreme − No": "{:+.2f}"}), width="stretch", hide_index=True)
    st.caption("Carreras por 100 pitches para la ofensiva (negativo = bueno para el pitcher). Efectos fijos de pitcher, bateador, temporada y conteo. "
               "Regiones aproximadas, de adentro hacia afuera: corazón, borde, persecución y desperdicio (`location_plus_v2.region`).")
st.markdown("**Probable explicación:** en el corazón hay mucho contacto, y en altitud ese contacto es más caro. Fuera de la zona casi no hay "
            "contacto, así que casi no cambia.")
st.markdown('<div class="warnbox"><b>No significa "nunca lances al centro".</b> Es una asociación promedio: los strikes en el centro también '
            'sirven para no caer atrás en la cuenta, y la región se definió de forma aproximada.</div>', unsafe_allow_html=True)
levels(
    "Location+ responde: con este tipo de pitch y en este conteo, ¿el pitcher lo puso en un buen lugar o en uno peligroso?",
    "Un slider abajo y afuera con dos strikes suele ser buen lugar; el mismo slider en el centro con el pitcher atrás en la cuenta, no. "
    "Location+ compara cada pitch con la ubicación típica de su mismo contexto.",
    "Mismo árbol de probabilidades que Pitching+, con solo ubicación (lado relativo al bateador y altura), conteo, tipo de pitch y manos. "
    "Puntaje por pitch = E[Δ | ubicación, contexto] − promedio de E[Δ] en ese contexto. Detalle: `reports/location_plus_v2.md`.")

# ---------------------------------------------------------------------------
# 6. Tipos de lanzamiento
# ---------------------------------------------------------------------------

section("tipos", "6 · ¿Qué tipos de lanzamiento conservan mejor su valor en altitud?", "obs", "expl")
ty = D["types"].copy()
tyi = ty.set_index("tipo")
st.markdown('<div class="warnbox"><b>Es una comparación descriptiva, ajustada por contexto, no el efecto causal de elegir un tipo de pitch.</b> '
            "Cada número compara al mismo pitcher contra el mismo bateador (efectos fijos de pitcher, bateador, temporada y conteo), pero quién "
            "lanza cada tipo, cuándo, contra quién y en qué ubicación sigue siendo distinto. \"Lanzar un slider\" no causa por sí mismo "
            "menos pérdida.</div>", unsafe_allow_html=True)


def _t(tp, col):
    return float(tyi.loc[tp, col])


ci = lambda tp, col, k=1: (k * (_t(tp, col) - 1.96 * _t(tp, col + "_se")), k * (_t(tp, col) + 1.96 * _t(tp, col + "_se")))  # noqa: E731
low = TP["menor_costo_extreme"]
lw_ = TP["mayor_perdida_whiff"]
hbs = TP["mayor_perdida_hb"]
mc = TP["menor_cambio_whiff"]
hi_ = TP["mayor_costo_extreme"]
lines = [f"**{low}**: en promedio, el menor costo en Extreme entre los tipos con muestra razonable ({_t(low, 'delta_por_100_E'):+.2f} carreras por "
         f"100 pitches; IC 95% {ci(low, 'delta_por_100_E')[0]:+.2f} a {ci(low, 'delta_por_100_E')[1]:+.2f}), y pierde poco IVB ({_t(low, 'ivb_E'):+.1f} pulg.).",
         f"**{lw_}**: la mayor pérdida de whiff/swing ({100 * _t(lw_, 'whiff_swing_E'):+.1f} pp), con {_t(lw_, 'ivb_E'):+.1f} pulg. de IVB y "
         f"{_t(lw_, 'hb_abs_E'):+.1f} pulg. de movimiento horizontal.",
         f"**{' y '.join(hbs)}**: parecen perder más movimiento horizontal ("
         + ", ".join(f"{h} {_t(h, 'hb_abs_E'):+.1f} pulg." for h in hbs) + ").",
         f"**{mc}**: poco cambio de whiff ({100 * _t(mc, 'whiff_swing_E'):+.1f} pp; IC 95% {ci(mc, 'whiff_swing_E', 100)[0]:+.1f} a "
         f"{ci(mc, 'whiff_swing_E', 100)[1]:+.1f}) y poco movimiento horizontal perdido ({_t(mc, 'hb_abs_E'):+.1f} pulg.).",
         f"**{hi_}**: el mayor costo promedio ({_t(hi_, 'delta_por_100_E'):+.2f} carreras/100); "
         + (f"el daño cuando hay contacto aumenta de manera importante (RV/BIP {_t(hi_, 'rv_bip_E'):+.3f})."
            if TP["mayor_aumento_rv_bip"] == hi_ else f"el mayor aumento del daño por contacto es del {TP['mayor_aumento_rv_bip']}.")]
for sm in TP["muestra_pequena"]:
    lines.append(f"**{sm}**: aparentemente robusto en whiff ({100 * _t(sm, 'whiff_swing_E'):+.1f} pp), pero con solo {int(_t(sm, 'pitches')):,} pitches "
                 f"el IC 95% va de {ci(sm, 'whiff_swing_E', 100)[0]:+.1f} a {ci(sm, 'whiff_swing_E', 100)[1]:+.1f} pp. **No es \"el mejor pitch\"**: "
                 "la incertidumbre es muy alta.")
c = st.columns([1, 1.15])
with c[0]:
    st.markdown(badges("expl") + " **Hallazgos exploratorios (en estos datos, en promedio)**", unsafe_allow_html=True)
    st.markdown("\n".join(f"- {s}" for s in lines))
    st.caption(f"\"Muestra razonable\" = al menos {TP['min_pitches_comparacion']:,} pitches del tipo. Costo total = cambio del valor realizado del "
               "pitch (carreras por 100 pitches para la ofensiva) en Extreme frente a No Altitude.")
c[1].plotly_chart(charts.pitch_type_scatter(ty, TP["min_pitches_comparacion"]), width="stretch")

st.markdown("#### Explora un tipo")
a, b = st.columns([1, 2.4])
sel_t = a.selectbox("Tipo de lanzamiento", ty["tipo"].tolist(), index=ty["tipo"].tolist().index(low), key="type_sel")
r = tyi.loc[sel_t]
small = r["pitches"] < TP["min_pitches_comparacion"]
a.markdown(card(f"{sel_t} · costo en Extreme", f"{r['delta_por_100_E']:+.2f}",
                f"carreras/100 (IC 95% {ci(sel_t, 'delta_por_100_E')[0]:+.2f} a {ci(sel_t, 'delta_por_100_E')[1]:+.2f}) · Medium {r['delta_por_100_M']:+.2f}"),
           unsafe_allow_html=True)
a.markdown(f"- Movimiento: IVB {r['ivb_E']:+.1f} pulg. (de {r['ivb_no']:.1f} en No Altitude), |HB| {r['hb_abs_E']:+.1f} (de {r['hb_abs_no']:.1f}).\n"
           f"- Whiff/swing: {100 * r['whiff_swing_E']:+.1f} pp (de {pct(r['whiff_swing_no'])}).\n"
           f"- Daño por bola en juego: {r['rv_bip_E']:+.3f} carreras.\n- {int(r['pitches']):,} pitches, {pct(r['pct_extreme'], 0)} en Extreme.")
if small:
    a.markdown('<div class="dangerbox">Muestra pequeña: intervalos muy anchos. No conviene sacar conclusiones de este tipo.</div>', unsafe_allow_html=True)
b.plotly_chart(charts.pitch_type_detail(r), width="stretch")
b.caption("Cada barra: cambio contra No Altitude con efectos fijos (mismo pitcher, mismo bateador). Las líneas son IC 95%. "
          "El movimiento compara al mismo pitcher con el mismo tipo en la misma temporada.")
m_sel = st.radio("Comparar todos los tipos en:", list(charts.TYPE_METRICS), format_func=lambda m: charts.TYPE_METRICS[m][0], horizontal=True,
                 index=len(charts.TYPE_METRICS) - 1, key="type_metric")
st.plotly_chart(charts.pitch_type_compare(ty, m_sel, sel_t), width="stretch")
levels(
    f"En altitud todos los lanzamientos pierden algo de movimiento y de valor, pero no todos lo mismo. En estos datos, el {low} es el que menos "
    f"valor pierde en promedio y el {lw_} el que más swings fallados pierde.",
    f"Hipótesis: los pitches que dependen más del giro (Magnus) para moverse pierden más cuando el aire es menos denso; el {low} pierde poco "
    f"movimiento vertical ({_t(low, 'ivb_E'):+.1f} pulg.). Pero cada pitcher es distinto: estos promedios no dicen qué pasará con un pitch concreto.",
    "Para cada tipo, regresión con efectos fijos de pitcher, bateador, temporada y conteo sobre swings (whiff, BIP), bolas en juego (RV) y todos "
    "los pitches (valor realizado RE24 ×100). Movimiento: efectos fijos de pitcher × tipo × temporada. Errores agrupados. Fuente: "
    "`data/web/pitch_type_altitude_effects.csv`, `reports/p2_pitcher_profiles_altitude.md`.")

# ---------------------------------------------------------------------------
# 7. Pitching+
# ---------------------------------------------------------------------------

section("pitching", "7 · Pitching+: cuánto vale realmente el lanzamiento", "pred")
st.markdown("**Pitching+ intenta medir el valor total esperado de un lanzamiento considerando su calidad física, su ubicación, el contexto y la "
            "probabilidad de cada resultado.** No es una suma ponderada de Stuff+ y Location+: sale de un modelo de probabilidades.")
st.markdown("""
<div class="pipeline">
 <div class="node"><b>Características físicas</b><br><small>velocidad, movimiento, spin, eje, extensión, salida, ángulos</small></div>
 <div class="arrow">+</div>
 <div class="node"><b>Ubicación</b><br><small>PlateLocSide, PlateLocHeight</small></div>
 <div class="arrow">+</div>
 <div class="node"><b>Contexto</b><br><small>conteo, tipo de pitch, manos, ambiente*</small></div>
 <div class="arrow">→</div>
 <div class="node"><b>Probabilidades</b><br><small>P(swing), P(whiff | swing), P(foul | contacto), P(resultado | en juego), P(bola / strike / golpeado)</small></div>
 <div class="arrow">→</div>
 <div class="node"><b>E[Δ] = Σ P(evento) × valor(evento)</b><br><small>valor en carreras según el conteo (RE24)</small></div>
 <div class="arrow">→</div>
 <div class="node"><b>Pitching+</b><br><small>escala 100</small></div>
</div>""", unsafe_allow_html=True)
st.caption("*El ambiente entra a través de la física del pitch (movimiento y llegada) y del ajuste simple por estadio; el modelo no ve el parque directamente.")
ex = E.simulate(E.median_inputs("Four-Seam"))
c = st.columns([1.6, 1])
with c[0]:
    st.markdown(badges("sim") + " **Ejemplo real del modelo:** recta (Four-Seam) mediana de la liga, al centro de la zona, conteo 0-0, No Altitude.", unsafe_allow_html=True)
    st.plotly_chart(charts.sankey(ex.probs, EVENT_LABELS), width="stretch")
with c[1]:
    st.markdown(card("Pitching+ del ejemplo", f"{ex.pitching_plus:.0f}", f"E[Δ] = {ex.xrv_per_100:+.2f} carreras por 100 pitches", hero=True), unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"- P(swing) = {ex.p_swing:.1%}\n- P(whiff | swing) = {ex.p_whiff_swing_model:.1%}\n- P(bola en juego) = {ex.p_bip:.1%}\n"
                f"- Daño esperado si hay bola en juego = {ex.e_delta_bip:+.3f} carreras\n"
                f"- Daño esperado de contacto = P(BIP) × E[RV | BIP] = {100 * ex.contact_damage:+.2f} carreras por 100 pitches")
st.markdown('<div class="formula">E[Δ] = Σ_evento P(evento | física, ubicación, contexto) × Δ(evento, conteo)\n'
            'Daño esperado de contacto = Σ_k∈en juego P(k) · Δ(k) = P(BIP) × E[RV | BIP]   ← la parte del árbol que se encarece en altitud\n'
            'Pitching+ (lanzamiento) = 100 + 10 · (media − E[Δ]) / sd       ← menos carreras esperadas = más alto\n'
            'Pitching+ (pitcher)     = 100 + 10 · (μ − θ̂) / τ                ← θ̂ con shrinkage; τ = sd de habilidad real</div>', unsafe_allow_html=True)
VP = S["validacion_perfil"]
st.markdown(f"**Comparación.** Stuff+ explica la parte física y Location+ la de ejecución; Pitching+ las combina con el contexto en un solo valor "
            f"esperado. Por pitcher, Pitching+ es estable (r año a año = {VP['pitching_plus']['estabilidad']:.2f}) y se relaciona con las carreras "
            f"en altitud del año siguiente (r = {VP['pitching_plus']['r_altitud_t1']:.2f}). Para elegir pitchers en altitud, el perfil por "
            f"resultados AEPP predice mejor (sección 12).")
st.markdown('<div class="okbox"><b>Pitching+ no depende de un claim causal.</b> Su lógica para ambientes de altitud es: <i>"en estos contextos '
            'observados, ciertos resultados tienen un costo diferente"</i> (el contacto cuesta más, el ponche protege más). No dice "la altitud causa '
            'exactamente X". Eso permite construir una métrica útil sin pedirle al diseño más de lo que identifica.</div>', unsafe_allow_html=True)
levels(
    "Pitching+ pregunta: si este pitch se lanzara muchas veces en esta misma situación, ¿cuántas carreras le costaría o le ahorraría al "
    "pitcher en promedio? 100 es el lanzamiento promedio; más alto es mejor.",
    "Un pitch puede tener gran Stuff pero quedar en el centro con la cuenta en contra, o tener Stuff normal y ubicarse perfecto. Pitching+ "
    "pondera todo según lo que realmente pasa en el juego: strikes, whiffs, fouls y lo que sucede cuando la pelota se pone en juego.",
    "Cinco modelos LightGBM condicionales: swing; resultado del take (bola / strike / golpeado); whiff | swing; foul | contacto; resultado | en "
    "juego (out, 1B, 2B, 3B, HR). Las 10 probabilidades suman 1. Δ(evento, conteo) = V(estado siguiente) − V(conteo) con la recursión RE24 "
    "por conteo (el foul con 2 strikes es un auto-lazo). Golpeado = "
    f"{S['hbp_run_value']:+.3f} carreras (corregido en P0). Índice principal de P2: ajuste simple por temporada × estadio anónimo, porque en "
    "P0 el ajuste simple predijo igual o mejor que el contrafactual físico.")

# ---------------------------------------------------------------------------
# 8. Simulador
# ---------------------------------------------------------------------------

section("simulador", "8 · Construye tu lanzamiento", "sim")
st.markdown("Describe un lanzamiento tal como lo mediría el Trackman **en el ambiente que elijas**. Los modelos finales del pipeline lo evalúan "
            "en tiempo real. Los resultados son **estimaciones del modelo**, no valores observados.")

# ubicación elegida con clic en la zona (se aplica antes de dibujar los controles)
sel = st.session_state.get("zone_chart")
try:
    pts = sel["selection"]["points"] if sel else []
except (KeyError, TypeError):
    pts = []
if pts:
    gx, gz = E.location_grid()
    idx = pts[0].get("point_index")
    if idx is not None and 0 <= int(idx) < len(gx):
        p0 = (round(float(gx[int(idx)]), 2), round(float(gz[int(idx)]), 2))
    else:  # respaldo: coordenadas que manda la gráfica
        p0 = (round(float(pts[0]["x"]), 2), round(float(pts[0]["y"]), 2))
    if st.session_state.get("_last_click") != p0:
        st.session_state["_last_click"] = p0
        st.session_state["plate_x"], st.session_state["plate_z"] = p0

ctrl, zone, out = st.columns([1.0, 1.45, 0.95])
with ctrl:
    pt = st.selectbox("Tipo de lanzamiento", PITCH_TYPES, key="sim_type")
    a, b = st.columns(2)
    hand = a.radio("Mano del pitcher", ["Right", "Left"], format_func=lambda v: "derecho" if v == "Right" else "zurdo", key="sim_hand")
    bat = b.radio("Mano del bateador", ["Right", "Left"], format_func=lambda v: "derecho" if v == "Right" else "zurdo", key="sim_bat")
    env = st.select_slider("Ambiente", CATS, format_func=lambda v: CAT_LABEL[v], key="sim_env")
    a, b = st.columns(2)
    balls = a.number_input("Bolas", 0, 3, 0, key="sim_balls")
    strikes = b.number_input("Strikes", 0, 2, 0, key="sim_strikes")
    dom = E.type_domain(pt)
    vals = {}
    with st.expander("Características físicas", expanded=True):
        st.caption(f"Límites = percentiles 0.5–99.5 de los {pt} reales; valor inicial = mediana.")
        steps = {"velo": 0.5, "ivb": 0.5, "hb_arm": 0.5, "spin_rate": 25.0, "extension": 0.1, "rel_height": 0.1, "rel_side_arm": 0.1}
        for k, col in PHYS_INPUTS.items():
            p = dom["percentiles"][col]
            lo, hi, med = float(p["0.5"]), float(p["99.5"]), float(p["50"])
            stp = steps[k]
            lo, hi, med = np.floor(lo / stp) * stp, np.ceil(hi / stp) * stp, round(med / stp) * stp
            vals[k] = st.slider(LABELS[k], float(lo), float(hi), float(med), float(stp), key=f"{k}_{pt}")
        auto_axis = st.checkbox("Eje de giro coherente con el movimiento", True, key=f"axis_auto_{pt}")
        implied = float(E.implied_axis(vals))
        axis = None if auto_axis else st.slider(LABELS["spin_axis"] + " (marco de un derecho)", 0.0, 359.0, round(implied), 1.0, key=f"axis_{pt}")
    st.session_state.setdefault("plate_x", 0.0)
    st.session_state.setdefault("plate_z", 2.5)
    a, b = st.columns(2)
    plate_x = a.slider("PlateLocSide (pies)", -1.8, 1.8, step=0.05, key="plate_x")
    plate_z = b.slider("PlateLocHeight (pies)", 0.6, 4.4, step=0.05, key="plate_z")

inputs = {"pitch_type": pt, "p_throws": hand, "b_side": bat, **vals, "spin_axis": axis, "plate_x": plate_x, "plate_z": plate_z,
          "balls": int(balls), "strikes": int(strikes), "env": env}
res = E.simulate(inputs)
base = E.baseline(res.inputs)
interp = E.interpret(res, base)

with zone:
    surf = E.location_surface(inputs)
    st.plotly_chart(charts.zone_picker(surf, plate_x, plate_z, bat), width="stretch", key="zone_chart", on_select="rerun",
                    selection_mode="points")
    st.caption("Haz clic en la zona para mover el lanzamiento (o usa los controles). Color = Location+ de esa ubicación para este tipo, "
               f"conteo y manos. Blanco = menos de {LOC_MIN_CELL} pitches reales de ese tipo en la celda: ahí el modelo extrapola.")

tctx = E.pitch_type_context(pt)
rctx = E.region_context(plate_x, plate_z)

with out:
    st.markdown(card("Pitching+", f"{res.pitching_plus:.0f}", f"percentil {res.percentiles['pitching_plus']:.0f} de todos los lanzamientos · "
                     f"{res.percentiles_tipo['pitching_plus']:.0f} entre los {pt}", hero=True), unsafe_allow_html=True)
    st.markdown("")
    a, b = st.columns(2)
    a.markdown(card("Stuff+ Physical", f"{res.stuff_plus:.0f}", f"percentil {res.percentiles_tipo['stuff_plus']:.0f} entre los {pt}"), unsafe_allow_html=True)
    b.markdown(card("Location+", f"{res.location_plus:.0f}", f"percentil {res.percentiles_tipo['location_plus']:.0f} entre los {pt}"), unsafe_allow_html=True)
    st.markdown("")
    a, b = st.columns(2)
    a.markdown(f'<div class="mini"><div class="label">P(whiff | swing)</div><div class="value">{res.p_whiff_swing_model:.1%}</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="mini"><div class="label">P(bola en juego)</div><div class="value">{res.p_bip:.1%}</div></div>', unsafe_allow_html=True)
    st.caption("Escala de lanzamiento: 100 = lanzamiento promedio de la liga, 10 puntos = 1 desviación estándar entre lanzamientos. "
               "Stuff+ compara contra todos los tipos (las rectas generan menos whiff por swing que los rompientes); por eso también se "
               "muestra el percentil dentro del tipo.")

# contexto histórico observado: tipo de pitch y región de la zona
c = st.columns(2)
if tctx["muestra_suficiente"]:
    w = 100 * tctx["whiff_swing_E"]
    wtxt = (f"pierde {abs(w):.1f} pp de whiff por swing en Extreme" if w < 0 and tctx["whiff_distinto_de_0"]
            else f"cambia {w:+.1f} pp de whiff por swing en Extreme (no distinguible de 0)")
    if tctx["orden_costo"] == 1:
        ctxt = f"Es el tipo con <b>menor pérdida promedio de valor</b> en Extreme ({tctx['costo_E']:+.2f} carreras/100)."
    elif tctx["orden_costo"] == tctx["n_tipos"]:
        ctxt = f"Es el tipo con <b>mayor pérdida promedio de valor</b> en Extreme ({tctx['costo_E']:+.2f} carreras/100)."
    else:
        ctxt = f"Pérdida promedio de valor en Extreme: {tctx['costo_E']:+.2f} carreras/100 ({tctx['orden_costo']}° de {tctx['n_tipos']}, de menor a mayor)."
    hot = " hot" if tctx["orden_costo"] == tctx["n_tipos"] else ""
    c[0].markdown(f'<div class="ctx{hot}"><div class="tag">Observado · tipo de pitch</div>'
                  f"Históricamente, el <b>{pt}</b> {wtxt}. {ctxt}<br><small><i>Promedio del tipo, no destino garantizado para este pitch "
                  f"específico.</i></small></div>", unsafe_allow_html=True)
else:
    c[0].markdown(f'<div class="ctx"><div class="tag">Observado · tipo de pitch</div>El <b>{pt}</b> tiene poca muestra ({tctx["pitches"]:,} pitches): '
                  "su cambio promedio en Extreme es muy incierto y no lo usamos como referencia.</div>", unsafe_allow_html=True)
o, n = rctx["orden_perdida"], rctx["n_regiones"]
where = "una de las que <b>más</b>" if o == 1 else ("la que <b>menos</b>" if o == n else "una región intermedia en cuanto al")
hot = " hot" if o == 1 else ""
c[1].markdown(f'<div class="ctx{hot}"><div class="tag">Observado · región de la zona</div><b>{REG_LABEL[rctx["region"]]}</b>: '
              f"históricamente, es {where} valor pierde en Extreme ({rctx['extreme_menos_no']:+.2f} carreras/100 frente a No Altitude).<br>"
              "<small><i>Asociación promedio. Location+ no cambia con el ambiente en el modelo; esta nota es el dato observado.</i></small></div>",
              unsafe_allow_html=True)

if res.warnings:
    st.markdown('<div class="dangerbox"><b>Cuidado con la extrapolación:</b><ul>' + "".join(f"<li>{w}</li>" for w in res.warnings) + "</ul></div>",
                unsafe_allow_html=True)
else:
    st.markdown('<div class="okbox">Este lanzamiento está dentro del rango común observado para su tipo (inputs entre los percentiles 1 y 99 y '
                'combinación dentro del percentil 99 de los lanzamientos reales).</div>', unsafe_allow_html=True)

why, more = st.tabs(["¿Por qué? Desglose de probabilidades", "Más detalles"])
with why:
    rest = res.xrv_per_100 - 100 * res.contact_damage
    st.markdown(f"""<div class="chain">
<div class="step"><div class="t">Riesgo de ponerla en juego</div><div class="v">{res.p_bip:.1%}</div><small>P(BIP)</small></div><div class="op">×</div>
<div class="step"><div class="t">Daño si entra en juego</div><div class="v">{res.e_delta_bip:+.3f}</div><small>E[RV | BIP], carreras</small></div><div class="op">=</div>
<div class="step"><div class="t">Daño esperado de contacto</div><div class="v">{100 * res.contact_damage:+.2f}</div><small>carreras por 100 pitches</small></div><div class="op">+</div>
<div class="step"><div class="t">Resto del pitch</div><div class="v">{rest:+.2f}</div><small>bolas, strikes, whiffs, fouls (por 100)</small></div><div class="op">=</div>
<div class="step"><div class="t">Valor esperado del pitch</div><div class="v">{res.xrv_per_100:+.2f}</div><small>E[Δ] por 100 (+ = carreras para la ofensiva)</small></div><div class="op">→</div>
<div class="step final"><div class="t">Pitching+</div><div class="v">{res.pitching_plus:.0f}</div><small>menos carreras esperadas = más alto</small></div>
</div>""", unsafe_allow_html=True)
    a, b, cc, d = st.columns(4)
    a.markdown(stat(f"{res.p_swing:.1%}", "P(swing)"), unsafe_allow_html=True)
    b.markdown(stat(f"{res.p_whiff_swing_model:.1%}", f"P(whiff | swing) · Stuff+ Physical: {res.p_whiff_swing_stuff:.1%}"), unsafe_allow_html=True)
    cc.markdown(stat(f"{res.p_whiff:.1%}", "P(whiff) por pitch"), unsafe_allow_html=True)
    d.markdown(stat(f"{res.p_bip:.1%}", "P(bola en juego) por pitch"), unsafe_allow_html=True)
    st.caption("Dos formas de bajar el daño de contacto: bajar P(BIP) (evitar contacto) o bajar E[RV | BIP] (contacto menos dañino). La primera "
               "es una habilidad mucho más estable entre temporadas (sección 3).")
    c = st.columns([1.3, 1])
    with c[0]:
        st.markdown("**Probabilidad de cada resultado** (la marca roja es el lanzamiento mediano de su tipo en la misma ubicación y conteo)")
        st.plotly_chart(charts.prob_bars(res.probs, EVENT_LABELS, base.probs), width="stretch")
    with c[1]:
        st.markdown("**Interpretación automática**")
        if interp["fortalezas"]:
            st.markdown("**Fortalezas**\n" + "\n".join(f"- {s}" for s in interp["fortalezas"]))
        if interp["debilidades"]:
            st.markdown("**Debilidades**\n" + "\n".join(f"- {s}" for s in interp["debilidades"]))
        if not interp["fortalezas"] and not interp["debilidades"]:
            st.markdown("Ningún componente se aleja claramente del lanzamiento mediano de su tipo en este contexto.")
        st.markdown(f"**Interpretación.** {interp['interpretacion']}")
        st.caption(f"Referencia: lanzamiento mediano de su tipo en la misma ubicación, conteo, manos y ambiente (Pitching+ {base.pitching_plus:.0f}).")

with more:
    st.markdown(f"**Historia del {pt} en altitud** " + badges("obs"), unsafe_allow_html=True)
    st.plotly_chart(charts.pitch_type_detail(E.type_effects.loc[pt]), width="stretch", key="sim_type_detail")
    st.caption("Promedio observado del tipo (efectos fijos de pitcher, bateador, temporada y conteo), con IC 95%. Detalle en la sección 6.")
    st.markdown("**¿Qué podría mejorar este lanzamiento?** " + badges("sim"), unsafe_allow_html=True)
    st.markdown("Cambios pequeños, **uno a la vez**, de ½ desviación estándar del tipo, que no salen del rango 5–95% observado, o "
                "movimientos de 3 pulgadas hacia ubicaciones con suficientes pitches reales. Es lo que estima el modelo, no una receta: en la "
                "realidad las variables físicas no se mueven por separado.")
    imp = E.improvements(inputs)
    if imp.empty:
        st.info("No hay cambios dentro del rango observado para comparar.")
    else:
        show = imp.rename(columns={"cambio": "cambio", "pitching_plus": "Pitching+", "delta_pitching_plus": "Δ Pitching+", "stuff_plus": "Stuff+",
                                   "location_plus": "Location+"})
        st.dataframe(show.style.format({"Pitching+": "{:.1f}", "Δ Pitching+": "{:+.1f}", "Stuff+": "{:.1f}", "Location+": "{:.1f}"})
                     .background_gradient(subset=["Δ Pitching+"], cmap="RdBu", vmin=-5, vmax=5), width="stretch", hide_index=True)
        st.caption("Diferencias menores a ~1 punto están dentro del ruido del modelo (los árboles responden de forma escalonada a cambios "
                   "pequeños); no conviene leerlas como mejoras reales.")
    st.markdown("**Rango observado de cada input (dominio de entrenamiento)**")
    dc = E.domain_check(res.inputs)
    st.dataframe(dc["tabla"].style.format({"valor": "{:.2f}", "percentil": "{:.0f}", "p5": "{:.2f}", "p95": "{:.2f}"}), width="stretch", hide_index=True)
    st.caption(f"Distancia de Mahalanobis del conjunto: {dc['mahalanobis']:.2f} (percentil 99 de los lanzamientos reales: {dc['mahalanobis_p99']:.2f}). "
               f"Pitches reales del tipo en esa celda de ubicación: {dc['pitches_en_celda']}.")

# ---------------------------------------------------------------------------
# 9. Pitching+ general vs. para ambientes de altitud
# ---------------------------------------------------------------------------

section("generalvsalt", "9 · ¿Qué cambia en nuestro Pitching+? General vs. ambientes de altitud", "sim", "obs")
EV = D["envv"]
CBL = EV["comparacion_lanzamientos"]
RG = EV["ganancia_relativa_mismo_conteo"]["correlaciones"]
st.markdown('<div class="bigidea"><b>Stuff+</b> = física. <b>Location+</b> = ejecución. <b>Pitching+</b> = valor total. El mismo lanzamiento '
            "(misma física, misma ubicación, mismo conteo) puede valer distinto si cambia <b>lo que cuesta cada resultado</b>.</div>",
            unsafe_allow_html=True)
st.markdown("La sección 11 cambia la **física** del pitch (cómo se mueve en otro aire). Aquí el pitch es **exactamente el mismo**: las "
            "probabilidades del modelo no cambian; cambia el valor en carreras de cada resultado.")
gva = E.general_vs_altitude(res)
g_, a_ = gva["general"], gva["alt"]
c = st.columns([1.15, 1])
with c[0]:
    st.markdown(f"**Tu lanzamiento** ({pt}, conteo {int(balls)}-{int(strikes)})")
    tb = pd.DataFrame({
        "métrica": ["Stuff+ Physical", "Location+", "P(whiff) por pitch", "P(bola en juego)", "E[RV | BIP]: run value de una bola en juego",
                    "Daño de contacto frente a seguir el turno (por 100)", "Valor esperado E[Δ] (por 100)", "Pitching+"],
        "General (liga)": [f"{res.stuff_plus:.0f}", f"{res.location_plus:.0f}", f"{gva['p_whiff']:.1%}", f"{gva['p_bip']:.1%}",
                           f"{g_['e_rv_bip']:+.3f}", f"{100 * g_['dano_contacto']:+.2f}", f"{100 * g_['xrv']:+.2f}", f"{g_['pitching_plus']:.1f}"],
        gva["alt_label"]: [f"{res.stuff_plus:.0f}", f"{res.location_plus:.0f}", f"{gva['p_whiff']:.1%}", f"{gva['p_bip']:.1%}",
                           f"{a_['e_rv_bip']:+.3f}", f"{100 * a_['dano_contacto']:+.2f}", f"{100 * a_['xrv']:+.2f}", f"{a_['pitching_plus']:.1f}"]})
    st.dataframe(tb, width="stretch", hide_index=True)
    st.caption("Stuff+, Location+ y las probabilidades no cambian: es el mismo lanzamiento. Cambia lo que vale cada resultado. Una bola en juego "
               "vale más para la ofensiva en Extreme (E[RV | BIP] más alto), pero el turno completo también es más caro. Aquí Pitching+ se compara "
               "contra todos los lanzamientos de la liga sin el ajuste por ambiente del simulador, por eso puede diferir ≈ 1 punto del de arriba.")
with c[1]:
    st.markdown(card(f"Pitching+ General → {gva['alt_label']}", f"{g_['pitching_plus']:.1f} → {a_['pitching_plus']:.1f}",
                     f"frente al típico de este conteo: {gva['cambio_relativo_100']:+.2f} carreras por 100 pitches", hero=True),
                unsafe_allow_html=True)
    st.markdown(f"**¿Por qué cambió?** {gva['explicacion']}")
    st.plotly_chart(charts.contribution_bars(gva["contribuciones"]), width="stretch", key="contrib_chart")

st.markdown("#### ¿Qué lanzamientos ganan valor relativo en Extreme? " + badges("obs", "sim"), unsafe_allow_html=True)
c = st.columns([1.4, 1])
c[0].plotly_chart(charts.relative_gain_bars(D["relgain"]), width="stretch")
with c[1]:
    st.markdown(f"- En toda la liga, el orden de los lanzamientos por Pitching+ general y por Pitching+ Extreme correlaciona "
                f"**{CBL['spearman_general_extreme']:.3f}**; el cambio mediano es de {CBL['cambio_abs_mediana']:.2f} puntos.\n"
                f"- **Con dos strikes**, donde el swing fallado es ponche, ganan los lanzamientos de swing-and-miss (correlación con P(whiff) "
                f"{RG['dos strikes']['whiff_vs_general']:+.2f}).\n"
                f"- **Con 0 o 1 strike**, evitar el contacto casi no gana valor ({RG['0 o 1 strike']['whiff_vs_general']:+.2f}): si no hay contacto, "
                "el turno sigue en un ambiente donde todo cuesta más.")
    st.markdown('<div class="okbox">Coherente con P1: cambiar una bola en juego por un <b>ponche</b> vale ≈ '
                f'{pct(K_UP, 0)} más en Extreme. En el valor por lanzamiento eso aparece donde el whiff <b>termina</b> el turno.</div>',
                unsafe_allow_html=True)

st.markdown("#### Dos lanzamientos con el mismo Stuff+ pueden valer distinto " + badges("sim"), unsafe_allow_html=True)
surf_ok = surf[surf["pitches_en_celda"] >= LOC_MIN_CELL]
best_loc = surf_ok.loc[surf_ok["location_plus"].idxmax()] if len(surf_ok) else None
pair = [("En el corazón de la zona", 0.0, 2.5)] + ([("En la mejor ubicación con soporte para este tipo y conteo", float(best_loc["plate_x"]),
                                                   float(best_loc["plate_z"]))] if best_loc is not None else [])
c = st.columns(len(pair))
for col, (lab, px, pz) in zip(c, pair):
    rr = E.simulate({**inputs, "plate_x": px, "plate_z": pz})
    col.markdown(card(lab, f"Pitching+ {rr.pitching_plus:.0f}", f"Stuff+ {rr.stuff_plus:.0f} · Location+ {rr.location_plus:.0f} · "
                      f"P(BIP) {rr.p_bip:.1%} · ubicación ({px:+.2f}, {pz:.2f}) pies"), unsafe_allow_html=True)
st.caption("Misma física (tu lanzamiento), dos ubicaciones: Stuff+ casi no cambia; Location+ y Pitching+ sí. Pitching+ no es Stuff+ + Location+: "
           "combina física, ubicación y contexto en el valor esperado.")

with st.expander("Cómo se calcula (qué vale cada resultado en cada ambiente)"):
    C_ = EV["conjuntos"]
    tv = pd.DataFrame({"resultado": ["base por bolas", "ponche", "golpeado", "out en juego", "sencillo", "doble", "triple", "jonrón"],
                       **{C_[k]["etiqueta"]: [C_[k]["terminales"][e] for e in ("walk", "strikeout", "hbp", "out", "single", "double", "triple", "home_run")]
                          for k in ("general", "no_altitude", "extreme")}})
    st.dataframe(tv.style.format({c_: "{:+.3f}" for c_ in tv.columns if c_ != "resultado"}), width="stretch", hide_index=True)
    st.markdown(f"- **General** = los valores de toda la liga: exactamente los del Pitching+ oficial (diferencia máxima "
                f"{EV['verificacion']['max_dif_xrv_general_vs_oficial']:.0e}).\n"
                "- **Extreme** = la misma recursión RE24 por conteo con los pitches de Extreme: valores de cada resultado y de cada conteo "
                f"(V(0-0) = {C_['extreme']['V']['0-0']:+.3f} contra {C_['general']['V']['0-0']:+.3f} en la liga).\n"
                "- **Calibración del costo de contacto:** el modelo no ve el ambiente y subestima el daño por bola en juego en altitud (P2). "
                f"A la rama de bola en juego se le suma lo que falta según lo observado en ese ambiente ({C_['extreme']['shift_bip']:+.4f} "
                "carreras en Extreme). No hay ningún bono manual.")
levels(
    "Imagina el mismo lanzamiento en dos ligas: en una, cada hit cuesta más y cada turno también. Pitching+ para altitud vuelve a calcular "
    "cuánto vale el lanzamiento con los precios de ese ambiente.",
    "Con dos strikes, un pitch que hace fallar swings vale más en altitud porque el ponche evita un contacto más caro. Antes de dos strikes, "
    "un pitch que no genera contacto solo lleva a otro conteo, y en altitud el resto del turno también es más caro.",
    "Mismas P(evento) del modelo final; Δ(evento, conteo) de la recursión RE24 con los pitches de cada ambiente (valores terminales y "
    "frecuencias de transición) más la calibración de la rama BIP (Δ observado − predicho). Escala 100 + 10·(media − E[Δ])/sd sobre todos "
    "los lanzamientos reales. Detalle: `reports/pitching_plus_general_vs_altitude.md`.")

# ---------------------------------------------------------------------------
# 10. ¿Es el mismo un buen lanzamiento en cualquier estadio?
# ---------------------------------------------------------------------------

section("optimo", "10 · ¿Es el mismo un buen lanzamiento en cualquier estadio?", "sim", "expl")
OP, OC, OL, OH, OM = D["opt_prof"], D["opt_cmp"], D["opt_loc"], D["opt_heat"], D["opt_meta"]
oc_e, oc_f = OC[OC["alterno"] == "extreme"], OC[OC["alterno"] == "extreme_fisica"]
n_poco_e, n_poco_f = int((oc_e["clase"] == "Poco").sum()), int((oc_f["clase"] == "Poco").sum())
if n_poco_e == len(oc_e):
    headline = "Un gran lanzamiento sigue siendo un gran lanzamiento."
elif n_poco_e >= len(oc_e) / 2:
    headline = "La definición de un buen lanzamiento es estable, pero el ambiente cambia el margen óptimo en algunos tipos."
else:
    headline = "El lanzamiento ideal cambia de forma apreciable entre ambientes."
st.markdown(f'<div class="bigidea"><b>{headline}</b><br><small>Con los costos de Extreme, el perfil óptimo cambia menos que el ruido de muestreo '
            f"en {n_poco_e} de {len(oc_e)} combinaciones tipo × conteo (correlación de rangos ≥ {oc_e['spearman'].min():.3f}). Con la física "
            f"aproximada del aire de Extreme, en {n_poco_f} de {len(oc_f)}.</small></div>", unsafe_allow_html=True)
st.markdown("**Cómo se buscó:** entre lanzamientos **reales** de cada tipo (no se inventan combinaciones imposibles), se buscan los que "
            "maximizan Stuff+ Physical, Pitching+ general y Pitching+ Extreme en contextos fijos. El óptimo se reporta como el **perfil del top 1%** "
            "(mediana y rango) y como el **mejor lanzamiento observado**, con incertidumbre por bootstrap de pitchers.")
c = st.columns([1, 1, 1.3])
opt_t = c[0].selectbox("Tipo de lanzamiento", PITCH_TYPES, index=PITCH_TYPES.index("Slider"), key="opt_type")
opt_ctx = c[1].radio("Conteo", list(OM["contextos"]), format_func=lambda k: OM["contextos"][k], key="opt_ctx")
layer = c[2].radio("Ambiente para \"Extreme\"", ["extreme", "extreme_fisica"], key="opt_layer",
                   format_func=lambda k: "Costos de Extreme (validado)" if k == "extreme" else "Costos + física aproximada del aire")


def _p(obj, ctx):
    return OP[(OP["tipo"] == opt_t) & (OP["objetivo"] == obj) & (OP["contexto"] == ctx)].iloc[0]


pg, pa, ps = _p("general", opt_ctx), _p(layer, opt_ctx), _p("stuff", "todos")
cm = OC[(OC["tipo"] == opt_t) & (OC["alterno"] == layer) & (OC["contexto"] == opt_ctx)].iloc[0]
units = {"RelSpeed": ("velocidad", "mph", 1), "InducedVertBreak": ("IVB", "pulg.", 1), "HorzBreak": ("HB (lado del brazo)", "pulg.", 1),
         "SpinRate": ("spin", "rpm", 0), "Extension": ("extensión", "pies", 2)}
tip = OM["tipos"][opt_t]
c = st.columns(3)
for col, (lab, pr, extra) in zip(c, [("Mejor versión General", pg, ""), (f"Mejor versión {'Extreme' if layer == 'extreme' else 'Extreme (física aprox.)'}", pa, ""),
                                      ("Mejor Stuff+ (para comparar)", ps, "")]):
    rows = "".join(f'<div class="row"><span>{u[0]}</span><b>{pr[k + "_med"]:.{u[2]}f} {u[1]}</b></div>' for k, u in units.items())
    rng = "".join(f'<div class="row"><span>rango (p10–p90)</span><b>{pr[k + "_p10"]:.{u[2]}f}–{pr[k + "_p90"]:.{u[2]}f}</b></div>'
                  for k, u in list(units.items())[:1])
    score = (f'<div class="row"><span>Stuff+ (media del top 1%)</span><b>{pr["top_stuff_plus"]:.0f}</b></div>'
             + (f'<div class="row"><span>Location+ (ubicación típica)</span><b>{pr["location_plus_tipica"]:.0f}</b></div>'
                f'<div class="row"><span>Pitching+ (vecinos reales)</span><b>{pr["puntaje_perfil_vecinos"]:.1f}</b></div>'
                f'<div class="row"><span>mejor observado</span><b>{pr["puntaje_mejor_observado"]:.1f} '
                f'{"(estable)" if pr["mejor_estable"] else "(inestable)"}</b></div>' if pr["objetivo"] != "stuff" else
                f'<div class="row"><span>Stuff+ (vecinos reales)</span><b>{pr["puntaje_perfil_vecinos"]:.1f}</b></div>'))
    probs_ = (f'<div class="row"><span>P(whiff) · P(BIP)</span><b>{pr["top_p_whiff"]:.1%} · {pr["top_p_bip"]:.1%}</b></div>'
              if pd.notna(pr.get("top_p_whiff")) else "")
    col.markdown(f'<div class="rkcard"><div class="pid">{lab}</div>{rows}{rng}{score}{probs_}</div>', unsafe_allow_html=True)
loc_txt = ", ".join(f"{'mismo lado' if s_ == 'same' else 'lado contrario'} ({v[0]:+.2f}, {v[1]:.2f}) pies" for s_, v in tip["ubicacion_tipica"].items())
st.caption(f"Experimento 1 (física sola): ubicación fija en la ubicación típica del tipo — {loc_txt}. {tip['n_candidatos']:,} candidatos "
           f"reales de {tip['n_pitchers']} pitchers. Pitching+ en escala de lanzamiento.")

for pr_ in (pg, pa):
    if isinstance(pr_.get("frontera"), str):
        st.markdown(f'<div class="warnbox">El perfil {pr_["objetivo"]} queda en la <b>frontera</b> de los datos (fuera de p1–p99 de los '
                    f'candidatos en {pr_["frontera"].replace(";", ", ")}). Es real, pero poco común: leerlo con cautela.</div>', unsafe_allow_html=True)
clase = cm["clase"]
st.markdown(f'<div class="{"okbox" if clase == "Poco" else "warnbox"}"><b>¿Cambió? {clase}.</b> Distancia entre los perfiles óptimos: '
            f'{cm["distancia_z"]:.2f} desviaciones estándar, contra un ruido de muestreo de {cm["ruido_q95"]:.2f} (q95 del bootstrap por pitcher). '
            f'La variable que más se mueve es <b>{units.get(cm["variable_que_mas_cambia"], (cm["variable_que_mas_cambia"],))[0]}</b> '
            f'({cm["z_" + cm["variable_que_mas_cambia"]]:+.2f} sd). El orden de todos los candidatos correlaciona {cm["spearman"]:.3f}; '
            f'comparten {100 * cm["jaccard_top1"]:.0f}% del top 1%; el mejor {opt_t} general queda en el percentil '
            f'{cm["percentil_en_alt_del_mejor_general"]:.2f} con este criterio.</div>', unsafe_allow_html=True)
c = st.columns([1, 1.3])
pct_rows = {"General": {k: pg[k + "_pct"] for k in units}, ("Extreme" if layer == "extreme" else "Extreme (física aprox.)"): {k: pa[k + "_pct"] for k in units},
            "Mejor Stuff+": {k: ps[k + "_pct"] for k in units}}
c[0].plotly_chart(charts.profile_radar(pct_rows, {k: u[0] for k, u in units.items()}), width="stretch", key="opt_radar")
with c[1]:
    st.markdown("**Ubicación: ¿la misma sigue siendo óptima?** (experimento 2, lanzamiento fijo con muy buen Stuff+)")
    side = st.radio("Enfrentamiento", ["same", "opposite"], horizontal=True, key="opt_side",
                    format_func=lambda s_: "mismo lado" if s_ == "same" else "lado contrario")
    hh = OH[(OH["tipo"] == opt_t) & (OH["enfrentamiento"] == side)]
    if len(hh):
        st.plotly_chart(charts.location_pair(hh, "Pitching+ general", "Pitching+ Extreme" + (" (física aprox.)" if layer != "extreme" else ""),
                                             "pp_extreme" if layer == "extreme" else "pp_extreme_fisica"), width="stretch", key="opt_heat_chart")
        st.caption("Conteo 0-0. Estrella = mejor ubicación. Solo celdas con al menos "
                   f"{OM['loc_min']} lanzamientos reales del tipo en ese enfrentamiento. Marco de un derecho.")
    else:
        st.info("No hay suficientes celdas con soporte para este tipo y enfrentamiento.")

lc = OL[(OL["tipo"] == opt_t) & (OL["contexto"] == "0-0") & (OL["enfrentamiento"] == side)].set_index("objetivo")
if len(lc):
    lg, la = lc.loc["general"], lc.loc[layer]
    fg = str(lg.get("fijo_mejor_celda_general", ""))
    fa = str(lg.get("fijo_mejor_celda_extreme" if layer == "extreme" else "fijo_mejor_celda_extreme_fisica", ""))
    st.markdown(f"**Mejor combinación física + ubicación (0-0):** General ({lg['x']:+.1f}, {lg['z']:.1f}) con Pitching+ {lg['pitching_plus']:.1f}; "
                f"{'Extreme' if layer == 'extreme' else 'Extreme con física aprox.'} ({la['x']:+.1f}, {la['z']:.1f}) con Pitching+ {la['pitching_plus']:.1f}. "
                f"Orden de todas las combinaciones: correlación {lg['spearman_combos']:.3f}. Para el lanzamiento fijo, la mejor celda "
                f"{'es la misma' if fg == fa else 'cambia'} ({fg} → {fa}).")
st.markdown(f"**Mejor Stuff ≠ mejor pitch.** Para el {opt_t}, el orden por Stuff+ y por Pitching+ general correlaciona "
            f"{cm['stuff_vs_general_spearman']:.2f} y sus top 1% comparten {100 * cm['jaccard_stuff_general']:.0f}%. Esa diferencia es mucho mayor "
            "que la que produce el ambiente: lo que hace fallar swings no es lo mismo que lo que más vale en cada conteo.")
with st.expander("Resultado por tipo de lanzamiento y conteo"):
    gt = OC[OC["alterno"].isin(["extreme", "extreme_fisica"])].copy()
    gt["capa"] = gt["alterno"].map({"extreme": "costos", "extreme_fisica": "física aprox."})
    gt = gt.pivot_table(index=["tipo", "contexto"], columns="capa", values=["clase", "spearman", "jaccard_top1"], aggfunc="first")
    gt.columns = [f"{a} · {b}" for a, b in gt.columns]
    st.dataframe(gt.reset_index(), width="stretch", hide_index=True)
    st.caption(OM["clases"])
st.markdown('<div class="warnbox"><b>No es una receta.</b> El óptimo del modelo es una combinación de características favorable según los '
            "datos; no considera biomecánica, salud, tunneling, secuencias, repertorio ni fatiga. No significa \"todo pitcher debe lanzar este "
            "pitch\".</div>", unsafe_allow_html=True)
levels(
    "Buscamos, entre lanzamientos reales, cuál sería la mejor versión de cada tipo con los precios de la liga y con los de Extreme, y "
    "comparamos. Si casi no cambia, un buen lanzamiento es bueno en cualquier lado.",
    "El ambiente cambia el margen, no la definición: lo que hace bueno a un slider o a una recta es casi lo mismo en cualquier estadio. Lo "
    "que sí cambia mucho es la pregunta: \"¿qué hace fallar swings?\" no es lo mismo que \"¿qué vale más en este conteo?\".",
    "Candidatos = lanzamientos reales de No Altitude (hasta 4,000 por tipo). Objetivos: Stuff+ Physical, Pitching+ con valores de la liga, "
    "Pitching+ con valores de Extreme y con física re-simulada en Extreme. Perfil = mediana del top 1%; puntaje con 25 vecinos reales; "
    "bootstrap por pitcher (200); distancia en sd del tipo. Detalle: `reports/optimal_pitch_by_environment.md`.")

# ---------------------------------------------------------------------------
# 11. Comparar ambientes
# ---------------------------------------------------------------------------

section("ambientes", "11 · Compara ambientes: el mismo lanzamiento en tres ambientes", "sim")
SH = S["contrafactual_split_half"]
st.markdown(f'<div class="warnbox"><b>Simulación contrafactual del modelo · simulación física aproximada / contrafactual explicativo.</b> '
            f'El modelo físico sobreestima ≈ {pct(SH["sobreprediccion_ivb"], 0)} el cambio de IVB (validación split-half, sección 14). Se toma el lanzamiento que construiste, se reconstruye su trayectoria y '
            "se vuela de nuevo con el drag y el Magnus medidos en cada categoría. Es una estimación para un ambiente de altitud según el modelo, "
            "no un resultado observado ni el valor en un estadio real. El modelo solo ve el cambio físico del pitch: no incluye el vuelo más largo "
            "de la pelota bateada, así que subestima el daño de contacto en altitud.</div>", unsafe_allow_html=True)
be = res.by_env.copy()
st.plotly_chart(charts.env_compare(be), width="stretch")
tbl = be[["ambiente", "pitching_plus", "stuff_plus", "location_plus", "xrv_per_100", "p_whiff", "p_whiff_swing_model", "p_bip", "ivb", "hb_arm", "vaa"]].copy()
tbl["ambiente"] = tbl["ambiente"].map(CAT_LABEL)
tbl.columns = ["ambiente", "Pitching+ (vs. su ambiente)", "Stuff+ (vs. su ambiente)", "Location+", "E[Δ] por 100", "P(whiff)", "P(whiff | swing)",
               "P(BIP)", "IVB", "HB (lado del brazo)", "VAA"]
st.dataframe(tbl.style.format({c: "{:.1f}" for c in tbl.columns if "+" in c} | {"E[Δ] por 100": "{:+.2f}", "P(whiff)": "{:.1%}", "P(whiff | swing)": "{:.1%}",
                                                                               "P(BIP)": "{:.1%}", "IVB": "{:.1f}", "HB (lado del brazo)": "{:.1f}", "VAA": "{:.2f}"}),
             width="stretch", hide_index=True)
st.caption("Cómo leerlo: **E[Δ]** es el valor en carreras del mismo pitch en cada ambiente (cuánto cambia). Stuff+ y Pitching+ comparan contra el "
           "lanzamiento típico **de ese mismo ambiente** (ajuste simple), así que casi no cambian: en altitud todos los pitches pierden movimiento.")

# ---------------------------------------------------------------------------
# 12. Ranking
# ---------------------------------------------------------------------------

section("ranking", "12 · Pitchers con perfil más favorable para ambientes de altitud", "pred")
st.markdown(f"**AEPP** (Altitude Environment Pitcher Profile) = carreras evitadas por 100 turnos frente a un pitcher promedio, usando lo que vale "
            f"cada resultado (ponche, base por bolas, golpeado, bola en juego) en los ambientes de altitud. **Los pesos no se eligieron a mano**: "
            f"son los valores en carreras medidos en esos ambientes.")
c = st.columns(4)
c[0].markdown(stat(f"{AE['r_altitud_t1']:.2f}", "AEPP vs. carreras en altitud del año siguiente (r)"), unsafe_allow_html=True)
c[1].markdown(stat(f"{AE['r_pitching_plus_altitud_t1']:.2f}", f"Pitching+ con el mismo criterio (P(AEPP mejor) = {pct(AE['p_mejor_que_pitching_plus'], 0)})"), unsafe_allow_html=True)
c[2].markdown(stat(f"{AE['r_kbb_altitud_t1']:.2f}", "K−BB% con shrinkage: AEPP NO lo supera"), unsafe_allow_html=True)
c[3].markdown(stat(f"{AE['rho_pesos_altitud_vs_no']:.3f}", "correlación de rankings con pesos de altitud vs. sin altitud"), unsafe_allow_html=True)
st.markdown(f'<div class="bigidea">La altitud <b>no crea un tipo completamente diferente de buen pitcher</b>. Aumenta ligeramente el valor de '
            f'habilidades que ya son buenas, especialmente <b>generar ponches y evitar contacto</b> (el "bono de altitud" correlaciona '
            f'{AE["r_bono_k"]:.2f} con el K%).</div>', unsafe_allow_html=True)

prof = D["prof"].copy()
RT = S["ranking_top"]
st.markdown("### El ranking tiene mucha incertidumbre")
c = st.columns(3)
c[0].markdown(stat(f"{AE['ancho_mediano_rango_80']:.0f}", f"posiciones: amplitud mediana del intervalo de 80% del lugar (de {AE['n_elegibles']} pitchers)", "pos"),
              unsafe_allow_html=True)
c[1].markdown(stat(f"{RT['lugar_1_intervalo'][0]:.0f}–{RT['lugar_1_intervalo'][1]:.0f}", "intervalo del lugar del #1: la cima está mejor identificada"),
              unsafe_allow_html=True)
c[2].markdown(stat(f"{RT['n_p_top25_mayor_50']}", "pitchers con P(top 25) mayor a 50%"), unsafe_allow_html=True)
st.markdown("Por eso no conviene leer una posición rígida (\"#37\"). **P(top 25)** —la proporción de simulaciones de la incertidumbre en que el "
            "pitcher queda entre los 25 primeros— comunica mejor qué tan seguro es su lugar. Muchos pitchers del centro son prácticamente intercambiables.")
top = prof.sort_values("rank").head(4)
c = st.columns(4)
for col, (_, q) in zip(c, top.iterrows()):
    col.markdown(f"""<div class="rkcard"><div class="pid">{q['pitcher_anon_id']}</div>
<div class="row"><span>AEPP</span><b>{q['aepp']:+.1f}</b></div>
<div class="row"><span>Ranking estimado</span><b>{q['rank_lo80']:.0f}–{q['rank_hi80']:.0f}</b></div>
<div class="row"><span>Probabilidad top 25</span><b>{q['p_top25']:.0%}</b></div>
<div class="row"><span>Stuff+</span><b>{q['stuff_plus_physical']:.0f}</b></div>
<div class="row"><span>Location+</span><b>{q['location_plus']:.0f}</b></div></div>""", unsafe_allow_html=True)
st.caption(f"Los cuatro AEPP más altos. Ranking estimado = intervalo de 80% del lugar ({RT['simulaciones']:,} simulaciones de la incertidumbre "
           "de cada métrica).")

f1, f2, f3 = st.columns([1, 1, 1.4])
min_p = f1.slider("Mínimo de pitches", int(AE["min_pitches"]), 3000, int(AE["min_pitches"]), 100, key="rk_min")
min_alt = f2.slider("Mínimo de pitches en Medium + Extreme", 0, 1500, 0, 50, key="rk_alt")
query = f3.text_input("Buscar pitcher (ID anónimo)", "", key="rk_q")
view = prof[(prof["n_pitches"] >= min_p) & (prof["pitches_medium"] + prof["pitches_extreme"] >= min_alt)]
if query:
    view = view[view["pitcher_anon_id"].str.contains(query.strip(), case=False)]
tv = pd.DataFrame({
    "pitcher": view["pitcher_anon_id"], "AEPP": view["aepp"],
    "intervalo AEPP (80%)": [f"{a:+.1f} a {b:+.1f}" for a, b in zip(view["aepp_lo80"], view["aepp_hi80"])],
    "lugar estimado": view["rank"], "intervalo del lugar (80%)": [f"{a:.0f}–{b:.0f}" for a, b in zip(view["rank_lo80"], view["rank_hi80"])],
    "P(top 25)": view["p_top25"], "pitches": view["n_pitches"], "K%": view["k_pct"], "BB%": view["bb_pct"], "Whiff/Swing": view["whiff_swing"],
    "Stuff+": view["stuff_plus_physical"], "Location+": view["location_plus"], "Pitching+": view["pitching_plus"],
    "en Extreme": view["pitches_extreme"], "en Medium": view["pitches_medium"]})
st.dataframe(tv.style.format({"lugar estimado": "{:.0f}", "pitches": "{:,.0f}", "en Extreme": "{:,.0f}", "en Medium": "{:,.0f}", "AEPP": "{:+.2f}",
                              "P(top 25)": "{:.0%}", "Pitching+": "{:.0f}", "Stuff+": "{:.0f}", "Location+": "{:.0f}", "Whiff/Swing": "{:.1%}",
                              "K%": "{:.1%}", "BB%": "{:.1%}"})
             .bar(subset=["P(top 25)"], color="#b7d3f2", vmin=0, vmax=1).bar(subset=["AEPP"], color="#f4b6be", align="zero"),
             width="stretch", hide_index=True, height=380)
st.caption(f"{len(view)} pitchers. Métricas con shrinkage y ajustadas por entorno; Pitching+, Stuff+ y Location+ en **escala de pitcher** (todas sus "
           f"temporadas). El intervalo de 80% del lugar tiene una amplitud mediana de {AE['ancho_mediano_rango_80']:.0f} posiciones: la cima está bien "
           "separada, el centro de la tabla es casi intercambiable.")
st.plotly_chart(charts.ranking_forest(prof.sort_values("rank").head(25)), width="stretch")

st.markdown("### Ficha del pitcher")
opts = view.sort_values("rank")["pitcher_anon_id"].tolist() or prof.sort_values("rank")["pitcher_anon_id"].tolist()
pid = st.selectbox("Pitcher", opts, key="pitcher_sel")
p = prof.set_index("pitcher_anon_id").loc[pid]
c = st.columns(5)
c[0].markdown(card("AEPP", f"{p['aepp']:+.2f}", f"80%: {p['aepp_lo80']:+.2f} a {p['aepp_hi80']:+.2f} · lugar #{int(p['rank'])} ({int(p['rank_lo80'])}–{int(p['rank_hi80'])})", hero=True),
              unsafe_allow_html=True)
c[1].markdown(card("Pitching+", f"{p['pitching_plus']:.0f}", f"80%: {p['pitching_plus_lo80']:.0f}–{p['pitching_plus_hi80']:.0f}"), unsafe_allow_html=True)
c[2].markdown(card("Stuff+ Physical", f"{p['stuff_plus_physical']:.0f}", f"80%: {p['stuff_plus_physical_lo80']:.0f}–{p['stuff_plus_physical_hi80']:.0f}"), unsafe_allow_html=True)
c[3].markdown(card("Location+", f"{p['location_plus']:.0f}", f"80%: {p['location_plus_lo80']:.0f}–{p['location_plus_hi80']:.0f}"), unsafe_allow_html=True)
c[4].markdown(card("Muestra", f"{int(p['n_pitches']):,}", f"{int(p['n_pa']):,} turnos · Extreme {int(p['pitches_extreme']):,} · Medium {int(p['pitches_medium']):,} · temporadas {p['temporadas']}"),
              unsafe_allow_html=True)
c = st.columns(6)
for col, (m, lab) in zip(c, [("whiff_swing", "Whiff/Swing"), ("k_pct", "K%"), ("bip_pct", "BIP%"), ("bb_pct", "BB%"), ("hbp_pct", "HBP%"), ("hard_pct", "Hard contact%")]):
    col.markdown(card(lab, pct(p[m]), f"80%: {pct(p[m + '_lo80'])}–{pct(p[m + '_hi80'])} · crudo {pct(p[m + '_crudo'])}"), unsafe_allow_html=True)
st.markdown(f"**¿Por qué aparece {'alto' if p['rank'] <= len(prof) / 2 else 'bajo'}?** {p['explicacion']}")
if any(k in p["explicacion"] for k in ("contacto que permite", "contacto duro", "rodados")):
    st.markdown(f'<div class="warnbox"><b>Ojo con las frases sobre el contacto</b> ("el contacto que permite hace poco/mucho daño", "contacto duro", '
                f'"rodados"): describen lo que pasó, no una habilidad comprobada. Son métricas poco estables (r año a año: run value/BIP '
                f'{STAB["rv_bip"]["estabilidad_cruda"]:+.2f}, hard contact {STAB["hard_pct"]["estabilidad_cruda"]:.2f}, HR/BIP '
                f'{STAB["hr_bip"]["estabilidad_cruda"]:.2f}) y pueden reflejar ruido, defensa, estadio o muestra.</div>', unsafe_allow_html=True)
mx = D["mix"][D["mix"]["pitcher_anon_id"] == pid]
c = st.columns([1, 2])
c[0].plotly_chart(charts.mix_donut(mx), width="stretch")
c[1].plotly_chart(charts.pitcher_type_bars(mx), width="stretch")
ok = mx[mx["muestra_suficiente"]]
if len(ok):
    st.markdown(f"Mejor arma según Stuff+ (relativo a su tipo): **{ok.loc[ok['stuff_plus_tipo'].idxmax(), 'AutoPitchType']}** · mejor Location+: "
                f"**{ok.loc[ok['location_plus_tipo'].idxmax(), 'AutoPitchType']}** · mayor whiff por swing: **{ok.loc[ok['whiff_swing'].idxmax(), 'AutoPitchType']}**. "
                "Solo tipos con al menos 50 pitches.")
st.caption("Lectura correcta: su perfil presenta características que históricamente son más (o menos) favorables en estos contextos. No es una "
           "garantía de rendimiento ni una prueba de que el pitcher sea \"resistente a la altitud\".")
st.markdown(badges("lim") + f" **No hay arquetipos claros.** Al agrupar pitchers, la mejor separación tiene silueta {max(S['arquetipos']['silueta'].values()):.2f} "
            "(se exigía ≥ 0.25). En lugar de grupos como \"Power Ace\" o \"Contact Manager\" hay un **continuo** entre swing-and-miss y control, así que "
            "no se inventaron categorías.", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 13. Sensibilidad
# ---------------------------------------------------------------------------

section("sensibilidad", "13 · Sensibilidad observada a la altitud — análisis exploratorio", "expl", "lim")
st.markdown(f'<div class="dangerbox"><b>No es un ranking de pitchers "resistentes".</b> Se intentó medir qué pitchers rinden relativamente mejor '
            f"bajo altitud, pero la métrica presenta demasiada variabilidad para considerarla una habilidad estable.<ul>"
            f"<li>Confiabilidad ≈ <b>{SE['confiabilidad']:.2f}</b>.</li>"
            f"<li>Para llegar a 0.5 harían falta ≈ <b>{SE['pitches_para_0_5']:,.0f}</b> pitches en Extreme por pitcher; la mediana tiene {SE['mediana_pitches_extreme']:.0f}.</li>"
            f"<li>Con shrinkage, <b>{SE['intervalos_cruzan_0']} de {SE['n_pitchers']}</b> intervalos de 80% cruzan 0.</li>"
            f"<li>La sensibilidad que predice el modelo físico no predice la observada: Pearson <b>{SE['r_mod_emp']:+.2f}</b>, Spearman "
            f"<b>{SX['r_mod_emp_spearman']:+.2f}</b>.</li></ul><b>No se usa para recomendar pitchers.</b></div>",
            unsafe_allow_html=True)
c = st.columns([1.2, 1])
c[0].plotly_chart(charts.sensitivity_scatter(D["sens"], SE["r_mod_emp"]), width="stretch")
with c[1]:
    st.markdown("**Por qué es tan ruidosa**")
    st.markdown(f"- Muestra: pocos pitches en Extreme por pitcher.\n- Rivales: parte de la \"sensibilidad\" es la calidad de los bateadores enfrentados "
                f"(r = {SE['r_rivales']:+.2f}).\n- Estadios: no todos los Extreme son físicamente iguales.\n- Selección: solo se puede medir a pitchers con mucho uso.")
    st.markdown(f"**¿Cuánto podría correlacionar, como máximo?** Con confiabilidad ≈ {SE['confiabilidad']:.2f}, incluso un modelo perfecto "
                f"correlacionaría a lo sumo ≈ √{SE['confiabilidad']:.2f} ≈ **{SX['techo_correlacion']:.2f}** con la sensibilidad observada. Es una "
                "orientación de la teoría clásica de la medición (error de medición independiente, confiabilidad bien estimada), no una cota universal.")
    with st.expander("Ver la tabla exploratoria (con advertencia)"):
        sv = D["sens"][["rank_menos_sensible", "pitcher_anon_id", "s_emp", "s_emp_lo80", "s_emp_hi80", "s_emp_shrunk", "s_shrunk_lo80", "s_shrunk_hi80",
                        "factor_shrinkage", "n_extreme", "n_no", "confiabilidad_global"]]
        st.dataframe(sv.style.format(precision=2), width="stretch", hide_index=True, height=300)
        st.caption(D["sens"]["advertencia"].iloc[0])

# ---------------------------------------------------------------------------
# 14. Física
# ---------------------------------------------------------------------------

section("fisica", "14 · Física y contrafactual: herramienta explicativa", "sim", "lim")
st.markdown("Con menos densidad del aire hay menos **drag** (la pelota frena menos) y menos **fuerza de Magnus** (el giro mueve menos la pelota). "
            "Por eso los pitches se mueven menos en altitud. El modelo físico reconstruye la trayectoria completa y la vuela de nuevo en cada "
            "ambiente con la densidad medida en los datos.")
st.markdown('<div class="warnbox"><b>Simulación física aproximada / contrafactual explicativo.</b> Su función es explicar, simular y separar '
            "mecanismos. <b>No</b> mostró una ventaja predictiva frente a una normalización simple por estadio anónimo (P0); por eso el índice "
            "principal usa el ajuste simple.</div>", unsafe_allow_html=True)
SHX = SH["extreme"]
c = st.columns([1.3, 1])
c[0].plotly_chart(charts.counterfactual_check(SHX, {"InducedVertBreak": "IVB (pulg.)", "HorzBreak": "HB (pulg.)", "ZoneSpeed": "ZoneSpeed (mph)",
                                                    "VertApprAngle": "VAA (°)"}), width="stretch")
with c[1]:
    st.markdown(badges("lim") + " **¿Qué tan bien reproduce lo observado?**", unsafe_allow_html=True)
    st.markdown(f"- Con los parámetros estimados en una mitad de los pitchers y evaluado en la otra, reproduce bien ZoneSpeed "
                f"({SHX['ZoneSpeed']['predicho']:+.2f} vs. {SHX['ZoneSpeed']['observado']:+.2f} mph) y VAA ({SHX['VertApprAngle']['predicho']:+.2f}° vs. "
                f"{SHX['VertApprAngle']['observado']:+.2f}°). La ubicación y los ángulos de llegada (VAA, HAA) se reconstruyen casi exactos (P0).\n"
                f"- **Sobreestima ≈ {pct(SH['sobreprediccion_ivb'], 0)} el cambio de IVB** ({SHX['InducedVertBreak']['predicho']:+.2f} vs. "
                f"{SHX['InducedVertBreak']['observado']:+.2f} pulg.).\n\n"
                "No invalida el modelo, pero no debe leerse como una simulación física exacta.")
paths = E.trajectory_paths(inputs)
end = paths.groupby("ambiente", sort=False).tail(1).set_index("ambiente")
st.markdown(badges("sim") + f" **Tu lanzamiento** ({pt}) soltado exactamente igual (misma posición y velocidad de salida) en los tres "
            f"ambientes. En Extreme cruza el plato {end.loc['Extreme Altitude', 'dif_altura_pulg']:+.1f} pulgadas en altura y "
            f"{end.loc['Extreme Altitude', 'dif_lado_pulg']:+.1f} en lado respecto de No Altitude: es el movimiento que se pierde con menos drag y "
            "menos Magnus.", unsafe_allow_html=True)
st.plotly_chart(charts.trajectories(paths), width="stretch")
c = st.columns([1.2, 1])
c[0].plotly_chart(charts.stadium_physics(D["stadiums"]), width="stretch")
ES = S["estadios"]
with c[1]:
    st.markdown("**Extreme no es homogéneo**")
    st.markdown(f"- Drag en Extreme: **{ES['drag_extreme'][0]:.2f}–{ES['drag_extreme'][1]:.2f}**; Magnus: **{ES['magnus_extreme'][0]:.2f}–{ES['magnus_extreme'][1]:.2f}** "
                f"(I² = {pct(ES['I2_drag_extreme'], 0)} y {pct(ES['I2_magnus_extreme'], 0)}: la diferencia entre estadios es real, no ruido).\n"
                f"- Hay traslape físico con Medium (drag de Medium {ES['drag_medium'][0]:.2f}–{ES['drag_medium'][1]:.2f}).")
    st.markdown("La etiqueta de altitud resume un entorno, pero **no todos los estadios de la misma categoría son físicamente iguales**.")
    st.markdown(f"**Estadios anónimos** (reconstruidos solo con el dataset): {pct(ES['pct_juegos_resueltos'])} de los juegos asignados; ARI ≥ "
                f"{ES['ari_submuestra_80_min']:.3f}; {ES['grupos_misma_categoria']}/{ES['grupos_multitemporada']} grupos conservan su categoría entre "
                f"temporadas; test–retest de drag {ES['test_retest_drag']:.2f} y de Magnus {ES['test_retest_magnus']:.2f}.")
ty = D["types"]
st.plotly_chart(charts.bars_simple(ty["tipo"], ty["ivb_E"], "Cambio de movimiento vertical inducido en Extreme por tipo de pitch (pulgadas, mismo pitcher)",
                                   fmt="{:+.1f}", yaxis_title="pulgadas"), width="stretch")
st.caption("Negativo = menos movimiento vertical inducido. La curva tiene IVB negativo (cae), así que su +1.6 también es pérdida de "
           "movimiento: cae menos. Efecto promedio de cada tipo comparando al mismo pitcher; no es el efecto causal de elegir ese pitch.")

# ---------------------------------------------------------------------------
# 15. Metodología
# ---------------------------------------------------------------------------

section("metodo", "15 · Qué funcionó y qué no", "lim")
c = st.columns(2)
c[0].markdown(f"""<div class="okbox"><b>Funcionó</b><ul>
<li>Separar Stuff+ y Location+: la ubicación explica {pct(SS['eta2_ubicacion_physical'])} de Stuff+ Physical (antes {pct(SS['eta2_ubicacion_v1'], 0)}); r entre ambos = {SL['r_stuff']:+.2f}.</li>
<li>Perfil swing-and-miss: whiffs, ponches y pocas bolas en juego valen más donde el contacto cuesta más.</li>
<li>Evidencia de mayor costo del contacto: {CONS[0]}/{CONS[1]} estadio-temporadas.</li>
<li>AEPP: pesos en carreras sin elección manual; predice mejor que Pitching+ las carreras en altitud.</li>
<li>Estadios anónimos estables, construidos solo con los datos ({pct(ES['pct_juegos_resueltos'])} asignados).</li>
<li>Reproducibilidad: todo se regenera con el pipeline desde los datos crudos.</li></ul></div>""", unsafe_allow_html=True)
c[1].markdown(f"""<div class="dangerbox"><b>No funcionó (y lo reportamos)</b><ul>
<li>El contrafactual físico no mejora la predicción frente al ajuste simple por estadio.</li>
<li>La sensibilidad individual no es estable (confiabilidad {SE['confiabilidad']:.2f}; modelo vs. observado r = {SE['r_mod_emp']:+.2f}).</li>
<li>AEPP no supera a un K−BB% con shrinkage ({AE['r_altitud_t1']:.2f} vs. {AE['r_kbb_altitud_t1']:.2f}).</li>
<li>No aparecen arquetipos claros de pitcher.</li>
<li>No se puede separar la altitud del resto del parque: Medium se parece a Extreme.</li></ul></div>""", unsafe_allow_html=True)
TS = S["temporadas"]
yrs = sorted(TS["pitches"])
st.markdown("### Limitaciones " + badges("lim"), unsafe_allow_html=True)
c = st.columns(2)
with c[0]:
    st.markdown(f"""**Validación temporal limitada.** Solo hay {len(TS['pares_validacion'])} pares de temporadas ({' y '.join(TS['pares_validacion'])}),
y {yrs[-1]} es una temporada parcial ({TS['juegos'][yrs[-1]]:,} juegos frente a {TS['juegos'][yrs[-2]]:,} en {yrs[-2]}). No hacemos afirmaciones de
estabilidad de largo plazo.

**Calidad de los rivales.** Los efectos del ambiente controlan por pitcher, bateador, temporada y conteo, pero las métricas individuales que
usan los rankings **no están totalmente ajustadas por la calidad de los bateadores enfrentados**. Parte de la sensibilidad observada se relaciona con
los rivales (r = {SE['r_rivales']:+.2f}).

**Altitud vs. parque.** Pocos estadios por categoría: la asociación es fuerte y consistente, pero no es identificación causal de la altitud.
Medium se parece mucho a Extreme.

**Contrafactual físico.** Simulación aproximada: sobreestima ≈ {pct(SH['sobreprediccion_ivb'], 0)} el cambio de IVB (sección 14).

**Simulador.** Usa los modelos finales (entrenados con todos los datos) y no ve el parque: sus diferencias entre ambientes son solo físicas.""")
with c[1]:
    hp = TS["hard_pct_bip"]
    ev = TS["ev_media_bip"]
    st.markdown("**Medición del contacto.** La velocidad de salida (EV) y el contacto duro dependen de la calibración del Trackman de cada estadio. "
                f"El contacto duro promedio cambia de una temporada a otra ({pct(hp[yrs[0]])} en {yrs[0]} → {pct(hp[yrs[1]])} en {yrs[1]}), y no "
                "podemos separar cuánto es cambio real del juego y cuánto es calibración:")
    st.dataframe(pd.DataFrame({"temporada": yrs, "hard contact% (EV ≥ 95 mph, bolas en juego)": [hp[y] for y in yrs],
                               "EV media en bolas en juego (mph)": [ev[y] for y in yrs], "pitches": [TS["pitches"][y] for y in yrs]})
                 .style.format({"hard contact% (EV ≥ 95 mph, bolas en juego)": "{:.1%}", "EV media en bolas en juego (mph)": "{:.1f}", "pitches": "{:,.0f}"}),
                 width="stretch", hide_index=True)
    st.markdown("Por eso los análisis de contacto duro y los mapas de calor que dependen del contacto llevan esta advertencia: **la EV no es una "
                "medición perfectamente comparable entre estadios y temporadas**. El run value (lo que pasó en el juego) no depende de esa calibración.")
    st.markdown(f"**Daño de contacto por pitcher.** Es casi ruido en estos datos (r año a año {STAB['rv_bip']['estabilidad_cruda']:+.2f}).")

with st.expander("Metodología y reproducibilidad"):
    st.markdown("- Datos: Trackman anónimo del hackathon; ninguna fuente externa. Estadios reconstruidos como `ANON_STADIUM_xx` (P0).\n"
                f"- Golpeado corregido en P0 (run value {S['hbp_run_value']:+.3f}).\n"
                "- Validación año a año con bootstrap de pitchers; validación temporal estricta en P0 (entrenar solo con temporadas anteriores).\n"
                "- Pipeline: `.venv\\Scripts\\python -m src.pipeline` regenera todo, incluidas las tablas de esta página (`data/web/`).\n"
                "- Reportes: `reports/p0_altitude_pipeline.md`, `reports/p2_pitcher_profiles_altitude.md`, `reports/auditoria_matematica.md`, "
                "`reports/web_final_validation.md`.")

# ---------------------------------------------------------------------------
# 16. Qué mejoraríamos
# ---------------------------------------------------------------------------

section("futuro", "16 · Qué mejoraríamos con más datos", "hip")
st.markdown(f"""1. **Más temporadas** para estimar la sensibilidad individual (hoy harían falta ≈ {SE['pitches_para_0_5']:,.0f} pitches en Extreme por
   pitcher para una confiabilidad de 0.5).
2. **Ajustar completamente por la calidad de los rivales** en las métricas individuales.
3. **Modelar el daño esperado de contacto** a partir de velocidad de salida, ángulo de salida y tipo de contacto, en lugar de usar solo el
   resultado observado.
4. **Separar estadios por densidad física medida** en lugar de por categoría.
5. **Mejorar el contrafactual físico** para corregir la sobrepredicción de ≈ {pct(SH['sobreprediccion_ivb'], 0)} del cambio de IVB.""")
st.markdown("**Extensión importante (no implementada): calidad de contacto esperada**")
st.markdown('<div class="formula">Expected Contact Quality = f( Exit Velocity, Launch Angle, características del pitch, ubicación, contexto )</div>',
            unsafe_allow_html=True)
st.markdown("Objetivo: separar mejor la **habilidad del pitcher** de la **defensa**, el **parque** y la **suerte**. Requiere corregir antes la "
            "calibración de la velocidad de salida por estadio y temporada (sección 15).")

st.markdown("---")
st.markdown('<div class="bigidea">En los ambientes de altitud analizados, <b>el contacto cuesta más y los whiffs/ponches son más difíciles de generar</b>. '
            "Por ello, los pitchers con perfiles de swing-and-miss adquieren una ventaja relativa, aunque el perfil favorable sigue siendo muy "
            "parecido al perfil de un buen pitcher en general.</div>", unsafe_allow_html=True)
st.markdown('<div class="bigidea"><b>Pitching+</b> es la herramienta principal para convertir Stuff, Location, contexto y probabilidades de resultado '
            "en una estimación interpretable del valor de cada lanzamiento.</div>", unsafe_allow_html=True)
st.markdown('<p class="foot">Todas las cifras se leen de <code>data/web/</code> y de los modelos finales del pipeline. Identificadores anónimos.</p>',
            unsafe_allow_html=True)
