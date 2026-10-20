"""Dashboard Streamlit — Pitching+ v3 neutral por altitud.

Dos usos:
1. Pitch hipotético (p. ej. un prospecto): se describe el pitch y la
   categoría donde se midió su movimiento; el dashboard re-simula la
   trayectoria completa en las 3 categorías de altitud y la valúa en
   carreras con Pitching+ v3 (ver dashboard/scoring.py).
2. Pitcher existente: Pitching+ oficial (neutral por altitud, con intervalo
   de 80%), el mismo índice con altitud, su exposición a cada altitud, el
   costo de altitud de su arsenal y el detalle por tipo de pitch.

Para correrlo:  .venv\\Scripts\\streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import scoring  # noqa: E402
from src.reports import scouting_report_v2  # noqa: E402


@st.cache_resource(show_spinner="Cargando modelos de Pitching+ v3...")
def load_artifact():
    return scoring.load_artifact()


@st.cache_resource(show_spinner="Cargando datos de referencia...")
def load_reference():
    return scoring.load_ratios(), scoring.compute_pitch_defaults(), scoring.load_count_shares(), scoring.load_altitude_effects()


@st.cache_resource(show_spinner="Cargando índice oficial y pitches...")
def load_pitcher_data():
    official = scoring.load_official()
    resources = scoring.load_scouting_resources()
    return official, resources, scoring.load_scatter_sample(resources, official)


st.set_page_config(page_title="Pitching+ y altitud", layout="wide")
st.title("Pitching+ v3 y el efecto de la altitud")
st.caption(
    "Índice oficial: **Pitching+ v3 neutral por altitud**. Los valores están en **carreras por cada 100 pitches** "
    "(negativo = el pitch evita carreras). El efecto de la altitud se calcula re-simulando la trayectoria completa del pitch "
    "con la densidad del aire medida en los datos (`reports/altitude_counterfactual_v2.md`)."
)

artifact = load_artifact()
ratios, defaults, count_shares, effects = load_reference()

tab_pitch, tab_pitcher = st.tabs(["Pitch hipotético", "Pitcher existente"])

# ---------------------------------------------------------------------------
# 1. Pitch hipotético
# ---------------------------------------------------------------------------
with tab_pitch:
    st.markdown(
        "Describe un pitch y la altitud donde se midió su movimiento. El dashboard construye la trayectoria que reproduce "
        "ese movimiento y la vuelve a lanzar en las tres categorías de altitud (mismo punto de salida, mismo objetivo en el plato)."
    )
    with st.expander("Supuestos"):
        st.markdown(
            f"- **Ubicación:** centro de la zona (altura {scoring.TARGET_PLATE_HEIGHT} pies). Un pitch al centro recibe mucho "
            "contacto, así que el efecto del entorno de bola en juego pesa más que en un pitch promedio.\n"
            "- **Conteo:** neutral — se evalúa en los 12 conteos y se promedia con la frecuencia real de cada uno.\n"
            "- **Punto de salida lateral, drag típico y diferencias contra la recta:** mediana de la liga para ese tipo de pitch, "
            "mano y categoría.\n"
            "- **SpinAxis:** se deduce de la dirección del movimiento (IVB, HB).\n"
            "- **HorzBreak** en la convención de Trackman (vista del cátcher): positivo hacia el lado del brazo de un derecho."
        )
    with st.form("pitch_form"):
        reference_category = st.selectbox(
            "¿A qué altitud se midió este movimiento?", scoring.ALTITUDE_CATEGORIES, index=0,
            format_func=lambda c: scoring.ALTITUDE_LABELS_ES[c],
            help="Nivel del mar para un prospecto de otra liga; la altitud real si es un pitch medido en casa.",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            pitch_type = st.selectbox("Tipo de pitch", scoring.PITCH_TYPES, index=0)
            pitcher_throws = st.selectbox("Mano del pitcher", scoring.PITCHER_THROWS_OPTIONS)
            batter_side = st.selectbox("Lado del bateador", scoring.BATTER_SIDE_OPTIONS)
        with c2:
            rel_speed = st.number_input("Velocidad de salida (mph)", 60.0, 105.0, 93.0, 0.1)
            ivb = st.number_input("InducedVertBreak (pulg)", -25.0, 30.0, 16.0, 0.1)
            hb = st.number_input("HorzBreak (pulg)", -25.0, 25.0, 8.0, 0.1)
        with c3:
            spin = st.number_input("SpinRate (rpm)", 500.0, 3500.0, 2300.0, 10.0)
            extension = st.number_input("Extensión (pies)", 4.0, 8.0, 6.2, 0.1)
            rel_height = st.number_input("Altura de salida (pies)", 3.0, 7.5, 5.8, 0.1)
        submitted = st.form_submit_button("Calcular", type="primary")

    if submitted:
        inputs = {
            "RelSpeed": rel_speed, "InducedVertBreak": ivb, "HorzBreak": hb, "SpinRate": spin, "Extension": extension,
            "RelHeight": rel_height, "AutoPitchType": pitch_type, "PitcherThrows": pitcher_throws, "BatterSide": batter_side,
        }
        st.session_state["pitch_result"] = scoring.compute_altitude_scenarios(inputs, reference_category, artifact, ratios, defaults)
        st.session_state["pitch_inputs"] = inputs
        st.session_state["pitch_ref"] = reference_category

    if "pitch_result" in st.session_state:
        result = st.session_state["pitch_result"]
        inputs = st.session_state["pitch_inputs"]
        ref = st.session_state["pitch_ref"]
        st.subheader("Valor del pitch por altitud")
        cols = st.columns(3)
        ref_value = result.loc[result["altitude_category"] == ref, "runs_per_100"].iloc[0]
        for col, (_, r) in zip(cols, result.iterrows()):
            cat = r["altitude_category"]
            label = scoring.ALTITUDE_LABELS_ES[cat] + (" (medido)" if cat == ref else " (simulado)")
            with col:
                st.metric(
                    label, f"{r['runs_per_100']:+.2f} carreras/100",
                    delta=None if cat == ref else f"{r['runs_per_100'] - ref_value:+.2f}",
                    delta_color="inverse",
                )
                st.caption(
                    f"vs. el {inputs['AutoPitchType']} típico en este entorno: **{r['runs_saved_vs_typical_per_100']:+.2f}** "
                    "carreras evitadas/100"
                )
        no_alt = result.loc[result["altitude_category"] == "No Altitude"].iloc[0]
        ext = result.loc[result["altitude_category"] == "Extreme Altitude"].iloc[0]
        diff = ext["runs_per_100"] - no_alt["runs_per_100"]
        pre_contact = ext["antes_contacto_vs_no_per_100"]
        st.markdown(
            f"**Altitud extrema vs. nivel del mar:** {diff:+.2f} carreras por cada 100 pitches "
            f"(≈{diff * scoring.STARTER_PITCHES / 100:+.1f} por temporada de abridor, ≈{diff * scoring.RELIEVER_PITCHES / 100:+.1f} de relevista). "
            f"De eso, **{pre_contact:+.2f}** ocurre antes del contacto (swing, bolas/strikes, whiffs, fouls, incluida la mayor "
            "probabilidad de que la pelota quede en juego) y el resto es el resultado de la bola en juego, que mezcla calidad del "
            "contacto y entorno del parque (atribución de Shapley; es contable, no causal)."
        )
        note = scoring.altitude_note(inputs["AutoPitchType"], effects)
        if note:
            st.info(note)
        with st.expander("Trayectoria simulada en cada categoría"):
            show = result[["altitude_category", "InducedVertBreak", "HorzBreak", "ZoneSpeed", "VertApprAngle", "ZoneTime", "p_whiff", "runs_per_100"]]
            st.dataframe(show.set_index("altitude_category").round(3))

        official, resources, scatter = load_pitcher_data()
        fig = px.scatter(
            scatter, x="HorzBreak", y="InducedVertBreak", color="pitching_plus", color_continuous_scale="RdYlGn", opacity=0.45,
            hover_data=["AutoPitchType"],
            labels={"HorzBreak": "HorzBreak (pulg)", "InducedVertBreak": "InducedVertBreak (pulg)", "pitching_plus": "Pitching+ oficial"},
            title="Movimiento en contexto — muestra de pitches reales coloreada por el Pitching+ oficial de su pitcher",
        )
        fig.add_trace(go.Scatter(
            x=result["HorzBreak"], y=result["InducedVertBreak"], mode="markers+text",
            text=[scoring.ALTITUDE_LABELS_ES[c] for c in result["altitude_category"]], textposition="top center",
            marker=dict(symbol="star", size=18, color="black", line=dict(color="white", width=1.5)), name="Pitch ingresado",
        ))
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 2. Pitcher existente
# ---------------------------------------------------------------------------
with tab_pitcher:
    official, resources, _ = load_pitcher_data()
    pitcher_ids = sorted(official["pitcher_anon_id"].unique())
    selected = st.selectbox("pitcher_anon_id", ["(elige un pitcher)"] + pitcher_ids)
    if selected != "(elige un pitcher)":
        seasons = official.loc[official["pitcher_anon_id"] == selected].sort_values("year")
        st.subheader(f"Pitching+ oficial — {selected}")
        cols = st.columns(len(seasons))
        for col, (_, r) in zip(cols, seasons.iterrows()):
            with col:
                st.metric(
                    f"{int(r['year'])} · {int(r['n_pitches']):,} pitches", f"{r['pitching_plus']:.1f}",
                    delta=f"{r['pitching_plus'] - r['pitching_plus_con_altitud']:+.1f} vs. con altitud",
                )
                st.caption(
                    f"Intervalo de credibilidad 80%: [{r['pitching_plus_lo80']:.1f}, {r['pitching_plus_hi80']:.1f}] · "
                    f"percentil {100 * r['percentil_temporada']:.0f} · {100 * r['pct_extreme_altitude']:.0f}% de sus pitches en altitud extrema"
                )
        table = seasons[["year", "n_pitches", "pitching_plus", "pitching_plus_lo80", "pitching_plus_hi80", "pitching_plus_con_altitud",
                         "stuff_plus_v1",
                         "pct_no_altitude", "pct_medium_altitude", "pct_extreme_altitude",
                         "costo_altitud_total_por_100", "costo_altitud_antes_contacto_por_100"]].set_index("year")
        st.dataframe(table.round(2))
        st.caption(
            "`pitching_plus` = neutral por altitud con shrinkage (oficial; 100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). El intervalo es de credibilidad "
            "empírico-bayesiano: refleja el tamaño de muestra, no la incertidumbre de los modelos. "
            "`stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado). "
            "`costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar."
        )
        year = st.selectbox("Temporada para el detalle del arsenal", seasons["year"].tolist()[::-1])
        arsenal = scouting_report_v2.arsenal_table(selected, int(year), resources)
        st.subheader(f"Arsenal {year}")
        st.dataframe(
            arsenal[["n", "uso_pct", "velo", "ivb", "hb", "spin", "whiff_swing_vs_tipo_pp", "carreras_evitadas_vs_liga_por_100",
                     "carreras_evitadas_vs_liga_en_altitud_por_100", "costo_antes_contacto", "costo_total"]].round(2)
        )
        row = seasons.loc[seasons["year"] == year].iloc[0]
        st.subheader("Recomendaciones")
        for rec in scouting_report_v2.recommendations(row, arsenal, resources):
            st.markdown(f"- {rec}")
