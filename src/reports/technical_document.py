"""Documento técnico de entrega (6–10 páginas): `reports/documento_tecnico.md`, `.html` y `.pdf`.

Todas las cifras se leen de los outputs del pipeline (data/web/, reports/benchmark/). El PDF se imprime con Microsoft Edge
en modo headless si está disponible; si no, quedan el Markdown y el HTML.
"""

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "data" / "web"
BENCH = ROOT / "reports" / "benchmark"
MD_PATH = ROOT / "reports" / "documento_tecnico.md"
HTML_PATH = ROOT / "reports" / "documento_tecnico.html"
PDF_PATH = ROOT / "reports" / "documento_tecnico.pdf"
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"), Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def m(x, d=1):
    """Número con signo y menos tipográfico."""
    return f"{x:+.{d}f}".replace("-", "−")


FIG_DIR = ROOT / "reports" / "figures" / "documento"


def calibration_figure(VH: dict) -> str:
    """Calibración por deciles del valor predicho en los cuatro esquemas de holdout (PNG para el documento)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=150)
    lab = {"pitcher_holdout": "pitcher holdout", "season_holdout_2026": "season holdout", "park_holdout_extreme": "park holdout: Extreme",
           "park_holdout_medium": "park holdout: Medium"}
    for k, e in VH.items():
        d = pd.DataFrame(e["calibracion_deciles"])
        ax.plot(100 * d["predicho"], 100 * d["realizado"], marker="o", ms=3.5, lw=1.2, label=lab[k])
    lim = [-9, 12]
    ax.plot(lim, lim, color="#8892a6", lw=1, ls="--", label="calibración perfecta")
    ax.set_xlabel("valor predicho (carreras por 100 pitches)")
    ax.set_ylabel("valor realizado (carreras por 100 pitches)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibracion_holdout.png")
    plt.close(fig)
    return "figures/documento/calibracion_holdout.png"


def build() -> str:
    S = J(WEB / "summary.json")
    EV = J(WEB / "pitching_env_values.json")
    VH = J(WEB / "validation_holdout.json")["esquemas"]
    TV = J(BENCH / "validacion_temporal.json")["prespecified"]
    PR = J(WEB / "prospect_reference.json")
    OO = J(WEB / "outs_on_play_check.json")
    OC = pd.read_csv(WEB / "optimal_pitch_comparison.csv")
    SH = pd.read_csv(WEB / "pitch_shape_altitude_effects.csv")
    TY = pd.read_csv(WEB / "pitch_type_altitude_effects.csv").set_index("tipo")
    AR = pd.read_csv(WEB / "arsenal_altitude.csv")
    ef, ss, sl, ae, se, es, dm = S["efectos"], S["stuff"], S["location"], S["aepp"], S["sensibilidad"], S["estadios"], S["dano"]
    vp = S["validacion_perfil"]
    stab = {r["metrica"]: r for r in S["estabilidad_metricas"]}
    oc_e, oc_f = OC[OC["alterno"] == "extreme"], OC[OC["alterno"] == "extreme_fisica"]
    rg = EV["ganancia_relativa_mismo_conteo"]["correlaciones"]
    fig_cal = calibration_figure(VH)
    L = []
    add = L.extend

    add([
        "# Pitching+ para ambientes de altitud — Documento técnico",
        "",
        "*Hackathon de analítica deportiva · datos Trackman anónimos 2024–2026 · proyecto reproducible (`python -m src.pipeline`)*",
        "",
        "## 1. Resumen ejecutivo",
        "",
        "El reto pregunta cómo medir la calidad de pitcheo cuando se lanza en altitud extrema (las condiciones de la Ciudad de México) y qué "
        "pitchers y lanzamientos convienen en ese ambiente. El dataset está anonimizado: para cada juego solo indica si fue en No Altitude, "
        "Medium o Extreme Altitude. Respondemos con la categoría **Extreme Altitude**, sin identificar qué estadio anónimo es cuál.",
        "",
        f"- **El juego cambia en altitud.** Comparando al mismo pitcher contra el mismo bateador, en Extreme el whiff por swing baja "
        f"{m(100 * ef['whiff_swing']['extreme'])} pp, el ponche por turno {m(100 * ef['k_pa']['extreme'])} pp y cada bola en juego vale "
        f"{m(ef['rv_bip']['extreme'], 3)} carreras más, en {ef['whiff_swing']['consistencia_extreme'][0]} de {ef['whiff_swing']['consistencia_extreme'][1]} "
        "combinaciones estadio anónimo × temporada.",
        f"- **El contacto cuesta más y evitarlo vale más:** cambiar una bola en juego por un ponche vale ≈ {pct(S['pesos']['k_vs_bip_aumento_extreme'], 0)} "
        f"más en Extreme. {pct(dm['fraccion_por_dano_extreme'], 0)} del aumento del daño de contacto viene del costo de cada bola en juego, no de "
        "que haya más.",
        "- **Tres métricas separadas y un valor total:** Stuff+ Physical (física del pitch), Location+ (ubicación) y **Pitching+** (valor esperado "
        "del lanzamiento en carreras con un árbol de probabilidades).",
        f"- **Generaliza fuera de muestra:** supera al modelo nulo por conteo en pitcher, season y park holdout (mejora del MSE de "
        f"{pct(min(e['skill_vs_conteo'] for e in VH.values()))} a {pct(max(e['skill_vs_conteo'] for e in VH.values()))}) y está calibrado "
        f"(pendiente {min(e['pendiente_calibracion'] for e in VH.values()):.2f}–{max(e['pendiente_calibracion'] for e in VH.values()):.2f}).",
        f"- **Perfil favorable:** swing-and-miss (whiffs, ponches, pocas bolas en juego). El ranking AEPP usa pesos en carreras de los ambientes "
        f"de altitud y correlaciona {ae['rho_pesos_altitud_vs_no']:.3f} con el ranking con pesos de nivel del mar: la altitud no crea otro tipo de "
        "buen pitcher, premia más lo que ya era bueno.",
        f"- **Un gran lanzamiento sigue siendo un gran lanzamiento:** con los costos de Extreme el lanzamiento óptimo de cada tipo cambia menos "
        f"que el ruido de muestreo en {int((oc_e['clase'] == 'Poco').sum())} de {len(oc_e)} combinaciones tipo × conteo.",
        "",
        "## 2. Marco teórico",
        "",
        "**Sabermetría.** El valor de un evento se mide en carreras con la matriz de carreras esperadas por estado de outs y corredores (RE24): "
        "el *run value* de una jugada es RE después − RE antes + carreras anotadas (Tango, Lichtman y Dolphin, 2007). Los índices \"plus\" "
        "(Stuff+, Location+, Pitching+) expresan la calidad del pitcheo en una escala con media 100 y 10 puntos por desviación estándar; separan "
        "las características físicas del pitch, su ubicación y su valor total, que se estabilizan con muchos menos lanzamientos que los resultados "
        "(FanGraphs, 2022).",
        "",
        "**Física del béisbol.** Un lanzamiento está sometido a gravedad, arrastre (*drag*) y la fuerza de Magnus producida por el giro; ambas "
        "fuerzas aerodinámicas son proporcionales a la densidad del aire (Adair, 2002; Nathan, 2008). A mayor altitud, el aire es menos denso: "
        "la pelota frena menos y el giro la mueve menos, así que los pitches se mueven menos y la pelota bateada vuela más. El Trackman ajusta "
        "una trayectoria de aceleración constante (9 parámetros) de la que salen el movimiento vertical inducido (IVB) y el horizontal (HB).",
        "",
        f"**Entorno.** Con los propios datos estimamos la fuerza del aire en cada estadio anónimo: en Extreme el drag es "
        f"{es['drag_extreme'][0]:.2f}–{es['drag_extreme'][1]:.2f} y el Magnus {es['magnus_extreme'][0]:.2f}–{es['magnus_extreme'][1]:.2f} veces el de "
        f"No Altitude (I² = {pct(es['I2_drag_extreme'], 0)}: los estadios de Extreme no son iguales entre sí). Los estadios se reconstruyen sin "
        f"información externa ({pct(es['pct_juegos_resueltos'])} de los juegos asignados; {es['grupos_misma_categoria']}/{es['grupos_multitemporada']} "
        "grupos conservan su categoría entre temporadas).",
        "",
        "## 3. Datos",
        "",
        f"{S['temporadas']['pitches']['2024'] + S['temporadas']['pitches']['2025'] + S['temporadas']['pitches']['2026']:,} lanzamientos Trackman de "
        f"{sum(S['temporadas']['juegos'].values()):,} juegos (2024, 2025 y 2026 parcial), con velocidad, spin, eje, salida, extensión, movimiento, "
        "ubicación, conteo, resultado del pitch y de la jugada, y la categoría de altitud de cada juego. Se corrigió el valor del golpeado "
        f"(run value {m(S['hbp_run_value'], 3)}) y se reconstruyó el estado de corredores por turno para RE24. Ninguna fuente externa.",
        "",
        "## 4. Decisiones de modelado",
        "",
        "**Arquitectura: un árbol de probabilidades valuado en carreras.** Pitching+ encadena cinco clasificadores LightGBM (Ke et al., 2017), "
        "cada uno entrenado en su universo: P(swing); el resultado si no hay swing (bola, strike cantado, golpeado); P(whiff | swing); "
        "P(foul | contacto); y el resultado de la bola en juego (out, sencillo, doble, triple, jonrón). Las diez probabilidades suman 1 y el valor "
        "esperado es",
        "",
        "> E[Δ] = Σ_evento P(evento | física, ubicación, conteo) × Δ(evento, conteo),",
        "",
        "donde Δ es el cambio de carreras esperadas del turno: V(siguiente conteo) − V(conteo), o el valor terminal si el evento termina el "
        "turno. V(conteo) se obtiene hacia atrás desde los valores terminales con las frecuencias observadas de cada transición (el foul con dos "
        "strikes es un auto-lazo).",
        "",
        "**¿Por qué no regresar directamente el run value?** Porque el run value de un pitch individual es casi todo ruido: un modelo de "
        "regresión directo con solo física quedó peor que el promedio por conteo. El árbol aprovecha que cada paso (swing, whiff, contacto) sí es "
        "predecible y valúa cada resultado con su costo en carreras. Tampoco sumamos Stuff+ y Location+: el valor de un buen movimiento depende de "
        "dónde termina el pitch y del conteo, y esa interacción la calcula el árbol.",
        "",
        "**Función de pérdida.** Log-loss (entropía cruzada) binaria o multiclase en cada sub-modelo; hiperparámetros ajustados por sub-modelo con "
        "validación agrupada por pitcher. La log-loss premia probabilidades bien calibradas, que es lo que necesita un valor esperado.",
        "",
        "**Features.**",
        "",
        "- **Física:** velocidad, spin rate, eje de giro, altura y lado de salida, extensión, IVB, HB, pérdida de velocidad y ángulos de llegada "
        "*residualizados contra la ubicación* (para que no codifiquen dónde terminó el pitch).",
        "- **Arsenal (relacionales):** diferencia de velocidad, IVB y HB contra la recta del mismo pitcher, y el tipo de pitch.",
        "- **Contexto:** ubicación (lado y altura, espejadas para zurdos), conteo y si pitcher y bateador son del mismo lado.",
        f"- **Stuff+ Physical** usa solo física y arsenal (la ubicación explica {pct(ss['eta2_ubicacion_physical'])} de su predicción, contra "
        f"{pct(ss['eta2_ubicacion_v1'], 0)} de la versión de referencia v1). **Location+** usa solo ubicación, conteo, tipo y manos "
        f"(correlación con Stuff+ {m(sl['r_stuff'], 2)}).",
        "",
        "**Ambiente.** Los modelos no reciben la altitud como variable. El ambiente se trata con un ajuste simple: a cada valor se le resta su "
        "promedio por temporada × estadio anónimo. Probamos también un contrafactual físico (re-simular cada trayectoria con el drag y el Magnus "
        "de otro ambiente); en la validación temporal estricta el ajuste simple predijo igual o mejor (sección 7), así que el contrafactual quedó "
        "como herramienta explicativa.",
        "",
        "## 5. Definición formal de las escalas",
        "",
        "Sea θ̂ᵢ el promedio de un pitcher (P(whiff | swing) para Stuff+, valor de ubicación para Location+, E[Δ] para Pitching+), ajustado por "
        "ambiente. El *shrinkage* empírico-bayesiano normal–normal (Efron y Morris, 1975) lo encoge hacia la media de la liga:",
        "",
        "> θ̃ᵢ = μ + Bᵢ(θ̂ᵢ − μ), con Bᵢ = τ² / (τ² + σ²/nᵢ),",
        "",
        "donde τ² es la varianza de la habilidad real entre pitchers y σ² la varianza por lanzamiento. La escala es",
        "",
        "> Índice+ = 100 + 10 · s · (θ̃ᵢ − μ) / τ,",
        "",
        "con s = +1 si más es mejor (Stuff+) y s = −1 si menos carreras es mejor (Location+ y Pitching+). Así, 100 es el pitcher promedio y 10 "
        "puntos una desviación estándar de **habilidad real**, no de ruido. Para un **lanzamiento individual** (simulador) se usa la escala de "
        "lanzamiento: 100 + 10·s·(v − media)/sd sobre todos los lanzamientos reales.",
        "",
        "**AEPP** (perfil para ambientes de altitud): carreras evitadas por 100 turnos frente a un pitcher promedio, con los valores en carreras "
        "medidos en Medium + Extreme y tasas con *shrinkage*:",
        "",
        "> AEPP = −100 · [Σ_{o ∈ K, BB, HBP} (rₒ − r̄ₒ)(vₒ − v_BIP) + r_BIP · (RV/BIPᵢ − RV/BIP̄)].",
        "",
        "## 6. El efecto de la altitud",
        "",
        "| métrica (Extreme vs. No Altitude, efectos fijos de pitcher, bateador, temporada y conteo) | No Altitude | Medium | Extreme [IC 95%] |",
        "|---|---:|---:|---:|",
    ])
    for k, lab, sc, d in (("whiff_swing", "whiff por swing", 100, 1), ("k_pa", "ponche por turno", 100, 1), ("bip_swing", "bola en juego por swing", 100, 1),
                          ("rv_bip", "run value por bola en juego", 1, 3)):
        e = ef[k]
        u = " pp" if sc == 100 else ""
        add([f"| {lab} | {sc * e['no_altitude']:.{d}f}{'%' if sc == 100 else ''} | {m(sc * e['medium'], d)}{u} | {m(sc * e['extreme'], d)}{u} "
             f"[{m(sc * e['extreme_ic95'][0], d)}, {m(sc * e['extreme_ic95'][1], d)}] |"])
    fs = TY.loc["Four-Seam"]
    sl_ = TY.loc["Slider"]
    add([
        "",
        "![Daño de contacto por ambiente: P(bola en juego), daño por bola en juego y su producto](figures/p2/contact_damage_by_environment.png)",
        "",
        "![Valor de evitar el contacto: carreras que ahorra un ponche frente a una bola en juego, por ambiente](figures/p2/value_of_avoiding_contact.png)",
        "",
        f"Medium se parece a Extreme (run value por bola en juego {m(ef['rv_bip']['medium'], 3)} vs. {m(ef['rv_bip']['extreme'], 3)}): hablamos de "
        "ambientes de altitud, no de un umbral. Altitud y estadio están confundidos (pocos estadios por categoría): son asociaciones consistentes, "
        "no un efecto causal aislado de la altitud.",
        "",
        "**Por tipo y por forma de pitch.** Comparando al mismo pitcher consigo mismo:",
        "",
        f"- la recta pierde {m(fs['ivb_E'])} pulg. de IVB y {m(100 * fs['whiff_swing_E'])} pp de whiff/swing;",
        f"- el slider es el tipo con menor costo promedio en Extreme ({m(sl_['delta_por_100_E'], 2)} carreras/100 pitches);",
        f"- sinker y changeup pierden más movimiento horizontal ({m(TY.loc['Sinker', 'hb_abs_E'])} y {m(TY.loc['Changeup', 'hb_abs_E'])} pulg.).",
        "",
        "Dentro de cada tipo, los pitches que más dependen del giro pierden más:",
        "",
        "| tipo · forma (movimiento a nivel del mar) | Δ Stuff+ | Δ whiff/swing | Δ contacto duro | Δ HR/BIP |",
        "|---|---:|---:|---:|---:|",
    ])
    for t in ["Four-Seam", "Sinker", "Slider", "Curveball", "Changeup"]:
        for f in ["bajo", "alto"]:
            r = SH[(SH["tipo"] == t) & (SH["forma"] == f)]
            if len(r):
                r = r.iloc[0]
                add([f"| {t} · {f} {r['eje'].split(' (')[0]} | {m(r['stuff_E'])} | {m(100 * r['whiff_swing_E'])} pp | {m(100 * r['hard_E'])} pp | "
                     f"{m(100 * r['hr_bip_E'])} pp |"])
    add([
        "",
        "Explicación física: con menos Magnus, el pitch que depende del giro para \"subir\", barrer o caer pierde más pulgadas, y con ellas parte "
        "del swing-and-miss; lo que se pierde en whiff se paga en contacto, que en altitud es más caro. Contacto duro = velocidad de salida ≥ 95 mph, "
        f"sensible a la calibración del Trackman (el hard contact% promedio pasa de {pct(S['temporadas']['hard_pct_bip']['2024'])} en 2024 a "
        f"{pct(S['temporadas']['hard_pct_bip']['2025'])} en 2025).",
        "",
        f"**Arsenal.** El Stuff+ calculado solo con pitches de Extreme es más alto en arsenales de recta + slider ({AR.iloc[0]['arsenal']}: "
        f"{AR.iloc[0]['stuff_plus_extreme']:.1f}; {AR.iloc[1]['arsenal']}: {AR.iloc[1]['stuff_plus_extreme']:.1f}) y correlaciona "
        f"{PR['arsenal_corr_stuff_extreme_general']:.2f} con el Stuff+ general del pitcher: el arsenal que rinde en Extreme es el que rinde en general.",
        "",
        "## 7. Validación fuera de muestra",
        "",
        "**Nivel lanzamiento (modelo final, misma especificación e hiperparámetros).** Objetivo: el valor realizado del pitch, Δ(evento, conteo). "
        "Modelo nulo: el promedio del valor por conteo (\"xRunValue promedio por conteo\"), estimado en entrenamiento. El dataset no identifica "
        "estadios: el *park holdout* deja fuera **todos** los estadios de una categoría.",
        "",
        "| esquema | pitches de prueba | RMSE modelo | RMSE nulo | mejora del MSE | pendiente de calibración | AUC Stuff+ (nulo por tipo) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    lab = {"pitcher_holdout": "pitcher holdout (k=5)", "season_holdout_2026": "season holdout (→ 2026)",
           "park_holdout_extreme": "park holdout: Extreme", "park_holdout_medium": "park holdout: Medium"}
    for k, e in VH.items():
        add([f"| {lab[k]} | {e['n_prueba']:,} | {e['rmse_modelo']:.4f} | {e['rmse_base_conteo']:.4f} | {m(100 * e['skill_vs_conteo'], 2)}% | "
             f"{e['pendiente_calibracion']:.2f} | {e['stuff']['auc']:.3f} ({e['stuff']['auc_base_tipo']:.3f}) |"])
    pe = VH["park_holdout_extreme"]
    add([
        "",
        f"![Calibración fuera de muestra: valor predicho contra realizado por deciles, en los cuatro esquemas]({fig_cal})",
        "",
        "El valor de un pitch individual es muy ruidoso, por eso el RMSE baja poco en absoluto; lo importante es que el modelo supera al nulo y "
        f"queda calibrado en pitchers, temporada y ambientes que no vio. Sin haber visto Extreme, subestima un poco su valor realizado "
        f"({m(pe['media_predicha'], 4)} vs. {m(pe['media_realizada'], 4)} carreras por pitch), lo que confirma que el contacto ahí cuesta más y "
        "justifica el ajuste por ambiente.",
        "",
        f"**Nivel pitcher (temporada t → t+1).** Estabilidad año a año: Stuff+ Physical {stab['stuff_plus_physical']['estabilidad_cruda']:.2f}, "
        f"Pitching+ {stab['pitching_plus']['estabilidad_cruda']:.2f}, Location+ {stab['location_plus']['estabilidad_cruda']:.2f}, whiff/swing "
        f"{stab['whiff_swing']['estabilidad_cruda']:.2f}, run value por bola en juego {m(stab['rv_bip']['estabilidad_cruda'], 2)}. En la validación "
        "temporal estricta (entrenando solo con temporadas anteriores, 2025 → 2026), el ajuste simple por temporada × estadio anónimo (M1b) "
        f"predice las carreras del año siguiente con r = {TV['100']['r']['t_m1b']['r']:.3f} (mín. 100 pitches; {TV['300']['r']['t_m1b']['r']:.3f} con "
        f"mín. 300), por encima del contrafactual físico ({TV['100']['r']['t_m2']['r']:.3f}) y sin ajuste ({TV['100']['r']['t_m0']['r']:.3f}). AEPP "
        f"predice las carreras en altitud del año siguiente con r = {ae['r_altitud_t1']:.2f}, mejor que Pitching+ ({ae['r_pitching_plus_altitud_t1']:.2f}; "
        f"P = {pct(ae['p_mejor_que_pitching_plus'], 0)}) e igual que K−BB% con *shrinkage* ({ae['r_kbb_altitud_t1']:.2f}); lo reportamos tal cual.",
        "",
        "## 8. Uso para el cuerpo técnico",
        "",
        "- **Simulador:** el pitching coach ingresa velocidad, IVB, HB, spin, extensión, salida, ubicación y conteo y obtiene Stuff+, Location+, "
        "Pitching+, P(whiff), P(bola en juego), el daño esperado de contacto y alertas si la combinación sale de lo observado. El mismo pitch se "
        "compara lado a lado en altitud y a nivel del mar, por física (re-simulación de la trayectoria) y por costo de los resultados "
        f"(correlación de rangos general vs. Extreme {EV['comparacion_lanzamientos']['spearman_general_extreme']:.3f}; con dos strikes gana el "
        f"swing-and-miss: correlación {m(rg['dos strikes']['whiff_vs_general'], 2)}).",
        "- **Ranking AEPP** con intervalo del lugar y P(top 25); la incertidumbre es grande (amplitud mediana del intervalo de 80%: "
        f"{ae['ancho_mediano_rango_80']:.0f} posiciones).",
        "- **Recomendaciones:** formas de pitch que pierden más en altitud, mapa de movimiento de los pitches más y menos efectivos, arsenales y "
        "un flujo para perfilar prospectos con una calculadora de AEPP.",
        f"- **Lanzamiento óptimo por ambiente:** buscado entre lanzamientos reales (sin extrapolar); con la física aproximada del aire de Extreme el "
        f"perfil óptimo sigue dentro del ruido en {int((oc_f['clase'] == 'Poco').sum())} de {len(oc_f)} combinaciones.",
        "",
        "## 9. Alcance y fronteras del modelo",
        "",
        "- **No causal:** altitud y estadio están confundidos; Medium se parece a Extreme.",
        f"- **Sensibilidad individual a la altitud no medible hoy:** confiabilidad {se['confiabilidad']:.2f}; harían falta ≈ {se['pitches_para_0_5']:,.0f} "
        f"pitches en Extreme por pitcher para 0.5; {se['intervalos_cruzan_0']} de {se['n_pitchers']} intervalos cruzan 0. Hablamos de perfil favorable, "
        "no de resistencia demostrada.",
        "- **Contrafactual físico aproximado:** sobreestima ≈ "
        f"{pct(S['contrafactual_split_half']['sobreprediccion_ivb'], 0)} el cambio de IVB (validación split-half); el modelo no ve el vuelo de la "
        "pelota bateada.",
        "- **Validación temporal limitada:** dos pares de temporadas (2024 → 2025, 2025 → 2026); 2026 es parcial.",
        "- **Calidad de rivales:** los efectos del ambiente la controlan; las métricas por pitcher del ranking no del todo.",
        "- **Datos de contacto:** la velocidad de salida depende de la calibración de cada estadio y temporada; el daño por bola en juego de un "
        f"pitcher casi no se repite (r = {m(stab['rv_bip']['estabilidad_cruda'], 2)}).",
        "- **Outs no registrados:** `OutsOnPlay` casi no registra las dobles matanzas. En "
        f"{pct(OO['pct_turnos']['No Altitude'])} de los turnos de No Altitude y {pct(OO['pct_turnos']['Extreme Altitude'])} de los de Extreme "
        f"(en rodados, {pct(OO['pct_rodados']['No Altitude'])} y {pct(OO['pct_rodados']['Extreme Altitude'])}) el siguiente turno empieza con un "
        "out más de los registrados, así que el RE24 valora esos outs como si no hubieran ocurrido. El sesgo aproximado en el run value por bola "
        f"en juego es {m(OO['sesgo_rv_bip']['No Altitude'], 3)} en No Altitude y {m(OO['sesgo_rv_bip']['Extreme Altitude'], 3)} en Extreme: "
        f"explicaría ≈ {m(OO['sesgo_extreme_menos_no'], 3)} de las {m(ef['rv_bip']['extreme'], 3)} carreras de diferencia, dentro de su intervalo "
        "de confianza. No lo corregimos porque el dataset no trae corredores en base para saber qué out faltó.",
        "- **El óptimo no es una receta:** no considera biomecánica, salud, secuencias, repertorio ni fatiga.",
        "",
        "## Referencias",
        "",
        "- Adair, R. K. (2002). *The Physics of Baseball* (3.ª ed.). HarperCollins.",
        "- Efron, B., y Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical "
        "Association*, 70(350), 311–319.",
        "- FanGraphs (2022). Stuff+, Location+ y Pitching+ (glosario y metodología).",
        "- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems* 30.",
        "- Nathan, A. M. (2008). The effect of spin on the flight of a baseball. *American Journal of Physics*, 76(2), 119–124.",
        "- Tango, T. M., Lichtman, M., y Dolphin, A. (2007). *The Book: Playing the Percentages in Baseball*. Potomac Books.",
        "",
        "*Reportes de respaldo: `reports/validacion_fuera_de_muestra.md`, `reports/p0_altitude_pipeline.md`, `reports/p2_pitcher_profiles_altitude.md`, "
        "`reports/pitching_plus_general_vs_altitude.md`, `reports/optimal_pitch_by_environment.md`, `reports/auditoria_matematica.md`.*",
        "",
    ])
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Markdown → HTML (subconjunto usado en el documento) → PDF
# ---------------------------------------------------------------------------


def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    return s


def to_html(md: str) -> str:
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("!["):
            mm = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
            out.append(f'<figure><img src="{mm.group(2)}"><figcaption>{_inline(mm.group(1))}</figcaption></figure>')
        elif ln.startswith("> "):
            out.append(f'<p class="eq">{_inline(ln[2:])}</p>')
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip("|").split(" | ")] for r in rows if not re.match(r"^\|[-:| ]+\|$", r)]
            t = "<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells[0]) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:]) + "</tbody></table>"
            out.append(t)
            continue
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    css = """
    @page { size: Letter; margin: 2cm 2cm 2cm 2cm; }
    body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #1d2433; }
    h1 { font-size: 19pt; color: #13213c; margin: 0 0 4pt; }
    h2 { font-size: 13pt; color: #13213c; border-bottom: 1px solid #d6dbe6; padding-bottom: 2pt; margin: 14pt 0 6pt; }
    p { margin: 5pt 0; text-align: justify; } ul { margin: 4pt 0 6pt 16pt; padding: 0; } li { margin: 2pt 0; }
    table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 9pt; page-break-inside: avoid; }
    th, td { border: 1px solid #d6dbe6; padding: 3pt 5pt; } th { background: #eef1f6; text-align: left; }
    td { text-align: right; } td:first-child { text-align: left; }
    .eq { font-family: Consolas, 'Cambria Math', monospace; background: #f5f7fb; padding: 5pt 8pt; border-left: 3px solid #d1495b; }
    code { font-family: Consolas, monospace; font-size: 9pt; }
    figure { margin: 6pt 0 8pt; text-align: center; page-break-inside: avoid; } figure img { max-width: 88%; max-height: 6.2cm; }
    figcaption { font-size: 8.5pt; color: #5b6577; margin-top: 2pt; }
    """
    return f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Documento técnico</title><style>{css}</style></head><body>{''.join(out)}</body></html>"


def to_pdf() -> bool:
    exe = next((p for p in EDGE if p.exists()), None) or (Path(shutil.which("msedge")) if shutil.which("msedge") else None)
    if exe is None:
        return False
    subprocess.run([str(exe), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF_PATH}", HTML_PATH.as_uri()],
                   check=False, capture_output=True, timeout=120)
    return PDF_PATH.exists()


def run() -> None:
    md = build()
    MD_PATH.write_text(md, encoding="utf-8")
    HTML_PATH.write_text(to_html(md), encoding="utf-8")
    print("PDF:", to_pdf(), "· palabras:", len(md.split()))


if __name__ == "__main__":
    run()
