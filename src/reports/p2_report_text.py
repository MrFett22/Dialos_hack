"""Texto de `reports/p2_pitcher_profiles_altitude.md` (lo arma `src/reports/p2_report.py`)."""

import numpy as np

from src.analysis import p2_common as pc
from src.analysis import pitcher_profiles as pp
from src.models import stuff_plus_physical as spp
from src.reports.p2_report import CF_WARNING, _rel

LAB_INT = {"whiff_swing": "Whiff/Swing", "k_pct": "K%", "bb_pct": "pocas BB", "bip_pct": "pocas BIP", "rv_bip": "poco daño por BIP",
           "hard_pct": "poco contacto duro", "gb_pct": "muchos rodados", "stuff_plus_physical": "Stuff+ Physical", "location_plus": "Location+"}


def _hrow(res: dict, label: str) -> dict:
    return next(r for r in res["hyp"]["h1_h2"] if r["variable"] == label)


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _fx(res, label, key="beta_extreme"):
    return _hrow(res, label)["efectos_fijos"][key]


def build_report(res: dict, stuff: dict, loc: dict, figs: dict, web: dict) -> str:
    h3 = res["hyp"]["h3"]
    G = {c: h3["por_categoria"][c]["G"] for c in pc.CATS}
    g_ratio = G["Extreme Altitude"] / G["No Altitude"] - 1
    lw = res["pesos"]
    kb = {k: lw.loc[k, "K_menos_BIP"] for k in lw.index}
    bb = {k: lw.loc[k, "BB_menos_BIP"] for k in lw.index}
    alt = "Altitud (Medium + Extreme)"
    gb_gap = {k: lw.loc[k, "v_aire"] - lw.loc[k, "v_GB"] for k in lw.index}
    hard_gap = {k: lw.loc[k, "v_duro"] - lw.loc[k, "v_no_duro"] for k in lw.index}
    rates = res["tasas_ambiente"].set_index("ambiente")
    cons = res["consistencia"]
    ce, cm = cons["Extreme Altitude"], cons["Medium Altitude"]
    dmg = res["dano"]
    dt = dmg["tabla"]
    dcat = dt[dt["nivel"] == "categoria"].set_index("ambiente")
    dec = dmg["descomposicion"]
    chk = dmg["chain_check"]
    t = res["pitchers"]
    e = t[res["elegibles"].to_numpy()].sort_values("rank")
    val = res["validacion"]
    v100, v300 = val[100], val[300]
    stab = res["estabilidad_metricas"]
    ir = res["interacciones"].set_index("caracteristica")
    arch = res["arquetipos"]
    ty = res["tipos"]
    mix = res["mezcla"]
    sens = res["sensibilidad"]
    rel = sens["info"]["confiabilidad_spearman_brown"]
    r_me = sens["r_mod_emp"]
    s_tab = sens["tabla"]
    rank_width = float((e["rank_hi80"] - e["rank_lo80"]).median())
    rho_w = float(e["aepp"].corr(e["aepp_no_altitude"], method="spearman"))
    bono_k = float(e["bono_altitud"].corr(e["k_pct"]))
    k_gain = kb["Extreme"] / kb["No Altitude"] - 1
    WH, KP, BS, RV = "whiff por swing", "ponche por turno", "bola en juego por swing", "run value RE24 por bola en juego"
    main_vars = [(WH, "Whiff/Swing"), (KP, "K/PA"), (BS, "Bola en juego por swing"), (RV, "Run value por BIP (carreras)")]
    gN0 = G["No Altitude"]
    L = []
    add = L.extend

    # 1 ----------------------------------------------------------------------
    add([
        "# P2 — Perfiles de pitcher para ambientes de altitud",
        "",
        "Generado por `src/reports/p2_report.py` con los resultados de `src/analysis/pitcher_profiles.py`, `src/models/stuff_plus_physical.py` "
        "y `src/models/location_plus_v2.py` (pasos `stuff_physical`, `location_v2`, `profiles` y `p2_report` del pipeline). Usa solo el dataset "
        "del hackathon; estadios y pitchers con identificadores anónimos (`ANON_STADIUM_xx`, `pitcher_xxxxx`).",
        "",
        "# 1. Resumen ejecutivo",
        "",
        "**Los contextos de altitud media y extrema presentan patrones consistentes de menor capacidad de generar whiffs y ponches, junto con "
        "un mayor costo esperado cuando la pelota se pone en juego.** Es evidencia contextual fuerte, pero no identificación causal pura del "
        "efecto de la altitud.",
        "",
        f"- **El juego cambia (sección 2).** En Extreme, con el mismo pitcher contra el mismo bateador: whiff por swing {100 * _fx(res, WH):+.1f} pp, "
        f"ponches por turno {100 * _fx(res, KP):+.1f} pp, run value por bola en juego {_fx(res, RV):+.3f} carreras; convertir una bola en juego "
        f"en un whiff vale ≈ {_pct(g_ratio, 0)} más. Los cuatro efectos van en la misma dirección en **{ce[WH]['en_direccion']}/{ce[WH]['total']}** "
        "estadio-temporadas de Extreme. Medium se comporta de manera muy parecida.",
        f"- **Por qué evitar contacto vale más (sección 7).** El daño esperado de contacto sube de {100 * dcat.loc['No Altitude', 'dano_bip_por_pitch']:.2f} "
        f"a {100 * dcat.loc['Extreme', 'dano_bip_por_pitch']:.2f} carreras por 100 pitches. {_pct(dec['Extreme']['por_dano'] / dec['Extreme']['total'], 0)} "
        f"del aumento viene de que cada bola en juego hace más daño; solo {_pct(dec['Extreme']['por_probabilidad'] / dec['Extreme']['total'], 0)} de que "
        "haya más bolas en juego.",
        f"- **Perfil favorable (secciones 9–10).** Lo que más vale en altitud es **poncharse bateadores y evitar bolas en juego**: cambiar una bola "
        f"en juego por un ponche vale {_pct(k_gain, 0)} más en Extreme. El puntaje AEPP (pesos medidos en los datos, no elegidos a mano) predice "
        f"las carreras del año siguiente en juegos de altitud mejor que Pitching+ (r = {v100['aepp']['carreras_altitud_t1']['r']:.2f} vs. "
        f"{v100['pitching_plus']['carreras_altitud_t1']['r']:.2f}, P = {_pct(v100['aepp']['delta_altitud_vs_pitching_plus']['p_mejor'], 0)}), "
        "pero **igual que un simple K−BB% con shrinkage**.",
        f"- **Resultado importante:** el perfil favorable para altitud es casi el mismo que el perfil bueno en cualquier parte (correlación de "
        f"rangos con pesos de altitud vs. de No Altitude = {rho_w:.3f}). La altitud agrega un **bono pequeño** a los pitchers que ponchan mucho "
        f"(correlación bono–K% = {bono_k:.2f}); no produce una lista distinta de pitchers.",
        f"- **Stuff+ Physical y Location+ (secciones 4–6)** son métricas separadas y casi independientes (r = {loc['validacion']['r_stuff_physical']:+.2f}). "
        f"La ubicación explica {_pct(stuff['ubicacion_eta2']['Stuff+ Physical'])} de la varianza de Stuff+ Physical, contra "
        f"{_pct(stuff['ubicacion_eta2']['Stuff+ v1'])} en Stuff+ v1, y Stuff+ Physical predice el whiff/swing del año siguiente mejor que v4.",
        f"- **Resultados negativos que se conservan:**",
        "  - el contrafactual físico no supera al ajuste simple;",
        f"  - la sensibilidad individual observada tiene confiabilidad ≈ {rel:.2f};",
        f"  - la sensibilidad modelada no predice la observada (r = {r_me:.2f});",
        f"  - los arquetipos de pitcher no forman grupos claros (silueta máxima {max(arch['silueta'].values()):.2f}).",
        "",
        "Orden de prioridad de esta etapa: 1) evidencia de que el juego cambia, 2) perfil favorable, 3) Stuff+ Physical, 4) Location+, "
        "5) ranking de perfiles, 6) sensibilidad individual (secundaria), 7) contrafactual (explicativo).",
        "",
        "---",
        "",
    ])

    # 2 ----------------------------------------------------------------------
    add([
        "# 2. Evidencia de que el juego cambia en contextos de altitud",
        "",
        "> **En una frase:** los contextos de altitud media y extrema presentan patrones consistentes de menor capacidad de generar whiffs y "
        "ponches, junto con un mayor costo esperado cuando la pelota se pone en juego. Es evidencia contextual fuerte, **no** identificación "
        "causal pura del efecto de la altitud.",
        "",
        "## 2.1 Qué cambia y cuánto",
        "",
        "Diferencias respecto de No Altitude con **efectos fijos de pitcher y bateador** (además de temporada y conteo): comparan al mismo "
        "pitcher contra el mismo bateador. IC 95% con errores agrupados por juego (`reports/hipotesis_altitud.md`).",
        "",
        "| variable | No Altitude (promedio) | Medium − No | Extreme − No [IC 95%] | cambio relativo en Extreme | diferencia cruda E − N |",
        "|---|---:|---:|---|---:|---:|",
    ])
    for lab, name in main_vars:
        fx, raw = _hrow(res, lab)["efectos_fijos"], _hrow(res, lab)["crudo"]["beta_extreme"]
        lo, hi = fx["ic95_extreme"]
        if lab == RV:
            L.append(f"| **{name}** | {fx['media_no_altitude']:+.3f} | {fx['beta_medium']:+.3f} | **{fx['beta_extreme']:+.3f}** [{lo:+.3f}, {hi:+.3f}] | "
                     f"de {fx['media_no_altitude']:+.3f} a {fx['media_no_altitude'] + fx['beta_extreme']:+.3f} | {raw:+.3f} |")
        else:
            L.append(f"| **{name}** | {_pct(fx['media_no_altitude'])} | {100 * fx['beta_medium']:+.1f} pp | **{100 * fx['beta_extreme']:+.1f} pp** "
                     f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {_pct(fx['beta_extreme'] / fx['media_no_altitude'], 0)} | {100 * raw:+.1f} pp |")
    gE, gN = h3["por_categoria"]["Extreme Altitude"], h3["por_categoria"]["No Altitude"]
    add([
        f"| **Valor de evitar contacto (G)** | {gN['G']:.3f} carreras | {h3['medium_menos_no']:+.3f} | **{h3['extreme_menos_no']:+.3f}** "
        f"[{h3['ic95'][0]:+.3f}, {h3['ic95'][1]:+.3f}] | **{_pct(g_ratio, 0)} más** | — |",
        "",
        f"G mide cuántas carreras más le cuesta al pitcher una bola en juego que un whiff en el mismo conteo. En Extreme, convertir una bola en "
        f"juego en un whiff ahorra {gE['G']:.3f} carreras, contra {gN['G']:.3f} en No Altitude: **≈ {_pct(g_ratio, 0)} más**. Medido por turno "
        f"(sección 7.3), cambiar una bola en juego por un ponche vale {_pct(k_gain, 0)} más: el resultado es prácticamente el mismo.",
        "",
        "Tasas crudas por ambiente (sin controles, para dar escala):",
        "",
        "| ambiente | pitches | turnos | whiff/swing | K/PA | BB/PA | BIP/PA | run value por BIP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for a, r in rates.iterrows():
        L.append(f"| {a} | {int(r['pitches']):,} | {int(r['turnos']):,} | {_pct(r['whiff_swing'])} | {_pct(r['k_pa'])} | {_pct(r['bb_pa'])} | "
                 f"{_pct(r['bip_pa'])} | {r['rv_bip']:+.3f} |")
    add([
        "",
        "## 2.2 Consistencia por estadio-temporada",
        "",
        "Cada estadio anónimo de cada temporada se compara por separado contra No Altitude, con los mismos efectos fijos. Si el efecto viniera "
        "de un solo estadio o de una sola temporada, muchas combinaciones tendrían el signo contrario.",
        "",
        "| variable | Extreme: combinaciones en la dirección | Medium: combinaciones en la dirección |",
        "|---|---:|---:|",
    ])
    for lab, name in main_vars:
        L.append(f"| {name} | **{ce[lab]['en_direccion']}/{ce[lab]['total']}** | {cm[lab]['en_direccion']}/{cm[lab]['total']} |")
    add([
        f"| G mayor que en No Altitude | **{sum(v > gN0 for v in ce['G'].values())}/{len(ce['G'])}** | {sum(v > gN0 for v in cm['G'].values())}/{len(cm['G'])} |",
        "",
        f"Nota sobre el conteo: la bitácora previa reportaba 13/13 estadio-temporadas con la reconstrucción de estadios anterior a P0. Con los "
        f"grupos anónimos estables de P0 hay {ce[WH]['total']} combinaciones estadio-temporada de Extreme, y el patrón se mantiene en todas. "
        "El hallazgo no cambia; cambió la unidad de conteo.",
        "",
        f"![Valor de evitar contacto]({_rel(figs['avoid'])})",
        "",
        "## 2.3 Medium se parece mucho a Extreme",
        "",
        f"- Run value por bola en juego: Medium {_fx(res, RV, 'beta_medium'):+.3f} vs. Extreme {_fx(res, RV):+.3f} carreras.",
        f"- Valor de evitar contacto: Medium {h3['medium_menos_no']:+.3f} vs. Extreme {h3['extreme_menos_no']:+.3f} carreras.",
        f"- Ponches por turno: Medium {100 * _fx(res, KP, 'beta_medium'):+.1f} pp vs. Extreme {100 * _fx(res, KP):+.1f} pp.",
        f"- La diferencia más clara está en el whiff por swing: Medium {100 * _fx(res, WH, 'beta_medium'):+.1f} pp vs. Extreme {100 * _fx(res, WH):+.1f} pp. "
        f"En Medium, además, el efecto no aparece en todos los estadios ({cm[WH]['en_direccion']}/{cm[WH]['total']}).",
        "",
        "Como Medium muestra casi el mismo aumento en el daño del contacto, **no podemos afirmar que la altitud extrema cause directamente todos "
        "estos efectos**: una parte puede venir de otras características que comparten esos estadios.",
        "",
        "## 2.4 Lo que NO podemos atribuir causalmente a la altitud",
        "",
        "- Cada categoría agrupa pocos estadios anónimos (5–6 por temporada en Extreme). La altitud no se puede separar de las dimensiones del "
        "parque, el clima, la superficie o la calibración del Trackman de cada estadio.",
        "- Los controles por pitcher y bateador quitan el efecto de quién juega, pero no el de las otras cosas que tienen en común los estadios de altitud.",
        "- El efecto es el promedio de esos estadios; cada uno tiene su propia magnitud (`reports/estadios_anonimos.md`).",
        "- Lo que sí es sólido: el patrón (menos whiffs y ponches, contacto más caro) es grande, aparece en todos los estadio-temporada de Extreme "
        "y no depende de quién lanzó o bateó.",
        "",
        "---",
        "",
    ])

    # 3-4 --------------------------------------------------------------------
    sp = stuff["pitch"]
    y100 = stuff["yoy"]["100"]["res"]
    add([
        "# 3. Stuff+ v1 — benchmark predictivo",
        "",
        f"- Modelo de P(whiff) sobre **todos** los pitches: AUC 0.7315 en su universo; sobre swings, AUC {sp['Stuff+ v1 (P(whiff) por pitch)']['auc_swings']:.3f}.",
        f"- Es el mejor en AUC pitch a pitch porque **ve la ubicación de forma implícita** (ángulos de salida y de llegada) y la categoría de altitud: "
        f"la ubicación explica {_pct(stuff['ubicacion_eta2']['Stuff+ v1'])} de la varianza de sus predicciones dentro de cada tipo de pitch.",
        "- Se conserva **solo como benchmark predictivo**. No es la definición conceptual de Stuff+ (calidad física del pitch).",
        "",
        "# 4. Stuff+ Physical",
        "",
        "Detalle completo, con la auditoría variable por variable: `reports/stuff_plus_physical.md`.",
        "",
        "**Objetivo:** medir la calidad de las características físicas del lanzamiento sin dar crédito directo a la ubicación final ni al ambiente.",
        "",
        "## 4.1 Auditoría de features",
        "",
        "| Variable | Física | Contiene ubicación | Contiene altitud | Posible leakage | Usar |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in spp.FEATURE_AUDIT:
        L.append("| " + " | ".join(row[:6]) + " |")
    add([
        "",
        "## 4.2 Target",
        "",
        "Se usa **P(whiff | swing)**: la capacidad de hacer fallar al bateador **una vez que decide hacer swing**. No mide la capacidad de "
        "inducir el swing, que depende sobre todo de la ubicación y del conteo (Location+). Candidatos descartados:",
        "",
        "- whiff por pitch: mezcla Stuff con Location; es el target de v1;",
        "- evitar contacto: es el mismo target;",
        "- valor esperado sin ubicación: sin ubicación, el swing y el take son casi ruido, y el componente de contacto no predice el año siguiente.",
        "",
        "## 4.3 ¿Stuff observado o neutralizado?",
        "",
        "El movimiento (IVB, HB) cambia con la densidad del aire.",
        "",
        "- Es parte real de lo que el bateador ve (\"stuff observado\").",
        "- Pero si se comparan pitchers que lanzaron en ambientes distintos, castiga a quien lanzó en altitud; por eso hace falta un \"stuff neutralizado\".",
        "",
        "No se eliminó el movimiento. Se calcularon tres versiones con el mismo modelo:",
        "",
        "| versión | qué hace con el ambiente | r con % de pitches en Extreme | estabilidad (mín. 100) | whiff/swing t+1 |",
        "|---|---|---:|---:|---:|",
    ])
    for k, lab, how in [("stuff_plus_physical_observado", "Observado", "nada (lo que el bateador vio)"),
                        ("stuff_plus_physical", "**Oficial**", "ajuste simple temporada × estadio anónimo"),
                        ("stuff_plus_physical_neutral", "Neutral", "contrafactual físico a No Altitude")]:
        L.append(f"| {lab} | {how} | {stuff['pitcher'][k]['r_pct_extreme']:+.3f} | {y100[k]['estabilidad']['r']:.3f} | {y100[k]['whiff_swing_t1']['r']:.3f} |")
    add([
        "",
        f"- La versión observada castiga al pitcher por lanzar en altitud (r = {stuff['pitcher']['stuff_plus_physical_observado']['r_pct_extreme']:+.2f} "
        "con su % de pitches en Extreme). Las otras dos lo neutralizan.",
        f"- La oficial (ajuste simple) predice el whiff/swing del año siguiente mejor que la neutral.",
        f"- Las dos correlacionan {stuff['r_oficial_neutral']:.3f}, por debajo del umbral de 0.98 fijado antes, así que se conservan ambas: la "
        "**oficial para rankear** y la neutral solo para explicar el mecanismo físico.",
        "",
        "---",
        "",
    ])

    # 5 ----------------------------------------------------------------------
    lv = loc["validacion"]["100"]
    add([
        "# 5. Location+",
        "",
        "Detalle: `reports/location_plus_v2.md`. **Pregunta:** dado el tipo de pitch y el contexto (conteo, manos), ¿qué tan favorable es la "
        "ubicación donde terminó el pitch?",
        "",
        "- Usa el mismo árbol de probabilidades de Pitching+ (swing, take, whiff, foul, resultado en juego), valuado en carreras.",
        "- Sus variables son **solo** la ubicación (lado relativo al bateador y altura), el conteo, el tipo de pitch y las manos. No usa velocidad ni movimiento.",
        "- El puntaje por pitch es el valor esperado en esa ubicación menos el promedio del mismo contexto (tipo × conteo × manos): solo "
        "cuenta dónde terminó el pitch, no qué pitch era ni en qué conteo.",
        f"- Estabilidad año a año {lv['estabilidad']['r']:.2f}.",
        f"- Predice el BB% del año siguiente (r = {lv['bb_t1']['r']:+.2f}) **mejor que el propio BB%** (r = {lv['bb_propio_t1']['r']:+.2f}).",
        "",
        "Valor por región de la zona en cada ambiente, con efectos fijos de pitcher, bateador, temporada y conteo. Carreras por 100 pitches; "
        "positivo = malo para el pitcher.",
        "",
        "| región | Δ en No Altitude | Medium − No | Extreme − No |",
        "|---|---:|---:|---:|",
    ])
    for r in loc["regiones"]:
        L.append(f"| {r['region']} | {r['delta_medio_no_altitude_por_100']:+.2f} | {r['medium_menos_no_por_100']:+.2f} ± {1.96 * r['ee_medium']:.2f} | "
                 f"{r['extreme_menos_no_por_100']:+.2f} ± {1.96 * r['ee_extreme']:.2f} |")
    add([
        "",
        "Lectura: en altitud **los pitches en el corazón de la zona son los que más valor pierden**, porque ahí hay más contacto y el contacto "
        "es más caro; los pitches fuera de la zona casi no cambian. Es una asociación, no una regla de ubicación.",
        "",
        "Hay mapas de calor por tipo de pitch, mano del bateador y grupo de conteo en `reports/figures/p2/location_*.png`. Muestran zonas que "
        "**suelen** ser más favorables para cada tipo de pitch; ninguna zona garantiza un out.",
        "",
        "![Location+ slider](figures/p2/location_slider.png)",
        "",
        "---",
        "",
        "# 6. Comparación de métricas",
        "",
        "## 6.1 Stuff+",
        "",
        "| métrica | AUC whiff \\| swing | ubicación en la predicción | estabilidad (mín. 100) | whiff/swing t+1 | carreras t+1 | separación conceptual |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for key, name, auc_key, eta, sep in [
            ("stuff_plus_v1", "Stuff+ v1 benchmark", "Stuff+ v1 (P(whiff) por pitch)", "Stuff+ v1", "mezcla física, ubicación y altitud"),
            ("stuff_plus_v4", "Stuff+ v4", "Stuff+ v4", "Stuff+ v4", "física; neutral por contrafactual; incluye SpeedDrop y lado del bateador"),
            ("stuff_plus_physical", "**Stuff+ Physical**", "Stuff+ Physical (puntaje, lados marginalizados)", "Stuff+ Physical",
             "física; ajuste simple por estadio; lado del bateador marginalizado")]:
        L.append(f"| {name} | {sp[auc_key]['auc_swings']:.3f} | {_pct(stuff['ubicacion_eta2'][eta])} | {y100[key]['estabilidad']['r']:.3f} | "
                 f"{y100[key]['whiff_swing_t1']['r']:.3f} | {y100[key]['carreras_t1']['r']:.3f} | {sep} |")
    d4 = y100["stuff_plus_physical"]["delta_vs_v4_whiff"]
    add([
        "",
        f"**Decisión:** Stuff+ Physical es el Stuff+ conceptual del proyecto. No se eligió por AUC: pierde AUC frente a v1, porque ya no ve la "
        f"ubicación. Lo que cuenta es que es más estable, predice el whiff/swing del año siguiente igual que v1 y mejor que v4 "
        f"(Δr = {d4['delta_r']:+.3f}, P = {_pct(d4['p_mejor'], 0)}), es interpretable (solo física) y es útil para scouting: dice cuánto "
        "swing-and-miss genera el pitch por sí mismo.",
        "",
        "## 6.2 Todas las métricas por pitcher: estabilidad y relación con carreras en altitud del año siguiente",
        "",
        f"Pitcher-temporadas con ≥ 100 pitches. *Estabilidad* = correlación año a año del valor ajustado (sin shrinkage). La última columna es la "
        f"correlación con las carreras evitadas por 100 turnos en juegos de Medium y Extreme del año siguiente (≥ {pp.MIN_ALT_PA_T1} turnos). "
        "El signo esperado depende de si la métrica es \"mayor = mejor\".",
        "",
        "| métrica | mayor es mejor | estabilidad [IC 95%] | r con carreras en altitud t+1 [IC 95%] | lectura |",
        "|---|:---:|---|---|---|",
    ])
    for _, r in stab.iterrows():
        s_ok = r["estabilidad_cruda"] >= 0.35
        p_ok = (r["r_lo"] > 0) if r["mayor_es_mejor"] else (r["r_hi"] < 0)
        reading = "estable y predictiva" if s_ok and p_ok else ("estable, relación débil con carreras" if s_ok else "poco estable (casi ruido de muestra)")
        L.append(f"| {r['etiqueta']} | {'sí' if r['mayor_es_mejor'] else 'no'} | {r['estabilidad_cruda']:.2f} [{r['estabilidad_lo']:.2f}, {r['estabilidad_hi']:.2f}] | "
                 f"{r['r_carreras_altitud_t1']:+.2f} [{r['r_lo']:+.2f}, {r['r_hi']:+.2f}] | {reading} |")
    add([
        "",
        "Lo importante para el perfil:",
        "",
        "- **Whiff/swing, K% y BIP% son estables.**",
        "- El daño por bola en juego (run value por BIP, hard contact%, HR/BIP) **casi no se repite de un año a otro**: en estos datos, limitar el "
        "daño una vez que hay contacto es sobre todo ruido y contexto, no una habilidad estable del pitcher.",
        "- Por eso, en el perfil esa parte entra muy encogida hacia el promedio.",
        "",
        "## 6.3 Pitching+: índice principal simple",
        "",
        "En P0 el ajuste simple por entorno predijo igual o mejor que el contrafactual físico. Por eso **el Pitching+ de P2 es el de ajuste simple "
        "por temporada × estadio anónimo (M1b)**. El Pitching+ neutral físico se conserva como explicación física, herramienta contrafactual y "
        f"análisis de mecanismos: *{CF_WARNING}*",
        "",
        "---",
        "",
    ])

    # 7 ----------------------------------------------------------------------
    cal = chk["calibracion"].set_index("ambiente")
    add([
        "# 7. Valor de evitar contacto",
        "",
        "## 7.1 Formalización",
        "",
        "    Daño esperado de contacto por pitch = P(BIP) × E[RunValue | BIP]",
        "",
        "En el árbol de probabilidades de Pitching+:",
        "",
        "    P(BIP) = P(swing) · (1 − P(whiff | swing)) · (1 − P(foul | contacto))",
        "    E[Δ | BIP] = Σ_k P(k | BIP) · Δ(k, conteo),   k ∈ {out, 1B, 2B, 3B, HR}",
        "    componente de bola en juego de E[Δ] = Σ_k P(k) · Δ(k) = P(BIP) · E[Δ | BIP]",
        "",
        f"Verificación numérica sobre las predicciones OOF de Pitching+: la identidad se cumple con error máximo {chk['max_abs_identidad']:.1e}, y el "
        f"producto del árbol reproduce P(BIP) con error máximo {chk['max_abs_arbol']:.1e}. **Evitar contacto** (bajar P(BIP)) y **contacto menos "
        "peligroso** (bajar E[Δ | BIP]) son las dos ramas del mismo término.",
        "",
        "## 7.2 Cómo cambia cada componente entre ambientes",
        "",
        "| ambiente | P(BIP) por pitch [IC 95%] | E[RV \\| BIP] en carreras [IC 95%] | daño por 100 pitches [IC 95%] | P(BIP) por turno |",
        "|---|---|---|---|---:|",
    ])
    for a in ["No Altitude", "Medium", "Extreme"]:
        r = dcat.loc[a]
        L.append(f"| {a} | {_pct(r['p_bip_por_pitch'], 2)} [{_pct(r['p_bip_por_pitch_lo'], 2)}, {_pct(r['p_bip_por_pitch_hi'], 2)}] | "
                 f"{r['rv_por_bip']:+.3f} [{r['rv_por_bip_lo']:+.3f}, {r['rv_por_bip_hi']:+.3f}] | "
                 f"{100 * r['dano_bip_por_pitch']:+.2f} [{100 * r['dano_bip_por_pitch_lo']:+.2f}, {100 * r['dano_bip_por_pitch_hi']:+.2f}] | {_pct(r['p_bip_por_turno'])} |")
    add([
        "",
        "Descomposición del aumento del daño frente a No Altitude (carreras por 100 pitches):",
        "",
        "| ambiente | total [IC 95%] | por más bolas en juego | por más daño por bola en juego | interacción |",
        "|---|---|---:|---:|---:|",
    ])
    for c in ["Medium", "Extreme"]:
        d = dec[c]
        L.append(f"| {c} | {100 * d['total']:+.2f} [{100 * d['ic95_total'][0]:+.2f}, {100 * d['ic95_total'][1]:+.2f}] | {100 * d['por_probabilidad']:+.2f} | "
                 f"{100 * d['por_dano']:+.2f} | {100 * d['interaccion']:+.2f} |")
    add([
        "",
        f"![Daño de contacto por ambiente]({_rel(figs['damage'])})",
        "",
        "**Lectura:** en altitud casi no cambia *cuántas* bolas en juego hay por pitch; cambia *cuánto cuesta cada una*. Por eso evitar el contacto "
        "vale más aunque el pitcher no permita más contacto que antes.",
        "",
        "Calibración del árbol por ambiente (el modelo de Pitching+ no recibe el ambiente como variable):",
        "",
        "| ambiente | P(BIP) predicho | observado | E[Δ \\| BIP] predicho | observado |",
        "|---|---:|---:|---:|---:|",
    ])
    for a, r in cal.iterrows():
        L.append(f"| {a} | {_pct(r['p_bip_predicho'], 2)} | {_pct(r['p_bip_observado'], 2)} | {r['delta_bip_predicho']:+.3f} | {r['delta_bip_observado']:+.3f} |")
    add([
        "",
        "El árbol acierta la probabilidad de contacto en todos los ambientes, pero **subestima el daño por bola en juego en altitud**. La física "
        "del pitch explica solo una parte del aumento; el resto viene del ambiente (vuelo de la pelota, parque). Es otra razón para medir el "
        "ambiente con el ajuste simple por estadio.",
        "",
        "## 7.3 Pesos lineales por ambiente (valor RE24 de cada resultado del turno)",
        "",
        "| ambiente | ponche | base por bolas | golpeado | bola en juego | ponche − bola en juego [IC 95%] | BB − bola en juego | aire − rodado | duro − no duro |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ])
    for a, r in lw.iterrows():
        L.append(f"| {a} | {r['v_K']:+.3f} | {r['v_BB']:+.3f} | {r['v_HBP']:+.3f} | {r['v_BIP']:+.3f} | {r['K_menos_BIP']:+.3f} "
                 f"[{r['K_menos_BIP_lo']:+.3f}, {r['K_menos_BIP_hi']:+.3f}] | {r['BB_menos_BIP']:+.3f} | {gb_gap[a]:+.3f} | {hard_gap[a]:+.3f} |")
    add([
        "",
        f"- **Ponche en lugar de bola en juego:** vale {abs(kb['No Altitude']):.3f} carreras en No Altitude y {abs(kb['Extreme']):.3f} en Extreme "
        f"(**{_pct(k_gain, 0)} más**). Es la misma conclusión que G (≈ {_pct(g_ratio, 0)}), medida por turno en vez de por swing.",
        f"- **Base por bolas en lugar de bola en juego:** cuesta {bb['No Altitude']:.3f} en No Altitude y {bb['Extreme']:.3f} en Extreme. En altitud la "
        "base por bolas es relativamente *menos* grave, porque la alternativa (la bola en juego) es más cara.",
        f"- **Rodado en lugar de batazo al aire:** ahorra {gb_gap['No Altitude']:.3f} carreras en No Altitude y {gb_gap['Extreme']:.3f} en Extreme: en "
        "altitud el rodado protege más.",
        f"- **Contacto duro:** la diferencia duro − no duro pasa de {hard_gap['No Altitude']:.3f} a {hard_gap['Extreme']:.3f}.",
        "",
        "---",
        "",
    ])

    # 8 ----------------------------------------------------------------------
    add([
        "# 8. Pitch mix",
        "",
        "Efecto de Extreme por tipo de pitch frente a No Altitude:",
        "",
        "- los resultados, con efectos fijos de pitcher, bateador, temporada y conteo;",
        "- el movimiento y la velocidad, comparando dentro del mismo pitcher × tipo × temporada.",
        "",
        "**Es el promedio de cada tipo de pitch, no el efecto causal de elegir ese pitch**: quién lanza cada tipo, y cuándo, también cambia.",
        "",
        "| tipo | pitches | whiff/swing en No Altitude | Δ whiff/swing | Δ BIP/swing | Δ run value por BIP | Δ IVB (pulg.) | Δ HB absoluto (pulg.) | costo en Extreme (carreras/100 pitches) | costo en Medium |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, r in ty.iterrows():
        L.append(f"| {r['tipo']} | {r['pitches']:,} | {_pct(r['whiff_swing_no'])} | {100 * r['whiff_swing_E']:+.1f} ± {196 * r['whiff_swing_E_se']:.1f} pp | "
                 f"{100 * r['bip_swing_E']:+.1f} pp | {r['rv_bip_E']:+.3f} ± {1.96 * r['rv_bip_E_se']:.3f} | {r['ivb_E']:+.1f} (de {r['ivb_no']:.1f}) | "
                 f"{r['hb_abs_E']:+.1f} (de {r['hb_abs_no']:.1f}) | {r['delta_por_100_E']:+.2f} ± {1.96 * r['delta_por_100_E_se']:.2f} | {r['delta_por_100_M']:+.2f} |")
    best = ty.sort_values("delta_por_100_E").iloc[0]
    worst = ty.sort_values("whiff_swing_E").iloc[0]
    add([
        "",
        f"- **El que más whiff pierde:** {worst['tipo']} ({100 * worst['whiff_swing_E']:+.1f} pp), con la mayor pérdida de movimiento vertical "
        f"inducido ({worst['ivb_E']:+.1f} pulgadas).",
        "- **Los que más movimiento horizontal pierden:** sinker y changeup (≈ −4 pulgadas).",
        f"- **El de menor costo total en Extreme:** {best['tipo']} ({best['delta_por_100_E']:+.2f} carreras por 100 pitches).",
        "- **Los que casi no pierden whiff:** cutter y splitter; el splitter tiene poca muestra.",
        "- **Los que más daño permiten cuando hay contacto:** splitter y curveball, con intervalos amplios.",
        "",
        "**¿Hay una mezcla más robusta?** Para cada pitcher se calculó el costo esperado de su mezcla (Σ participación × costo del tipo):",
        "",
        f"- se relaciona con la sensibilidad que predice el modelo (r = {mix['r_costo_mezcla_s_mod']:+.2f});",
        f"- **no** se relaciona con la sensibilidad observada (r = {mix['r_costo_mezcla_s_emp']:+.2f}, {mix['n']} pitchers).",
        "",
        "Una mezcla cargada a sliders y cutters pierde menos *en promedio*, pero los datos no muestran que esos pitchers rindan relativamente "
        "mejor en altitud.",
        "",
        "---",
        "",
    ])

    # 9 ----------------------------------------------------------------------
    add([
        "# 9. Perfil favorable para ambientes de altitud",
        "",
        "**Pregunta:** ¿qué características parecen especialmente valiosas cuando el contexto penaliza más el contacto? Se respondió con tres "
        "piezas, sin asumir la respuesta:",
        "",
        "1. **Contabilidad de carreras por ambiente** (pesos lineales, sección 7.3): cuánto vale cada resultado del turno en cada ambiente.",
        "2. **Estabilidad y poder predictivo** de cada característica (sección 6.2): cuáles son habilidad del pitcher y cuáles ruido.",
        "3. **Interacción con el ambiente** (abajo): si ser mejor en una característica previene relativamente más carreras en altitud.",
        "",
        "## 9.1 ¿Vale más cada característica en altitud? (asociación)",
        "",
        "Diseño cruzado: la característica del pitcher se mide en una mitad de sus juegos y las carreras de sus turnos en la otra mitad, para que "
        "el mismo ruido no aparezca en ambos lados.",
        "",
        "- Pendiente = carreras evitadas por 100 turnos por cada desviación estándar de mejora.",
        "- IC 95% por bootstrap de pitchers.",
        "",
        "| característica (+1 sd = mejor) | No Altitude | Medium | Extreme | Extreme − No [IC 95%] | P(mayor en Extreme) |",
        "|---|---:|---:|---:|---|---:|",
    ])
    for m, r in ir.iterrows():
        L.append(f"| {LAB_INT[m]} | {r['pend_No Altitude']:+.2f} | {r['pend_Medium']:+.2f} | {r['pend_Extreme']:+.2f} | "
                 f"{r['dif_Extreme']:+.2f} [{r['dif_Extreme_lo']:+.2f}, {r['dif_Extreme_hi']:+.2f}] | {_pct(r['p_mayor_Extreme'], 0)} |")
    sig = [LAB_INT[m] for m, r in ir.iterrows() if r["dif_Extreme_lo"] > 0]
    add([
        "",
        f"![Interacciones]({_rel(figs['inter'])})",
        "",
        f"- Diferencias Extreme − No con IC que excluye 0: **{', '.join(sig) if sig else 'ninguna'}**.",
        "- Generar whiffs y ponches, y el Stuff+ Physical que los produce, tiende a valer más en Extreme (P ≥ 95%), como predice la contabilidad de carreras.",
        "- Son 9 características × 2 ambientes: con tantas comparaciones, algún intervalo puede excluir 0 por azar. Se leen como un **patrón "
        "exploratorio coherente**, no como pruebas individuales.",
        "- Dar pocas bases por bolas también aparece con más valor en Extreme, al revés de lo que indican los pesos lineales (allí una BB es "
        "relativamente menos grave). Lo más probable es que los pitchers con control también sean mejores en otras cosas, así que no se "
        "interpreta como un efecto propio del control.",
        "- El daño por bola en juego, el contacto duro y el porcentaje de rodados no muestran con claridad un valor marginal distinto.",
        "",
        "## 9.2 El perfil ideal, según los datos",
        "",
        "| característica | ¿más valiosa en altitud? | ¿es habilidad estable? | conclusión |",
        "|---|---|---|---|",
        f"| Whiff/Swing, K% | sí: un ponche en vez de una BIP vale {_pct(k_gain, 0)} más; pendiente mayor en Extreme | sí (≈ 0.45–0.51) | **núcleo del perfil** |",
        "| Pocas bolas en juego (BIP%) | sí (es la otra cara del ponche) | sí (≈ 0.52) | **núcleo del perfil** |",
        "| Stuff+ Physical | sí (pendiente mayor en Extreme) | sí (≈ 0.76) | **fuente física del whiff**; entra a través del whiff y el K% |",
        "| Location+ / pocas BB | ambiguo (los pesos dicen que la BB pesa relativamente menos) | sí (≈ 0.54 / 0.35) | importante, pero no especialmente más en altitud |",
        "| Poco daño por BIP / poco contacto duro | el daño pesa más en altitud | **no** (≈ 0 / 0.11) | valioso, pero no medible como habilidad: entra muy encogido |",
        f"| Rodados (GB%) | el rodado protege más (aire − rodado: {gb_gap['No Altitude']:.2f} → {gb_gap['Extreme']:.2f}) | sí (≈ 0.41) | ayuda poco a predecir carreras; secundario |",
        "",
        "## 9.3 Puntaje compuesto sin pesos arbitrarios: AEPP",
        "",
        "Las carreras de un turno se descomponen **exactamente** por su resultado:",
        "",
        "    R = Σ_o r_o · v_o,   o ∈ {K, BB, HBP, BIP},   Σ_o r_o = 1",
        "    R_i − R̄ = Σ_{o≠BIP} (r_o,i − r̄_o)·(v_o − v_BIP) + r_BIP,i · d_i",
        "    AEPP_i = −100 · (R_i − R̄)   [carreras evitadas por 100 turnos frente a un pitcher promedio]",
        "",
        "- r_o,i: tasas del pitcher por turno (K%, BB%, HBP%; BIP% = 1 − las otras), ajustadas por entorno y con shrinkage.",
        "- d_i: su daño por bola en juego relativo a la liga, con shrinkage. Casi todo se encoge al promedio, porque no es estable.",
        "- v_o: **pesos lineales de los ambientes de altitud (Medium + Extreme)** medidos en los datos (sección 7.3).",
        "",
        "Ningún peso se eligió a mano: el peso de cada característica es lo que vale en carreras en ese ambiente. También se exploraron PCA y "
        "clustering (sección 11): describen cómo se agrupan las características, pero no dan pesos con significado en carreras. Como "
        "comparación simple se usa K−BB% con shrinkage (sección 9.4). No se ajustó una regresión para elegir pesos: con dos pares de "
        "temporadas habría riesgo de sobreajuste, y la contabilidad de carreras ya da pesos con justificación.",
        "",
        "Antes de calcularlo se fijó una regla para usar Stuff+ Physical y Location+ como covariables (media a priori del shrinkage de K%, BB% "
        "y daño por BIP): se adoptan solo si mejoran la predicción del mismo componente en t+1 con P ≥ 0.80.",
        "",
        "| componente | Δr t+1 con covariables [IC 95%] | P | ¿se adopta? |",
        "|---|---|---:|---|",
    ])
    for m, v in res["covariables"].items():
        d = v[100]["delta"]
        L.append(f"| {m} | {d['delta_r']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {_pct(d['p_mejor'], 0)} | {'sí' if v['adoptar'] else 'no'} |")
    add([
        "",
        "Ninguna mejora lo suficiente, así que el perfil usa shrinkage simple. Stuff+ Physical y Location+ aparecen en la ficha del pitcher pero "
        "no cambian su puntaje: su información ya está en el whiff, el K% y el BB% observados.",
        "",
        "## 9.4 Validación del perfil",
        "",
        f"Se predicen, con la pitcher-temporada t, las carreras evitadas por 100 turnos en t+1 (ajustadas por entorno): en juegos de altitud "
        f"(≥ {pp.MIN_ALT_PA_T1} turnos de altitud en t+1) y en todos los juegos.",
        "",
    ])
    for mn, v in [(100, v100), (300, v300)]:
        add([f"**Mínimo {mn} pitches** ({v['n_pares']} pares; {v['n_pares_altitud']} con suficientes turnos de altitud)", "",
             "| puntaje | estabilidad | carreras t+1 (todos) | carreras t+1 en altitud | Δ vs Pitching+ en altitud (P) |", "|---|---|---|---|---|"])
        for c, lab in val["candidatos"].items():
            x = v[c]
            dd = x.get("delta_altitud_vs_pitching_plus")
            dtxt = f"{dd['delta_r']:+.3f} ({_pct(dd['p_mejor'], 0)})" if dd else "referencia"
            L.append(f"| {lab} | {pc.fmt_ci(x['estabilidad'])} | {pc.fmt_ci(x['carreras_t1'])} | {pc.fmt_ci(x['carreras_altitud_t1'])} | {dtxt} |")
        dn = v["aepp"]["delta_altitud_vs_pesos_no"]
        add(["", f"AEPP con pesos de altitud vs. el mismo perfil con pesos de No Altitude, prediciendo carreras en altitud: "
                 f"Δr = {dn['delta_r']:+.3f} [{dn['ci95'][0]:+.3f}, {dn['ci95'][1]:+.3f}], P = {_pct(dn['p_mejor'], 0)}.", ""])
    add([
        "**Lectura honesta:**",
        "",
        f"- El perfil por resultados (AEPP, o simplemente K−BB% con shrinkage) predice las carreras en altitud del año siguiente mejor que "
        f"Pitching+ (r ≈ {v100['aepp']['carreras_altitud_t1']['r']:.2f} vs. {v100['pitching_plus']['carreras_altitud_t1']['r']:.2f}).",
        "- AEPP **no** supera a K−BB%: aporta interpretabilidad (está en carreras y separa ponches, bases por bolas y daño), no más precisión.",
        "- Usar los pesos de altitud en lugar de los de No Altitude **no** mejora la predicción de forma detectable. El perfil favorable para "
        "altitud y el perfil bueno en general ordenan casi igual a los pitchers.",
        "",
        "## 9.5 K al nivel correcto (turno)",
        "",
        "- **Unidad:** el turno, representado por su pitch terminal (ponche, base por bolas, golpeado o bola en juego). Cada ponche cuenta una "
        "vez; ningún resultado se copia a los pitches del turno (verificado en `tests/test_p2.py`).",
        "- **Features:** para el efecto del ambiente, efectos fijos de pitcher, bateador y temporada (sección 2). Para el pitcher, su K% por "
        "turno ajustado por entorno, con shrinkage normal-normal (τ ≈ 0.04–0.045 por temporada).",
        "- **Temporalidad:** la validación usa la temporada t para predecir t+1, sin información futura.",
        "- **Pesos:** cada turno pesa 1; los promedios por pitcher se ponderan por sus turnos.",
        f"- **K% por ambiente (crudo):** No Altitude {_pct(rates.loc['No Altitude', 'k_pa'])}, Medium {_pct(rates.loc['Medium', 'k_pa'])}, "
        f"Extreme {_pct(rates.loc['Extreme', 'k_pa'])}. Con controles, {100 * _fx(res, KP):+.1f} pp en Extreme.",
        "",
        "---",
        "",
    ])

    # 10 ---------------------------------------------------------------------
    add([
        "# 10. Ranking principal de pitchers",
        "",
        "## Pitchers con perfil más favorable para ambientes de altitud",
        "",
        f"**Altitude Environment Pitcher Profile (AEPP).** Elegibles: ≥ {pp.MIN_PITCHES_RANK} pitches y ≥ {pp.MIN_PA_RANK} turnos en 2024–2026 "
        f"({len(e)} pitchers). No usa la sensibilidad observada. Los intervalos de 80% del puntaje y del lugar salen de {pp.N_DRAWS:,} extracciones "
        "de los posteriores de cada componente.",
        "",
        "> Lectura correcta: *su perfil presenta características que históricamente son más favorables en estos contextos.* No significa que el "
        "pitcher vaya a rendir mejor en altitud con seguridad, ni que sea \"resistente a la altitud\".",
        "",
        "| # | pitcher | AEPP [80%] | lugar [80%] | P(top 25) | pitches (Extreme / Medium) | K% | BB% | whiff/swing | Stuff+ Phys | Location+ | por qué aparece alto |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for _, r in e.head(25).iterrows():
        expl = r["explicacion"].split(" Su perfil presenta")[0].replace("Aparece alto porque: ", "")
        L.append(f"| {int(r['rank'])} | {r['pitcher_anon_id']} | {r['aepp']:+.2f} [{r['aepp_lo80']:+.2f}, {r['aepp_hi80']:+.2f}] | "
                 f"{int(r['rank_lo80'])}–{int(r['rank_hi80'])} | {_pct(r['p_top25'], 0)} | {int(r['n_pitches']):,} ({int(r['pitches_extreme']):,} / "
                 f"{int(r['pitches_medium']):,}) | {_pct(r['k_pct'])} | {_pct(r['bb_pct'])} | {_pct(r['whiff_swing'])} | "
                 f"{r['stuff_plus_physical']:.0f} | {r['location_plus']:.0f} | {expl} |")
    add([
        "",
        f"![Ranking]({_rel(figs['ranking'])})",
        "",
        f"La tabla completa ({len(e)} pitchers, con ficha y explicación) está en `data/web/pitcher_rankings.csv` y `data/web/pitcher_altitude_profiles.csv`.",
        "",
        "## 10.1 Ficha de pitcher",
        "",
        "Cada fila de `pitcher_altitude_profiles.csv` contiene:",
        "",
        "- **Muestra:** pitches, turnos y swings, y pitches por ambiente.",
        "- **Índices:** Stuff+ Physical, Location+ y Pitching+, cada uno con intervalo de 80%.",
        "- **Métricas de resultado:** whiff/swing, whiff/pitch, K%, BB%, HBP%, contact%, BIP%, run value/BIP, hard contact%, GB% y HR/BIP. Cada "
        "una en versión cruda, ajustada y con shrinkage, con su intervalo, su muestra y su factor de encogimiento.",
        "- **Perfil:** mezcla de pitches; AEPP con intervalo y lugar; arquetipo; percentiles; y una explicación en texto.",
        "",
        "## 10.2 ¿Quién gana más con los pesos de altitud? (bono de altitud)",
        "",
        f"Bono = AEPP con pesos de altitud − AEPP con pesos de No Altitude. Es pequeño (sd {e['bono_altitud'].std():.2f} carreras por 100 turnos, "
        f"contra {e['aepp'].std():.2f} del AEPP). Depende sobre todo del K% (r = {bono_k:.2f}) y en parte del BB% "
        f"(r = {e['bono_altitud'].corr(e['bb_pct']):+.2f}): en altitud, ponchar vale más y la base por bolas pesa relativamente menos que una "
        "bola en juego, así que el ponchador con algo de descontrol pierde menos que en No Altitude.",
        "",
        "| pitcher | lugar AEPP | bono [80%] | K% | BB% |",
        "|---|---:|---|---:|---:|",
    ])
    for _, r in e.sort_values("bono_altitud", ascending=False).head(10).iterrows():
        L.append(f"| {r['pitcher_anon_id']} | {int(r['rank'])} | {r['bono_altitud']:+.2f} [{r['bono_altitud_lo80']:+.2f}, {r['bono_altitud_hi80']:+.2f}] | "
                 f"{_pct(r['k_pct'])} | {_pct(r['bb_pct'])} |")
    add(["", "---", ""])

    # 11 ---------------------------------------------------------------------
    cent = arch["centroides"]
    pv, ld = arch["pca"]["varianza_explicada"], arch["pca"]["cargas"]
    add([
        "# 11. Arquetipos de pitchers",
        "",
        f"k-means sobre {', '.join(pp.ARCH_FEATURES)} (estandarizados y orientados: + = mejor para el pitcher), con k = 2…7. Regla fijada antes: "
        "usar nombres de arquetipo solo si la silueta es ≥ 0.25 y la estabilidad (ARI en bootstrap) es ≥ 0.6.",
        "",
        "| k | " + " | ".join(str(k) for k in arch["silueta"]) + " |",
        "|---|" + "---:|" * len(arch["silueta"]),
        "| silueta | " + " | ".join(f"{v:.3f}" for v in arch["silueta"].values()) + " |",
        "",
        f"El mejor k es {arch['k']}, con silueta {arch['silueta'][arch['k']]:.3f} y ARI de estabilidad {arch['ari_estabilidad']:.2f}. "
        + ("**Se usan nombres.**" if arch["nombres_validos"] else
           "**Los datos no producen grupos claros, así que no se usan los nombres de arquetipo propuestos (Whiff dominant, Contact manager, etc.).**"),
        "",
        "| perfil | n | centroide (z orientado) | AEPP medio | bono de altitud medio |",
        "|---|---:|---|---:|---:|",
    ])
    for _, r in cent.iterrows():
        z = ", ".join(f"{c} {r[c]:+.2f}" for c in pp.ARCH_FEATURES)
        L.append(f"| {r['nombre']} | {int(r['n'])} | {z} | {r['aepp_medio']:+.2f} | {r['bono_altitud_medio']:+.2f} |")
    add([
        "",
        f"Componentes principales: PC1 explica {_pct(pv[0], 0)} de la varianza y PC2 {_pct(pv[1], 0)}. Cargas de PC1: "
        + ", ".join(f"{c} {ld.loc[c, 'PC1']:+.2f}" for c in pp.ARCH_FEATURES) + ".",
        "",
        "**Lectura:** en lugar de grupos hay un **continuo** con un eje dominante: *swing-and-miss / Stuff* en un extremo y *control / ubicación* "
        "en el otro. Los pitchers del lado de swing-and-miss tienen en promedio un bono de altitud positivo, y los del lado de control uno "
        "negativo. Es coherente con la sección 7.3: en altitud el ponche vale más y la base por bolas pesa relativamente menos.",
        "",
        "---",
        "",
    ])

    # 12-15 ------------------------------------------------------------------
    sm = sens["s_mod"]
    pr = sens["por_rol"]
    pl = sens["por_equipo_local"]
    sel = sens["seleccion"]
    fac = sens["factores"]
    need = sens["pitches_necesarios"]
    ry = sens["r_entre_temporadas"]
    add([
        "# 12. Sensibilidad modelada",
        "",
        f"*{CF_WARNING}*",
        "",
        "- **Definición:** para cada pitch, el valor esperado proyectado a Extreme menos el proyectado a No Altitude (mismo pitch, contrafactual "
        "físico), promediado por pitcher. Todos los pitchers reciben los mismos ratios de densidad del aire por tipo de pitch.",
        f"- Dispersión entre pitchers: sd = {sm['s_mod_rel'].std():.2f} carreras por 100 pitches (relativa a la liga).",
        f"- La mezcla de tipos de pitch explica R² = 0.44 de la sensibilidad modelada (`reports/sensibilidad_altitud.md`). El costo de la mezcla de la "
        f"sección 8 correlaciona r = {mix['r_costo_mezcla_s_mod']:+.2f} con ella.",
        "- Es un **perfil de pitches** (qué tipo de pitches pierde más con menos densidad), no una medición de cómo reacciona el pitcher.",
        "- Usos válidos: visualizar cambios físicos, simular ambientes, explicar qué variables cambian y separar mecanismos.",
        "",
        "# 13. Sensibilidad observada",
        "",
        "- **Definición:** S_emp = 100·[media_Extreme(Δ̃) − media_No(Δ̃)], con Δ̃ = valor realizado del pitch menos el promedio de su temporada y "
        "categoría. Positivo = le fue relativamente peor en Extreme.",
        f"- Pitchers con ≥ 150 pitches en Extreme y en No Altitude: {sens['info']['n_pitchers']}.",
        f"- La sd observada ({sens['info']['sd_observada']:.2f}) es casi igual al error estándar típico ({sens['info']['se_medio']:.2f}): la variación "
        "entre pitchers es del mismo tamaño que el ruido.",
        f"- **Confiabilidad (mitades de juegos, Spearman–Brown) = {rel:.2f}.**",
        f"- Entre temporadas (pitchers con ≥ 75 pitches en cada ambiente en dos temporadas): r = {ry['r']:+.2f} [{ry['ci95'][0]:+.2f}, {ry['ci95'][1]:+.2f}], "
        f"{sens['n_pares_temporadas']} pares.",
        "",
        "## 13.1 ¿Por qué la confiabilidad es tan baja?",
        "",
        "| posible causa | evidencia | lectura |",
        "|---|---|---|",
        f"| tamaño de muestra | mediana {int(s_tab['n_extreme'].median())} pitches en Extreme; para confiabilidad 0.5 harían falta ≈ {need.get(0.5, float('nan')):,.0f}, y para 0.7 ≈ {need.get(0.7, float('nan')):,.0f} | **causa principal**: ningún pitcher tiene esa muestra |",
        f"| calidad de los rivales | r(rivales más fuertes en Extreme que en No, S_emp) = {fac['rivales_E_menos_N']['r_con_s_emp']:+.2f} | parte de la \"sensibilidad\" es a quién enfrentó |",
        f"| estadio | r(drag promedio de sus estadios de Extreme, S_emp) = {fac['drag_extreme']['r_con_s_emp']:+.2f} | algo de contexto: no todos los Extreme son iguales |",
        f"| cambio de pitch mix | r(cambio de mezcla entre Extreme y No, S_emp) = {fac['cambio_mezcla']['r_con_s_emp']:+.2f} | poco |",
        f"| abridor / relevista | abridores {pr.loc['abridor', 'mean']:+.2f} (n={int(pr.loc['abridor', 'size'])}), relevistas {pr.loc['relevista', 'mean']:+.2f} (n={int(pr.loc['relevista', 'size'])}) | sin diferencia relevante |",
        f"| equipo local en Extreme | {pl.loc[True, 'mean']:+.2f} (n={int(pl.loc[True, 'size'])}) vs. {pl.loc[False, 'mean']:+.2f} (n={int(pl.loc[False, 'size'])}) | pequeña, dentro del ruido |",
        f"| selección | quienes tienen S_emp: Pitching+ neutral medio {sel.loc[True, 'pitching_plus_neutral']:.1f} y {sel.loc[True, 'n_pitches']:,.0f} pitches, contra {sel.loc[False, 'pitching_plus_neutral']:.1f} y {sel.loc[False, 'n_pitches']:,.0f} | solo se mide a pitchers con mucho uso |",
        f"| regresión a la media | factor de shrinkage medio {s_tab['factor_shrinkage'].mean():.2f} | casi todo se encoge a 0 |",
        "",
        "No se intentó \"arreglar\" la confiabilidad. **Sigue siendo baja y es una limitación del proyecto.**",
        "",
        "# 14. Ranking exploratorio de sensibilidad",
        "",
        f"> ⚠️ **ADVERTENCIA METODOLÓGICA.** Este ranking se incluye porque fue una de las hipótesis exploradas durante el proyecto. Sin embargo, su "
        f"confiabilidad estimada es únicamente {rel:.2f}, por lo que **NO debe interpretarse como evidencia de una capacidad estable del pitcher "
        "para resistir la altitud**. No se usa en el perfil ni en el ranking principal.",
        "",
        "<details><summary>Ver el ranking exploratorio (los 10 con menor sensibilidad encogida)</summary>",
        "",
        "| lugar | pitcher | S_emp cruda [80%] | S_emp con shrinkage [80%] | shrinkage | pitches Extreme / No | S_mod relativa | confiabilidad |",
        "|---:|---|---|---|---:|---|---:|---:|",
    ])
    for _, r in s_tab.sort_values("rank_menos_sensible").head(10).iterrows():
        L.append(f"| {int(r['rank_menos_sensible'])} | {r['pitcher_anon_id']} | {r['s_emp']:+.2f} [{r['s_emp_lo80']:+.2f}, {r['s_emp_hi80']:+.2f}] | "
                 f"{r['s_emp_shrunk']:+.2f} [{r['s_shrunk_lo80']:+.2f}, {r['s_shrunk_hi80']:+.2f}] | {r['factor_shrinkage']:.2f} | "
                 f"{int(r['n_extreme'])} / {int(r['n_no'])} | {r['s_mod_rel']:+.2f} | {rel:.2f} |")
    add([
        "",
        "</details>",
        "",
        f"Con shrinkage, {int(((s_tab['s_shrunk_lo80'] < 0) & (s_tab['s_shrunk_hi80'] > 0)).sum())} de {len(s_tab)} intervalos de 80% cruzan 0. "
        "Los intervalos crudos que no cruzan 0 son en su mayoría los extremos esperados por azar entre 294 pitchers. La tabla completa, "
        "con la advertencia en cada fila, está en `data/web/pitcher_sensitivity.csv`.",
        "",
        "# 15. Model vs observed sensitivity",
        "",
        f"**Hallazgo negativo importante:** corr(sensibilidad modelada, sensibilidad observada) = **{r_me:+.2f}** (Spearman "
        f"{sens['r_mod_emp_spearman']:+.2f}; {sens['info']['n_pitchers']} pitchers).",
        "",
        "Interpretación: **nuestro contrafactual físico no identifica actualmente qué pitchers muestran mejor adaptación observada a los ambientes "
        "de altitud.**",
        "",
        f"- Con una confiabilidad observada de {rel:.2f}, ni un modelo perfecto podría correlacionar más de ≈ {np.sqrt(rel):.2f} con la medida observada.",
        "- El problema es doble: el contrafactual describe tipos de pitch, y la medida observada es casi toda ruido.",
        "",
        f"![Modelo vs observado]({_rel(figs['sens'])})",
        "",
        "---",
        "",
    ])

    # 16-18 ------------------------------------------------------------------
    add([
        "# 16. Incertidumbre",
        "",
        f"- **Ranking principal:** el intervalo de 80% del lugar tiene una amplitud mediana de {rank_width:.0f} posiciones (de {len(e)}). La cima "
        f"está bien separada (el #1 queda entre {int(e.iloc[0]['rank_lo80'])} y {int(e.iloc[0]['rank_hi80'])}), pero el centro de la tabla es casi "
        "intercambiable. Por eso se publican P(top 25) y el intervalo, no solo el lugar.",
        "- **Componentes:** con ~300 turnos, el K% de un pitcher tiene un error estándar de ≈ 2 pp. El daño por bola en juego casi no tiene señal "
        "(confiabilidad típica < 0.1 por temporada) y se encoge casi por completo.",
        "- **Pesos lineales:** tienen IC de 95% por bootstrap de juegos (sección 7.3). No se propagan al intervalo del AEPP; si se propagaran, el "
        "intervalo sería algo más ancho.",
        "- **Pitchers con pocos pitches:** quedan fuera del ranking (mínimo 300 pitches y 75 turnos). En la ficha se reportan la muestra y el factor de shrinkage.",
        f"- **Sensibilidad:** los intervalos son mucho más anchos que las diferencias entre pitchers (confiabilidad {rel:.2f}).",
        "",
        "# 17. Limitaciones",
        "",
        "- Altitud y estadio no se pueden separar: hay pocos estadios por categoría (sección 2.4), y Medium se parece mucho a Extreme.",
        "- \"Extreme\" mezcla estadios con densidades distintas (P0). Algunos IDs de Extreme probablemente son el mismo estadio sin enlazar entre temporadas.",
        "- El ajuste simple por entorno quita el promedio de cada estadio-temporada, incluida la parte que se deba a qué pitchers lanzan ahí.",
        "- Las métricas por pitcher no se ajustan por la calidad de los bateadores (los efectos del ambiente sí).",
        "- Solo hay dos pares de temporadas para validar (2024→2025 y 2025→2026), y 2026 es una temporada parcial.",
        "- En estos datos, el daño de contacto de cada pitcher es casi ruido: un perfil de \"contact manager\" no se puede medir con confianza.",
        "- Las interacciones de la sección 9.1 son asociaciones con muchas comparaciones.",
        "- El contrafactual físico usa modelos entrenados con todos los datos y sobreestima el cambio de IVB en ≈ 20% (P0).",
        "- La velocidad de salida y el contacto duro dependen de la calibración del Trackman de cada estadio (el hard contact% promedio sube de 2024 a 2025).",
        "",
        "# 18. Conclusiones",
        "",
        "## Hallazgos robustos",
        "",
        f"1. En los contextos de altitud el juego cambia: menos whiffs ({100 * _fx(res, WH):+.1f} pp por swing), menos ponches "
        f"({100 * _fx(res, KP):+.1f} pp por turno) y contacto más caro ({_fx(res, RV):+.3f} carreras por bola en juego), en "
        f"{ce[WH]['en_direccion']}/{ce[WH]['total']} estadio-temporadas de Extreme.",
        f"2. Evitar el contacto vale ≈ {_pct(g_ratio, 0)} más por swing, y cambiar una bola en juego por un ponche ≈ {_pct(k_gain, 0)} más por turno.",
        "3. El aumento del daño de contacto viene casi todo del daño por bola en juego, no de que haya más bolas en juego.",
        "4. Stuff+ Physical mide la física sin ubicación ni ambiente, es muy estable y predice el whiff futuro. Location+ mide la ubicación por separado y predice las bases por bolas.",
        "5. El perfil favorable en altitud es el de **swing-and-miss**: whiffs, ponches y pocas bolas en juego.",
        "",
        "## Hallazgos exploratorios",
        "",
        "1. Whiff/swing, K% y Stuff+ Physical se asocian con más carreras evitadas en Extreme que en No Altitude.",
        "2. El slider es el tipo que menos valor pierde en Extreme; la recta, el sinker y el changeup son los que más movimiento pierden.",
        "3. Los pitches en el corazón de la zona son los que más valor pierden en altitud.",
        "4. La sensibilidad observada se relaciona en parte con la calidad de los rivales enfrentados en cada ambiente.",
        "",
        "## Resultados negativos",
        "",
        "1. El contrafactual físico no supera al baseline trivial de ajuste por entorno (P0).",
        f"2. La sensibilidad individual observada tiene confiabilidad ≈ {rel:.2f}.",
        f"3. La sensibilidad modelada no predice la observada (r = {r_me:.2f}).",
        "4. Medium se parece mucho a Extreme: no hay identificación causal pura de la altitud.",
        "5. El perfil favorable para altitud casi no difiere del perfil bueno en general, y los pesos de altitud no mejoran la predicción de forma detectable.",
        "6. No hay arquetipos claros de pitcher; hay un continuo entre swing-and-miss y control.",
        "7. Usar Stuff+ y Location+ como covariables no mejora el shrinkage de K%, BB% ni del daño de contacto.",
        "",
        "## Hipótesis futuras",
        "",
        f"1. Con más temporadas, volver a medir la sensibilidad individual (harían falta ≈ {need.get(0.5, float('nan')):,.0f} pitches en Extreme por pitcher para llegar a confiabilidad 0.5).",
        "2. Ajustar las métricas por pitcher por la calidad de los bateadores enfrentados.",
        "3. Modelar el daño de contacto con velocidad y ángulo de salida esperados, para separar mejor habilidad de suerte.",
        "4. Separar los estadios de Medium y Extreme por su densidad medida, en lugar de por categoría.",
        "",
        "---",
        "",
        "# ¿Qué significa todo esto en béisbol?",
        "",
        "**Por qué una bola en juego puede ser más peligrosa.** En los estadios de altitud el aire es menos denso: la pelota bateada vuela más y "
        "los pitches se mueven menos. Cuando el bateador hace contacto, es más probable que sea hit o extra base. En nuestros datos, una bola en "
        f"juego en Extreme le cuesta al pitcher unas {_fx(res, RV):.2f} carreras más que en un estadio sin altitud.",
        "",
        "**Por qué generar un whiff puede valer más.** Un whiff es un swing que no puede convertirse en hit. Si cada contacto es más caro, cada "
        f"contacto evitado vale más: en Extreme, cambiar una bola en juego por un swing fallado ahorra alrededor de {_pct(g_ratio, 0)} más carreras "
        "que en un estadio a nivel del mar. Al mismo tiempo, allí es más difícil conseguir whiffs, porque los pitches se mueven menos.",
        "",
        "**Qué es Stuff+.** Una calificación de qué tan difícil de conectar es un pitch por sus características físicas: velocidad, giro, "
        "movimiento, desde dónde sale y con qué ángulo llega. No importa dónde terminó el pitch ni en qué estadio se lanzó. 100 es el promedio "
        "de la liga, y cada 10 puntos equivalen a una desviación estándar de talento.",
        "",
        "**Qué es Location+.** Una calificación de qué tan buenos lugares encuentra el pitcher para cada tipo de pitch en cada conteo. Mide dónde "
        "terminó la pelota, no qué tan bueno era el pitch. Los pitchers con Location+ alto dan menos bases por bolas.",
        "",
        "**Qué es Pitching+.** Una calificación del valor total de los pitches (física, ubicación y conteo juntos), expresada en carreras. En esta "
        "etapa se usa la versión con un ajuste simple por estadio, que predijo tan bien como la versión que modela la física del aire.",
        "",
        "**Qué significa tener un perfil favorable.** Que el pitcher hace bien justo lo que más vale en altitud: hace fallar a los bateadores, "
        "poncha y permite pocas bolas en juego. Nuestro puntaje AEPP traduce eso a carreras evitadas por cada 100 bateadores enfrentados, usando "
        "lo que vale cada resultado en los estadios de altitud.",
        "",
        "**Por qué \"perfil favorable\" NO es lo mismo que \"demostradamente resistente\".** \"Perfil favorable\" dice que el pitcher tiene las "
        "características que, en general, funcionan mejor en altitud. \"Resistente\" diría que ese pitcher en particular pierde menos que los "
        "demás cuando lanza en altitud. Intentamos medir eso directamente y los datos no alcanzan: la diferencia de cada pitcher entre estadios "
        f"es casi toda suerte y contexto (confiabilidad {rel:.2f}). Por eso recomendamos perfiles, no garantías.",
        "",
        "---",
        "",
        "Tablas para la futura página: " + ", ".join(f"`data/web/{p.name}`" for p in web.values()) + ".",
        "",
    ])
    return "\n".join(L)
