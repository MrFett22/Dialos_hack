# Validación final de la página web

Página: `web/app.py` (Streamlit). Motor de inferencia: `web/engine.py`. Figuras: `web/charts.py`. Datos: `data/web/`
(pasos `p2_report` y `web_data` del pipeline). Plan previo: `reports/web_architecture_plan.md`.

```bash
.venv\Scripts\streamlit run web/app.py
```

## 1. Tests

`.venv\Scripts\python -m pytest tests`: **53 PASS** (40 anteriores + 13 en `tests/test_web.py`; 3 de ellos agregados con la ampliación, §6). También pasan como
último paso del pipeline (`reports/pipeline_run.md`).

| test | qué verifica | resultado |
|---|---|---|
| 1 `test_average_pitch_gives_average_score` | 150 lanzamientos reales al azar dan Pitching+, Stuff+ y Location+ con media dentro de 100 ± 5; el lanzamiento mediano de cada tipo da valores razonables | PASS |
| 2 `test_out_of_range_inputs_warn` | 110 mph en una recta produce advertencia por variable y por combinación; la distancia de Mahalanobis crece; una ubicación sin soporte avisa | PASS |
| 3 `test_location_changes_location_plus_not_stuff` | mover la ubicación cambia Location+ y casi no cambia Stuff+ Physical | PASS |
| 4 `test_physical_changes_move_stuff` | cambiar velocidad y movimiento cambia Stuff+ y deja Location+ idéntico | PASS |
| 5 `test_probabilities_sum_to_one` | los 10 eventos suman 1 (error < 1e-9) en los tres ambientes; P(BIP) = suma de los resultados en juego | PASS |
| 6 `test_pitching_plus_decreases_with_expected_runs` | Pitching+ es estrictamente decreciente en E[Δ]; en lanzamientos del mismo ambiente, ordenar por E[Δ] ordena al revés por Pitching+ | PASS |
| 7 `test_missing_values_do_not_break` | inputs vacíos, NaN, tipo de pitch inexistente o conteo inválido: el motor imputa la mediana del tipo, avisa y no falla | PASS |
| extra `test_improvements_stay_in_range` | "¿Qué podría mejorar?" solo propone cambios dentro de los percentiles 5–95 del tipo y ordena por Δ Pitching+ | PASS |
| extra `test_summary_matches_sources` | las cifras de `summary.json` coinciden con sus fuentes (efectos, G, golpeado posterior al fix, sensibilidad, número de elegibles) | PASS |
| extra `test_page_renders_without_exceptions` | la página completa corre en `streamlit.testing` sin excepciones, también al cambiar de tipo de pitch y de ambiente, de tipo y métrica en la sección 6 y con un tipo de muestra pequeña; aparecen las advertencias de no causalidad | PASS |
| ampliación `test_contact_damage_is_bip_branch_of_tree` | daño de contacto del simulador = Σ_k∈BIP P(k)·Δ(k, conteo) = P(BIP) × E[RV \| BIP]; Σ de todas las ramas = E[Δ] | PASS |
| ampliación `test_region_and_pitch_type_context` | región de la ubicación, cambio observado de la región (el corazón es la que más pierde) y contexto del tipo (menor costo = orden 1; muestra pequeña sin orden) | PASS |
| ampliación `test_extended_summary_matches_sources` | ahorro del rodado, valor por BIP, versiones de Stuff, techo √confiabilidad, split-half de IVB, tipo de menor costo, amplitud del rango y P(top 25) coinciden con sus fuentes | PASS |

## 2. Validación de cálculos

**Escala del simulador.** Sobre 300 lanzamientos reales tomados al azar y pasados por el motor (con sus valores físicos, ubicación,
conteo, manos y ambiente):

| puntaje | media | sd |
|---|---:|---:|
| Pitching+ | 99.5 | 10.1 |
| Stuff+ Physical | 100.4 | 9.6 |
| Location+ | 99.3 | 10.2 |

Coincide con la definición (100 = lanzamiento promedio, 10 = 1 sd). Es una prueba de consistencia interna: el motor reconstruye
la trayectoria y usa diferencias típicas contra la recta, y aun así reproduce la distribución de los lanzamientos reales.

**Escala de lanzamiento vs. escala de pitcher.** Para un lanzamiento individual no se usa la escala de los pitchers (μ, τ del
shrinkage): daría valores absurdos (Location+ de −86 a 242 entre los percentiles 5 y 95), porque un lanzamiento varía mucho más
que el promedio de un pitcher. La página lo explica junto a cada número. El ranking y la ficha usan la escala de pitcher.

**Lanzamiento mediano de cada tipo** (centro de la zona, 0-0, No Altitude, pitcher y bateador derechos):

| tipo | Stuff+ | Location+ | Pitching+ | P(whiff \| swing) | P(BIP) | E[Δ] por 100 |
|---|---:|---:|---:|---:|---:|---:|
| Four-Seam | 89.6 | 103.1 | 103.0 | 8.8% | 29.6% | −2.08 |
| Sinker | 86.4 | 104.4 | 103.6 | 6.8% | 32.1% | −2.41 |
| Cutter | 94.0 | 101.3 | 100.5 | 11.2% | 30.8% | −0.77 |
| Slider | 105.0 | 101.8 | 100.2 | 15.8% | 27.9% | −0.60 |
| Curveball | 101.1 | 102.0 | 102.9 | 10.6% | 17.4% | −2.02 |
| Changeup | 97.3 | 103.9 | 101.2 | 17.2% | 29.9% | −1.15 |
| Splitter | 99.5 | 102.4 | 101.8 | 16.8% | 27.5% | −1.47 |

Stuff+ compara contra todos los tipos: las rectas generan menos whiff por swing que los rompientes, así que una recta mediana
queda debajo de 100. Por eso la página muestra también el percentil dentro del tipo, y la interpretación automática usa ese
percentil (fortaleza ≥ 75, debilidad ≤ 25).

**Comparación de ambientes (contrafactual).** Una recta mediana soltada igual en los tres ambientes cruza el plato 2.7 pulgadas
más abajo y 1.7 pulgadas menos hacia el lado del brazo en Extreme. Su E[Δ] pasa de −2.08 a +0.20 carreras por 100 pitches
(No Altitude → Extreme) cuando el pitcher apunta al mismo lugar. Es el mismo sentido que los efectos observados de P1/P2.

**Reproducibilidad.** `python -m src.pipeline --from web_data` regenera `data/web/` byte a byte: solo cambió `README.md`, por el
cambio intencional de su texto. Todos los textos numéricos de la página leen `summary.json`; no hay cifras escritas a mano.

**Rendimiento.** Carga de modelos ≈ 0.15 s (una vez, con caché); una simulación ≈ 76 ms; "¿qué podría mejorar?" (≈ 14
simulaciones) ≈ 1 s.

## 3. Componentes (verificados en el navegador, 1400 × 900)

| sección | componentes | verificación |
|---|---|---|
| Hero | pregunta central, cinco cifras clave e idea principal en grande | se ve; cifras de `summary.json` |
| 1 Evidencia | 4 cifras grandes; barras con IC de Medium y Extreme (whiff, K, BIP/swing, RV/BIP); dispersión por estadio-temporada; aviso causal; tres niveles de explicación | ok; separé Medium y Extreme en filas distintas |
| 2 El contacto cuesta más | 3 paneles P(BIP), E[RV\|BIP] y daño; dona de descomposición (93% / 1%); dispersión por estadio anónimo | ok |
| 3 Qué buscamos | ponche vs. BIP por ambiente (+27%); valor de cada característica por ambiente | ok |
| 4 Stuff+ Physical | 1.9% vs. 46% de ubicación, estabilidad 0.76, comparación con v1 y v4 | ok |
| 5 Location+ | mapa de calor interactivo (tipo, enfrentamiento, conteo), independencia de Stuff+, BB% futuro | ok |
| 6 Pitching+ | diagrama de la arquitectura, Sankey del árbol de probabilidades con un ejemplo real del modelo, fórmulas y comparación | ok |
| 7 Construye tu lanzamiento | tipo, manos, ambiente, conteo y 7 controles físicos con límites del dominio; zona de strike clicable coloreada por Location+; tarjetas de Pitching+ (grande), Stuff+ y Location+ con percentiles; mecanismo; probabilidades contra el lanzamiento mediano; interpretación; alertas; "¿qué podría mejorar?"; tabla de dominio | ok; ver hallazgos |
| 8 Compara ambientes | barras y tabla de E[Δ], P(whiff \| swing), P(BIP), IVB, Stuff+, Location+ y Pitching+, con la marca "Simulación contrafactual del modelo" | ok |
| 9 Ranking | 4 cifras de validación (incluye "AEPP NO supera a K−BB%" y 0.986), filtros, tabla con intervalos, forest del top 25, ficha del pitcher con pitch mix y mejor arma por tipo, nota de arquetipos | ok; corregí el formato de los lugares |
| 10 Sensibilidad | caja de advertencia (0.11, ≈ 2,500 pitches, 294/294, r = 0.06), dispersión modelo vs. observado, tabla dentro de un expander | ok |
| 11 Física | trayectoria con la misma salida (diferencia en pulgadas y punto de cruce), drag y Magnus por estadio anónimo, IVB por tipo | ok; ver hallazgos |
| 12 Qué funcionó y qué no | dos columnas, limitaciones y metodología en expanders, mensaje final | ok |

**Problemas encontrados y corregidos durante la verificación:**

1. **Selección en la zona de strike.** Con datos binarios de Plotly, el punto seleccionado podía llegar sin `x`/`y` y la página
   fallaba (`KeyError`). Ahora la ubicación se toma del índice del punto en una rejilla fija (`Engine.location_grid`); `x`/`y` quedan
   como respaldo.
2. **Caché del motor.** Después de cambiar `engine.py`, Streamlit conservaba el objeto anterior en caché; un reinicio lo resuelve.
   Solo afecta al desarrollo.
3. **Trayectorias indistinguibles.** Con el supuesto de comando las tres trayectorias llegan al mismo punto y se superponen. Se cambió
   a "misma salida" y se grafica la diferencia contra No Altitude en pulgadas.
4. **Interpretación injusta con las rectas.** Comparar contra todos los tipos marcaba como debilidad el Stuff+ de una recta normal.
   Ahora usa el percentil dentro del tipo.
5. **Detalles visuales:** etiquetas de barras encimadas, lugares del ranking con decimales, leyendas sobre títulos, dona del pitch mix.

## 4. Advertencias del simulador (reglas de extrapolación)

- Cada input fuera de los percentiles 1–99 de su tipo genera una advertencia con el rango observado.
- Si la combinación supera el percentil 99 de la distancia de Mahalanobis de los lanzamientos reales del tipo: "fuera del rango común
  observado… la predicción tiene menor confiabilidad".
- Con eje de giro manual, se avisa si se aleja del que implica el movimiento más que el percentil 99 real.
- Una ubicación con menos de 20 pitches reales del tipo en su celda de 0.25 pies genera aviso; en la zona aparece en blanco.
- Los sliders están limitados a los percentiles 0.5–99.5 del tipo.
- Inputs faltantes → mediana del tipo, con aviso.
- En la muestra de 300 lanzamientos reales, 17.7% activa al menos un aviso, sobre todo por ubicaciones poco frecuentes o por una
  variable en el 1% de una cola. Es lo esperable con siete variables evaluadas a la vez.

## 5. Limitaciones

- **Clic en la zona.** La selección se verificó emitiendo el evento de selección de Plotly. Los clics sintéticos del navegador de
  pruebas no disparan la selección de Plotly, tampoco en la app mínima de ejemplo de Streamlit. Con un mouse real, el clic y el
  arrastre de caja seleccionan la celda. Si fallara, los controles de PlateLocSide y PlateLocHeight hacen lo mismo.
- Las diferencias entre ambientes del simulador son **solo físicas**: el modelo no ve el parque ni el vuelo de la pelota bateada, así que
  subestima el daño de contacto en altitud (P2 §7.2). La página lo dice en la sección 8.
- El ajuste por ambiente del simulador es por **categoría**. El índice publicado usa temporada × estadio anónimo, pero el usuario
  no elige estadio.
- El simulador usa los modelos **finales** (entrenados con todos los datos), no las predicciones fuera de muestra con que se validaron
  los índices.
- Las diferencias de la tabla de mejoras menores a ~1 punto están dentro del ruido escalonado de los árboles; la página lo advierte.
- Las diferencias contra la recta (`velo_diff_vs_fb`, etc.) usan el valor típico del tipo: el simulador no conoce el resto del
  repertorio del pitcher.
- Streamlit es una app local de un solo proceso; para publicarla haría falta un servidor con Python (no es una página estática).

## 6. Ampliación (requisitos complementarios)

La estructura no cambió: Pitching+ sigue siendo el protagonista. Secciones de la página ahora: 1 Evidencia · 2 El contacto cuesta más ·
3 Qué buscamos · 4 Stuff+ Physical · 5 Location+ · **6 Tipos de lanzamiento (nueva)** · 7 Pitching+ · 8 Construye tu lanzamiento ·
9 Compara ambientes · 10 Ranking · 11 Sensibilidad · 12 Física · 13 Qué funcionó y qué no · **14 Qué mejoraríamos (nueva)**.

| requisito | dónde | fuente de las cifras |
|---|---|---|
| Tipos de pitch: Δ whiff, Δ BIP/swing, Δ RV/BIP, IVB, HB, costo Extreme y Medium; selector; dispersión whiff vs. costo | sección 6 | `pitch_type_altitude_effects.csv`; reglas en `summary.json → tipos_pitch` (muestra razonable ≥ 20,000 pitches) |
| Advertencia "descriptivo/ajustado por contexto, no causal"; Splitter con muestra pequeña | sección 6 y simulador | ídem |
| Contexto del tipo en el simulador ("promedio del tipo, no destino garantizado") | sección 8, vista rápida y "Más detalles" | `Engine.pitch_type_context` |
| Corazón de la zona pierde más valor; nota de la región en el simulador | sección 5 y sección 8 | `summary.json → location.regiones`; `Engine.region_context` (misma `location_plus_v2.region`) |
| Rodado como segunda defensa (0.140 → 0.202) y pirámide conceptual | sección 2 | `summary.json → ahorro_rodado` (v_aire − v_GB de los pesos lineales) |
| P(BIP), E[RV \| BIP], daño de contacto, P(whiff) por pitch, E[Δ] | sección 8, pestaña "¿Por qué?" | modelo final; identidad probada en `test_contact_damage_is_bip_branch_of_tree` |
| Dos formas de ser bueno; estabilidad de cada métrica | sección 3 | `summary.json → estabilidad_metricas` |
| Advertencia en la ficha cuando la explicación usa métricas de contacto | sección 10 | ídem |
| Stuff observado / oficial / neutral solo en "Comparar neutral vs. observado" | sección 4 (expander) | `summary.json → stuff_versiones` |
| Ranking: intervalo AEPP, lugar estimado e intervalo, P(top 25), pitches, K%, BB%, whiff, Stuff+, Location+; tarjetas de los 4 primeros | sección 10 | `pitcher_altitude_profiles.csv`; `summary.json → ranking_top` |
| Sensibilidad: Pearson y Spearman; techo √0.11 ≈ 0.33 con sus supuestos | sección 11 | `summary.json → sensibilidad_extra` |
| Contrafactual "simulación física aproximada": split-half (IVB sobrepredicho 19.8%) | secciones 9 y 12 | `summary.json → contrafactual_split_half` (de `altitude_counterfactual_v2.json`) |
| Calibración de EV por temporada; validación temporal limitada (2026 parcial); calidad de rivales | sección 13 | `summary.json → temporadas` |
| Medium ≈ Extreme (RV/BIP +0.073 / +0.085 / +0.015) | sección 1 | `summary.json → valor_bip` |
| Pitching+ sin claim causal; conclusión general | secciones 7 y cierre | texto |
| Qué mejoraríamos y Expected Contact Quality (no implementado) | sección 14 | texto; cifras de `summary.json` |

**Verificación en el navegador (1400 × 900):** sin excepciones, sin scroll horizontal y 31 gráficas renderizadas. La cadena
P(BIP) × E[RV | BIP] → Pitching+ cabe en una fila. Se corrigieron tres cosas: las etiquetas encimadas en la dispersión de tipos, la
leyenda sobre el título en la gráfica de regiones, y una frase de la sección 13 que afirmaba algo no medido sobre la calibración de la EV.
Ahora la frase solo reporta el cambio de hard contact% (21.5% en 2024 → 26.2% en 2025) y dice que no se puede separar calibración de
cambio real.

**Reproducibilidad:** `python -m src.reports.web_data` solo cambia `summary.json` (campos nuevos); el resto de `data/web/` es idéntico.
