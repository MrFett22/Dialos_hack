# Plan de arquitectura de la página web — Pitching+ para ambientes de altitud

Auditoría de los assets disponibles al cierre de P2 (commit `57cbd73`) y diseño de la página. Este plan se escribió **antes** de construirla.

## 0. Decisión de plataforma

| opción | ventaja | problema | decisión |
|---|---|---|---|
| Página estática (HTML/JS) | se publica como un archivo | el simulador necesita 11 modelos LightGBM (~11,500 árboles), la re-simulación de trayectorias y la ingeniería de features v2 (espejo de zurdos, VAA/HAA ajustados por ubicación). Portarlo todo a JS duplicaría la lógica y podría desalinearse con el pipeline | no |
| API + front-end propio | máximo control visual | dos servicios que mantener | no para el hackathon |
| **Streamlit (Python) con CSS propio y Plotly** | usa **los mismos módulos y artefactos del pipeline** (cero lógica duplicada); ya existe un dashboard Streamlit y `.claude/launch.json` | estética más limitada; se compensa con CSS y componentes propios | **sí** |

Se crea una app nueva en `web/`. El dashboard anterior (`dashboard/`) queda como legado, porque usa el índice neutral físico como oficial y eso ya no es la decisión vigente.

## 1. Estructura de la página (una sola página con navegación lateral)

| # | sección | tipo de evidencia | contenido principal |
|---|---|---|---|
| 1 | Hero | — | pregunta central y mensaje principal en grande |
| 2 | Evidencia: el juego cambia | **observado / asociación** | whiff/swing, K/PA, RV/BIP, G; consistencia 16/16 estadio-temporadas; comparación con Medium; límite causal |
| 3 | Insight: el contacto cuesta más | **observado** | P(BIP) × E[RV\|BIP] por ambiente; descomposición (93% daño / 1% cantidad); dispersión por estadio anónimo |
| 4 | Qué buscamos en un pitcher | **observado + asociación** | pesos lineales por ambiente (K vs BIP +27%); valor de cada característica por ambiente |
| 5 | Stuff+ Physical | **predictivo** | qué mide, variables físicas, 1.9% vs 46% de ubicación, estabilidad 0.76, comparación v1/v4 |
| 6 | Location+ | **predictivo** | qué mide, mapas de calor (tipo × mano × conteo), independencia de Stuff+, predicción de BB% |
| 7 | **Pitching+** (sección principal) | **predictivo** | árbol de probabilidades, E[Δ] = Σ P·Δ, escala, comparación con Stuff+/Location+ |
| 8 | **Construye tu lanzamiento** (interactivo principal) | **simulación del modelo** | controles físicos, zona de strike clicable, conteo, manos y ambiente; Stuff+, Location+ y Pitching+; probabilidades; interpretación; "¿qué podría mejorar?"; alertas de extrapolación |
| 9 | Compara ambientes | **simulación contrafactual** | el mismo pitch en No Altitude / Medium / Extreme |
| 10 | Ranking de pitchers | **predictivo (perfil)** | AEPP con intervalos; validación (r 0.26 vs 0.15; no supera a K−BB%); correlación 0.986; ficha del pitcher con pitch mix |
| 11 | Sensibilidad exploratoria | **exploratorio (no confiable)** | advertencia, confiabilidad 0.11, r 0.06, 294/294 intervalos cruzan 0 |
| 12 | Física / contrafactual | **simulación** | drag, Magnus, trayectoria original vs. simulada, estadios anónimos heterogéneos |
| 13 | Metodología, qué funcionó y qué no, limitaciones | — | transparencia, reproducibilidad |

Cada sección importante tiene tres niveles en pestañas (**Simple · Béisbol · Técnico**), con el técnico al final. Cada bloque lleva una **etiqueta del tipo de evidencia**: Observado, Asociación, Predictivo, Simulación, Exploratorio o Hipótesis.

## 2. Tablas disponibles (`data/web/`, generadas por `src/reports/p2_report.py`)

| tabla | filas × cols | columnas clave | sección | ¿recalcular? |
|---|---|---|---|---|
| `contact_damage_by_environment.csv` | 28 × 24 | nivel, ambiente, p_bip_por_pitch, rv_por_bip, dano_bip_por_pitch (+ IC) | 3 | no |
| `linear_weights_by_environment.csv` | 4 × 18 | v_K, v_BB, v_HBP, v_BIP, K_menos_BIP (+ IC), v_GB, v_aire | 4 | no |
| `characteristic_value_by_environment.csv` | 9 × 19 | pend_{ambiente} (+ IC), dif_Extreme, p_mayor | 4 | no |
| `stuff_plus_physical.csv` | 1,628 × 13 | por pitcher-temporada | 5 (distribución) | no |
| `location_plus.csv` | 1,628 × 9 | por pitcher-temporada | 6 (distribución) | no |
| `location_plus_heatmap.csv` | 28,592 × 7 | tipo, enfrentamiento, grupo_conteo, loc_in, PlateLocHeight, carreras_por_100 | 6 | no |
| `pitch_type_altitude_effects.csv` | 7 × 40 | Δ whiff, Δ RV/BIP, Δ IVB/HB, costo por tipo | 12 | no |
| `pitcher_altitude_profiles.csv` | 599 × 121 | ficha completa del pitcher | 10 | no |
| `pitcher_rankings.csv` | 599 × 15 | rank, aepp (+ IC), rank_lo/hi, p_top25, explicación | 10 | **no se usa**: es un subconjunto de `pitcher_altitude_profiles` (se conserva para descargas) |
| `pitcher_sensitivity.csv` | 294 × 16 | s_emp, s_emp_shrunk (+ IC), confiabilidad, advertencia | 11 | no |

**Faltantes, que se agregan con un paso nuevo del pipeline (`src/reports/web_data.py`) y nunca a mano:**

| nuevo archivo | contenido | sale de |
|---|---|---|
| `summary.json` | todas las cifras de los textos: efectos E/M con IC, 16/16, G, 27%, daño 0.26→1.62, 93%/1%, validación del AEPP, 0.986, confiabilidad 0.11, r 0.06, 2,500 pitches, 294/294, Stuff+ 1.9%/46%/0.76, Location+ vs BB%, HBP +0.398, estabilidad de los estadios | `hipotesis_altitud.json`, `p2_profiles_results.pkl`, `stuff_plus_physical.json`, `location_plus_v2.json`, `estadios_anonimos.json` (nuevo), `run_value_chain` |
| `pitcher_pitch_types.csv` | por pitcher elegible × tipo: uso, Stuff+ por tipo, Location+ por tipo, whiff/swing | pitches OOF de Stuff+ Physical y Location+ |
| `anon_stadiums.csv` | ID anónimo, categoría, ratios de drag y Magnus con IC | `anon_stadium_ratios.parquet` |
| `simulator_domain.json` | por tipo de pitch: percentiles 1/5/50/95/99 de cada input, media y covarianza, cuantiles de la distancia de Mahalanobis | datos del pipeline |
| `simulator_scales.json` | μ y τ de Stuff+, Location+ y Pitching+; desplazamientos por ambiente; mezcla de lados | resultados del pipeline |

## 3. Modelos disponibles

| modelo | archivo | uso en la web | estado |
|---|---|---|---|
| Pitching+ v3 final (5 sub-modelos + Δ por conteo + coeficientes de ubicación) | `src/models/artifacts/pitching_plus_v3.pkl` | probabilidades de eventos y E[Δ] del lanzamiento | existe (regenerado después del fix de HBP) |
| Stuff+ Physical final | `src/models/artifacts/stuff_plus_physical.pkl` | P(whiff \| swing) físico | **falta**: se agrega `train_final()` |
| Location+ v2 final (5 sub-modelos + referencia por contexto) | `src/models/artifacts/location_plus_v2.pkl` | valor de la ubicación | **falta**: se guardan los modelos que ya entrena `heatmap_grid` |
| Trayectoria (`trajectory.synthesize_pitch`) y ratios de densidad | `altitude_density_ratios.parquet` | comparar ambientes y graficar trayectorias | existe |
| Defaults por tipo × mano × categoría | `dashboard_pitch_defaults.parquet` (`dashboard/scoring.py`) | lado de salida, drag típico, diferencias contra la recta | existe |

## 4. Inferencia necesaria (`web/engine.py`, sin Streamlit)

- `simulate(inputs) -> Result`: (1) sintetiza la trayectoria en las 3 categorías a partir del pitch descrito en su ambiente de referencia; (2) construye las features v2; (3) aplica los tres modelos; (4) escala.
- `improvements(inputs)`: evalúa cambios pequeños de una variable a la vez, dentro del rango observado del tipo.
- `domain_check(inputs)`: percentiles, rango observado, distancia de Mahalanobis y su percentil.
- `trajectory_paths(inputs)`: puntos (x, y, z) del vuelo en cada ambiente, para la gráfica.
- Caché: los modelos se cargan una sola vez (`st.cache_resource`); cada simulación cuesta unas pocas filas × 11 modelos, en milisegundos.

## 5. Componentes interactivos

1. Selector de tipo de pitch y mano del pitcher, y sliders físicos cuyos límites y defaults salen de `simulator_domain.json`.
2. Zona de strike clicable (Plotly `on_select`) con sliders de respaldo para PlateLocSide y PlateLocHeight.
3. Conteo, mano del bateador y ambiente.
4. Resultado: tres tarjetas de score (Pitching+ la más grande), barras de probabilidades por evento, P(whiff | swing), P(BIP), E[Δ | BIP], E[Δ] e interpretación automática.
5. "¿Qué podría mejorar?": tabla ordenada por Δ Pitching+.
6. Comparar ambientes: tabla y barras, con la marca **Simulación contrafactual del modelo**.
7. Ranking: tabla filtrable (muestra mínima, ambiente con más pitches) con intervalos y selector de pitcher para la ficha.
8. Ficha del pitcher: métricas con intervalo, pitch mix (dona y barras por tipo) y "¿por qué aparece alto o bajo?".

## 6. Visualizaciones

| visualización | sección |
|---|---|
| Barras con IC de los efectos E y M (whiff, K, RV/BIP, G) | 2 |
| Matriz estadio-temporada (16/16) | 2 |
| Las dos piezas del daño de contacto (3 paneles) y dispersión por estadio | 3 |
| Pesos lineales y "valor de cada característica por ambiente" (forest plot) | 4 |
| Barras de comparación Stuff+ v1 / v4 / Physical | 5 |
| Mapas de calor de Location+ | 6 |
| Diagrama del árbol de probabilidades (Sankey) | 7 |
| Barras de probabilidades del lanzamiento simulado y zona de strike | 8 |
| Trayectoria en 3D/2D original vs. simulada; drag y Magnus por estadio anónimo | 12 |
| Ranking con intervalos (forest) | 10 |
| Dispersión de sensibilidad modelada vs. observada | 11 |

## 7. Inputs del simulador

Solo variables que los modelos usan, o que determinan una que usan:

| input | modelo que lo usa |
|---|---|
| tipo de pitch | los tres |
| mano del pitcher, mano del bateador | los tres (como `same_side`; Location+ también `bat_side`) |
| velocidad de salida (RelSpeed) | Stuff+, Pitching+ |
| IVB, HB (movimiento) | Stuff+, Pitching+ (y definen la trayectoria) |
| spin rate | Stuff+, Pitching+ |
| spin axis | Stuff+, Pitching+ (default derivado del movimiento; se puede ajustar) |
| extensión, altura de salida, lado de salida | Stuff+, Pitching+ (y la trayectoria) |
| ubicación (PlateLocSide, PlateLocHeight) | Location+, Pitching+ (y VAA/HAA ajustados) |
| conteo (bolas, strikes) | Location+, Pitching+ |
| ambiente en el que se describe el movimiento | trayectoria y ajuste ambiental |

Derivados, no editables: VAA/HAA (de la trayectoria), `vaa_loc_adj`/`haa_loc_adj`, diferencias contra la recta (típicas del tipo, o 0 para rectas) y SpeedDrop (solo Pitching+; Stuff+ Physical no lo usa).

## 8. Outputs del simulador

Stuff+ Physical, Location+ y Pitching+ (escala 100), P(evento) de los 10 eventos (suman 1), P(swing), P(whiff | swing), P(BIP), E[Δ | BIP], E[Δ] en carreras por 100 pitches, interpretación (fortalezas y debilidades), alertas de dominio, mejoras posibles y comparación entre ambientes.

**Escala de un lanzamiento individual.** Se usa la misma fórmula que para los pitchers, 100 + 10·(μ − valor)/τ, con μ y τ de la temporada más reciente. Se lee como "un pitcher cuyos lanzamientos fueran todos como este". Un lanzamiento individual varía más que el promedio de un pitcher, así que los valores extremos son más frecuentes; la página lo explica junto al número.

## 9. Reglas de extrapolación

1. Por input: percentil dentro del tipo de pitch. Fuera de [p1, p99] → advertencia; fuera de [p5, p95] → aviso suave.
2. Conjunto: distancia de Mahalanobis respecto de los pitches reales del tipo (velocidad, IVB, HB, spin, extensión, altura y lado de salida). Si supera el percentil 99 de los pitches reales → advertencia "fuera del rango común observado; la predicción tiene menor confiabilidad".
3. Ubicación: si la celda tiene pocos pitches reales de ese tipo → aviso.
4. Los sliders se limitan a [p0.5, p99.5] del tipo; nada fuera del soporte se puede pedir.
5. "¿Qué podría mejorar?" solo propone cambios de ±½ sd del tipo y sin salir de [p5, p95].
6. Con datos faltantes (input vacío) → se usa la mediana del tipo y se avisa.

## 10. Tests (`tests/test_web.py`)

1. Un pitch mediano de su tipo, en una ubicación y un conteo típicos, da un score cercano a 100.
2. Un input fuera de rango produce advertencia.
3. Cambiar solo la ubicación cambia Location+ y no Stuff+ Physical.
4. Cambiar solo variables físicas cambia Stuff+ Physical.
5. Las probabilidades de los eventos suman 1.
6. Pitching+ es monótono decreciente en E[Δ].
7. Inputs faltantes no rompen el motor (se imputan y se avisa).
8. (extra) Streamlit `AppTest`: la página completa se ejecuta sin excepciones.
9. (extra) Todas las cifras de `summary.json` coinciden con sus fuentes.

## 11. Riesgos técnicos

| riesgo | mitigación |
|---|---|
| Extrapolación del simulador | reglas de la sección 9, límites en los sliders y advertencias visibles |
| Escala de un pitch individual (valores extremos) | explicarla junto al número y mostrar también la posición en la distribución real |
| El ambiente en el simulador viene solo de la física (el modelo no ve el parque) | etiqueta "Simulación contrafactual"; texto de que el modelo subestima el daño de contacto en altitud (P2 §7.2) |
| Cifras desalineadas con el pipeline | todos los textos leen `summary.json`; un test lo verifica |
| Rendimiento con 11 modelos | caché de recursos; inferencia de pocas filas |
| Selección por clic en Plotly | sliders de respaldo para la ubicación |
| Anonimización | solo `ANON_STADIUM_xx` y `pitcher_xxxxx`; ningún nombre real en el código ni en los textos |

## 12. Arquitectura de archivos

```
web/
  app.py            # Streamlit: narrativa, layout, componentes (sin lógica de modelos)
  engine.py         # inferencia: simulate, improvements, domain_check, trajectory_paths
  charts.py         # figuras Plotly reutilizables
  style.css         # estilo propio
src/models/stuff_plus_physical.py   # + train_final() → artifacts/stuff_plus_physical.pkl
src/models/location_plus_v2.py      # + guarda artifacts/location_plus_v2.pkl
src/data/parks.py                   # + reports/benchmark/estadios_anonimos.json
src/reports/web_data.py             # summary.json, pitcher_pitch_types.csv, anon_stadiums.csv, simulator_*.json
src/pipeline.py                     # + paso web_data
tests/test_web.py
.claude/launch.json                 # + configuración "web"
reports/web_final_validation.md
README.md                           # cómo correr la página
```

## 13. Ampliación posterior

Se agregó una sección 6 ("¿Qué tipos de lanzamiento conservan mejor su valor en altitud?") entre Location+ y Pitching+, y una sección 14
("Qué mejoraríamos con más datos") al final. El simulador se reorganizó en vista rápida / "¿Por qué?" / "Más detalles" para no mostrar todo a
la vez. Detalle y fuentes en `reports/web_final_validation.md` §6.
