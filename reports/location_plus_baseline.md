# Location+ v1 — modelo baseline (GAM / pygam, solo ubicacion + conteo + lateralidad)

Pregunta: dado (aproximadamente) el mismo pitch fisico, ¿cuanto cambia el resultado SOLO por donde se ubico y en que conteo? Deliberadamente NO se usa ninguna feature fisica del pitch (RelSpeed, SpinRate, movimiento, angulos) — esas son de Stuff+.

Features: PlateLocSide, PlateLocHeight, Balls, Strikes, BatterSide, AutoPitchType — PlateLocSide/PlateLocHeight con tensor spline 2D (te()), el resto como terminos de factor (f()).

Filas usadas: 627,665 (de las cuales se excluyeron 71 por BatterSide nulo)
- Train: 503,077 pitches, 837 pitchers
- Test: 124,588 pitches, 208 pitchers

## is_called_strike (target oficial de Location+)

| | AUC | log-loss | n | base rate |
|---|---:|---:|---:|---:|
| Modelo (test) | 0.8468 | 0.3184 | 124,588 | 0.1539 |
| Modelo (train) | 0.8492 | 0.3173 | 503,077 | 0.1542 |
| Baseline ingenuo (test) — promedio por (count, BatterSide) | 0.7295 | 0.3825 | 124,588 | 0.1539 |

Mejora del modelo sobre el baseline (test): AUC +0.1174, log-loss -0.0640 (mas bajo es mejor).

Heatmap: `reports/figures/location_heatmaps/is_called_strike_heatmaps.png`

## is_chase (target oficial de Location+)

| | AUC | log-loss | n | base rate |
|---|---:|---:|---:|---:|
| Modelo (test) | 0.7993 | 0.4348 | 124,588 | 0.2232 |
| Modelo (train) | 0.8001 | 0.4326 | 503,077 | 0.2222 |
| Baseline ingenuo (test) — promedio por (count, BatterSide) | 0.6352 | 0.5115 | 124,588 | 0.2232 |

Mejora del modelo sobre el baseline (test): AUC +0.1641, log-loss -0.0767 (mas bajo es mejor).

Heatmap: `reports/figures/location_heatmaps/is_chase_heatmaps.png`

## is_whiff (SOLO COMPARACION, no oficial)

| | AUC | log-loss | n | base rate |
|---|---:|---:|---:|---:|
| Modelo (test) | 0.6839 | 0.3361 | 124,588 | 0.1160 |
| Modelo (train) | 0.6809 | 0.3301 | 503,077 | 0.1127 |
| Baseline ingenuo (test) — promedio por (count, BatterSide) | 0.5757 | 0.3548 | 124,588 | 0.1160 |

Mejora del modelo sobre el baseline (test): AUC +0.1082, log-loss -0.0187 (mas bajo es mejor).

Heatmap: `reports/figures/location_heatmaps/is_whiff_heatmaps.png`

## Comparacion con el ablation de Stuff+

Location+ (solo ubicacion+conteo+lateralidad, SIN fisica del pitch) obtiene AUC=0.6839 para `is_whiff` — comparar contra Stuff+ (solo fisica, sin ubicacion/conteo), que obtuvo AUC=0.7297 en reports/stuff_plus_baseline.md. 
**Confirmacion simetrica del hallazgo del ablation de Stuff+:** aqui, con SOLO ubicacion/conteo/lateralidad (sin fisica), `is_whiff` es el target que peor predice el modelo de los tres — menos que is_called_strike y is_chase, que si son fundamentalmente de ubicacion/conteo. Confirma que is_whiff depende de la fisica del pitch (Stuff+), no de donde se ubico (Location+); is_called_strike/is_chase son al reves: dependen de ubicacion, no de fisica.

## Notas de los heatmaps

Grilla PlateLocSide en [-2.0,2.0] ft x PlateLocHeight en [0.0,5.0] ft (no el rango completo del dataset, que llega a pitches muy desviados e inutiles para visualizar la zona). AutoPitchType fijo en 'Four-Seam' (el mas comun). El rectangulo negro es una zona de referencia aproximada (17in de plato + radio de pelota, ~rodilla a letras) — NO es la zona dinamica por bateador (`in_strike_zone`) del dataset, solo una guia visual.

Verificar visualmente en las figuras: la zona de is_called_strike deberia verse mas generosa/ancha en conteo de pitcher (0-0) que en conteo de bateador (3-2) si el modelo aprendio el sesgo real de los umpires; is_chase deberia concentrarse fuera del rectangulo de referencia, mas en 3-2 (el bateador se ve forzado a perseguir mas cerca de la zona) que en 0-0.

`BatterSide` tiene 4 valores reales en el dato (no solo Left/Right), por eso los heatmaps salen con 4 filas: Right=364,986, Left=262,491, Undefined=167, Switch=21. Los paneles de `Switch` y `Undefined` (21 y 167 pitches respectivamente en el subconjunto usado para entrenar, <0.04% del dataset cada uno) son estimaciones del factor `f()` con muestra muy chica — tratar esos dos paneles especificos con cautela, no como una lectura confiable del sesgo real de umpire para esos grupos.

Modelo guardado en `src/models/artifacts/location_plus_v1.pkl` (is_called_strike y is_chase oficiales; is_whiff guardado aparte como comparison_models, no es parte de Location+).
