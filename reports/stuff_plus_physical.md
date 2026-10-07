# Stuff+ Physical — calidad física del pitch sin crédito a ubicación ni ambiente

Generado por `src/models/stuff_plus_physical.py`. Las decisiones (target, features, versión oficial y regla para reportar una o dos métricas) están en el docstring y se fijaron **antes** de entrenar.

## 1. Auditoría de features

| Variable | Física | Contiene ubicación | Contiene altitud | Posible leakage | Usar | Motivo |
|---|---|---|---|---|---|---|
| RelSpeed | sí | no | mínima (se mide en la salida) | no | sí | velocidad de salida del brazo |
| EffectiveVelo | sí | no | mínima | no | no | redundante: velocidad + extensión |
| ZoneSpeed | sí | no | sí (drag) | altitud | no | velocidad al llegar = salida − pérdida por drag |
| SpeedDrop | parcial | no | sí, casi toda (drag) | altitud | no | RelSpeed − ZoneSpeed: mide sobre todo el aire (quitada respecto de v4) |
| ZoneTime | parcial | no | sí (drag) | altitud | no | redundante con velocidad, extensión y drag |
| SpinRate | sí | no | mínima (se mide en la salida) | no | sí | giro |
| SpinAxis | sí | no | no | no | sí | eje de giro (espejado para zurdos) |
| Tilt | sí | no | no | no | no | el mismo eje en formato de reloj (redundante) |
| HorzBreak | sí | no | sí (Magnus) | altitud parcial | sí | movimiento horizontal; el ambiente se trata con el ajuste por estadio |
| InducedVertBreak | sí | no | sí (Magnus) | altitud parcial | sí | movimiento vertical inducido; ídem |
| VertBreak | sí | no | sí | altitud | no | incluye gravedad; redundante con IVB + velocidad |
| pfxx / pfxz | sí | no | sí | altitud | no | las mismas componentes de movimiento (redundantes) |
| RelHeight | sí | no | no | no | sí | altura de salida |
| RelSide | sí | no | no | no | sí | lado de salida (espejado) |
| Extension | sí | no | no | no | sí | extensión |
| VertRelAngle / HorzRelAngle | parcial | sí (a dónde apunta) | no | ubicación | no | el ángulo de salida decide en buena parte dónde cruza |
| VertApprAngle / HorzApprAngle | parcial | sí (fuerte) | parcial | ubicación | no (crudos) | un pitch bajo llega más inclinado por estar bajo |
| vaa_loc_adj / haa_loc_adj | sí | residualizada | parcial | baja | sí | ángulo de llegada menos lo que explica la ubicación (recta por tipo de pitch) |
| x0…az0 (9 parámetros) | sí | vx0, vz0 sí | ax, az sí | ubicación y altitud | no | redundantes con salida y movimiento; las velocidades iniciales codifican el destino |
| magnus_residual_vert / _horz | derivada | no | sí (supone elevaciones) | altitud | no | construida con la categoría de altitud |
| altitude_level / altitude_category | no | no | sí | ambiente | no | es el ambiente |
| PlateLocHeight / PlateLocSide | no | sí | no | ubicación | no | es la ubicación (Location+) |
| Balls / Strikes / count | no | no | no | contexto | no | es el conteo (Location+ / Pitching+) |
| velo_diff_vs_fb / ivb_diff_vs_fb / hb_diff_vs_fb | sí (arsenal) | no | parcial (se cancela en parte) | baja | sí | contraste con la recta del pitcher |
| tunnel_similarity | derivada | sí (usa VAA/HAA crudos) | parcial | ubicación | no | usa ángulos de llegada sin ajustar |
| pitch_usage_pct / prev_pitch_type | no | no | no | uso y secuencia | no | decisiones de uso, no física |
| AutoPitchType | sí (clasificación) | no | no | no | sí | tipo de pitch |
| same_side (mano bateador vs pitcher) | no | no | no | contexto del enfrentamiento | solo para entrenar | se marginaliza al predecir (mezcla de lados de la liga) |

Features finales: `RelSpeed`, `SpinRate`, `SpinAxis`, `RelHeight`, `RelSide`, `Extension`, `InducedVertBreak`, `HorzBreak`, `vaa_loc_adj`, `haa_loc_adj`, `velo_diff_vs_fb`, `ivb_diff_vs_fb`, `hb_diff_vs_fb`, `AutoPitchType`, `same_side`. Zurdos espejados (marco de un derecho).

## 2. Target

| candidato | qué mide | problema | decisión |
|---|---|---|---|
| P(whiff \| swing) | capacidad de hacer fallar al bateador **una vez que decide tirar** | no mide si induce el swing | **elegido** |
| whiff por pitch | lo anterior **más** la capacidad de inducir el swing | el swing depende sobre todo de ubicación y conteo: mezcla Stuff con Location | descartado (es el target de v1) |
| evitar contacto | 1 − P(contacto \| swing) | es el mismo target que P(whiff \| swing) | equivalente |
| valor esperado sin ubicación | todo el árbol en carreras | sin ubicación, swing y take son casi ruido; el componente de contacto es estable pero no predice t+1 (v3 DIPS) | descartado; el daño de contacto va en el perfil |

## 3. Modelo

- OOF por pitcher (GroupKFold k=5) sobre 293,559 swings. Tasa base 0.2424.
- AUC del modelo (con el lado real del bateador): 0.6508; log-loss 0.5280.
- AUC del puntaje (lados marginalizados con la mezcla de la liga, 50.5% mismo lado): 0.6454; log-loss 0.5299.

## 4. Comparación pitch a pitch (universo: swings)

| modelo | AUC whiff \| swing | log-loss | nota |
|---|---:|---:|---|
| Stuff+ v1 (P(whiff) por pitch) | 0.6858 | — | target distinto: whiff sobre todos los pitches; su log-loss no es comparable |
| Stuff+ v4 | 0.6507 | 0.5281 |  |
| Stuff+ Physical (modelo) | 0.6508 | 0.5280 |  |
| Stuff+ Physical (puntaje, lados marginalizados) | 0.6454 | 0.5299 |  |

Stuff+ v1 se entrenó para whiff por pitch (AUC 0.7315 en su universo, todos los pitches); aquí se evalúa también sobre swings para ponerlo en la misma pregunta.

## 5. Sensibilidad a la ubicación y al ambiente

*Ubicación:* fracción de la varianza de la predicción pitch a pitch que explica la celda de ubicación (0.5 × 0.5 pies) **dentro** de cada tipo de pitch. *Ambiente:* diferencia media de la predicción para el mismo pitcher, tipo de pitch y temporada (puntos porcentuales de whiff).

| modelo | varianza explicada por ubicación | Medium − No (pp) | Extreme − No (pp) |
|---|---:|---:|---:|
| Stuff+ v1 | 45.6% | -0.42 | -0.80 |
| Stuff+ v4 | 2.2% | -1.06 | -1.80 |
| Stuff+ Physical | 1.9% | -1.07 | -2.08 |
| Stuff+ Physical ajustado por estadio | — | -0.16 | -0.35 |
| Stuff+ Physical neutral | — | +0.30 | +0.42 |

Por pitcher-temporada (≥ 100 pitches):

| puntaje | r con % de pitches en zona | r con % de pitches en Extreme |
|---|---:|---:|
| Stuff+ v1 benchmark | +0.088 | -0.144 |
| Stuff+ v4 | -0.015 | -0.003 |
| Physical observado | +0.021 | -0.163 |
| Physical (oficial) | -0.002 | -0.015 |
| Physical neutral | -0.004 | -0.006 |

## 6. Estabilidad y predicción del año siguiente

Resultados t+1 con ajuste simple por entorno. IC 95% por bootstrap de pitchers (1,000 réplicas).

**Mínimo 100 pitches (437 pares)**

| puntaje | estabilidad | whiff/swing t+1 | K% t+1 | carreras evitadas t+1 |
|---|---|---|---|---|
| Stuff+ v1 benchmark | +0.668 [+0.589, +0.735] | +0.332 [+0.249, +0.417] | +0.301 [+0.211, +0.382] | +0.168 [+0.078, +0.255] |
| Stuff+ v4 | +0.769 [+0.726, +0.807] | +0.304 [+0.207, +0.398] | +0.264 [+0.170, +0.351] | +0.142 [+0.053, +0.236] |
| Physical observado | +0.745 [+0.701, +0.786] | +0.323 [+0.232, +0.402] | +0.283 [+0.188, +0.364] | +0.163 [+0.077, +0.251] |
| Physical (oficial) | +0.760 [+0.715, +0.800] | +0.340 [+0.252, +0.418] | +0.297 [+0.201, +0.378] | +0.173 [+0.087, +0.262] |
| Physical neutral | +0.780 [+0.738, +0.817] | +0.308 [+0.212, +0.398] | +0.281 [+0.188, +0.369] | +0.152 [+0.062, +0.247] |

| puntaje | Δr whiff/swing t+1 vs v4 | Δr whiff/swing t+1 vs v1 | Δr carreras t+1 vs v1 |
|---|---|---|---|
| Physical (oficial) | +0.036 [+0.001, +0.072] (P=98%) | +0.008 [-0.076, +0.087] (P=59%) | +0.006 [-0.068, +0.084] (P=56%) |
| Physical neutral | +0.004 [-0.013, +0.019] (P=66%) | -0.024 [-0.113, +0.064] (P=30%) | -0.016 [-0.097, +0.070] (P=37%) |
| Physical observado | +0.018 [-0.023, +0.057] (P=80%) | -0.010 [-0.091, +0.066] (P=40%) | -0.005 [-0.078, +0.072] (P=44%) |

**Mínimo 300 pitches (245 pares)**

| puntaje | estabilidad | whiff/swing t+1 | K% t+1 | carreras evitadas t+1 |
|---|---|---|---|---|
| Stuff+ v1 benchmark | +0.710 [+0.626, +0.780] | +0.274 [+0.160, +0.400] | +0.236 [+0.097, +0.363] | +0.049 [-0.076, +0.190] |
| Stuff+ v4 | +0.791 [+0.733, +0.842] | +0.280 [+0.155, +0.408] | +0.231 [+0.106, +0.354] | +0.076 [-0.054, +0.214] |
| Physical observado | +0.744 [+0.685, +0.797] | +0.337 [+0.223, +0.457] | +0.265 [+0.141, +0.383] | +0.084 [-0.039, +0.219] |
| Physical (oficial) | +0.754 [+0.697, +0.808] | +0.356 [+0.249, +0.467] | +0.287 [+0.170, +0.401] | +0.106 [-0.025, +0.243] |
| Physical neutral | +0.793 [+0.739, +0.842] | +0.295 [+0.176, +0.421] | +0.257 [+0.134, +0.377] | +0.097 [-0.033, +0.233] |

| puntaje | Δr whiff/swing t+1 vs v4 | Δr whiff/swing t+1 vs v1 | Δr carreras t+1 vs v1 |
|---|---|---|---|
| Physical (oficial) | +0.076 [+0.027, +0.122] (P=100%) | +0.081 [-0.029, +0.199] (P=92%) | +0.057 [-0.070, +0.188] (P=81%) |
| Physical neutral | +0.015 [-0.007, +0.039] (P=90%) | +0.021 [-0.100, +0.144] (P=64%) | +0.048 [-0.082, +0.180] (P=77%) |
| Physical observado | +0.057 [+0.003, +0.104] (P=98%) | +0.063 [-0.043, +0.181] (P=87%) | +0.035 [-0.090, +0.159] (P=72%) |

## 7. Por tipo de pitch

AUC de whiff | swing dentro de cada tipo (swings):

| tipo      |   swings |   whiff/swing observado |   Stuff+ v1 |   Stuff+ v4 |   Stuff+ Physical |
|:----------|---------:|------------------------:|------------:|------------:|------------------:|
| Four-Seam |    99535 |                  0.1987 |      0.6497 |      0.6216 |            0.6133 |
| Sinker    |    49784 |                  0.1463 |      0.623  |      0.5865 |            0.5759 |
| Cutter    |    12752 |                  0.2239 |      0.5787 |      0.6032 |            0.5959 |
| Slider    |    66323 |                  0.3192 |      0.6869 |      0.591  |            0.5817 |
| Curveball |    23083 |                  0.3303 |      0.7227 |      0.5769 |            0.5661 |
| Changeup  |    38055 |                  0.2922 |      0.6535 |      0.6226 |            0.6219 |
| Splitter  |     3695 |                  0.3288 |      0.6671 |      0.5963 |            0.5995 |

P(whiff | swing) promedio del puntaje por tipo (observado y proyectado a No Altitude):

| AutoPitchType   |   p_whiff_swing_obs |   p_whiff_swing_neutral |
|:----------------|--------------------:|------------------------:|
| Four-Seam       |              0.1938 |                  0.22   |
| Sinker          |              0.1438 |                  0.1525 |
| Cutter          |              0.2232 |                  0.2316 |
| Slider          |              0.3104 |                  0.3197 |
| Curveball       |              0.3276 |                  0.3425 |
| Changeup        |              0.2898 |                  0.3022 |
| Splitter        |              0.3204 |                  0.326  |

## 8. ¿Una o dos métricas?

- r(oficial, neutral) = **0.957**; r(oficial, observado sin ajuste) = 0.974; r(oficial, v4) = 0.946; r(oficial, v1) = 0.681 (pitcher-temporadas con ≥ 100 pitches).
- La correlación queda por debajo de 0.98: las dos versiones ordenan distinto a algunos pitchers, así que se conservan ambas. `stuff_plus_physical` (ajuste simple) es la oficial; `stuff_plus_physical_neutral` es la contrafactual y se usa para explicar, no para rankear.

Distribución del oficial (pitcher-temporadas): media 99.2, sd 10.1.
