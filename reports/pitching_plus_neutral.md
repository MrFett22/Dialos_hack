# Pitching+ v3 neutral por altitud — índice oficial

Generado por `src/models/pitching_plus_neutral.py`. Tabla oficial: `data/processed/pitching_plus_official.parquet`.

## Qué es

- Cada pitch se **proyecta a nivel del mar** (No Altitude) con el contrafactual de trayectoria completa (`reports/altitude_counterfactual_v2.md`) y se valúa en carreras con la cadena de Pitching+ v3, con predicciones out-of-fold por pitcher.
- Se neutraliza el **efecto total** de la altitud (calidad del pitch + entorno de bola en juego), por decisión del usuario.
- **Shrinkage empírico-bayesiano:** el promedio de cada pitcher-temporada se acerca a la media de la liga según su volumen; un pitcher con pocos pitches queda cerca de 100 con un intervalo ancho, en vez de quedar fuera del ranking.
- **Escala:** 100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ.
- **Incertidumbre:** intervalo de credibilidad empírico-bayesiano de 80%. Es la distribución posterior del modelo normal-normal con μ, τ y σ fijos en sus estimaciones; refleja cuántos pitches tiene el pitcher. NO incluye la incertidumbre de μ, τ y σ, de los modelos ML, de los ratios de densidad ni de los hiperparámetros, así que es más angosta que la incertidumbre total.

## Parámetros del shrinkage por temporada (carreras/pitch)

| temporada | μ (media liga) | τ (habilidad real) | σ (pitch a pitch) | pitches para encoger a la mitad | pitchers usados |
|---|---:|---:|---:|---:|---:|
| 2024 | -0.00411 | 0.00654 | 0.0537 | 67 | 436 |
| 2025 | -0.00553 | 0.00586 | 0.0544 | 86 | 474 |
| 2026 | -0.00556 | 0.00651 | 0.0540 | 69 | 368 |

*Pitches para encoger a la mitad* = n con el que el score de un pitcher se acerca 50% a la media (σ²/τ²).

## ¿Funciona la neutralización?

Correlación del score con el % de pitches del pitcher en cada categoría (pitcher-temporadas con ≥100 pitches):

| score | corr. con % Extreme | corr. con % Medium |
|---|---:|---:|
| Pitching+ v3 con altitud (mismo shrinkage) | -0.325 | -0.072 |
| Pitching+ v3 neutral | -0.065 | +0.035 |
| Pitching+ v3 neutral + shrinkage (oficial) | -0.062 | +0.032 |

Promedio del score según dónde lanzó el pitcher la mayoría de sus pitches:

| entorno | n | con altitud | neutral | neutral + shrinkage |
|---|---:|---:|---:|---:|
| mayoría Extreme | 311 | 95.12 | 98.22 | 98.86 |
| mayoría No Altitude | 560 | 102.10 | 98.56 | 99.13 |
| mixto | 407 | 97.31 | 98.23 | 98.75 |

## Benchmark año-contra-año — mínimo 100 pitches (437 pares)

| métrica | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |
|---|---|---|---|---|
| Pitching+ v1 | 0.536 [0.466, 0.600] | 0.188 [0.097, 0.278] | 0.240 [0.140, 0.335] | 0.171 [0.061, 0.274] |
| Pitching+ v3 con altitud | 0.560 [0.489, 0.626] | 0.177 [0.089, 0.265] | 0.177 [0.069, 0.278] | 0.125 [0.016, 0.228] |
| Pitching+ v3 neutral | 0.567 [0.495, 0.633] | 0.141 [0.048, 0.232] | 0.176 [0.066, 0.275] | 0.114 [-0.000, 0.227] |
| Pitching+ v3 neutral + shrinkage (oficial) | 0.581 [0.512, 0.643] | 0.153 [0.062, 0.244] | 0.185 [0.076, 0.285] | 0.123 [0.010, 0.234] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.349 [0.260, 0.429] | 0.202 [0.112, 0.283] | 0.284 [0.197, 0.362] | 0.357 [0.269, 0.442] |
| *K-BB% observado* | 0.263 [0.179, 0.342] | 0.181 [0.096, 0.264] | 0.263 [0.179, 0.342] | 0.296 [0.212, 0.376] |

Delta contra Pitching+ v3 con altitud (bootstrap pareado):

| métrica | estabilidad | carreras evitadas t+1 |
|---|---|---|
| Pitching+ v3 neutral | +0.007 [-0.039, +0.055] · P=62% | -0.036 [-0.081, +0.010] · P=6% |
| Pitching+ v3 neutral + shrinkage (oficial) | +0.021 [-0.021, +0.066] · P=84% | -0.024 [-0.069, +0.022] · P=15% |

## Benchmark año-contra-año — mínimo 300 pitches (245 pares)

| métrica | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |
|---|---|---|---|---|
| Pitching+ v1 | 0.605 [0.514, 0.678] | 0.200 [0.072, 0.320] | 0.299 [0.165, 0.426] | 0.192 [0.032, 0.332] |
| Pitching+ v3 con altitud | 0.596 [0.502, 0.673] | 0.221 [0.099, 0.344] | 0.238 [0.095, 0.376] | 0.148 [0.008, 0.281] |
| Pitching+ v3 neutral | 0.627 [0.540, 0.696] | 0.208 [0.078, 0.330] | 0.237 [0.091, 0.375] | 0.128 [-0.019, 0.273] |
| Pitching+ v3 neutral + shrinkage (oficial) | 0.632 [0.547, 0.700] | 0.210 [0.079, 0.330] | 0.239 [0.095, 0.375] | 0.132 [-0.012, 0.277] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.293 [0.166, 0.404] | 0.106 [-0.017, 0.224] | 0.229 [0.104, 0.345] | 0.367 [0.237, 0.479] |
| *K-BB% observado* | 0.323 [0.205, 0.430] | 0.193 [0.064, 0.315] | 0.323 [0.205, 0.430] | 0.377 [0.261, 0.478] |

Delta contra Pitching+ v3 con altitud (bootstrap pareado):

| métrica | estabilidad | carreras evitadas t+1 |
|---|---|---|
| Pitching+ v3 neutral | +0.031 [-0.027, +0.091] · P=86% | -0.013 [-0.065, +0.044] · P=32% |
| Pitching+ v3 neutral + shrinkage (oficial) | +0.036 [-0.021, +0.096] · P=90% | -0.011 [-0.064, +0.046] · P=34% |

## Prueba secundaria (post-hoc): carreras del año siguiente ajustadas por entorno

**Definida después de ver resultados.** La validación preespecificada (carreras observadas t+1, mín. 100) y la comparación obligatoria contra el ajuste trivial por temporada × categoría están en `reports/baselines.md`: el índice neutral no supera a ese ajuste trivial, así que esta tabla no demuestra que la física aporte valor predictivo.

Las carreras observadas en t+1 incluyen el efecto del parque, y la mayoría de los pitchers repite parque: un score que conserva la altitud se beneficia de esa persistencia. Aquí el resultado t+1 son las carreras evitadas **sobre lo esperado en su categoría de altitud** (valor realizado de cada pitch menos el promedio de su categoría y temporada).

| mín. pitches | pares | con altitud | neutral | neutral + shrinkage | Δ oficial vs con altitud | carreras ajustadas t (baseline) |
|---|---:|---|---|---|---|---|
| 100 | 437 | 0.165 [0.074, 0.251] | 0.169 [0.077, 0.261] | 0.181 [0.090, 0.272] | +0.016 [-0.028, +0.060] · P=78% | 0.165 [0.090, 0.239] |
| 300 | 245 | 0.215 [0.077, 0.334] | 0.279 [0.146, 0.401] | 0.279 [0.145, 0.401] | +0.065 [+0.010, +0.123] · P=99% | 0.271 [0.147, 0.378] |

## Top 15 (≥100 pitches)

| pitcher_anon_id   |   year |   n_pitches |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |   pitching_plus_con_altitud |   pct_extreme_altitude |   costo_altitud_total_por_100 |
|:------------------|-------:|------------:|----------------:|---------------------:|---------------------:|----------------------------:|-----------------------:|------------------------------:|
| pitcher_00619     |   2024 |        1464 |         130.961 |              128.274 |              133.649 |                     125.413 |                  0.583 |                         1.187 |
| pitcher_00469     |   2026 |         283 |         129.115 |              123.45  |              134.78  |                     130.639 |                  0.148 |                         0.828 |
| pitcher_00465     |   2026 |         357 |         128.227 |              123.078 |              133.376 |                     128.446 |                  0.095 |                         1.397 |
| pitcher_00069     |   2025 |         807 |         124.379 |              120.399 |              128.359 |                     120.187 |                  0.445 |                         1.766 |
| pitcher_00115     |   2025 |         623 |         120.68  |              116.214 |              125.146 |                     116.641 |                  0.09  |                         1.077 |
| pitcher_00680     |   2024 |         562 |         120.193 |              116.001 |              124.386 |                     113.97  |                  0.63  |                         1.232 |
| pitcher_00714     |   2025 |         660 |         119.863 |              115.509 |              124.217 |                     113.623 |                  0.218 |                         1.513 |
| pitcher_00285     |   2025 |         181 |         119.657 |              112.38  |              126.934 |                     111.389 |                  0.547 |                         1.61  |
| pitcher_00036     |   2026 |         528 |         119.407 |              115.058 |              123.756 |                     124.17  |                  0     |                         1.128 |
| pitcher_00263     |   2026 |         102 |         118.994 |              110.863 |              127.125 |                     123.264 |                  0     |                         1.17  |
| pitcher_00081     |   2025 |        1622 |         118.762 |              115.884 |              121.64  |                     118.424 |                  0.109 |                         0.962 |
| pitcher_00904     |   2024 |         379 |         118.396 |              113.417 |              123.375 |                     113.539 |                  0.683 |                         1.066 |
| pitcher_00703     |   2024 |         556 |         118.297 |              114.084 |              122.51  |                     112.027 |                  0.554 |                         1.351 |
| pitcher_00456     |   2026 |         220 |         117.951 |              111.699 |              124.203 |                     120.325 |                  0.155 |                         0.865 |
| pitcher_00059     |   2024 |         443 |         117.899 |              113.243 |              122.555 |                     120.642 |                  0.113 |                         1.537 |

## Pitchers que más suben al neutralizar (≥300 pitches)

| pitcher_anon_id   |   year |   n_pitches |   pct_extreme_altitude |   pitching_plus_con_altitud |   pitching_plus |   ajuste_altitud_carreras_por_100 |   cambio_ranking |
|:------------------|-------:|------------:|-----------------------:|----------------------------:|----------------:|----------------------------------:|-----------------:|
| pitcher_00534     |   2025 |         528 |                   1    |                       94.28 |          104.79 |                              1.23 |              262 |
| pitcher_00193     |   2025 |         605 |                   0.75 |                       95.43 |          104.59 |                              1.12 |              224 |
| pitcher_01032     |   2026 |         500 |                   0.61 |                       97.27 |          107.27 |                              1.21 |              221 |
| pitcher_00851     |   2025 |         464 |                   0.71 |                       99.03 |          109.77 |                              1.25 |              209 |
| pitcher_00812     |   2025 |         642 |                   0.71 |                       95.56 |          104.08 |                              1.07 |              205 |
| pitcher_00776     |   2026 |         554 |                   0.79 |                       96.57 |          104.94 |                              1.09 |              201 |
| pitcher_00948     |   2024 |         715 |                   0.69 |                       95.91 |          104.08 |                              1.07 |              196 |
| pitcher_00144     |   2025 |         518 |                   0.74 |                      100.51 |          112.53 |                              1.32 |              192 |
| pitcher_00236     |   2026 |         783 |                   0.89 |                       95.82 |          103.54 |                              1.03 |              190 |
| pitcher_00169     |   2025 |         477 |                   0.61 |                       90.75 |           98.68 |                              1.06 |              190 |

`cambio_ranking` = posiciones que sube en el ranking de su temporada (todos los pitchers) al neutralizar. `ajuste_altitud_carreras_por_100` = carreras por 100 pitches que el entorno le cargaba.

## Pitchers con muestra chica: el shrinkage en acción

| pitcher_anon_id   |   year |   n_pitches |   pitching_plus_sin_shrinkage |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |
|:------------------|-------:|------------:|------------------------------:|----------------:|---------------------:|---------------------:|
| pitcher_00084     |   2024 |           1 |                        156.54 |          100.83 |                88.11 |               113.55 |
| pitcher_00655     |   2026 |           3 |                        142.67 |          101.79 |                89.24 |               114.33 |
| pitcher_00933     |   2026 |          13 |                        141.38 |          106.58 |                94.83 |               118.34 |
| pitcher_00433     |   2026 |           7 |                        139.74 |          103.67 |                91.47 |               115.88 |
| pitcher_00820     |   2026 |           4 |                        138.78 |          102.13 |                89.67 |               114.59 |
