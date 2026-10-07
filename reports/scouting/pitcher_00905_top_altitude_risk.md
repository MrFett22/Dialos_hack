# Ficha de scouting — pitcher_00905

Índice oficial: **Pitching+ v3 neutral por altitud** (100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). Altitud extrema = varios estadios de la categoría Extreme del dataset (estadios anónimos; el dataset no identifica estadios reales). Generado por `src/reports/scouting_report_v2.py`.

## 1. Resumen por temporada

|   year |   n_pitches |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |   percentil_temporada |   pitching_plus_con_altitud |   stuff_plus_v1 |   pct_no_altitude |   pct_medium_altitude |   pct_extreme_altitude |   costo_altitud_total_por_100 |   costo_altitud_antes_contacto_por_100 |
|-------:|------------:|----------------:|---------------------:|---------------------:|----------------------:|----------------------------:|----------------:|------------------:|----------------------:|-----------------------:|------------------------------:|---------------------------------------:|
|   2024 |         480 |          110.33 |               105.83 |               114.82 |                    93 |                      113.33 |          105.59 |                82 |                    11 |                      7 |                          1.75 |                                   0.76 |
|   2025 |         457 |          104.99 |                99.88 |               110.09 |                    80 |                      108.66 |          102.63 |                83 |                     2 |                     15 |                          1.76 |                                   0.71 |
|   2026 |          91 |          103.21 |                94.81 |               111.62 |                    71 |                      107.31 |          100.12 |               100 |                     0 |                      0 |                          1.58 |                                   0.49 |

`pitching_plus` = neutral por altitud con shrinkage (oficial) · `lo80`/`hi80` = intervalo de credibilidad empírico-bayesiano de 80% · `pitching_plus_con_altitud` = mismo índice sin quitar el entorno · `stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado) · `pct_*` = % de sus pitches en cada categoría · `costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar.

## 2. Arsenal (2025)

| AutoPitchType   |   n |   uso % |   velo (mph) |   IVB (pulg) |   HB (pulg) |   spin (rpm) |   stuff: whiff/swing vs su tipo (pp) |   carreras evitadas/100 vs liga (nivel del mar) |   carreras evitadas/100 vs liga (altitud extrema) |   costo altitud: antes del contacto/100 |   costo altitud: total/100 |
|:----------------|----:|--------:|-------------:|-------------:|------------:|-------------:|-------------------------------------:|------------------------------------------------:|--------------------------------------------------:|----------------------------------------:|---------------------------:|
| Four-Seam       | 285 |   62.36 |        88.2  |        20.08 |        7.85 |      2330.06 |                                 0.02 |                                            0.85 |                                              0.07 |                                    0.93 |                       2.23 |
| Slider          |  92 |   20.13 |        82.81 |         7.4  |        0.35 |      2183.36 |                                 1.25 |                                            0.1  |                                              0.05 |                                    0.26 |                       0.7  |
| Changeup        |  67 |   14.66 |        83.73 |        11.18 |       10.62 |      1857.21 |                                -1.57 |                                           -1    |                                             -1.28 |                                    0.49 |                       1.43 |

*Stuff: whiff/swing vs su tipo*: puntos porcentuales de probabilidad de swing-and-miss que el modelo espera por las características físicas del pitch (sin ubicación, a nivel del mar), contra el promedio de la liga para ese mismo tipo. Solo es comparable entre pitches del mismo tipo. *Carreras evitadas/100 vs liga*: cuántas carreras evita cada 100 pitches de ese tipo contra el promedio de la liga del mismo tipo en el mismo entorno (nivel del mar o altitud extrema). *Costo de altitud*: carreras extra cada 100 pitches en altitud extrema; *antes del contacto* = swing, take, whiff y foul, incluida la mayor probabilidad de contacto (atribución de Shapley); *total* = incluye el resultado de la bola en juego (que mezcla calidad del contacto y parque).

## 3. Recomendaciones (2025)

- **Nivel:** Pitching+ 105.0 (percentil 80 de su temporada), intervalo de credibilidad de 80% [99.9, 110.1] (solo refleja el tamaño de muestra; no la incertidumbre de los modelos).
- **Riesgo de altitud ALTO (cuartil superior de la liga) (según el modelo):** en altitud extrema su arsenal permite +1.76 carreras más por cada 100 pitches que a nivel del mar; de eso, +0.71 ocurre antes del contacto (menos whiffs, más pelotas en juego) y el resto es el resultado de la bola en juego, que mezcla calidad del contacto y parque. La parte antes del contacto equivale a ≈+19.5 carreras por temporada de abridor (≈+6.4 de relevista). Es una sensibilidad implicada por el modelo (depende de qué lanza), no una resistencia a la altitud medida en sus propios juegos (`reports/sensibilidad_altitud.md`).
- **Pitch que más pierde antes del contacto en altitud:** Four-Seam (+0.93 carreras/100).
- **Mejor pitch a nivel del mar:** Four-Seam (+0.85 carreras/100 vs. el promedio de liga del mismo tipo).
- **En altitud extrema Four-Seam sigue siendo su mejor pitch** frente al promedio de su tipo en el mismo entorno.
