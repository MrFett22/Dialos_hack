# Ficha de scouting — pitcher_00081

Índice oficial: **Pitching+ v3 neutral por altitud** (100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). Altitud extrema = varios estadios de la categoría Extreme del dataset (estadios anónimos; el dataset no identifica estadios reales). Generado por `src/reports/scouting_report_v2.py`.

## 1. Resumen por temporada

|   year |   n_pitches |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |   percentil_temporada |   pitching_plus_con_altitud |   stuff_plus_v1 |   pct_no_altitude |   pct_medium_altitude |   pct_extreme_altitude |   costo_altitud_total_por_100 |   costo_altitud_antes_contacto_por_100 |
|-------:|------------:|----------------:|---------------------:|---------------------:|----------------------:|----------------------------:|----------------:|------------------:|----------------------:|-----------------------:|------------------------------:|---------------------------------------:|
|   2025 |        1622 |          118.76 |               115.88 |               121.64 |                    99 |                      118.42 |          106.08 |                26 |                    64 |                     11 |                          0.96 |                                   0.37 |

`pitching_plus` = neutral por altitud con shrinkage (oficial) · `lo80`/`hi80` = intervalo de credibilidad empírico-bayesiano de 80% · `pitching_plus_con_altitud` = mismo índice sin quitar el entorno · `stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado) · `pct_*` = % de sus pitches en cada categoría · `costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar.

## 2. Arsenal (2025)

| AutoPitchType   |   n |   uso % |   velo (mph) |   IVB (pulg) |   HB (pulg) |   spin (rpm) |   stuff: whiff/swing vs su tipo (pp) |   carreras evitadas/100 vs liga (nivel del mar) |   carreras evitadas/100 vs liga (altitud extrema) |   costo altitud: antes del contacto/100 |   costo altitud: total/100 |
|:----------------|----:|--------:|-------------:|-------------:|------------:|-------------:|-------------------------------------:|------------------------------------------------:|--------------------------------------------------:|----------------------------------------:|---------------------------:|
| Four-Seam       | 488 |   30.09 |        91.81 |        13.41 |        4.8  |      2421.23 |                                -0.35 |                                            0.55 |                                              0.81 |                                    0.37 |                       1.19 |
| Slider          | 449 |   27.68 |        83.35 |         3.5  |      -11.3  |      2791.51 |                                 1.53 |                                            1.69 |                                              1.7  |                                    0.36 |                       0.65 |
| Sinker          | 258 |   15.91 |        91.6  |         8.25 |       12.83 |      2321.29 |                                -1.7  |                                            1.19 |                                              1.05 |                                    0.3  |                       1.34 |
| Curveball       | 173 |   10.67 |        79.45 |        -9.44 |      -13.51 |      2915.59 |                                 1.7  |                                            1.39 |                                              1.15 |                                    0.65 |                       0.99 |
| Cutter          | 164 |   10.11 |        89.82 |         7.85 |       -2.32 |      2633.05 |                                 3.18 |                                            1.88 |                                              1.9  |                                    0.26 |                       0.66 |
| Changeup        |  86 |    5.3  |        83.82 |         2.35 |        9.46 |      1796.99 |                                 2.43 |                                            0.75 |                                              1.12 |                                    0.41 |                       0.78 |

*Stuff: whiff/swing vs su tipo*: puntos porcentuales de probabilidad de swing-and-miss que el modelo espera por las características físicas del pitch (sin ubicación, a nivel del mar), contra el promedio de la liga para ese mismo tipo. Solo es comparable entre pitches del mismo tipo. *Carreras evitadas/100 vs liga*: cuántas carreras evita cada 100 pitches de ese tipo contra el promedio de la liga del mismo tipo en el mismo entorno (nivel del mar o altitud extrema). *Costo de altitud*: carreras extra cada 100 pitches en altitud extrema; *antes del contacto* = swing, take, whiff y foul, incluida la mayor probabilidad de contacto (atribución de Shapley); *total* = incluye el resultado de la bola en juego (que mezcla calidad del contacto y parque).

## 3. Recomendaciones (2025)

- **Nivel:** Pitching+ 118.8 (percentil 99 de su temporada), intervalo de credibilidad de 80% [115.9, 121.6] (solo refleja el tamaño de muestra; no la incertidumbre de los modelos).
- **Riesgo de altitud intermedio (según el modelo):** en altitud extrema su arsenal permite +0.96 carreras más por cada 100 pitches que a nivel del mar; de eso, +0.37 ocurre antes del contacto (menos whiffs, más pelotas en juego) y el resto es el resultado de la bola en juego, que mezcla calidad del contacto y parque. La parte antes del contacto equivale a ≈+10.3 carreras por temporada de abridor (≈+3.4 de relevista). Es una sensibilidad implicada por el modelo (depende de qué lanza), no una resistencia a la altitud medida en sus propios juegos (`reports/sensibilidad_altitud.md`).
- **Pitch que más pierde antes del contacto en altitud:** Curveball (+0.65 carreras/100).
- **Mejor pitch a nivel del mar:** Cutter (+1.88 carreras/100 vs. el promedio de liga del mismo tipo).
- **En altitud extrema Cutter sigue siendo su mejor pitch** frente al promedio de su tipo en el mismo entorno.
