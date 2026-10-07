# Ficha de scouting — pitcher_00619

Índice oficial: **Pitching+ v3 neutral por altitud** (100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). Altitud extrema = varios estadios de la categoría Extreme del dataset (estadios anónimos; el dataset no identifica estadios reales). Generado por `src/reports/scouting_report_v2.py`.

## 1. Resumen por temporada

|   year |   n_pitches |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |   percentil_temporada |   pitching_plus_con_altitud |   stuff_plus_v1 |   pct_no_altitude |   pct_medium_altitude |   pct_extreme_altitude |   costo_altitud_total_por_100 |   costo_altitud_antes_contacto_por_100 |
|-------:|------------:|----------------:|---------------------:|---------------------:|----------------------:|----------------------------:|----------------:|------------------:|----------------------:|-----------------------:|------------------------------:|---------------------------------------:|
|   2024 |        1464 |          130.96 |               128.27 |               133.65 |                   100 |                      125.41 |          131.61 |                20 |                    21 |                     58 |                          1.19 |                                   0.57 |

`pitching_plus` = neutral por altitud con shrinkage (oficial) · `lo80`/`hi80` = intervalo de credibilidad empírico-bayesiano de 80% · `pitching_plus_con_altitud` = mismo índice sin quitar el entorno · `stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado) · `pct_*` = % de sus pitches en cada categoría · `costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar.

## 2. Arsenal (2024)

| AutoPitchType   |   n |   uso % |   velo (mph) |   IVB (pulg) |   HB (pulg) |   spin (rpm) |   stuff: whiff/swing vs su tipo (pp) |   carreras evitadas/100 vs liga (nivel del mar) |   carreras evitadas/100 vs liga (altitud extrema) |   costo altitud: antes del contacto/100 |   costo altitud: total/100 |
|:----------------|----:|--------:|-------------:|-------------:|------------:|-------------:|-------------------------------------:|------------------------------------------------:|--------------------------------------------------:|----------------------------------------:|---------------------------:|
| Four-Seam       | 668 |   45.63 |        94.37 |        15.08 |        8.07 |      2490.04 |                                 6.41 |                                            2.47 |                                              2.19 |                                    0.69 |                       1.73 |
| Slider          | 464 |   31.69 |        84.72 |         0.87 |       -8.9  |      2792.43 |                                 9.17 |                                            1.25 |                                              1.32 |                                    0.34 |                       0.58 |
| Curveball       | 230 |   15.71 |        81.26 |       -13.08 |      -10.49 |      2915.58 |                                10.91 |                                            2.34 |                                              2.03 |                                    0.82 |                       1.07 |
| Changeup        |  62 |    4.23 |        86.57 |         4.68 |       10.47 |      1725.79 |                                 9.08 |                                            0.56 |                                              1.2  |                                    0.14 |                       0.5  |
| Cutter          |  21 |    1.43 |        89.25 |         6.58 |       -2.21 |      2770.02 |                                 9.93 |                                            2.24 |                                              2.44 |                                    0.26 |                       0.48 |
| Sinker          |  18 |    1.23 |        93.4  |        11.04 |       13.63 |      2408.65 |                                 2.37 |                                            1.49 |                                              1.19 |                                    0.45 |                       1.5  |

*Stuff: whiff/swing vs su tipo*: puntos porcentuales de probabilidad de swing-and-miss que el modelo espera por las características físicas del pitch (sin ubicación, a nivel del mar), contra el promedio de la liga para ese mismo tipo. Solo es comparable entre pitches del mismo tipo. *Carreras evitadas/100 vs liga*: cuántas carreras evita cada 100 pitches de ese tipo contra el promedio de la liga del mismo tipo en el mismo entorno (nivel del mar o altitud extrema). *Costo de altitud*: carreras extra cada 100 pitches en altitud extrema; *antes del contacto* = swing, take, whiff y foul, incluida la mayor probabilidad de contacto (atribución de Shapley); *total* = incluye el resultado de la bola en juego (que mezcla calidad del contacto y parque).

## 3. Recomendaciones (2024)

- **Nivel:** Pitching+ 131.0 (percentil 100 de su temporada), intervalo de credibilidad de 80% [128.3, 133.6] (solo refleja el tamaño de muestra; no la incertidumbre de los modelos).
- **Riesgo de altitud intermedio (según el modelo):** en altitud extrema su arsenal permite +1.19 carreras más por cada 100 pitches que a nivel del mar; de eso, +0.57 ocurre antes del contacto (menos whiffs, más pelotas en juego) y el resto es el resultado de la bola en juego, que mezcla calidad del contacto y parque. La parte antes del contacto equivale a ≈+15.6 carreras por temporada de abridor (≈+5.1 de relevista). Es una sensibilidad implicada por el modelo (depende de qué lanza), no una resistencia a la altitud medida en sus propios juegos (`reports/sensibilidad_altitud.md`).
- **Pitch que más pierde antes del contacto en altitud:** Curveball (+0.82 carreras/100).
- **Mejor pitch a nivel del mar:** Four-Seam (+2.47 carreras/100 vs. el promedio de liga del mismo tipo).
- **En altitud extrema el mejor pitch cambia a Cutter** (+2.44 vs. +2.19 de Four-Seam, contra el promedio de su tipo también en altitud): considerar darle más uso en casa.
- **Contexto:** lanzó 58% de sus pitches en altitud extrema; el score con altitud (125.4) lo subestima — el neutral corrige ese entorno.
