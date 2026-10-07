# Ficha de scouting — pitcher_00529

Índice oficial: **Pitching+ v3 neutral por altitud** (100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). Altitud extrema = varios estadios de la categoría Extreme del dataset (estadios anónimos; el dataset no identifica estadios reales). Generado por `src/reports/scouting_report_v2.py`.

## 1. Resumen por temporada

|   year |   n_pitches |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |   percentil_temporada |   pitching_plus_con_altitud |   stuff_plus_v1 |   pct_no_altitude |   pct_medium_altitude |   pct_extreme_altitude |   costo_altitud_total_por_100 |   costo_altitud_antes_contacto_por_100 |
|-------:|------------:|----------------:|---------------------:|---------------------:|----------------------:|----------------------------:|----------------:|------------------:|----------------------:|-----------------------:|------------------------------:|---------------------------------------:|
|   2024 |          53 |           95.67 |                86.08 |               105.26 |                    38 |                       91.23 |          120.07 |                47 |                    30 |                     23 |                          2.14 |                                   1.01 |
|   2025 |         551 |           97.26 |                92.55 |               101.97 |                    47 |                       97.28 |          114.2  |                69 |                    23 |                      8 |                          2    |                                   0.87 |
|   2026 |         180 |           97.32 |                90.58 |               104.05 |                    41 |                       94.51 |          108.41 |                48 |                     0 |                     52 |                          1.75 |                                   0.75 |

`pitching_plus` = neutral por altitud con shrinkage (oficial) · `lo80`/`hi80` = intervalo de credibilidad empírico-bayesiano de 80% · `pitching_plus_con_altitud` = mismo índice sin quitar el entorno · `stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado) · `pct_*` = % de sus pitches en cada categoría · `costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar.

## 2. Arsenal (2025)

| AutoPitchType   |   n |   uso % |   velo (mph) |   IVB (pulg) |   HB (pulg) |   spin (rpm) |   stuff: whiff/swing vs su tipo (pp) |   carreras evitadas/100 vs liga (nivel del mar) |   carreras evitadas/100 vs liga (altitud extrema) |   costo altitud: antes del contacto/100 |   costo altitud: total/100 |
|:----------------|----:|--------:|-------------:|-------------:|------------:|-------------:|-------------------------------------:|------------------------------------------------:|--------------------------------------------------:|----------------------------------------:|---------------------------:|
| Changeup        | 342 |   62.07 |        77.1  |        15.93 |      -13.33 |      1959.2  |                                 6.42 |                                            0.42 |                                             -0.65 |                                    1    |                       2.21 |
| Four-Seam       | 131 |   23.77 |        85.32 |        18.8  |      -10.02 |      2251.94 |                                -2.51 |                                           -0.68 |                                             -1.17 |                                    0.83 |                       1.94 |
| Curveball       |  44 |    7.99 |        74.14 |        -3    |        4.31 |      2133.67 |                                -5.51 |                                           -0.5  |                                             -0.43 |                                    0.16 |                       0.68 |
| Sinker          |  18 |    3.27 |        85.74 |        14.39 |      -11.1  |      2194.72 |                                -1.99 |                                           -0.75 |                                             -1.38 |                                    0.38 |                       1.83 |

*Stuff: whiff/swing vs su tipo*: puntos porcentuales de probabilidad de swing-and-miss que el modelo espera por las características físicas del pitch (sin ubicación, a nivel del mar), contra el promedio de la liga para ese mismo tipo. Solo es comparable entre pitches del mismo tipo. *Carreras evitadas/100 vs liga*: cuántas carreras evita cada 100 pitches de ese tipo contra el promedio de la liga del mismo tipo en el mismo entorno (nivel del mar o altitud extrema). *Costo de altitud*: carreras extra cada 100 pitches en altitud extrema; *antes del contacto* = swing, take, whiff y foul, incluida la mayor probabilidad de contacto (atribución de Shapley); *total* = incluye el resultado de la bola en juego (que mezcla calidad del contacto y parque).

## 3. Recomendaciones (2025)

- **Nivel:** Pitching+ 97.3 (percentil 47 de su temporada), intervalo de credibilidad de 80% [92.5, 102.0] (solo refleja el tamaño de muestra; no la incertidumbre de los modelos).
- **Riesgo de altitud ALTO (cuartil superior de la liga) (según el modelo):** en altitud extrema su arsenal permite +2.00 carreras más por cada 100 pitches que a nivel del mar; de eso, +0.87 ocurre antes del contacto (menos whiffs, más pelotas en juego) y el resto es el resultado de la bola en juego, que mezcla calidad del contacto y parque. La parte antes del contacto equivale a ≈+23.9 carreras por temporada de abridor (≈+7.8 de relevista). Es una sensibilidad implicada por el modelo (depende de qué lanza), no una resistencia a la altitud medida en sus propios juegos (`reports/sensibilidad_altitud.md`).
- **Pitch que más pierde antes del contacto en altitud:** Changeup (+1.00 carreras/100).
- **Mejor pitch a nivel del mar:** Changeup (+0.42 carreras/100 vs. el promedio de liga del mismo tipo).
- **En altitud extrema el mejor pitch cambia a Curveball** (-0.43 vs. -0.65 de Changeup, contra el promedio de su tipo también en altitud): considerar darle más uso en casa.
