# Ficha de scouting — pitcher_00465

Índice oficial: **Pitching+ v3 neutral por altitud** (100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). Altitud extrema = varios estadios de la categoría Extreme del dataset (estadios anónimos; el dataset no identifica estadios reales). Generado por `src/reports/scouting_report_v2.py`.

## 1. Resumen por temporada

|   year |   n_pitches |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |   percentil_temporada |   pitching_plus_con_altitud |   stuff_plus_v1 |   pct_no_altitude |   pct_medium_altitude |   pct_extreme_altitude |   costo_altitud_total_por_100 |   costo_altitud_antes_contacto_por_100 |
|-------:|------------:|----------------:|---------------------:|---------------------:|----------------------:|----------------------------:|----------------:|------------------:|----------------------:|-----------------------:|------------------------------:|---------------------------------------:|
|   2024 |         556 |          113.23 |               109.02 |               117.44 |                    97 |                      117.15 |          120.78 |                79 |                    16 |                      5 |                          1.41 |                                   0.54 |
|   2025 |         502 |          112.78 |               107.88 |               117.69 |                    97 |                      114.47 |          109.13 |                69 |                    21 |                     10 |                          1.3  |                                   0.49 |
|   2026 |         357 |          128.23 |               123.08 |               133.38 |                   100 |                      128.45 |          123.02 |                73 |                    18 |                     10 |                          1.4  |                                   0.61 |

`pitching_plus` = neutral por altitud con shrinkage (oficial) · `lo80`/`hi80` = intervalo de credibilidad empírico-bayesiano de 80% · `pitching_plus_con_altitud` = mismo índice sin quitar el entorno · `stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado) · `pct_*` = % de sus pitches en cada categoría · `costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar.

## 2. Arsenal (2026)

| AutoPitchType   |   n |   uso % |   velo (mph) |   IVB (pulg) |   HB (pulg) |   spin (rpm) |   stuff: whiff/swing vs su tipo (pp) |   carreras evitadas/100 vs liga (nivel del mar) |   carreras evitadas/100 vs liga (altitud extrema) |   costo altitud: antes del contacto/100 |   costo altitud: total/100 |
|:----------------|----:|--------:|-------------:|-------------:|------------:|-------------:|-------------------------------------:|------------------------------------------------:|--------------------------------------------------:|----------------------------------------:|---------------------------:|
| Four-Seam       | 202 |   56.58 |        95.55 |        17.33 |       10.86 |      2320.55 |                                 4    |                                            2.21 |                                              1.97 |                                    0.62 |                       1.69 |
| Curveball       | 103 |   28.85 |        81.52 |       -14.99 |      -11.42 |      2935.26 |                                22.91 |                                            3.27 |                                              2.92 |                                    0.8  |                       1.1  |
| Changeup        |  46 |   12.89 |        87.16 |         8.52 |       13.3  |      1725.09 |                                10.54 |                                            0.63 |                                              0.94 |                                    0.16 |                       0.84 |

*Stuff: whiff/swing vs su tipo*: puntos porcentuales de probabilidad de swing-and-miss que el modelo espera por las características físicas del pitch (sin ubicación, a nivel del mar), contra el promedio de la liga para ese mismo tipo. Solo es comparable entre pitches del mismo tipo. *Carreras evitadas/100 vs liga*: cuántas carreras evita cada 100 pitches de ese tipo contra el promedio de la liga del mismo tipo en el mismo entorno (nivel del mar o altitud extrema). *Costo de altitud*: carreras extra cada 100 pitches en altitud extrema; *antes del contacto* = swing, take, whiff y foul, incluida la mayor probabilidad de contacto (atribución de Shapley); *total* = incluye el resultado de la bola en juego (que mezcla calidad del contacto y parque).

## 3. Recomendaciones (2026)

- **Nivel:** Pitching+ 128.2 (percentil 100 de su temporada), intervalo de credibilidad de 80% [123.1, 133.4] (solo refleja el tamaño de muestra; no la incertidumbre de los modelos).
- **Riesgo de altitud ALTO (cuartil superior de la liga) (según el modelo):** en altitud extrema su arsenal permite +1.40 carreras más por cada 100 pitches que a nivel del mar; de eso, +0.61 ocurre antes del contacto (menos whiffs, más pelotas en juego) y el resto es el resultado de la bola en juego, que mezcla calidad del contacto y parque. La parte antes del contacto equivale a ≈+16.9 carreras por temporada de abridor (≈+5.5 de relevista). Es una sensibilidad implicada por el modelo (depende de qué lanza), no una resistencia a la altitud medida en sus propios juegos (`reports/sensibilidad_altitud.md`).
- **Pitch que más pierde antes del contacto en altitud:** Curveball (+0.80 carreras/100).
- **Mejor pitch a nivel del mar:** Curveball (+3.27 carreras/100 vs. el promedio de liga del mismo tipo).
- **En altitud extrema Curveball sigue siendo su mejor pitch** frente al promedio de su tipo en el mismo entorno.
