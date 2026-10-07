# Ficha de scouting — pitcher_00703

Índice oficial: **Pitching+ v3 neutral por altitud** (100 = valor de un pitch promedio de la liga (media ponderada por pitches de los pitchers con ≥100 en la temporada); 10 puntos = 1 desviación estándar de habilidad real τ). Altitud extrema = varios estadios de la categoría Extreme del dataset (estadios anónimos; el dataset no identifica estadios reales). Generado por `src/reports/scouting_report_v2.py`.

## 1. Resumen por temporada

|   year |   n_pitches |   pitching_plus |   pitching_plus_lo80 |   pitching_plus_hi80 |   percentil_temporada |   pitching_plus_con_altitud |   stuff_plus_v1 |   pct_no_altitude |   pct_medium_altitude |   pct_extreme_altitude |   costo_altitud_total_por_100 |   costo_altitud_antes_contacto_por_100 |
|-------:|------------:|----------------:|---------------------:|---------------------:|----------------------:|----------------------------:|----------------:|------------------:|----------------------:|-----------------------:|------------------------------:|---------------------------------------:|
|   2024 |         556 |          118.3  |               114.08 |               122.51 |                    99 |                      112.03 |          114.67 |                28 |                    16 |                     55 |                          1.35 |                                   0.57 |
|   2025 |          70 |           99.75 |                90.23 |               109.27 |                    58 |                       95.67 |          101.44 |                23 |                     0 |                     77 |                          1.34 |                                   0.4  |

`pitching_plus` = neutral por altitud con shrinkage (oficial) · `lo80`/`hi80` = intervalo de credibilidad empírico-bayesiano de 80% · `pitching_plus_con_altitud` = mismo índice sin quitar el entorno · `stuff_plus_v1` = Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita —ángulos de salida y llegada— y la categoría de altitud como features; no está neutralizado) · `pct_*` = % de sus pitches en cada categoría · `costo_altitud_*_por_100` = carreras extra por cada 100 pitches si los lanzara en altitud extrema en vez de a nivel del mar.

## 2. Arsenal (2024)

| AutoPitchType   |   n |   uso % |   velo (mph) |   IVB (pulg) |   HB (pulg) |   spin (rpm) |   stuff: whiff/swing vs su tipo (pp) |   carreras evitadas/100 vs liga (nivel del mar) |   carreras evitadas/100 vs liga (altitud extrema) |   costo altitud: antes del contacto/100 |   costo altitud: total/100 |
|:----------------|----:|--------:|-------------:|-------------:|------------:|-------------:|-------------------------------------:|------------------------------------------------:|--------------------------------------------------:|----------------------------------------:|---------------------------:|
| Four-Seam       | 323 |   58.09 |        93.03 |        16.9  |        7.99 |      2430.02 |                                 7    |                                            1.84 |                                              1.6  |                                    0.73 |                       1.69 |
| Changeup        | 112 |   20.14 |        84.27 |         4.51 |        6.59 |      1318.92 |                                 7.39 |                                            1.2  |                                              1.21 |                                    0.55 |                       1.14 |
| Slider          |  69 |   12.41 |        85.18 |         3.51 |       -3.44 |      2141.03 |                                 4.56 |                                           -0.06 |                                              0.19 |                                    0.07 |                       0.4  |
| Splitter        |  33 |    5.94 |        83.54 |         2.65 |        5.8  |      1161.99 |                                 4.14 |                                            0.82 |                                              0.85 |                                    0.3  |                       0.87 |

*Stuff: whiff/swing vs su tipo*: puntos porcentuales de probabilidad de swing-and-miss que el modelo espera por las características físicas del pitch (sin ubicación, a nivel del mar), contra el promedio de la liga para ese mismo tipo. Solo es comparable entre pitches del mismo tipo. *Carreras evitadas/100 vs liga*: cuántas carreras evita cada 100 pitches de ese tipo contra el promedio de la liga del mismo tipo en el mismo entorno (nivel del mar o altitud extrema). *Costo de altitud*: carreras extra cada 100 pitches en altitud extrema; *antes del contacto* = swing, take, whiff y foul, incluida la mayor probabilidad de contacto (atribución de Shapley); *total* = incluye el resultado de la bola en juego (que mezcla calidad del contacto y parque).

## 3. Recomendaciones (2024)

- **Nivel:** Pitching+ 118.3 (percentil 99 de su temporada), intervalo de credibilidad de 80% [114.1, 122.5] (solo refleja el tamaño de muestra; no la incertidumbre de los modelos).
- **Riesgo de altitud ALTO (cuartil superior de la liga) (según el modelo):** en altitud extrema su arsenal permite +1.35 carreras más por cada 100 pitches que a nivel del mar; de eso, +0.57 ocurre antes del contacto (menos whiffs, más pelotas en juego) y el resto es el resultado de la bola en juego, que mezcla calidad del contacto y parque. La parte antes del contacto equivale a ≈+15.8 carreras por temporada de abridor (≈+5.2 de relevista). Es una sensibilidad implicada por el modelo (depende de qué lanza), no una resistencia a la altitud medida en sus propios juegos (`reports/sensibilidad_altitud.md`).
- **Pitch que más pierde antes del contacto en altitud:** Four-Seam (+0.73 carreras/100).
- **Mejor pitch a nivel del mar:** Four-Seam (+1.84 carreras/100 vs. el promedio de liga del mismo tipo).
- **En altitud extrema Four-Seam sigue siendo su mejor pitch** frente al promedio de su tipo en el mismo entorno.
- **Contexto:** lanzó 55% de sus pitches en altitud extrema; el score con altitud (112.0) lo subestima — el neutral corrige ese entorno.
