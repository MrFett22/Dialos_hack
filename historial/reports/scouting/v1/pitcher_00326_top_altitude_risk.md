# Reporte de scouting — pitcher_00326

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Sinker          |        1081 |    63.9 |                96   |                             100   |                   99.8 |
| Changeup        |         297 |    17.6 |               100.8 |                              98.5 |                   99.7 |
| Slider          |         276 |    16.3 |               101.6 |                              99.4 |                  101.2 |
| Four-Seam       |          25 |     1.5 |               104.4 |                             101.1 |                  100.5 |
| Curveball       |           8 |     0.5 |               104.8 |                             101.8 |                  101.8 |
| Cutter          |           5 |     0.3 |               101.1 |                              99.6 |                  100.2 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2024 |         887 |          103.73 |
|   2025 |         235 |          105.84 |
|   2026 |         570 |           98.57 |

### Shrinkage bayesiano (`historial/reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |         284 |           -0.0236 |              -0.0162 |             -0.0137 |
| fastballs |        1111 |            0.022  |               0.0075 |              0.0042 |
| offspeed  |         297 |           -0.0289 |              -0.0127 |             -0.0068 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Slider          |                                     -2.13 |
| Sinker          |                                     -1.57 |
| Four-Seam       |                                     -1.06 |
| Changeup        |                                     -0.65 |

## 4. Percentil vs. liga

Pitching+ = **98.6** en la temporada **2026** -> percentil **30** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Sinker** (64% de uso): riesgo de perdida de valor en altitud extrema (~+4.5 carreras/temporada proyectadas para un abridor, ver `historial/reports/runs_translation.md`).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~4.3 carreras/temporada perdidas en altitud extrema para un abridor — en el extremo alto de la muestra de tipos de pitch analizados (rango ~3.5-4.5).
- Percentil 30 de la liga en 2026 — desempeño cercano al promedio de la liga.