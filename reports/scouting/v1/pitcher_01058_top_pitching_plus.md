# Reporte de scouting — pitcher_01058

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Sinker          |         315 |    40.8 |                96.8 |                             101.7 |                  101.9 |
| Slider          |         158 |    20.5 |               109   |                             101.2 |                  105   |
| Curveball       |         157 |    20.3 |               107.9 |                             101.1 |                  103.9 |
| Changeup        |         135 |    17.5 |                97.6 |                             100.3 |                  101   |
| Four-Seam       |           4 |     0.5 |               115.2 |                             105.2 |                   98.4 |
| Sweeper         |           2 |     0.3 |               104   |                              96.4 |                  103.8 |
| Splitter        |           1 |     0.1 |               105.3 |                              94.3 |                  100.4 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2025 |         471 |          114.79 |
|   2026 |         301 |          113.17 |

### Shrinkage bayesiano (`reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |         315 |           -0.0259 |              -0.0081 |             -0.0137 |
| fastballs |         319 |           -0.0203 |               0.0025 |              0.0042 |
| offspeed  |         136 |            0.0074 |              -0.0071 |             -0.0068 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Curveball       |                                    -10.55 |
| Changeup        |                                     -1.89 |
| Slider          |                                     -1.51 |
| Sinker          |                                     -1.06 |

## 4. Percentil vs. liga

Pitching+ = **113.2** en la temporada **2026** -> percentil **99** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Sinker** (41% de uso): riesgo de perdida de valor en altitud extrema (~+4.5 carreras/temporada proyectadas para un abridor, ver `reports/runs_translation.md`).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~4.0 carreras/temporada perdidas en altitud extrema para un abridor — en el extremo alto de la muestra de tipos de pitch analizados (rango ~3.5-4.5).
- Percentil 99 de la liga en 2026 — desempeño de elite (top 1%).