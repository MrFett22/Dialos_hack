# Reporte de scouting — pitcher_00489

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Sinker          |         224 |    72.7 |                99   |                              99.7 |                  100.3 |
| Slider          |          56 |    18.2 |               106.3 |                             100   |                  101.2 |
| Four-Seam       |          21 |     6.8 |               110.1 |                             101.5 |                  103.1 |
| Cutter          |           3 |     1   |                95.2 |                              97.2 |                  100.8 |
| Curveball       |           2 |     0.6 |                96.9 |                              95.1 |                   98.5 |
| Changeup        |           2 |     0.6 |               111.3 |                              96.4 |                  100   |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2026 |         308 |          104.87 |

### Shrinkage bayesiano (`reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |          58 |            0.0366 |              -0.0104 |             -0.0137 |
| fastballs |         248 |            0.0045 |               0.0047 |              0.0042 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Slider          |                                     -2.4  |
| Four-Seam       |                                     -1.12 |
| Sinker          |                                      1.47 |

## 4. Percentil vs. liga

Pitching+ = **104.9** en la temporada **2026** -> percentil **80** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Sinker** (73% de uso): riesgo de perdida de valor en altitud extrema (~+4.5 carreras/temporada proyectadas para un abridor, ver `reports/runs_translation.md`).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~4.3 carreras/temporada perdidas en altitud extrema para un abridor — en el extremo alto de la muestra de tipos de pitch analizados (rango ~3.5-4.5).
- Percentil 80 de la liga en 2026 — desempeño cercano al promedio de la liga.