# Reporte de scouting — pitcher_00096

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Sinker          |        1167 |    71   |                95.6 |                             100.1 |                  100.1 |
| Slider          |         162 |     9.9 |               106   |                             100.6 |                  103.5 |
| Curveball       |         137 |     8.3 |               108.2 |                             101.3 |                  103.4 |
| Sweeper         |          81 |     4.9 |               106.8 |                             100   |                  102.8 |
| Four-Seam       |          57 |     3.5 |                98.9 |                              99.8 |                   99.2 |
| Changeup        |          39 |     2.4 |               100.4 |                             100.2 |                   99.9 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2024 |         710 |          103.4  |
|   2025 |         536 |          106.58 |
|   2026 |         397 |          103.85 |

### Shrinkage bayesiano (`reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |         299 |           -0.0132 |              -0.0156 |             -0.0137 |
| fastballs |        1224 |           -0.0056 |              -0.0069 |              0.0042 |
| offspeed  |          39 |           -0.0799 |              -0.0085 |             -0.0068 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Changeup        |                                     -2.35 |
| Curveball       |                                     -2.31 |
| Four-Seam       |                                     -1.63 |
| Slider          |                                     -1.55 |
| Sinker          |                                     -1.5  |

## 4. Percentil vs. liga

Pitching+ = **103.8** en la temporada **2026** -> percentil **72** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Sinker** (71% de uso): riesgo de perdida de valor en altitud extrema (~+4.5 carreras/temporada proyectadas para un abridor, ver `reports/runs_translation.md`).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~4.3 carreras/temporada perdidas en altitud extrema para un abridor — en el extremo alto de la muestra de tipos de pitch analizados (rango ~3.5-4.5).
- Percentil 72 de la liga en 2026 — desempeño cercano al promedio de la liga.