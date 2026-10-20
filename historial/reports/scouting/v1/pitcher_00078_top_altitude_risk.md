# Reporte de scouting — pitcher_00078

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Sinker          |         249 |    70.3 |                94.9 |                             100.1 |                  100.2 |
| Slider          |          57 |    16.1 |               106.5 |                              99.6 |                  102.8 |
| Four-Seam       |          23 |     6.5 |                96.6 |                              99.3 |                  100.1 |
| Cutter          |          23 |     6.5 |               105.7 |                             100   |                  102.1 |
| Curveball       |           2 |     0.6 |                94.9 |                              92.6 |                  100.2 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2025 |         263 |          107.68 |
|   2026 |          91 |           95.8  |

⚠️ **2026: muestra chica (n=91<100)** — usar este Pitching+ con cautela, sujeto a ruido/shrinkage.

### Shrinkage bayesiano (`historial/reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |          59 |           -0.0498 |              -0.0131 |             -0.0137 |
| fastballs |         295 |           -0.0121 |               0.0003 |              0.0042 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Slider          |                                     -2.45 |
| Four-Seam       |                                     -1.32 |
| Sinker          |                                     -1.22 |
| Cutter          |                                     -0.58 |

## 4. Percentil vs. liga

Pitching+ = **107.7** en la temporada **2025** -> percentil **96** dentro de 474 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Sinker** (70% de uso): riesgo de perdida de valor en altitud extrema (~+4.5 carreras/temporada proyectadas para un abridor, ver `historial/reports/runs_translation.md`).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~4.3 carreras/temporada perdidas en altitud extrema para un abridor — en el extremo alto de la muestra de tipos de pitch analizados (rango ~3.5-4.5).
- Muestra chica (n=91 pitches en su ultima temporada, <100): tratar el Pitching+ agregado con cautela — es candidato a shrinkage sustancial hacia la media de su cluster (ver seccion de shrinkage bayesiano si esta disponible).