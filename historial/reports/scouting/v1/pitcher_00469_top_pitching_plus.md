# Reporte de scouting — pitcher_00469

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Slider          |         123 |    43.5 |               115.7 |                             102.1 |                  106.7 |
| Sinker          |         118 |    41.7 |                96   |                             101.2 |                  101   |
| Curveball       |          39 |    13.8 |               114.7 |                             102   |                  105.3 |
| Four-Seam       |           2 |     0.7 |                98.8 |                              99.4 |                  100.6 |
| Changeup        |           1 |     0.4 |               113.6 |                              97.8 |                  100.3 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2026 |         283 |          116.25 |

### Shrinkage bayesiano (`historial/reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |         162 |           -0.0594 |              -0.0161 |             -0.0137 |
| fastballs |         120 |           -0.0246 |               0.0042 |              0.0042 |
| offspeed  |           1 |            0      |              -0.0067 |             -0.0068 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Sinker          |                                     -2.31 |
| Slider          |                                     -1.86 |
| Curveball       |                                     -1.61 |

## 4. Percentil vs. liga

Pitching+ = **116.3** en la temporada **2026** -> percentil **100** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Sin un tipo dominante claro de alto/bajo riesgo de altitud (mayor uso: Slider, 43%) — el impacto de altitud depende del mix completo del arsenal, no de un solo pitch (ver tabla de arsenal).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~4.0 carreras/temporada perdidas en altitud extrema para un abridor — en el rango medio de la muestra analizada.
- Percentil 100 de la liga en 2026 — desempeño de elite (top 0%).