# Reporte de scouting — pitcher_00402

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Slider          |          46 |    45.5 |               107.4 |                             103.5 |                  105.9 |
| Four-Seam       |          34 |    33.7 |               101.5 |                              98   |                  101.3 |
| Sinker          |          17 |    16.8 |                95   |                             100.1 |                  100.3 |
| Curveball       |           3 |     3   |               111.2 |                              99.4 |                  105.1 |
| Changeup        |           1 |     1   |               124.3 |                             103.7 |                  102.9 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2026 |         101 |          113.65 |

### Shrinkage bayesiano (`historial/reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |          49 |           -0.0875 |              -0.0184 |             -0.0137 |
| fastballs |          51 |            0.0084 |               0.0046 |              0.0042 |
| offspeed  |           1 |            0      |              -0.0068 |             -0.0068 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Sinker          |                                     -2.18 |
| Slider          |                                     -1.11 |
| Four-Seam       |                                     -0.97 |

## 4. Percentil vs. liga

Pitching+ = **113.6** en la temporada **2026** -> percentil **99** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Sin un tipo dominante claro de alto/bajo riesgo de altitud (mayor uso: Slider, 46%) — el impacto de altitud depende del mix completo del arsenal, no de un solo pitch (ver tabla de arsenal).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~3.7 carreras/temporada perdidas en altitud extrema para un abridor — en el rango medio de la muestra analizada.
- Percentil 99 de la liga en 2026 — desempeño de elite (top 1%).