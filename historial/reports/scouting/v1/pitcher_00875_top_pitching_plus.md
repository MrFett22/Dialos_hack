# Reporte de scouting — pitcher_00875

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Cutter          |         212 |    34.1 |               103.3 |                             102   |                  103.4 |
| Four-Seam       |         203 |    32.7 |                99.3 |                             100.9 |                  101.4 |
| Slider          |         114 |    18.4 |               104   |                             100.2 |                  102.9 |
| Sinker          |          63 |    10.1 |                93.8 |                             101.2 |                  100.4 |
| Splitter        |          16 |     2.6 |               105.9 |                              98.2 |                  102.8 |
| Changeup        |          10 |     1.6 |               104.5 |                              99.7 |                  103.8 |
| Curveball       |           3 |     0.5 |                92.2 |                             101.3 |                  100.9 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2024 |         272 |          108.98 |
|   2026 |         349 |          114.97 |

### Shrinkage bayesiano (`historial/reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |         117 |           -0.0285 |              -0.0164 |             -0.0137 |
| fastballs |         478 |           -0.024  |              -0.0026 |              0.0042 |
| offspeed  |          26 |            0.0167 |              -0.0053 |             -0.0068 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Slider          |                                     -2.38 |
| Sinker          |                                     -1.76 |
| Cutter          |                                     -1.6  |
| Splitter        |                                     -1.47 |
| Changeup        |                                     -1.27 |
| Four-Seam       |                                     -1.06 |

## 4. Percentil vs. liga

Pitching+ = **115.0** en la temporada **2026** -> percentil **99** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Cutter** (34% de uso): riesgo de perdida de valor en altitud extrema (~+3.9 carreras/temporada proyectadas para un abridor, ver `historial/reports/runs_translation.md`).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~3.8 carreras/temporada perdidas en altitud extrema para un abridor — en el rango medio de la muestra analizada.
- Percentil 99 de la liga en 2026 — desempeño de elite (top 1%).