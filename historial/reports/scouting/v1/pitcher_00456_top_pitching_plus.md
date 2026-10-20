# Reporte de scouting — pitcher_00456

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Four-Seam       |         153 |    69.5 |               105.4 |                             100.9 |                  102.2 |
| Curveball       |          63 |    28.6 |               110.3 |                             100.6 |                  106.9 |
| Cutter          |           4 |     1.8 |               100.8 |                             106.4 |                  104.7 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2026 |         220 |           115.5 |

### Shrinkage bayesiano (`historial/reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |          63 |           -0.0851 |              -0.0154 |             -0.0137 |
| fastballs |         157 |           -0.0061 |               0.0019 |              0.0042 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Curveball       |                                     -2.15 |
| Four-Seam       |                                     -0.79 |

## 4. Percentil vs. liga

Pitching+ = **115.5** en la temporada **2026** -> percentil **100** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Four-Seam** (70% de uso): de los tipos que MENOS pierden valor en altitud (`historial/reports/runs_translation.md`) — arsenal relativamente resiliente en ese frente (aunque sigue perdiendo valor neto, no gana).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~3.6 carreras/temporada perdidas en altitud extrema para un abridor — en el extremo bajo de la muestra (rango ~3.5-4.5) — comparativamente resiliente.
- Percentil 100 de la liga en 2026 — desempeño de elite (top 0%).