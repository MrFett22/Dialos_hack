# Reporte de scouting — pitcher_00107

## 1. Arsenal completo

| AutoPitchType   |   n_pitches |   uso % |   Stuff+ (is_whiff) |   Location+ (called_strike+chase) |   Pitching+ individual |
|:----------------|------------:|--------:|--------------------:|----------------------------------:|-----------------------:|
| Sinker          |         208 |    76.8 |                99.5 |                             100.3 |                  101.3 |
| Slider          |          58 |    21.4 |               110.8 |                              99.5 |                  102.7 |
| Curveball       |           4 |     1.5 |               111.8 |                             102.8 |                  109.5 |
| Four-Seam       |           1 |     0.4 |                95.8 |                              96.3 |                   89.2 |

*Stuff+/Location+ aqui son los componentes is_whiff / (is_called_strike+is_chase) del modelo Pitching+ combinado, no los modelos standalone — ver docstring de `src/reports/scouting_report.py`.*

## 2. Pitching+ agregado (por temporada)

|   year |   n_pitches |   pitching_plus |
|-------:|------------:|----------------:|
|   2026 |         271 |          107.91 |

### Shrinkage bayesiano (`reports/bayesian_hierarchical.md`)

Este pitcher SI fue parte de la muestra de 180k del modelo bayesiano jerarquico para el/los cluster(es) listados abajo — se compara su promedio crudo de run_value (sin pooling) contra el valor 'encogido' (posterior de `alpha_pc`, que combina su propia señal con la media de su cluster segun cuanta muestra tiene):

| cluster   |   n_pitches |   run_value_crudo |   run_value_encogido |   media_del_cluster |
|:----------|------------:|------------------:|---------------------:|--------------------:|
| breaking  |          62 |           -0.0066 |              -0.0132 |             -0.0137 |
| fastballs |         209 |           -0.0236 |               0.0017 |              0.0042 |

## 3. Delta de altitud proyectado por tipo de pitch

Usando `dashboard/scoring.compute_altitude_scenarios` (mismo mecanismo del dashboard), con el promedio REAL de RelSpeed/movimiento/spin de ESTE pitcher para cada tipo, BatterSide fijo en 'Right' (neutral):

| AutoPitchType   |   delta Pitching+ (Extreme - No Altitude) |
|:----------------|------------------------------------------:|
| Sinker          |                                     -2.32 |
| Slider          |                                     -1.95 |

## 4. Percentil vs. liga

Pitching+ = **107.9** en la temporada **2026** -> percentil **94** dentro de 368 pitcher-temporadas confiables (n_pitches>=100) de ese mismo año.

## 5. Recomendacion

- Arsenal dominado por **Sinker** (77% de uso): riesgo de perdida de valor en altitud extrema (~+4.5 carreras/temporada proyectadas para un abridor, ver `reports/runs_translation.md`).
- Proyeccion agregada (ponderada por el uso real de su arsenal): ~4.3 carreras/temporada perdidas en altitud extrema para un abridor — en el extremo alto de la muestra de tipos de pitch analizados (rango ~3.5-4.5).
- Percentil 94 de la liga en 2026 — desempeño de elite (top 6%).