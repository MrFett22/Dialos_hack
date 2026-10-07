# Datos de la página (data/web/)

Generados por el pipeline (`src/reports/p2_report.py` y `src/reports/web_data.py`); los usa `web/app.py`. Identificadores anónimos. No editar a mano.

| archivo | contenido |
|---|---|
| `summary.json` | cifras que citan los textos de la página (efectos, consistencia, daño, pesos, validación, sensibilidad, estadios) |
| `simulator_scales.json` | escalas de lanzamiento (media, sd, percentiles) y ajuste por ambiente del simulador |
| `simulator_domain.json` | soporte de los datos por tipo de pitch (percentiles, Mahalanobis, celdas de ubicación) |
| `pitcher_pitch_types.csv` | uso, Stuff+ y Location+ por tipo (100 = promedio del tipo) y whiff/swing de cada pitcher del ranking |
| `anon_stadiums.csv` | ratios de drag y Magnus de cada estadio anónimo (IC 95%) |
| `pitcher_altitude_profiles.csv` | ficha completa de los pitchers elegibles (P2) |
| `pitcher_rankings.csv` | ranking AEPP (subconjunto de la ficha, para descargar) |
| `pitcher_sensitivity.csv` | sensibilidad observada — exploratoria, con advertencia en cada fila |
| `stuff_plus_physical.csv` | Stuff+ Physical por pitcher-temporada |
| `location_plus.csv` | Location+ por pitcher-temporada |
| `location_plus_heatmap.csv` | mapas de calor de Location+ (tipo × enfrentamiento × grupo de conteo) |
| `pitch_type_altitude_effects.csv` | efecto de Extreme y Medium por tipo de pitch |
| `contact_damage_by_environment.csv` | P(BIP), E[RV|BIP] y daño por ambiente y por estadio anónimo |
| `linear_weights_by_environment.csv` | valor RE24 de cada resultado del turno por ambiente |
| `characteristic_value_by_environment.csv` | valor de cada característica por ambiente (diseño cruzado) |

La sensibilidad observada es exploratoria (confiabilidad ≈ 0.11). El contrafactual físico es explicativo: no mostró ventaja predictiva frente a la normalización simple por estadio anónimo.
