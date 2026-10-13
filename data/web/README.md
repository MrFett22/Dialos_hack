# Datos de la página (data/web/)

Generados por el pipeline (`src/reports/p2_report.py`, `src/reports/web_data.py` y `src/analysis/pitch_environment.py`); los usa `web/app.py`. Identificadores anónimos. No editar a mano.

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
| `pitching_env_values.json` | Δ(evento, conteo), valores terminales, calibración del costo de contacto y escalas: liga, No Altitude, Medium y Extreme |
| `pitching_general_vs_altitude.csv` | Pitching+ general vs. Extreme por tipo de pitch y por decil de P(BIP) |
| `pitching_relative_gain_by_whiff.csv` | ganancia de valor relativo en Extreme frente al típico del conteo, por quintil de P(whiff) |
| `optimal_pitch_profiles.csv` | perfil óptimo (top 1%) y mejor observado por tipo, objetivo y conteo |
| `optimal_pitch_comparison.csv` | cuánto cambia el óptimo entre objetivos (distancia, ruido bootstrap, rangos, top 1% compartido) |
| `optimal_pitch_location.csv` | mejor combinación física + ubicación por tipo, enfrentamiento, conteo y objetivo |
| `optimal_location_heatmap.csv` | Pitching+ por ubicación para un lanzamiento fijo de muy buen Stuff+ (general vs. Extreme) |
| `optimal_location_fixed_pitch.csv` | el lanzamiento fijo de cada tipo usado en el mapa de ubicaciones |
| `optimal_pitch_meta.json` | parámetros del análisis del óptimo (contextos, umbrales, rangos de los candidatos) |

La sensibilidad observada es exploratoria (confiabilidad ≈ 0.11). El contrafactual físico es explicativo: no mostró ventaja predictiva frente a la normalización simple por estadio anónimo.
