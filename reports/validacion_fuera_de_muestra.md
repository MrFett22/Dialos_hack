# Validación fuera de muestra de los modelos finales

Generado por `src/validation/holdout_final.py`. Misma especificación e hiperparámetros que los artefactos finales (Pitching+ v3 y Stuff+ Physical). El dataset no identifica estadios: el *park holdout* deja fuera una categoría completa de ambiente (todos sus estadios anónimos), que es más exigente que dejar fuera un solo estadio.

**Objetivo:** el valor realizado de cada pitch, Δ(evento, conteo) con valores RE24 por conteo del conjunto de entrenamiento. **Baselines nulos:** media del Δ realizado por conteo ("xRunValue promedio por conteo") y por conteo × tipo de pitch.

## 1. Error y mejora sobre el modelo nulo (Pitching+)

| esquema | pitches de prueba | RMSE modelo | RMSE nulo (conteo) | mejora del MSE vs. conteo | vs. conteo × tipo |
|---|---:|---:|---:|---:|---:|
| Pitcher holdout (k=5 por pitcher) | 627,185 | 0.2350 | 0.2407 | +4.65% | +4.60% |
| Season holdout (2024–25 → 2026) | 129,594 | 0.2339 | 0.2398 | +4.83% | +4.83% |
| Park holdout: sin Extreme → Extreme | 180,794 | 0.2420 | 0.2471 | +4.10% | +4.06% |
| Park holdout: sin Medium → Medium | 154,048 | 0.2395 | 0.2449 | +4.33% | +4.30% |

El valor de un pitch individual es muy ruidoso (un mismo pitch puede terminar en strike, foul o jonrón), así que el RMSE por pitch baja poco en términos absolutos; lo relevante es que el modelo supera al nulo en todos los esquemas, también en pitchers, temporada y ambientes que no vio.

## 2. Calibración

| esquema | media predicha | media realizada | pendiente realizado ~ predicho | intercepto | ECE (deciles) | correlación por pitcher (mín. 100) | pendiente por pitcher |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pitcher holdout (k=5 por pitcher) | +0.0001 | +0.0000 | 0.98 | -0.0001 | 0.0013 | 0.55 (821) | 1.13 |
| Season holdout (2024–25 → 2026) | -0.0008 | -0.0022 | 1.00 | -0.0013 | 0.0021 | 0.52 (368) | 1.18 |
| Park holdout: sin Extreme → Extreme | +0.0084 | +0.0106 | 0.96 | +0.0026 | 0.0035 | 0.36 (476) | 1.04 |
| Park holdout: sin Medium → Medium | +0.0030 | +0.0062 | 0.98 | +0.0033 | 0.0035 | 0.39 (395) | 1.14 |

Pendiente 1 e intercepto 0 = calibración perfecta. Deciles del valor predicho (pitcher holdout):

| decil | predicho | realizado | n |
|---:|---:|---:|---:|
| 1 | -0.0770 | -0.0734 | 62,719 |
| 2 | -0.0454 | -0.0447 | 62,718 |
| 3 | -0.0336 | -0.0323 | 62,719 |
| 4 | -0.0239 | -0.0232 | 62,718 |
| 5 | -0.0133 | -0.0138 | 62,719 |
| 6 | -0.0003 | -0.0027 | 62,718 |
| 7 | +0.0149 | +0.0135 | 62,718 |
| 8 | +0.0293 | +0.0278 | 62,719 |
| 9 | +0.0434 | +0.0428 | 62,718 |
| 10 | +0.1064 | +0.1060 | 62,719 |

## 3. Sub-modelos del árbol (log-loss contra la tasa base; AUC)

| esquema | ¿swing? | resultado del take | whiff \| swing | foul \| contacto | resultado en juego |
|---|---|---|---|---|---|
| Pitcher holdout (k=5 por pitcher) | 0.450 vs. 0.691 (+35%); AUC 0.867 | 0.204 vs. 0.640 (+68%) | 0.461 vs. 0.554 (+17%); AUC 0.764 | 0.667 vs. 0.693 (+4%); AUC 0.624 | 0.958 vs. 0.971 (+1%) |
| Season holdout (2024–25 → 2026) | 0.446 vs. 0.691 (+35%); AUC 0.870 | 0.208 vs. 0.661 (+68%) | 0.460 vs. 0.555 (+17%); AUC 0.765 | 0.668 vs. 0.693 (+4%); AUC 0.622 | 0.958 vs. 0.972 (+1%) |
| Park holdout: sin Extreme → Extreme | 0.447 vs. 0.691 (+35%); AUC 0.870 | 0.211 vs. 0.633 (+67%) | 0.451 vs. 0.542 (+17%); AUC 0.764 | 0.668 vs. 0.692 (+3%); AUC 0.616 | 1.005 vs. 1.019 (+1%) |
| Park holdout: sin Medium → Medium | 0.447 vs. 0.691 (+35%); AUC 0.870 | 0.200 vs. 0.635 (+68%) | 0.457 vs. 0.553 (+17%); AUC 0.769 | 0.666 vs. 0.692 (+4%); AUC 0.626 | 0.994 vs. 1.008 (+1%) |

## 4. Stuff+ Physical: P(whiff | swing) sin ubicación

| esquema | swings de prueba | AUC modelo | AUC nulo (tasa por tipo) | log-loss | log-loss nulo |
|---|---:|---:|---:|---:|---:|
| Pitcher holdout (k=5 por pitcher) | 58,712 (por pliegue) | 0.651 | 0.601 | 0.5280 | 0.5412 |
| Season holdout (2024–25 → 2026) | 60,320 | 0.650 | 0.603 | 0.5295 | 0.5420 |
| Park holdout: sin Extreme → Extreme | 84,164 | 0.650 | 0.609 | 0.5173 | 0.5285 |
| Park holdout: sin Medium → Medium | 71,957 | 0.650 | 0.606 | 0.5274 | 0.5391 |

## 5. Lectura

- El modelo generaliza a **pitchers no vistos**, a una **temporada futura** y a **ambientes completos que no vio**: la mejora sobre el nulo y la calibración se mantienen en los tres esquemas.
- En el park holdout de Extreme el modelo **subestima un poco** el valor realizado (predicho +0.0084 vs. realizado +0.0106 carreras por pitch): sin haber visto Extreme, no sabe que ahí el contacto cuesta más. Es el mismo hallazgo de P2 §7.2 y la razón de ajustar por ambiente (índice oficial) y de calibrar el costo de contacto en el Pitching+ para altitud.
- La correlación por pitcher baja de 0.55 a 0.36 en el park holdout de Extreme, en parte porque cada pitcher tiene menos pitches de prueba (solo los de ese ambiente); la pendiente sigue cerca de 1.
- La validación de los índices por pitcher (estabilidad y predicción del año siguiente) está en `reports/validacion_temporal.md` y `reports/p2_pitcher_profiles_altitude.md`.
