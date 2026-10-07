# Stuff+ v1 — modelo baseline (LightGBM, solo features fisicas)

## Decision: target oficial de Stuff+ v1 = solo `is_whiff`

**Stuff+ v1 queda con un solo target: `is_whiff`** — "capacidad de generar swing-and-miss por calidad fisica pura". Mas adelante, cuando se construya Location+, se le suma `is_chase` como segunda pieza de este mismo indice de calidad-de-pitch.

`run_value` **se entrena y se evalua en esta corrida, pero solo como diagnostico/referencia comparativa — NO se guarda en el artefacto `.pkl` y NO es un target de Stuff+.** La razon es empirica, no de diseño a priori: el LightGBM de run_value con solo features fisicas queda ligeramente PEOR que el baseline ingenuo de "promedio de run_value por count" (ver seccion de apendice abajo) — evidencia directa de que run_value depende mucho mas del conteo/situacion que de la calidad fisica pura del pitch. Meter un target que el modelo no predice mejor que un lookup trivial dentro de "Stuff+" diluiria la interpretacion del indice (un componente que no aporta señal real, solo ruido) y mezclaria dos preguntas distintas: "¿que tan bueno es este pitch en si mismo?" (Stuff+) vs. "¿que tanto valor de carreras genera en ESTA situacion especifica?" (que necesita conteo+ubicacion, y le corresponde a xRunValue completo en el sub-modelo 3 / Pitching+).

Filas usadas (pitcher_holdout != excluded): 627,736
- Train: 503,140 pitches, 837 pitchers
- Test: 124,596 pitches, 208 pitchers

Features: RelSpeed, EffectiveVelo, ZoneSpeed, SpeedDrop, ZoneTime, SpinRate, SpinAxis, RelHeight, RelSide, Extension, VertBreak, InducedVertBreak, HorzBreak, VertRelAngle, HorzRelAngle, VertApprAngle, HorzApprAngle, altitude_level, magnus_residual_vert, magnus_residual_horz, PitcherThrows, BatterSide, AutoPitchType

Explicitamente NO incluidas: ubicacion (PlateLocHeight/Side, in/outside_strike_zone), conteo (Balls, Strikes, count), y features de arsenal (velo/ivb/hb_diff_vs_fb, tunnel_similarity, pitch_usage_pct, prev_pitch_type) — esas son para Location+/Pitching+, no para este Stuff+ puro.

## is_whiff (clasificacion) — target oficial de Stuff+ v1

| | AUC | log-loss | n | base rate |
|---|---:|---:|---:|---:|
| Modelo (test) | 0.7297 | 0.3236 | 124,596 | 0.1160 |
| Modelo (train) | 0.7734 | 0.3035 | 503,140 | 0.1127 |
| Baseline ingenuo (test) — promedio is_whiff por AutoPitchType | 0.5943 | 0.3527 | 124,596 | 0.1160 |

Mejora del modelo sobre el baseline (test): AUC +0.1354, log-loss -0.0291 (mas bajo es mejor).

Top 10 features por importancia (is_whiff):

| feature | importancia |
|---|---:|
| HorzRelAngle | 1493 |
| RelSide | 1447 |
| VertApprAngle | 1284 |
| VertRelAngle | 1278 |
| HorzApprAngle | 1130 |
| RelHeight | 1121 |
| SpeedDrop | 723 |
| Extension | 722 |
| SpinRate | 611 |
| magnus_residual_vert | 595 |

## Aporte de features de altitud/fisica

Ablation sobre is_whiff, mismo split pitcher_holdout, mismos hiperparametros: 3 subconjuntos de features. 'altitud' aqui es `altitude_level` (ordinal), la misma columna que usa Stuff+ v1 (ver seccion de features arriba) — el equivalente de `altitude_category` en este pipeline.

| version | features (n) | AUC test | log-loss test |
|---|---:|---:|---:|
| Completa (todas) | 23 | 0.7297 | 0.3236 |
| Sin altitud ni magnus | 20 | 0.7306 | 0.3233 |
| Solo altitud (sin magnus) | 21 | 0.7306 | 0.3233 |

**La version completa y 'solo altitud' dan resultados casi identicos** (diferencia de AUC <= 0.002) — senal de que `magnus_residual_vert/horz` es redundante para el PODER PREDICTIVO del modelo de is_whiff. Esto no cambia su rol en el analisis explicativo de H1 (Semana 2, reports/magnus_analysis.md y reports/run_value_validation.md): ahi el residual de Magnus sirvio para AISLAR el efecto de la densidad de aire sobre el movimiento, una pregunta distinta a "que tanta senal nueva le agrega a un modelo de whiff que ya tiene InducedVertBreak/HorzBreak crudos". Es esperable que aporte poco aqui: el residual es built a partir de columnas (RelSpeed, SpinRate, SpinAxis, InducedVertBreak, HorzBreak) que YA estan en el modelo — LightGBM puede aproximar esa misma relacion combinando esas columnas crudas en sus splits, sin necesitar el residual pre-calculado.

'Sin altitud ni magnus' no pierde poder predictivo notable frente a las otras dos versiones — la altitud no esta aportando señal incremental relevante para is_whiff especificamente (puede seguir siendo relevante para otros targets, como run_value).

**Nota metodologica importante para el contrafactual de altitud de Semana 5:** este resultado significa que el efecto de altitud sobre is_whiff esta completamente mediado por el movimiento observado (IVB/HB) — no hay efecto directo adicional. Un contrafactual que cambie altitude_category dejando el movimiento observado fijo va a medir ~0 por diseño, no porque la altitud no importe. Ver `reports/notas_metodologicas.md` para el diseño correcto (encadenar el modelo de Magnus antes de Stuff+).


## Apendice — diagnostico de `run_value` (NO es parte de Stuff+ v1)

Esta seccion existe solo para documentar la evidencia detras de la decision de arriba. El modelo de abajo NO se guarda en `stuff_plus_v1.pkl`.

Pitches con run_value nulo, excluidos de este target (bordes de datos de la reconstruccion de plate appearances, ver reports/run_value_validation.md): 2,989

| | RMSE | R^2 | n | std(y) |
|---|---:|---:|---:|---:|
| Modelo (test) | 0.2753 | 0.0032 | 123,939 | 0.2758 |
| Modelo (train) | 0.2709 | 0.0349 | 500,808 | 0.2758 |
| Baseline ingenuo (test) — promedio run_value por count | 0.2729 | 0.0205 | 123,939 | 0.2758 |

Mejora del modelo sobre el baseline (test): RMSE +0.0024 (mas bajo es mejor), R^2 -0.0173.

Top 10 features por importancia (run_value):

| feature | importancia |
|---|---:|
| Extension | 1039 |
| VertApprAngle | 1011 |
| VertRelAngle | 995 |
| SpinRate | 994 |
| SpeedDrop | 989 |
| RelHeight | 984 |
| HorzRelAngle | 936 |
| RelSide | 918 |
| HorzApprAngle | 847 |
| magnus_residual_horz | 737 |

## Hiperparametros

Defaults razonables, sin tuning: n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1

Modelo guardado en `src/models/artifacts/stuff_plus_v1.pkl` (pickle con is_whiff_model, feature_columns y metadata — sin run_value_model, ver decision arriba).
