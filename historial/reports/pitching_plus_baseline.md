# Pitching+ v1 — modelo combinado (LightGBM, Stuff+ + Location+ + arsenal)

Filas usadas (pitcher_holdout != excluded): 627,736

Features (33): RelSpeed, EffectiveVelo, ZoneSpeed, SpeedDrop, ZoneTime, SpinRate, SpinAxis, RelHeight, RelSide, Extension, VertBreak, InducedVertBreak, HorzBreak, VertRelAngle, HorzRelAngle, VertApprAngle, HorzApprAngle, altitude_level, magnus_residual_vert, magnus_residual_horz, PlateLocHeight, PlateLocSide, Balls, Strikes, velo_diff_vs_fb, ivb_diff_vs_fb, hb_diff_vs_fb, tunnel_similarity, pitch_usage_pct, PitcherThrows, BatterSide, AutoPitchType, prev_pitch_type

## Tabla comparativa: 8 targets x 3 esquemas de holdout (metricas de test)

### Targets binarios (AUC / log-loss)

| target | pitcher_holdout AUC | log-loss | season_holdout AUC | log-loss | altitude_holdout AUC | log-loss |
|---|---:|---:|---:|---:|---:|---:|
| is_whiff | 0.7575 | 0.3138 | 0.7582 | 0.3081 | 0.7581 | 0.2995 |
| is_chase | 0.9173 | 0.2942 | 0.9171 | 0.2945 | 0.9163 | 0.2926 |
| is_called_strike | 0.8840 | 0.2855 | 0.8810 | 0.3012 | 0.8820 | 0.2843 |
| is_weak_contact | 0.7677 | 0.2409 | 0.7709 | 0.2430 | 0.7694 | 0.2446 |
| is_groundball | 0.7812 | 0.2777 | 0.7856 | 0.2901 | 0.7803 | 0.2931 |
| is_hard_contact | 0.8105 | 0.0277 | 0.8110 | 0.0305 | 0.8125 | 0.0291 |
| is_out_in_pa | 0.6292 | 0.6211 | 0.6269 | 0.6187 | 0.6225 | 0.6389 |

### run_value (RMSE / R²)

| target | pitcher_holdout RMSE | R² | season_holdout RMSE | R² | altitude_holdout RMSE | R² |
|---|---:|---:|---:|---:|---:|---:|
| run_value | 0.2697 | 0.0435 | 0.2667 | 0.0428 | 0.2863 | 0.0398 |

### n de train/test por esquema (referencia)

| target | pitcher train/test | season train/test | altitude train/test |
|---|---|---|---|
| is_whiff | 503,140/124,596 | 497,596/130,140 | 446,932/180,804 |
| is_chase | 503,140/124,596 | 497,596/130,140 | 446,932/180,804 |
| is_called_strike | 503,140/124,596 | 497,596/130,140 | 446,932/180,804 |
| is_weak_contact | 496,836/123,164 | 490,536/129,464 | 441,188/178,812 |
| is_groundball | 503,140/124,596 | 497,596/130,140 | 446,932/180,804 |
| is_hard_contact | 496,835/123,164 | 490,535/129,464 | 441,188/178,811 |
| is_out_in_pa | 500,808/123,939 | 495,654/129,093 | 444,599/180,148 |
| run_value | 500,808/123,939 | 495,654/129,093 | 444,599/180,148 |

## Sensibilidad por esquema — lectura

- is_whiff: AUC estable entre esquemas (spread=0.0007) — generaliza de forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud).
- is_chase: AUC estable entre esquemas (spread=0.0010) — generaliza de forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud).
- is_called_strike: AUC estable entre esquemas (spread=0.0029) — generaliza de forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud).
- is_weak_contact: AUC estable entre esquemas (spread=0.0033) — generaliza de forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud).
- is_groundball: AUC estable entre esquemas (spread=0.0053) — generaliza de forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud).
- is_hard_contact: AUC estable entre esquemas (spread=0.0020) — generaliza de forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud).
- is_out_in_pa: AUC estable entre esquemas (spread=0.0067) — generaliza de forma similar sin importar que se mantenga fuera del train (pitcher, temporada o altitud).
- **run_value**: R² por esquema = pitcher_holdout=0.0435, season_holdout=0.0428, altitude_holdout=0.0398 — comparar contra el baseline por count de Stuff+ v1 (R²=0.0205) en historial/reports/stuff_plus_baseline.md; aqui con ubicacion+conteo+arsenal completos se espera una mejora real, a diferencia del Stuff+ puro (que quedo peor que ese baseline).

## Comparacion contra sub-modelos individuales (pitcher_holdout test)

| target | Pitching+ (combinado) AUC | Sub-modelo individual | AUC individual | Δ AUC |
|---|---:|---|---:|---:|
| is_whiff | 0.7575 | Stuff+ v1 | 0.7297 | +0.0279 |
| is_called_strike | 0.8840 | Location+ v1 | 0.8468 | +0.0371 |
| is_chase | 0.9173 | Location+ v1 | 0.7993 | +0.1180 |

**Interpretacion:** un Δ AUC grande y positivo indica que agregar el resto del arsenal de features (fuera del dominio del sub-modelo individual) SI aporta señal incremental real — no son redundantes. Un Δ cercano a 0 indica que el sub-modelo individual ya capturaba toda la señal disponible en su propio dominio, y las features de los otros sub-modelos no agregan nada nuevo para ESE target especifico (coherente con la logica de separar Stuff+/Location+/Pitching+ en primer lugar: cada uno deberia ser mejor en su propio dominio de target).

Modelo guardado en `src/models/artifacts/pitching_plus_v1.pkl` — 8 modelos (uno por target), todos entrenados sobre `pitcher_holdout` (el esquema oficial, igual criterio que Stuff+/Location+ v1). Las metricas de season_holdout/altitude_holdout de arriba son solo de validacion/diagnostico, esos modelos no se guardan por separado.
