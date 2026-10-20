# Corrección del golpeado (HBP): antes vs. después

Generado por `src/validation/hbp_before_after.py`. *Antes* = pipeline del commit `db1eed6`; *después* = pipeline regenerado desde cero con `is_hit_by_pitch` reconstruida desde PitchCall (`src/features/targets.py`).

## 1. Target

| | antes | después |
|---|---:|---:|
| pitches con is_hit_by_pitch = 1 | 0 | 2,276 |
| run_value medio del golpeado | +0.0038 | +0.3976 |
| golpeados con run_value = 0 | 2,228 | 32 |
| run_value medio de la base por bolas | +0.3815 | +0.3784 |
| pitches con run_value distinto | — | 159,097 |
| pitches sin run_value (turno no resoluble) | 3,168 | 3,092 |

El run_value cambia también fuera de los golpeados: el golpeado ahora cierra el turno y pone un corredor en primera, así que cambian los estados base-out del resto de la media entrada y la propia matriz RE24.

## 2. Valores terminales (carreras)

| evento | antes | después | cambio |
|---|---:|---:|---:|
| walk | +0.3813 | +0.3783 | -0.0030 |
| strikeout | -0.3283 | -0.3344 | -0.0061 |
| hbp | +0.0039 | +0.3971 | +0.3932 |
| out | -0.3011 | -0.3075 | -0.0063 |
| single | +0.5153 | +0.5118 | -0.0035 |
| double | +0.8323 | +0.8360 | +0.0037 |
| triple | +1.0369 | +1.0410 | +0.0041 |
| home_run | +1.4788 | +1.4719 | -0.0069 |

## 3. V(b, s): carreras esperadas del turno desde cada conteo

| conteo | antes | después | cambio |
|---|---:|---:|---:|
| 0-0 | +0.0142 | +0.0147 | +0.0005 |
| 0-1 | -0.0348 | -0.0348 | -0.0000 |
| 0-2 | -0.1058 | -0.1067 | -0.0009 |
| 1-0 | +0.0586 | +0.0579 | -0.0006 |
| 1-1 | -0.0008 | -0.0015 | -0.0007 |
| 1-2 | -0.0811 | -0.0823 | -0.0012 |
| 2-0 | +0.1299 | +0.1284 | -0.0015 |
| 2-1 | +0.0570 | +0.0555 | -0.0015 |
| 2-2 | -0.0324 | -0.0344 | -0.0019 |
| 3-0 | +0.2497 | +0.2476 | -0.0021 |
| 3-1 | +0.1738 | +0.1712 | -0.0025 |
| 3-2 | +0.0783 | +0.0754 | -0.0029 |

## 4. Δ(golpeado, conteo) y conteos afectados

| conteo | Δ(hbp) antes | Δ(hbp) después | Δ(bola) después |
|---|---:|---:|---:|
| 0-0 | -0.0103 | +0.3824 | +0.0432 |
| 0-1 | +0.0386 | +0.4319 | +0.0333 |
| 0-2 | +0.1097 | +0.5038 | +0.0244 |
| 1-0 | -0.0547 | +0.3392 | +0.0705 |
| 1-1 | +0.0047 | +0.3986 | +0.0570 |
| 1-2 | +0.0849 | +0.4794 | +0.0479 |
| 2-0 | -0.1261 | +0.2687 | +0.1192 |
| 2-1 | -0.0531 | +0.3416 | +0.1157 |
| 2-2 | +0.0363 | +0.4315 | +0.1097 |
| 3-0 | -0.2459 | +0.1495 | +0.1307 |
| 3-1 | -0.1699 | +0.2259 | +0.2071 |
| 3-2 | -0.0744 | +0.3218 | +0.3029 |

Conteos con Δ(hbp) < 0 (el golpeado "favorece" al pitcher): **7 antes → 0 después**. Cambio máximo en el Δ de los demás eventos: 0.0074 carreras.

## 5. Pitching+ oficial

- Pitcher-temporadas comparadas: 1,628 (1,278 con ≥100 pitches).
- Correlación antes/después (≥100): Pearson 0.9880, Spearman 0.9865.
- Cambio en puntos (≥100): media -0.07, sd 1.35, |cambio| máximo 5.90; 137 pitcher-temporadas se mueven ≥2 puntos.
- Correlación entre el cambio y la tasa de golpeados del pitcher: -0.564 (los que más golpean bajan: antes el golpeado se les contaba casi como un strike).

### Pitchers más afectados (≥100 pitches)

| pitcher_anon_id   |   year |   n_pitches |   hbp_por_100 |   pitching_plus_antes |   pitching_plus_despues |   cambio |   rank_antes |   rank_despues |
|:------------------|-------:|------------:|--------------:|----------------------:|------------------------:|---------:|-------------:|---------------:|
| pitcher_00384     |   2025 |         134 |          1.49 |                 96.83 |                   90.93 |    -5.9  |          336 |            460 |
| pitcher_01007     |   2025 |         160 |          0.62 |                 85.31 |                   79.47 |    -5.84 |          534 |            563 |
| pitcher_00386     |   2025 |         141 |          0.71 |                105    |                   99.46 |    -5.54 |          121 |            243 |
| pitcher_00840     |   2025 |         548 |          0.55 |                104.23 |                   98.75 |    -5.48 |          136 |            267 |
| pitcher_00774     |   2025 |         501 |          0.6  |                 97.22 |                   91.93 |    -5.29 |          322 |            441 |
| pitcher_00916     |   2025 |         106 |          0    |                 94.4  |                   89.2  |    -5.2  |          390 |            490 |
| pitcher_00901     |   2025 |         349 |          0.86 |                 96.47 |                   91.27 |    -5.2  |          341 |            454 |
| pitcher_00632     |   2026 |         169 |          1.18 |                108.92 |                  103.88 |    -5.04 |           51 |            131 |
| pitcher_00786     |   2026 |         108 |          0    |                 87.65 |                   82.71 |    -4.93 |          464 |            492 |
| pitcher_00168     |   2024 |         415 |          1.2  |                 98.73 |                   93.8  |    -4.93 |          270 |            375 |
| pitcher_00577     |   2025 |         432 |          1.62 |                 97.28 |                   92.37 |    -4.91 |          320 |            437 |
| pitcher_00002     |   2025 |         196 |          0.51 |                104.27 |                   99.48 |    -4.79 |          135 |            242 |
