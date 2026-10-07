# Stuff+ v2 — features corregidas (mano, ubicación escondida, redundancias, arsenal)

Generado por `src/models/stuff_plus_v2.py`. Benchmark de `src/validation/benchmark.py` (pares 2024→2025 y 2025→2026, bootstrap pareado por pitcher, 2,000 réplicas). Candidata oficial declarada antes de ver resultados: **v2d_arsenal**.

## Cambios de features (escalera de ablation)

| variante | features | cambio |
|---|---:|---|
| v1 | 23 | referencia |
| v2a_mano | 22 | v2a: espejo de zurdos + platoon |
| v2b_ubicacion | 20 | v2b: + VAA/HAA ajustados por ubicación, sin ángulos de salida |
| v2c_limpio | 13 | v2c: − redundancias, altitud y Magnus |
| v2d_arsenal | 16 | v2d: + diferencias contra la recta (candidata oficial) |

Features finales de v2d_arsenal: RelSpeed, SpeedDrop, SpinRate, SpinAxis, RelHeight, RelSide, Extension, InducedVertBreak, HorzBreak, vaa_loc_adj, haa_loc_adj, velo_diff_vs_fb, ivb_diff_vs_fb, hb_diff_vs_fb, same_side, AutoPitchType

- **Espejo de zurdos:** se invierte el signo de `HorzBreak`, `RelSide`, `HorzRelAngle`, `HorzApprAngle`, `magnus_residual_horz`, `hb_diff_vs_fb`, `PlateLocSide`; `SpinAxis` → 360 − SpinAxis. Mano Undefined/nula se infiere de `RelSide`. `PitcherThrows`/`BatterSide` se reemplazan por `same_side` (mismo lado / lado contrario).
- **Ajuste por ubicación:** `vaa_loc_adj` = VertApprAngle − (a + b·PlateLocHeight) y `haa_loc_adj` = HorzApprAngle − (a + b·PlateLocSide), con la recta ajustada por tipo de pitch. Es el ángulo de llegada "a ubicación constante": cuánto más plano o inclinado llega el pitch que uno promedio de su tipo en el mismo punto.

Pendiente global VAA ~ altura: b=1.396 °/pie; HAA ~ lado: b=1.251 °/pie.

## Métricas pitch a pitch (OOF, `is_whiff`)

| variante | AUC | log-loss | Brier | ECE |
|---|---:|---:|---:|---:|
| stuff_plus_v1 | 0.7315 | 0.3184 | 0.0932 | 0.0046 |
| v2a_mano | 0.7360 | 0.3168 | 0.0929 | 0.0043 |
| v2b_ubicacion | 0.6336 | 0.3425 | 0.0983 | 0.0019 |
| v2c_limpio | 0.6319 | 0.3428 | 0.0984 | 0.0020 |
| v2d_arsenal | 0.6396 | 0.3414 | 0.0981 | 0.0021 |
| diag_v1_swing | 0.7410 | 0.4757 | 0.1529 | 0.0032 |
| diag_v2d_swing | 0.6509 | 0.5269 | 0.1737 | 0.0032 |

Los diagnósticos `diag_*_swing` se evalúan solo sobre swings (su universo), así que su AUC no es comparable 1 a 1 con el resto.

Bajar el AUC pitch a pitch es **esperado** al quitar información de ubicación: el v1 predecía mejor en parte porque veía dónde cruzaba el pitch. Lo que importa es el benchmark de abajo.

## Benchmark año-contra-año — mínimo 100 pitches (437 pares, 322 pitchers)

| variante | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |
|---|---|---|---|---|
| stuff_plus_v1 | 0.668 [0.594, 0.732] | 0.182 [0.098, 0.267] | 0.243 [0.155, 0.330] | 0.362 [0.277, 0.443] |
| v2a_mano | 0.661 [0.559, 0.746] | 0.159 [0.077, 0.245] | 0.225 [0.130, 0.314] | 0.345 [0.261, 0.429] |
| v2b_ubicacion | 0.734 [0.659, 0.798] | 0.147 [0.065, 0.231] | 0.157 [0.061, 0.249] | 0.285 [0.196, 0.379] |
| v2c_limpio | 0.724 [0.629, 0.802] | 0.139 [0.055, 0.222] | 0.147 [0.052, 0.240] | 0.269 [0.177, 0.364] |
| v2d_arsenal | 0.693 [0.599, 0.777] | 0.160 [0.077, 0.246] | 0.137 [0.036, 0.231] | 0.310 [0.219, 0.403] |
| diag_v1_swing | 0.617 [0.521, 0.699] | 0.146 [0.051, 0.237] | 0.180 [0.074, 0.274] | 0.379 [0.292, 0.464] |
| diag_v2d_swing | 0.718 [0.663, 0.774] | 0.190 [0.105, 0.275] | 0.166 [0.066, 0.261] | 0.314 [0.228, 0.399] |
| *Whiff% observado (whiffs/pitch)* | 0.487 [0.388, 0.576] | 0.193 [0.114, 0.268] | 0.290 [0.197, 0.375] | 0.487 [0.388, 0.576] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.349 [0.260, 0.429] | 0.202 [0.112, 0.283] | 0.284 [0.197, 0.362] | 0.357 [0.269, 0.442] |
| *K-BB% observado* | 0.263 [0.179, 0.342] | 0.181 [0.096, 0.264] | 0.263 [0.179, 0.342] | 0.296 [0.212, 0.376] |

**Delta contra v1 (bootstrap pareado; P = probabilidad de que la variante sea mejor):**

| variante | estabilidad | carreras evitadas t+1 | whiff% t+1 |
|---|---|---|---|
| v2a_mano | -0.007 [-0.056, +0.037] · P=44% | -0.023 [-0.050, +0.006] · P=6% | -0.017 [-0.046, +0.012] · P=13% |
| v2b_ubicacion | +0.066 [+0.019, +0.115] · P=100% | -0.035 [-0.089, +0.018] · P=10% | -0.076 [-0.137, -0.015] · P=1% |
| v2c_limpio | +0.056 [+0.002, +0.116] · P=98% | -0.043 [-0.099, +0.012] · P=6% | -0.092 [-0.154, -0.029] · P=0% |
| v2d_arsenal | +0.024 [-0.039, +0.093] · P=78% | -0.022 [-0.086, +0.047] · P=27% | -0.051 [-0.123, +0.023] · P=8% |
| diag_v1_swing | -0.052 [-0.110, +0.007] · P=5% | -0.036 [-0.102, +0.032] · P=14% | +0.017 [-0.058, +0.098] · P=66% |
| diag_v2d_swing | +0.050 [-0.002, +0.105] · P=97% | +0.008 [-0.061, +0.077] · P=59% | -0.048 [-0.119, +0.023] · P=8% |

**Delta contra el mejor baseline observado:**

| variante | carreras evitadas t+1 | whiff% t+1 |
|---|---|---|
| stuff_plus_v1 | -0.020 [-0.123, +0.091] · P=35% (vs obs_csw) | -0.125 [-0.210, -0.036] · P=0% (vs obs_whiff_rate) |
| v2a_mano | -0.042 [-0.150, +0.071] · P=22% (vs obs_csw) | -0.142 [-0.229, -0.050] · P=0% (vs obs_whiff_rate) |
| v2b_ubicacion | -0.054 [-0.158, +0.056] · P=15% (vs obs_csw) | -0.201 [-0.304, -0.097] · P=0% (vs obs_whiff_rate) |
| v2c_limpio | -0.063 [-0.169, +0.049] · P=13% (vs obs_csw) | -0.217 [-0.319, -0.115] · P=0% (vs obs_whiff_rate) |
| v2d_arsenal | -0.041 [-0.153, +0.074] · P=24% (vs obs_csw) | -0.176 [-0.281, -0.075] · P=0% (vs obs_whiff_rate) |
| diag_v1_swing | -0.056 [-0.164, +0.057] · P=16% (vs obs_csw) | -0.107 [-0.186, -0.025] · P=1% (vs obs_whiff_rate) |
| diag_v2d_swing | -0.012 [-0.120, +0.098] · P=42% (vs obs_csw) | -0.172 [-0.273, -0.074] · P=0% (vs obs_whiff_rate) |

## Benchmark año-contra-año — mínimo 300 pitches (245 pares, 193 pitchers)

| variante | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |
|---|---|---|---|---|
| stuff_plus_v1 | 0.711 [0.628, 0.779] | 0.040 [-0.074, 0.166] | 0.172 [0.047, 0.291] | 0.327 [0.209, 0.442] |
| v2a_mano | 0.708 [0.630, 0.775] | 0.041 [-0.080, 0.167] | 0.147 [0.012, 0.277] | 0.311 [0.192, 0.428] |
| v2b_ubicacion | 0.748 [0.676, 0.810] | 0.051 [-0.073, 0.183] | 0.096 [-0.040, 0.220] | 0.283 [0.162, 0.402] |
| v2c_limpio | 0.747 [0.676, 0.809] | 0.039 [-0.080, 0.164] | 0.080 [-0.058, 0.199] | 0.269 [0.143, 0.390] |
| v2d_arsenal | 0.735 [0.669, 0.791] | 0.065 [-0.063, 0.194] | 0.076 [-0.060, 0.209] | 0.327 [0.199, 0.441] |
| diag_v1_swing | 0.666 [0.572, 0.736] | 0.039 [-0.092, 0.176] | 0.089 [-0.058, 0.230] | 0.350 [0.206, 0.469] |
| diag_v2d_swing | 0.739 [0.677, 0.792] | 0.095 [-0.039, 0.224] | 0.121 [-0.015, 0.252] | 0.323 [0.198, 0.435] |
| *Whiff% observado (whiffs/pitch)* | 0.537 [0.402, 0.648] | 0.111 [-0.008, 0.226] | 0.268 [0.132, 0.387] | 0.537 [0.402, 0.648] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.293 [0.166, 0.404] | 0.106 [-0.017, 0.224] | 0.229 [0.104, 0.345] | 0.367 [0.237, 0.479] |
| *K-BB% observado* | 0.323 [0.205, 0.430] | 0.193 [0.064, 0.315] | 0.323 [0.205, 0.430] | 0.377 [0.261, 0.478] |

**Delta contra v1 (bootstrap pareado; P = probabilidad de que la variante sea mejor):**

| variante | estabilidad | carreras evitadas t+1 | whiff% t+1 |
|---|---|---|---|
| v2a_mano | -0.003 [-0.042, +0.035] · P=45% | +0.001 [-0.050, +0.051] · P=51% | -0.015 [-0.060, +0.027] · P=26% |
| v2b_ubicacion | +0.037 [-0.029, +0.103] · P=87% | +0.011 [-0.068, +0.091] · P=60% | -0.043 [-0.120, +0.031] · P=12% |
| v2c_limpio | +0.037 [-0.031, +0.105] · P=86% | -0.001 [-0.080, +0.083] · P=49% | -0.058 [-0.139, +0.022] · P=7% |
| v2d_arsenal | +0.024 [-0.049, +0.103] · P=75% | +0.025 [-0.070, +0.126] · P=69% | +0.001 [-0.097, +0.102] · P=50% |
| diag_v1_swing | -0.045 [-0.127, +0.030] · P=12% | -0.001 [-0.104, +0.105] · P=50% | +0.023 [-0.091, +0.152] · P=64% |
| diag_v2d_swing | +0.029 [-0.045, +0.107] · P=78% | +0.055 [-0.047, +0.163] · P=83% | -0.004 [-0.109, +0.104] · P=47% |

**Delta contra el mejor baseline observado:**

| variante | carreras evitadas t+1 | whiff% t+1 |
|---|---|---|
| stuff_plus_v1 | -0.153 [-0.298, -0.000] · P=2% (vs obs_k_minus_bb) | -0.211 [-0.351, -0.080] · P=0% (vs obs_whiff_rate) |
| v2a_mano | -0.153 [-0.294, -0.000] · P=2% (vs obs_k_minus_bb) | -0.226 [-0.360, -0.098] · P=0% (vs obs_whiff_rate) |
| v2b_ubicacion | -0.142 [-0.290, +0.011] · P=4% (vs obs_k_minus_bb) | -0.254 [-0.400, -0.122] · P=0% (vs obs_whiff_rate) |
| v2c_limpio | -0.154 [-0.300, +0.006] · P=3% (vs obs_k_minus_bb) | -0.269 [-0.420, -0.131] · P=0% (vs obs_whiff_rate) |
| v2d_arsenal | -0.128 [-0.275, +0.033] · P=6% (vs obs_k_minus_bb) | -0.210 [-0.349, -0.079] · P=0% (vs obs_whiff_rate) |
| diag_v1_swing | -0.154 [-0.317, +0.010] · P=3% (vs obs_k_minus_bb) | -0.187 [-0.297, -0.085] · P=0% (vs obs_whiff_rate) |
| diag_v2d_swing | -0.098 [-0.248, +0.062] · P=11% (vs obs_k_minus_bb) | -0.215 [-0.359, -0.081] · P=0% (vs obs_whiff_rate) |

## Importancia de features — v2d_arsenal (gain, modelo final)

| feature | % del gain total |
|---|---:|
| velo_diff_vs_fb | 25.7% |
| InducedVertBreak | 8.4% |
| ivb_diff_vs_fb | 7.0% |
| RelSpeed | 6.9% |
| vaa_loc_adj | 6.8% |
| SpinAxis | 4.8% |
| AutoPitchType | 4.7% |
| hb_diff_vs_fb | 4.6% |
| HorzBreak | 4.5% |
| Extension | 4.5% |
| RelHeight | 4.4% |
| SpinRate | 4.3% |
| haa_loc_adj | 4.1% |
| SpeedDrop | 3.9% |
| RelSide | 3.8% |
| same_side | 1.6% |
