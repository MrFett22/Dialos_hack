# Stuff+ v4 — P(whiff | swing), neutral por altitud, con shrinkage

Generado por `src/models/stuff_plus_v4.py`. Diseño y criterio de aceptación declarados en el docstring antes de correr.

## Modelo

- P(whiff | swing) OOF sobre 293,559 swings: AUC=0.6507, log-loss=0.5281, tasa base=0.2424.
- Features (sin ubicación ni conteo): RelSpeed, SpeedDrop, SpinRate, SpinAxis, RelHeight, RelSide, Extension, InducedVertBreak, HorzBreak, vaa_loc_adj, haa_loc_adj, velo_diff_vs_fb, ivb_diff_vs_fb, hb_diff_vs_fb, same_side, AutoPitchType
- Score: promedio de P(whiff | swing) de todos los pitches del pitcher, proyectados a nivel del mar; shrinkage empírico-bayesiano; 10 puntos = 1 desviación estándar de habilidad real.

| temporada | μ P(whiff\|swing) | τ | pitches para encoger a la mitad |
|---|---:|---:|---:|
| 2024 | 0.2510 | 0.0345 | 8 |
| 2025 | 0.2526 | 0.0328 | 9 |
| 2026 | 0.2571 | 0.0356 | 8 |

## Criterio principal: carreras del año siguiente ajustadas por entorno

| mín. pitches | pares | Stuff+ v1 | v4 con altitud | v4 neutral | **v4 oficial (neutral + shrinkage)** | Δ v4 oficial vs v1 |
|---|---:|---|---|---|---|---|
| 100 | 437 | 0.194 [0.102, 0.285] | 0.172 [0.080, 0.261] | 0.145 [0.053, 0.241] | 0.145 [0.052, 0.241] | -0.050 [-0.127, +0.036] · P=13% |
| 300 | 245 | 0.068 [-0.063, 0.200] | 0.117 [-0.021, 0.247] | 0.119 [-0.023, 0.252] | 0.119 [-0.023, 0.251] | +0.050 [-0.073, +0.174] · P=78% |

## Benchmark año-contra-año — mínimo 100 pitches (437 pares)

| métrica | estabilidad | carreras observadas t+1 | K% t+1 | whiff% t+1 |
|---|---|---|---|---|
| Stuff+ v1 | 0.668 [0.594, 0.732] | 0.182 [0.098, 0.267] | 0.314 [0.224, 0.393] | 0.362 [0.277, 0.443] |
| Stuff+ v4 con altitud | 0.734 [0.687, 0.778] | 0.175 [0.089, 0.261] | 0.294 [0.197, 0.382] | 0.327 [0.238, 0.412] |
| Stuff+ v4 neutral | 0.768 [0.726, 0.807] | 0.135 [0.045, 0.228] | 0.262 [0.165, 0.350] | 0.298 [0.204, 0.387] |
| Stuff+ v4 neutral + shrinkage | 0.769 [0.727, 0.807] | 0.134 [0.044, 0.226] | 0.261 [0.164, 0.349] | 0.298 [0.204, 0.388] |
| *Whiff% observado (whiffs/pitch)* | 0.487 [0.388, 0.576] | 0.193 [0.114, 0.268] | 0.414 [0.325, 0.493] | 0.487 [0.388, 0.576] |
| *K% observado (ponches/PA)* | 0.420 [0.333, 0.494] | 0.195 [0.117, 0.275] | 0.420 [0.333, 0.494] | 0.377 [0.293, 0.456] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.349 [0.260, 0.429] | 0.202 [0.112, 0.283] | 0.363 [0.270, 0.444] | 0.357 [0.269, 0.442] |

Delta contra Stuff+ v1 (bootstrap pareado):

| métrica | estabilidad | carreras observadas t+1 | whiff% t+1 |
|---|---|---|---|
| Stuff+ v4 con altitud | +0.066 [+0.004, +0.133] · P=98% | -0.007 [-0.079, +0.069] · P=44% | -0.035 [-0.108, +0.038] · P=17% |
| Stuff+ v4 neutral | +0.100 [+0.036, +0.169] · P=100% | -0.047 [-0.128, +0.035] · P=14% | -0.064 [-0.145, +0.017] · P=6% |
| Stuff+ v4 neutral + shrinkage | +0.100 [+0.038, +0.171] · P=100% | -0.048 [-0.129, +0.033] · P=13% | -0.064 [-0.146, +0.017] · P=6% |

Contra el mejor baseline observado:

| métrica | whiff% t+1 | carreras observadas t+1 |
|---|---|---|
| Stuff+ v1 | -0.125 [-0.210, -0.036] · P=0% (vs obs_whiff_rate) | -0.020 [-0.123, +0.091] · P=35% (vs obs_csw) |
| Stuff+ v4 con altitud | -0.160 [-0.264, -0.060] · P=0% (vs obs_whiff_rate) | -0.027 [-0.135, +0.084] · P=32% (vs obs_csw) |
| Stuff+ v4 neutral | -0.189 [-0.303, -0.079] · P=0% (vs obs_whiff_rate) | -0.066 [-0.183, +0.052] · P=14% (vs obs_csw) |
| Stuff+ v4 neutral + shrinkage | -0.188 [-0.303, -0.079] · P=0% (vs obs_whiff_rate) | -0.067 [-0.184, +0.051] · P=13% (vs obs_csw) |

## Benchmark año-contra-año — mínimo 300 pitches (245 pares)

| métrica | estabilidad | carreras observadas t+1 | K% t+1 | whiff% t+1 |
|---|---|---|---|---|
| Stuff+ v1 | 0.711 [0.628, 0.779] | 0.040 [-0.074, 0.166] | 0.243 [0.108, 0.367] | 0.327 [0.209, 0.442] |
| Stuff+ v4 con altitud | 0.740 [0.681, 0.792] | 0.077 [-0.053, 0.205] | 0.252 [0.122, 0.378] | 0.331 [0.206, 0.446] |
| Stuff+ v4 neutral | 0.790 [0.733, 0.841] | 0.053 [-0.079, 0.187] | 0.219 [0.087, 0.350] | 0.275 [0.140, 0.401] |
| Stuff+ v4 neutral + shrinkage | 0.791 [0.733, 0.841] | 0.052 [-0.079, 0.186] | 0.219 [0.086, 0.349] | 0.275 [0.140, 0.401] |
| *Whiff% observado (whiffs/pitch)* | 0.537 [0.402, 0.648] | 0.111 [-0.008, 0.226] | 0.435 [0.308, 0.552] | 0.537 [0.402, 0.648] |
| *K% observado (ponches/PA)* | 0.488 [0.382, 0.582] | 0.190 [0.069, 0.308] | 0.488 [0.382, 0.582] | 0.457 [0.350, 0.550] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.293 [0.166, 0.404] | 0.106 [-0.017, 0.224] | 0.319 [0.197, 0.431] | 0.367 [0.237, 0.479] |

Delta contra Stuff+ v1 (bootstrap pareado):

| métrica | estabilidad | carreras observadas t+1 | whiff% t+1 |
|---|---|---|---|
| Stuff+ v4 con altitud | +0.030 [-0.043, +0.108] · P=80% | +0.037 [-0.068, +0.146] · P=74% | +0.004 [-0.103, +0.110] · P=53% |
| Stuff+ v4 neutral | +0.080 [+0.005, +0.160] · P=98% | +0.013 [-0.104, +0.129] · P=59% | -0.051 [-0.172, +0.064] · P=20% |
| Stuff+ v4 neutral + shrinkage | +0.080 [+0.006, +0.161] · P=98% | +0.012 [-0.104, +0.128] · P=58% | -0.051 [-0.172, +0.063] · P=20% |

Contra el mejor baseline observado:

| métrica | whiff% t+1 | carreras observadas t+1 |
|---|---|---|
| Stuff+ v1 | -0.211 [-0.351, -0.080] · P=0% (vs obs_whiff_rate) | -0.153 [-0.298, -0.000] · P=2% (vs obs_k_minus_bb) |
| Stuff+ v4 con altitud | -0.206 [-0.353, -0.066] · P=0% (vs obs_whiff_rate) | -0.116 [-0.270, +0.047] · P=8% (vs obs_k_minus_bb) |
| Stuff+ v4 neutral | -0.262 [-0.418, -0.110] · P=0% (vs obs_whiff_rate) | -0.141 [-0.299, +0.027] · P=5% (vs obs_k_minus_bb) |
| Stuff+ v4 neutral + shrinkage | -0.262 [-0.419, -0.110] · P=0% (vs obs_whiff_rate) | -0.141 [-0.299, +0.026] · P=5% (vs obs_k_minus_bb) |

## P(whiff | swing) promedio por tipo de pitch (a nivel del mar)

| AutoPitchType   |   mean |   size |
|:----------------|-------:|-------:|
| Other           | 0.3458 |     60 |
| Sweeper         | 0.3436 |    571 |
| Curveball       | 0.336  |  53269 |
| Slider          | 0.3254 | 137513 |
| Splitter        | 0.3209 |   7212 |
| OneSeamFastBall | 0.3169 |      1 |
| Changeup        | 0.299  |  77730 |
| TwoSeamFastBall | 0.2989 |      1 |
| Knuckleball     | 0.292  |      1 |
| Cutter          | 0.2313 |  26122 |
| Four-Seam       | 0.2194 | 215235 |
| Sinker          | 0.1524 | 109470 |

## Top 10 Stuff+ v4 (≥300 pitches)

| pitcher_anon_id   |   year |   n_pitches |   stuff_plus |   stuff_plus_lo80 |   stuff_plus_hi80 |   stuff_plus_sin_shrinkage |   p_whiff_swing_neutral |   stuff_plus_con_altitud |
|:------------------|-------:|------------:|-------------:|------------------:|------------------:|---------------------------:|------------------------:|-------------------------:|
| pitcher_00952     |   2024 |         893 |       132.36 |            131.17 |            133.54 |                     132.64 |                    0.36 |                   129.33 |
| pitcher_00619     |   2024 |        1464 |       130.01 |            129.09 |            130.94 |                     130.17 |                    0.36 |                   125.73 |
| pitcher_00465     |   2026 |         357 |       129.78 |            127.9  |            131.66 |                     130.43 |                    0.37 |                   131.93 |
| pitcher_01127     |   2025 |         693 |       127.69 |            126.24 |            129.15 |                     128.06 |                    0.34 |                   133.24 |
| pitcher_00616     |   2025 |         591 |       126.38 |            124.8  |            127.95 |                     126.78 |                    0.34 |                   125.85 |
| pitcher_01117     |   2025 |         964 |       125.6  |            124.36 |            126.84 |                     125.84 |                    0.34 |                   114.94 |
| pitcher_00595     |   2025 |         735 |       124.62 |            123.21 |            126.04 |                     124.93 |                    0.33 |                   126.17 |
| pitcher_00482     |   2024 |         336 |       124.55 |            122.63 |            126.47 |                     125.12 |                    0.34 |                   120.28 |
| pitcher_00866     |   2025 |         735 |       124.03 |            122.61 |            125.45 |                     124.33 |                    0.33 |                   118.51 |
| pitcher_00205     |   2024 |         849 |       123.7  |            122.48 |            124.92 |                     123.91 |                    0.33 |                   118.26 |

Distribución (todas las pitcher-temporadas): media=99.09, std=10.04, min=66.87, max=149.53.
