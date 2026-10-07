# Stuff+ v3 / Pitching+ v3 — cadena de probabilidades valuada en carreras

Generado por `src/models/run_value_chain.py`. Benchmark de `src/validation/benchmark.py` (pares 2024→2025 y 2025→2026, bootstrap pareado por pitcher, 2,000 réplicas).

## Estructura

Cada pitch se descompone en eventos excluyentes: take → {bola, strike cantado, golpeado}; swing → {whiff, foul, en juego → {out, 1B, 2B, 3B, HR}}. Cinco sub-modelos LightGBM condicionales (swing; resultado del take; whiff | swing; foul | contacto; resultado | en juego), out-of-fold por pitcher.

- **Stuff+ v3:** features de v2d (sin ubicación ni conteo); valores Δ promediados sobre conteos.
- **Pitching+ v3:** features de v2d + `PlateLocHeight`, `PlateLocSide` (espejado por mano) + `Balls`, `Strikes`; Δ en el conteo real.
- **Score:** `100 + 10·z` de −E[Δ] promedio por pitcher-temporada (z dentro de cada año). Más alto = mejor.

## Valores en carreras

Valores terminales (promedio de run_value RE24 en el pitch que termina el turno):

| evento | carreras |
|---|---:|
| walk | +0.378 |
| hbp | +0.397 |
| strikeout | -0.334 |
| out | -0.307 |
| single | +0.512 |
| double | +0.836 |
| triple | +1.041 |
| home_run | +1.472 |

V(conteo) = carreras esperadas del turno desde ese conteo:

| conteo | V | % de pitches |
|---|---:|---:|
| 0-0 | +0.015 | 25.9% |
| 0-1 | -0.035 | 12.3% |
| 0-2 | -0.107 | 6.0% |
| 1-0 | +0.058 | 10.8% |
| 1-1 | -0.002 | 10.1% |
| 1-2 | -0.082 | 8.9% |
| 2-0 | +0.128 | 3.8% |
| 2-1 | +0.055 | 5.5% |
| 2-2 | -0.034 | 7.8% |
| 3-0 | +0.248 | 1.3% |
| 3-1 | +0.171 | 2.5% |
| 3-2 | +0.075 | 5.0% |

Δ(evento) promediado sobre conteos (valores que usa Stuff+ v3):

| evento | Δ carreras |
|---|---:|
| ball | +0.0759 |
| called_strike | -0.1275 |
| hbp | +0.3887 |
| whiff | -0.1275 |
| foul | -0.0474 |
| out | -0.3158 |
| single | +0.5034 |
| double | +0.8276 |
| triple | +1.0326 |
| home_run | +1.4635 |

Chequeo de coherencia: V(0-0) = +0.0147 vs. run_value promedio por turno observado = +0.0147.

## Sub-modelos (OOF, evaluados en su propio universo)

| índice | sub-modelo | n | AUC | log-loss | tasa base |
|---|---|---:|---:|---:|---:|
| stuff | swing | 627,185 | 0.5452 | 0.6879 | 0.4681 |
| stuff | take | 333,626 | multiclase | 0.6286 | — |
| stuff | whiff | 293,559 | 0.6507 | 0.5281 | 0.2424 |
| stuff | foul | 222,412 | 0.5700 | 0.6847 | 0.4840 |
| stuff | bip | 114,759 | multiclase | 0.9796 | — |
| pitching | swing | 627,185 | 0.8671 | 0.4509 | 0.4681 |
| pitching | take | 333,626 | multiclase | 0.2058 | — |
| pitching | whiff | 293,559 | 0.7627 | 0.4617 | 0.2424 |
| pitching | foul | 222,412 | 0.6227 | 0.6677 | 0.4840 |
| pitching | bip | 114,759 | multiclase | 0.9689 | — |

Comparación directa con v1: `whiff | swing` reemplaza a `is_whiff` sobre todos los pitches; `resultado | en juego` reemplaza a `is_weak_contact`/`is_groundball`/`is_hard_contact`/`is_out_in_pa`.

## Calibración del valor esperado

- Media de E[Δ] predicho (Pitching+ v3) = -0.00183 vs. Δ realizado = +0.00000 carreras/pitch.
- Correlación por pitcher-temporada (mín. 100 pitches, misma temporada) entre E[Δ] predicho y Δ realizado: Pitching+ v3 r=0.499, Stuff+ v3 r=0.339.

## Diagnóstico: ¿de dónde sale la diferencia entre pitchers?

Contribución de cada grupo de eventos a la varianza de E[Δ] entre pitcher-temporadas (mín. 100 pitches):

| índice | take (bola/strike cantado) | whiff | foul | bola en juego |
|---|---:|---:|---:|---:|
| stuff | 7% | 24% | 3% | 66% |
| pitching | 48% | 26% | 3% | 24% |

Cada componente como índice por separado (estabilidad y correlación con resultados observados en t+1):

Mínimo 100 pitches:

| componente | estabilidad | carreras t+1 | K-BB% t+1 |
|---|---:|---:|---:|
| stuff_take | 0.750 | -0.113 | -0.076 |
| stuff_whiff | 0.735 | 0.139 | 0.124 |
| stuff_foul | 0.765 | -0.003 | -0.012 |
| stuff_bip | 0.711 | 0.032 | -0.007 |
| pitching_take | 0.504 | 0.102 | 0.172 |
| pitching_whiff | 0.530 | 0.221 | 0.247 |
| pitching_foul | 0.415 | -0.002 | 0.043 |
| pitching_bip | 0.648 | 0.012 | -0.078 |

Mínimo 300 pitches:

| componente | estabilidad | carreras t+1 | K-BB% t+1 |
|---|---:|---:|---:|
| stuff_take | 0.787 | 0.044 | -0.041 |
| stuff_whiff | 0.743 | 0.033 | 0.061 |
| stuff_foul | 0.782 | 0.027 | -0.068 |
| stuff_bip | 0.719 | 0.123 | 0.096 |
| pitching_take | 0.563 | 0.176 | 0.217 |
| pitching_whiff | 0.581 | 0.145 | 0.229 |
| pitching_foul | 0.415 | 0.055 | 0.026 |
| pitching_bip | 0.670 | 0.069 | 0.015 |

El componente de bola en juego de Stuff+ es muy estable pero casi no predice el año siguiente: el modelo aprende patrones de contacto persistentes (p. ej. por tipo de pitch) que no se traducen en prevención de carreras. Motiva la variante **DIPS** (resultado en juego = promedio de liga), agregada **después** de ver estos resultados y por eso validada también en cada par de temporadas por separado.

## Benchmark año-contra-año — mínimo 100 pitches (437 pares, 322 pitchers)

| métrica | estabilidad | carreras evitadas t+1 | K-BB% t+1 | K% t+1 | whiff% t+1 |
|---|---|---|---|---|---|
| stuff_plus_v1 | 0.668 [0.594, 0.732] | 0.182 [0.098, 0.267] | 0.243 [0.155, 0.330] | 0.314 [0.224, 0.393] | 0.362 [0.277, 0.443] |
| stuff_plus_v3 | 0.728 [0.672, 0.776] | 0.032 [-0.061, 0.130] | 0.010 [-0.097, 0.117] | 0.125 [0.020, 0.225] | 0.079 [-0.030, 0.188] |
| stuff_plus_v3_dips | 0.767 [0.724, 0.806] | 0.050 [-0.044, 0.147] | 0.063 [-0.036, 0.162] | 0.196 [0.102, 0.287] | 0.182 [0.095, 0.267] |
| pitching_plus_v1 | 0.536 [0.466, 0.600] | 0.188 [0.097, 0.278] | 0.240 [0.140, 0.335] | 0.156 [0.052, 0.260] | 0.171 [0.061, 0.274] |
| pitching_plus_v3 | 0.566 [0.494, 0.633] | 0.166 [0.074, 0.257] | 0.175 [0.069, 0.277] | 0.114 [-0.007, 0.229] | 0.096 [-0.015, 0.204] |
| pitching_plus_v3_dips | 0.501 [0.425, 0.572] | 0.205 [0.112, 0.292] | 0.231 [0.130, 0.327] | 0.122 [0.016, 0.228] | 0.118 [0.005, 0.222] |
| *Whiff% observado (whiffs/pitch)* | 0.487 [0.388, 0.576] | 0.193 [0.114, 0.268] | 0.290 [0.197, 0.375] | 0.414 [0.325, 0.493] | 0.487 [0.388, 0.576] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.349 [0.260, 0.429] | 0.202 [0.112, 0.283] | 0.284 [0.197, 0.362] | 0.363 [0.270, 0.444] | 0.357 [0.269, 0.442] |
| *K% observado (ponches/PA)* | 0.420 [0.333, 0.494] | 0.195 [0.117, 0.275] | 0.269 [0.178, 0.349] | 0.420 [0.333, 0.494] | 0.377 [0.293, 0.456] |
| *K-BB% observado* | 0.263 [0.179, 0.342] | 0.181 [0.096, 0.264] | 0.263 [0.179, 0.342] | 0.294 [0.209, 0.374] | 0.296 [0.212, 0.376] |
| *Run value observado (carreras evitadas/100 pitches)* | 0.084 [-0.010, 0.170] | 0.084 [-0.010, 0.170] | 0.136 [0.052, 0.217] | 0.138 [0.051, 0.217] | 0.148 [0.063, 0.229] |

**v3 contra v1 (bootstrap pareado):**

| comparación | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |
|---|---|---|---|---|
| stuff_plus_v3 vs stuff_plus_v1 | +0.060 [-0.022, +0.148] · P=91% | -0.150 [-0.267, -0.031] · P=1% | -0.233 [-0.344, -0.121] · P=0% | -0.283 [-0.392, -0.172] · P=0% |
| stuff_plus_v3_dips vs stuff_plus_v1 | +0.099 [+0.032, +0.173] · P=100% | -0.132 [-0.232, -0.031] · P=1% | -0.180 [-0.271, -0.086] · P=0% | -0.180 [-0.270, -0.085] · P=0% |
| pitching_plus_v3 vs pitching_plus_v1 | +0.031 [-0.023, +0.085] · P=87% | -0.022 [-0.094, +0.047] · P=28% | -0.065 [-0.121, -0.011] · P=1% | -0.075 [-0.128, -0.016] · P=1% |
| pitching_plus_v3_dips vs pitching_plus_v1 | -0.035 [-0.099, +0.032] · P=16% | +0.017 [-0.049, +0.086] · P=70% | -0.008 [-0.074, +0.061] · P=39% | -0.052 [-0.121, +0.017] · P=7% |

**Contra el mejor baseline observado:**

| modelo | carreras evitadas t+1 | K-BB% t+1 |
|---|---|---|
| stuff_plus_v1 | -0.020 [-0.123, +0.091] · P=35% (vs obs_csw) | -0.046 [-0.128, +0.041] · P=16% (vs obs_whiff_rate) |
| stuff_plus_v3 | -0.170 [-0.287, -0.046] · P=0% (vs obs_csw) | -0.279 [-0.403, -0.159] · P=0% (vs obs_whiff_rate) |
| stuff_plus_v3_dips | -0.152 [-0.274, -0.024] · P=1% (vs obs_csw) | -0.227 [-0.331, -0.123] · P=0% (vs obs_whiff_rate) |
| pitching_plus_v1 | -0.013 [-0.116, +0.086] · P=41% (vs obs_csw) | -0.050 [-0.164, +0.069] · P=21% (vs obs_whiff_rate) |
| pitching_plus_v3 | -0.035 [-0.143, +0.076] · P=27% (vs obs_csw) | -0.115 [-0.236, +0.007] · P=3% (vs obs_whiff_rate) |
| pitching_plus_v3_dips | +0.003 [-0.111, +0.116] · P=53% (vs obs_csw) | -0.058 [-0.182, +0.062] · P=17% (vs obs_whiff_rate) |

## Benchmark año-contra-año — mínimo 300 pitches (245 pares, 193 pitchers)

| métrica | estabilidad | carreras evitadas t+1 | K-BB% t+1 | K% t+1 | whiff% t+1 |
|---|---|---|---|---|---|
| stuff_plus_v1 | 0.711 [0.628, 0.779] | 0.040 [-0.074, 0.166] | 0.172 [0.047, 0.291] | 0.243 [0.108, 0.367] | 0.327 [0.209, 0.442] |
| stuff_plus_v3 | 0.738 [0.668, 0.793] | 0.121 [-0.013, 0.243] | 0.071 [-0.062, 0.200] | 0.163 [0.028, 0.289] | 0.153 [0.021, 0.273] |
| stuff_plus_v3_dips | 0.793 [0.741, 0.833] | 0.085 [-0.050, 0.218] | 0.029 [-0.119, 0.175] | 0.187 [0.055, 0.324] | 0.197 [0.079, 0.314] |
| pitching_plus_v1 | 0.605 [0.514, 0.678] | 0.200 [0.072, 0.320] | 0.299 [0.165, 0.426] | 0.174 [0.020, 0.313] | 0.192 [0.032, 0.332] |
| pitching_plus_v3 | 0.612 [0.514, 0.688] | 0.226 [0.099, 0.345] | 0.254 [0.112, 0.386] | 0.169 [0.014, 0.310] | 0.132 [-0.012, 0.265] |
| pitching_plus_v3_dips | 0.550 [0.447, 0.636] | 0.222 [0.097, 0.336] | 0.263 [0.108, 0.404] | 0.156 [-0.002, 0.305] | 0.102 [-0.045, 0.243] |
| *Whiff% observado (whiffs/pitch)* | 0.537 [0.402, 0.648] | 0.111 [-0.008, 0.226] | 0.268 [0.132, 0.387] | 0.435 [0.308, 0.552] | 0.537 [0.402, 0.648] |
| *CSW% observado (strikes cantados + whiffs)/pitch* | 0.293 [0.166, 0.404] | 0.106 [-0.017, 0.224] | 0.229 [0.104, 0.345] | 0.319 [0.197, 0.431] | 0.367 [0.237, 0.479] |
| *K% observado (ponches/PA)* | 0.488 [0.382, 0.582] | 0.190 [0.069, 0.308] | 0.293 [0.174, 0.392] | 0.488 [0.382, 0.582] | 0.457 [0.350, 0.550] |
| *K-BB% observado* | 0.323 [0.205, 0.430] | 0.193 [0.064, 0.315] | 0.323 [0.205, 0.430] | 0.381 [0.261, 0.492] | 0.377 [0.261, 0.478] |
| *Run value observado (carreras evitadas/100 pitches)* | 0.158 [0.029, 0.279] | 0.158 [0.029, 0.279] | 0.229 [0.115, 0.338] | 0.221 [0.106, 0.332] | 0.226 [0.118, 0.334] |

**v3 contra v1 (bootstrap pareado):**

| comparación | estabilidad | carreras evitadas t+1 | K-BB% t+1 | whiff% t+1 |
|---|---|---|---|---|
| stuff_plus_v3 vs stuff_plus_v1 | +0.027 [-0.070, +0.125] · P=70% | +0.081 [-0.069, +0.216] · P=85% | -0.101 [-0.232, +0.027] · P=7% | -0.173 [-0.313, -0.027] · P=1% |
| stuff_plus_v3_dips vs stuff_plus_v1 | +0.082 [+0.006, +0.168] · P=99% | +0.045 [-0.087, +0.173] · P=75% | -0.143 [-0.263, -0.015] · P=1% | -0.129 [-0.262, +0.004] · P=3% |
| pitching_plus_v3 vs pitching_plus_v1 | +0.007 [-0.056, +0.072] · P=57% | +0.027 [-0.045, +0.099] · P=75% | -0.046 [-0.112, +0.017] · P=8% | -0.060 [-0.131, +0.010] · P=5% |
| pitching_plus_v3_dips vs pitching_plus_v1 | -0.055 [-0.136, +0.020] · P=8% | +0.022 [-0.075, +0.118] · P=67% | -0.037 [-0.116, +0.039] · P=18% | -0.090 [-0.183, +0.004] · P=3% |

**Contra el mejor baseline observado:**

| modelo | carreras evitadas t+1 | K-BB% t+1 |
|---|---|---|
| stuff_plus_v1 | -0.153 [-0.298, -0.000] · P=2% (vs obs_k_minus_bb) | -0.151 [-0.268, -0.027] · P=1% (vs obs_k_minus_bb) |
| stuff_plus_v3 | -0.072 [-0.221, +0.071] · P=17% (vs obs_k_minus_bb) | -0.252 [-0.397, -0.110] · P=0% (vs obs_k_minus_bb) |
| stuff_plus_v3_dips | -0.108 [-0.254, +0.033] · P=7% (vs obs_k_minus_bb) | -0.294 [-0.441, -0.136] · P=0% (vs obs_k_minus_bb) |
| pitching_plus_v1 | +0.007 [-0.121, +0.137] · P=54% (vs obs_k_minus_bb) | -0.023 [-0.146, +0.087] · P=36% (vs obs_k_minus_bb) |
| pitching_plus_v3 | +0.033 [-0.091, +0.161] · P=71% (vs obs_k_minus_bb) | -0.069 [-0.198, +0.049] · P=14% (vs obs_k_minus_bb) |
| pitching_plus_v3_dips | +0.029 [-0.090, +0.151] · P=69% (vs obs_k_minus_bb) | -0.060 [-0.179, +0.056] · P=16% (vs obs_k_minus_bb) |

## Validación por par de temporadas (carreras evitadas t+1, delta contra v1)

| mín. pitches | par | n | stuff_plus_v3 | stuff_plus_v3_dips | pitching_plus_v3 | pitching_plus_v3_dips |
|---|---|---:|---|---|---|---|
| 100 | 2024->2025 | 222 | 0.029 (-0.134, P=9%) | 0.075 (-0.088, P=15%) | 0.193 (-0.028, P=28%) | 0.258 (+0.037, P=73%) |
| 100 | 2025->2026 | 215 | 0.050 (-0.144, P=3%) | 0.038 (-0.156, P=1%) | 0.148 (-0.032, P=20%) | 0.168 (-0.011, P=41%) |
| 300 | 2024->2025 | 146 | 0.078 (+0.030, P=60%) | 0.073 (+0.025, P=58%) | 0.219 (-0.046, P=17%) | 0.237 (-0.029, P=35%) |
| 300 | 2025->2026 | 99 | 0.220 (+0.165, P=90%) | 0.176 (+0.121, P=86%) | 0.229 (+0.096, P=96%) | 0.192 (+0.060, P=78%) |

- Distribución stuff_plus_v3: n=1,628 · media=100.00 · std=9.99 · min=27.07 · max=138.37
- Distribución stuff_plus_v3_dips: n=1,628 · media=100.00 · std=9.99 · min=24.44 · max=147.14
- Distribución pitching_plus_v3: n=1,628 · media=100.00 · std=9.99 · min=43.01 · max=139.21
- Distribución pitching_plus_v3_dips: n=1,628 · media=100.00 · std=9.99 · min=37.09 · max=152.71
