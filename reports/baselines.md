# Baselines obligatorios del ajuste ambiental

Generado por `src/validation/baselines.py`. Todos los scores usan las mismas predicciones OOF, el mismo shrinkage y la misma escala; solo cambia cómo se trata el entorno (tabla en el docstring del módulo). Criterio de éxito en todas las comparaciones: Δr > 0 y P(Δr > 0) ≥ 0.90 (bootstrap pareado por pitcher, 2,000 réplicas).

## 1. Validación PREESPECIFICADA (§7.2: carreras observadas t+1, mín. 100, contra Pitching+ v1)

### Mínimo 100 pitches (437 pares) — **criterio oficial**

| score | estabilidad | carreras observadas t+1 |
|---|---|---|
| Pitching+ v1 (referencia preespecificada) | 0.536 [0.466, 0.600] | 0.188 [0.097, 0.278] |
| M0 · sin ajuste ambiental | 0.560 [0.489, 0.626] | 0.177 [0.089, 0.265] |
| M1 · ajuste simple temporada × categoría | 0.577 [0.511, 0.638] | 0.169 [0.081, 0.258] |
| M1b · ajuste simple temporada × estadio anónimo | 0.567 [0.499, 0.631] | 0.175 [0.085, 0.264] |
| M2 · contrafactual físico (oficial) | 0.581 [0.512, 0.643] | 0.153 [0.062, 0.244] |
| M3a · contrafactual físico con densidad por estadio | 0.587 [0.517, 0.650] | 0.156 [0.064, 0.246] |
| M3b · contrafactual físico + ajuste por estadio | 0.576 [0.506, 0.641] | 0.159 [0.068, 0.249] |

| modelo | referencia | Δr [IC 95%] · P · cumple |
|---|---|---|
| M0 · sin ajuste ambiental | Pitching+ v1 (referencia preespecificada) | -0.011 [-0.070, +0.046] · P=36% · ❌ |
| M1 · ajuste simple temporada × categoría | Pitching+ v1 (referencia preespecificada) | -0.019 [-0.093, +0.057] · P=32% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | Pitching+ v1 (referencia preespecificada) | -0.013 [-0.085, +0.060] · P=37% · ❌ |
| M2 · contrafactual físico (oficial) | Pitching+ v1 (referencia preespecificada) | -0.035 [-0.111, +0.044] · P=20% · ❌ |
| M3a · contrafactual físico con densidad por estadio | Pitching+ v1 (referencia preespecificada) | -0.032 [-0.103, +0.045] · P=21% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | Pitching+ v1 (referencia preespecificada) | -0.029 [-0.105, +0.049] · P=24% · ❌ |
| M1 · ajuste simple temporada × categoría | M0 · sin ajuste ambiental | -0.008 [-0.047, +0.033] · P=35% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M0 · sin ajuste ambiental | -0.002 [-0.044, +0.038] · P=48% · ❌ |
| M2 · contrafactual físico (oficial) | M0 · sin ajuste ambiental | -0.024 [-0.069, +0.022] · P=15% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M0 · sin ajuste ambiental | -0.020 [-0.065, +0.024] · P=19% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M0 · sin ajuste ambiental | -0.018 [-0.065, +0.028] · P=22% · ❌ |
| M0 · sin ajuste ambiental | M1 · ajuste simple temporada × categoría | +0.008 [-0.033, +0.047] · P=65% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M1 · ajuste simple temporada × categoría | +0.006 [-0.010, +0.023] · P=77% · ❌ |
| M2 · contrafactual físico (oficial) | M1 · ajuste simple temporada × categoría | -0.016 [-0.035, +0.005] · P=7% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1 · ajuste simple temporada × categoría | -0.013 [-0.038, +0.013] · P=17% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1 · ajuste simple temporada × categoría | -0.010 [-0.035, +0.017] · P=22% · ❌ |
| M0 · sin ajuste ambiental | M1b · ajuste simple temporada × estadio anónimo | +0.002 [-0.038, +0.044] · P=52% · ❌ |
| M1 · ajuste simple temporada × categoría | M1b · ajuste simple temporada × estadio anónimo | -0.006 [-0.023, +0.010] · P=23% · ❌ |
| M2 · contrafactual físico (oficial) | M1b · ajuste simple temporada × estadio anónimo | -0.022 [-0.048, +0.004] · P=5% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1b · ajuste simple temporada × estadio anónimo | -0.019 [-0.041, +0.003] · P=4% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1b · ajuste simple temporada × estadio anónimo | -0.016 [-0.036, +0.005] · P=6% · ❌ |

### Mínimo 300 pitches (245 pares) — sensibilidad (no preespecificada)

| score | estabilidad | carreras observadas t+1 |
|---|---|---|
| Pitching+ v1 (referencia preespecificada) | 0.605 [0.514, 0.678] | 0.200 [0.072, 0.320] |
| M0 · sin ajuste ambiental | 0.596 [0.502, 0.673] | 0.221 [0.099, 0.344] |
| M1 · ajuste simple temporada × categoría | 0.617 [0.529, 0.687] | 0.206 [0.080, 0.327] |
| M1b · ajuste simple temporada × estadio anónimo | 0.603 [0.505, 0.680] | 0.206 [0.082, 0.327] |
| M2 · contrafactual físico (oficial) | 0.632 [0.547, 0.700] | 0.210 [0.079, 0.330] |
| M3a · contrafactual físico con densidad por estadio | 0.639 [0.553, 0.708] | 0.222 [0.095, 0.340] |
| M3b · contrafactual físico + ajuste por estadio | 0.626 [0.533, 0.699] | 0.211 [0.085, 0.331] |

| modelo | referencia | Δr [IC 95%] · P · cumple |
|---|---|---|
| M0 · sin ajuste ambiental | Pitching+ v1 (referencia preespecificada) | +0.022 [-0.054, +0.098] · P=70% · ❌ |
| M1 · ajuste simple temporada × categoría | Pitching+ v1 (referencia preespecificada) | +0.006 [-0.103, +0.112] · P=53% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | Pitching+ v1 (referencia preespecificada) | +0.007 [-0.107, +0.120] · P=53% · ❌ |
| M2 · contrafactual físico (oficial) | Pitching+ v1 (referencia preespecificada) | +0.010 [-0.102, +0.121] · P=56% · ❌ |
| M3a · contrafactual físico con densidad por estadio | Pitching+ v1 (referencia preespecificada) | +0.022 [-0.091, +0.131] · P=65% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | Pitching+ v1 (referencia preespecificada) | +0.011 [-0.107, +0.126] · P=56% · ❌ |
| M1 · ajuste simple temporada × categoría | M0 · sin ajuste ambiental | -0.016 [-0.071, +0.039] · P=28% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M0 · sin ajuste ambiental | -0.015 [-0.076, +0.048] · P=31% · ❌ |
| M2 · contrafactual físico (oficial) | M0 · sin ajuste ambiental | -0.011 [-0.064, +0.046] · P=34% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M0 · sin ajuste ambiental | +0.001 [-0.059, +0.062] · P=52% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M0 · sin ajuste ambiental | -0.010 [-0.071, +0.053] · P=38% · ❌ |
| M0 · sin ajuste ambiental | M1 · ajuste simple temporada × categoría | +0.016 [-0.039, +0.071] · P=72% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M1 · ajuste simple temporada × categoría | +0.001 [-0.021, +0.023] · P=54% · ❌ |
| M2 · contrafactual físico (oficial) | M1 · ajuste simple temporada × categoría | +0.005 [-0.018, +0.027] · P=64% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1 · ajuste simple temporada × categoría | +0.016 [-0.015, +0.049] · P=83% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1 · ajuste simple temporada × categoría | +0.005 [-0.025, +0.039] · P=63% · ❌ |
| M0 · sin ajuste ambiental | M1b · ajuste simple temporada × estadio anónimo | +0.015 [-0.048, +0.076] · P=69% · ❌ |
| M1 · ajuste simple temporada × categoría | M1b · ajuste simple temporada × estadio anónimo | -0.001 [-0.023, +0.021] · P=46% · ❌ |
| M2 · contrafactual físico (oficial) | M1b · ajuste simple temporada × estadio anónimo | +0.004 [-0.026, +0.035] · P=58% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1b · ajuste simple temporada × estadio anónimo | +0.015 [-0.011, +0.041] · P=86% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1b · ajuste simple temporada × estadio anónimo | +0.005 [-0.017, +0.026] · P=66% · ❌ |

## 2. Validación SECUNDARIA (post-hoc): carreras t+1 ajustadas por temporada × categoría

Resultado t+1 = valor realizado de cada pitch menos el promedio de su temporada × categoría. Favorece por construcción al ajuste simple que usa la misma partición.

### Mínimo 100 pitches (437 pares)

| score | r con carreras ajustadas t+1 |
|---|---|
| M0 · sin ajuste ambiental | 0.165 [0.074, 0.251] |
| M1 · ajuste simple temporada × categoría | 0.191 [0.100, 0.276] |
| M1b · ajuste simple temporada × estadio anónimo | 0.198 [0.109, 0.285] |
| M2 · contrafactual físico (oficial) | 0.181 [0.090, 0.272] |
| M3a · contrafactual físico con densidad por estadio | 0.188 [0.097, 0.278] |
| M3b · contrafactual físico + ajuste por estadio | 0.190 [0.096, 0.280] |
| carreras ajustadas en t (baseline observado) | 0.165 [0.090, 0.239] |

| modelo | referencia | Δr [IC 95%] · P · cumple |
|---|---|---|
| M1 · ajuste simple temporada × categoría | M0 · sin ajuste ambiental | +0.026 [-0.013, +0.067] · P=91% · ✅ |
| M1b · ajuste simple temporada × estadio anónimo | M0 · sin ajuste ambiental | +0.033 [-0.008, +0.073] · P=95% · ✅ |
| M2 · contrafactual físico (oficial) | M0 · sin ajuste ambiental | +0.016 [-0.028, +0.060] · P=78% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M0 · sin ajuste ambiental | +0.023 [-0.023, +0.067] · P=85% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M0 · sin ajuste ambiental | +0.025 [-0.023, +0.070] · P=87% · ❌ |
| M0 · sin ajuste ambiental | M1 · ajuste simple temporada × categoría | -0.026 [-0.067, +0.013] · P=9% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M1 · ajuste simple temporada × categoría | +0.007 [-0.008, +0.022] · P=81% · ❌ |
| M2 · contrafactual físico (oficial) | M1 · ajuste simple temporada × categoría | -0.010 [-0.032, +0.012] · P=20% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1 · ajuste simple temporada × categoría | -0.004 [-0.029, +0.023] · P=41% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1 · ajuste simple temporada × categoría | -0.002 [-0.028, +0.026] · P=47% · ❌ |
| M0 · sin ajuste ambiental | M1b · ajuste simple temporada × estadio anónimo | -0.033 [-0.073, +0.008] · P=5% · ❌ |
| M1 · ajuste simple temporada × categoría | M1b · ajuste simple temporada × estadio anónimo | -0.007 [-0.022, +0.008] · P=19% · ❌ |
| M2 · contrafactual físico (oficial) | M1b · ajuste simple temporada × estadio anónimo | -0.017 [-0.042, +0.009] · P=11% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1b · ajuste simple temporada × estadio anónimo | -0.010 [-0.033, +0.013] · P=19% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1b · ajuste simple temporada × estadio anónimo | -0.008 [-0.030, +0.014] · P=25% · ❌ |

### Mínimo 300 pitches (245 pares)

| score | r con carreras ajustadas t+1 |
|---|---|
| M0 · sin ajuste ambiental | 0.215 [0.077, 0.334] |
| M1 · ajuste simple temporada × categoría | 0.279 [0.154, 0.396] |
| M1b · ajuste simple temporada × estadio anónimo | 0.280 [0.156, 0.396] |
| M2 · contrafactual físico (oficial) | 0.279 [0.145, 0.401] |
| M3a · contrafactual físico con densidad por estadio | 0.292 [0.158, 0.412] |
| M3b · contrafactual físico + ajuste por estadio | 0.285 [0.151, 0.403] |
| carreras ajustadas en t (baseline observado) | 0.271 [0.147, 0.378] |

| modelo | referencia | Δr [IC 95%] · P · cumple |
|---|---|---|
| M1 · ajuste simple temporada × categoría | M0 · sin ajuste ambiental | +0.064 [+0.009, +0.117] · P=99% · ✅ |
| M1b · ajuste simple temporada × estadio anónimo | M0 · sin ajuste ambiental | +0.065 [+0.004, +0.124] · P=98% · ✅ |
| M2 · contrafactual físico (oficial) | M0 · sin ajuste ambiental | +0.065 [+0.010, +0.123] · P=99% · ✅ |
| M3a · contrafactual físico con densidad por estadio | M0 · sin ajuste ambiental | +0.078 [+0.015, +0.142] · P=99% · ✅ |
| M3b · contrafactual físico + ajuste por estadio | M0 · sin ajuste ambiental | +0.071 [+0.008, +0.135] · P=98% · ✅ |
| M0 · sin ajuste ambiental | M1 · ajuste simple temporada × categoría | -0.064 [-0.117, -0.009] · P=1% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M1 · ajuste simple temporada × categoría | +0.001 [-0.020, +0.022] · P=54% · ❌ |
| M2 · contrafactual físico (oficial) | M1 · ajuste simple temporada × categoría | +0.000 [-0.026, +0.026] · P=53% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1 · ajuste simple temporada × categoría | +0.013 [-0.020, +0.049] · P=79% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1 · ajuste simple temporada × categoría | +0.007 [-0.026, +0.041] · P=65% · ❌ |
| M0 · sin ajuste ambiental | M1b · ajuste simple temporada × estadio anónimo | -0.065 [-0.124, -0.004] · P=2% · ❌ |
| M1 · ajuste simple temporada × categoría | M1b · ajuste simple temporada × estadio anónimo | -0.001 [-0.022, +0.020] · P=46% · ❌ |
| M2 · contrafactual físico (oficial) | M1b · ajuste simple temporada × estadio anónimo | -0.001 [-0.032, +0.033] · P=49% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1b · ajuste simple temporada × estadio anónimo | +0.012 [-0.015, +0.042] · P=80% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1b · ajuste simple temporada × estadio anónimo | +0.006 [-0.019, +0.031] · P=67% · ❌ |

## 2. Validación SECUNDARIA (post-hoc): carreras t+1 ajustadas por temporada × estadio anónimo

Resultado t+1 = valor realizado de cada pitch menos el promedio de su temporada × estadio anónimo. Favorece por construcción al ajuste simple que usa la misma partición.

### Mínimo 100 pitches (437 pares)

| score | r con carreras ajustadas t+1 |
|---|---|
| M0 · sin ajuste ambiental | 0.164 [0.078, 0.247] |
| M1 · ajuste simple temporada × categoría | 0.191 [0.103, 0.273] |
| M1b · ajuste simple temporada × estadio anónimo | 0.201 [0.116, 0.285] |
| M2 · contrafactual físico (oficial) | 0.180 [0.090, 0.269] |
| M3a · contrafactual físico con densidad por estadio | 0.191 [0.102, 0.279] |
| M3b · contrafactual físico + ajuste por estadio | 0.193 [0.103, 0.280] |
| carreras ajustadas en t (baseline observado) | 0.168 [0.093, 0.240] |

| modelo | referencia | Δr [IC 95%] · P · cumple |
|---|---|---|
| M1 · ajuste simple temporada × categoría | M0 · sin ajuste ambiental | +0.027 [-0.013, +0.068] · P=91% · ✅ |
| M1b · ajuste simple temporada × estadio anónimo | M0 · sin ajuste ambiental | +0.037 [-0.005, +0.078] · P=96% · ✅ |
| M2 · contrafactual físico (oficial) | M0 · sin ajuste ambiental | +0.016 [-0.029, +0.061] · P=77% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M0 · sin ajuste ambiental | +0.028 [-0.019, +0.071] · P=88% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M0 · sin ajuste ambiental | +0.029 [-0.020, +0.075] · P=89% · ❌ |
| M0 · sin ajuste ambiental | M1 · ajuste simple temporada × categoría | -0.027 [-0.068, +0.013] · P=9% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M1 · ajuste simple temporada × categoría | +0.010 [-0.004, +0.026] · P=91% · ✅ |
| M2 · contrafactual físico (oficial) | M1 · ajuste simple temporada × categoría | -0.011 [-0.034, +0.013] · P=19% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1 · ajuste simple temporada × categoría | +0.001 [-0.025, +0.028] · P=53% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1 · ajuste simple temporada × categoría | +0.002 [-0.026, +0.030] · P=56% · ❌ |
| M0 · sin ajuste ambiental | M1b · ajuste simple temporada × estadio anónimo | -0.037 [-0.078, +0.005] · P=4% · ❌ |
| M1 · ajuste simple temporada × categoría | M1b · ajuste simple temporada × estadio anónimo | -0.010 [-0.026, +0.004] · P=9% · ❌ |
| M2 · contrafactual físico (oficial) | M1b · ajuste simple temporada × estadio anónimo | -0.021 [-0.047, +0.006] · P=7% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1b · ajuste simple temporada × estadio anónimo | -0.009 [-0.034, +0.014] · P=22% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1b · ajuste simple temporada × estadio anónimo | -0.008 [-0.032, +0.015] · P=24% · ❌ |

### Mínimo 300 pitches (245 pares)

| score | r con carreras ajustadas t+1 |
|---|---|
| M0 · sin ajuste ambiental | 0.219 [0.082, 0.337] |
| M1 · ajuste simple temporada × categoría | 0.276 [0.149, 0.394] |
| M1b · ajuste simple temporada × estadio anónimo | 0.283 [0.155, 0.399] |
| M2 · contrafactual físico (oficial) | 0.277 [0.140, 0.399] |
| M3a · contrafactual físico con densidad por estadio | 0.297 [0.165, 0.417] |
| M3b · contrafactual físico + ajuste por estadio | 0.289 [0.155, 0.408] |
| carreras ajustadas en t (baseline observado) | 0.264 [0.140, 0.375] |

| modelo | referencia | Δr [IC 95%] · P · cumple |
|---|---|---|
| M1 · ajuste simple temporada × categoría | M0 · sin ajuste ambiental | +0.057 [+0.003, +0.111] · P=98% · ✅ |
| M1b · ajuste simple temporada × estadio anónimo | M0 · sin ajuste ambiental | +0.064 [+0.003, +0.124] · P=98% · ✅ |
| M2 · contrafactual físico (oficial) | M0 · sin ajuste ambiental | +0.058 [+0.001, +0.118] · P=98% · ✅ |
| M3a · contrafactual físico con densidad por estadio | M0 · sin ajuste ambiental | +0.078 [+0.014, +0.145] · P=99% · ✅ |
| M3b · contrafactual físico + ajuste por estadio | M0 · sin ajuste ambiental | +0.070 [+0.006, +0.139] · P=98% · ✅ |
| M0 · sin ajuste ambiental | M1 · ajuste simple temporada × categoría | -0.057 [-0.111, -0.003] · P=2% · ❌ |
| M1b · ajuste simple temporada × estadio anónimo | M1 · ajuste simple temporada × categoría | +0.007 [-0.015, +0.028] · P=75% · ❌ |
| M2 · contrafactual físico (oficial) | M1 · ajuste simple temporada × categoría | +0.001 [-0.026, +0.028] · P=55% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1 · ajuste simple temporada × categoría | +0.021 [-0.012, +0.057] · P=88% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1 · ajuste simple temporada × categoría | +0.013 [-0.020, +0.048] · P=77% · ❌ |
| M0 · sin ajuste ambiental | M1b · ajuste simple temporada × estadio anónimo | -0.064 [-0.124, -0.003] · P=2% · ❌ |
| M1 · ajuste simple temporada × categoría | M1b · ajuste simple temporada × estadio anónimo | -0.007 [-0.028, +0.015] · P=25% · ❌ |
| M2 · contrafactual físico (oficial) | M1b · ajuste simple temporada × estadio anónimo | -0.006 [-0.039, +0.028] · P=36% · ❌ |
| M3a · contrafactual físico con densidad por estadio | M1b · ajuste simple temporada × estadio anónimo | +0.014 [-0.014, +0.044] · P=82% · ❌ |
| M3b · contrafactual físico + ajuste por estadio | M1b · ajuste simple temporada × estadio anónimo | +0.006 [-0.018, +0.032] · P=68% · ❌ |

## 3. Neutralización (pitcher-temporadas con ≥100 pitches)

Correlación del score con el % de pitches en Extreme (total y por temporada) y con la exposición física (promedio del ratio de drag de los estadios anónimos donde lanzó; < 1 = más altitud).

| score | % Extreme | 2024 | 2025 | 2026 | exposición física |
|---|---:|---:|---:|---:|---:|
| M0 · sin ajuste ambiental | -0.325 | -0.200 | -0.384 | -0.393 | +0.398 |
| M1 · ajuste simple temporada × categoría | -0.042 | +0.028 | -0.075 | -0.082 | +0.069 |
| M1b · ajuste simple temporada × estadio anónimo | -0.024 | +0.053 | -0.069 | -0.058 | +0.022 |
| M2 · contrafactual físico (oficial) | -0.062 | +0.062 | -0.104 | -0.150 | +0.082 |
| M3a · contrafactual físico con densidad por estadio | -0.054 | +0.065 | -0.101 | -0.129 | +0.038 |
| M3b · contrafactual físico + ajuste por estadio | -0.014 | +0.068 | -0.057 | -0.053 | +0.010 |
