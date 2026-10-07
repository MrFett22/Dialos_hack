# Validación temporal estricta (entrenamiento con temporadas < t)

Generado por `src/validation/temporal.py`. Qué se estima con qué datos: ver el docstring del módulo (todo con temporadas < t, salvo los hiperparámetros, que son globales, y el shrinkage y el ajuste trivial, que usan solo predicciones de la temporada t, sin resultados).

Única pareja evaluable: **2025 → 2026** (para puntuar 2024 no hay temporadas anteriores). *Original* = los scores del pipeline normal (OOF por pitcher con las 3 temporadas) restringidos a la misma pareja y los mismos pitchers.

## Resultado preespecificado: carreras observadas t+1

### Mínimo 100 pitches (215 pares)

| score | r [IC 95%] |
|---|---|
| M0 original | 0.161 [0.045, 0.274] |
| M1 original | 0.179 [0.065, 0.297] |
| M1B original | 0.198 [0.081, 0.316] |
| M2 original | 0.159 [0.038, 0.282] |
| M3A original | 0.167 [0.048, 0.291] |
| M0 temporal estricto | 0.158 [0.042, 0.267] |
| M1 temporal estricto | 0.172 [0.054, 0.283] |
| M1B temporal estricto | 0.191 [0.069, 0.303] |
| M2 temporal estricto | 0.148 [0.024, 0.264] |
| M3A temporal estricto | 0.153 [0.028, 0.271] |

| comparación | Δr [IC 95%] · P · cumple (P ≥ 0.90) |
|---|---|
| M1 temporal estricto vs M0 temporal estricto | +0.014 [-0.038, +0.070] · P=71% · ❌ |
| M2 temporal estricto vs M0 temporal estricto | -0.010 [-0.062, +0.043] · P=36% · ❌ |
| M2 temporal estricto vs M1 temporal estricto | -0.024 [-0.052, +0.003] · P=4% · ❌ |
| M3A temporal estricto vs M1 temporal estricto | -0.019 [-0.050, +0.012] · P=11% · ❌ |
| M3A temporal estricto vs M1B temporal estricto | -0.038 [-0.067, -0.009] · P=1% · ❌ |
| M2 temporal estricto vs M1B temporal estricto | -0.043 [-0.073, -0.012] · P=0% · ❌ |
| M2 original vs M1 temporal estricto | -0.013 [-0.065, +0.039] · P=32% · ❌ |

### Mínimo 300 pitches (99 pares)

| score | r [IC 95%] |
|---|---|
| M0 original | 0.234 [0.039, 0.409] |
| M1 original | 0.238 [0.035, 0.419] |
| M1B original | 0.266 [0.064, 0.443] |
| M2 original | 0.233 [0.037, 0.406] |
| M3A original | 0.261 [0.062, 0.431] |
| M0 temporal estricto | 0.273 [0.084, 0.431] |
| M1 temporal estricto | 0.275 [0.067, 0.446] |
| M1B temporal estricto | 0.302 [0.094, 0.464] |
| M2 temporal estricto | 0.258 [0.055, 0.427] |
| M3A temporal estricto | 0.279 [0.077, 0.444] |

| comparación | Δr [IC 95%] · P · cumple (P ≥ 0.90) |
|---|---|
| M1 temporal estricto vs M0 temporal estricto | +0.003 [-0.073, +0.078] · P=54% · ❌ |
| M2 temporal estricto vs M0 temporal estricto | -0.014 [-0.082, +0.053] · P=33% · ❌ |
| M2 temporal estricto vs M1 temporal estricto | -0.017 [-0.058, +0.018] · P=18% · ❌ |
| M3A temporal estricto vs M1 temporal estricto | +0.004 [-0.045, +0.048] · P=55% · ❌ |
| M3A temporal estricto vs M1B temporal estricto | -0.023 [-0.066, +0.016] · P=12% · ❌ |
| M2 temporal estricto vs M1B temporal estricto | -0.044 [-0.092, +0.000] · P=3% · ❌ |
| M2 original vs M1 temporal estricto | -0.042 [-0.134, +0.045] · P=17% · ❌ |

## Resultado secundario (post-hoc): carreras ajustadas por entorno t+1

### Mínimo 100 pitches (215 pares)

| score | r [IC 95%] |
|---|---|
| M0 original | 0.132 [0.009, 0.252] |
| M1 original | 0.173 [0.058, 0.297] |
| M1B original | 0.186 [0.067, 0.308] |
| M2 original | 0.160 [0.038, 0.287] |
| M3A original | 0.167 [0.047, 0.295] |
| M0 temporal estricto | 0.125 [0.007, 0.242] |
| M1 temporal estricto | 0.162 [0.042, 0.282] |
| M1B temporal estricto | 0.175 [0.054, 0.294] |
| M2 temporal estricto | 0.136 [0.013, 0.261] |
| M3A temporal estricto | 0.143 [0.018, 0.270] |

| comparación | Δr [IC 95%] · P · cumple (P ≥ 0.90) |
|---|---|
| M1 temporal estricto vs M0 temporal estricto | +0.037 [-0.016, +0.092] · P=92% · ✅ |
| M2 temporal estricto vs M0 temporal estricto | +0.011 [-0.041, +0.063] · P=67% · ❌ |
| M2 temporal estricto vs M1 temporal estricto | -0.026 [-0.059, +0.004] · P=5% · ❌ |
| M3A temporal estricto vs M1 temporal estricto | -0.019 [-0.053, +0.017] · P=14% · ❌ |
| M3A temporal estricto vs M1B temporal estricto | -0.032 [-0.064, +0.002] · P=3% · ❌ |
| M2 temporal estricto vs M1B temporal estricto | -0.039 [-0.075, -0.003] · P=2% · ❌ |
| M2 original vs M1 temporal estricto | -0.003 [-0.068, +0.058] · P=48% · ❌ |

### Mínimo 300 pitches (99 pares)

| score | r [IC 95%] |
|---|---|
| M0 original | 0.181 [-0.028, 0.384] |
| M1 original | 0.267 [0.079, 0.447] |
| M1B original | 0.285 [0.101, 0.461] |
| M2 original | 0.261 [0.062, 0.449] |
| M3A original | 0.278 [0.079, 0.458] |
| M0 temporal estricto | 0.225 [0.050, 0.400] |
| M1 temporal estricto | 0.305 [0.135, 0.460] |
| M1B temporal estricto | 0.323 [0.153, 0.476] |
| M2 temporal estricto | 0.281 [0.102, 0.449] |
| M3A temporal estricto | 0.296 [0.124, 0.461] |

| comparación | Δr [IC 95%] · P · cumple (P ≥ 0.90) |
|---|---|
| M1 temporal estricto vs M0 temporal estricto | +0.080 [+0.002, +0.157] · P=98% · ✅ |
| M2 temporal estricto vs M0 temporal estricto | +0.056 [-0.016, +0.131] · P=94% · ✅ |
| M2 temporal estricto vs M1 temporal estricto | -0.023 [-0.077, +0.021] · P=18% · ❌ |
| M3A temporal estricto vs M1 temporal estricto | -0.009 [-0.064, +0.041] · P=36% · ❌ |
| M3A temporal estricto vs M1B temporal estricto | -0.027 [-0.081, +0.017] · P=14% · ❌ |
| M2 temporal estricto vs M1B temporal estricto | -0.041 [-0.100, +0.012] · P=7% · ❌ |
| M2 original vs M1 temporal estricto | -0.044 [-0.163, +0.061] · P=23% · ❌ |

## Neutralización (corr. con % de pitches en Extreme, ≥100 pitches)

| score | 2025 | 2026 |
|---|---:|---:|
| M0 temporal estricto | -0.326 | -0.381 |
| M1 temporal estricto | -0.024 | -0.070 |
| M1B temporal estricto | -0.022 | -0.047 |
| M2 temporal estricto | -0.093 | -0.139 |
| M3A temporal estricto | -0.079 | -0.121 |
