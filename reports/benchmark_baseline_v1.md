# Benchmark año-contra-año — línea base v1

Referencia contra la cual se medirá cada mejora futura de los modelos (features, targets, contrafactual, tuning). Generado por `src/validation/benchmark.py`; resultados crudos en `reports/benchmark/baseline_v1.json`.

## Qué se mide

- **Estabilidad:** correlación de la métrica de un pitcher en la temporada *t* con la misma métrica en *t+1*. Una métrica de habilidad debe repetirse; si no se repite, mide ruido.
- **Poder predictivo:** correlación de la métrica en *t* con resultados **observados** en *t+1*. Resultado principal: `obs_rv_prevented` (carreras evitadas por cada 100 pitches, a partir del run value RE24 observado).
- **Baselines:** estadísticas tradicionales observadas en *t* (whiff%, CSW%, K%, K-BB%, run value). Un modelo que no supera al K-BB% observado del año anterior no está aportando información nueva.
- Pares de temporadas: 2024→2025, 2025→2026 (2026 es parcial). Correlación de Pearson; IC 95% por bootstrap de 2,000 réplicas remuestreando **pitchers** (seed=42).
- Stuff+ v1 se escala aquí por primera vez a `100 + 10·z` por temporada, con predicciones out-of-fold (GroupKFold por pitcher, k=5) del mismo modelo de `is_whiff` de `src/models/stuff_plus.py`.

## Cobertura de pares

- Mínimo 100 pitches en ambas temporadas: **437 pares** de 322 pitchers.
- Mínimo 300 pitches en ambas temporadas: **245 pares** de 193 pitchers.

Solo ~280 pitchers aparecen en dos temporadas consecutivas (de ~600 por temporada) — la muestra de pares es chica, y eso se refleja en intervalos de confianza anchos.

## Resultados — mínimo 100 pitches por temporada

### Estabilidad (r de t con t+1)

| métrica | tipo | r [IC 95%] |
|---|---|---|
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | modelo | 0.668 [0.594, 0.732] |
| Pitching+ v1 (OOF, oficial) | modelo | 0.536 [0.466, 0.600] |
| Whiff% observado (whiffs/pitch) | baseline observado | 0.487 [0.388, 0.576] |
| CSW% observado (strikes cantados + whiffs)/pitch | baseline observado | 0.349 [0.260, 0.429] |
| K% observado (ponches/PA) | baseline observado | 0.420 [0.333, 0.494] |
| K-BB% observado | baseline observado | 0.263 [0.179, 0.342] |
| Run value observado (carreras evitadas/100 pitches) | baseline observado | 0.084 [-0.010, 0.170] |

### Poder predictivo (r de la métrica en t con el resultado observado en t+1)

| métrica | Carreras evitadas/100 pitches (t+1) | K-BB% (t+1) | K% (t+1) | Whiff% (t+1) |
|---|---|---|---|---|
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | 0.182 [0.098, 0.267] | 0.243 [0.155, 0.330] | 0.314 [0.224, 0.393] | 0.362 [0.277, 0.443] |
| Pitching+ v1 (OOF, oficial) | 0.188 [0.097, 0.278] | 0.240 [0.140, 0.335] | 0.156 [0.052, 0.260] | 0.171 [0.061, 0.274] |
| Whiff% observado (whiffs/pitch) | 0.193 [0.114, 0.268] | 0.290 [0.197, 0.375] | 0.414 [0.325, 0.493] | 0.487 [0.388, 0.576] |
| CSW% observado (strikes cantados + whiffs)/pitch | 0.202 [0.112, 0.283] | 0.284 [0.197, 0.362] | 0.363 [0.270, 0.444] | 0.357 [0.269, 0.442] |
| K% observado (ponches/PA) | 0.195 [0.117, 0.275] | 0.269 [0.178, 0.349] | 0.420 [0.333, 0.494] | 0.377 [0.293, 0.456] |
| K-BB% observado | 0.181 [0.096, 0.264] | 0.263 [0.179, 0.342] | 0.294 [0.209, 0.374] | 0.296 [0.212, 0.376] |
| Run value observado (carreras evitadas/100 pitches) | 0.084 [-0.010, 0.170] | 0.136 [0.052, 0.217] | 0.138 [0.051, 0.217] | 0.148 [0.063, 0.229] |

### Modelos vs. el mejor baseline observado (delta de r, bootstrap pareado)

| modelo | resultado t+1 | mejor baseline | Δr [IC 95%] | P(modelo mejor) |
|---|---|---|---|---|
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | Carreras evitadas/100 pitches (t+1) | CSW% observado (strikes cantados + whiffs)/pitch | -0.020 [-0.123, +0.091] | 35% |
| Pitching+ v1 (OOF, oficial) | Carreras evitadas/100 pitches (t+1) | CSW% observado (strikes cantados + whiffs)/pitch | -0.013 [-0.116, +0.086] | 41% |
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | K-BB% (t+1) | Whiff% observado (whiffs/pitch) | -0.046 [-0.128, +0.041] | 16% |
| Pitching+ v1 (OOF, oficial) | K-BB% (t+1) | Whiff% observado (whiffs/pitch) | -0.050 [-0.164, +0.069] | 21% |
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | K% (t+1) | K% observado (ponches/PA) | -0.107 [-0.191, -0.017] | 1% |
| Pitching+ v1 (OOF, oficial) | K% (t+1) | K% observado (ponches/PA) | -0.265 [-0.374, -0.149] | 0% |
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | Whiff% (t+1) | Whiff% observado (whiffs/pitch) | -0.125 [-0.210, -0.036] | 0% |
| Pitching+ v1 (OOF, oficial) | Whiff% (t+1) | Whiff% observado (whiffs/pitch) | -0.316 [-0.445, -0.195] | 0% |

## Resultados — mínimo 300 pitches por temporada

### Estabilidad (r de t con t+1)

| métrica | tipo | r [IC 95%] |
|---|---|---|
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | modelo | 0.711 [0.628, 0.779] |
| Pitching+ v1 (OOF, oficial) | modelo | 0.605 [0.514, 0.678] |
| Whiff% observado (whiffs/pitch) | baseline observado | 0.537 [0.402, 0.648] |
| CSW% observado (strikes cantados + whiffs)/pitch | baseline observado | 0.293 [0.166, 0.404] |
| K% observado (ponches/PA) | baseline observado | 0.488 [0.382, 0.582] |
| K-BB% observado | baseline observado | 0.323 [0.205, 0.430] |
| Run value observado (carreras evitadas/100 pitches) | baseline observado | 0.158 [0.029, 0.279] |

### Poder predictivo (r de la métrica en t con el resultado observado en t+1)

| métrica | Carreras evitadas/100 pitches (t+1) | K-BB% (t+1) | K% (t+1) | Whiff% (t+1) |
|---|---|---|---|---|
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | 0.040 [-0.074, 0.166] | 0.172 [0.047, 0.291] | 0.243 [0.108, 0.367] | 0.327 [0.209, 0.442] |
| Pitching+ v1 (OOF, oficial) | 0.200 [0.072, 0.320] | 0.299 [0.165, 0.426] | 0.174 [0.020, 0.313] | 0.192 [0.032, 0.332] |
| Whiff% observado (whiffs/pitch) | 0.111 [-0.008, 0.226] | 0.268 [0.132, 0.387] | 0.435 [0.308, 0.552] | 0.537 [0.402, 0.648] |
| CSW% observado (strikes cantados + whiffs)/pitch | 0.106 [-0.017, 0.224] | 0.229 [0.104, 0.345] | 0.319 [0.197, 0.431] | 0.367 [0.237, 0.479] |
| K% observado (ponches/PA) | 0.190 [0.069, 0.308] | 0.293 [0.174, 0.392] | 0.488 [0.382, 0.582] | 0.457 [0.350, 0.550] |
| K-BB% observado | 0.193 [0.064, 0.315] | 0.323 [0.205, 0.430] | 0.381 [0.261, 0.492] | 0.377 [0.261, 0.478] |
| Run value observado (carreras evitadas/100 pitches) | 0.158 [0.029, 0.279] | 0.229 [0.115, 0.338] | 0.221 [0.106, 0.332] | 0.226 [0.118, 0.334] |

### Modelos vs. el mejor baseline observado (delta de r, bootstrap pareado)

| modelo | resultado t+1 | mejor baseline | Δr [IC 95%] | P(modelo mejor) |
|---|---|---|---|---|
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | Carreras evitadas/100 pitches (t+1) | K-BB% observado | -0.153 [-0.298, -0.000] | 2% |
| Pitching+ v1 (OOF, oficial) | Carreras evitadas/100 pitches (t+1) | K-BB% observado | +0.007 [-0.121, +0.137] | 54% |
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | K-BB% (t+1) | K-BB% observado | -0.151 [-0.268, -0.027] | 1% |
| Pitching+ v1 (OOF, oficial) | K-BB% (t+1) | K-BB% observado | -0.023 [-0.146, +0.087] | 36% |
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | K% (t+1) | K% observado (ponches/PA) | -0.245 [-0.365, -0.129] | 0% |
| Pitching+ v1 (OOF, oficial) | K% (t+1) | K% observado (ponches/PA) | -0.314 [-0.459, -0.170] | 0% |
| Stuff+ v1 (predictivo; con ubicación implícita y altitud) | Whiff% (t+1) | Whiff% observado (whiffs/pitch) | -0.211 [-0.351, -0.080] | 0% |
| Pitching+ v1 (OOF, oficial) | Whiff% (t+1) | Whiff% observado (whiffs/pitch) | -0.345 [-0.563, -0.156] | 0% |

## Métricas pitch a pitch — Stuff+ v1 (OOF, `is_whiff`)

| AUC | log-loss | Brier | ECE (calibración) | tasa base |
|---:|---:|---:|---:|---:|
| 0.7315 | 0.3184 | 0.0932 | 0.0046 | 0.1133 |

ECE = error de calibración esperado (promedio ponderado de |probabilidad predicha − tasa observada| por decil). Importa porque Pitching+ y Stuff+ promedian probabilidades: si están descalibradas, el promedio se sesga.

| decil | predicho | observado | n |
|---:|---:|---:|---:|
| 1 | 0.0180 | 0.0133 | 62,774 |
| 2 | 0.0384 | 0.0321 | 62,774 |
| 3 | 0.0518 | 0.0497 | 62,773 |
| 4 | 0.0651 | 0.0638 | 62,774 |
| 5 | 0.0817 | 0.0781 | 62,773 |
| 6 | 0.1014 | 0.1008 | 62,774 |
| 7 | 0.1256 | 0.1248 | 62,773 |
| 8 | 0.1582 | 0.1587 | 62,774 |
| 9 | 0.2041 | 0.2117 | 62,773 |
| 10 | 0.2816 | 0.3004 | 62,774 |

## Distribución del Stuff+ v1 (pitcher-temporada, escala 100+10z)

n=1,628 · media=100.00 · std=9.99 · min=60.97 · max=169.19
