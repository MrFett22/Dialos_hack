# Ajuste de hiperparámetros — Pitching+ v3

Generado por `src/models/tuning.py`. Diseño y criterio de adopción declarados en el docstring antes de correr.

## Búsqueda (validación por pitcher, learning_rate 0.1)

| sub-modelo | filas ajuste | filas validación | log-loss config. base | log-loss mejor | parámetros elegidos |
|---|---:|---:|---:|---:|---|
| swing | 300,000 | 117,938 | 0.45141 | 0.45017 | hojas=53, min_hijos=1762, λ=0.00453, α=0.838, col=0.85, sub=0.73, árboles=278 |
| take | 275,836 | 57,790 | 0.20447 | 0.20156 | hojas=18, min_hijos=990, λ=0.115, α=0.783, col=0.99, sub=0.77, árboles=330 |
| whiff | 235,798 | 57,761 | 0.46278 | 0.46185 | hojas=29, min_hijos=785, λ=0.531, α=0.00354, col=0.95, sub=0.76, árboles=394 |
| foul | 178,996 | 43,416 | 0.66864 | 0.66771 | hojas=25, min_hijos=597, λ=1.43, α=0.067, col=0.87, sub=0.69, árboles=372 |
| bip | 91,227 | 23,532 | 0.95616 | 0.95234 | hojas=19, min_hijos=971, λ=0.0173, α=0.224, col=0.97, sub=0.90, árboles=128 |

## Criterio 1 y 2: log-loss y calibración out-of-fold (5 folds por pitcher)

| sub-modelo | log-loss base | log-loss ajustado | Δ | ECE base | ECE ajustado |
|---|---:|---:|---:|---:|---:|
| swing | 0.45087 | 0.45045 | -0.00042 | 0.0061 | 0.0068 |
| take | 0.20575 | 0.20446 | -0.00130 | — | — |
| whiff | 0.46170 | 0.46123 | -0.00047 | 0.0055 | 0.0050 |
| foul | 0.66775 | 0.66713 | -0.00062 | 0.0043 | 0.0050 |
| bip | 0.96887 | 0.95828 | -0.01060 | — | — |

## Criterio 3: benchmark del índice oficial (neutral + shrinkage)

| mín. pitches | carreras ajustadas t+1: base | ajustado | Δ |
|---|---|---|---|
| 100 | 0.153 [0.060, 0.246] | 0.164 [0.070, 0.258] | +0.011 [-0.018, +0.042] · P(mejor)=76% |
| 300 | 0.267 [0.123, 0.388] | 0.276 [0.144, 0.398] | +0.008 [-0.028, +0.045] · P(mejor)=68% |

| mín. pitches | estabilidad base | estabilidad ajustado | carreras observadas t+1: base | ajustado |
|---|---|---|---|---|
| 100 | 0.605 [0.539, 0.663] | 0.610 [0.545, 0.669] | 0.132 [0.039, 0.223] | 0.134 [0.044, 0.225] |
| 300 | 0.653 [0.569, 0.720] | 0.658 [0.578, 0.724] | 0.213 [0.082, 0.335] | 0.206 [0.078, 0.327] |

Correlación de rangos (Spearman) entre el índice base y el ajustado: **0.967**.

## Decisión según el criterio

- Log-loss OOF mejora en los 5 sub-modelos: **sí**
- Calibración no empeora (tolerancia 0.002): **sí**
- Benchmark no empeora (P(peor) < 90%): **sí**
- **Se adopta.**
