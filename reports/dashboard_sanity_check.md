# Chequeo de sanity del dashboard — identidad + proyeccion vs. realidad

Reusa la logica real de `dashboard/scoring.py` (la misma que corre `dashboard/app.py`), no una reimplementacion paralela — este chequeo valida el mecanismo end-to-end, no solo la logica aislada.

## 1. Identidad: No Altitude -> No Altitude (residual contra si mismo)

28 pitches reales lanzados en No Altitude (7 AutoPitchType x hasta 4 cuartiles de SpinRate), ingresados al dashboard (`reference_category='No Altitude'`) — la salida 'No Altitude' debe coincidir con el movimiento real ingresado (el residual de Magnus se resta y se vuelve a sumar bajo la MISMA densidad de aire, asi que la diferencia deberia ser 0 salvo error de punto flotante).

**Diferencia maxima observada en los 28 pitches: 8.88e-16 pulgadas** — practicamente cero (error de punto flotante, no una discrepancia real) en el 100% de los casos. Confirma que el residual de Magnus se calcula y se reaplica correctamente.

## 2. Proyeccion Extreme Altitude vs. pitches reales similares (SIN controlar lateralidad)

Para cada uno de los 28 pitches, se comparo la proyeccion 'Extreme Altitude' del dashboard contra pitches REALES del dataset con el mismo AutoPitchType y SpinRate dentro de ±150 rpm, que efectivamente se lanzaron en Extreme Altitude — SIN filtrar por `PitcherThrows` (ver seccion 3 para por que esto importa).

| AutoPitchType | SpinRate | n similares | z (IVB) | z (HB) | veredicto |
|---|---:|---:|---:|---:|---|
| Changeup | 1283 | 1988 | +2.73 | -1.43 | FUERA de 2 desviaciones estandar (max|z|=2.73) — revisar |
| Changeup | 1784 | 6722 | -1.07 | -1.25 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Changeup | 1904 | 6547 | -0.66 | +0.78 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Changeup | 2287 | 2994 | +1.52 | +0.68 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Curveball | 2076 | 1637 | +0.07 | -0.33 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Curveball | 2541 | 5679 | -1.83 | -0.92 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Curveball | 2685 | 5556 | -0.04 | +1.48 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Curveball | 2775 | 4854 | -0.22 | +1.17 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Cutter | 2107 | 2984 | -1.71 | +0.45 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Cutter | 2331 | 4975 | -0.43 | +0.02 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Cutter | 2578 | 3657 | -1.85 | -0.21 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Cutter | 2683 | 2026 | +0.84 | -1.26 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Four-Seam | 2178 | 29021 | +0.57 | -1.80 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Four-Seam | 2329 | 41556 | +1.64 | +0.20 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Four-Seam | 2357 | 41063 | +0.18 | +0.65 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Four-Seam | 2446 | 33361 | -0.80 | -1.77 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Sinker | 2119 | 14191 | +0.58 | -1.57 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Sinker | 2202 | 19050 | -0.73 | +0.69 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Sinker | 2290 | 20656 | +0.55 | -1.68 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Sinker | 2381 | 17290 | -0.61 | +0.96 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Slider | 1315 | 393 | -0.41 | -0.28 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Slider | 2360 | 18874 | -0.48 | -0.11 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Slider | 2464 | 19276 | -0.26 | -1.31 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Slider | 3409 | 128 | -0.67 | -0.98 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Splitter | 945 | 883 | -0.10 | -0.20 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Splitter | 1043 | 1075 | +1.60 | -0.01 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |
| Splitter | 1104 | 1074 | +0.80 | +0.46 | DENTRO de 1 desviacion estandar — proyeccion muy consistente con la realidad |
| Splitter | 1253 | 803 | +1.58 | -1.41 | DENTRO de 2 desviaciones estandar — proyeccion razonablemente consistente |

**27 de 28 pitches evaluables** (n>=20 pitches reales similares) caen dentro de 2 desviaciones estandar.

## 3. Investigacion: ¿el sesgo positivo de HorzBreak es sistematico?

La corrida inicial (n=5) mostro los 5 z-scores de HorzBreak positivos. Con la muestra ampliada:

| metrica | n | % positivos | media(z) | p-value (signo, binomial) | p-value (t-test, media=0) |
|---|---:|---:|---:|---:|---:|
| z(HB), sin controlar lateralidad | 28 | 39% | -0.321 | 0.345 | 0.099 |
| z(IVB), sin controlar lateralidad (control/referencia) | 28 | 43% | +0.028 | 0.572 | 0.897 |

**Hipotesis investigada — confusor de lateralidad del pitcher (NO el modelo fisico):** `HorzBreak` tiene signo OPUESTO segun `PitcherThrows` (confirmado en el dataset completo, no solo en la muestra):

| PitcherThrows   |   Changeup |   Curveball |   Cutter |   Four-Seam |   Sinker |   Slider |   Splitter |
|:----------------|-----------:|------------:|---------:|------------:|---------:|---------:|-----------:|
| Left            |     -11.76 |        7.96 |    -1.12 |       -8.05 |   -12.95 |     4.86 |      -8.84 |
| Right           |      11.58 |       -7.93 |    -0.39 |        8.33 |    14.02 |    -3.92 |       5.89 |

El chequeo de la seccion 2 comparaba la proyeccion de UN pitch (con una lateralidad fija, la del pitch de origen) contra un 'pool' de pitches reales similares que mezclaba LHP y RHP sin filtrar — como ambos tienen HorzBreak de signo opuesto, la media del pool se acerca artificialmente a 0 frente a un pitch individual con signo definido, generando un sesgo de z-score que no tiene que ver con el modelo de Magnus, sino con la metodologia de comparacion.

**Re-chequeo controlando por `PitcherThrows`** (el pool de pitches similares ahora tambien exige la misma lateralidad que el pitch de origen):

| metrica | n | % positivos | media(z) | p-value (signo, binomial) | p-value (t-test, media=0) |
|---|---:|---:|---:|---:|---:|
| z(HB), controlando lateralidad | 28 | 50% | -0.080 | 1.000 | 0.551 |
| z(IVB), controlando lateralidad (control/referencia) | 28 | 43% | +0.009 | 0.572 | 0.967 |

### Conclusion

**El sesgo NO se confirmo como sistematico con la muestra mas grande** — los 5 z-scores positivos de la corrida inicial fueron coincidencia de muestra chica, no un patron real.