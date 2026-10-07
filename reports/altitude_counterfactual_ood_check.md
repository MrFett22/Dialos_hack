# Chequeo de out-of-distribution (OOD) — contrafactual de altitud

Verifica si el hallazgo de `reports/altitude_counterfactual.md` (is_whiff sube al mover el mismo pitch a otra altitud) es un artefacto de extrapolacion: para cada uno de los 100,000 pitches de la muestra contrafactual, se evalua si (InducedVertBreak_cf, HorzBreak_cf, RelSpeed, SpinRate) — la combinacion que se le da a `pitching_plus_v1.pkl` para predecir bajo cada escenario — cae dentro de lo que el modelo realmente vio en entrenamiento para esa combinacion (AutoPitchType, altitude_category objetivo).

**Referencia usada: el TRAIN de `pitcher_holdout`** (503,112 pitches, no el dataset completo) — es el universo exacto que ajusto `pitching_plus_v1.pkl`, asi que es la pregunta correcta para 'extrapolacion del modelo', no una pregunta general de plausibilidad fisica.

3 metodos, agrupando por (AutoPitchType, altitude_category objetivo): (1) percentiles 1-99 de train (OOD si CUALQUIERA de las 4 features cae afuera); (2) distancia de Mahalanobis (covarianza LedoitWolf) contra chi2 al 99% con 4 gl; (3) Isolation Forest (contamination='auto'). Los metodos (2)-(3) solo se ajustan para grupos con n_train>=200 (22 de 27 grupos AutoPitchType x altitude_category califican) — grupos mas chicos solo tienen el chequeo de percentiles.

## 1. % OOD por escenario contrafactual (los 3 metodos)

| escenario simulado | n | % OOD (percentiles 1-99) | % OOD (Mahalanobis, solo grupos con ref) | % OOD (Isolation Forest, solo grupos con ref) | % filas con grupo de referencia solido (n_train>=200) |
|---|---:|---:|---:|---:|---:|
| No Altitude | 100,000 | 7.02% | 1.20% | 16.30% | 99.9% |
| Medium Altitude | 100,000 | 13.81% | 2.76% | 25.58% | 99.9% |
| Extreme Altitude | 100,000 | 10.39% | 2.59% | 21.34% | 100.0% |

### Calibracion: tasa 'natural' de OOD sin contrafactual

Mismo chequeo, mismos 3 metodos, misma referencia de train — pero sobre pitches REALES (sin recalcular movimiento) de los pitchers de `pitcher_holdout` TEST (nunca vistos en train), cada uno contra su propia `altitude_category` real. Esto mide cuanto OOD aparece solo por variacion normal pitcher-a-pitcher, sin ningun cambio de altitud — el piso de comparacion para saber si el % de la seccion 1 es alto DE VERDAD o es del orden del ruido de metodo/generalizacion normal.

| n | % OOD percentiles | % OOD Mahalanobis | % OOD IsoForest |
|---:|---:|---:|---:|
| 124,496 | 8.52% | 1.74% | 18.54% |

## 2. Desglose por AutoPitchType x escenario (solo tipos con n>=100 en la muestra)

| escenario | AutoPitchType | n muestra | n_train referencia | % OOD percentiles | % OOD Mahalanobis | % OOD IsoForest |
|---|---|---:|---:|---:|---:|---:|
| Extreme Altitude | Cutter | 4,164 | 8,116 | 8.26% | 1.20% | 14.27% |
| Extreme Altitude | Slider | 21,924 | 33,559 | 9.04% | 4.81% | 15.01% |
| Extreme Altitude | Changeup | 12,391 | 15,601 | 9.95% | 1.82% | 23.02% |
| Extreme Altitude | Sinker | 17,458 | 24,661 | 10.30% | 1.98% | 23.39% |
| Extreme Altitude | Curveball | 8,490 | 11,322 | 10.60% | 2.03% | 23.64% |
| Extreme Altitude | Four-Seam | 34,318 | 50,278 | 11.31% | 2.11% | 23.89% |
| Extreme Altitude | Splitter | 1,150 | 1,856 | 20.35% | 1.13% | 26.09% |
| Medium Altitude | Slider | 21,924 | 27,740 | 10.67% | 4.44% | 20.58% |
| Medium Altitude | Sinker | 17,458 | 19,312 | 14.19% | 2.13% | 25.16% |
| Medium Altitude | Curveball | 8,490 | 11,386 | 14.24% | 2.07% | 27.31% |
| Medium Altitude | Four-Seam | 34,318 | 43,395 | 14.51% | 2.78% | 27.57% |
| Medium Altitude | Changeup | 12,391 | 14,575 | 15.08% | 1.42% | 27.61% |
| Medium Altitude | Cutter | 4,164 | 4,700 | 15.85% | 2.26% | 27.31% |
| Medium Altitude | Splitter | 1,150 | 1,144 | 21.91% | 1.04% | 26.87% |
| No Altitude | Splitter | 1,150 | 2,534 | 5.48% | 1.04% | 10.61% |
| No Altitude | Four-Seam | 34,318 | 79,118 | 5.78% | 0.98% | 14.04% |
| No Altitude | Sinker | 17,458 | 44,169 | 5.84% | 1.30% | 14.58% |
| No Altitude | Curveball | 8,490 | 20,334 | 5.91% | 1.12% | 15.72% |
| No Altitude | Changeup | 12,391 | 33,483 | 7.21% | 0.95% | 18.04% |
| No Altitude | Slider | 21,924 | 47,901 | 9.74% | 1.77% | 19.41% |
| No Altitude | Cutter | 4,164 | 7,398 | 9.94% | 0.53% | 23.25% |