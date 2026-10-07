# z-test de is_whiff por altitud, controlando por AutoPitchType

El chequeo de OOD (`reports/altitude_counterfactual_ood_check.md`) descarto que el contrafactual (is_whiff sube al mover un pitch a Extreme Altitude) sea un artefacto de extrapolacion: el % de combinaciones fuera de distribucion es del mismo orden que el que ya existe naturalmente al evaluar pitchers de test nunca vistos en train. Este reporte cierra el otro extremo de la pregunta: si el z-test observacional agregado de `reports/run_value_validation.md` (is_whiff -7.7% relativo Extreme vs No Altitude, z=-9.52) sobrevive al controlar por `AutoPitchType`, o si es composicion de tipos de pitch.

## 1. z-test DENTRO de cada AutoPitchType (Extreme Altitude vs No Altitude)

| AutoPitchType | n Extreme Altitude | rate Extreme Altitude | n No Altitude | rate No Altitude | diff | diff % relativo | z | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Four-Seam | 61,678 | 0.0829 | 99,354 | 0.1008 | -0.0178 | -17.69% | -11.92 | 0.00e+00 |
| Slider | 42,507 | 0.1502 | 59,705 | 0.1574 | -0.0072 | -4.57% | -3.14 | 1.72e-03 |
| Sinker | 31,596 | 0.0607 | 54,918 | 0.0701 | -0.0095 | -13.50% | -5.38 | 7.65e-08 |
| Changeup | 19,227 | 0.1347 | 41,287 | 0.1443 | -0.0096 | -6.67% | -3.17 | 1.54e-03 |
| Curveball | 14,374 | 0.1480 | 25,092 | 0.1404 | +0.0077 | +5.47% | 2.10 | 3.60e-02 |
| Cutter | 10,094 | 0.1066 | 9,612 | 0.1088 | -0.0022 | -2.04% | -0.50 | 6.15e-01 |
| Splitter | 2,200 | 0.1750 | 3,536 | 0.1581 | +0.0169 | +10.70% | 1.68 | 9.30e-02 |

**4 de 7** tipos de pitch principales muestran la MISMA direccion (is_whiff mas bajo en Extreme Altitude) con p<0.05 dentro del propio tipo — si el efecto agregado fuera puramente composicion, esperariamos que la mayoria de estos z-tests individuales NO fueran significativos o cambiaran de signo.

> **Nota de cautela sobre Curveball.** El patron de Curveball (direccion OPUESTA al resto: +5.47%, mas whiffs en Extreme Altitude) esta confirmado por dos metodos independientes — este z-test crudo y el analisis contrafactual de `reports/altitude_counterfactual.md` (seccion "Curveball especificamente") — lo cual le da mas credibilidad que a un hallazgo de una sola fuente. **Pero su significancia estadistica es notablemente mas debil que la del efecto agregado o que la de los tipos de pitch que dominan la muestra:** p=0.036 vs. p<0.001 en el CMH pooled (seccion 2) y vs. p<0.001 en Four-Seam. A diferencia de Four-Seam (z=-11.92, n=161k combinados) o Sinker (z=-5.38, n=86k), el z de Curveball (2.10, n=39k combinados) apenas cruza el umbral convencional de 0.05 — y con 7 tipos de pitch principales testeados aqui, un p=0.036 sin corregir por comparaciones multiples **no sobreviviria** una correccion de Bonferroni (umbral efectivo ≈0.05/7≈0.0071). Splitter, que va en la misma direccion que Curveball (+10.70%), ni siquiera alcanza p<0.05 (p=0.093). **Recomendacion para cualquier documento tecnico que cite este hallazgo:** presentar el patron de Curveball como "un patron real pero con mayor incertidumbre relativa" — reproducible en dos metodos distintos y con una direccion consistente, pero apoyado en una senal estadistica mas debil y en una muestra mas chica que el hallazgo principal (menos whiffs en altitud, que si es robusto y sobrevive multiples chequeos) — no con el mismo nivel de confianza que ese hallazgo principal.

## 2. Test de Cochran-Mantel-Haenszel (pooled, controla por AutoPitchType)

- Odds ratio pooled (Extreme Altitude vs No Altitude): **0.9055** (OR<1 = menos whiffs en la categoria de mayor altitud, dentro de cada estrato de tipo de pitch)
- Chi2 de Mantel-Haenszel (con correccion de continuidad): 107.09, p-value: 0.00e+00
- Eventos observados (whiffs en Extreme Altitude, sumado sobre estratos): 19,597 vs. esperados bajo independencia: 20,687.8

## 3. Estandarizacion directa (reponderar por mezcla GLOBAL de AutoPitchType)

Mismo metodo que `reports/magnus_analysis.md` para el residual de Magnus: recalcular is_whiff promedio de cada altitude_category reponderando con la mezcla GLOBAL fija de tipos de pitch, para aislar el efecto de composicion del efecto dentro-de-tipo.

### Composicion de AutoPitchType por altitude_category (%)

| altitude_category   |   Four-Seam |   Slider |   Sinker |   Changeup |   Curveball |   Cutter |   Splitter |
|:--------------------|------------:|---------:|---------:|-----------:|------------:|---------:|-----------:|
| No Altitude         |       33.85 |    20.34 |    18.71 |      14.07 |        8.55 |     3.27 |       1.2  |
| Medium Altitude     |       35.74 |    23.34 |    15.22 |      11.38 |        9.09 |     4.26 |       0.97 |
| Extreme Altitude    |       33.95 |    23.4  |    17.39 |      10.58 |        7.91 |     5.56 |       1.21 |

Test chi-cuadrado de independencia: chi2=4344.1, p=0.00e+00. **Cramer's V = 0.0587** (V<0.1 = asociacion despreciable segun convenciones estandar).

### is_whiff promedio: crudo vs. ajustado por composicion

| altitude_category | crudo | ajustado por composicion |
|---|---:|---:|
| No Altitude | 0.1170 | 0.1176 |
| Medium Altitude | 0.1124 | 0.1115 |
| Extreme Altitude | 0.1079 | 0.1078 |

Brecha cruda (No Altitude - Extreme Altitude): +0.0092. Brecha ajustada por composicion: +0.0098. Composicion de AutoPitchType explica ~-7.0% de la brecha cruda (si este numero es chico, el efecto es genuinamente dentro-de-tipo, no un artefacto de que Extreme Altitude tira una mezcla distinta de pitches).
