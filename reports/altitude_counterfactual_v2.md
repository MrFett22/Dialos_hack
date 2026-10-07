# Contrafactual de altitud v2 — trayectoria completa + Pitching+ v3

Generado por `src/analysis/altitude_counterfactual_v2.py` (física en `src/features/trajectory.py`). Reemplaza al contrafactual v1 (`reports/altitude_counterfactual.md`). Pitches: 627,736.

## 1. Densidad del aire medida con los datos (no supuesta)

Drag/v² y Magnus/v² son proporcionales a la densidad del aire. Comparando al **mismo pitcher, mismo tipo de pitch y misma temporada** entre categorías (≥15 pitches por celda), la mediana de los ratios estima la densidad efectiva de cada categoría relativa a No Altitude.

| categoría | ratio drag | ratio Magnus | grupos | elevación equivalente (drag) | supuesto v1 |
|---|---:|---:|---:|---|---:|
| Medium Altitude | 0.906 | 0.834 | 1,832 | ≈884 m | 0.930 |
| Extreme Altitude | 0.854 | 0.770 | 2,165 | ≈1,385 m | 0.834 |

Por tipo de pitch:

| categoría | tipo | ratio drag | ratio Magnus | grupos |
|---|---|---:|---:|---:|
| Medium Altitude | Changeup | 0.904 | 0.791 | 243 |
| Medium Altitude | Curveball | 0.907 | 0.821 | 162 |
| Medium Altitude | Cutter | 0.923 | 0.868 | 74 |
| Medium Altitude | Four-Seam | 0.888 | 0.842 | 587 |
| Medium Altitude | Sinker | 0.893 | 0.834 | 309 |
| Medium Altitude | Slider | 0.931 | 0.840 | 443 |
| Extreme Altitude | Changeup | 0.854 | 0.712 | 271 |
| Extreme Altitude | Curveball | 0.861 | 0.740 | 170 |
| Extreme Altitude | Cutter | 0.880 | 0.821 | 80 |
| Extreme Altitude | Four-Seam | 0.836 | 0.781 | 710 |
| Extreme Altitude | Sinker | 0.836 | 0.764 | 384 |
| Extreme Altitude | Slider | 0.895 | 0.793 | 533 |

**Lectura.** Si solo cambiara la densidad del aire, drag y Magnus bajarían en la misma proporción. El Magnus baja bastante más, sobre todo en Medium Altitude. Eso apunta a algo más que densidad: calibración de Trackman por estadio, clima o la pelota. Es consistente con el residual vertical anómalo de Medium Altitude que v1 dejó sin explicar (`reports/magnus_analysis.md`). Para el contrafactual se usan los ratios medidos de cada componente, que reproducen lo que el modelo realmente ve.

## 2. Validación split-half de la re-simulación

Ratios estimados con la mitad de los pitchers. Con la **otra mitad**, se proyectan sus pitches de No Altitude a la categoría destino y se compara el cambio predicho contra el cambio observado en esos mismos pitchers.

| categoría | feature | grupos | cambio observado | cambio predicho | corr. entre grupos |
|---|---|---:|---:|---:|---:|
| Medium Altitude | InducedVertBreak | 967 | -1.677 | -1.835 | 0.747 |
| Medium Altitude | HorzBreak | 967 | -1.261 | -1.085 | 0.837 |
| Medium Altitude | ZoneSpeed | 967 | +0.903 | +0.762 | 0.116 |
| Medium Altitude | VertApprAngle | 967 | -0.078 | -0.116 | 0.417 |
| Medium Altitude | ZoneTime | 967 | -0.003 | -0.002 | 0.023 |
| Medium Altitude | VertBreak | 967 | -1.299 | -1.546 | 0.630 |
| Extreme Altitude | InducedVertBreak | 1061 | -2.201 | -2.638 | 0.806 |
| Extreme Altitude | HorzBreak | 1061 | -1.817 | -1.503 | 0.897 |
| Extreme Altitude | ZoneSpeed | 1061 | +1.244 | +1.202 | 0.143 |
| Extreme Altitude | VertApprAngle | 1061 | -0.165 | -0.156 | 0.481 |
| Extreme Altitude | ZoneTime | 1061 | -0.004 | -0.003 | 0.049 |
| Extreme Altitude | VertBreak | 1061 | -1.640 | -2.157 | 0.703 |

El contrafactual v1 predecía cambio **cero** en ZoneSpeed, VAA, ZoneTime y VertBreak, porque no los recalculaba.

**Identidad:** proyectar cada pitch a su propia categoría cambia las features como máximo en 1.00e-05 (error de punto flotante).

## 3. Brecha observada entre categorías vs. lo que atribuye el modelo

Carreras por 100 pitches (positivo = más carreras permitidas). *Realizado* = valor en carreras del evento que ocurrió (misma escala Δ por conteo de Pitching+ v3). *Predicho* = E[Δ] de Pitching+ v3 con las features observadas.

| categoría | realizado | predicho (obs.) | proyectado a No Altitude | proyectado a Extreme |
|---|---:|---:|---:|---:|
| No Altitude | -0.720 | -0.510 | -0.522 | +0.608 |
| Medium Altitude | +0.471 | +0.244 | -0.502 | +0.573 |
| Extreme Altitude | +0.762 | +0.585 | -0.452 | +0.603 |

Brecha Extreme − No: realizada **+1.48**, predicha por el modelo **+1.10** carreras/100 pitches. Pitching+ v3 no tiene ninguna variable de altitud, pero **la detecta por la firma física de la trayectoria** (drag y Magnus más bajos) y le atribuye casi toda la diferencia entre parques, incluido el entorno ofensivo (vuelo de la pelota, dimensiones). Proyectar un pitch a otra categoría mueve esa firma, así que el contrafactual captura el **efecto total** del entorno de altitud, no solo la pérdida de calidad del pitch. La sección 5 los separa.

## 4. Efecto total de la altitud en carreras (Pitching+ v3, Extreme − No Altitude)

Cada pitch se proyecta a ambas categorías (mismo pitcher, misma salida, mismo objetivo en el plato) y se predice E[Δ carreras] con los modelos finales de Pitching+ v3. Positivo = más carreras permitidas. IC 95% por bootstrap de pitchers (1,000 réplicas).

| tipo de pitch | carreras/100 pitches [IC 95%] | Δ Pitching+ v3 | abridor/temporada | relevista/temporada | v1: Δ Pitching+ v1 | v1: abridor |
|---|---|---:|---:|---:|---:|---:|
| Four-Seam | +1.448 [+1.414, +1.479] | -19.92 | +39.82 | +13.03 | -0.839 | +3.58 |
| Sinker | +1.199 [+1.173, +1.223] | -16.49 | +32.97 | +10.79 | -1.059 | +4.52 |
| Changeup | +1.143 [+1.116, +1.168] | -15.72 | +31.44 | +10.29 | -0.951 | +4.06 |
| Splitter | +0.904 [+0.845, +0.952] | -12.43 | +24.85 | +8.13 | -0.817 | +3.49 |
| Curveball | +0.756 [+0.721, +0.791] | -10.39 | +20.78 | +6.80 | -0.829 | +3.54 |
| Cutter | +0.682 [+0.654, +0.709] | -9.38 | +18.76 | +6.14 | -0.904 | +3.86 |
| Slider | +0.651 [+0.639, +0.664] | -8.96 | +17.91 | +5.86 | -0.843 | +3.60 |
| **todos** | +1.095 [+1.077, +1.112] | -15.06 | +30.10 | +9.85 | | |

Conversión a puntos de Pitching+ v3: 10 puntos = 1 desviación estándar del E[Δ] promedio por pitcher-temporada → 1,375 puntos por carrera/pitch. A diferencia de v1, la conversión a carreras por temporada es directa (el score ya está en carreras), no una regresión circular.

## 5. ¿Por dónde pasa el efecto? Atribución por eslabón de la cadena

El cambio en E[Δ] se reparte entre los 5 eslabones de la cadena con valores de Shapley (`src/models/chain_attribution.py`): promedia, sobre todos los órdenes posibles, cuánto cambia E[Δ] al pasar cada eslabón de No Altitude a Extreme. Los componentes suman exactamente el efecto total. Es una **atribución contable entre dos predicciones del modelo, no una descomposición causal**.

- **Antes del contacto** = swing + take + whiff + foul: lo que el pitch provoca hasta que la pelota queda en juego, incluido el aumento de la probabilidad de contacto. Es la parte atribuible al pitch con más confianza.
- **Bola en juego** = cambio en el resultado del contacto (out/1B/2B/3B/HR). Mezcla calidad del contacto con el entorno del parque (la pelota vuela más en altitud); con este modelo no se pueden separar.

La versión anterior de esta sección llamaba 'calidad del pitch' a take + whiff + foul por **grupo de eventos**. Eso omitía que menos whiffs implican más pelotas en juego: ese efecto, causado por el pitch, quedaba contado como 'bola en juego' (auditoría 2026-10-05, C3).

| tipo de pitch | antes del contacto, carreras/100 [IC 95%] | abridor/temporada | bola en juego, carreras/100 | abridor/temporada |
|---|---|---:|---:|---:|
| Four-Seam | +0.520 [+0.502, +0.537] | +14.30 | +0.928 | +25.52 |
| Sinker | +0.340 [+0.328, +0.352] | +9.35 | +0.859 | +23.62 |
| Changeup | +0.414 [+0.397, +0.430] | +11.39 | +0.729 | +20.05 |
| Splitter | +0.400 [+0.347, +0.453] | +11.00 | +0.504 | +13.85 |
| Curveball | +0.324 [+0.293, +0.353] | +8.90 | +0.432 | +11.88 |
| Cutter | +0.170 [+0.158, +0.183] | +4.67 | +0.513 | +14.09 |
| Slider | +0.270 [+0.260, +0.281] | +7.43 | +0.381 | +10.48 |

Detalle por eslabón (carreras/100 pitches, Shapley):

| tipo de pitch | decisión de swing | resultado del take (bola / strike cantado / golpeado) | whiff dado swing | foul dado contacto | resultado de la bola en juego (calidad del contacto + parque) | total |
|---|---:|---:|---:|---:|---:|---:|
| Four-Seam | +0.191 | +0.028 | +0.164 | +0.137 | +0.928 | +1.448 |
| Sinker | +0.211 | +0.046 | +0.044 | +0.039 | +0.859 | +1.199 |
| Changeup | +0.106 | +0.017 | +0.245 | +0.046 | +0.729 | +1.143 |
| Splitter | +0.124 | +0.010 | +0.233 | +0.033 | +0.504 | +0.904 |
| Curveball | +0.109 | -0.005 | +0.187 | +0.033 | +0.432 | +0.756 |
| Cutter | +0.095 | +0.004 | +0.056 | +0.014 | +0.513 | +0.682 |
| Slider | +0.131 | +0.008 | +0.120 | +0.010 | +0.381 | +0.651 |

Mecanismo (Extreme − No Altitude, promedio por pitch): cambio físico del pitch y cambio en las probabilidades de la cadena. Los cambios físicos son los del contrafactual (mismo pitch, mismo objetivo).

| tipo de pitch | IVB (pulg.) | HB (pulg.) | ZoneSpeed (mph) | VAA (°) | P(swing) (pp) | P(whiff dado swing) (pp) | P(en juego) (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Four-Seam | -4.04 | -1.31 | +1.37 | -0.289 | -0.32 | -2.29 | +2.09 |
| Sinker | -2.66 | -2.09 | +1.33 | -0.157 | -0.60 | -0.62 | +0.63 |
| Changeup | -2.33 | -1.61 | +1.08 | -0.116 | -0.63 | -3.24 | +1.18 |
| Splitter | -1.15 | -1.58 | +1.14 | +0.003 | -0.61 | -2.29 | +0.98 |
| Curveball | +2.08 | +0.91 | +0.96 | +0.315 | +2.31 | -1.79 | +1.91 |
| Cutter | -2.07 | +0.11 | +0.91 | -0.124 | -0.65 | -0.83 | +0.23 |
| Slider | -0.75 | +0.39 | +0.78 | +0.006 | -0.13 | -1.41 | +0.62 |

Lo que **no** se puede separar con estos modelos: dentro de 'bola en juego', la parte que se debe a que el pitch provoca peor contacto de la que se debe a que la pelota vuela más en el parque.
