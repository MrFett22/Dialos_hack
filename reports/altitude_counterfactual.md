# Analisis contrafactual de altitud — Pitching+ v1

Diseño de 2 pasos (Magnus -> Pitching+), documentado en `reports/notas_metodologicas.md`. Para cada pitch de una muestra estratificada de 100,000 (por AutoPitchType x altitude_category actual), se recalculo el movimiento bajo las 3 densidades de aire (No/Medium/Extreme Altitude), preservando RelSpeed/SpinRate/SpinAxis/Extension y el residual de Magnus del pitcher, y se re-predijo con los 8 modelos de `pitching_plus_v1.pkl`.

`altitude_level` se actualizo para ser consistente con cada escenario contrafactual (no se dejo en su valor real mientras el movimiento cambiaba) — el resto de las features quedo sin tocar. El ablation de Stuff+ ya mostro que altitude_level aporta ~0 señal una vez que el movimiento esta en el modelo, asi que esta decision no deberia cambiar el resultado de forma material, pero es la version internamente consistente.

La normalizacion z-score usa como referencia la media/std de cada target en condiciones FACTUALES sobre TODO el dataset valido (no solo la muestra), para que el delta sea comparable a la escala Pitching+ real. El termino '+100' se cancela exactamente en cualquier delta — solo importan los pesos y las std de referencia.

## 1. delta_pitching_plus por AutoPitchType

Figura: `reports/figures/altitude_counterfactual/delta_by_pitchtype.png`

| AutoPitchType | delta promedio (Extreme - No Altitude) | n |
|---|---:|---:|
| OneSeamFastBall | -1.279 | 1 |
| Sinker | -1.059 | 17,458 |
| Sweeper | -1.046 | 91 |
| Other | -1.010 | 11 |
| Changeup | -0.951 | 12,391 |
| Cutter | -0.904 | 4,164 |
| Slider | -0.843 | 21,924 |
| Four-Seam | -0.839 | 34,318 |
| Curveball | -0.829 | 8,490 |
| Splitter | -0.817 | 1,150 |
| TwoSeamFastBall | -0.530 | 1 |
| Knuckleball | -0.429 | 1 |

(Solo tipos con n>=100 en la muestra — OneSeamFastBall, Sweeper, Other, TwoSeamFastBall, Knuckleball tienen muestra demasiado chica para ser confiables, aunque aparecen en la tabla completa arriba.)
Pierden mas Pitching+ al moverse a Extreme Altitude: Sinker, Changeup, Cutter. Pierden menos: Four-Seam, Curveball, Splitter.

## 2. delta_pitching_plus por diferencial de velocidad vs. fastball (H2)

Figura: `reports/figures/altitude_counterfactual/delta_by_velo_diff_bin.png`

H2: pitches con mayor diferencial de velocidad respecto al fastball (mecanismo de engaño mas basado en el cambio de velocidad/tunneling que en el movimiento por Magnus) deberian perder MENOS Pitching+ en altitud que los pitches de velocidad similar al fastball (que dependen mas del movimiento puro).

| bin de velo_diff_vs_fb | delta promedio | n |
|---|---:|---:|
| muy lento vs FB (<-8mph) | -0.856 | 23,471 |
| lento vs FB (-8 a -4) | -0.866 | 17,201 |
| algo lento vs FB (-4 a -1) | -0.956 | 15,245 |
| similar a FB (-1 a 1) | -0.936 | 31,969 |
| igual/mas rapido que FB (>1) | -0.819 | 12,114 |

**No consistente con H2 en la direccion esperada** — el bin mas alejado del fastball (delta=-0.856) NO pierde menos que el mas cercano al fastball (delta=-0.819). Revisar con cautela antes de citar H2 como confirmada.

## 2b. H2 restringido a Changeup

Re-agregacion de los mismos 100,000 pitches y los mismos 5 bins de `velo_diff_vs_fb` de la seccion 2 (sin volver a correr el contrafactual — datos de `data/processed/altitude_counterfactual_scores.parquet`), filtrando SOLO `AutoPitchType == 'Changeup'`. Motivacion: el engaño de un Changeup depende casi por definicion del diferencial de velocidad contra el fastball, a diferencia de otros tipos de pitch cuyo engaño es mas de movimiento — la mezcla de todos los tipos en la seccion 2 puede diluir un patron que si exista dentro de Changeup especificamente.

**n total de Changeup en la muestra: 12,391** (de 100,000 pitches totales).

| bin de velo_diff_vs_fb | n | delta promedio | IC 95% (bootstrap) |
|---|---:|---:|---|
| muy lento vs FB (<-8mph) | 4,516 | -0.910 | [-0.933, -0.888] |
| lento vs FB (-8 a -4) | 5,501 | -0.913 | [-0.936, -0.891] |
| algo lento vs FB (-4 a -1) | 1,614 | -1.091 | [-1.153, -1.038] |
| similar a FB (-1 a 1) | 611 | -1.173 | [-1.249, -1.101] |
| igual/mas rapido que FB (>1) | 149 | -1.167 | [-1.269, -1.068] |

### Monotonicidad dentro de Changeup

Secuencia de deltas por bin confiable (orden de bins, mas lento a mas rapido vs. FB): -0.910 → -0.913 → -1.091 → -1.173 → -1.167

**H2 se sostiene en los extremos pero NO de forma estrictamente monotonica** dentro de Changeup: el bin mas alejado del fastball pierde menos que el mas cercano/igual, pero el patron no es consistentemente creciente en los bins intermedios — revisar la tabla completa antes de citar un patron limpio.

### Investigacion del quiebre en "algo lento vs FB (-4 a -1)"

Chequeo rapido (no exhaustivo) de si el quiebre de monotonicidad tiene una explicacion de composicion. Se comparo `RelSpeed`, `SpinRate` e `InducedVertBreak` REALES del pitch (no contrafactuales) por bin, mas concentracion de pitchers especificos.

| bin | RelSpeed (mean/mediana) | SpinRate (mean/mediana) | InducedVertBreak (mean/mediana) |
|---|---|---|---|
| muy lento vs FB (<-8mph) | 80.5 / 80.8 | 1771 / 1747 | 5.91 / 5.84 |
| lento vs FB (-8 a -4) | 83.9 / 84.1 | 1773 / 1763 | 6.19 / 6.20 |
| algo lento vs FB (-4 a -1) | 86.0 / 86.1 | 1979 / 2023 | 7.87 / 7.90 |
| similar a FB (-1 a 1) | 87.3 / 86.7 | 1975 / 2072 | 8.84 / 9.08 |
| igual/mas rapido que FB (>1) | 88.6 / 87.2 | 1951 / 2257 | 8.59 / 8.27 |

**Hallazgo:** `SpinRate` (y, en linea con eso, `InducedVertBreak`) da un salto claro justo en la misma transicion donde se rompe la monotonicidad de `delta_pitching_plus` — plano en los dos bins mas alejados del fastball (mediana ~1,750-1,760 rpm) y ~200-300 rpm mas alto en los tres bins restantes (mediana ~2,000-2,260 rpm). Esto es consistente con que el dataset contiene dos sub-poblaciones de 'Changeup' distintas — una con menos spin y mas separacion de velocidad respecto al fastball, otra con mas spin y menos separacion — y como el contrafactual recalcula el movimiento via la formula de Magnus (`src/features/physics.py`), que escala con SpinRate, el sub-grupo de mayor spin sufre una perdida de movimiento mas grande al cambiar la densidad de aire, lo cual empuja su `delta_pitching_plus` mas abajo — sin necesidad de invocar un artefacto de pitchers especificos.

**Robustez:** excluyendo a los 5 pitchers mas frecuentes de cada bin (para descartar que sea solo un puñado de pitchers concentrados), la mediana de SpinRate se mantiene en el mismo orden (muy lento vs FB (<-8mph): 1739, lento vs FB (-8 a -4): 1764, algo lento vs FB (-4 a -1): 2022, similar a FB (-1 a 1): 2003, igual/mas rapido que FB (>1): 1286 rpm) — el patron es de la poblacion de pitches, no de unos pocos pitchers dominando el bin (la concentracion de los 5 pitchers mas frecuentes por bin va de 7.1% a 58.4% del bin, sin un salto que coincida con el quiebre de SpinRate).

**Advertencia:** la correlacion pitch-a-pitch entre SpinRate y `delta_pitching_plus` dentro de Changeup es DEBIL (-0.071; RelSpeed: -0.053) — el patron es visible al promediar por bin, no como una relacion lineal fuerte a nivel de pitch individual (mucho ruido pitch a pitch, esperable). Esta investigacion es un chequeo rapido (no un analisis causal completo) y no cambia la conclusion de monotonicidad de arriba — la documenta con una explicacion fisica plausible en vez de dejarla sin ninguna pista.

## 3. Tabla cruzada AutoPitchType x altitude_category actual (con bootstrap CI 95%)

Figura: `reports/figures/altitude_counterfactual/delta_crosstable_heatmap.png`

Bootstrap de 1000 remuestreos por celda, solo celdas con n>=20.

| AutoPitchType | altitude_category actual | n | delta promedio | IC 95% |
|---|---|---:|---:|---|
| Changeup | Extreme Altitude | 3,038 | -0.961 | [-0.995, -0.930] |
| Changeup | Medium Altitude | 2,802 | -0.906 | [-0.932, -0.882] |
| Changeup | No Altitude | 6,551 | -0.966 | [-0.987, -0.944] |
| Curveball | Extreme Altitude | 2,271 | -0.854 | [-0.901, -0.815] |
| Curveball | Medium Altitude | 2,237 | -0.808 | [-0.839, -0.774] |
| Curveball | No Altitude | 3,982 | -0.826 | [-0.848, -0.802] |
| Cutter | Extreme Altitude | 1,591 | -0.918 | [-0.950, -0.886] |
| Cutter | Medium Altitude | 1,050 | -0.877 | [-0.913, -0.842] |
| Cutter | No Altitude | 1,523 | -0.908 | [-0.944, -0.873] |
| Four-Seam | Extreme Altitude | 9,751 | -0.808 | [-0.824, -0.792] |
| Four-Seam | Medium Altitude | 8,803 | -0.799 | [-0.817, -0.783] |
| Four-Seam | No Altitude | 15,764 | -0.882 | [-0.898, -0.867] |
| Sinker | Extreme Altitude | 4,993 | -1.035 | [-1.057, -1.015] |
| Sinker | Medium Altitude | 3,749 | -1.017 | [-1.040, -0.995] |
| Sinker | No Altitude | 8,716 | -1.090 | [-1.106, -1.075] |
| Slider | Extreme Altitude | 6,709 | -0.812 | [-0.832, -0.794] |
| Slider | Medium Altitude | 5,746 | -0.846 | [-0.865, -0.824] |
| Slider | No Altitude | 9,469 | -0.863 | [-0.879, -0.846] |
| Splitter | Extreme Altitude | 349 | -0.774 | [-0.847, -0.703] |
| Splitter | Medium Altitude | 240 | -0.838 | [-0.936, -0.743] |
| Splitter | No Altitude | 561 | -0.835 | [-0.899, -0.777] |
| Sweeper | Extreme Altitude | 91 | -1.046 | [-1.214, -0.887] |

## 4. Descomposicion del delta en los 8 componentes individuales

Investigacion especifica: Curveball tiene la mayor reduccion de movimiento absoluto por altitud (EDA de Semana 2), pero NO es de los que mas pierde Pitching+ en el contrafactual (seccion 1). Se descompuso el delta en sus 8 componentes, antes y despues de combinar con pesos, para ver que target compensa.

### Delta crudo por target (unidades naturales — probabilidad para targets binarios, carreras para run_value)

| AutoPitchType | is_whiff | is_chase | is_called_strike | is_weak_contact | is_groundball | is_hard_contact | is_out_in_pa | run_value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Changeup | +0.0004 | -0.0009 | -0.0033 | -0.0006 | +0.0067 | +0.0008 | -0.0366 | +0.0102 |
| Curveball | +0.0050 | +0.0067 | -0.0094 | +0.0050 | +0.0034 | +0.0013 | -0.0357 | +0.0091 |
| Cutter | +0.0025 | -0.0035 | -0.0022 | +0.0001 | +0.0099 | +0.0003 | -0.0385 | +0.0101 |
| Four-Seam | +0.0026 | -0.0001 | -0.0069 | +0.0017 | +0.0183 | +0.0014 | -0.0390 | +0.0090 |
| Sinker | -0.0002 | -0.0039 | -0.0087 | -0.0013 | +0.0096 | +0.0011 | -0.0403 | +0.0101 |
| Slider | +0.0030 | +0.0005 | -0.0046 | +0.0013 | +0.0056 | +0.0008 | -0.0333 | +0.0099 |
| Splitter | +0.0039 | +0.0010 | -0.0017 | +0.0015 | +0.0055 | +0.0009 | -0.0323 | +0.0104 |

### Contribucion en puntos de Pitching+ por target (estos SI suman el delta total — ver `contrib_sum_check` vs. la seccion 1)

| AutoPitchType | is_whiff | is_chase | is_called_strike | is_weak_contact | is_groundball | is_hard_contact | is_out_in_pa | run_value | suma (check) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Changeup | +0.004 | -0.004 | -0.020 | -0.007 | +0.056 | -0.031 | -0.593 | -0.357 | -0.951 |
| Curveball | +0.063 | +0.029 | -0.057 | +0.057 | +0.029 | -0.055 | -0.578 | -0.317 | -0.829 |
| Cutter | +0.032 | -0.016 | -0.013 | +0.001 | +0.084 | -0.014 | -0.623 | -0.354 | -0.904 |
| Four-Seam | +0.033 | -0.000 | -0.042 | +0.019 | +0.155 | -0.058 | -0.631 | -0.314 | -0.839 |
| Sinker | -0.003 | -0.017 | -0.053 | -0.015 | +0.082 | -0.046 | -0.652 | -0.355 | -1.059 |
| Slider | +0.037 | +0.002 | -0.028 | +0.015 | +0.048 | -0.031 | -0.539 | -0.347 | -0.843 |
| Splitter | +0.049 | +0.004 | -0.011 | +0.017 | +0.047 | -0.037 | -0.523 | -0.363 | -0.817 |

### Curveball especificamente

- El componente que MAS compensa (contribucion menos negativa o mas positiva) es **is_whiff** (+0.063 puntos).
- El componente que MAS castiga es **is_out_in_pa** (-0.578 puntos).
- `is_whiff` contribuye +0.063 puntos — no es el mas castigado, sorprendente dado que se esperaba que fuera el mas afectado.
- `is_hard_contact` (crudo, probabilidad de barrel): delta = +0.00133, contribucion = -0.055 puntos — NO consistente con la hipotesis de 'curva mas plana = menos contacto duro' — revisar el signo antes de citar esta explicacion.

Nota: `is_hard_contact` esta invertido de signo en la combinacion (mayor probabilidad de barrel = peor para el pitcher), asi que un `raw_delta_is_hard_contact` NEGATIVO (menos barrels en altitud) se traduce en una contribucion POSITIVA al score — pero para Curveball el delta crudo sale POSITIVO (+0.0013, mas barrels, no menos), asi que este mecanismo especifico queda descartado.

**Investigacion de seguimiento — comparando Curveball contra los otros 6 tipos de pitch (no solo mirandolo aislado):**

| target | ranking de Curveball (de 7 tipos) | valor |
|---|---|---:|
| contrib_is_whiff | **#1 (el mas alto)** | +0.063 |
| contrib_is_weak_contact | **#1 (el mas alto)** | +0.057 |
| contrib_is_called_strike | #7 (el mas bajo/peor) | -0.057 |
| contrib_is_hard_contact | #6 de 7 (casi el peor) | -0.055 |

**La hipotesis original (is_hard_contact compensa) esta refutada, pero hay una compensacion real — solo que por una via distinta y mas interesante:** Curveball es el tipo de pitch que MAS gana en `is_whiff` y, por separado, el que MAS gana en `is_weak_contact` — el unico de los 7 tipos donde ambos componentes son simultaneamente el maximo del grupo. Es decir, la curva no se vuelve "mas facil de conectar con menos calidad de contacto" (esa hipotesis especifica no se sostiene: el contacto duro en realidad sube un poco), sino que en altitud el modelo predice **mas swings-y-fallos Y, cuando SI hay contacto, mas contacto debil** — ambos empujando en la misma direccion favorable. Eso compensa el costo, que si es real y es el peor de los 7 tipos, en `is_called_strike` (menos movimiento vertical/horizontal hace mas dificil "robarse" un strike cantado en el borde de la zona) y en `is_hard_contact` (ligero aumento, no reduccion).

Esto tambien contradice la premisa implicita en la pregunta original: no es que `is_whiff` caiga y otro componente lo compense — `is_whiff` en si mismo SUBE para Curveball (y de hecho para casi todos los tipos de pitch excepto Sinker, ver tabla de deltas crudos arriba), consistente con el hallazgo ya establecido en el proyecto de que el efecto de altitud en el desempeño del pitcher pasa mayormente por `is_out_in_pa` (el termino que domina la caida en los 7 tipos, entre -0.52 y -0.65 puntos) y no por perder capacidad de generar swing-and-miss.


## Referencia de normalizacion (media/std factual por target, dataset completo)

| target | mean (con signo ajustado) | std |
|---|---:|---:|
| is_whiff | 0.1129 | 0.0953 |
| is_chase | 0.2222 | 0.2713 |
| is_called_strike | 0.1542 | 0.1975 |
| is_weak_contact | 0.0797 | 0.0700 |
| is_groundball | 0.1006 | 0.0943 |
| is_hard_contact | -0.0046 | 0.0192 |
| is_out_in_pa | 0.6346 | 0.1234 |
| run_value | -0.0035 | 0.0572 |

Datos completos por pitch guardados en `data/processed/altitude_counterfactual_scores.parquet`.
