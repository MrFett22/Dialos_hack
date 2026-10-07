# Modelo bayesiano jerarquico (PyMC) — run_value con shrinkage pitcher-dentro-de-cluster

## 0. Decisiones de diseño

**Unidad modelada: `run_value` a nivel de PITCH**, no el score Pitching+ a nivel pitcher-temporada. La jerarquia de 3 niveles pedida (pitch -> pitcher -> cluster) necesita el nivel de pitch para que el shrinkage sea real: lo que se encoge es el promedio de un pitcher estimado con mas o menos precision segun su N real de pitches, y eso solo se modela bien con una verosimilitud pitch a pitch (ver docstring de este script para el detalle completo).

**Nesting real:** un pitcher lanza en varios clusters a la vez (su arsenal mezcla fastballs/breaking/offspeed), asi que el nivel intermedio de la jerarquia es la combinacion **(pitcher_anon_id, cluster)** — cada una con su propio intercepto `alpha_pc`, encogido hacia la media de SU cluster (`mu_cluster`). Esto responde directamente 'que tan bueno es el repertorio de fastballs de este pitcher' en vez de forzar a cada pitcher a un unico cluster.

**Altitud:** `altitude_level` (0/1/2) entra con un coeficiente fijo global (`beta_altitude_global`) y una desviacion aleatoria por cluster no-centrada (`beta_altitude_cluster[c] = beta_altitude_global + sigma_altitude_cluster * offset[c]`) — el coeficiente que realmente multiplica `altitude_level` en la verosimilitud es el de cluster, parcialmente pooled hacia el efecto global.

## 1. Datos y muestra

- Pitches validos (mismo filtro que `pitcher_holdout != excluded` de `data/processed/splits/splits.parquet`, mas exclusion de run_value nulo y de AutoPitchType fuera de los 3 clusters): **624,115**
- Muestra usada para el modelo: **180,018** pitches (28.8% del universo valido)
- Pitchers con <= 150 pitches validos: **292** — se incluyeron TODOS sus pitches (no se sub-muestrearon), justamente porque son el caso de uso de interes. El resto de los pitchers se sub-muestreo estratificado por (pitcher_anon_id, altitude_category) para llegar al total.
- Grupos (pitcher_anon_id x cluster) en el modelo: **3,008**

## 2. Sampling y diagnosticos de convergencia

NUTS, 4 cadenas, 1000 draws + 1000 tuning cada una (target_accept=0.95). Tiempo de sampling: 67.3 min.

- Divergencias: **11** de 4,000 draws post-tuning (0.275%)
- r_hat maximo (sobre 3,020 parametros, incluye los 3,008 `alpha_pc`): **1.0100** (0 parametros con r_hat > 1.01)
- ess_bulk minimo: **1174** (0 parametros con ess_bulk < 400)
- ess_tail minimo: **643** (0 parametros con ess_tail < 400)

**Convergencia OK (r_hat/ESS).** Los 3,020 parametros chequeados (incluyendo los 3,008 interceptos `alpha_pc`) cumplen r_hat<1.01 y ess_bulk/ess_tail>=400 — sin excepciones. Los resultados de las secciones siguientes se reportan con confianza en la muestra posterior.

**Nota sobre divergencias:** hubo 11 divergencias en 4,000 draws post-tuning (0.275%) — una tasa baja (bien por debajo del 1% que suele usarse como referencia informal de alarma) y consistente con r_hat/ESS perfectos en todos los parametros, asi que no invalida el resultado. Se deja documentado por transparencia, no como problema sin resolver: si se quisiera eliminarlas del todo, subir `target_accept` por encima de 0.95 o `max_treedepth` por encima de 12 probablemente las resuelve, a costa de mas tiempo de sampling.

Trace plots (parametros de nivel superior): `reports/figures/bayesian_hierarchical/trace_top_level.png`

### Resumen posterior — parametros de nivel superior

|                                  |   mean |    sd |   hdi_3% |   hdi_97% |   r_hat |   ess_bulk |   ess_tail |
|:---------------------------------|-------:|------:|---------:|----------:|--------:|-----------:|-----------:|
| mu_global                        | -0.005 | 0.008 |   -0.022 |     0.011 |       1 |       2652 |       2476 |
| sigma_cluster                    |  0.014 | 0.007 |    0.004 |     0.027 |       1 |       3252 |       3000 |
| mu_cluster[fastballs]            |  0.004 | 0.001 |    0.002 |     0.007 |       1 |       3782 |       3248 |
| mu_cluster[breaking]             | -0.014 | 0.002 |   -0.017 |    -0.011 |       1 |       4454 |       3966 |
| mu_cluster[offspeed]             | -0.007 | 0.002 |   -0.011 |    -0.003 |       1 |       3805 |       3477 |
| sigma_pc                         |  0.01  | 0.001 |    0.007 |     0.012 |       1 |       1255 |       1723 |
| beta_altitude_global             |  0.009 | 0.002 |    0.004 |     0.013 |       1 |       1412 |        643 |
| sigma_altitude_cluster           |  0.002 | 0.003 |    0     |     0.007 |       1 |       1174 |        879 |
| beta_altitude_cluster[fastballs] |  0.009 | 0.001 |    0.007 |     0.011 |       1 |       4213 |       3423 |
| beta_altitude_cluster[breaking]  |  0.009 | 0.001 |    0.007 |     0.011 |       1 |       4440 |       3458 |
| beta_altitude_cluster[offspeed]  |  0.008 | 0.002 |    0.005 |     0.011 |       1 |       4272 |       3290 |
| sigma_obs                        |  0.275 | 0     |    0.275 |     0.276 |       1 |       9235 |       2866 |

## 3. Efecto de altitud sobre run_value, por cluster (IC 94%)

`beta_altitude_cluster[c]` = runs esperados adicionales por pitch, por cada nivel de `altitude_level` (0->1->2), para el cluster c. Multiplicado por 2 da el delta total No Altitude -> Extreme Altitude, comparable directamente contra el contrafactual (seccion 4).

| cluster | mean (por nivel) | IC 94% (por nivel) | mean x2 (No->Extreme) | IC 94% x2 |
|---|---:|---|---:|---|
| fastballs | +0.00908 | [+0.00732, +0.01078] | +0.01817 | [+0.01465, +0.02156] |
| breaking | +0.00920 | [+0.00697, +0.01139] | +0.01839 | [+0.01394, +0.02279] |
| offspeed | +0.00844 | [+0.00512, +0.01105] | +0.01688 | [+0.01024, +0.02211] |

Efecto fijo global (referencia, pooled entre clusters): +0.00870 por nivel, IC 94% [+0.00415, +0.01261].

## 4. Shrinkage: pitchers de muestra chica (n<100), crudo vs. jerarquico

Para cada (pitcher_anon_id, cluster) con n<100 pitches en la muestra, se compara el promedio crudo de `run_value` (sin pooling, solo esos pitches) contra la media posterior de `alpha_pc` (el intercepto jerarquico, que combina la señal propia del pitcher con la media de su cluster segun cuanta muestra tiene). `shrinkage_pct` = que tanto de la distancia entre el crudo y la media del cluster se cerro el shrinkage (100% = colapso total a la media del cluster, 0% = sin shrinkage).

| pitcher_anon_id | cluster | n pitches | crudo (sin shrinkage) | jerarquico (shrunk) | media del cluster | shrinkage % |
|---|---|---:|---:|---:|---:|---:|
| pitcher_00843 | offspeed | 1 | -0.1574 | -0.0071 | -0.0068 | 99.9% |
| pitcher_00021 | offspeed | 6 | -0.0863 | -0.0075 | -0.0068 | 99.2% |
| pitcher_00085 | offspeed | 13 | +0.0562 | -0.0058 | -0.0068 | 98.4% |
| pitcher_00384 | offspeed | 21 | -0.0054 | -0.0069 | -0.0068 | 98.5% |
| pitcher_00673 | breaking | 31 | +0.1637 | -0.0067 | -0.0137 | 96.1% |
| pitcher_01122 | breaking | 44 | -0.0745 | -0.0175 | -0.0137 | 93.8% |
| pitcher_00196 | fastballs | 66 | +0.0927 | +0.0111 | +0.0042 | 92.1% |
| pitcher_00103 | offspeed | 99 | +0.0037 | -0.0069 | -0.0068 | 99.5% |

**Lectura esperada:** a menor n_pitches, mayor shrinkage_pct (mas peso a la media del cluster, menos al promedio crudo, propio ruidoso de pocos pitches). La tabla muestra esa tendencia general (99.9% en n=1 -> 92.1% en n=66), aunque no es estrictamente monotonica fila a fila (n=99 sale con 99.5%, mas alto que n=66) — `shrinkage_pct` es una metrica RELATIVA (que tanto se cerro la distancia al promedio del cluster), asi que un pitcher cuyo promedio crudo ya cayo por azar cerca de la media de su cluster (como `pitcher_00103`, distancia cruda de solo 0.0105) puede mostrar un % alto sin que eso signifique mas shrinkage real en terminos absolutos — el patron de fondo (menos muestra = mas peso a la media del cluster) es real y esta en `sigma_pc` de la seccion 2, que es el parametro que efectivamente controla cuanto se encoge cada grupo.

## 5. Comparacion contra el contrafactual de altitud

Comparacion directa: `raw_delta_run_value` de `reports/altitude_counterfactual.md` (seccion 4, columna run_value — delta de la PREDICCION del modelo LightGBM Extreme vs. No Altitude, manteniendo fijo RelSpeed/SpinRate/residual de Magnus) contra `beta_altitude_cluster` x2 de este modelo bayesiano (delta OBSERVADO en run_value real, estimado con partial pooling, sin recalcular movimiento fisico — dos metodos completamente distintos sobre la misma pregunta).

### Por AutoPitchType individual (referencia, contrafactual)

| AutoPitchType | delta_run_value contrafactual | n |
|---|---:|---:|
| Four-Seam | +0.00898 | 34,318 |
| Curveball | +0.00907 | 8,490 |
| Slider | +0.00992 | 21,924 |
| Cutter | +0.01012 | 4,164 |
| Sinker | +0.01015 | 17,458 |
| Changeup | +0.01022 | 12,391 |
| Splitter | +0.01038 | 1,150 |

### Por cluster: contrafactual vs. modelo bayesiano

| cluster | delta_run_value contrafactual (No->Extreme) | delta_run_value bayesiano (No->Extreme, IC94%) |
|---|---:|---|
| fastballs | +0.00943 | +0.01817 [+0.01465, +0.02156] |
| breaking | +0.00969 | +0.01839 [+0.01394, +0.02279] |
| offspeed | +0.01023 | +0.01688 [+0.01024, +0.02211] |

**Direccion (signo):** coincide entre ambos metodos en los 3 clusters (ambos deberian ser positivos: mas runs permitidos en altitud extrema = peor para el pitcher).

**Nota de escala:** el contrafactual muestra deltas de run_value MUY similares entre clusters (todos ~0.009-0.010, ver tabla por AutoPitchType arriba) — el hallazgo de que Sinker pierde mas y Curveball/Splitter pierden menos (`reports/altitude_counterfactual.md` seccion 1) esta dominado por `is_out_in_pa`, no por `run_value` en si mismo (ver la descomposicion de la seccion 4 de ese reporte) — asi que **no se espera que este modelo bayesiano (que solo mira run_value) reproduzca el mismo ranking de perdida por cluster que el Pitching+ compuesto**, solo el mismo SIGNO/direccion general. Si el modelo bayesiano muestra diferencias grandes de magnitud entre clusters en run_value especificamente, eso seria una señal nueva no vista en el contrafactual (o una discrepancia a investigar, ver seccion 6).

## 6. Discrepancias a investigar

- **fastballs**: el delta del contrafactual (+0.00943) cae FUERA del IC 94% bayesiano (+0.01817 [+0.01465, +0.02156]) — revisar si es diferencia real entre metodos (el contrafactual mantiene el residual de Magnus del pitcher fijo y recalcula solo el movimiento teorico; el modelo bayesiano usa la variacion observada real entre altitudes, que incluye cualquier otro confusor no modelado) o ruido de muestra.
- **breaking**: el delta del contrafactual (+0.00969) cae FUERA del IC 94% bayesiano (+0.01839 [+0.01394, +0.02279]) — revisar si es diferencia real entre metodos (el contrafactual mantiene el residual de Magnus del pitcher fijo y recalcula solo el movimiento teorico; el modelo bayesiano usa la variacion observada real entre altitudes, que incluye cualquier otro confusor no modelado) o ruido de muestra.
- **offspeed**: el delta del contrafactual (+0.01023) cae FUERA del IC 94% bayesiano (+0.01688 [+0.01024, +0.02211]) — revisar si es diferencia real entre metodos (el contrafactual mantiene el residual de Magnus del pitcher fijo y recalcula solo el movimiento teorico; el modelo bayesiano usa la variacion observada real entre altitudes, que incluye cualquier otro confusor no modelado) o ruido de muestra.

**Verificacion aritmetica de la hipotesis "18%/82%" — NO cuadra, documentado sin forzar el ajuste.** La primera version de este reporte proponia que el modelo bayesiano sale ~1.8-2x mas grande porque absorbe el canal `is_out_in_pa` (~82% de la brecha observacional cruda, segun la descomposicion Oaxaca-Blinder de `reports/run_value_validation.md`) mientras el contrafactual aislaria solo el canal de "valor condicional" (~18%). Si esa hipotesis fuera literalmente correcta — contrafactual = ~18% del efecto, bayesiano = ~100% — el ratio bayesiano/contrafactual esperado seria **1/0.18 ≈ 5.56x**, no el 1.65x-1.93x observado (promedio 1.83x; ver tabla de la seccion 5). Dicho al reves: para que el ratio observado (1.83x) fuera consistente con esa hipotesis, el contrafactual tendria que estar capturando **~52-61%** del efecto total en cada cluster (1/ratio), no ~18%. La discrepancia entre el 18% esperado y el ~55% implicado es un factor ~3x — demasiado grande para atribuirlo a ruido de muestra en los IC 94% reportados.

**Conclusion: la hipotesis del canal `is_out_in_pa` como explicacion COMPLETA de la brecha esta descartada por esta verificacion; el signo de la explicacion (el bayesiano captura mas que el contrafactual) sigue siendo razonable, pero la cifra especifica de "82% via outs" no es la magnitud correcta para explicar ESTE ratio.** La razon de fondo, sin sobre-ajustar una narrativa nueva a esta cifra: el 18%/82% de `run_value_validation.md` es una descomposicion Oaxaca-Blinder de la **brecha observacional agregada cruda** (promedio de `run_value` real por categoria de altitud, separado en "cuanto viene de que P(out) difiere" vs "cuanto viene de que el valor condicional a out/no-out difiere"). El delta del contrafactual es una cantidad **distinta por construccion**: la prediccion de UN modelo LightGBM entrenado con location+conteo+arsenal+fisica completos, evaluada dos veces con solo el movimiento fisico perturbado — no es "el canal de valor condicional" de esa descomposicion, es el efecto marginal de un input especifico dentro de un modelo predictivo con muchos otros inputs fijos en sus valores observados. **Proporcion de una brecha observacional agregada** y **proporcion de un efecto marginal por pitch en un modelo distinto** no tienen por que coincidir numericamente, aunque ambas midan "cuanto explica la fisica pura" en un sentido amplio — son construcciones estadisticas diferentes sobre poblaciones/condicionamientos ligeramente distintos. Se deja como discrepancia residual sin explicacion numerica precisa, en vez de inventar una segunda hipotesis ad-hoc para cuadrar el ~3x que sobra.
