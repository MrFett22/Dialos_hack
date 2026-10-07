# Causal Forest (econml CausalForestDML) — efecto heterogeneo de altitud sobre run_value

## 0. Decisiones de diseño

**Tratamiento: `altitude_level` continuo ordinal (0/1/2), no binario.** Mismo diseño que `reports/bayesian_hierarchical.md` — permite comparar magnitudes directamente (no solo signo/ranking) multiplicando por 2 para obtener el equivalente No->Extreme, y preserva el 100% del dataset (no hay que descartar Medium Altitude). Ver docstring de este script para la justificacion completa.

**Heterogeneidad (X):** InducedVertBreak, HorzBreak, SpinRate, RelSpeed, velo_diff_vs_fb + AutoPitchType (one-hot, 7 tipos) — 12 columnas totales.
**Controles (W, doble ML):** Balls, Strikes, year + PitcherThrows + BatterSide (one-hot) — 6 columnas totales.

## 1. Datos y muestra

- Pitches validos (mismo filtro `pitcher_holdout != excluded`, mas run_value/velo_diff_vs_fb/PitcherThrows/BatterSide no nulos, mas AutoPitchType restringido a los 7 tipos principales — mismos que `reports/bayesian_hierarchical.md`): **623,785**
- Muestra usada para el forest: **180,001** pitches (28.9% del universo valido, estratificada por AutoPitchType x altitude_category)

## 2. Ajuste del modelo

`CausalForestDML(model_y=LGBMRegressor, model_t=LGBMRegressor, discrete_treatment=False, n_estimators=400, min_samples_leaf=50, max_samples=0.45, cv=3, honest=True)`. Tiempo de ajuste: 2.3 min.

**ATE global (efecto promedio, por nivel de altitude_level):** +0.00566 runs/pitch, IC 94% [-0.01325, +0.02457]. Equivalente No->Extreme (x2): +0.01132 [-0.02649, +0.04913].

## 3. CATE agregado por AutoPitchType

`cate_no_to_extreme` = efecto causal condicional promedio de mover ese tipo de pitch de No Altitude a Extreme Altitude (runs adicionales esperados por pitch). IC calculado con la inferencia asintotica nativa de CausalForestDML (`ate_interval`, alpha=0.06 -> 94%).

| AutoPitchType | n | CATE por nivel | CATE No->Extreme | IC 94% (No->Extreme) |
|---|---:|---:|---:|---|
| Curveball | 15,303 | +0.00802 | +0.01605 | [-0.02087, +0.05296] |
| Sinker | 31,451 | +0.00755 | +0.01511 | [-0.02528, +0.05550] |
| Changeup | 22,328 | +0.00726 | +0.01451 | [-0.02433, +0.05335] |
| Splitter | 2,072 | +0.00623 | +0.01247 | [-0.02393, +0.04887] |
| Slider | 39,502 | +0.00517 | +0.01033 | [-0.02232, +0.04299] |
| Four-Seam | 61,840 | +0.00413 | +0.00827 | [-0.03145, +0.04799] |
| Cutter | 7,505 | +0.00317 | +0.00635 | [-0.02844, +0.04113] |

## 4. CATE agregado por rango de InducedVertBreak

| rango IVB | n | CATE por nivel (media) | sd | CATE No->Extreme |
|---|---:|---:|---:|---:|
| <0 in | 27,999 | +0.00664 | 0.00659 | +0.01327 |
| 0-5 in | 30,038 | +0.00527 | 0.00595 | +0.01054 |
| 5-10 in | 38,900 | +0.00669 | 0.00701 | +0.01339 |
| 10-15 in | 48,196 | +0.00508 | 0.00648 | +0.01017 |
| >15 in | 34,868 | +0.00486 | 0.00681 | +0.00971 |

## 5. Comparacion de heterogeneidad contra los otros 3 metodos

**Ranking por metodo** (1 = mas afectado por altitud / peor para el pitcher, 7 = menos afectado / mejor). Convencion de signo unificada antes de rankear: para contrafactual y z-test, mas negativo = peor; para el causal forest y el bayesiano, mas positivo (mas runs permitidos) = peor.

**Nota sobre el bayesiano:** ese modelo se ajusto a nivel de CLUSTER (fastballs/breaking/offspeed), no de AutoPitchType individual — su columna aqui repite el valor del cluster para cada tipo que pertenece a el (marcado explicitamente), no es una estimacion pitch-type-especifica independiente como las otras 3.

| AutoPitchType | cluster | rank contrafactual | rank z-test (whiff) | rank causal forest | rank bayesiano (cluster) |
|---|---|---:|---:|---:|---:|
| Four-Seam | fastballs | 5 | 1 | 6 | 3 |
| Sinker | fastballs | 1 | 2 | 2 | 3 |
| Cutter | fastballs | 3 | 5 | 7 | 3 |
| Slider | breaking | 4 | 4 | 5 | 1 |
| Curveball | breaking | 6 | 6 | 1 | 1 |
| Changeup | offspeed | 2 | 3 | 3 | 6 |
| Splitter | offspeed | 7 | 7 | 4 | 6 |

### Lectura — ADVERTENCIA: el ranking del causal forest NO es estadisticamente distinguible del ruido

**Los 7 intervalos de confianza de la seccion 3 incluyen cero, Y se solapan entre si casi por completo** (ej. Curveball [-0.021, +0.053] vs Cutter [-0.028, +0.041] — los dos extremos del ranking — se solapan en practicamente todo su rango). Esto significa que **ningun par de tipos de pitch es estadisticamente distinguible entre si** con este diseño/muestra, y tampoco ninguno es distinguible de cero. El ATE global tampoco excluye cero (IC 94% [-0.013, +0.025]).

**Consecuencia para la comparacion de rankings:** el ranking por punto-estimado de la tabla de arriba (Curveball #1 "peor", Sinker #2, Cutter #7 "mejor") **no debe leerse como que el causal forest confirma o contradice** el ranking de los otros 3 metodos — con estos IC, ese ranking especifico es en gran medida ruido de estimacion, no señal. Reportarlo como si fuera una cuarta confirmacion/contradiccion independiente seria sobre-interpretar un punto estimado sin respaldo estadistico, exactamente el tipo de error que este proyecto ha evitado en los reportes anteriores (ver `reports/bayesian_hierarchical.md` seccion 6, mismo estandar de honestidad).

**Por que el causal forest tiene tan poca potencia aqui, a diferencia del bayesiano:** `run_value` es un target extremadamente ruidoso (std≈0.276, mayormente ceros con valores grandes solo en pitches que terminan una plate appearance — ver `reports/targets_summary.md`), y el efecto de altitud es diminuto en comparacion (~0.005-0.015 runs/pitch/nivel). El modelo bayesiano jerarquico logra intervalos angostos porque IMPONE una estructura fuerte (partial pooling hacia 3 medias de cluster, con `sigma_pc`/`sigma_altitude_cluster` estimados y compartidos entre miles de grupos) — practicamente toda la potencia estadistica se concentra en estimar bien esos 3 clusters. El causal forest, en cambio, es no-parametrico: estima el CATE como funcion continua de 5 covariables + tipo de pitch sin imponer esa estructura de 3 grupos, lo cual es mas flexible pero cuesta muchisima mas varianza por grupo dado el mismo N — con run_value de esta magnitud de ruido, hace falta bastante mas de 180k pitches (o una estructura mas restringida, como forzar splits solo por cluster en vez de por covariable continua) para que el causal forest tenga la potencia que si tiene el bayesiano.

**Lo unico que SI es leible con algo de confianza:** el signo del ATE global (+0.00566 por nivel) coincide en DIRECCION con los otros 3 metodos (mas runs permitidos en altitud = peor para el pitcher), aunque su IC no excluye cero — es consistente con, no independiente-y-significativamente-confirmatorio-de, el hallazgo de las otras 3 fuentes.

- Correlacion de Spearman entre rankings (reportada por transparencia, NO como evidencia de acuerdo/desacuerdo dado lo anterior): contrafactual vs causal forest = +0.036; z-test vs causal forest = -0.143; contrafactual vs z-test = +0.607 (estos dos ultimos SI son metodos con inferencia mas potente/directa por tipo de pitch, y coinciden razonablemente).
- Con muestra mas grande (dataset completo, ~624k) o un forest restringido a splittear solo por cluster (replicando la estructura de 3 grupos del bayesiano en vez de dejar que el forest elija splits libremente sobre 5 covariables continuas), es probable que los IC se achicaran lo suficiente para una comparacion con mas poder — queda como trabajo futuro, no implementado aqui por alcance.
