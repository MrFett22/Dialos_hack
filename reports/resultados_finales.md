# Resultados finales — Stuff+ LMB con corrección por altitud

Documento consolidado de todos los hallazgos del proyecto, con cifras exactas y su archivo fuente citado entre paréntesis para verificación rápida. No reemplaza a los reportes individuales (más detalle metodológico en cada uno) — es el punto de entrada único para redactar el documento técnico final.

---

## 1. Datos

**Dataset crudo:** 635,002 pitches, 1,134 pitchers únicos, temporadas 2024/2025/2026 (`historial/reports/eda_summary.md`). Distribución por año: 2024=243,254 · 2025=256,144 · 2026=135,604 (`historial/reports/eda_summary.md`).

**Composición por altitud:** Extreme Altitude (el planteamiento original lo asumía como el estadio de la Ciudad de México) = 28.9% del total (183,267 pitches) — volumen suficiente para modelar con confianza (`historial/reports/eda_summary.md`).

**Volumen por pitcher:** mediana 331 pitches/pitcher, p10=22, p1=8, máximo 4,222. **309 pitchers (27.2%) tienen <100 pitches** — motivó el shrinkage bayesiano de la sección 3 (`historial/reports/eda_summary.md`).

**Targets construidos** (`reports/targets_summary.md`):
- `is_whiff`: 71,742 positivos, 0 nulos, 447 inconsistencias lógicas (whiff+contact simultáneo) sin resolver.
- `is_called_strike`: 97,579 positivos, 0 discrepancias vs `PitchCall`.
- `run_value`: reconstruido sobre 162,035 plate appearances en 36,995 medias-entradas; 3,168 pitches sin estado RE válido.

**Validación de run_value** (`historial/reports/run_value_validation.md`): matriz RE24 de la LMB reconstruida sigue el patrón correcto (decrece con outs, crece con corredores) y está en escala razonable vs. la referencia MLB (*The Book*). Se encontró y corrigió un bug de reconciliación en `_advance_on_hit`: bajó las inconsistencias en PA con hit de **18.94–19.16% a 0.97%** (404 de 41,579 PA con hit). El fix no cambió el hallazgo de altitud (Δ<0.001 carreras en los 3 grupos). **Descomposición Oaxaca-Blinder de la brecha de run_value por altitud: ~82% viene de la diferencia en tasa de outs (P(out)), ~18% de valor condicional** — hallazgo fundacional que se retoma en la sección 4.

**Arsenal** (`reports/arsenal_summary.md`): 24 de 1,718 combinaciones pitcher-year (1.4%, 440 pitches) sin fastball identificable. Nulos en `velo_diff_vs_fb`/`ivb_diff_vs_fb`/`hb_diff_vs_fb`/`tunnel_similarity`: 3,448.

**Nulos en columnas de identidad** (`reports/pitcher_id_null_investigation.md`):
- `pitcher_anon_id` nulo: 3,008 filas (0.47%), 95.4% concentradas en 2026, en solo 34 de 2,127 juegos — patrón MNAR (pitcher entero invisible, no ruido aleatorio).
- `altitude_category` nulo: 3,019 filas (0.48%), problema **distinto** (mapeo estadio→altitud faltante en 10 juegos), solo 316 filas se solapan con el problema anterior.
- Decisión aplicada: excluir la unión de ambos (5,711 filas, 0.90%) de los 3 sub-modelos por igual.

**Splits** (`reports/splits_summary.md`): 627,736 pitches válidos, 1,045 pitchers únicos, seed=42.
- `pitcher_holdout`: 837 pitchers train / 208 test (19.9%).
- `season_holdout`: 2026 a test (497,596 train / 130,140 test); 37 pitchers flageados con cobertura 2026 sospechosamente baja.
- `altitude_holdout`: Extreme Altitude a test (446,932 train / 180,804 test).

---

## 2. Modelos

**Stuff+ v1** (`historial/reports/stuff_plus_baseline.md`): target oficial único `is_whiff`. AUC test = **0.7297** vs. baseline ingenuo 0.5943 (+0.1354). `run_value` con solo física queda *peor* que el baseline por count (R²=0.0032 vs 0.0205) — por eso no es target de Stuff+. Ablation: quitar `altitude_level` + `magnus_residual` no cuesta AUC (0.7297 → 0.7306, diferencia de ruido).

**Location+ v1** (`historial/reports/location_plus_baseline.md`): GAM, solo ubicación+conteo+lateralidad. `is_called_strike` AUC=**0.8468**, `is_chase` AUC=**0.7993** (ambos oficiales). Confirmación simétrica del ablation de Stuff+: con solo ubicación, `is_whiff` predice peor (AUC 0.6839) que con física.

**SHAP sobre Stuff+** (`historial/reports/shap_findings.md`): top-5 global por importancia — VertApprAngle (0.2985), HorzRelAngle (0.2861), VertRelAngle (0.2709), RelSide (0.1840), RelHeight (0.1208). `altitude_level` es la feature **menos importante** de las 23 (rank 23, mean|SHAP|=0.0076), confirmando el ablation. `magnus_residual_horz` (rank 10) se investigó y se concluyó que es redundancia (correlación -0.809 con SpinAxis), no señal nueva.

**Pitching+ v1 combinado** (`historial/reports/pitching_plus_baseline.md`): 33 features, 8 targets, LightGBM. AUC `pitcher_holdout`: is_whiff=0.7575, is_chase=0.9173, is_called_strike=0.8840, is_weak_contact=0.7677, is_groundball=0.7812, is_hard_contact=0.8105, is_out_in_pa=0.6292; run_value R²=0.0435. Mejora sobre sub-modelos individuales: is_whiff +0.0279 AUC vs. Stuff+ solo, is_called_strike +0.0371 y is_chase +0.1180 vs. Location+ solo — confirma que combinar aporta señal real, no redundancia.

**Pitching+ score OOF (versión oficial)** (`reports/pitching_plus_score_oof.md`): n=1,628 pitcher-temporadas, mean=100.00, std=**6.32**, rango [53.88, 131.13]. Correlación con la versión in-sample = **0.9429**, overlap top-10 = 9/10, bottom-10 = 3/10 — ranking robusto al sesgo in-sample, pero esta es la versión a usar de aquí en más.

---

## 3. Validación

**Diseño de 2 pasos del contrafactual** (`historial/reports/notas_metodologicas.md`): un contrafactual ingenuo (cambiar solo `altitude_level` dejando el movimiento fijo) mide ~0 por construcción, dado que el ablation de Stuff+ ya mostró que el efecto de altitud está completamente mediado por el movimiento observado. Diseño correcto: recalcular movimiento vía Magnus preservando el residual del pitcher, luego predecir con Stuff+/Pitching+.

**Modelo físico de Magnus** (`reports/magnus_analysis.md`): offset SpinAxis→dirección de break calibrado empíricamente en 178.67° (std=8.72°, redondeado a 180°) sobre 633,447 pitches. El modelo sobreestima la magnitud del break ~1.46x (razón teórica/observada mediana) — consistente con no separar spin útil de giroscópico. El residual de Magnus reduce la varianza explicada por altitud en 79.2% (vertical) y 98.9% (horizontal) vs. el movimiento crudo. **Hallazgo abierto sin resolver:** Medium Altitude tiene el residual vertical más negativo de las 3 categorías, patrón no-monotónico que sobrevive al descartar composición de AutoPitchType (Cramer's V=0.0587, ajuste apenas mueve -2.6214→-2.6341).

**Chequeo OOD del contrafactual** (`historial/reports/altitude_counterfactual_ood_check.md`): % de combinaciones fuera de distribución (percentiles 1-99, referencia = train de `pitcher_holdout`, 503,112 pitches) — No Altitude 7.02%, Medium Altitude 13.81%, Extreme Altitude 10.39%. **Calibración clave: la tasa "natural" de OOD en pitches reales de test (nunca vistos en train) es 8.52%** — prácticamente el mismo orden que los escenarios contrafactuales. Conclusión: **no es un artefacto de extrapolación**.

**z-test estratificado por tipo de pitch** (`historial/reports/altitude_whiff_stratified.md`): Cochran-Mantel-Haenszel pooled, controlando por AutoPitchType — OR=**0.9055**, chi2=107.09, p<0.001. El efecto agregado sobrevive al control por tipo de pitch; composición de tipos explica ~-7% de la brecha cruda (prácticamente nada). Curveball muestra dirección opuesta (+5.47% relativo, z=2.10, **p=0.036**) — señalado explícitamente como "patrón real pero con mayor incertidumbre relativa" (no sobrevive corrección de Bonferroni con 7 tipos testeados, umbral≈0.0071).

**Sanity check del dashboard** (`historial/reports/dashboard_sanity_check.md`): identidad (reconstruir bajo la misma categoría de referencia) — diferencia máxima **8.88e-16 pulgadas** sobre 28 pitches (error de punto flotante puro). Proyección Extreme Altitude vs. pitches reales similares: **27 de 28 pitches (96%) caen dentro de 2 desviaciones estándar** de la distribución real observada. Investigación de sesgo direccional en HorzBreak: con n=5 los 5 z-scores salían positivos, pero con n=28 el sesgo **no se confirmó sistemático** (39% positivos, p=0.345) — coincidencia de muestra chica, no un patrón real ni relacionado con el hallazgo de Medium Altitude de `magnus_analysis.md`.

---

## 4. Efecto de altitud

**Delta de Pitching+ por tipo de pitch** (Extreme − No Altitude), contrafactual sobre muestra estratificada de 100,000 pitches (`historial/reports/altitude_counterfactual.md` sección 1):

| AutoPitchType | delta Pitching+ | n |
|---|---:|---:|
| Sinker | −1.059 | 17,458 |
| Changeup | −0.951 | 12,391 |
| Cutter | −0.904 | 4,164 |
| Slider | −0.843 | 21,924 |
| Four-Seam | −0.839 | 34,318 |
| Curveball | −0.829 | 8,490 |
| Splitter | −0.817 | 1,150 |

**Los 7 tipos pierden Pitching+ en altitud extrema — ninguno gana.** Sinker pierde más; Curveball, Four-Seam y Splitter pierden menos (no ganan valor).

**Descomposición en 8 componentes** (`historial/reports/altitude_counterfactual.md` sección 4): `is_out_in_pa` domina la caída en los 7 tipos (entre −0.52 y −0.65 puntos). Caso Curveball: `is_whiff` (+0.063) e `is_weak_contact` (+0.057) son, cada uno, el máximo de los 7 tipos — compensan parcialmente la pérdida en `is_called_strike` (peor de los 7, −0.057) e `is_out_in_pa`.

**H2 (diferencial de velocidad vs. fastball)** (`historial/reports/altitude_counterfactual.md` sección 2): **no consistente** en la muestra agregada de todos los tipos (bin más lejano del fastball pierde −0.856, el más cercano −0.819, dirección contraria a la esperada). Restringido a Changeup (sección 2b): se sostiene en los extremos (−0.910 vs. −1.167) pero **no de forma monotónica** — quiebre explicado por un salto real de SpinRate (~1,750→~2,000+ rpm) entre sub-poblaciones de Changeup, verificado robusto a exclusión de los 5 pitchers más frecuentes por bin.

**Traducción a carreras/temporada** (`historial/reports/runs_translation.md`): pendiente Pitching+→runs = **−0.001551 runs/pitch por punto** (R²=0.7177, n=1,278 pitcher-temporadas con ≥100 pitches).

| AutoPitchType | Abridor (2,500-3,000 pitches) | Relevista (800-1,000 pitches) |
|---|---|---|
| Sinker | +4.10 a +4.93 (≈**4.52**) | +1.31 a +1.64 (≈1.48) |
| Changeup | +3.69 a +4.43 (≈4.06) | +1.18 a +1.48 (≈1.33) |
| Cutter | +3.50 a +4.21 (≈3.86) | +1.12 a +1.40 (≈1.26) |
| Slider | +3.27 a +3.92 (≈3.60) | +1.05 a +1.31 (≈1.18) |
| Four-Seam | +3.25 a +3.91 (≈3.58) | +1.04 a +1.30 (≈1.17) |
| Curveball | +3.21 a +3.86 (≈3.54) | +1.03 a +1.29 (≈1.16) |
| Splitter | +3.17 a +3.80 (≈3.49) | +1.01 a +1.27 (≈1.14) |

(positivo = carreras permitidas de más = pérdida de valor). El mecanismo aislado de "más whiffs con la curva" vale solo ~0.27 carreras/temporada — no revierte el resultado neto de Curveball (sigue perdiendo ~3.5).

**Modelo bayesiano jerárquico** (`historial/reports/bayesian_hierarchical.md`): 180,018 pitches (28.8% del universo válido), 3,008 grupos pitcher×cluster, NUTS 4 cadenas × 2,000 draws. **Convergencia perfecta**: r_hat máximo 1.0100 (0 parámetros por encima del umbral), ess_bulk mínimo 1,174, 11 divergencias/4,000 (0.275%, no invalidante). Efecto de altitud por cluster (No→Extreme, IC 94%): fastballs +0.01817 [+0.01465,+0.02156], breaking +0.01839 [+0.01394,+0.02279], offspeed +0.01688 [+0.01024,+0.02211] runs/pitch — **misma dirección en los 3 clusters**, pero ~1.8-2x más grande que el contrafactual (avg ratio 1.83x). Se verificó aritméticamente que esto NO se explica por la partición 82%/18% del run_value_validation.md (ratio esperado 5.56x bajo esa hipótesis) — queda como discrepancia residual entre "proporción de brecha observacional" y "proporción de efecto marginal en un modelo distinto", sin forzar una explicación numérica exacta. Shrinkage confirmado: pitcher con n=1 pitch se encoge 99.9% hacia la media de su cluster.

**Causal Forest** (`historial/reports/causal_forest.md`): 180,001 pitches, `CausalForestDML` con tratamiento continuo. ATE global +0.00566 runs/pitch/nivel, **IC 94% incluye cero [−0.01325, +0.02457]**. Los 7 CATE por tipo de pitch tienen ICs que se solapan casi por completo (ningún par distinguible entre sí) — **el ranking por tipo de pitch NO es estadísticamente distinguible del ruido con esta muestra**. Solo el signo del ATE global es consistente en dirección con los otros 3 métodos. Spearman: contrafactual vs. z-test = +0.607 (razonable), ambos vs. causal forest ≈ 0.

**Síntesis de 4 métodos:** contrafactual, z-test estratificado y bayesiano (a nivel de cluster) coinciden en dirección (altitud reduce el valor del pitcher) con evidencia estadística sólida. El causal forest confirma la dirección del efecto agregado pero está subpotenciado para resolver heterogeneidad por tipo de pitch a este tamaño de muestra — no confirma ni contradice el ranking de los otros 3.

---

## 5. Recomendaciones

De `historial/reports/runs_translation.md` y `reports/scouting/`:

- **Un abridor sinker-heavy pierde ~4.5 carreras/temporada** (rango 4.1-4.9) al jugar en altitud extrema vs. nivel del mar — el mayor impacto entre los 7 tipos analizados.
- **Curveball, Four-Seam y Splitter son los que MENOS pierden** (~3.5 carreras/temporada) — no presentar como que "ganan" valor, la cifra sigue siendo una pérdida neta.
- Para un relevista (800-1,000 pitches/temporada), el mismo efecto por tipo de pitch representa ~1.1-1.5 carreras/temporada — menor solo por volumen, no por un efecto distinto por pitch.
- **Reportes de scouting individuales** (`reports/scouting/README.md`, 10 pitchers): ejemplo verificado — `pitcher_00107` (Sinker 77% de uso de su arsenal) recibe la recomendación automática "riesgo de pérdida de valor en altitud extrema (~+4.5 carreras/temporada proyectadas)" generada por regla determinística, consistente con la tabla de arriba; percentil 94 en su temporada 2026 (n_pool=368 pitcher-temporadas confiables). `pitcher_00469` (top Pitching+=116.25, percentil 100) sirve de ejemplo de arsenal balanceado (Slider 43.5% / Sinker 41.7% / Curveball 13.8%) sin tipo dominante de alto riesgo.
- El dashboard interactivo (`historial/dashboard/app.py`) permite evaluar cualquier pitch hipotético (prospecto externo o pitcher actual con referencia de altitud ajustable) usando el mismo mecanismo de 2 pasos validado en la sección 3.

---

## 6. Límites del análisis

1. **Elevación de las categorías de altitud es un supuesto, no un dato medido.** No Altitude=50m, Medium=800m, Extreme=1,900m — la altitud real de los estadios no está en el dataset, así que el punto medio usado para Extreme es conservador (`reports/magnus_analysis.md`).
2. **El modelo de Magnus no separa spin útil de spin giroscópico**, por lo que sobreestima la magnitud del break ~1.46x — la comparación *relativa* entre categorías de altitud no se ve afectada, pero la magnitud absoluta del residual sí (`reports/magnus_analysis.md`).
3. **Hallazgo abierto sin explicación:** el residual vertical de Magnus en Medium Altitude es sistemáticamente el más negativo de las 3 categorías, sin explicación de composición de tipos de pitch — candidatos no investigados: efecto de drag, composición de pitchers/equipos por estadio (`reports/magnus_analysis.md`).
4. **H2 (diferencial de velocidad vs. fastball) no está confirmada** en la muestra agregada; dentro de Changeup se sostiene solo en los extremos, no monotónicamente, y el quiebre coincide con una sub-población de mayor SpinRate no explicada más allá de una hipótesis física plausible (`historial/reports/altitude_counterfactual.md`).
5. **El hallazgo de Curveball (dirección opuesta en is_whiff) tiene significancia estadística débil** (p=0.036, no sobrevive corrección de Bonferroni con 7 tipos) — reportado con menor confianza que el hallazgo principal (`historial/reports/altitude_whiff_stratified.md`, `historial/reports/runs_translation.md`).
6. **Discrepancia de magnitud sin resolver entre el modelo bayesiano y el contrafactual** (~1.8-2x), verificada aritméticamente como NO explicable por la partición 82%/18% de `run_value_validation.md` — queda como diferencia metodológica de fondo entre "brecha observacional" y "efecto marginal en un modelo" (`historial/reports/bayesian_hierarchical.md`).
7. **El Causal Forest está subpotenciado** para resolver heterogeneidad por tipo de pitch con 180k pitches — sus 7 intervalos de confianza se solapan casi por completo; no debe citarse como confirmación ni contradicción del ranking por tipo de pitch de los otros métodos (`historial/reports/causal_forest.md`).
8. **El score de Pitching+ es in-sample para pitchers de train** en la versión original (corregido por la versión OOF, que es la oficial) — usar siempre `pitching_plus_scores_oof.parquet`, nunca la versión in-sample, para análisis nuevos (`reports/pitching_plus_score_oof.md`).
9. **`is_out_in_pa` (AUC=0.6292) es el target con peor poder predictivo de los 8**, aunque es el que domina el efecto de altitud en la descomposición — el modelo predictivo de ESTE target específico es el más débil del ensemble, algo a tener en cuenta al interpretar magnitudes finas (`historial/reports/pitching_plus_baseline.md`).
10. **El dashboard completa 24 de 33 features del modelo con supuestos** (mediana histórica por tipo de pitch, conteo 0-0, ubicación al centro de la zona) cuando el usuario solo provee 9 — validado end-to-end contra datos reales (sección 3), pero no reemplaza una predicción con las features reales completas de un pitch específico.
11. **Reconstrucción de ocupación de bases** en run_value sigue teniendo 0.97% de casos degenerados sin resolver del todo (404 de 41,579 PA con hit) — no afecta runs/outs reales, solo el detalle de a qué base llegó cada corredor (`historial/reports/run_value_validation.md`).

---

## 7. Bitácora de mejoras (post-v1)

Registro cronológico de cada cambio hecho después de la versión v1 descrita arriba. Cada entrada dice qué se hizo, por qué, y cómo cambió el benchmark de la sección 7.2. Las secciones 1-6 describen la v1 y **no se reescriben**: cuando una mejora cambie una cifra, se anota aquí y se señala qué cifra de arriba queda superada.

**Restricción del hackathon:** no se pueden pedir datos adicionales a los organizadores (p. ej. estadio por partido o elevación real). Toda mejora sobre la limitación de altitud debe salir del dataset entregado.

### 7.1 Paso 0 — Entorno reproducible y control de versiones (2026-10-03)

- Entorno nuevo `.venv` (Python 3.14) con todas las dependencias; versiones exactas en `requirements-lock.txt`. Las carpetas vendorizadas `.pytools/` y `.pytools_bayes/` quedan obsoletas (incompletas, mezclaban versiones, compiladas para un Python 3.12 que ya no existe en la máquina).
- Se verificó que los datos, los 3 modelos v1 y el Pitching+ OOF cargan y reproducen las cifras reportadas (media=100.00, std=6.32).
- Repositorio git local con la versión v1 congelada en el tag `baseline-v1`. Datos y artefactos pesados (`bayesian_hierarchical.nc`, `causal_forest.pkl`) fuera de git; son reproducibles con sus scripts.
- Para correr cualquier script: `.venv\Scripts\python -m src.<modulo>`.

### 7.2 Paso 1 — Benchmark año-contra-año: la vara para medir mejoras (2026-10-03)

**Qué se hizo.** `src/validation/benchmark.py` mide cada métrica a nivel pitcher-temporada con dos criterios, sobre los pares 2024→2025 y 2025→2026:
- **Estabilidad:** correlación de la métrica en *t* con la misma métrica en *t+1*.
- **Poder predictivo:** correlación de la métrica en *t* con resultados **observados** en *t+1*. El resultado principal son las carreras evitadas por cada 100 pitches (run value RE24 observado).

Se compara contra baselines tradicionales observados en *t* (whiff%, CSW%, K%, K-BB%, run value). Los IC 95% salen de un bootstrap de 2,000 réplicas por pitcher, y las comparaciones entre métricas usan bootstrap pareado. Como subproducto, **Stuff+ v1 se escala por primera vez a 100+10z** por pitcher-temporada, con predicciones out-of-fold (std=9.99, rango [60.97, 169.19]). Detalle completo en `reports/benchmark_baseline_v1.md`; cifras crudas en `reports/benchmark/baseline_v1.json`.

**Línea base v1** (mínimo 100 pitches por temporada, 437 pares de 322 pitchers):

| métrica | estabilidad r | predice carreras evitadas t+1 | predice whiff% t+1 |
|---|---:|---:|---:|
| Stuff+ v1 | **0.668** | 0.181 | 0.362 |
| Pitching+ v1 | 0.545 | 0.169 | 0.172 |
| Whiff% observado | 0.487 | 0.196 | **0.487** |
| CSW% observado | 0.349 | **0.205** | 0.357 |
| K-BB% observado | 0.263 | 0.178 | 0.296 |
| Run value observado | 0.085 | 0.085 | 0.150 |

**Hallazgos.**
1. **Ningún modelo v1 predice el año siguiente mejor que una estadística observada simple.**
   - Para carreras en *t+1*, Stuff+ y Pitching+ quedan por debajo de CSW% (Δr de −0.024 y −0.036, con IC que incluyen cero).
   - Para whiff% en *t+1*, el whiff% observado le gana claramente al Stuff+ (Δr = −0.125, IC [−0.210, −0.036]), aunque Stuff+ se entrena justamente para predecir whiff.
   - Con mínimo 300 pitches, Pitching+ empata con K-BB% (Δr = +0.007, 54% de probabilidad de ser mejor).
2. **Los modelos son más estables que las estadísticas observadas, pero predicen peor.** Esa combinación sugiere que capturan rasgos persistentes del pitcher que no se traducen en resultados. Encaja con el diagnóstico de que Stuff+ v1 se apoya en el punto de salida y en ángulos que codifican ubicación y "huella" del pitcher.
3. **La calibración pitch a pitch de Stuff+ es buena** (ECE=0.0046, AUC OOF=0.7315), con una leve subestimación en el decil más alto (0.282 predicho vs. 0.300 observado). El problema no es de calibración, sino de qué información usa el modelo.
4. **La muestra de pares es chica.** Solo ~280 pitchers aparecen en temporadas consecutivas, así que los IC son anchos (±0.08 a ±0.13). Una mejora debe mover r en ~0.05-0.10 para distinguirse del ruido.
5. **Anomalía a seguir:** con mínimo 300 pitches, el poder predictivo de Stuff+ sobre carreras cae a 0.039 (vs. 0.181 con mínimo 100). Puede ser un efecto de rol (abridores vs. relevistas), pero no se investigó.

**Criterio de aceptación para los pasos siguientes** (fijado antes de ver los resultados de cualquier mejora):
- **Principal:** poder predictivo sobre carreras evitadas en *t+1* (mínimo 100 pitches), con delta pareado contra v1 > 0 y P(mejor) ≥ 90%.
- **Específico de Stuff+:** poder predictivo sobre whiff% en *t+1*.
- **Secundario:** estabilidad. No se acepta un cambio que suba la estabilidad pero baje el poder predictivo.
- **Meta de fondo:** que los modelos superen al mejor baseline observado. Hoy no lo hacen.

**Cifras de arriba que esto matiza:** la sección 2 presenta a Stuff+ y Pitching+ con métricas pitch a pitch (AUC). A nivel pitcher-temporada, ninguno de los dos supera todavía a las estadísticas tradicionales para proyectar el año siguiente.

### 7.3 Paso 2 — Features de Stuff+: mano del pitcher, ubicación escondida, redundancias (2026-10-03)

**Qué se hizo.** `src/models/stuff_plus_v2.py` prueba 4 cambios en escalera. Cada variante agrega uno sobre la anterior, con los mismos hiperparámetros de v1 y predicciones OOF:

| variante | cambio | features |
|---|---|---:|
| v2a | espejo de zurdos (signo de HB, RelSide, ángulos horizontales; SpinAxis → 360 − SpinAxis) + indicador `same_side` en lugar de `PitcherThrows`/`BatterSide` | 22 |
| v2b | + VAA/HAA ajustados por ubicación (residual de una recta ángulo ~ PlateLocHeight/Side por tipo de pitch); se quitan los ángulos de salida | 20 |
| v2c | − redundancias (VertBreak, EffectiveVelo, ZoneSpeed, ZoneTime) − altitud − residuales de Magnus | 13 |
| v2d | + diferencias de velocidad y movimiento contra la recta del pitcher (candidata declarada de antemano) | 16 |

También se corrieron dos **diagnósticos** con el target `P(whiff | swing)` (entrenando solo con swings), como adelanto del Paso 3. Detalle completo en `reports/stuff_plus_v2.md`.

**Resultados** (mínimo 100 pitches, delta pareado contra v1):

| variante | AUC pitch a pitch | estabilidad | predice carreras t+1 | Δ vs v1 (carreras) | predice whiff% t+1 |
|---|---:|---:|---:|---|---:|
| v1 | 0.732 | 0.668 | 0.181 | — | 0.362 |
| v2a mano | 0.736 | 0.661 | 0.159 | −0.022 (P=6%) | 0.345 |
| v2b ubicación | 0.634 | 0.734 | 0.148 | −0.032 (P=12%) | 0.285 |
| v2c limpio | 0.632 | 0.724 | 0.140 | −0.041 (P=7%) | 0.269 |
| v2d arsenal | 0.640 | 0.693 | 0.163 | −0.017 (P=31%) | 0.310 |
| diag. v2d, whiff \| swing | 0.651* | 0.718 | **0.193** | +0.013 (P=65%) | 0.314 |

\* AUC sobre swings solamente; no es comparable 1 a 1 con las demás filas. Con mínimo 300 pitches, el diagnóstico v2d whiff|swing es el mejor en carreras t+1 (0.097 vs. 0.039 de v1; Δ=+0.058, P=85%).

**Hallazgos.**
1. **La mayor parte de lo que predecía Stuff+ v1 era ubicación, no "stuff".** Al quitar la ubicación escondida en VAA y en los ángulos de salida, el AUC baja de 0.73 a 0.63. VertApprAngle tiene correlación 0.78 con PlateLocHeight. El Stuff+ v1 funciona en la práctica como un "Stuff + Location". Esto matiza el top-5 de SHAP de la sección 2: los ángulos dominaban por la ubicación que codifican.
2. **Quitar la ubicación hace al índice más estable, pero no más predictivo.** Con el target `is_whiff` sobre todos los pitches, la ubicación decide en buena parte si el bateador tira. Sin ella, el modelo pierde la señal principal de su propio target.
3. **El diagnóstico apunta al target, no a las features.** Con las mismas features "limpias" de v2d pero entrenando P(whiff | swing), el índice recupera el poder predictivo sobre carreras y queda como el mejor de todos (0.193 con mín. 100; 0.097 con mín. 300). Aun así no alcanza el umbral de aceptación.
4. **El espejo de zurdos sube el AUC** (0.732 → 0.736), pero no cambia el benchmark de forma distinguible del ruido.
5. **En v2d, la variable más importante es la diferencia de velocidad contra la recta (26% del gain).** Le siguen el IVB, la diferencia de IVB contra la recta y la velocidad.

**Decisión** (según el criterio fijado en 7.2):
- **Ninguna variante v2 se acepta como oficial**, porque ninguna mejora las carreras t+1 con P ≥ 90%. **Stuff+ v1 sigue siendo la versión oficial.**
- Las features v2d (espejo, ángulos ajustados por ubicación, sin redundancias, con diferencias de arsenal) **se conservan como base del Paso 3**, porque son conceptualmente correctas y en combinación con el target condicionado dan el mejor resultado observado hasta ahora.
- Artefacto `src/models/artifacts/stuff_plus_v2.pkl` (v2d entrenado con todos los pitches) guardado solo como referencia; no reemplaza a v1.

**Cifras de arriba que esto matiza:** sección 2, "SHAP sobre Stuff+". Los ángulos de salida y de llegada dominan la importancia porque codifican ubicación, no porque sean la parte más valiosa del "stuff".

### 7.4 Paso 3 — Cadena de probabilidades condicionales valuada en carreras (2026-10-03)

**Qué se hizo.** `src/models/run_value_chain.py` reemplaza los 8 targets binarios independientes y sus pesos elegidos a mano por la estructura estándar de Stuff+/PitchingBot. Cada pitch se descompone en eventos excluyentes:
- **take:** bola, strike cantado o golpeado;
- **swing:** whiff, foul o bola en juego, y la bola en juego termina en out, 1B, 2B, 3B o HR.

Se entrenan 5 sub-modelos condicionales, cada uno en su universo correcto: P(swing), resultado del take, P(whiff | swing), P(foul | contacto) y resultado | en juego. Las probabilidades se multiplican por el **valor en carreras de cada evento según el conteo**. Esos valores salen de los run values RE24 observados, con una recursión hacia atrás sobre los 12 conteos (V(0-0)=+0.014, que coincide exactamente con el run value promedio por turno observado). Los eventos se definen desde `PitchCall`, porque hay ~1,400 filas con `is_swing` inconsistente.

Se construyeron dos índices:
- **Stuff+ v3:** features de v2d, sin ubicación ni conteo; valores neutrales al conteo.
- **Pitching+ v3:** features de v2d + ubicación + conteo; valores en el conteo real.

**Hallazgo intermedio.** En Stuff+ v3, **el 66% de la varianza entre pitchers viene del componente de bola en juego**. Ese componente es muy estable (0.71), pero casi no predice las carreras del año siguiente (0.04): es el fenómeno DIPS, el pitcher controla poco el resultado del contacto. Por eso se agregó una variante **DIPS**, con el resultado en juego fijado al promedio de liga. Es **post-hoc** (decidida después de ver los datos), así que se validó además en cada par de temporadas por separado. Detalle completo en `reports/run_value_chain.md`.

**Resultados** (predicción de carreras evitadas en t+1, delta pareado contra la v1 correspondiente):

| índice | mín. 100 pitches | Δ vs v1 | mín. 300 pitches | Δ vs v1 |
|---|---:|---|---:|---|
| Stuff+ v1 | 0.181 | — | 0.039 | — |
| Stuff+ v3 | 0.030 | −0.151 (P=1%) | 0.115 | +0.076 (P=82%) |
| Stuff+ v3 DIPS | 0.040 | −0.141 (P=1%) | 0.074 | +0.035 (P=68%) |
| Pitching+ v1 | 0.169 | — | 0.194 | — |
| Pitching+ v3 | 0.154 | −0.016 (P=34%) | 0.227 | +0.034 (P=80%) |
| Pitching+ v3 DIPS | 0.189 | +0.019 (P=72%) | 0.229 | +0.035 (P=76%) |

Por par de temporadas, Pitching+ v3 DIPS (mín. 100) da +0.045 en 2024→2025 y −0.014 en 2025→2026: la mejora no es consistente.

**Hallazgos.**
1. **La cadena no sirve para un Stuff+ sin ubicación.** Sin ubicación, P(swing) es casi aleatorio (AUC 0.545) y el resultado del take tampoco se puede predecir. Esos componentes meten ruido, y el componente "take" correlaciona incluso negativamente con las carreras t+1 (−0.13). Para Stuff+ "puro", la señal útil está en P(whiff | swing): el diagnóstico del Paso 2 (0.193) sigue siendo el mejor Stuff+ sin ubicación.
2. **Pitching+ v3 rinde igual que v1 con una estructura metodológicamente correcta.** No usa pesos inventados, no usa la etiqueta `is_out_in_pa` copiada a todo el turno ni la regresión directa de `run_value` (R²=0.04), y las probabilidades suman 1. En carreras t+1 es equivalente o un poco mejor (mín. 300: 0.227 vs. 0.194). En K-BB% y whiff% t+1 es algo peor, porque ya no pondera de más los whiffs.
3. **Los sub-modelos con ubicación son fuertes**: P(swing) tiene AUC 0.867 y P(whiff | swing) 0.763. El de calidad de contacto es débil, como predice DIPS.
4. **El benchmark tiene poca potencia.** Las carreras observadas tienen una estabilidad año contra año de apenas 0.085-0.155. Eso pone un techo teórico de r ≈ 0.3-0.4 a cualquier métrica, y los IC de los deltas son de ±0.07. Mejoras reales de ~0.03 no se pueden confirmar con ~440 pares.

**Decisión.**
- **Ninguna variante cumple el criterio de aceptación** (P ≥ 90% de mejora en carreras t+1). **Stuff+ v1 y Pitching+ v1 siguen siendo los oficiales.**
- Queda pendiente una **decisión del usuario**: adoptar Pitching+ v3 por equivalencia de rendimiento y mejor metodología, aunque no por mejora demostrada. Corrige los problemas señalados de pesos arbitrarios, traducción a carreras circular y etiquetas mal condicionadas.

**Cifras de arriba que esto matiza:**
- Sección 2, pesos de Pitching+ (`historial/reports/pitching_plus_score.md`): la cadena v3 obtiene los pesos de las carreras y no de una tabla elegida a mano, con poder predictivo equivalente.
- Sección 4, traducción a carreras: la pendiente de −0.001551 runs/pitch por punto se calculó contra una predicción que forma parte del mismo score. En v3, el score **ya está en carreras** (E[Δ] por pitch), así que la traducción es directa y no circular.

### 7.5 Decisión — Pitching+ v3 adoptado como índice oficial (2026-10-03)

**El usuario aprobó adoptar Pitching+ v3, en su versión completa (sin DIPS), como índice oficial.** La razón es **mejor metodología con rendimiento equivalente**, no una mejora demostrada en el benchmark (ver 7.4). Corrige cuatro problemas de v1:
- pesos elegidos a mano;
- traducción a carreras circular;
- targets mal condicionados;
- la etiqueta `is_out_in_pa` copiada a todos los pitches del turno.

Se elige la versión completa y no la DIPS porque fue la diseñada antes de ver resultados; la mejora de DIPS no fue consistente entre pares de temporadas.

- Scores oficiales: columna `pitching_plus_v3` de `data/processed/run_value_chain_scores.parquet` (OOF).
- Modelos finales para inferencia: `src/models/artifacts/pitching_plus_v3.pkl`. Contiene los 5 sub-modelos entrenados con todos los pitches, los valores Δ por conteo y los coeficientes del ajuste por ubicación. Se generan con `python -m src.models.run_value_chain --train-final`.
- **Stuff+ sigue en v1.** El dashboard y los reportes de scouting todavía usan v1; se migran en el Paso 5.

### 7.6 Paso 4 — Contrafactual de altitud coherente: trayectoria completa (2026-10-03)

**Qué se hizo.** `src/features/trajectory.py` re-simula la trayectoria completa de cada pitch con los 9 parámetros de Trackman (posición, velocidad y aceleración en y=50 pies). Primero se validó que esa reconstrucción reproduce las columnas del dataset: PlateLoc, ZoneSpeed, ZoneTime, VAA, HAA, RelHeight y RelSide con r ≥ 0.997, e IVB, HB y VertBreak con r ≥ 0.995. Para proyectar un pitch a otra categoría de altitud:
1. La aceleración sin gravedad se descompone en **drag** (a lo largo de la velocidad) y **Magnus** (perpendicular). Cada componente se escala por su ratio de densidad efectiva.
2. Se reconstruye el punto de salida y se vuelve a lanzar el pitch al **mismo objetivo en el plato** (supuesto de comando), recalculando la velocidad en zona, el tiempo de vuelo, el VAA, el HAA, el IVB, el HB, el VertBreak y las diferencias contra la recta.
3. Los cambios se aplican como deltas sobre lo observado. Proyectar un pitch a su propia categoría lo deja igual (máx. |Δ| = 1e-05).

`src/analysis/altitude_counterfactual_v2.py` usa esa re-simulación con los modelos finales de Pitching+ v3. Detalle completo en `reports/altitude_counterfactual_v2.md`.

**Hallazgo 1: la densidad del aire se puede medir con el propio dataset, y no coincide con lo supuesto.** Drag/v² y Magnus/v² son proporcionales a la densidad. Comparando al mismo pitcher, con el mismo tipo de pitch y en la misma temporada entre categorías (1,800-2,200 grupos):

| categoría | ratio drag | ratio Magnus | supuesto v1 |
|---|---:|---:|---:|
| Medium vs No | 0.906 (≈1,050 m) | 0.834 | 0.930 (800 m) |
| Extreme vs No | 0.854 (≈1,600 m) | 0.770 | 0.834 (1,900 m) |

El Magnus baja bastante más que el drag; con densidad sola deberían bajar igual. Hay algo más que densidad: calibración de Trackman por estadio, clima o la pelota. **Esto explica la anomalía de Medium Altitude** que v1 dejó abierta en el límite 3 de la sección 6. Esto resuelve, **con los datos entregados**, el límite 1 de la sección 6 (elevaciones supuestas): el contrafactual ya no depende de ninguna elevación supuesta.

**Hallazgo 2: la re-simulación reproduce los cambios observados.** En la validación split-half, los ratios se estiman con la mitad de los pitchers y se predice el cambio No → Extreme en la otra mitad (1,061 grupos):

| feature | cambio observado | cambio predicho | v1 predecía |
|---|---:|---:|---:|
| InducedVertBreak (pulg.) | −2.20 | −2.64 | sí cambiaba |
| HorzBreak (pulg.) | −1.82 | −1.50 | sí cambiaba |
| ZoneSpeed (mph) | +1.24 | +1.20 | 0 |
| VertApprAngle (°) | −0.165 | −0.156 | 0 |
| VertBreak (pulg.) | −1.64 | −2.16 | 0 |

El IVB y el VertBreak se sobreestiman ~20%, pero la dirección y el orden de magnitud son correctos en todas las features. El contrafactual v1 dejaba fijos el VAA y la velocidad en zona.

**Hallazgo 3: el modelo detecta la altitud por la firma física de la trayectoria.** Pitching+ v3 no tiene ninguna variable de altitud. Aun así, de la brecha realizada Extreme − No de **+1.47 carreras/100 pitches**, el modelo atribuye **+1.15** a las features del pitch. Por eso proyectar un pitch a otra categoría captura el **efecto total del entorno de altitud**, y no solo la pérdida de calidad del pitch.

**Efecto de la altitud (Extreme − No, carreras/100 pitches; positivo = peor para el pitcher):**

| tipo | total | calidad del pitch (take+whiff+foul) | bola en juego | calidad del pitch, abridor/temporada | v1: abridor/temporada |
|---|---:|---:|---:|---:|---:|
| Four-Seam | +1.50 | **+0.41** | +1.10 | +11.1 | +3.6 |
| Sinker | +1.28 | +0.27 | +1.01 | +7.5 | +4.5 |
| Changeup | +1.14 | +0.31 | +0.83 | +8.5 | +4.1 |
| Splitter | +0.85 | +0.32 | +0.53 | +8.9 | +3.5 |
| Curveball | +0.82 | +0.25 | +0.58 | +6.8 | +3.5 |
| Cutter | +0.71 | +0.16 | +0.55 | +4.3 | +3.9 |
| Slider | +0.67 | +0.23 | +0.44 | +6.2 | +3.6 |

**Lectura.**
1. **El efecto de la altitud es mucho mayor que lo que estimaba v1.** Solo la parte de calidad del pitch ya vale ~4-11 carreras por temporada para un abridor. El efecto total, con la bola en juego, vale ~18-41.
2. **El tipo más afectado ya no es el sinker, sino la recta de 4 costuras.** Es la que más depende del "levantamiento" por Magnus, y el Magnus es la componente que más baja. Cutter y slider son los que menos pierden.
3. **La bola en juego explica el 60-75% del efecto total.** Esa parte mezcla calidad del contacto con el entorno del parque (vuelo de la pelota, dimensiones), y con estos datos no se puede separar. Se reporta aparte y no debe venderse como "pérdida de calidad del pitch".
4. **Los IC son muy estrechos (±0.01-0.05) porque solo capturan el muestreo de pitchers**, no la incertidumbre del modelo (un solo modelo final, predicciones in-sample para los deltas). Subestiman la incertidumbre real.
5. **Medium Altitude:** la brecha realizada contra No Altitude (+1.18 carreras/100 pitches) es mayor que la que atribuye el modelo (+0.80). Queda espacio para efectos de parque que no pasan por la trayectoria.

**Cifras de arriba que esto supera:**
- Sección 4, tabla "Delta de Pitching+ por tipo de pitch" y tabla de carreras por temporada: el ranking (Sinker primero) y la magnitud (~3.5-4.5 carreras/abridor) quedan reemplazados por las tablas de esta sección.
- Sección 6, límites 1 (elevaciones supuestas) y 3 (anomalía de Medium Altitude): resueltos o explicados con los ratios medidos.

### 7.7 Paso 5 — Índice oficial neutral por altitud, shrinkage, fichas y dashboard (2026-10-03)

**Decisión del usuario:** el índice oficial neutraliza el **efecto total** de la altitud (calidad del pitch + entorno de bola en juego). La pregunta que responde es qué tan bueno es cada pitcher si todos lanzaran al nivel del mar.

**Qué se hizo** (`src/models/pitching_plus_neutral.py`, detalle completo en `reports/pitching_plus_neutral.md`):
1. **Proyección a nivel del mar:** cada pitch se proyecta a No Altitude con el contrafactual del Paso 4 y se valúa con la cadena de Pitching+ v3. Las predicciones son **out-of-fold**: los modelos se entrenan con pitches observados y predicen los pitches proyectados de pitchers que nunca vieron.
2. **Shrinkage empírico-bayesiano** por pitcher-temporada. Reemplaza el filtro duro de 100 pitches: un pitcher con muestra chica queda cerca de 100 con un intervalo ancho, en lugar de quedar fuera del ranking. Con unos **65-80 pitches**, el score de un pitcher ya se encoge a la mitad hacia la media.
3. **Escala:** 10 puntos = 1 desviación estándar de **habilidad real** (τ) en la temporada. Cada pitcher-temporada lleva un **intervalo de 80%**.
4. **Tabla oficial única:** `data/processed/pitching_plus_official.parquet`. Contiene el Pitching+ oficial con su intervalo, el mismo índice con altitud (en la misma escala), el % de pitches en cada categoría, el costo de altitud personal (total y calidad) y el percentil.

**¿Funciona la neutralización?** Correlación del score con el % de pitches del pitcher en altitud extrema (mín. 100 pitches):

| score | corr. con % Extreme |
|---|---:|
| Pitching+ v3 con altitud | −0.333 |
| **Pitching+ v3 neutral + shrinkage (oficial)** | **−0.074** |

Pitchers que lanzan mayormente en Extreme: 94.99 con altitud → **98.68** neutral. Mayormente en No Altitude: 102.43 → **99.45**. Al neutralizar, la diferencia entre entornos pasa de 7.4 a 0.8 puntos. Pitchers con 60-80% de sus pitches en altitud suben más de 200 lugares en el ranking de su temporada.

**Benchmark año-contra-año:**

| métrica | mín. 100: carreras observadas t+1 | mín. 100: carreras ajustadas por entorno t+1 | mín. 300: carreras observadas t+1 | mín. 300: carreras ajustadas t+1 |
|---|---:|---:|---:|---:|
| Pitching+ v3 con altitud | 0.158 | 0.138 | 0.223 | 0.208 |
| **Pitching+ v3 neutral + shrinkage** | 0.132 | **0.153** (Δ +0.015, P=77%) | 0.213 | **0.267** (Δ +0.059, P=98-99%) |

- Contra las **carreras observadas** del año siguiente, el neutral predice un poco peor. Es lo esperado: la mayoría de los pitchers repite parque, y un score que conserva la altitud se beneficia de esa persistencia.
- La prueba justa para un índice neutral es contra las **carreras ajustadas por entorno**: el valor realizado de cada pitch menos el promedio de su categoría y temporada. Ahí **el neutral es mejor**, y con mín. 300 pitches la mejora cumple el criterio de 7.2 (P ≥ 90%).
- Transparencia: esta prueba se definió después de ver el primer resultado. Está motivada por la teoría (qué debe predecir un índice neutral), no elegida para favorecer al modelo.

**Fichas de scouting v2** (`src/reports/scouting_report_v2.py`, `reports/scouting/README.md`):
- 5 mejores Pitching+ y 5 con mayor costo de altitud.
- Cada ficha incluye un resumen por temporada con intervalo y el arsenal con carreras evitadas contra la liga, tanto a nivel del mar como en altitud extrema.
- También incluye el costo de altitud por pitch (calidad y total) y recomendaciones por reglas fijas.
- La regla "en casa conviene apoyarse en X" compara el **valor** del pitch en altitud contra su tipo, no solo cuánto pierde. Una primera versión recomendaba el pitch que menos pierde aunque fuera el peor del arsenal; se corrigió.
- Las fichas v1 quedan en `reports/scouting/v1/`.

**Dashboard v3** (`historial/dashboard/app.py`, `dashboard/scoring.py`):
- **Pitch hipotético:** construye la trayectoria que reproduce el movimiento ingresado en la categoría donde se midió, la re-simula en las 3 categorías con los ratios medidos y la valúa en carreras/100 con Pitching+ v3. La valuación es neutral al conteo (promedio de los 12 conteos ponderado por frecuencia) y compara contra el pitch típico de su tipo. Separa la pérdida de calidad del pitch del entorno de bola en juego.
- **Pitcher existente:** Pitching+ oficial con intervalo, el índice con altitud, la exposición a cada altitud, el arsenal y las recomendaciones.
- Verificado con `streamlit.testing.AppTest`: las dos pestañas corren sin excepciones.
- Supuesto visible en la app: el pitch hipotético va al centro de la zona, lo que exagera el peso del contacto.

**Pruebas automáticas** (`tests/test_trajectory.py`, 6 pruebas, todas pasan):
- identidad del contrafactual;
- dirección física del efecto (menos IVB y más velocidad en zona en altitud);
- orden de los ratios de densidad;
- el pitch sintético reproduce exactamente el movimiento ingresado;
- el shrinkage encoge más a las muestras chicas y recupera τ en datos simulados;
- simetría zurdo/derecho del dashboard.

**Limpieza:** se eliminaron `src/reports/scouting_report.py` y `src/analysis/dashboard_sanity_check.py` (v1), que dependían del mecanismo de scoring v1 del dashboard. Siguen disponibles en el tag `baseline-v1`.

**Cifras de arriba que esto supera:**
- Sección 2, Pitching+ score OOF (media 100, std 6.32, top `pitcher_00469` 116.25): el índice oficial es ahora `pitching_plus` de `pitching_plus_official.parquet`.
- Sección 5, recomendaciones y ejemplos de scouting (`pitcher_00107`, `pitcher_00469`): reemplazados por las fichas v2.
- Sección 6, límite 10 (dashboard con 24 de 33 features supuestas): el dashboard v3 calcula todas las features de trayectoria con física coherente. Los supuestos que quedan son la ubicación, el punto de salida lateral, el drag típico y las diferencias contra la recta.

### 7.8 Stuff+ v4 — P(whiff | swing), neutral por altitud (2026-10-03)

**Qué se hizo** (`src/models/stuff_plus_v4.py`, detalle completo en `reports/stuff_plus_v4.md`). El diseño y el criterio se declararon antes de correr. El modelo estima P(whiff | swing) con las features v2d, sin ubicación ni conteo (AUC OOF sobre swings = 0.651). Se predice para todos los pitches proyectados a nivel del mar, y el promedio por pitcher-temporada lleva el mismo shrinkage y la misma escala que el Pitching+ oficial. `src/models/pitching_plus_neutral.park_adjusted_test` se generalizó para evaluar cualquier score.

**Resultados** (delta pareado contra Stuff+ v1):

| métrica | Stuff+ v1 | Stuff+ v4 (neutral + shrinkage) | Δ |
|---|---:|---:|---|
| Carreras ajustadas por entorno t+1, mín. 100 | 0.184 | 0.149 | −0.035 (P=21%) |
| Carreras ajustadas por entorno t+1, mín. 300 | 0.057 | 0.119 | +0.062 (P=84%) |
| Whiff% t+1, mín. 100 | 0.362 | 0.298 | −0.064 (P=6%) |
| Estabilidad, mín. 100 | 0.668 | 0.769 | +0.100 (P=100%) |

**Hallazgos.**
1. **No cumple el criterio de aceptación** (P ≥ 90% en carreras ajustadas t+1, mín. 100). Con mín. 100 pitches predice peor que v1; con mín. 300 predice mejor, pero sin llegar al umbral.
2. **Es más estable que v1, pero predice peor el whiff%.** Es el mismo patrón del Paso 2: sin ubicación, el modelo mide rasgos persistentes del arsenal que no se convierten del todo en resultados.
3. **Neutralizar por altitud no ayuda a Stuff+** (v4 con altitud: 0.175; neutral: 0.149), a diferencia de Pitching+. En Pitching+ la mejora venía de quitar el entorno de bola en juego, y Stuff+ v4 no tiene ese componente.
4. **Limitación estructural: P(whiff | swing) castiga a los pitchers de contacto.** El sinker promedio tiene 0.15, contra 0.34 de la curva. Un sinkerista que genera rolas débiles queda abajo aunque prevenga carreras. Es parte de por qué predice mal las carreras.

**Decisión:** ver 7.9.

### 7.9 Decisión — Stuff+ oficial = "Stuff+ v1 (incluye ubicación)" (2026-10-03)

**El usuario eligió mantener Stuff+ v1 como el Stuff+ oficial, renombrado con honestidad: "Stuff+ v1 (incluye ubicación)".** Es la versión que mejor predice, pero no es un Stuff+ "puro": una parte importante de su señal es la ubicación del pitch, escondida en el VAA y los ángulos de salida (§7.3). Tampoco está neutralizado por altitud.

- **Índice principal del proyecto:** Pitching+ v3 neutral por altitud (§7.7). Es el único validado contra carreras del año siguiente ajustadas por entorno.
- **Stuff+ v1 (incluye ubicación):** nueva columna `stuff_plus_v1_incluye_ubicacion` de `data/processed/pitching_plus_official.parquet`, con escala 100 + 10·z por temporada (promedio OOF de P(whiff)). Se muestra en las fichas y en el dashboard. Correlación con el Pitching+ oficial: 0.47 (mín. 100 pitches). En el código, el identificador `stuff_plus_v1` se conserva por compatibilidad; las etiquetas del benchmark ya dicen "incluye ubicación".
- **Stuff+ v4 en uso acotado:** P(whiff | swing) solo se usa **dentro de cada tipo de pitch**, en la columna "stuff: whiff/swing vs su tipo (pp)" del arsenal (fichas y dashboard). Esa columna da los puntos porcentuales de swing-and-miss esperados por las características físicas del pitch contra el promedio de la liga del mismo tipo. Así se evita su sesgo contra los pitchers de contacto (sinker 0.15 vs. curva 0.34), que solo aparece al comparar tipos distintos.

### 7.10 Documento técnico reescrito (2026-10-03)

`reports/documento_tecnico.md` se reescribió sobre la versión oficial (Pitching+ v3 neutral por altitud). Reemplaza al borrador basado en v1 y resuelve sus marcas `[PENDIENTE]`. La formulación matemática completa está en `historial/reports/resumen_matematico.md`.

### 7.11 Ajuste de hiperparámetros de Pitching+ v3 (2026-10-03)

**Qué se hizo** (`src/models/tuning.py`, detalle completo en `reports/tuning.md`). El diseño y el criterio se declararon antes de correr. Se usó Optuna (TPE, 25 ensayos por sub-modelo, el primero con la configuración base) para minimizar el log-loss en pitchers de validación, con una partición por pitcher (80/20, seed 7). Durante la búsqueda se usó learning rate 0.1 con early stopping; el modelo final usa 0.05 y el doble de árboles. El benchmark año-contra-año **no** entró en el ajuste.

**Criterio de adopción y resultado:**

| criterio | resultado |
|---|---|
| Log-loss OOF mejora en los 5 sub-modelos | ✅ swing −0.0004, take −0.0013, whiff −0.0005, foul −0.0006, bola en juego −0.0106 |
| La calibración (ECE) no empeora más de 0.002 | ✅ cambios de −0.0005 a +0.0007 |
| El índice oficial no empeora (P(peor) < 90%) | ✅ carreras ajustadas t+1: 0.153 → 0.164 (mín. 100, P(mejor) = 76%); 0.267 → 0.276 (mín. 300, P = 68%) |

**Se adoptó** (`python -m src.models.tuning --activate`). Los parámetros quedaron en `src/models/artifacts/pitching_plus_v3_tuned_params.json`. Con ellos se reentrenaron los modelos finales y se regeneraron el contrafactual, la tabla oficial y las fichas. Las fichas que ya no entran en la selección se borran automáticamente.

**Lectura:**
- La mejora es pequeña y no se puede confirmar en el benchmark.
- El ranking casi no cambia (correlación de rangos 0.967).
- Los modelos elegidos son más regularizados: muchos menos árboles en bola en juego (128 vs. 500) y min_child_samples entre 600 y 1,800 (antes 20).
- La mayor ganancia está en el sub-modelo de resultado en juego, que era el más sobreajustado.

**Cifras que cambian:**

| cifra | antes | ahora |
|---|---:|---:|
| Efecto total de la altitud (carreras/100) | +1.14 | +1.10 |
| Brecha Extreme − No atribuida por el modelo (carreras/100) | +1.15 | +1.09 |
| corr(score neutral, % en Extreme) | −0.074 | −0.054 |
| Carreras ajustadas t+1, mín. 300 | 0.267 | 0.276 (Δ vs. con altitud = +0.067, P = 99%) |

El documento técnico (v3.1) y `resumen_matematico.md` ya tienen las cifras nuevas. Las secciones 7.6 y 7.7 conservan las cifras previas al ajuste como registro histórico.

**Cómo regenerar todo el pipeline v3** (en orden):

```
.venv\Scripts\python -m src.validation.benchmark
.venv\Scripts\python -m src.models.run_value_chain
.venv\Scripts\python -m src.models.tuning              # opcional: re-ajusta; luego --activate
.venv\Scripts\python -m src.models.run_value_chain --train-final
.venv\Scripts\python -m src.analysis.altitude_counterfactual_v2
.venv\Scripts\python -m src.models.pitching_plus_neutral
.venv\Scripts\python -m src.models.stuff_plus_v4
.venv\Scripts\python -m src.reports.scouting_report_v2
.venv\Scripts\python -m pytest tests
.venv\Scripts\streamlit run historial/dashboard/app.py
```

### 7.12 Auditoría matemática y de implementación (2026-10-05)

Se contrastó `resumen_matematico.md` contra el código y se corrieron pruebas numéricas sobre los datos existentes. No se modificó código. Informe completo: `reports/auditoria_matematica.md`.

- **Correcto y verificado numéricamente:** la cadena de probabilidades (Σ P = 1 con error ≤ 6e−16), la recursión de V(b,s), el signo de Δ, la física del contrafactual (descomposición exacta, reproyección analítica sin circularidad, sin features congeladas en el modelo oficial), el shrinkage y la escala.
- **Error encontrado:** el golpeado (HBP) vale ≈ 0 carreras porque `targets.py` usa `is_hit_by_pitch` (siempre 0) para cerrar el turno. Δ(hbp) sale negativo en 7 de 12 conteos y mueve hasta 10 puntos de Pitching+ a algunos pitchers.
- **Leakage temporal:** existe, pero es pequeño. Reentrenando todo solo con temporadas ≤ t, la mejora del índice neutral sobre el índice con altitud se mantiene: Δr = +0.061, P = 98.6% (antes +0.067, P = 98.7%).
- **Matiz a la conclusión principal:** un ajuste trivial por (temporada, categoría de altitud), sin física, predice igual las carreras ajustadas t+1 (0.275 vs. 0.276 con mín. 300; mejor con mín. 100). El criterio pre-registrado de 7.2 (carreras observadas, mín. 100) no se cumple.
- **Pendiente:** el usuario decide qué correcciones aplicar y en qué orden (plan P0–P3 en el informe).

### 7.13 Auditoría inicial del proyecto "CDMX Como Variable" (2026-10-05)

Inventario del proyecto, reproducción de resultados y lo que falta para el nuevo enfoque. Reporte completo: `historial/AUDITORIA_INICIAL.md`.

- Las cifras vigentes se reproducen exactamente (AUC Stuff+ v1 0.7315, ratios ρ̂, −0.327 → −0.054, 0.208 → 0.276).
- **Extreme Altitude no es un solo estadio.** Se reconstruyeron los equipos anónimos con el grafo de pitchers que lanzan juntos (k = 20 en 2024, 97% de pureza). Entre 5 y 6 equipos juegan de local en Extreme. Su ratio de drag dentro de pitcher va de 0.81 a 0.90, así que son estadios físicamente distintos. El dataset no trae estadio, equipo ni clima.
- Lectura descriptiva preliminar de la hipótesis del analista: en Extreme la bola en juego vale +0.090 carreras, contra +0.019 en No Altitude. Con 0-2, la diferencia entre un whiff y una bola en juego es 0.43 carreras en Extreme y 0.34 en No Altitude. Está confundido por equipos y estadios; no es una prueba.
- **Pendiente de decisión del usuario:** usar el calendario público de la LMB para identificar los juegos de un estadio concreto, o trabajar con los estadios de altitud extrema reconstruidos y anónimos.

### 7.14 Correcciones P0 y P1 de la auditoría (2026-10-05)

Detalle completo: `historial/reports/validacion_post_auditoria.md`. Pipeline regenerado desde los datos crudos con `python -m src.pipeline` (21 pasos, todos OK).

- **P0.1 HBP.** `targets.py` reconstruye el golpeado desde PitchCall (0 → 2,276). Su valor pasa de +0.004 a +0.397 carreras, y Δ(hbp) < 0 pasa de 7 conteos a 0. Pitching+ cambia hasta 5.9 puntos (corr 0.988). Las etiquetas y probabilidades de los modelos son idénticas, así que los hiperparámetros siguen siendo válidos (`historial/reports/hbp_antes_despues.md`).
- **P0.3 Baseline obligatorio.** El neutral físico **no supera** al ajuste trivial temporada × categoría: 0.279 vs. 0.279 con mín. 300, y 0.181 vs. 0.191 con mín. 100 (`reports/baselines.md`). **Se retira** la conclusión "la neutralización física mejora la predicción" (§7.7, §7.11).
- **P0.4 Validación preespecificada (§7.2).** No se cumple: neutral 0.153 vs. Pitching+ v1 0.188 (P = 20%). La prueba ajustada por entorno queda como secundaria post-hoc. **Se retira** "cumple el criterio de 7.2" (§7.7).
- **P0.5 Temporal estricta.** Todo entrenado con temporadas < t (pareja 2025→2026). No hay señal de inflación por leakage, y el neutral tampoco supera al trivial (`reports/validacion_temporal.md`).
- **P1:**
  - el techo √r_stab queda como orientación bajo supuestos;
  - intervalo de *credibilidad* empírico-bayesiano;
  - atribución de Shapley del efecto de altitud: antes del contacto +0.39/100, bola en juego +0.71/100. Reemplaza a "calidad del pitch" (+0.32).
  - `stuff_plus_v1` declarado como modelo que usa ubicación implícita y altitud;
  - escala: 100 = pitch promedio de la liga; se conserva;
  - estadios reconstruidos: Extreme = 4–5 estadios, ningún estadio identificable;
  - hipótesis del analista respaldadas en esos estadios: menos whiff, más contacto, menos K, contacto más caro; G_E − G_N = +0.046;
  - la sensibilidad individual a la altitud no es medible (confiabilidad 0.11).
- **Tests:** 21 automáticos pasan (15 nuevos). El test nuevo de ratios inválidos encontró una falla silenciosa en `trajectory.lookup_ratios`, ya corregida.
- **Pendiente (P2/P3, sin tocar):**
  - revisión del shrinkage;
  - efecto de altitud con predicciones OOF;
  - sobrepredicción de IVB;
  - limpieza de duplicados;
  - reescritura completa del documento técnico (v3.2 lleva correcciones puntuales);
  - modelo interpretable de whiff y modelo de K.

### 7.15 Reformulación: Pitching+ para condiciones de altitud extrema · P0 (2026-10-06)

**Decisión del usuario (2026-10-05).**
- Se respeta por completo la anonimización del dataset: no se usará información externa (calendarios, nombres, número de equipos) ni se intentará identificar equipos, estadios o pitchers reales.
- Los resultados se expresan por categoría de altitud y por grupo de estadio anónimo (`ANON_STADIUM_xx`).
- Ventaja: todo se reproduce solo con los datos entregados.

**P0 completo.** Detalle: `reports/p0_altitude_pipeline.md`.
- **HBP:** corregido desde `targets.py`, pipeline regenerado.
- **Grupos anónimos estables** (`src/data/parks.py`, `reports/estadios_anonimos.md`):
  - clustering por temporada con k elegido por un criterio interno;
  - enlace entre temporadas por roster (algoritmo húngaro);
  - 98.4% de los juegos resueltos;
  - 19 de 19 grupos conservan su categoría entre temporadas;
  - test–retest físico de 0.96 (drag) y 0.99 (Magnus).
- **Extreme no es homogéneo:**
  - 8 grupos con drag de 0.83 a 0.90 y Magnus de 0.72 a 0.84 (I² ≥ 95%);
  - las categorías se traslapan físicamente con Medium.
- **Jerarquía de modelos:** M0 sin ajuste; M1 temporada × categoría; M1b temporada × estadio anónimo; M2 contrafactual por categoría; M3a contrafactual con densidad por estadio; M3b M2 + ajuste por estadio.
  - Criterio preespecificado: ningún modelo lo cumple.
  - Validación post-hoc: M3a es el mejor punto (P = 79–88%, no llega al umbral).
  - Validación temporal estricta: **gana M1b en las 4 combinaciones**.
  - La física no tiene valor predictivo adicional demostrado; sí tiene valor explicativo y contrafactual.
- **Tests:** 25 PASS (4 nuevos de grupos anónimos).
- **Pendiente:** P1, que espera indicación del usuario.

### 7.16 P2: perfiles de pitcher para ambientes de altitud (2026-10-06)

Reporte: `reports/p2_pitcher_profiles_altitude.md`. Tablas para la página: `data/web/`.

- **Evidencia de que el juego cambia:** se conservan los hallazgos de P1 (whiff/swing −2.9 pp, K/PA −4.7 pp, RV/BIP +0.068, G +27%). Con los grupos anónimos estables de P0, los cuatro efectos van en la misma dirección en 16/16 estadio-temporadas de Extreme (la cifra previa, 13/13, usaba la reconstrucción anterior a P0). Medium: 14/14, salvo whiff/swing (11/14).
- **Daño de contacto:** P(BIP) × E[RV|BIP] sube de 0.26 a 1.62 carreras/100 pitches en Extreme; ~93% del aumento es daño por BIP, ~1% más BIP. Identidad con el árbol verificada (error 1e-16).
- **Stuff+ Physical** (`src/models/stuff_plus_physical.py`): P(whiff | swing), sin SpeedDrop, lado del bateador marginalizado, ajuste simple por estadio. Ubicación = 1.9% de su varianza (v1: 45.6%); estabilidad 0.76; whiff/swing t+1 mejor que v4 (P = 98%).
- **Location+ v2** (`src/models/location_plus_v2.py`): árbol completo con solo ubicación y contexto; predice BB% t+1 mejor que el propio BB%; casi independiente de Stuff+ (r = −0.04).
- **AEPP**: pesos lineales de carreras por ambiente (sin pesos manuales). Predice carreras en altitud t+1 mejor que Pitching+ simple (0.26 vs 0.15, P = 98%), igual que K−BB%. Pesos de altitud vs. No Altitude: rangos 0.986, sin mejora predictiva. El "bono de altitud" depende del K%.
- **Error corregido durante P2:** el ajuste por entorno devolvía la media de cada temporada; con métricas muy encogidas eso invertía las correlaciones año a año (RV/BIP: −0.72). Ahora devuelve la media global.
- **Negativos:** arquetipos sin grupos claros (silueta 0.18); covariables Fay–Herriot no adoptadas; sensibilidad observada con confiabilidad 0.11 (harían falta ≈ 2,500 pitches en Extreme); sensibilidad modelada vs. observada r = 0.06.
- **Pipeline:** pasos `stuff_physical`, `location_v2`, `profiles` y `p2_report`. **Tests:** 40 PASS (15 nuevos en `tests/test_p2.py`).

### 7.17 Página web final (2026-10-06)

- Plan: `reports/web_architecture_plan.md`. Validación: `reports/web_final_validation.md`.
- **Página:** `web/app.py` (Streamlit), con la narrativa en el orden pedido: evidencia → contacto más caro → perfil swing-and-miss → Stuff+ Physical → Location+ → **Pitching+** → simulador → ambientes → ranking AEPP → sensibilidad exploratoria → física → qué funcionó y qué no. Cada bloque lleva su tipo de evidencia (observado, asociación, predictivo, simulación, exploratorio, limitación).
- **Simulador** (`web/engine.py`):
  - usa los modelos finales (Pitching+ v3, Stuff+ Physical y Location+ v2; los dos últimos se serializan ahora en `src/models/artifacts/`);
  - reconstruye la trayectoria con `synthesize_pitch` y aplica reglas de extrapolación (percentiles, Mahalanobis, soporte de ubicación);
  - usa escala por lanzamiento: 100 = lanzamiento promedio, sd = 10 (verificado con 300 lanzamientos reales: 99.5 / 100.4 / 99.3).
- **Datos:** paso nuevo `web_data` (`summary.json`, escalas, dominio, pitch mix por pitcher y estadios); `parks.py` ahora también escribe `reports/benchmark/estadios_anonimos.json`. Ninguna cifra de la página está escrita a mano.
- **Tests:** 50 PASS (10 nuevos en `tests/test_web.py`).

### 7.18 Página web: ampliación (pitch mix, regiones, estabilidad, incertidumbre) (2026-10-06)

Requisitos complementarios incorporados sin cambiar la estructura (Pitching+ sigue siendo el protagonista). Detalle en `reports/web_final_validation.md` §6.

- **Sección nueva 6:** "¿Qué tipos de lanzamiento conservan mejor su valor en altitud?". Incluye:
  - dispersión Δ whiff/swing vs. costo en Extreme;
  - selector de tipo y comparación por métrica, con IC 95%;
  - hallazgos generados de `pitch_type_altitude_effects.csv`, con la advertencia de que es descriptivo y no causal (Splitter marcado como muestra pequeña).
- **Location+:** gráfica de las cuatro regiones (el corazón pierde +2.92 carreras/100 en Extreme), con la advertencia "no significa nunca lances al centro".
- **Contacto:** rodado como segunda línea de defensa (ahorra 0.140 → 0.202 carreras) y pirámide conceptual del riesgo. Medium se parece a Extreme (RV/BIP +0.073 vs. +0.085; No Altitude +0.015).
- **Estabilidad:** "dos formas de ser bueno". Evitar el contacto (whiff 0.51, K 0.44, BIP 0.52) se repite; controlar el daño (RV/BIP −0.02, hard 0.11, HR/BIP 0.13) no. La ficha del pitcher advierte cuando la explicación usa métricas de contacto.
- **Simulador:**
  - vista rápida (Pitching+, Stuff+, Location+, whiff, BIP);
  - pestaña "¿Por qué?" con P(BIP) × E[RV | BIP] = daño de contacto (identidad probada contra el árbol);
  - pestaña "Más detalles";
  - contexto observado del tipo de pitch y de la región ("promedio del tipo, no destino garantizado").
- **Ranking:** intervalo de AEPP y de lugar, P(top 25), amplitud mediana de 294 posiciones, #1 en 1–3 y tarjetas de los cuatro primeros.
- **Metodología:**
  - Stuff observado / oficial / neutral (en un expander);
  - techo √0.11 ≈ 0.33 de la sensibilidad;
  - split-half del contrafactual (IVB sobrepredicho ~20%);
  - calibración de EV por temporada (hard% 21.5% → 26.2%);
  - validación con 2 pares de temporadas (2026 parcial, 447 juegos);
  - calidad de rivales;
  - sección 14, "Qué mejoraríamos con más datos", que incluye Expected Contact Quality.
- **Datos:** `summary.json` gana `valor_bip`, `ahorro_rodado`, `stuff_versiones`, `sensibilidad_extra`, `tipos_pitch`, `contrafactual_split_half`, `temporadas` y `ranking_top`. El resto de `data/web/` se reproduce idéntico.
- **Tests:** 53 PASS (3 nuevos y la prueba de la página ampliada).

### 7.19 Extensión: Pitching+ general vs. altitud, lanzamiento óptimo por ambiente y guía para presentar (2026-10-12)

- **Guía:** `reports/guia_para_presentacion.md`. Cada concepto en cuatro niveles: Stuff+ y sus versiones, Location+, Pitching+, RV/BIP, daño de contacto, AEPP, K−BB%, contrafactual, sensibilidad, causalidad y las dos extensiones. Incluye 23 preguntas difíciles con respuesta breve y extendida. Un test verifica que sus cifras coincidan con los datos.
- **Paso nuevo del pipeline:** `pitch_environment` (`src/analysis/pitch_environment.py` y `src/reports/pitch_environment_report.py`).
- **Parte B** (`reports/pitching_plus_general_vs_altitude.md`):
  - general = valores oficiales de la liga (diferencia 0);
  - Extreme = recursión RE24 con datos de Extreme más la calibración del costo de contacto (+0.0095);
  - mismo lanzamiento: correlación de rangos 0.997;
  - la ganancia relativa en Extreme se concentra en el swing-and-miss con dos strikes (+0.70).
- **Parte C** (`reports/optimal_pitch_by_environment.md`):
  - el óptimo se busca entre lanzamientos reales, con perfil del top 1%, vecinos reales, bootstrap por pitcher, contextos 0-0, 0-2 y 2-0, y experimentos de física sola y de física + ubicación;
  - con los costos de Extreme el óptimo cambia poco en 21/21 combinaciones; con la física aproximada, en 17/21;
  - mejor Stuff ≠ mejor pitch.
- **Página:** secciones nuevas 9 ("¿Qué cambia en nuestro Pitching+?") y 10 ("¿Es el mismo un buen lanzamiento en cualquier estadio?"); las siguientes se renumeraron a 11–16.
- **Tests:** 62 PASS.

### 7.20 Preparación de la entrega (2026-10-19)

- **Revisión contra los entregables del reto.** Se completaron los faltantes:
  - validación fuera de muestra del modelo final;
  - recomendaciones operativas: formas de pitch, contacto duro y HR, mapa de movimiento, arsenales y perfil de prospecto;
  - documento técnico nuevo de 7 páginas, con PDF.
- **Validación fuera de muestra** (`reports/validacion_fuera_de_muestra.md`):
  - mejora del MSE sobre el modelo nulo por conteo de 4.1–4.8%;
  - pendiente de calibración 0.96–1.00 en pitcher, season y park holdout;
  - en el park holdout de Extreme el modelo subestima un poco el valor realizado (el contacto cuesta más), coherente con P2 §7.2.
- **Formas de pitch:** los pitches que más dependen del giro pierden más en Extreme. Ejemplo: recta de mucho ride, −7.7 de Stuff+ y −7.9 pp de whiff/swing, contra −3.8 y −1.4 pp con poco ride.
- **Nombres:** decisión del usuario. Se usa "Extreme Altitude" y se explica que corresponde a la pregunta del reto sobre la Ciudad de México, sin identificar estadios. Se quitaron los nombres reales de todos los documentos.
- **Limpieza:** `historial/` con índice; README como índice de entrega; 42 capturas nuevas.
- **Tests:** 66 PASS.

