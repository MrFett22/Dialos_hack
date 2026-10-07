# Documento técnico — Pitching+ LMB neutral por altitud

> **⚠️ Versión 3.2 (2026-10-05), corregida después de la auditoría.** El resumen ejecutivo y las secciones marcadas con "(corregido)" reflejan el pipeline regenerado después de corregir el valor del golpeado y la validación con baselines obligatorios (`reports/validacion_post_auditoria.md`). Las tablas no marcadas conservan cifras de la v3.1 y pueden diferir en el segundo decimal. Esta versión **retira** cuatro conclusiones de la v3.1:
> - "el índice neutral predice mejor": no supera a un ajuste trivial por entorno;
> - "cumple el criterio de aceptación": el criterio preespecificado no se cumple;
> - "calidad del pitch" del efecto de altitud: era una descomposición incompleta;
> - "Extreme = Harp Helú": son 4–5 estadios anónimos.
>
> **Versión 3.1 (2026-10-03), con hiperparámetros ajustados.** Reemplaza a la versión basada en los modelos v1. Cada cifra cita su reporte fuente. El historial completo de decisiones, con lo que se probó y no funcionó, está en `reports/resultados_finales.md` §7. La formulación matemática detallada está en `reports/resumen_matematico.md`.

---

## Resumen ejecutivo (corregido)

- **Índice oficial: Pitching+ v3 neutral por altitud.**
  - Cada pitch se valúa en carreras con una cadena de probabilidades: ¿tira el bateador? → ¿falla? → ¿foul o en juego? → ¿qué resultado?
  - Antes de valuarlo, el pitch se **proyecta a nivel del mar** re-simulando su trayectoria completa.
  - Escala: 100 = pitch promedio de la liga (media ponderada por pitches); 10 puntos = 1 desviación estándar de habilidad real.
  - Cada pitcher lleva un intervalo de **credibilidad** empírico-bayesiano de 80%, que refleja su tamaño de muestra, no la incertidumbre de los modelos.
- **La densidad del aire se midió con los propios datos**, sin suponer elevaciones: en la categoría Extreme, el drag es 85.4% y el Magnus 77.0% del de No Altitude. Pero **Extreme agrupa 4–5 estadios locales distintos** (ratios de drag de 0.81 a 0.90), no identificados como el Harp Helú.
- **Altitud extrema contra nivel del mar, mismo pitch:** +1.10 carreras por cada 100 pitches.
  - +0.39 ocurren antes del contacto (menos whiffs, más pelotas en juego);
  - +0.71 vienen del resultado de la bola en juego, que mezcla calidad del contacto y parque.
  - Es una atribución de Shapley, contable y no causal.
- **Valor predictivo:**
  - Con el criterio preespecificado (carreras observadas del año siguiente, mín. 100) **ningún índice supera a Pitching+ v1**.
  - Contra carreras ajustadas por entorno (prueba post-hoc), el índice neutral supera al índice sin neutralizar, pero **un ajuste trivial por temporada × categoría logra exactamente lo mismo** (0.279 vs. 0.279 con mín. 300).
  - El contrafactual físico no aporta valor predictivo adicional; su valor es explicativo.
- **Hipótesis del analista de los Diablos**, respaldada en los datos de esos estadios (mismo pitcher contra el mismo bateador):
  - en Extreme hay menos whiffs (−2.9 pp por swing), más contacto en juego (+4.0 pp) y menos ponches (−4.7 pp por turno);
  - cada pelota en juego vale +0.068 carreras más;
  - evitar el contacto vale ≈ 27% más;
  - es asociación, no efecto causal aislado de la densidad del aire.
- **Stuff+ v1** (`stuff_plus_v1`) es un modelo predictivo de whiff que **usa ubicación implícita y la categoría de altitud**. No es una medida de "stuff" puro ni está neutralizado.

---

## 1. Problema y marco teórico

### 1.1 La escala "plus"

Como ERA+ u OPS+, un índice "plus" normaliza una métrica para que **100 sea el promedio de la liga** en esa temporada. En este proyecto (corregido), 100 = **el pitch promedio de la liga**: la media de E[Δ] ponderada por pitches de los pitchers con ≥100. El promedio simple por pitcher queda en ≈ 98–99, porque los pitchers de poco volumen son peores. **10 puntos equivalen a una desviación estándar de habilidad real** entre pitchers (τ, sección 6.2). Un Pitching+ de 120 significa que el pitcher está dos desviaciones estándar arriba del promedio.

### 1.2 Por qué importa la altitud

El movimiento de un pitch viene de dos fuerzas del aire:
- el **efecto Magnus**, perpendicular a la velocidad, que genera el quiebre;
- el **arrastre (drag)**, opuesto a la velocidad, que frena la pelota.

Las dos son proporcionales a la densidad del aire ρ. A mayor altitud hay menos densidad, y por lo tanto menos quiebre y menos frenado. Además, la pelota bateada vuela más. La LMB tiene equipos en estadios a altitudes muy distintas; el caso más extremo es el Estadio Alfredo Harp Helú de los Diablos Rojos (~2,240 msnm).

### 1.3 Limitación de los datos

El dataset no identifica el estadio. Solo trae `altitude_category` con 3 valores: No, Medium y Extreme Altitude. La categoría Extreme tiene **28.9% de los pitches** (`reports/eda_summary.md`), mucho más de lo que aportaría un solo estadio en una liga de 20 equipos, así que **casi seguro agrupa varios parques altos**. Las reglas del hackathon no permiten pedir datos adicionales, de modo que toda la corrección por altitud se construyó con la información entregada (sección 5.1).

---

## 2. Datos

| concepto | valor | fuente |
|---|---:|---|
| Pitches crudos | 635,002 (1,134 pitchers) | `eda_summary.md` |
| Pitches válidos | 627,736 (1,045 pitchers) | `splits_summary.md` |
| Temporadas | 2024 (243,254) · 2025 (256,144) · 2026 parcial (135,604) | `eda_summary.md` |
| Pitcher-temporadas | 1,628 | `pitching_plus_neutral.md` |
| Turnos reconstruidos | 162,035 | `run_value_validation.md` |

**Exclusiones.**
- 5,711 filas (0.90%) con identificador de pitcher nulo o categoría de altitud nula (`pitcher_id_null_investigation.md`).
- En la cadena de eventos, además, 551 pitches con `PitchCall` indefinido.
- Los eventos se definen a partir de `PitchCall` y no de `is_swing`: hay ~1,400 filas donde las dos columnas se contradicen.

**Run value.** Se reconstruyó una matriz de expectativa de carreras de 24 estados (bases × outs) para la LMB, validada contra la tabla RE24 de MLB. Durante la reconstrucción se corrigió un error en el avance de corredores en hits: las inconsistencias bajaron de 19% a 0.97% (`run_value_validation.md`).

---

## 3. Cómo se decide si un modelo es mejor

Un índice "plus" no se juzga por su precisión pitch a pitch (AUC), sino por dos propiedades a nivel pitcher-temporada (`src/validation/benchmark.py`, `reports/benchmark_baseline_v1.md`):

- **Estabilidad:** r_stab = corr(métrica en *t*, misma métrica en *t+1*).
- **Poder predictivo:** r_pred = corr(métrica en *t*, resultado **observado** en *t+1*). El resultado principal son las carreras evitadas por cada 100 pitches.

Los pares de temporadas son 2024→2025 y 2025→2026: 437 pares con ≥100 pitches en ambas temporadas y 245 con ≥300. La incertidumbre se mide con un bootstrap de 2,000 réplicas que remuestrea **pitchers**. Dos métricas se comparan con un **bootstrap pareado**.

**Criterio de aceptación, fijado antes de ver resultados:** un cambio se acepta si mejora el poder predictivo con probabilidad P ≥ 90%. Un cambio que solo hace la métrica más estable no se acepta.

**Línea base v1** (≥100 pitches):

| métrica | estabilidad | predice carreras t+1 | predice whiff% t+1 |
|---|---:|---:|---:|
| Stuff+ v1 | 0.668 | 0.181 | 0.362 |
| Pitching+ v1 | 0.545 | 0.169 | 0.172 |
| CSW% observado | 0.349 | 0.205 | 0.357 |
| Whiff% observado | 0.487 | 0.196 | 0.487 |

Los modelos v1 eran más estables que las estadísticas tradicionales, pero no predecían mejor.

**Estabilidad y poder predictivo (corregido: no es un techo universal).** Las carreras observadas de un pitcher casi no se repiten de un año a otro (r_stab ≈ 0.08–0.16). Bajo el modelo de medición clásico (o = talento + ruido independiente), r_pred ≤ √(r_stab/ρ_T), donde ρ_T es la correlación del talento real entre años. Solo si el talento no cambia (ρ_T = 1) la cota es √r_stab ≈ 0.3–0.4. Es una orientación del orden de magnitud, no una ley. Los IC de las comparaciones son de ±0.07: mejoras reales pero pequeñas no se pueden confirmar con esta muestra.

---

## 4. Modelos

### 4.1 Lecciones que dieron forma al diseño final

1. **Ubicación escondida** (`reports/stuff_plus_v2.md`). El ángulo vertical de llegada (VAA), la variable más importante de Stuff+ v1, tiene una correlación de **0.78** con la altura a la que cruza el pitch. Al reemplazarlo por su residual a ubicación constante,

   VAA_adj = VAA − (a_t + b_t · PlateLocHeight)   (ajustado por tipo de pitch t),

   y quitar los ángulos de salida, el AUC baja de **0.732 a 0.640**. La mayor parte de lo que predecía Stuff+ v1 era ubicación.
2. **Simetría de zurdos.** En los zurdos se reflejan las variables horizontales (HB, RelSide, ángulos horizontales y ubicación lateral) y SpinAxis → 360° − SpinAxis.
3. **DIPS** (`reports/run_value_chain.md`). En un índice sin ubicación, el 66% de la diferencia entre pitchers venía del resultado de la bola en juego. Ese componente es muy estable (0.71) pero casi no predice el año siguiente (0.04): los pitchers controlan poco qué pasa con el contacto.

### 4.2 Pitching+ v3: cadena de probabilidades valuada en carreras

`src/models/run_value_chain.py`. Cada pitch termina en uno de 10 eventos: bola, strike cantado, golpeado, whiff, foul, out, 1B, 2B, 3B o HR. Sus probabilidades se obtienen encadenando 5 modelos LightGBM, cada uno entrenado en su propio universo:

| sub-modelo | universo de entrenamiento | AUC (OOF, parámetros base) |
|---|---|---:|
| P(swing) | todos los pitches | 0.867 |
| resultado del take (bola, strike, golpeado) | takes | multiclase |
| P(whiff \| swing) | swings | 0.763 |
| P(foul \| contacto) | contactos | 0.623 |
| resultado en juego (out, 1B, 2B, 3B, HR) | bolas en juego | multiclase |

Las probabilidades de los 10 eventos se combinan así (π_sw = P(swing), π_wh = P(whiff | swing), π_f = P(foul | contacto)):

  P(whiff) = π_sw · π_wh
  P(foul) = π_sw · (1 − π_wh) · π_f
  P(resultado k en juego) = π_sw · (1 − π_wh) · (1 − π_f) · π_bip(k)

y suman 1 por construcción.

**Valor en carreras de cada evento.** Los valores terminales son el run value RE24 promedio de los turnos que terminan con ese evento: BB +0.381, K −0.328, out −0.301, 1B +0.515, 2B +0.832, 3B +1.037, HR +1.479. El valor de cada conteo sale de una recursión hacia atrás:

  V(b, s) = Σₑ π_bs(e) · v_siguiente(e, b, s)

Con 2 strikes, el foul deja el conteo igual (auto-lazo), así que V(b, 2) = Σ_{e≠foul} π · v / (1 − π_foul). Verificación: V(0-0) = +0.0142, igual al promedio observado por turno.

**Valor del pitch y score.** El evento e vale Δ(e, b, s) = v_siguiente − V(b, s), y el valor esperado del pitch es:

  E[Δᵢ] = Σₑ P̂(e | xᵢ) · Δ(e, bᵢ, sᵢ)

El score es el promedio de E[Δ] del pitcher, con el signo invertido para que más alto sea mejor.

**Variables:**
- características físicas sin redundancias: velocidad, spin, ángulo de spin, punto de salida, extensión, IVB, HB, VAA y HAA ajustados por ubicación, pérdida de velocidad;
- diferencias contra la recta del pitcher;
- mismo lado o lado contrario respecto al bateador;
- tipo de pitch, ubicación y conteo.

**Por qué reemplazó a v1.** Predice igual (carreras t+1: 0.154 vs. 0.169 con ≥100 pitches; 0.227 vs. 0.194 con ≥300) y corrige cuatro defectos de v1:
- pesos de componentes elegidos a mano;
- traducción circular a carreras;
- targets definidos sobre el universo equivocado;
- la etiqueta "out en el turno" copiada a todos los pitches del turno.

Además, el score ya está en carreras, así que su traducción a carreras es directa.

### 4.3 Stuff+

Se probó un Stuff+ "puro" (v4): P(whiff | swing) sin ubicación ni conteo, neutral por altitud (`reports/stuff_plus_v4.md`). Es más estable que v1 (0.769 vs. 0.668), pero **predice peor** las carreras del año siguiente (0.149 vs. 0.184; P = 21%) y el whiff% (0.298 vs. 0.362). Además castiga a los pitchers de contacto: el sinker promedio tiene P(whiff | swing) = 0.15, contra 0.34 de la curva.

**Decisión:**
- El Stuff+ vigente es **Stuff+ v1** (`stuff_plus_v1`; corregido): P(whiff) promedio por pitcher-temporada en escala 100 + 10·z. Es un modelo predictivo cuyas features incluyen ángulos de salida y llegada (ubicación implícita) y la categoría de altitud (`altitude_level`), así que no es independiente de ninguna de las dos.
- P(whiff | swing) se usa **solo para comparar pitches del mismo tipo**, en puntos porcentuales contra el promedio de la liga para ese tipo.
- Stuff+ v1 y Pitching+ oficial tienen una correlación de 0.49: miden cosas relacionadas pero distintas.

---

## 5. Efecto de la altitud

### 5.1 La densidad del aire, medida con los datos

`src/features/trajectory.py`, `reports/altitude_counterfactual_v2.md`.

**Trayectoria.** Trackman describe cada pitch con 9 parámetros: posición, velocidad y aceleración en y = 50 pies. Con aceleración constante, r(t) = r₀ + v₀t + ½at², se reconstruyen las mediciones del dataset con r ≥ 0.995.

**Drag y Magnus.** La aceleración sin gravedad, a_ng = a + gẑ, se separa en dos componentes:
- drag: D = (a_ng · v̂)·v̂, a lo largo de la velocidad;
- Magnus: M = a_ng − D, perpendicular a la velocidad.

Como |D| / |v|² y |M| / |v|² son proporcionales a la densidad del aire, comparar **al mismo pitcher, con el mismo tipo de pitch y en la misma temporada** entre categorías estima la densidad efectiva:

| categoría vs. nivel del mar | ratio drag | ratio Magnus | grupos | supuesto anterior |
|---|---:|---:|---:|---:|
| Medium | 0.906 (≈1,050 m) | 0.834 | 1,832 | 0.930 (800 m) |
| Extreme | 0.854 (≈1,600 m) | 0.770 | 2,165 | 0.834 (1,900 m) |

Los IC son de ±0.003. Si solo cambiara la densidad, drag y Magnus bajarían igual. **El Magnus baja bastante más**, sobre todo en Medium Altitude, así que hay algo más: calibración de Trackman por estadio, clima o la pelota. Esto explica la anomalía de Medium Altitude que la versión anterior dejó sin resolver. Los ratios son **efectivos** y reproducen lo que realmente miden los datos.

### 5.2 Contrafactual: el mismo pitch en otra altitud

Para proyectar un pitch de la categoría c a la c′:

1. Las aceleraciones se escalan: a′ = D · ρ̂_D(c′)/ρ̂_D(c) + M · ρ̂_M(c′)/ρ̂_M(c) − gẑ.
2. El pitch sale del **mismo punto** y se lanza al **mismo objetivo en el plato** (supuesto de comando), corrigiendo la dirección de salida.
3. Se recalculan la velocidad en zona, el tiempo de vuelo, VAA, HAA, IVB, HB, el quiebre total y las diferencias contra la recta.
4. Los cambios se aplican como diferencias sobre lo observado, así que proyectar un pitch a su propia categoría lo deja intacto (máx. |Δ| = 10⁻⁵).

**Validación con pitchers que no se usaron para medir los ratios** (split-half, No → Extreme, 1,061 grupos):

| feature | cambio observado | cambio predicho |
|---|---:|---:|
| IVB (pulg.) | −2.20 | −2.64 |
| HB (pulg.) | −1.82 | −1.50 |
| Velocidad en zona (mph) | +1.24 | +1.20 |
| VAA (°) | −0.165 | −0.156 |

La dirección y el orden de magnitud son correctos en todas las features; el IVB se sobreestima ~20%. El contrafactual anterior solo cambiaba IVB y HB, y predecía cambio cero en la velocidad en zona y el VAA.

### 5.3 Cuánto cuesta la altitud, por tipo de pitch

Cada pitch se proyecta a Extreme y a No Altitude y se valúa con Pitching+ v3. La tabla muestra la diferencia en carreras por cada 100 pitches; positivo = más carreras permitidas.

(Corregido.) Atribución de Shapley por eslabón de la cadena (`src/models/chain_attribution.py`). "Antes del contacto" = swing + take + whiff + foul, incluido el aumento de la probabilidad de que la pelota quede en juego. La versión anterior ("calidad del pitch" = take + whiff + foul por grupo de eventos) omitía ese término.

| tipo | total [IC 95%] | antes del contacto | bola en juego | antes del contacto, abridor/temporada* |
|---|---|---:|---:|---:|
| Four-Seam | +1.45 [1.41, 1.48] | **+0.52** | +0.93 | +14.3 |
| Changeup | +1.14 [1.12, 1.17] | +0.41 | +0.73 | +11.4 |
| Splitter | +0.90 [0.85, 0.95] | +0.40 | +0.50 | +11.0 |
| Sinker | +1.20 [1.17, 1.22] | +0.34 | +0.86 | +9.4 |
| Curveball | +0.76 [0.72, 0.79] | +0.32 | +0.43 | +8.9 |
| Slider | +0.65 [0.64, 0.66] | +0.27 | +0.38 | +7.4 |
| Cutter | +0.68 [0.65, 0.71] | +0.17 | +0.51 | +4.7 |
| **Todos** | **+1.10 [1.08, 1.11]** | +0.39 | +0.71 | |

\* 2,750 pitches por temporada.

**Lectura.**
- **La recta de 4 costuras es la que más pierde.** Es la que más depende del "levantamiento" por Magnus, y el Magnus es la componente que más baja. Cutter y slider son los que menos pierden.
- **La columna "bola en juego" mezcla dos cosas:** la calidad del contacto y el entorno del parque (la pelota vuela más). Con estos datos no se pueden separar, así que no debe presentarse como pérdida de calidad del pitch.
- **Por qué existe esa mezcla:** Pitching+ v3 no tiene ninguna variable de altitud, pero la **detecta por la firma física de la trayectoria**. De la brecha observada entre Extreme y No Altitude (+1.47 carreras/100), el modelo atribuye +1.09 a las características del pitch.

**Consistencia con la evidencia observacional de la versión anterior:**
- El test de Cochran-Mantel-Haenszel controlando por tipo de pitch da un odds ratio de whiff en altitud de 0.906 (p < 0.001; `altitude_whiff_stratified.md`).
- El modelo bayesiano jerárquico estima +0.017 a +0.018 carreras/pitch en altitud extrema (`bayesian_hierarchical.md`), es decir +1.7 a +1.8 carreras/100. Es del mismo orden que el efecto total actual (+1.10) y que la brecha observada (+1.47). La "discrepancia de 1.8-2x" que la versión anterior reportaba contra su contrafactual se explica porque ese contrafactual **subestimaba** el efecto: solo cambiaba IVB y HB.

---

## 6. Índice oficial: Pitching+ v3 neutral por altitud

`src/models/pitching_plus_neutral.py`, `reports/pitching_plus_neutral.md`. Tabla oficial: `data/processed/pitching_plus_official.parquet`.

### 6.1 Construcción

1. Cada pitch se proyecta a nivel del mar (sección 5.2). Por decisión del proyecto, se neutraliza el **efecto total** de la altitud: la calidad del pitch y el entorno de bola en juego.
2. Se predice E[Δ] con predicciones **out-of-fold**: los modelos se entrenan con pitches observados y predicen los pitches proyectados de pitchers que nunca vieron.
3. Para cada pitcher-temporada se calcula x̄ = promedio de E[Δ] de sus pitches.

### 6.2 Shrinkage e intervalos

Modelo empírico-bayesiano normal-normal, estimado por temporada. Sean σ² la varianza pitch a pitch dentro de cada pitcher y τ² la varianza real entre pitchers:

  τ² = Var(x̄) − E[σ²/n]
  B = (σ²/n) / (σ²/n + τ²)
  θ̂ = μ + (1 − B)(x̄ − μ)
  Pitching+ = 100 + 10 · (μ − θ̂)/τ

El intervalo de **credibilidad** empírico-bayesiano de 80% (corregido; antes llamado "IC 80%") es ±1.2816 · 10 · sd_post/τ, con sd_post = √((1 − B)·σ²/n). Es la posterior con μ, τ y σ fijos en sus estimaciones. **No incluye** la incertidumbre de esos parámetros, de los modelos ML, de los ratios de densidad ni de los hiperparámetros, así que es más angosto que la incertidumbre total.

Estimaciones: τ = 0.0056-0.0063 carreras/pitch y σ ≈ 0.051, así que **con unos 65-85 pitches el score de un pitcher se encoge a la mitad hacia el promedio**. Un pitcher con muestra chica ya no queda fuera del ranking: queda cerca de 100 y con un intervalo ancho (mediana del ancho = 11.2 puntos).

### 6.3 Validación

**¿Quita el efecto de la altitud?** Correlación del score con el % de pitches que el pitcher lanzó en altitud extrema (≥100 pitches):

| score | correlación | pitchers mayormente en Extreme (311) | mayormente a nivel del mar (560) |
|---|---:|---:|---:|
| con altitud | −0.327 | 95.13 | 102.24 |
| **neutral (oficial)** | **−0.054** | **99.06** | **99.14** |

La diferencia entre entornos pasa de 7.1 a 0.1 puntos.

**¿Predice mejor? (corregido)** Fuente: `reports/baselines.md`.

1. **Validación preespecificada** (§7.2 de la bitácora: carreras observadas t+1, mín. 100, contra Pitching+ v1, P ≥ 0.90). **No se cumple.** Pitching+ v1 0.188; con altitud 0.177; ajuste trivial 0.169; **neutral 0.153** (Δ = −0.035, P = 20%).
2. **Validación secundaria post-hoc** (carreras t+1 ajustadas por entorno). Se incluye el baseline obligatorio B1: E[Δ] observado menos su media por temporada × categoría, sin física.

| mínimo de pitches | con altitud (B0) | ajuste trivial (B1) | **neutral físico (oficial)** | neutral − B0 (P) | neutral − B1 (P) |
|---|---:|---:|---:|---|---|
| 100 | 0.165 | 0.191 | 0.181 | +0.016 (78%) | −0.010 (20%) |
| 300 | 0.215 | 0.279 | 0.279 | +0.065 (99%) | +0.000 (53%) |

**La ventaja del índice neutral sobre el índice con altitud la logra igual un ajuste trivial por entorno**, y la prueba ajustada favorece por construcción a cualquier ajuste por entorno. Con entrenamiento temporal estricto (todo con temporadas < t) el resultado es el mismo (`reports/validacion_temporal.md`). **Conclusión: el contrafactual físico no aporta valor predictivo adicional demostrado.** Su valor es explicativo: por qué cambia el pitch y en qué eslabón de la cadena se pierde valor.

### 6.4 Top 10 (≥300 pitches)

| pitcher | temporada | pitches | **Pitching+** [credibilidad 80%] | con altitud | Stuff+ v1 | % en Extreme |
|---|---:|---:|---|---:|---:|---:|
| pitcher_00619 | 2024 | 1,464 | **131.1** [128.5, 133.8] | 125.1 | 131.6 | 58% |
| pitcher_00465 | 2026 | 357 | **128.3** [123.2, 133.3] | 128.2 | 123.0 | 10% |
| pitcher_00069 | 2025 | 807 | **123.7** [119.7, 127.6] | 119.3 | 104.1 | 44% |
| pitcher_00115 | 2025 | 623 | **120.6** [116.2, 125.0] | 116.6 | 110.1 | 9% |
| pitcher_00680 | 2024 | 562 | **120.1** [116.0, 124.3] | 113.5 | 106.5 | 63% |
| pitcher_00714 | 2025 | 660 | **119.7** [115.4, 124.0] | 113.1 | 105.7 | 22% |
| pitcher_00108 | 2025 | 343 | **119.3** [113.6, 124.9] | 122.5 | 108.7 | 7% |
| pitcher_00036 | 2026 | 528 | **119.1** [114.9, 123.4] | 123.9 | 103.0 | 0% |
| pitcher_00081 | 2025 | 1,622 | **118.7** [115.8, 121.5] | 118.2 | 106.1 | 10% |
| pitcher_00019 | 2025 | 1,455 | **118.6** [115.6, 121.6] | 113.6 | 122.0 | 68% |

Los pitchers que lanzan mucho en altitud extrema suben al neutralizar: `pitcher_00680` (63% en Extreme) gana 6.6 puntos y `pitcher_00019` (68%) gana 5.0.

---

## 7. Herramientas para el cuerpo técnico

**Fichas de scouting** (`src/reports/scouting_report_v2.py`, `reports/scouting/`). Hay 10 fichas: los 5 mejores Pitching+ y los 5 con mayor costo de altitud. Cada una incluye:
- resumen por temporada con intervalo, percentil y exposición a cada altitud;
- arsenal por tipo de pitch, con:
  - stuff contra su mismo tipo;
  - carreras evitadas contra la liga a nivel del mar y en altitud extrema;
  - costo de altitud, separado en calidad del pitch y total;
- recomendaciones por reglas fijas, sin LLM. Por ejemplo, qué pitch vale más en casa comparado con su mismo tipo en ese entorno.

**Dashboard** (`dashboard/app.py`):
- **Pitch hipotético.** Construye la trayectoria que reproduce el movimiento medido, la re-simula en las 3 altitudes y la valúa en carreras por cada 100 pitches. La valuación es neutral al conteo (promedio de los 12 conteos ponderado por su frecuencia). Muestra la comparación contra el pitch típico de su tipo y separa calidad del pitch de entorno. Supuesto visible: el pitch va al centro de la zona.
- **Pitcher existente.** Índice oficial con intervalo, arsenal y recomendaciones.

---

## 8. Límites

1. **Los ratios de densidad son efectivos, no físicos.** Mezclan densidad, calibración de Trackman por estadio, clima y pelota. Además, cada categoría agrupa varios parques que no se pueden distinguir.
2. **El efecto de bola en juego no se puede separar del parque.** La columna "bola en juego" de 5.3 incluye el vuelo de la pelota y las dimensiones del parque. Solo la columna de calidad del pitch se atribuye con confianza al pitch.
3. **Los IC del efecto de altitud son demasiado estrechos.** Solo capturan qué pitchers entran en la muestra, no la incertidumbre del modelo (un solo modelo final, predicciones in-sample para las diferencias).
4. **El contrafactual sobreestima el cambio de IVB en ~20%.** Supone aceleración constante y que el pitcher corrige su dirección para llegar al mismo punto del plato.
5. **El benchmark tiene poca potencia.** Con ~440 pares y carreras casi no repetibles, mejoras reales de ~0.03 en r no se pueden confirmar. (Corregido) La mejora del índice neutral con ≥300 pitches la iguala un ajuste trivial por entorno, y el criterio preespecificado no se cumple.
6. **Stuff+ v1 usa ubicación implícita y la categoría de altitud** y no está neutralizado. Un Stuff+ sin ubicación predijo peor con estos datos (sección 4.3).
9. **(Nuevo) "Extreme" no es el Harp Helú.** Agrupa 4–5 estadios locales reconstruidos con densidades efectivas distintas (`reports/estadios_anonimos.md`). Todo resultado es un efecto promedio de altitud extrema.
10. **(Nuevo) La sensibilidad individual a la altitud no es medible** con estos datos (confiabilidad 0.11; `reports/sensibilidad_altitud.md`). La que implica el modelo describe el repertorio, no una habilidad demostrada.
7. **Calidad de datos.** 2026 es una temporada parcial; la anonimización de 2026 está incompleta; hay 447 pitches marcados como whiff y contacto a la vez; 21 pitches tienen una extensión imposible (>10 pies) y conservan sus features observadas en el contrafactual.
8. **El ajuste de hiperparámetros mejora poco.** Se ajustaron los 5 sub-modelos con Optuna (`reports/tuning.md`). El log-loss out-of-fold mejoró en los 5, pero poco (entre −0.0004 y −0.0106), y la mejora en el índice no se puede confirmar (P = 68-79%). El ranking casi no cambia: correlación de rangos de 0.967 con el índice sin ajustar. Los hiperparámetros se eligieron con una partición de pitchers que también participa en la evaluación out-of-fold, así que la estimación del log-loss es ligeramente optimista.

---

## 9. Reproducibilidad

- **Entorno:** `.venv` con Python 3.14; versiones exactas en `requirements-lock.txt`.
- **Versión anterior:** repositorio git con la línea base v1 en el tag `baseline-v1`.
- **Pruebas:** `tests/test_trajectory.py` (6 pruebas) cubre la identidad del contrafactual, la dirección física del efecto, el pitch sintético, el shrinkage y la simetría zurdo/derecho.

Secuencia para regenerar todo:

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
.venv\Scripts\streamlit run dashboard/app.py
```

---

## Apéndice — Historial de versiones

| versión | qué cambió | resultado contra el criterio | estado |
|---|---|---|---|
| v1 | Stuff+ (whiff), Pitching+ (8 targets, pesos a mano), contrafactual Magnus solo IVB/HB | línea base | Stuff+ v1 sigue oficial (renombrado) |
| Stuff+ v2 | espejo de zurdos, VAA/HAA ajustados, sin redundancias, arsenal | no mejora; revela la ubicación escondida | descartado |
| Pitching+ v3 | cadena de probabilidades en carreras | equivalente a v1, mejor metodología | **base del oficial** |
| Stuff+ v3 / DIPS | cadena sin ubicación | peor; revela el efecto DIPS | descartado |
| Contrafactual v2 | trayectoria completa, densidad medida | validado split-half | **en uso** |
| Pitching+ v3 neutral | proyección a nivel del mar + shrinkage | **+0.059, P = 99% (≥300 pitches)** | **oficial** |
| Stuff+ v4 | P(whiff \| swing) neutral | peor que v1 | solo comparación dentro del mismo tipo |
| Ajuste de hiperparámetros | Optuna sobre los 5 sub-modelos | log-loss mejora en los 5; índice +0.008-0.018 (no significativo) | **activo** |

Detalle de cada paso: `reports/resultados_finales.md` §7.1-7.11.
