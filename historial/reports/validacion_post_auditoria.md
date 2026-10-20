# Validación post-auditoría (P0 y P1)

**Fecha:** 2026-10-05.
- **Auditoría de origen:** `reports/auditoria_matematica.md` y `historial/AUDITORIA_INICIAL.md`.
- **Pipeline:** regenerado desde los datos crudos con `python -m src.pipeline` (21 pasos, todos OK; `reports/pipeline_run.md`).

**Reglas de este documento:**
- Toda cifra sale de un reporte generado por código. Se cita la fuente de cada una.
- Si una conclusión anterior dejó de ser válida, aquí se dice explícitamente.

---

## 1. Cambios aplicados

| problema | cambio | impacto | estado |
|---|---|---|---|
| **P0.1** El golpeado (HBP) valía ≈ 0 carreras: `is_hit_by_pitch` = 0 en todo el dataset | `targets.py` reconstruye `is_hit_by_pitch` desde `PitchCall == "HitByPitch"`. El golpeado ahora cierra el turno y pone al bateador en primera. Test que falla si el target queda vacío o si Δ(hbp) tiene el signo equivocado | v_HBP: +0.004 → +0.397. Δ(hbp) < 0 en 7 de 12 conteos → 0. Pitching+ cambia hasta 5.9 puntos | ✅ |
| **P0.2** Resultados intermedios reutilizados de corridas viejas | `src/pipeline.py`: ejecución reproducible de punta a punta, con `--refresh` en todo módulo con caché | 21 pasos regenerados desde los datos crudos | ✅ |
| **P0.3** No había baseline contra el cual medir el aporte de la física | `src/validation/baselines.py`: B0 (sin neutralizar), B1 (ajuste trivial temporada × categoría), B1b (ajuste lineal por % de categoría) y A (neutral físico), con el mismo shrinkage | **La física no supera al ajuste trivial** (§3) | ✅ |
| **P0.4** La validación "ajustada por entorno" se definió después de ver resultados | Se reporta primero la validación preespecificada de §7.2, sin cambiar el criterio. La ajustada queda como secundaria post-hoc | **El criterio preespecificado no se cumple** (§3) | ✅ |
| **P0.5** Leakage temporal | `src/validation/temporal.py`: todo se entrena con temporadas < t, incluida la matriz RE24 | La conclusión no cambia; no hay señal de inflación por leakage (§4) | ✅ |
| Falla silenciosa: con ratios ρ̂ inválidos, la proyección devolvía lo observado sin aviso | `trajectory.lookup_ratios` lanza un error explícito; test nuevo | Sin cambio en resultados (con los datos completos los ratios son válidos) | ✅ |
| **P1.1** "Techo" √r_stab presentado como teorema | Derivación con sus supuestos (§11.4); se elimina la palabra "techo" | Interpretación | ✅ |
| **P1.2** "IC 80%" | Renombrado a *intervalo de credibilidad empírico-bayesiano de 80%*, declarando qué no incluye (§6) | Interpretación | ✅ |
| **P1.3** "Calidad del pitch" = take + whiff + foul omitía el cambio en la tasa de contacto | Atribución de Shapley por eslabón de la cadena (`src/models/chain_attribution.py`), en el contrafactual, la tabla oficial, las fichas y el dashboard | "Antes del contacto" = +0.388 carreras/100 (35% del efecto) (§7) | ✅ |
| **P1.4** Stuff+ v1 descrito como "incluye ubicación" | Columna renombrada a `stuff_plus_v1`, con su descripción completa: ve ubicación implícita **y** la categoría de altitud (§8) | Interpretación | ✅ |
| **P1.5** La media de Pitching+ no es 100 | Análisis de la escala (`reports/escala_pitching_plus.md`). Se conserva la escala y se corrige la definición: "100 = pitch promedio de la liga" | Interpretación | ✅ |
| **P1.6 / P1.8** Definiciones de Stuff+, Location+ y Pitching+ | Contrastadas con el código (§8). Location+ no existe en el pipeline vigente | Documentación | ✅ |
| **P1.7** Hipótesis del analista (whiff/K/contacto) | `src/analysis/altitude_hypotheses.py`, con hipótesis y criterio fijados antes de correr (§9) | H1, H2 y H3 respaldadas, condicionales a los estadios de la muestra | ✅ (preliminar) |
| **P1.9** Calidad neutral vs. sensibilidad a la altitud | `src/analysis/sensitivity.py`: sensibilidad implicada por el modelo vs. sensibilidad empírica (§9.4) | La sensibilidad empírica casi no es medible (confiabilidad 0.11) | ✅ |
| **P1.10** Identificación del estadio de altitud extrema | `src/data/parks.py`: reconstrucción de equipos y estadios anónimos (§10) | Extreme = 4–5 estadios locales distintos; el estadio de altitud extrema no identificable | ✅ |
| Codificación de los reportes (acentos rotos) | `encoding="utf-8"` en todas las escrituras | — | ✅ |

**Pendientes de P2/P3 (no se tocaron, como se pidió):**
- revisión del shrinkage (μ por volumen, σ² por rol, guardia τ² ≤ 0);
- efecto de altitud con predicciones OOF;
- sobrepredicción de ~20% del cambio de IVB;
- limpieza de código duplicado (taxonomía de eventos en `temporal.py`);
- reescritura completa del documento técnico (por ahora lleva una nota de corrección).

---

## 2. HBP: antes vs. después

Fuente: `historial/reports/hbp_antes_despues.md` (`src/validation/hbp_before_after.py`).

**Cómo se identificó.** La columna `is_hit_by_pitch` del dataset vale 0 en las 633,447 filas. El golpeado real está en `PitchCall == "HitByPitch"`: 2,276 pitches. `play_result` coincide en 2,254; los otros 22 vienen marcados además como base por bolas. Verificación manual: después de un golpeado, el siguiente pitch de la media entrada es de otro bateador en el 99.96% de los casos y empieza en 0-0 en el 99.3%.

| | antes | después |
|---|---:|---:|
| pitches con `is_hit_by_pitch` = 1 | 0 | 2,276 |
| run value medio del golpeado | +0.0038 | **+0.3976** |
| run value medio de la base por bolas | +0.3815 | +0.3784 |
| conteos con Δ(hbp) < 0 | 7 de 12 | **0 de 12** |
| Δ(hbp) en 3-0 | −0.246 | +0.150 |
| pitches cuyo run value cambió | — | 159,097 |

El run value cambia también fuera de los golpeados: el golpeado ahora cierra el turno y pone un corredor en primera, así que cambian los estados base-out del resto de la media entrada y la propia matriz RE24.

**Cambio en V(b,s):** entre −0.0029 (3-2) y +0.0005 (0-0). V(0,0) = +0.0147.

**Cambio en los Δ de los demás eventos:** máximo 0.0074 carreras.

**Pitching+ oficial (pitcher-temporadas con ≥100 pitches):**
- correlación antes/después: 0.988 (Spearman 0.987);
- cambio con sd de 1.35 puntos y máximo de 5.9;
- 137 pitcher-temporadas se mueven ≥2 puntos;
- el cambio correlaciona −0.56 con la tasa de golpeados del pitcher: los que más golpean bajan, porque antes el golpeado se les contaba casi como un strike.

**Verificación de que no hay que re-ajustar hiperparámetros.** Las etiquetas de los 5 sub-modelos y sus probabilidades OOF son **idénticas** antes y después (diferencia máxima = 0.0). HBP solo cambia los valores en carreras, no lo que predicen los modelos. Los hiperparámetros se buscaron minimizando el log-loss de esas mismas etiquetas, así que siguen siendo válidos.

---

## 3. Baselines y validación preespecificada

Fuente: `reports/baselines.md`. Todos los scores usan las mismas predicciones OOF, el mismo shrinkage y la misma escala; solo cambia cómo se trata el entorno.

| score | cómo trata el entorno |
|---|---|
| B0 · con altitud | no lo trata |
| B1 · ajuste trivial | a cada pitch le resta el E[Δ] promedio de su temporada × categoría de altitud; **sin física** |
| B1b · ajuste lineal | quita al score B0 la parte explicada por el % de pitches del pitcher en cada categoría |
| A · neutral físico (oficial) | proyecta cada pitch a No Altitude con el contrafactual de trayectoria |

### 3.1 Validación preespecificada (bitácora §7.2, fijada el 2026-10-03 antes de ver cualquier mejora)

| | |
|---|---|
| Objetivo | predecir el desempeño del pitcher en la temporada siguiente |
| Métrica | correlación de Pearson entre el score en t y las **carreras evitadas observadas (RE24) por 100 pitches en t+1** |
| Horizonte | una temporada (2024→2025 y 2025→2026) |
| Mínimo de pitches | 100 en ambas temporadas (437 pares) |
| Criterio de éxito | Δr > 0 contra **Pitching+ v1** con P(Δr > 0) ≥ 0.90 (bootstrap pareado por pitcher, 2,000 réplicas) |

| score | r con carreras observadas t+1 | Δr vs. Pitching+ v1 | P | cumple |
|---|---:|---:|---:|---|
| Pitching+ v1 (referencia) | 0.188 | — | — | — |
| B0 · con altitud | 0.177 | −0.011 | 36% | ❌ |
| B1 · ajuste trivial | 0.169 | −0.019 | 32% | ❌ |
| B1b · ajuste lineal | 0.184 | −0.004 | 47% | ❌ |
| **A · neutral físico (oficial)** | **0.153** | **−0.035** | **20%** | ❌ |

**Resultado: ningún score cumple el criterio preespecificado.** El índice neutral oficial es el que peor predice las carreras observadas del año siguiente (A vs. B0: −0.024, P = 15%; A vs. B1: −0.016, P = 7%).

**¿Es apropiado el criterio original?** Hay un argumento para pensar que no lo es para un índice *neutral*: las carreras observadas incluyen el efecto del parque, y la mayoría de los pitchers repite parque. No se cambió el criterio. Ese argumento justifica una validación **secundaria y separada**, que se reporta como post-hoc.

### 3.2 Validación secundaria (post-hoc): carreras t+1 ajustadas por entorno

| mín. pitches | B0 | B1 trivial | A neutral físico | A − B0 (P) | **A − B1 (P)** |
|---|---:|---:|---:|---|---|
| 100 (437 pares) | 0.165 | **0.191** | 0.181 | +0.016 (78%) ❌ | **−0.010 (20%)** ❌ |
| 300 (245 pares) | 0.215 | **0.279** | **0.279** | +0.065 (99%) ✅ | **+0.000 (53%)** ❌ |

**Respuesta a la pregunta de P0.3: el modelo físico no aporta información predictiva sobre el ajuste ambiental simple.**
- Con mín. 300, A y B1 son iguales (Δ = 0.000).
- Con mín. 100, B1 es mejor que A.
- La ventaja de A sobre B0 (+0.065) se explica por la normalización ambiental, no por la física del contrafactual.
- Además, el resultado secundario se ajustó con la misma partición (temporada × categoría) que usa B1, así que esta prueba favorece por construcción a cualquier ajuste por entorno.

**Neutralización** (correlación con el % de pitches en Extreme, n ≥ 100): B0 −0.325; **B1 −0.042**; A −0.062. El ajuste trivial neutraliza un poco mejor.

**Lo que esto NO significa.** El contrafactual sigue teniendo valor **físico e interpretativo**:
- explica *por qué* cambia el valor de un pitch (menos movimiento, más velocidad en zona, menos whiffs);
- permite valuar un pitch hipotético en otra altitud (dashboard);
- da la atribución por eslabón (§7).

Lo que no tiene es valor **predictivo** adicional demostrado.

---

## 4. Validación temporal estricta

Fuente: `reports/validacion_temporal.md` (`src/validation/temporal.py`).

**Qué se entrena con qué:**

| componente | se estima con |
|---|---|
| matriz RE24 → run value | temporadas < t |
| valores terminales, V(b,s), Δ | temporadas < t |
| coeficientes del ajuste de VAA/HAA por ubicación | temporadas < t |
| ratios de densidad ρ̂ | temporadas < t (con < 2025: drag 0.864, Magnus 0.780) |
| los 5 sub-modelos LightGBM | temporadas < t |
| recta base del arsenal | temporada t, solo features (no usa resultados) |
| shrinkage (μ, τ, σ) y ajuste trivial | predicciones de la temporada t (no usan resultados) |
| **hiperparámetros** | **globales: excepción documentada** (se buscaron con las 3 temporadas, objetivo = log-loss de las etiquetas) |

Con t = 2024 no hay temporadas anteriores para entrenar, así que la **única pareja evaluable es 2025 → 2026** (215 pares con mín. 100; 99 con mín. 300).

| resultado t+1 | mín. | B0 orig. | A orig. | B0 temporal | B1 temporal | A temporal | A − B1 temporal (P) |
|---|---|---:|---:|---:|---:|---:|---|
| observado (preespecificado) | 100 | 0.161 | 0.159 | 0.158 | 0.172 | 0.148 | −0.024 (4%) |
| observado | 300 | 0.234 | 0.233 | 0.273 | 0.275 | 0.258 | −0.017 (18%) |
| ajustado (post-hoc) | 100 | 0.132 | 0.160 | 0.125 | 0.162 | 0.136 | −0.026 (5%) |
| ajustado (post-hoc) | 300 | 0.181 | 0.261 | 0.225 | 0.305 | 0.281 | −0.023 (18%) |

**Lectura:**
- **A vs. B: no hay señal de que el leakage infle los resultados.** Los valores temporales no son sistemáticamente menores que los originales (con mín. 300 son incluso mayores; con 99 pares eso es ruido de muestra).
- **Con entrenamiento temporal estricto, el neutral físico tampoco supera al ajuste trivial.** Queda por debajo en las 4 combinaciones.
- La validación temporal tiene **poca potencia** (una sola pareja de temporadas, ≤215 pares): sus IC son anchos.

---

## 5. Pitching+: resultados finales

Fuentes: `reports/pitching_plus_neutral.md`, `reports/escala_pitching_plus.md`.

| temporada | μ (carreras/pitch) | τ (habilidad real) | σ (pitch a pitch) | n½ = σ²/τ² | pitchers usados |
|---|---:|---:|---:|---:|---:|
| 2024 | −0.00411 | 0.00654 | 0.0537 | 67 | 436 |
| 2025 | −0.00553 | 0.00586 | 0.0544 | 86 | 474 |
| 2026 | −0.00556 | 0.00651 | 0.0540 | 69 | 368 |

σ subió de ≈0.051 a ≈0.054 con la corrección de HBP: un golpeado vale ahora ≈ +0.38 en vez de ≈ 0, así que E[Δ] varía más de un pitch a otro.

**Escala (P1.5).** Media de Pitching+ por pitcher:

| población | 2024 | 2025 | 2026 |
|---|---:|---:|---:|
| todos | 98.0 | 97.9 | 98.8 |
| n ≥ 100 | 98.9 | 98.7 | 99.3 |
| n < 100 | 94.5 | 93.8 | 97.5 |

sd con n ≥ 100: 8.6–8.9.

**¿Por qué no es 100?**
- 100 = μ = media de E[Δ] **ponderada por pitches** entre pitchers con ≥100. Con esa ponderación y sin shrinkage el promedio es exactamente 100.0 en las tres temporadas.
- El promedio **por pitcher** queda debajo porque los pitchers de más volumen son mejores (correlación de rangos volumen–score de 0.22–0.26) y los de menos de 100 pitches son peores.
- No es causa la neutralización: el índice con altitud tiene la misma construcción.
- La sd < 10 es **por diseño**: 10 puntos = 1 τ, y θ̂ está encogido.

**Decisión: se conserva la escala (opción A)** y se corrige su definición: *"100 = pitch promedio de la liga"*, no *"100 = pitcher promedio"*. Recalibrar haría que el 100 dependiera de cuántos pitchers de poco volumen hay en la temporada; sería cosmético.

**Neutralización:** corr(score, %Extreme) = −0.324 (con altitud) → −0.062 (oficial). Por temporada, el residuo no es homogéneo: 2026 queda en −0.15.

---

## 6. Incertidumbre: definición correcta

**Cálculo.** En el modelo normal-normal x̄_p | θ_p ~ N(θ_p, σ²/n_p) y θ_p ~ N(μ, τ²), la posterior de θ_p es normal:
- media θ̂_p = μ + (1 − B_p)(x̄_p − μ);
- varianza (1 − B_p)·σ²/n_p.

En la escala del índice, el intervalo es Pitching+ ± 1.2816·10·sd_post/τ.

**Tres conceptos distintos:**
- **Intervalo de confianza (frecuentista).** Es un procedimiento que, repetido sobre muchas muestras, cubre el parámetro fijo el 80% de las veces.
- **Intervalo de credibilidad (bayesiano).** Dado el modelo y los datos, el parámetro está ahí con 80% de probabilidad posterior.
- **El nuestro: intervalo de credibilidad empírico-bayesiano *plug-in*.** Es bayesiano en su forma, pero μ, τ y σ se estiman con los mismos datos y se tratan como conocidos.

**Qué incluye:** la incertidumbre por el número de pitches del pitcher (muestras chicas → intervalo ancho), con un ancho medio de 23 puntos con n ≤ 30 y de 6 puntos con n > 1,000.

**Qué NO incluye:**
- la incertidumbre de μ, τ y σ;
- la de los modelos LightGBM;
- la de los ratios de densidad ρ̂ y del contrafactual;
- la de los hiperparámetros;
- la dependencia entre pitches del mismo pitcher (σ² por mitades es ≈6% mayor que la usada).

**Por lo tanto, es más angosto que la incertidumbre total.**

**Cómo leerlo:** Pitcher A = 108 [104, 112] y Pitcher B = 109 [101, 117] **no** permite decir que B es mejor.

**Cambios de nombre aplicados:** reportes del índice, fichas de scouting, dashboard y documentación ("intervalo de credibilidad").

---

## 7. Efecto de la altitud extrema: descomposición correcta

Fuente: `reports/altitude_counterfactual_v2.md`.

**Efecto total** (mismo pitch proyectado a Extreme y a No Altitude, modelos finales): **+1.095 carreras por 100 pitches** [IC 95% por bootstrap de pitchers: +1.077, +1.112]. Ese IC es demasiado estrecho: el modelo es único y las predicciones son in-sample (pendiente P2).

**Atribución de Shapley por eslabón** (carreras/100, todos los pitches; los componentes suman exactamente el total, con error ≤ 4e−17):

| eslabón | carreras/100 | % |
|---|---:|---:|
| decisión de swing | +0.159 | 15% |
| resultado del take (bola / strike cantado / golpeado) | +0.021 | 2% |
| whiff dado swing | +0.142 | 13% |
| foul dado contacto | +0.066 | 6% |
| **antes del contacto (suma)** | **+0.388** | **35%** |
| resultado de la bola en juego | +0.707 | 65% |

**Mecanismo** (promedio por pitch, Extreme − No Altitude):
- P(whiff | swing) −1.82 pp;
- P(pelota en juego) +1.29 pp;
- P(swing) −0.16 pp.

Para la recta de 4 costuras:
- IVB −4.0 pulg., ZoneSpeed +1.4 mph, VAA −0.29°;
- P(whiff | swing) −2.3 pp;
- antes del contacto +0.52 carreras/100, bola en juego +0.93.

**Qué se puede separar y qué no:**

| componente | ¿separable? |
|---|---|
| 1. cambio físico del pitch | sí: lo da el contrafactual (IVB, HB, velocidad, VAA) |
| 2. cambio en la probabilidad de swing | sí (eslabón *swing*) |
| 3. cambio en la probabilidad de whiff | sí (eslabón *whiff*) |
| 4. cambio en el contacto | sí: P(en juego) y eslabones *whiff* y *foul* |
| 5. cambio en la calidad del contacto | **no por separado**: el eslabón *bola en juego* mezcla la calidad del contacto que provoca el pitch con el vuelo de la pelota y las dimensiones del parque |
| 6. cambio final en el valor esperado | sí (total) |

**Advertencias:**
- La atribución de Shapley es **contable** (reparte la diferencia entre dos predicciones del modelo), **no causal**.
- La versión anterior ("calidad del pitch" = take + whiff + foul = +0.319, antes de corregir HBP) omitía que menos whiffs implican más pelotas en juego. Esa cifra queda **reemplazada** por "antes del contacto" = +0.388.

---

## 8. Stuff+, Location+ y Pitching+: definiciones contra el código

| índice | definición deseada para la página | lo que hace el código hoy | ¿coincide? |
|---|---|---|---|
| **Stuff+ v1** (`stuff_plus_v1`) | calidad de las características físicas del lanzamiento | P(whiff) por pitch con LightGBM, entrenado con **todos** los pitches. Features: velocidad, spin, movimiento, extensión, punto de salida, **ángulos de salida y llegada (ubicación implícita)**, **`altitude_level` (categoría de altitud)**, residuales de Magnus y mano. OOF por pitcher; promedio por pitcher-temporada; 100 + 10·z por temporada, sin shrinkage | **No.** Es un modelo predictivo de whiff que ve ubicación y altitud. Su AUC OOF es 0.73: distingue moderadamente pitches con y sin whiff, pero buena parte de esa capacidad viene de dónde va el pitch (sin ubicación implícita el AUC baja a 0.64) |
| Stuff+ "físicamente neutralizado" | — | **No existe.** Lo más cercano es Stuff+ v4 (P(whiff \| swing), sin ubicación, proyectado a nivel del mar), pero no se adoptó: predice peor y está sesgado por tipo de pitch. Solo se usa *dentro de cada tipo* en las fichas | — |
| **Location+** | calidad de la ubicación | **No existe en el pipeline vigente.** Hay un Location+ v1 de legado (`src/models/location_plus.py`, un GAM de strike cantado y chase), que no está integrado ni validado contra la cadena | **No.** Hay que construirlo (p. ej., la misma cadena solo con ubicación + conteo) |
| **Pitching+ (oficial)** | valor esperado del lanzamiento considerando las características relevantes y el contexto | E[Δ carreras] de la cadena de 5 modelos con física + ubicación + conteo, proyectado a No Altitude, con shrinkage | **Sí**, con una aclaración: es *neutral por altitud* (valor esperado a nivel del mar) |

**Pitching+ ≠ Stuff+ + Location+.** No hay una suma: un solo modelo usa todas las features a la vez.

**Separación conceptual (P1.8)** para la página:
- Stuff+ → física;
- análisis whiff/K → respuesta del bateador (§9);
- análisis de contacto → consecuencia del contacto (§9);
- Pitching+ → valor integral.

Ningún índice usa K%, BB%, H%, HR% ni hard contact como feature. Son resultados, no entradas.

**Nombre de Stuff+ v1.** Se cambió de `stuff_plus_v1_incluye_ubicacion` a `stuff_plus_v1`. **Justificación:** el nombre anterior decía "incluye ubicación" pero omitía que el modelo también usa la categoría de altitud. Ahora la etiqueta completa en reportes, fichas y dashboard es: *"Stuff+ v1 (modelo predictivo de whiff; usa ubicación implícita y la categoría de altitud; no está neutralizado)"*.

---

## 9. Whiff, K y contacto: hipótesis del analista (preliminar)

Fuente: `reports/hipotesis_altitud.md`. Hipótesis, variables y criterio se fijaron en el docstring del módulo **antes** de correrlo.

**Diseño:**
- Modelo lineal con efectos fijos de pitcher, bateador, temporada y conteo.
- β_E = diferencia promedio Extreme − No Altitude **para el mismo pitcher contra el mismo bateador**.
- Errores estándar agrupados por juego.
- *Respaldada* = el IC 95% excluye 0 en la dirección de la hipótesis.

### 9.1 H1: ¿la altitud cambia la probabilidad de contacto?

| variable | media en No Altitude | β_E [IC 95%] | estadios en la dirección | veredicto |
|---|---:|---|---:|---|
| whiff por swing | 0.249 | **−0.029** [−0.034, −0.024] | 13/13 | RESPALDADA |
| bola en juego por swing | 0.378 | **+0.040** [+0.034, +0.045] | 13/13 | RESPALDADA |
| ponche por turno | 0.203 | **−0.047** [−0.052, −0.041] | 13/13 | RESPALDADA |

### 9.2 H2: ¿el contacto es más peligroso?

| variable (por bola en juego) | media en No Altitude | β_E [IC 95%] | estadios | veredicto |
|---|---:|---|---:|---|
| run value RE24 | +0.015 | **+0.068** [+0.058, +0.078] | 13/13 | RESPALDADA |
| hit | 0.331 | +0.053 [+0.045, +0.062] | 13/13 | RESPALDADA |
| extra base | 0.100 | +0.023 [+0.017, +0.028] | 13/13 | RESPALDADA |
| HR | 0.037 | +0.004 [+0.001, +0.008] | 9/13 | respaldada, **poco consistente** |
| velocidad de salida (mph) | 83.0 | +0.95 [+0.66, +1.24] | 11/13 | respaldada (ver advertencia) |
| barrel | 0.026 | −0.000 [−0.003, +0.003] | 7/13 | NO CONCLUYENTE |

### 9.3 H3: ¿evitar el contacto vale relativamente más?

G = cuánto más vale para el bateador una bola en juego que un whiff en el mismo conteo, con V(b,s) recalculado en cada categoría:

| categoría | G (carreras) [IC 95%] |
|---|---|
| No Altitude | 0.172 [0.168, 0.176] |
| Medium | 0.212 [0.206, 0.218] |
| Extreme | 0.218 [0.212, 0.224] |

**G_E − G_N = +0.046 [+0.038, +0.053] → RESPALDADA** (bootstrap de juegos). G es mayor en los 13 estadios de Extreme.

**Lectura para béisbol (condicionada).** En los estadios de la categoría Extreme, el mismo pitcher contra el mismo bateador:
- consigue menos swings fallidos (−2.9 puntos);
- permite más pelotas en juego;
- poncha menos (−4.7 puntos por turno).

Cada pelota en juego cuesta más carreras (+0.068). Convertir una bola en juego en un whiff vale ≈ 27% más que en No Altitude (0.218 vs. 0.172). **Los datos respaldan la idea del analista**, con estas advertencias:

1. **No es "la altitud" aislada.** Con 4–5 estadios por categoría, la altitud se confunde con dimensiones del parque, clima y calibración de Trackman.
2. **Medium se parece a Extreme.**
   - En H3: G_M − G_N = +0.040 contra +0.046 de Extreme.
   - En velocidad de salida, Medium tiene un efecto *mayor*: +1.45 mph contra +0.95.
   - Esto no encaja con un efecto que crezca solo con la densidad del aire. Sugiere que parte de las diferencias de velocidad de salida es calibración o parque.
3. **Los IC por juego son condicionales a estos estadios.** La consistencia en los 13 estadio-temporada da robustez, pero no generaliza a otros estadios de altura.
4. **Está pendiente:**
   - un modelo interpretable de P(whiff | características) (qué características aumentan el whiff, por tipo de pitch y altitud);
   - un modelo de K a nivel turno.

### 9.4 Calidad neutral vs. sensibilidad a la altitud (P1.9)

Fuente: `reports/sensibilidad_altitud.md`.

**Sensibilidad implicada por el modelo** (S_mod: el mismo pitch en Extreme vs. No Altitude):
- promedio de la liga: +1.09 carreras/100;
- sd entre pitchers: 0.24;
- **la mezcla de tipos de pitch explica R² = 0.44.**

Todos los pitchers reciben los mismos ratios de densidad, así que S_mod describe **qué lanza** el pitcher, no cómo reacciona él a la altitud.

**Sensibilidad empírica** (S_emp: lo que realmente le pasó en Extreme vs. No Altitude, relativo a la liga; 294 pitchers con ≥150 pitches en cada una):
- error estándar típico de 1.88 contra una sd observada de 1.99;
- correlación entre mitades de juegos de 0.06, **confiabilidad 0.11**: es casi todo ruido.

**¿El modelo predice la sensibilidad observada?** corr(S_mod, S_emp) = **+0.06**. No.

**Conclusión:** con estos datos **no se puede rankear pitchers por "resistencia a la altitud" observada**. El cuadrante "calidad alta + sensibilidad baja" solo puede mostrarse como *"buen pitcher con un repertorio que el modelo espera que pierda poco en altura"*. Además, calidad neutral y S_mod correlacionan +0.33: los mejores pitchers tienden a tener repertorios que el modelo espera que pierdan más (p. ej., rectas con mucho IVB).

---

## 10. el estadio de altitud extrema: qué se puede y qué no se puede afirmar

> **Decisión del proyecto (2026-10-05):** no se usará información externa (calendarios, nombres) ni se intentará identificar equipos, estadios o pitchers reales. Se respeta por completo la anonimización; los grupos se nombran `ANON_STADIUM_xx` (`reports/estadios_anonimos.md`). Las opciones de identificación de abajo quedan **descartadas**.

Fuente: `reports/estadios_anonimos.md` (`src/data/parks.py`).

**El dataset no trae:** estadio, equipo, local/visitante explícito, elevación, dimensiones ni clima.

**Reconstrucción.** Los compañeros de equipo lanzan y catchean del mismo lado del juego, y en la alta lanza el local. Con eso, un clustering espectral reconstruye equipos anónimos. k se elige por temporada (18/20/22) con un criterio que **no usa la altitud** (mínimo de juegos con local = visitante).

| temporada | k | juegos con local = visitante | pureza de categoría por equipo local | equipos con ≥80% de juegos locales en Extreme |
|---|---:|---:|---:|---:|
| 2024 | 20 | 0.0% | 100.0% | **5** |
| 2025 | 22 | 0.2% | 95.3% | **4** |
| 2026 | 22 | 0.9% | 91.3% | **4** |

En los estadios locales de Extreme, el ratio de drag dentro de pitcher va de **0.81 a 0.90**, con IC que no se traslapan entre los extremos, y el HR por bola en juego va de 2.8% a 5.1%.

**Se puede afirmar:**
- "Extreme Altitude" agrupa a 4–5 equipos locales con estadios **físicamente distintos**.
- Todo resultado actual es un **efecto de altitud extrema** promedio de esos estadios.

**No se puede afirmar:**
- cuál de ellos es el estadio de altitud extrema;
- cuáles pitchers son del equipo local;
- ningún resultado como "efecto el estadio de altitud extrema".

El de menor ratio de drag es un candidato natural al estadio más alto, pero eso es una **hipótesis**: altitudes cercanas, clima y calibración también mueven el ratio.

**Para identificarlo hace falta información externa**, por ejemplo el calendario público de la LMB contra las fechas de los juegos reconstruidas del PitchUID. Usarla es una **decisión del equipo** por las reglas del hackathon.

---

## 11. Conclusiones

### 11.1 Resultados demostrados

1. **La implementación matemática es fiel a la metodología.** Lo verifican los tests: Σ P = 1, recursión de Bellman con auto-lazo, signos de Δ, cinemática, D + M = a_ng, reproyección analítica, shrinkage, escala y eficiencia de Shapley.
2. **El golpeado estaba mal valuado y se corrigió desde la fuente.** Δ(hbp) > 0 en los 12 conteos.
3. **El contrafactual físico es coherente:**
   - reproduce la ubicación (error mediano < 0.01 ft);
   - llega al mismo objetivo;
   - las features cambian en la dirección física esperada;
   - ρ̂ se mide dentro de pitcher.
4. **Ajustar por entorno mejora la predicción de carreras futuras ajustadas por entorno** (mín. 300: +0.06 sobre el índice sin ajustar, P ≈ 99%), **pero eso lo logra igual un ajuste trivial** temporada × categoría.
5. **El contrafactual físico no aporta valor predictivo adicional sobre ese ajuste trivial**, ni con la validación original ni con la temporal estricta.
6. **El criterio preespecificado (§7.2) no se cumple:** ningún score supera a Pitching+ v1 prediciendo carreras observadas t+1 con mín. 100.
7. **No hay evidencia de que el leakage temporal infle los resultados.**
8. **"Extreme Altitude" no es un solo estadio:** son 4–5 estadios con densidades efectivas distintas.
9. **En esos estadios, con el mismo pitcher contra el mismo bateador:** menos whiffs (−2.9 pp por swing), más contacto en juego (+4.0 pp), menos ponches (−4.7 pp por turno) y contacto más valioso (+0.068 carreras por bola en juego). Es consistente en 13 de 13 estadio-temporada.
10. **Evitar el contacto vale más ahí:** G_E − G_N = +0.046 carreras por swing (≈ +27%).
11. **La sensibilidad individual a la altitud observada no es medible** con estos datos (confiabilidad 0.11).

### 11.2 Resultados sugerentes

- Medium se comporta casi como Extreme en contacto y en G, y tiene un efecto mayor en velocidad de salida. Sugiere que las diferencias no son solo de densidad del aire.
- El HR por bola en juego aumenta en promedio, pero no en todos los estadios (9/13).
- Los pitchers con mejor calidad neutral tienen repertorios que el modelo espera que pierdan más en altura (r = +0.33).
- La neutralización deja un residuo creciente en 2025–2026 (corr −0.10 / −0.15).

### 11.3 Hipótesis (no demostradas)

- Que alguno de los estadios reconstruidos sea el estadio de altitud extrema y que los efectos de H1–H3 apliquen específicamente ahí.
- Que el efecto sea causado por la densidad del aire y no por otras características del parque.
- Que haya pitchers "resistentes a la altitud" como habilidad individual.
- Que el contrafactual físico mejore las proyecciones en algún otro diseño (p. ej., pitchers que cambian de equipo y de parque).

### 11.4 Limitaciones

- **"Techo" √r_stab: no es un techo universal.** Bajo el modelo de medición clásico (o = T + e, con errores independientes entre años y de la métrica, e igual confiabilidad en t y t+1):
  - corr(m_t, o_{t+1}) ≤ √rel(o), con rel = Var(T)/Var(o);
  - además r_stab(o) = ρ_T·rel(o), donde ρ_T es la correlación del talento verdadero entre años;
  - así que la cota es √(r_stab/ρ_T), que solo es √r_stab si el talento no cambia entre años (ρ_T = 1); si cambia, la cota es más alta.
  - *En palabras sencillas:* si las carreras de un pitcher se repiten poco de un año a otro, ninguna métrica puede predecirlas muy bien, pero el número exacto depende de cuánto cambia el talento real de un año a otro.
  - Valores observados: r_stab de las carreras observadas = 0.08 (mín. 100) y 0.16 (mín. 300); de las ajustadas, 0.17 y 0.27. Se usan como **orientación**, no como cota.
- **Poca potencia:** 437 pares (mín. 100), 245 (mín. 300) y 99–215 en la validación temporal.
- **El efecto de altitud se calcula con modelos in-sample:** su IC es demasiado estrecho.
- **El contrafactual sobrepredice el cambio de IVB en ~20%** (split-half: −2.64 predicho vs. −2.20 observado).
- **Los ratios ρ̂ son efectivos**, no densidades físicas: mezclan densidad, clima, calibración y pelota.
- **Las hipótesis H1–H3 son asociaciones** con controles por pitcher y bateador, no efectos causales de la densidad del aire.

---

## 12. Tests

Fuente: `python -m pytest -v tests` (dentro de `src/pipeline.py`) y verificaciones adicionales.

| test | qué verifica | estado |
|---|---|---|
| test_event_probabilities_sum_to_one_including_edges | Σ P = 1 con π = 0 y 1 | PASS |
| test_oof_probabilities_sum_to_one | Σ P = 1 en los 627,185 pitches OOF, sin NaN | PASS |
| test_count_recursion_bellman_with_two_strike_foul_self_loop | Bellman en los 12 conteos; foul con 2 strikes = mismo estado, Δ = 0 | PASS |
| test_kinematics_reproduce_plate_location | r(T) reproduce PlateLoc (error mediano < 0.01 ft) | PASS |
| test_drag_magnus_decomposition_is_exact_and_orthogonal | D + M = a_ng, M ⟂ v, el drag frena | PASS |
| test_ratios_with_too_few_groups_fail_loudly | sin datos para ρ̂, la proyección falla en vez de devolver lo observado | PASS (test nuevo; falla real encontrada y corregida) |
| test_counterfactual_hits_same_target_without_nans_and_is_physical | mismo objetivo, sin NaN nuevos, cambios acotados y con el signo físico | PASS |
| test_shrinkage_factor_bounds_and_limits | B ∈ (0, 1), decreciente en n, n chico → μ, n grande → x̄ | PASS |
| test_score_sign_and_scale_definition | Pitching+ = 100 + 10(μ − θ̂)/τ; menos carreras → score más alto | PASS |
| test_official_scale_matches_definition | media ponderada sin shrinkage = 100; sd < 10 por diseño | PASS |
| test_shapley_is_efficient_and_null_link_gets_zero | Σφ = δ; un eslabón que no cambia recibe 0 | PASS |
| test_fix_is_hit_by_pitch_uses_pitchcall | reconstrucción de HBP | PASS |
| test_terminal_event_targets_not_empty | ningún indicador terminal vacío o constante | PASS (FAIL con los datos de antes de la corrección) |
| test_hbp_run_value_is_like_a_walk | run value del golpeado ≈ base por bolas | PASS (FAIL antes: +0.004) |
| test_event_value_signs_all_counts | signos de Δ en los 12 conteos, incluido Δ(hbp) ≥ Δ(bola) | PASS (FAIL antes) |
| test_trajectory.py (6 pruebas) | identidad, dirección física, ratios < 1, pitch sintético, shrinkage, simetría zurdo/derecho | PASS |
| Pipeline completo (`reports/pipeline_run.md`) | 21 pasos desde los datos crudos, desde el commit limpio `018366a` | PASS |
| Reproducibilidad | la re-ejecución completa desde `018366a` reproduce byte a byte todos los reportes, JSON y fichas (`git status` solo muestra `pipeline_run.md`) | PASS |
| Dashboard (`streamlit.testing.AppTest`) | ambas pestañas y el cálculo del pitch hipotético sin excepciones | PASS (falló una vez por una llave renombrada; corregido) |
| Identidad de etiquetas y probabilidades antes/después de HBP | diferencia máxima 0.0 | PASS |
| Sobrepredicción del cambio de IVB (split-half) | predicho −2.64 vs. observado −2.20 pulg. | **WARNING** |
| Residuo de neutralización por temporada | 2026: corr −0.15 | **WARNING** |
| Confiabilidad de la sensibilidad empírica | 0.11 | **WARNING** (no permite rankear) |
| Criterio preespecificado de §7.2 | ningún score lo cumple | **FAIL** (resultado científico, no error de código) |
| Física > ajuste trivial | Δ = 0.000 (P = 53%) con mín. 300 | **FAIL** (resultado científico) |
