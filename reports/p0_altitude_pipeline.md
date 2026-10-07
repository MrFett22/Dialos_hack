# P0 — Pipeline validado para el estudio de altitud extrema

**Proyecto:** *Pitching+ para condiciones de altitud extrema.*
**Fecha:** 2026-10-05.

**Pregunta central:** ¿cómo cambia el valor efectivo de un pitcheo bajo condiciones de altitud extrema y qué características permiten reducir el daño asociado con esos ambientes?

**Regla de anonimización.** El dataset se usa tal como fue entregado, anonimizado. No se usa información externa (calendarios, nombres, número de equipos de la liga). No se intenta identificar equipos, estadios ni pitchers reales. Los resultados se expresan por categoría de altitud (`No Altitude`, `Medium Altitude`, `Extreme Altitude`) y por grupo de estadio anónimo (`ANON_STADIUM_xx`).

**Ventaja metodológica:** todo el análisis se reproduce con los datos entregados, sin depender de identificación externa, calendarios ni supuestos sobre identidades.

**Reproducibilidad.** `python -m src.pipeline` regenera todo desde los datos crudos (21 pasos, todos OK; `reports/pipeline_run.md`). Ninguna cifra de este documento se escribió a mano: cada una sale de un reporte generado (se cita la fuente).

---

## 1. Corrección del golpeado (HBP)

Fuente: `reports/hbp_antes_despues.md`.

**Problema.** La columna `is_hit_by_pitch` del dataset vale 0 en las 633,447 filas. Por eso el golpeado:
- no cerraba el turno;
- no ponía al corredor en primera;
- valía ≈ 0 carreras.

**Variable correcta.** `PitchCall == "HitByPitch"` (2,276 pitches). `play_result` coincide en 2,254; los otros 22 vienen marcados además como base por bolas.

**Verificación manual.** Después de un golpeado, el siguiente pitch de la media entrada es de otro bateador en el 99.96% de los casos y empieza en 0-0 en el 99.3%.

**Corrección en la fuente.** `src/features/targets.py` (`fix_is_hit_by_pitch`) reconstruye la columna. Todo el pipeline se regeneró a partir de ahí: RE24 → valores por conteo → probabilidades → E[Δ] → contrafactual → Pitching+ → validaciones.

**Test automático.** `tests/test_targets.py`:
- falla si un indicador de evento terminal queda vacío o constante;
- falla si el golpeado no vale lo que una base gratis;
- falla si algún Δ(evento, conteo) tiene el signo equivocado.

Con los datos de antes de la corrección, los 3 tests de datos fallaban.

| | antes | después |
|---|---:|---:|
| pitches con `is_hit_by_pitch` = 1 | 0 | 2,276 |
| run value medio del golpeado | +0.004 | **+0.398** |
| run value medio de la base por bolas | +0.382 | +0.378 |
| conteos con Δ(hbp) < 0 (el golpeado "favorecía" al pitcher) | 7 de 12 | **0 de 12** |
| Δ(hbp) en 3-0 | −0.246 | +0.150 |

## 2. Cambios en resultados por la corrección

- **V(b,s)** cambia entre −0.0029 (3-2) y +0.0005 (0-0); V(0,0) = +0.0147.
- **Δ de los demás eventos:** cambio máximo de 0.0074 carreras.
- **Pitching+:** correlación antes/después de 0.988. Cambio máximo de 5.9 puntos, y 137 pitcher-temporadas (de 1,278 con ≥100 pitches) se mueven ≥ 2 puntos. Bajan los que más golpean: la correlación del cambio con su tasa de golpeados es −0.56.
- **Variación pitch a pitch del valor esperado:** σ ≈ 0.051 → 0.054, porque el golpeado ahora vale ≈ +0.38.
- **Sin cambio:** las etiquetas y las probabilidades de los 5 modelos son idénticas (diferencia máxima 0.0). Solo cambian los valores en carreras. Por eso los hiperparámetros ajustados siguen siendo válidos.

## 3. Definición de los grupos de estadio anónimos

Fuente: `reports/estadios_anonimos.md` (`src/data/parks.py`).

**Qué es un grupo.** Los juegos que un mismo equipo anónimo jugó como **local**. Si el equipo no cambió de casa, es su estadio.

**Cómo se reconstruye, sin usar la altitud ni ningún resultado:**
1. **Equipos por temporada.**
   - Los compañeros de equipo lanzan y catchean del mismo lado del mismo juego.
   - Con eso se arma un grafo de pitchers y catchers, ponderado por las mitades de juego que comparten, y se agrupa con clustering espectral.
   - El número de grupos k ∈ {16, 18, 20, 22, 24} se elige con un criterio interno: el mínimo de juegos con local = visitante. El resultado es k = 20 / 24 / 22 en 2024 / 2025 / 2026.
   - En la parte alta de cada entrada lanza el local, así que eso define al equipo local de cada juego.
2. **IDs estables entre temporadas.**
   - Los clusters de cada temporada se asignan uno a uno (algoritmo húngaro) a los IDs existentes según la similitud de sus rosters (índice de Jaccard).
   - El enlace se acepta solo si es claro: Jaccard ≥ 0.08 y ≥ 2.5 veces el segundo mejor.
3. **Calidad.**
   - Un grupo-temporada está *resuelto* si ≥ 80% de sus juegos locales cae en una sola categoría y tiene ≥ 15 juegos.
   - Si no, es *mixto* (probablemente dos equipos fusionados) y sus juegos solo se usan a nivel de categoría.

**Resultado:**
- 2,093 de 2,127 juegos (98.4%) quedan en un grupo resuelto.
- 25 IDs resueltos: **8 de Extreme, 6 de Medium y 11 de No Altitude**.
- Por temporada hay 5, 5 y 6 grupos de Extreme.

## 4. Estabilidad de los grupos

| prueba | resultado |
|---|---|
| Juegos con local = visitante | 0.09% |
| Pureza de categoría por grupo (la categoría no se usó para construir) | 100% (2024), 99.9% (2025), 99.6% (2026) |
| ARI con otras semillas del clustering (1 = idéntico) | 2024: 1.000 · 2025: ≥ 0.962 · 2026: ≥ 0.983 |
| ARI reconstruyendo con el 80% de los juegos | 2024: 1.000 · 2025: ≥ 0.997 · 2026: ≥ 0.916 |
| Grupos que aparecen en ≥ 2 temporadas y conservan su categoría | **19 de 19** (la categoría no se usó para enlazar) |
| Enlace entre temporadas | Jaccard mediano 0.18 contra 0.03 del segundo candidato; 10 de 44 clusters sin enlace claro (reciben ID nuevo) |
| Test–retest físico: ratio de drag del grupo en t vs. t+1 | corr = **0.96** (drag) y **0.99** (Magnus), 32 pares; solo Extreme: 0.68 y 0.94 (8 pares) |

**Lectura:** los grupos son estables, se reproducen con otras semillas y submuestras, y conservan su identidad física de una temporada a otra.

**Limitación.** Hay 8 IDs de Extreme, pero cada temporada tiene solo 5–6 grupos. Algunos IDs probablemente son el mismo estadio que no se pudo enlazar porque su roster rotó demasiado, o un equipo partido en dos clusters en 2026. Los análisis por estadio los tratan como grupos separados, lo que es conservador.

## 5. Diferencias físicas entre grupos

**Medición.** Ratio de drag/v² y Magnus/v² del **mismo pitcher, tipo de pitch y temporada** en cada grupo, contra el mismo pitcher en los **otros** estadios de No Altitude. 1 = igual al promedio de No Altitude; < 1 = el aire empuja menos.

| categoría | grupos | ratio de drag (rango) | ratio Magnus (rango) | sd real entre estadios (drag / Magnus) | I² (drag / Magnus) |
|---|---:|---|---|---|---|
| Extreme | 8 | 0.828 – 0.902 | 0.717 – 0.837 | 0.026 / 0.035 | 98% / 95% |
| Medium | 6 | 0.870 – 0.949 | 0.815 – 0.859 | 0.026 / 0.015 | 99% / 90% |
| No Altitude | 11 | 0.971 – 1.064 | 0.910 – 1.082 | 0.026 / 0.044 | 99% / 98% |

I² = fracción de la variación entre estadios que **no** es ruido de muestreo. El error típico de cada ratio es ≈ 0.004.

**Hallazgos:**
1. **"Extreme Altitude" no es un ambiente físico homogéneo.** Sus estadios difieren de forma real: el 98% de la variación en drag entre estadios no es ruido. Lo mismo pasa dentro de Medium y de No Altitude.
2. **Las categorías se traslapan físicamente.** El estadio de Extreme con más drag (0.902) tiene más que el de Medium con menos (0.870). Las categorías del dataset son etiquetas útiles, pero no fronteras físicas nítidas.
3. **En todos los estadios de Extreme el drag es menor que en todos los de No Altitude.** La dirección del efecto es consistente.

## 6. Baseline trivial obligatorio y jerarquía de modelos

Fuente: `reports/baselines.md` (`src/validation/baselines.py`). Todos los modelos usan las mismas predicciones out-of-fold, el mismo shrinkage y la misma escala; solo cambia cómo se trata el entorno.

| modelo | trato del entorno |
|---|---|
| **M0** sin ajuste | valor esperado con las características observadas |
| **M1** ajuste simple | menos su promedio por temporada × categoría de altitud |
| **M1b** ajuste simple por estadio | menos su promedio por temporada × estadio anónimo |
| **M2** contrafactual físico (índice oficial actual) | cada pitch proyectado a No Altitude con ratios de densidad por categoría × tipo de pitch |
| **M3a** contrafactual físico por estadio | igual, con el ratio de densidad del estadio anónimo del juego (98.6% de los pitches) |
| **M3b** contrafactual + ajuste por estadio | M2 menos su promedio por temporada × estadio anónimo |

**Elección de la especificación (estructura jerárquica: pitch → pitcher → juego → estadio anónimo → categoría):**
- **Ratios por estadio: estimación directa, sin modelo jerárquico.** Cada estadio tiene cientos de celdas y error de ≈ 0.004 contra una sd real entre estadios de ≈ 0.026. Un shrinkage hacia la media de su categoría movería cada ratio menos del 3%: no se justifica la complejidad.
- **Efecto de la categoría: no identificable por separado del estadio.** Cada estadio pertenece siempre a una sola categoría. En un modelo con efectos fijos de estadio, la categoría queda absorbida; solo se puede estimar como el promedio de sus 6–8 estadios. Generalizar "el efecto de la altitud extrema" a estadios fuera de la muestra requeriría efectos aleatorios de estadio estimados con 6–8 grupos, con muy poca información sobre su varianza. Para P1 se propone reportar efectos por estadio y su promedio por categoría, con incertidumbre por remuestreo de estadios, en lugar de afirmar un "efecto causal de la altitud".

## 7. Contrafactual físico

Fuente: `reports/altitude_counterfactual_v2.md`.

**Modelo:**
- Trayectoria de 9 parámetros con aceleración constante.
- La aceleración sin gravedad se separa en una componente paralela a la velocidad (*drag efectivo*) y una perpendicular (*Magnus efectivo*). Es una descomposición geométrica de lo observado, no la medición directa de las fuerzas físicas verdaderas.
- Cada componente se escala por la razón de densidades efectivas entre ambientes, y el pitch se vuelve a lanzar desde el mismo punto de salida al mismo objetivo en el plato.

**Efecto del mismo pitch en Extreme vs. No Altitude** (modelos finales): **+1.10 carreras por 100 pitches** [IC 95%: 1.08, 1.11]. Ese intervalo solo refleja el muestreo de pitchers.

**Atribución de Shapley por eslabón** (contable, no causal):

| eslabón | carreras/100 |
|---|---:|
| antes del contacto (swing + take + whiff + foul) | **+0.39** (35%) |
| resultado de la bola en juego (calidad del contacto + parque) | +0.71 (65%) |

**Mecanismo promedio:** P(whiff | swing) −1.8 puntos porcentuales y P(pelota en juego) +1.3 pp. En la recta de 4 costuras: IVB −4.0 pulgadas y velocidad en zona +1.4 mph.

**Tres utilidades distintas del contrafactual:**

| utilidad | pregunta | estado |
|---|---|---|
| **Predictiva** | ¿predice mejor el rendimiento futuro? | **No demostrada** frente a los ajustes simples (sección 8) |
| **Explicativa** | ¿permite entender cómo cambia físicamente un lanzamiento? | **Sí:** reproduce la trayectoria, las features cambian en la dirección física esperada y da la atribución por eslabón |
| **Contrafactual** | ¿cómo se comportaría el mismo lanzamiento en otro ambiente? | **Sí, bajo supuestos:** aceleración constante, mismo objetivo en el plato y ratios efectivos. El cambio de IVB se sobrepredice ~20% (validación split-half) |

## 8. Comparación predictiva

**Criterio de éxito** en todas las comparaciones: Δr > 0 y P(Δr > 0) ≥ 0.90 (bootstrap pareado por pitcher, 2,000 réplicas).

### 8.1 Validación preespecificada

Bitácora §7.2: carreras evitadas observadas en t+1, mín. 100 pitches, contra Pitching+ v1. 437 pares.

| modelo | r con carreras observadas t+1 | Δr vs. Pitching+ v1 (P) |
|---|---:|---|
| Pitching+ v1 (referencia) | 0.188 | — |
| M0 | 0.177 | −0.011 (36%) |
| M1 | 0.169 | −0.019 (32%) |
| M1b | 0.175 | −0.013 (37%) |
| M2 | 0.153 | −0.035 (20%) |
| M3a | 0.156 | −0.032 (21%) |
| M3b | 0.159 | −0.029 (24%) |

**Ningún modelo cumple el criterio preespecificado.**

### 8.2 Validación secundaria (post-hoc)

Carreras t+1 ajustadas por entorno, mín. 300 pitches, 245 pares.

| modelo | r (ajuste del resultado por categoría) | r (ajuste por estadio) |
|---|---:|---:|
| M0 | 0.215 | 0.219 |
| M1 | 0.279 | 0.276 |
| M1b | 0.280 | 0.283 |
| M2 | 0.279 | 0.277 |
| **M3a** | **0.292** | **0.297** |
| M3b | 0.285 | 0.289 |

**Contrastes clave:**
- **M2 − M1 = +0.000 (P = 53%).** El contrafactual por categoría no aporta nada sobre el ajuste simple.
- **M3a − M1 = +0.013 (P = 79%)** con el resultado ajustado por categoría, y **+0.021 (P = 88%)** con el ajustado por estadio. M3a − M1b = +0.012 (P = 80%). Es el mejor punto estimado, pero **no llega al umbral**.
- Con mín. 100 pitches, M1b es el mejor (0.198–0.201) y M3a queda por debajo (0.188–0.191).

### 8.3 Validación temporal estricta

Todo se entrena con temporadas < t, incluidas la matriz RE24 y los ratios por estadio. Única pareja evaluable: 2025 → 2026.

| resultado | mín. | M1 | M1b | M2 | M3a | M3a − M1b (P) |
|---|---|---:|---:|---:|---:|---|
| observado | 100 | 0.172 | **0.191** | 0.148 | 0.153 | −0.038 (1%) |
| observado | 300 | 0.275 | **0.302** | 0.258 | 0.279 | −0.023 (12%) |
| ajustado | 100 | 0.162 | **0.175** | 0.136 | 0.143 | −0.032 (3%) |
| ajustado | 300 | 0.305 | **0.323** | 0.281 | 0.296 | −0.027 (14%) |

**Con entrenamiento temporal estricto, el ajuste simple por estadio anónimo (M1b) es el mejor en las 4 combinaciones**, y ningún contrafactual físico lo supera. La ventaja de M3a en 8.2 no se sostiene fuera de la muestra temporal.

### 8.4 Neutralización

Correlación del score con el % de pitches del pitcher en Extreme y con su exposición física (ratio de drag promedio de los estadios donde lanzó). Pitcher-temporadas con ≥ 100 pitches.

| modelo | % Extreme | exposición física |
|---|---:|---:|
| M0 | −0.325 | +0.398 |
| M1 | −0.042 | +0.069 |
| M1b | −0.024 | +0.022 |
| M2 | −0.062 | +0.082 |
| M3a | −0.054 | +0.038 |
| M3b | −0.014 | +0.010 |

Los modelos que usan el estadio anónimo (M1b, M3b) son los que mejor eliminan la huella del ambiente.

**Respuesta a la pregunta de P0:** ¿qué valor adicional aporta la modelación física más allá de una normalización ambiental simple?
- **Predictivo:** ninguno demostrado. El mejor punto estimado de la validación post-hoc (M3a) no se sostiene en la validación temporal estricta, donde gana el ajuste simple por estadio.
- **Explicativo y contrafactual:** sí aporta (sección 7).

## 9. Tests

`python -m pytest tests`: **25 PASS**.

| archivo | qué verifica | estado |
|---|---|---|
| `tests/test_targets.py` (4) | reconstrucción de HBP; indicadores terminales no vacíos; golpeado ≈ base por bolas; signo de Δ en los 12 conteos | PASS (3 fallaban con los datos de antes de la corrección) |
| `tests/test_math.py` (11) | Σ P = 1 (incluidos π = 0 y 1, y 627,185 pitches OOF); Bellman con foul en 2 strikes; cinemática; D + M = a_ng; falla explícita con ratios inválidos; contrafactual sin NaN y físico; shrinkage; signo y escala; eficiencia de Shapley | PASS |
| `tests/test_parks.py` (4, nuevos) | ARI; IDs con formato `ANON_STADIUM_xx` (solo anónimos); calidad (≥ 95% resueltos, local ≠ visitante, categoría estable entre temporadas); ratios físicamente plausibles (todo Extreme < todo No Altitude); proyección con ratios iguales = identidad; fallback a la categoría | PASS |
| `tests/test_trajectory.py` (6) | identidad del contrafactual; dirección física; ratios < 1; pitch sintético; shrinkage; simetría zurdo/derecho | PASS |

**Verificaciones adicionales:**

| verificación | estado |
|---|---|
| Pipeline completo, 21 pasos | PASS |
| Dashboard (ambas pestañas, `streamlit.testing.AppTest`) | PASS |
| Identidad de etiquetas y probabilidades antes/después de HBP | PASS |
| Sobrepredicción de ~20% del cambio de IVB | WARNING |
| Criterio preespecificado | FAIL: resultado científico, no error de código |
| Física > ajuste simple | FAIL: resultado científico |

## 10. Conclusiones de P0

**Resultados demostrados:**
1. **El golpeado ya se valúa correctamente** (+0.398 carreras), desde la fuente del target, y todo el pipeline está regenerado.
2. **La reconstrucción de estadios anónimos es estable y física:**
   - 98.4% de los juegos resueltos;
   - ARI ≥ 0.91 con otras semillas y submuestras;
   - 19 de 19 grupos conservan su categoría entre temporadas;
   - test–retest físico de 0.96–0.99.
3. **"Extreme Altitude" no es un ambiente homogéneo.** Sus 8 grupos anónimos tienen ratios de drag de 0.83 a 0.90 y de Magnus de 0.72 a 0.84, y esas diferencias son reales (I² ≥ 95%). Las categorías se traslapan físicamente con Medium.
4. **Los baselines obligatorios ganan o empatan.** Ningún modelo cumple el criterio preespecificado. En la validación temporal estricta, el mejor es el ajuste simple por temporada × estadio anónimo (M1b), y la física no lo supera.
5. **El contrafactual físico conserva su valor explicativo y contrafactual,** no predictivo.

**Resultados sugerentes:**
- Usar la densidad del estadio anónimo (M3a) mejora un poco al contrafactual por categoría y es el mejor punto en la validación post-hoc (P = 79–88%). No se sostiene en la temporal estricta.
- Dentro de No Altitude también hay variación física real entre estadios (sd 0.026 en drag).

**Hipótesis para P1:**
- Que la diferencia de valor de la bola en juego entre Extreme y No Altitude persista controlando por estadio anónimo y composición.
- Que las características físicas que sostienen el whiff en Extreme sean identificables.

## 11. Limitaciones

1. **Grupos anónimos.**
   - Suponen que un equipo juega siempre como local en el mismo estadio.
   - Algunos IDs de Extreme probablemente son el mismo estadio sin enlazar.
   - 34 juegos (1.6%) quedan en grupos mixtos y solo se usan a nivel de categoría.
2. **Categoría y estadio no se separan.** El efecto de la categoría solo existe como promedio de 6–8 estadios. Ningún resultado de este proyecto debe presentarse como "efecto causal de la altitud".
3. **Ratios efectivos.** Los ratios de densidad mezclan densidad del aire, clima, calibración de Trackman por estadio y pelota.
4. **Pruebas post-hoc.** Las validaciones secundarias (ajustadas por entorno) y los modelos M1b, M3a y M3b se definieron después de ver la validación preespecificada. Son exploratorias y hay comparaciones múltiples.
5. **Potencia limitada.**
   - Validación original: 437 pares (mín. 100) y 245 (mín. 300).
   - Validación temporal: 215 y 99 pares de una sola pareja de temporadas.
6. **Componentes globales en la validación temporal:** la reconstrucción de grupos usa las 3 temporadas (solo quién jugó con quién, no resultados) y los hiperparámetros son globales.
7. **El efecto de altitud del contrafactual** usa modelos in-sample, así que su IC es demasiado estrecho.

---

# ¿Qué hicimos y por qué?

**El problema.** Queremos saber cuánto vale un lanzamiento y cómo cambia ese valor cuando se juega en estadios de gran altitud, donde el aire es menos denso. Para eso usamos una base de datos con más de 600 mil lanzamientos de tres temporadas. La base viene **anonimizada**: no dice en qué estadio se jugó cada partido ni qué equipo es cuál, solo si el juego fue en un ambiente de altitud nula, media o extrema. Decidimos **respetar esa anonimización**: no buscamos nombres reales ni usamos calendarios externos. Así, cualquiera puede repetir el análisis con los mismos datos.

**1. Arreglamos un error en los datos.** Cuando un lanzamiento golpea al bateador, el bateador avanza a primera base, igual que con una base por bolas. Pero una columna de la base de datos venía vacía, y nuestro sistema trataba el golpe como si no hubiera pasado nada; en algunos conteos, incluso como algo bueno para el pitcher. Lo corregimos desde el origen y recalculamos todo. Los pitchers que golpean más bateadores bajaron un poco en el ranking, como debe ser.

**2. Descubrimos que "altitud extrema" no es un solo lugar.** Aunque la base no dice el estadio, sí dice qué pitchers y receptores jugaron juntos en cada partido. Los compañeros de equipo juegan juntos, y el equipo local siempre lanza en la primera mitad de cada entrada. Con eso agrupamos los partidos por "equipo local" sin saber su nombre: los llamamos `ANON_STADIUM_01`, `ANON_STADIUM_02`, etc. Comprobamos que la agrupación es confiable:
- sale igual si repetimos el cálculo de otra forma;
- cada grupo conserva su categoría de altitud de un año a otro;
- la física de cada grupo se repite año con año.

Resultado: dentro de "altitud extrema" hay unos 5–8 estadios distintos, y en algunos el aire frena y desvía la pelota claramente menos que en otros. Es decir, "altitud extrema" es una etiqueta que agrupa ambientes parecidos, pero no idénticos.

**3. Pusimos a prueba nuestro modelo físico contra métodos simples.** Nuestro modelo físico calcula cómo se movería el mismo lanzamiento a nivel del mar: con más aire, la pelota se frena más y "quiebra" más. Con eso estimamos qué tan bueno es cada pitcher si todos jugaran en las mismas condiciones. La pregunta honesta es si eso **predice mejor el futuro** que algo mucho más simple: comparar a cada pitcher contra el promedio de su ambiente, por año y por estadio. La respuesta es **no**:
- los métodos simples predicen igual o mejor la temporada siguiente;
- cuando entrenamos solo con el pasado para predecir el futuro, que es la prueba más estricta, el método simple por estadio fue el mejor.

**4. Entonces, ¿para qué sirve el modelo físico?** Para **entender** y para **imaginar escenarios**. Nos dice:
- que una recta en altitud extrema pierde unas 4 pulgadas (≈10 cm) de "levantamiento" y llega más rápido al plato;
- que eso produce menos swings fallidos y más pelotas en juego;
- que la mayor parte del daño (dos tercios) aparece cuando la pelota ya está en juego.

Ese tipo de explicación no la da un promedio simple. Pero para **predecir**, el método simple es suficiente.

**5. Qué podemos y qué no podemos decir.**
- **Podemos decir** que en los estadios de altitud extrema de estos datos los lanzamientos pierden movimiento, se fallan menos swings y la pelota en juego cuesta más carreras.
- **No podemos decir** que eso se deba solo a la altura: cada estadio tiene además su propio clima, sus dimensiones y su calibración del radar, y con tan pocos estadios por categoría no se pueden separar.
- **Tampoco** podemos decir qué estadio real es cada grupo, y por decisión del proyecto no lo intentamos.

**Siguiente paso (P1, todavía no iniciado):** estudiar a fondo por qué la pelota en juego es más dañina en altitud extrema, qué características ayudan a provocar swings fallidos en esos ambientes y construir medidas separadas de "calidad física" (Stuff+) y de "ubicación" (Location+).
