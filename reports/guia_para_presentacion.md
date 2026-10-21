# Guía para presentar el proyecto

**Pitching+ para ambientes de altitud.** Esta guía no es otro reporte técnico. Es para defender el proyecto en voz alta frente a
profesores, científicos de datos, analistas de béisbol y personas que saben de béisbol pero poco de estadística.

Cada concepto tiene cuatro niveles:

1. **Respuesta corta** (15–30 segundos, para decirla de memoria).
2. **Explicación intuitiva** (qué significa en béisbol).
3. **Explicación técnica** (cómo se calculó de verdad).
4. **Pregunta probable del jurado** y la respuesta recomendada.

Todas las cifras salen de `data/web/summary.json`, `data/web/pitching_env_values.json` y de los reportes `reports/p2_pitcher_profiles_altitude.md`,
`reports/pitching_plus_general_vs_altitude.md` y `reports/optimal_pitch_by_environment.md`. Los estadios son anónimos (`ANON_STADIUM_xx`) y los
pitchers también (`pitcher_xxxxx`): no se usa ningún nombre real.

---

## 0. El proyecto en una frase (abrir y cerrar con esto)

> En los ambientes de altitud analizados, **el contacto cuesta más y los whiffs y ponches son más difíciles de generar**. Por eso los
> pitchers con perfil de **swing-and-miss** tienen una ventaja relativa, aunque el perfil favorable se parece mucho al de un buen pitcher en
> cualquier lado. **Pitching+** es nuestra herramienta para convertir física, ubicación y contexto en el valor esperado de cada lanzamiento.

**Las cuatro cifras que hay que saber de memoria (Extreme contra No Altitude, mismo pitcher contra el mismo bateador):**

| | cifra |
|---|---|
| Whiff por swing | **−2.9 puntos porcentuales** (de 24.9%) |
| Ponches por turno | **−4.7 pp** (de 20.3%) |
| Run value por bola en juego | **+0.068 carreras** |
| Valor de cambiar una bola en juego por un ponche | **≈ 27% más** |

El patrón aparece en **16 de 16** combinaciones estadio anónimo × temporada de Extreme.

---

## 1. Stuff+

### Respuesta corta
Stuff+ mide qué tan difícil es pegarle a un lanzamiento solo por cómo se mueve y qué tan rápido va, sin darle crédito por dónde lo
puso el pitcher ni por el ambiente. 100 es el promedio de la liga; cada 10 puntos son una desviación estándar.

### Explicación intuitiva
Es el "qué tan bueno es el pitch en sí". Una recta de 97 mph con mucho movimiento vertical o un slider con mucho quiebre hacen fallar más
swings, aunque el pitcher no los ubique perfecto. Más de 100 = mejor que el promedio; menos de 100 = peor. Un 110 está una desviación
estándar por encima; un 90, una por debajo.

### Explicación técnica
- **Target:** P(whiff | swing), la probabilidad de que el bateador falle cuando hace swing.
- **Modelo:** LightGBM con validación cruzada agrupada por pitcher (ningún pitcher está en entrenamiento y prueba a la vez).
- **Variables:** velocidad, spin rate, eje de giro, altura y lado de salida, extensión, movimiento vertical inducido (IVB) y horizontal (HB),
  ángulos de llegada **corregidos por la ubicación** (`vaa_loc_adj`, `haa_loc_adj`), diferencias contra la recta del pitcher y el tipo de pitch.
  El lado del bateador se promedia con la mezcla de la liga.
- **Por pitcher:** promedio con *shrinkage* empírico-bayesiano y escala 100 + 10·(θ̂ − μ)/τ, donde τ es la desviación estándar de la habilidad real.
- **Calidad:** estabilidad año a año 0.76; la ubicación explica solo 1.9% de su predicción.

### Pregunta probable
**"¿Por qué no usan velocidad y spin directamente?"** — Porque ninguna variable sola explica el whiff: lo que importa es la combinación
(movimiento, velocidad, salida, diferencia con la recta). El modelo aprende esas combinaciones y las resume en una sola escala comparable.

---

## 2. Los distintos Stuff+ del proyecto (muy importante)

### Respuesta corta
Probamos varias versiones. Stuff+ v1 predecía bien pero "hacía trampa": veía la ubicación y la altitud. Stuff+ Physical oficial es el que
usamos: mide solo la física, con un ajuste simple por ambiente, y es el más estable y el que mejor predice el whiff del año siguiente.

### Explicación intuitiva
Si un Stuff+ sabe dónde terminó el pitch, deja de medir "qué tan bueno es el pitch" y empieza a medir "qué tan bien lo ubicaron". Y si
sabe en qué estadio se lanzó, castiga o premia al pitcher por dónde le tocó lanzar.

### Explicación técnica

| versión | qué mide | usa ubicación | usa ambiente | para qué se usa |
|---|---|---|---|---|
| **Stuff+ v1** | P(whiff) por pitch | sí, implícita: ángulos de llegada sin corregir (la ubicación explica 46% de su predicción) | sí: `altitude_level` como variable | benchmark predictivo (AUC 0.686 en swings), no Stuff conceptual |
| **Stuff+ v4** | P(whiff \| swing) | no | se neutraliza con el contrafactual físico | intento de Stuff "puro"; predijo peor (whiff t+1: 0.304 vs. 0.332 de v1) |
| **Physical observado** | P(whiff \| swing) | no | nada: lo que el bateador vio | explicar lo que pasó; castiga a quien lanzó en altitud (r = −0.16 con % de pitches en Extreme) |
| **Physical oficial** | P(whiff \| swing) | no (1.9%) | ajuste simple por temporada × estadio anónimo | **ranking, scouting y la página** (estabilidad 0.76; whiff t+1 0.340) |
| **Physical neutral** | P(whiff \| swing) | no | contrafactual físico a No Altitude | explicar el mecanismo; no mostró ventaja predictiva (whiff t+1 0.308) |

- **v1 tiene el AUC más alto**, pero en parte porque usa información que no es "stuff": la ubicación y el ambiente.
- **v4 predijo peor** porque la neutralización física agrega error (el contrafactual sobreestima ≈ 20% el cambio de IVB) y pierde
  información real del ambiente.
- **Physical oficial** separa Stuff de Location (correlación −0.04 con Location+), es el más estable (0.76) y predice mejor el whiff del año
  siguiente que v4 (P = 98%).

### Pregunta probable
**"¿Por qué no se quedaron con el de AUC más alto?"** — Porque el AUC alto de v1 viene de ver la ubicación. Si Stuff+ incluye la ubicación,
ya no podemos separar "pitch bueno" de "pitch bien ubicado", y esa separación es la base de Stuff+, Location+ y Pitching+.

---

## 3. Location+

### Respuesta corta
Location+ mide qué tan buena fue la ubicación del lanzamiento, dado el tipo de pitch, el conteo y el bateador. 100 es la ubicación típica
en ese contexto.

### Explicación intuitiva
El mismo slider abajo y afuera con dos strikes suele ser un gran lugar; en el centro con el pitcher atrás en la cuenta, no. Por eso
importan el **conteo** (no es lo mismo 0-2 que 3-1) y el **tipo de pitch** (el lugar ideal de una recta no es el de un cambio).

### Explicación técnica
- Mismo árbol de probabilidades que Pitching+, pero solo con ubicación (lado relativo al bateador y altura), conteo, tipo de pitch y manos.
- Puntaje por pitch = E[Δ | ubicación, contexto] − promedio de E[Δ] en ese contexto.
- Casi independiente de Stuff+ (r = −0.04); predice el BB% del año siguiente (r = −0.39) mejor que el propio BB% (0.35).
- **Mapas de calor:** azul = zonas que **suelen** ser más favorables para ese tipo en ese conteo que su ubicación típica; rojo = menos. Solo se
  muestran celdas con suficientes pitches reales.
- **Hallazgo:** los pitches en el **corazón de la zona** son los que más valor pierden en altitud (+2.92 carreras por 100 pitches en
  Extreme, contra +0.16 en la zona de desperdicio). Probable razón: ahí hay mucho contacto, y el contacto es más caro.

### Pregunta probable
**"¿Entonces nunca hay que lanzar al centro en altitud?"** — No. Es una asociación promedio. Los strikes en el centro también sirven para
no caer atrás en la cuenta. Lo que decimos es que el contacto en el centro sale más caro en esos ambientes.

---

## 4. Pitching+ (la más importante)

### Respuesta corta
**Pitching+ mide cuánto valor esperamos que produzca un lanzamiento considerando sus características físicas, su ubicación, el contexto y las
probabilidades de cada resultado.** Si ese pitch se lanzara muchas veces en la misma situación, ¿cuántas carreras le costaría o le ahorraría al
pitcher en promedio? 100 es el lanzamiento promedio; más alto es mejor.

### Explicación intuitiva
Un pitch puede tener gran Stuff pero quedar en el centro con la cuenta en contra; otro puede tener Stuff normal y estar perfectamente
ubicado. Pitching+ los pone en la misma moneda: carreras.

### Explicación técnica
Cinco modelos LightGBM encadenados (el árbol del lanzamiento):

1. **P(swing)** — ¿le tira?
2. **P(resultado | no swing)** — bola, strike cantado o golpeado.
3. **P(whiff | swing)** — si le tira, ¿falla?
4. **P(foul | contacto)** — si hace contacto, ¿se va de foul?
5. **P(resultado | bola en juego)** — out, sencillo, doble, triple o jonrón.

Las 10 probabilidades suman 1. Luego:

**E[Δ] = Σ_evento P(evento | física, ubicación, contexto) × valor del evento en ese conteo**

El valor de cada evento sale de RE24: cuánto cambian las carreras esperadas del turno (una bola en 1-1 lleva a 2-1; un strike lleva a 1-2; un
sencillo termina el turno con su valor). Pitching+ = 100 + 10·(media − E[Δ])/sd: menos carreras esperadas = más Pitching+.

**¿Por qué no simplemente Stuff+ + Location+?** Porque no se suman: el valor de un buen movimiento depende de dónde termine el pitch y del
conteo. Un slider con gran quiebre vale mucho fuera de la zona con dos strikes y poco en el centro en 3-0. Pitching+ calcula esa interacción
con el modelo de probabilidades en lugar de sumar dos números con pesos inventados.

### Pregunta probable
**"¿Cómo convierten una probabilidad en carreras?"** — Multiplicando cada probabilidad por lo que vale ese resultado en carreras (RE24) y
sumando. Ejemplo: si un pitch tiene 30% de terminar en bola en juego y una bola en juego vale en promedio +0.05 carreras frente a seguir el turno,
esa rama aporta 0.30 × 0.05 = +0.015 carreras.

---

## 5. Run Value por Bola en Juego (RV/BIP)

### Respuesta corta
RV/BIP es el **valor esperado en carreras de una pelota una vez que fue puesta en juego**. Si sube, cada bola en juego le cuesta más
carreras al pitcher.

### Explicación intuitiva
Imagina que cada contacto es una multa. **BIP%** dice **qué tan seguido** pagas la multa. **RV/BIP** dice **qué tan cara** es cada multa.
En Extreme la multa es más cara: +0.068 carreras por bola en juego frente a No Altitude (promedios crudos: No Altitude +0.015, Medium +0.073,
Extreme +0.085).

### Explicación técnica
Promedio del run value RE24 de las jugadas que terminan con la pelota en juego. El efecto de +0.068 sale de una regresión con efectos fijos
de pitcher, bateador, temporada y conteo (mismo pitcher contra el mismo bateador), con errores agrupados por juego.

### Pregunta probable
**"¿No es lo mismo que BABIP?"** — Se parece, pero BABIP solo cuenta si fue hit. RV/BIP pesa cada resultado por su valor en carreras: un
jonrón cuenta mucho más que un sencillo.

---

## 6. Daño esperado de contacto

### Respuesta corta
**Daño esperado de contacto = P(BIP) × E[RV | BIP]**: qué tan seguido hay bola en juego por lo caro que sale cada una.

### Explicación intuitiva
Un pitcher se defiende de dos formas: (1) **evitando que haya bola en juego** (whiffs, ponches) o (2) **haciendo que el contacto sea menos
dañino** (rodados, contacto débil).

### Explicación técnica
- El daño por 100 pitches sube de 0.26 a 1.62 carreras de No Altitude a Extreme. **93% del aumento viene de que cada bola en juego hace más
  daño**; solo 1% de que haya más bolas en juego.
- La identidad P(BIP) × E[RV | BIP] coincide numéricamente con la rama de bola en juego del árbol de Pitching+.
- **Pero las dos defensas no valen lo mismo como habilidad.** Evitar contacto se repite de un año a otro (whiff/swing 0.51, K% 0.44, BIP%
  0.52). Controlar el daño casi no se repite (RV/BIP −0.02, hard contact 0.11, HR/BIP 0.13). Por eso el perfil se apoya en evitar contacto.

### Pregunta probable
**"¿Entonces no importa inducir rodados?"** — Sí importa, como segunda línea de defensa: un rodado en lugar de un batazo al aire ahorra
0.140 carreras en No Altitude y 0.202 en Extreme. Pero el % de rodados casi no predice las carreras del año siguiente en altitud, así que no
lo usamos como métrica central.

---

## 7. ¿Por qué nos importan tanto el whiff y el ponche?

### Respuesta corta
Porque en los ambientes de altitud el contacto cuesta más y, a la vez, los whiffs y ponches son más difíciles. Lo que se vuelve escaso y
valioso es justo evitar la bola en juego.

### Explicación intuitiva
Si cada hit cuesta más y además es más difícil poncharlos, el pitcher que todavía puede hacer fallar swings tiene una ventaja que vale más
que en otros parques.

### Explicación técnica
Whiff/swing −2.9 pp; K/PA −4.7 pp; RV/BIP +0.068; cambiar una bola en juego por un ponche vale ≈ 27% más por turno (0.432 contra 0.340
carreras), y convertir una bola en juego en whiff vale ≈ 27% más por swing.

### Pregunta probable
**"¿Qué significa 27% más?"** — Que la diferencia en carreras entre un ponche y una bola en juego pasa de 0.340 en No Altitude a 0.432 en
Extreme. Evitar el contacto "paga" 27% más.

---

## 8. AEPP (Altitude Environment Pitcher Profile)

### Respuesta corta
AEPP son las carreras evitadas por 100 turnos frente a un pitcher promedio, usando lo que vale cada resultado (ponche, base por bolas,
golpeado, bola en juego) en los ambientes de altitud. Los pesos no se eligieron a mano: son valores en carreras medidos en esos ambientes.

### Explicación intuitiva
Tomamos lo que hace el pitcher (cuántos ponches, bases y bolas en juego permite y qué tan caras son) y le ponemos los precios de los
ambientes de altitud.

### Explicación técnica
- AEPP = −100·[Σ_{resultado ≠ BIP} (tasa − tasa promedio)·(valor del resultado − valor de BIP) + tasa de BIP·(daño por BIP − promedio)],
  con tasas con *shrinkage*, ajustadas por entorno, y pesos lineales de Medium + Extreme.
- Predice las carreras en altitud del año siguiente mejor que Pitching+ (r = 0.26 contra 0.15; P = 98%).
- **No supera a K−BB% con shrinkage** (0.26 contra 0.25: empatados).
- El ranking con pesos de altitud y con pesos de No Altitude correlaciona **0.986**.
- La incertidumbre es grande: el intervalo de 80% del lugar tiene una amplitud mediana de **294 posiciones** (de 599). El #1 está entre el 1
  y el 3. Por eso mostramos **P(top 25)** además de la posición.

**¿Por qué sigue siendo útil?** Por la interpretabilidad: dice de dónde vienen las carreras evitadas (ponches, bases, contacto) con los precios
del ambiente.

### Pregunta probable
**"¿Por qué AEPP no supera a K−BB%?"** — Porque K−BB% ya captura casi todo lo que se repite en un pitcher; el daño por contacto que AEPP
agrega es casi ruido año a año. Lo reportamos tal cual: AEPP empata con K−BB%.

---

## 9. "Si K−BB% funciona igual, ¿para qué hicieron todo esto?"

### Respuesta corta
K−BB% es un excelente predictor simple y lo reconocemos. Nuestro proyecto responde preguntas que K−BB% no puede: **por qué** pasa, **qué
características físicas** lo generan, **qué ubicación** ayuda, **cómo cambia el valor con el ambiente** y **cómo evaluar un lanzamiento
individual antes de conocer el resultado**.

### Explicación intuitiva
K−BB% es el marcador final; Pitching+ es el análisis de cada jugada. Con K−BB% sabes que un pitcher poncha mucho; con Stuff+, Location+ y
Pitching+ sabes qué pitch, en qué lugar y en qué conteo lo logra, y puedes evaluar un lanzamiento nuevo (por ejemplo, de un prospecto) sin
esperar cientos de turnos.

### Explicación técnica
K−BB% necesita muchos turnos para estabilizarse y no se puede calcular para un lanzamiento. Pitching+ trabaja por pitch, usa la física
medida por el Trackman y está dentro de un modelo probabilístico que permite simular ("¿qué pasa si sube 2 mph?") y comparar ambientes.

---

## 10. Contrafactual físico

### Respuesta corta
Reconstruimos la trayectoria de cada pitch y la volvemos a "volar" con el drag y el Magnus medidos en otro ambiente. Sirve para simular,
explicar y estudiar mecanismos. **No** demostró predecir mejor que un ajuste simple por estadio.

### Explicación intuitiva
Es como preguntarle a la física: "si este mismo pitch se lanzara a nivel del mar, ¿cuánto más se movería?". Con menos aire, la pelota
frena menos (menos drag) y el giro la mueve menos (menos Magnus).

### Explicación técnica
- Cinemática con aceleración constante; drag y Magnus escalados por los ratios medidos por estadio anónimo.
- Reproduce bien la ubicación, ZoneSpeed y VAA, pero **sobreestima ≈ 20% el cambio de IVB** (validación split-half).
- En la validación temporal, el ajuste simple por temporada × estadio (modelo M1b) predijo igual o mejor que el contrafactual. Por eso es
  una **simulación física aproximada / contrafactual explicativo**, no el índice oficial.

### Pregunta probable
**"Si la física no predice mejor, ¿qué aporta?"** — Explica el mecanismo (por qué se pierde movimiento), permite simular escenarios y nos
mostró que no todos los estadios de Extreme son iguales (drag 0.83–0.90; Magnus 0.72–0.84).

---

## 11. Sensibilidad individual a la altitud

### Respuesta corta
"¿Qué pitcher es más resistente a la altitud?" **Hoy no lo podemos medir con suficiente confiabilidad.** La confiabilidad es ≈ 0.11 y los
294 intervalos (de 294 pitchers) cruzan el cero. Por eso hablamos de **perfil favorable para ambientes de altitud**, no de **resistencia
demostrada**.

### Explicación intuitiva
Para saber si un pitcher rinde distinto en altitud necesitaríamos muchísimos lanzamientos suyos en esos parques; la mayoría tiene muy pocos
(mediana: 312 pitches en Extreme).

### Explicación técnica
- Confiabilidad por mitades de juegos (Spearman–Brown) = 0.11. Para llegar a 0.5 harían falta ≈ 2,500 pitches en Extreme por pitcher.
- La sensibilidad que predice el modelo físico no predice la observada: Pearson +0.06, Spearman +0.02. Con confiabilidad 0.11, incluso un
  modelo perfecto correlacionaría como mucho ≈ √0.11 ≈ 0.33 con lo observado (orientación de teoría de medición, no cota universal).
- Parte de la "sensibilidad" es la calidad de los rivales (r = +0.24).

---

## 12. Altitud vs. estadio (causalidad)

### Respuesta corta
**No demostramos que la altitud por sí sola cause todo.** La categoría de altitud y el estadio están parcialmente confundidos, y Medium se
parece mucho a Extreme. La conclusión correcta es "**estos ambientes presentan estos patrones**", no "la altitud por sí sola los causa".

### Explicación intuitiva
Cada categoría tiene pocos estadios. Un parque alto también puede tener otras dimensiones, otro clima u otra calibración del Trackman. No
podemos separar todo eso.

### Explicación técnica
- Las comparaciones controlan por pitcher, bateador, temporada y conteo, y el patrón aparece en 16/16 estadio-temporadas de Extreme.
- Medium: run value por BIP +0.073 contra +0.085 de Extreme y +0.015 de No Altitude (promedios crudos). Por eso hablamos de "ambientes de
  altitud" y no de un umbral mágico.
- **Pitching+ no depende de un claim causal:** su lógica es "en estos contextos observados, ciertos resultados tienen un costo diferente".

---

## 13. Pitching+ general vs. Pitching+ para altitud (extensión)

### Respuesta corta
Es el **mismo lanzamiento** valuado con dos juegos de precios: los de toda la liga y los de Extreme. Stuff+ y Location+ no cambian. Lo que
cambia es lo que vale cada resultado.

### Explicación intuitiva
En Extreme una bola en juego vale más para la ofensiva, pero también es más caro todo el turno. Por eso el valor de un pitch cambia poco
en general, y **gana sobre todo el pitch de swing-and-miss con dos strikes**, donde el swing fallado es ponche.

### Explicación técnica
- General = los valores del Pitching+ oficial (verificado: diferencia 0).
- Extreme = la misma recursión RE24 por conteo con los pitches de Extreme, más una **calibración del costo de contacto**: el modelo
  subestima el daño por bola en juego en altitud, y se agrega lo que falta según lo observado. Sin pesos manuales.
- Lanzamiento por lanzamiento, el orden casi no cambia (correlación 0.997; cambio mediano 0.5 puntos).
- Frente al lanzamiento típico del mismo conteo: con dos strikes, la ganancia en Extreme sigue al swing-and-miss (correlación +0.70); con 0
  o 1 strike casi no (+0.07).

### Pregunta probable
**"¿No se contradice con que el contacto cuesta más?"** — No. La bola en juego sí cuesta más (+0.068 carreras). Pero si el pitch no termina
en contacto y el turno sigue, en Extreme el resto del turno también es más caro. Evitar el contacto rinde más cuando la alternativa es el
ponche: con dos strikes. Es exactamente el hallazgo de P1 visto lanzamiento por lanzamiento.

---

## 14. ¿Es el mismo un buen lanzamiento en cualquier estadio? (extensión)

### Respuesta corta
**Prácticamente sí.** Buscamos, entre lanzamientos reales, la mejor versión de cada tipo con los precios de la liga y con los de Extreme. El
óptimo casi no cambia: un gran lanzamiento sigue siendo un gran lanzamiento.

### Explicación intuitiva
No buscamos un pitch imposible: el "óptimo" sale de lanzamientos que de verdad existieron. Con los precios de Extreme, el mejor slider (o
la mejor recta) sigue siendo prácticamente el mismo. Lo que sí cambia mucho es la pregunta: "el pitch que más whiffs saca" no es "el pitch que
más vale".

### Explicación técnica
- Con los costos de Extreme, el perfil óptimo cambia menos que el ruido de muestreo en **21 de 21** combinaciones tipo × conteo; la correlación
  de rangos de los candidatos es ≥ 0.99.
- Con la física aproximada del aire de Extreme, el orden de los candidatos se mueve más (correlación mediana 0.87), pero el perfil óptimo
  sigue dentro del ruido en 17 de 21 combinaciones (el sinker es el que más cambia, por el IVB).
- "Mejor Stuff" y "mejor pitch" casi no se traslapan (top 1% compartido entre 0% y 13% en 0-0): esa diferencia es mucho mayor que la del
  ambiente.

### Pregunta probable
**"¿Entonces todo pitcher debería lanzar ese slider?"** — No. El óptimo es una combinación favorable según los datos; no considera
biomecánica, salud, tunneling, secuencias, repertorio ni fatiga.

---

## 15. Preguntas difíciles del jurado

Formato: **respuesta breve** (para contestar ya) y **respuesta extendida** (si piden más).

**1. ¿Por qué no identificaron los estadios reales?**
- *Breve:* Porque el dataset es anónimo y respetamos esa anonimización; además no podíamos usar información externa.
- *Extendida:* Reconstruimos estadios anónimos (`ANON_STADIUM_xx`) solo con los datos: 98.4% de los juegos asignados, grupos estables (ARI
  mínimo ≈ 0.92 al re-estimar con el 80% de los juegos) y 19/19 grupos que conservan su categoría entre temporadas. No intentamos ponerles nombre.

**2. ¿Por qué Stuff+ v1 no sirve como Stuff puro?**
- *Breve:* Porque ve la ubicación: el 46% de su predicción la explica dónde terminó el pitch.
- *Extendida:* Usa ángulos de llegada sin corregir y la altitud como variable. Su AUC es alto, pero mezcla "pitch bueno" con "pitch bien
  ubicado" y castiga o premia por el ambiente. Lo dejamos como benchmark.

**3. ¿Por qué usar machine learning?**
- *Breve:* Porque el valor de un pitch depende de combinaciones no lineales (velocidad × movimiento × salida × ubicación × conteo).
- *Extendida:* LightGBM capta esas interacciones; la validación cruzada por pitcher evita que el modelo "memorice" pitchers, y comparamos
  siempre contra baselines simples (por ejemplo, el promedio por tipo de pitch).

**4. ¿Qué significa AUC?**
- *Breve:* La probabilidad de que el modelo le dé más puntaje a un swing fallado que a uno con contacto, tomados al azar. 0.5 es azar; 1 es
  perfecto.
- *Extendida:* Stuff+ Physical tiene ≈ 0.65 sobre swings sin ver ubicación; v1 0.69 porque ve la ubicación.

**5. ¿Qué significa estabilidad?**
- *Breve:* La correlación de la métrica del mismo pitcher entre una temporada y la siguiente.
- *Extendida:* Si una métrica se repite, mide algo del pitcher; si no, es ruido o contexto. Stuff+ Physical 0.76; Pitching+ 0.57; whiff/swing
  0.51; RV/BIP −0.02. Solo hay dos pares de temporadas (2024→2025, 2025→2026; 2026 es parcial).

**6. ¿Por qué shrinkage?**
- *Breve:* Para no creerle de más a un pitcher con pocos lanzamientos.
- *Extendida:* El *shrinkage* empírico-bayesiano jala cada promedio hacia la media de la liga en proporción a su ruido: con poca muestra se
  acerca a la media; con mucha, se respeta su valor.

**7. ¿Por qué 100?**
- *Breve:* Es la convención de las métricas "plus": 100 es el promedio y cada 10 puntos una desviación estándar.
- *Extendida:* Permite comparar Stuff+, Location+ y Pitching+ en la misma escala. En el simulador, 100 = lanzamiento promedio de la liga; en el
  ranking, 100 = pitcher promedio.

**8. ¿Qué pasa con pitchers con pocos lanzamientos?**
- *Breve:* Sus métricas se encogen hacia el promedio y sus intervalos son anchos.
- *Extendida:* El ranking exige al menos 300 pitches y 75 turnos; aun así el intervalo del lugar tiene amplitud mediana de 294 posiciones.
  Por eso mostramos P(top 25).

**9. ¿Por qué AEPP no supera a K−BB%?**
- *Breve:* Porque K−BB% ya contiene casi todo lo que se repite; lo que AEPP agrega (daño por contacto) es casi ruido.
- *Extendida:* 0.26 contra 0.25 en carreras en altitud del año siguiente. Lo reportamos tal cual. AEPP sigue siendo útil porque explica de
  dónde vienen las carreras con los precios del ambiente.

**10. ¿Qué aporta la física?**
- *Breve:* Explicar el mecanismo y simular; no mejoró la predicción.
- *Extendida:* Mostró que en Extreme hay menos drag (0.83–0.90) y menos Magnus (0.72–0.84), que los estadios de Extreme no son iguales y por
  qué ciertos pitches pierden movimiento. Sobreestima ≈ 20% el cambio de IVB.

**11. ¿Podemos recomendar pitchers realmente?**
- *Breve:* Podemos señalar perfiles favorables, con su incertidumbre; no garantizar rendimiento.
- *Extendida:* El ranking AEPP es "perfil con características históricamente más favorables en estos contextos". La "resistencia a la altitud"
  individual no se puede medir hoy (confiabilidad 0.11).

**12. ¿Qué significa "27% más"?**
- *Breve:* Que la diferencia en carreras entre un ponche y una bola en juego es 27% mayor en Extreme (0.432 contra 0.340).
- *Extendida:* Lo mismo vale, aproximadamente, para convertir una bola en juego en whiff por swing (+27%).

**13. ¿Es causal?**
- *Breve:* No. Son asociaciones consistentes, con controles fuertes, pero altitud y estadio están confundidos.
- *Extendida:* Efectos fijos de pitcher, bateador, temporada y conteo; 16/16 estadio-temporadas. Pitching+ no necesita el claim causal:
  usa los costos observados de cada resultado.

**14. ¿Por qué Medium y Extreme son parecidos?**
- *Breve:* Porque el contacto se encarece casi igual en ambos (+0.058 y +0.068 por bola en juego frente a No Altitude).
- *Extendida:* El whiff baja menos en Medium (−1.3 pp). Además hay traslape físico: el drag de algunos estadios Medium (0.87–0.95) se acerca
  al de Extreme. Por eso hablamos de "ambientes de altitud".

**15. ¿Qué pitch es mejor en altitud?**
- *Breve:* En promedio, el slider es el que menos valor pierde en Extreme (+0.95 carreras por 100 pitches); no es "el mejor pitch".
- *Extendida:* Es una comparación descriptiva y ajustada por contexto, no el efecto causal de elegir un tipo. La recta pierde más whiff (−5.1
  pp); sinker y changeup pierden más movimiento horizontal.

**16. ¿El slider es realmente mejor?**
- *Breve:* No lo afirmamos. Es el tipo con menor costo promedio observado, no un efecto causal.
- *Extendida:* Quién lanza sliders, cuándo, contra quién y dónde es distinto. Y el análisis del óptimo muestra que la mejor versión de cada
  tipo casi no cambia con el ambiente.

**17. ¿Qué significa RV?**
- *Breve:* Run value: cuántas carreras esperadas suma o resta un resultado, según la situación del juego (RE24).
- *Extendida:* RE24 son las carreras esperadas hasta el final de la entrada según outs y corredores. El RV de una jugada es RE después − RE
  antes + carreras anotadas.

**18. ¿Cómo se convierte una probabilidad en carreras?**
- *Breve:* Probabilidad × valor en carreras, sumado sobre todos los resultados posibles.
- *Extendida:* E[Δ] = Σ P(evento) × Δ(evento, conteo); el valor de cada conteo se obtiene hacia atrás desde los resultados finales del turno
  (recursión por conteo; el foul con dos strikes no cambia el conteo).

**19. ¿Qué pasa con un groundball?**
- *Breve:* Es la segunda línea de defensa: un rodado en lugar de un batazo al aire ahorra 0.140 carreras en No Altitude y 0.202 en Extreme.
- *Extendida:* Pero el % de rodados casi no predice las carreras del año siguiente en altitud, y la primera defensa (evitar contacto) es mucho
  más estable como habilidad.

**20. ¿Qué harían con más datos?**
- *Breve:* Más temporadas, ajuste completo por calidad de rivales y un modelo de calidad de contacto con velocidad y ángulo de salida.
- *Extendida:* También separar estadios por densidad medida en vez de por categoría y corregir la sobrepredicción de IVB del contrafactual.
  Antes habría que corregir la calibración de la velocidad de salida por estadio y temporada (el hard contact% sube de 21.5% en 2024 a 26.2% en 2025).

**21. ¿Es lo mismo un buen lanzamiento en cualquier estadio?**
- *Breve:* Prácticamente sí: el óptimo de cada tipo casi no cambia con los costos de Extreme.
- *Extendida:* Lo que más cambia no es el ambiente sino el objetivo: el pitch con más Stuff+ no es el de más Pitching+.

**22. ¿Por qué Pitching+ predice peor que AEPP las carreras en altitud?**
- *Breve:* Porque Pitching+ mide la calidad esperada de cada lanzamiento; AEPP mide resultados por turno con los precios de altitud.
- *Extendida:* Pitching+ es más estable (0.57 contra 0.42) y sirve para evaluar pitches y escenarios; AEPP sirve para elegir pitchers.

**23. ¿Hay arquetipos de pitcher para altitud?**
- *Breve:* No encontramos grupos claros; hay un continuo entre swing-and-miss y control.
- *Extendida:* La mejor separación tiene silueta 0.18 (pedíamos ≥ 0.25), así que no inventamos categorías.

---

## 16. Lo que NO hay que decir

- "La altitud causa exactamente X." → Di: "en estos ambientes observamos X".
- "Este pitcher es resistente a la altitud." → Di: "tiene un perfil históricamente más favorable en estos contextos".
- "Este es el valor real del pitch en un estadio concreto." → Es una estimación del modelo para un ambiente.
- "Nunca lances al centro." → Es una asociación promedio.
- "El slider es el mejor pitch para altitud." → Es el tipo con menor costo promedio observado.
- "Todo pitcher debe lanzar este pitch óptimo." → El óptimo no considera biomecánica, salud, secuencias ni repertorio.

## 17. Cierre recomendado (30 segundos)

> En altitud, el contacto cuesta más y el swing-and-miss se vuelve más valioso. Pero eso no crea un tipo nuevo de pitcher: el perfil
> favorable sigue siendo el de un buen pitcher, con más premio al ponche. Y un gran lanzamiento sigue siendo un gran lanzamiento en cualquier
> estadio. Pitching+ nos permite ver todo esto pitch por pitch, con física, ubicación, contexto y el costo real de cada resultado.

## 18. Preguntas sobre la validación y las recomendaciones

**24. ¿Cómo saben que el modelo funciona con pitchers o estadios que no vio?**
- *Breve:* Lo reentrenamos dejando fuera pitchers, una temporada y ambientes completos, y lo comparamos con un modelo nulo (valor promedio por conteo).
- *Extendida:* En los cuatro esquemas (pitcher, season, park holdout de Extreme y de Medium) mejora el error cuadrático entre 4.1% y 4.8%, y la pendiente de calibración queda entre 0.96 y 1.00 (1 = perfecta). Sin haber visto Extreme, subestima un poco su valor realizado: el contacto ahí cuesta más. Como el dataset no identifica estadios, el *park holdout* deja fuera todos los estadios de una categoría.

**25. ¿Por qué el RMSE baja tan poco?**
- *Breve:* Porque el valor de un pitch individual es casi todo azar: el mismo pitch puede terminar en strike, foul o jonrón.
- *Extendida:* Lo importante es que el modelo supera al nulo de forma consistente y está calibrado: cuando predice un valor, en promedio eso es lo que pasa.

**26. ¿Qué le recomiendan al pitching coach?**
- *Breve:* Buscar swing-and-miss; desconfiar de los pitches que dependen mucho del giro (mucho ride, barrido o caída), porque son los que más pierden en altitud.
- *Extendida:* Las rectas de mucho ride pierden más Stuff+ y whiff en Extreme que las de poco ride (−7.7 contra −3.8 de Stuff+), y en los changeups de mucho fade el contacto duro sube más (+7.5 pp). Recta + slider es el arsenal con mayor Stuff+ con pitches de Extreme. Todo es descriptivo, no una receta.

**27. ¿Cómo perfilarían a un agente libre?**
- *Breve:* Stuff+ de cada pitch en el simulador, swing-and-miss (whiff/swing, K%), control (BB%, Location+), forma de sus pitches y su AEPP estimado con la calculadora.
- *Extendida:* La calculadora usa la misma fórmula y pesos del ranking; la tabla de referencia da los percentiles de los 599 pitchers. No medimos "resistencia a la altitud" individual porque hoy no es medible.

**28. ¿Revisaron la calidad de la columna de outs?**
- *Breve:* Sí. `OutsOnPlay` casi no registra las dobles matanzas; lo medimos y el sesgo es pequeño frente al efecto de altitud.
- *Extendida:* En 1.7% de los turnos de No Altitude y 2.3% de los de Extreme (en rodados, 3.4% y 4.8%) el siguiente turno empieza con un out más de los registrados. Ese out falta en el run value, un poco más en Extreme. Con la expectativa de carreras por número de outs, eso explicaría +0.002 de las +0.068 carreras más por bola en juego en Extreme: queda dentro del intervalo de confianza y no cambia ninguna conclusión. No lo corregimos porque el dataset no trae corredores en base para saber qué out faltó.
