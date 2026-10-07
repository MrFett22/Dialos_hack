# Validación de la reconstrucción de run_value

## 1. Matriz de 24 estados base-out: LMB (reconstruida) vs referencia MLB

Referencia: tabla RE24 de Tango/Lichtman/Dolphin, *The Book* (MLB 1999-2002, la
tabla mas citada en la comunidad sabermetrica). El entorno ofensivo varia por
año y liga, asi que esto es una comparacion de **orden de magnitud y patron**,
no un benchmark exacto.

| Estado (corredores) | LMB 0 outs | MLB 0 outs | LMB 1 out | MLB 1 out | LMB 2 outs | MLB 2 outs |
|---|---:|---:|---:|---:|---:|---:|
| Vacías        | 0.629 | 0.555 | 0.355 | 0.297 | 0.149 | 0.117 |
| 1B            | 1.070 | 0.953 | 0.688 | 0.573 | 0.298 | 0.251 |
| 2B            | 1.300 | 1.189 | 0.876 | 0.725 | 0.402 | 0.344 |
| 3B            | 1.598 | 1.482 | 0.991 | 0.983 | 0.366 | 0.387 |
| 1B-2B         | 1.778 | 1.573 | 1.143 | 0.971 | 0.501 | 0.466 |
| 1B-3B         | 1.778 | 1.904 | 1.018 | 1.243 | 0.430 | 0.538 |
| 2B-3B         | 1.795 | 2.052 | 1.186 | 1.467 | 0.481 | 0.634 |
| Bases llenas  | 2.353 | 2.417 | 1.433 | 1.650 | 0.622 | 0.815 |

**Lectura:**
- Patrón general correcto: en las 8 combinaciones de corredores, RE decrece
  monótonamente con más outs, y en cada out-count, RE crece monótonamente con
  más/mejores corredores (bases vacías es el mínimo, bases llenas el máximo).
- Escala consistente con MLB: LMB corre ~10-15% más alto que la referencia MLB
  en la mayoría de estados con 0-1 outs (liga más ofensiva), lo cual es
  razonable — no hay valores fuera de rango ni signos invertidos.
- Única anomalía notable: con 0 outs, "1B-3B" (1.778) y "2B-3B" (1.795) salen
  **más bajos** que en MLB relativo a "1B-2B" (1.778) — en MLB un corredor en
  3B añade más valor que uno en 2B; aquí los tres estados con dos corredores
  en 0 outs quedan casi empatados. Es plausible que sea ruido de muestra
  (menos innings que las 4 temporadas de MLB usadas en *The Book*) combinado
  con el sesgo descrito en la sección 2 (los hits no siempre reconcilian la
  ocupación real de bases contra RunsScored, lo que contamina ligeramente la
  clasificación de estados "antes" usada para construir esta tabla).

## 2. Ejemplo real: media entrada reconstruida pitch a pitch

`game_000001_2_Bottom` — 3 plate appearances, incluye un K, dos sencillos y un K final:

| PitchUID | bateador | count | PitchCall | ¿fin de PA? | estado antes (1B,2B,3B,outs) | resultado | carreras | estado después | run_value |
|---|---|---|---|---|---|---|---|---|---:|
| bae37820 | batter_00258 | 0-0 | StrikeCalled | No | (F,F,F,0) | | | (F,F,F,0) | 0.000 |
| c58b2160 | batter_00258 | 0-1 | HitByPitch | No | (F,F,F,0) | | | (F,F,F,0) | 0.000 |
| ecd28510 | batter_00202 | 0-0 | StrikeCalled | No | (F,F,F,0) | | | (F,F,F,0) | 0.000 |
| fe3422a0 | batter_00202 | 0-1 | StrikeSwinging | No | (F,F,F,0) | | | (F,F,F,0) | 0.000 |
| 14551d00 | batter_00202 | 0-2 | BallCalled | No | (F,F,F,0) | | | (F,F,F,0) | 0.000 |
| 2884a7a0 | batter_00202 | 1-2 | StrikeSwinging | **Sí** | (F,F,F,0) | Strikeout | 0 | (F,F,F,1) | **-0.275** |
| 4b4e4b60 | batter_00317 | 0-0 | StrikeCalled | No | (F,F,F,1) | | | (F,F,F,1) | 0.000 |
| 5d90d310 | batter_00317 | 0-1 | BallCalled | No | (F,F,F,1) | | | (F,F,F,1) | 0.000 |
| 7062f090 | batter_00317 | 1-1 | StrikeSwinging | No | (F,F,F,1) | | | (F,F,F,1) | 0.000 |
| 9b960d60 | batter_00317 | 1-2 | InPlay | **Sí** | (F,F,F,1) | Single | 0 | (T,F,F,1) | **+0.333** |
| c2248150 | batter_00385 | 0-0 | BallCalled | No | (T,F,F,1) | | | (T,F,F,1) | 0.000 |
| d1c5d140 | batter_00385 | 1-0 | StrikeSwinging | No | (T,F,F,1) | | | (T,F,F,1) | 0.000 |
| e18b4b00 | batter_00385 | 1-1 | StrikeSwinging | No | (T,F,F,1) | | | (T,F,F,1) | 0.000 |
| f2a5a660 | batter_00385 | 1-2 | FoulBallNotFieldable | No | (T,F,F,1) | | | (T,F,F,1) | 0.000 |
| 08b0f5e0 | batter_00385 | 1-2 | InPlay | **Sí** | (T,F,F,1) | Single | **1** | (T,T,F,2) | **+0.813** |
| 2d875990 | batter_00189 | 0-0 | BallCalled | No | (T,T,F,2) | | | (T,T,F,2) | 0.000 |
| 557a3f80 | batter_00189 | 1-0 | FoulBallNotFieldable | No | (T,T,F,2) | | | (T,T,F,2) | 0.000 |
| 6d66f930 | batter_00189 | 1-1 | BallCalled | No | (T,T,F,2) | | | (T,T,F,2) | 0.000 |
| 854c38d0 | batter_00189 | 2-1 | FoulBallNotFieldable | No | (T,T,F,2) | | | (T,T,F,2) | 0.000 |
| b05deaf0 | batter_00189 | 2-2 | FoulBallNotFieldable | No | (T,T,F,2) | | | (T,T,F,2) | 0.000 |
| ccc94ef0 | batter_00189 | 2-2 | FoulBallNotFieldable | No | (T,T,F,2) | | | (T,T,F,2) | 0.000 |
| e8a7dc40 | batter_00189 | 2-2 | StrikeSwinging | **Sí** | (T,T,F,2) | Strikeout | 0 | (F,F,F,3) | **-0.501** |

**Lo que funciona bien:** el encadenamiento entre PAs es correcto — el corredor
que llega a 1B con el sencillo de `batter_00317` sigue en el estado "antes" de
`batter_00385`, y los pitches que no terminan la PA correctamente mantienen
run_value=0 (el estado base-out solo cambia en el último pitch de cada PA).

**Lo que NO funciona del todo bien — hallazgo de esta verificación:** en la PA
de `batter_00385` (fila `08b0f5e0`), el sencillo con corredor en 1B produjo
**1 carrera real** (`RunsScored=1`), pero la heurística de avance-por-N-bases
solo mueve al corredor de 1B a 2B (no anota), y el estado "después" queda como
`(1B,2B ocupadas, 2 outs)` en vez de algo como `(solo 1B ocupada, 2 outs)`. El
`run_value` de +0.813 sí incorpora la carrera real (viene de RunsScored), pero
la ocupación de bases usada para el lookup de RE "después" es incorrecta —
sobrestima cuántos corredores quedaron en base. Esto pasó porque, a diferencia
de la rama de outs (`_generic_out_hold`, que sí reconcilia contra RunsScored),
la rama de hits (`_advance_on_hit`) avanza siempre exactamente N bases sin
verificar contra RunsScored real.

**Que tan seguido pasa esto:** de las 41,579 PA que terminaron en hit
(single/double/triple/HR), en **7,873 (18.94%)** las carreras implicadas por el
avance simple de N bases no coinciden con el RunsScored real — es decir, el
sencillo/doble/triple/HR movió corredores de forma distinta a la asumida. Eso
representa **4.86% del total de PA** del dataset.

## 3. Plate appearances con jugadas ambiguas para la heurística de bases

| Categoría | n PA | % del total de PA (162,035) |
|---|---:|---:|
| Out normal (bola en juego, no hit, sin ambigüedad relevante) | 66,781 | 41.21% |
| Fielder's Choice | 3,632 | 2.24% |
| Sacrifice | 1,926 | 1.19% |
| Error | 1,742 | 1.08% |
| **Subtotal ambiguo (FC + Sacrifice + Error)** | **7,300** | **4.51%** |
| Hits con RunsScored inconsistente con avance de N bases (hallazgo sección 2) | 7,873 | 4.86% |
| **Total de PA con incertidumbre real en la ocupación de bases** | **~15,173** | **~9.36%** |

Nota: "Out normal" también tiene una limitación menor no capturada arriba (un
corredor puede avanzar por tag-up en un fly out sin que se anote carrera, y la
heurística actual no lo modela — solo fuerza avance cuando hay carrera real),
pero no se puede cuantificar sin datos de trayectoria de la jugada.

## Conclusión (pre-fix)

El patrón de la matriz de 24 estados es correcto y está en la escala esperada
frente a MLB — la mecánica de RE24 (decrece con outs, crece con corredores)
funciona bien. El punto débil real es la reconstrucción de **ocupación de
bases en jugadas de hit**, que afecta a ~9.4% de las plate appearances. Runs y
outs (que vienen de RunsScored/OutsOnPlay reales) siempre son correctos; lo que
puede estar mal es a qué base exacta llegó cada corredor en esas PA, lo cual
sesga ligeramente el run_value asignado a esas jugadas específicas y a las
PA inmediatamente siguientes en la misma media entrada (por el encadenamiento
de estados).

---

## Actualización: fix de reconciliación en `_advance_on_hit`

Se corrigió `_advance_on_hit` para que reconcilie el avance ingenuo (N bases
para todos) contra `RunsScored` real, igual que ya hacía `_generic_out_hold`
para los outs:

- Si el avance ingenuo anota **menos** carreras que `RunsScored`, se hace
  anotar a los corredores que quedaron en base empezando por el **más
  adelantado** (3B, luego 2B, luego 1B) — no sabemos con certeza cuál corredor
  específico anotó, así que asumimos el que tenía menos distancia por recorrer.
- Si el avance ingenuo anota **más** carreras que `RunsScored`, se "deshace" la
  anotación del/los corredor(es) que partieron de la base **menos adelantada**
  (regla simétrica, igual de arbitraria en la identidad exacta) y se les
  coloca en la base libre más cercana a home.
- **Caso degenerado documentado:** en escenarios extremos (ej. bases llenas +
  hit donde ni el mínimo de carreras forzadas por el avance ingenuo ni el
  máximo de corredores disponibles alcanzan a explicar `RunsScored`, típicamente
  por un error de fildeo adicional no capturado por los flags single/double/etc.)
  no siempre hay una base libre para reconciliar por completo. Ocurre en
  **404 de 41,579 PA con hit (0.97%)** — se deja la ocupación de bases lo más
  cercana posible sin inventar una base inexistente; las carreras/outs
  (`RunsScored`/`OutsOnPlay`) siguen siendo siempre correctos.

### 1. Matriz de 24 estados, recalculada — ¿se resolvió la anomalía 1B-3B/2B-3B?

**Sí, se resolvió.**

| Estado (corredores) | LMB antes | LMB después | MLB (ref.) |
|---|---:|---:|---:|
| 1B-2B (0 outs) | 1.778 | **1.815** | 1.573 |
| 1B-3B (0 outs) | 1.778 (empatado) | **1.931** | 1.904 |
| 2B-3B (0 outs) | 1.795 (casi empatado) | **2.138** | 2.052 |

Antes del fix, los tres estados con dos corredores en 0 outs quedaban casi
empatados (1.778 / 1.778 / 1.795), lo cual no tenía sentido — un corredor en
3B debería valer más que uno en 2B. Después del fix, el orden correcto se
restablece claramente: **2B-3B (2.138) > 1B-3B (1.931) > 1B-2B (1.815)**,
igual que en la referencia MLB (2.052 > 1.904 > 1.573). La matriz completa
post-fix (todas las 32 combinaciones, incluyendo outs=3) queda arriba en la
sección "1. Matriz de 24 estados" de `reports/targets_summary.md`.

### 2. % de PA con hit con inconsistencia después del fix

Bajó de **18.94-19.16%** (pre-fix) a **0.97%** (404 de 41,579 PA con hit) —
y ese 0.97% restante son los casos degenerados documentados arriba (no un
bug: son plays donde el propio dato de single/double/etc. no alcanza a
explicar `RunsScored`, probablemente por un evento adicional como un error de
fildeo). En términos del total de PA del dataset, eso es **0.25%** (404 de
162,035), bajando desde el 4.86% original.

### 3. run_value y xRunValue regenerados

`data/processed/pitches_with_targets.parquet` y `reports/targets_summary.md`
ya están regenerados con la matriz y el run_value corregidos (pipeline
completo re-ejecutado).

### 4. Cambio en run_value promedio por altitude_category

| altitude_category | mean ANTES | mean DESPUÉS | Δ | std antes | std después |
|---|---:|---:|---:|---:|---:|
| Extreme Altitude | 0.0480 | 0.0473 | -0.0007 | 0.6012 | 0.5708 |
| Medium Altitude   | 0.0328 | 0.0323 | -0.0005 | 0.5875 | 0.5608 |
| No Altitude       | -0.0154 | -0.0159 | -0.0005 | 0.5358 | 0.5145 |

**El hallazgo se mantiene y prácticamente no cambió.** El ranking
Extreme > Medium > No Altitude es idéntico antes y después, y la magnitud del
promedio se movió menos de 0.001 carreras en los tres grupos — un cambio
irrelevante frente al tamaño de la diferencia entre grupos (~0.06 carreras
entre Extreme y No Altitude). El bug de reconciliación **no era la causa** del
patrón de altitud: introducía ruido dentro de cada media entrada (a qué base
exacta llega cada corredor), pero ese ruido se promedia de forma parecida en
los tres grupos de altitud y no sesgaba la comparación entre ellos. Lo que sí
cambió notablemente fue la **precisión interna de la matriz de 24 estados**
(sección 1), que es lo que hacía falta corregir para poder confiar en
comparaciones más finas (por ejemplo, run_value por tipo de pitch o por
pitcher individual, donde el ruido de reconciliación pudo pesar más en
plate appearances específicas).

Nota aparte: `is_hard_contact` (barrel) por altitude_category **no cambió en
absoluto** con este fix (Extreme=0.0050, Medium=0.0052, No Altitude=0.0043,
igual que antes) — es un target que depende solo de ExitSpeed/Angle, no de la
reconstrucción de bases, así que nunca estuvo afectado por este bug. De hecho
esa métrica en particular no muestra un patrón claro de "más hard contact en
Extreme Altitude" (Medium Altitude es ligeramente más alto que Extreme); el
hallazgo de altitud que sí es consistente y robusto es el de `run_value`
promedio, no el de `is_hard_contact`.

---

## Descomposición: ¿de dónde viene el efecto de altitud en run_value?

Hipótesis a probar: ¿el efecto viene más de "menos whiffs / menos comando
efectivo" (menos capacidad del pitcher de generar swing-and-miss o de robarse
strikes) que de "más contacto duro"?

### Comparación de targets por altitud (Extreme vs No Altitude), con prueba de
### dos proporciones (z-test, n>180k por grupo)

| target | Extreme | Medium | No Altitude | Δ (Ext-NoAlt) | % relativo | z |
|---|---:|---:|---:|---:|---:|---:|
| **is_out_in_pa** | 0.6072 | 0.6220 | 0.6601 | **-0.0529** | **-8.0%** | **-36.91** |
| is_whiff | 0.1080 | 0.1124 | 0.1170 | -0.0090 | -7.7% | -9.52 |
| is_chase | 0.2171 | 0.2197 | 0.2268 | -0.0097 | -4.3% | -7.83 |
| is_called_strike | 0.1507 | 0.1501 | 0.1583 | -0.0076 | -4.8% | -7.01 |
| is_groundball | 0.1093 | 0.1012 | 0.0948 | +0.0145 | +15.2% | +16.13 |
| is_weak_contact | 0.0811 | 0.0763 | 0.0792 | +0.0019 | +2.4% | +2.34 |
| is_hard_contact | 0.0050 | 0.0052 | 0.0043 | +0.0008 | +18.1% | +3.81 |

**Lectura:** `is_whiff`, `is_chase` e `is_called_strike` bajan de forma
consistente y monótona (Extreme < Medium < No Altitude) en los tres — todos
son indicadores de "el pitcher engaña/domina menos al bateador" (menos swing-
and-miss, menos persuadir al bateador de perseguir fuera de zona, menos
robarse strikes cantados). `is_out_in_pa` baja en la misma dirección y con,
por mucho, el z-score más grande de toda la tabla (-36.91): es el efecto
dominante. En cambio `is_hard_contact` sube solo 0.08 puntos porcentuales en
términos absolutos — estadísticamente "significativo" solo porque el n es
enorme (>180k por grupo), pero irrelevante en magnitud frente a los otros
efectos.

### Descomposición algebraica del run_value promedio

Separando el run_value promedio por PA en `P(out) × RV|out + P(no_out) × RV|no_out`:

| altitude_category | P(out) | RV \| out | RV \| no_out (embasado) |
|---|---:|---:|---:|
| Extreme Altitude | 0.6171 | -0.3131 | +0.6281 |
| Medium Altitude  | 0.6294 | -0.3132 | +0.6190 |
| No Altitude      | 0.6737 | -0.3047 | +0.5804 |

`P(out)` cae **5.66 puntos porcentuales** de No Altitude a Extreme Altitude —
coherente con el -8.0% relativo de `is_out_in_pa` de arriba. Usando una
descomposición tipo Oaxaca-Blinder (promediando ambos órdenes de base) sobre
la brecha total de run_value (0.0473 − (−0.0159) = 0.0632 carreras):

| Componente | Contribución | % de la brecha total |
|---|---:|---:|
| **Diferencia en tasa de out (P(out))** | **+0.0517** | **~82%** |
| Diferencia en valor condicional (RV\|out y RV\|no_out) | +0.0115 | ~18% |

### Conclusión sobre la hipótesis

**Se confirma.** ~82% de la ventaja de run_value en Extreme Altitude viene de
que el bateador hace out con menos frecuencia (menos whiffs, menos chases,
menos called strikes → menos outs por PA), no de que la bola salga golpeada
más fuerte cuando hay contacto. El 18% restante ("valor condicional a
embasarse/hacer out") es más ambiguo: incluye tanto un posible efecto de
contacto de calidad (hits con más bases, más carreras impulsadas) como el
efecto mecánico de que, con más corredores en base en general en un entorno
más ofensivo, tanto los hits como los outs "valen más" en términos de RE —
esto no se puede separar limpiamente del contacto duro con las columnas
disponibles, pero en cualquier caso es un componente secundario. `is_hard_contact`
(la métrica más directa de "contacto duro" vía Statcast barrel) prácticamente
no varía por altitud, así que no sostiene la narrativa de "más contacto duro";
la narrativa de "menos comando/menos swing-and-miss" sí está fuertemente
sostenida por los datos.
