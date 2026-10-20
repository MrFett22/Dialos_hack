# Investigación: pitcher_anon_id nulo (3,008 filas en el dataset limpio)

## 1. ¿Hay un patrón? Sí — no es aleatorio

**Concentración por year:** 95.4% de los nulos (3,122 de 3,272 en el dato crudo)
son de **year=2026**. En 2024 y 2025 el nulo es prácticamente residual.

| year | % del total de pitches | % de los nulos |
|---|---:|---:|
| 2024 | 38.3% | 4.0% |
| 2025 | 40.3% | 0.6% |
| 2026 | 21.3% | **95.4%** |

**Concentración por altitude_category:** sobre-representado en Extreme Altitude
y en el bucket sin altitud asignada:

| altitude_category | % del total | % de los nulos | sobre-representación |
|---|---:|---:|---:|
| Extreme Altitude | 28.9% | **52.7%** | 1.8x |
| altitude_category NaN | 0.5% | **9.7%** | ~20x |
| No Altitude | 46.3% | 37.2% | 0.8x |
| Medium Altitude | 24.4% | 0.5% | 0.02x |

**AutoPitchType:** sin sesgo relevante — la mezcla de tipos de pitch entre las
filas nulas es prácticamente igual a la del dataset completo.

**Concentración por juego:** los nulos vienen de solo **34 de 2,127 juegos**
(1.6%). No están dispersos aleatoriamente entre juegos.

**Dentro de esos 34 juegos, los nulos NO están dispersos pitch a pitch — vienen
en bloques cronológicos contiguos grandes**, y ese patrón coincide con medias
entradas completas de un pitcher específico. Ejemplo real, `game_000486`
(85.1% de sus 282 pitches nulos):

- **Todas** las medias entradas "Bottom" del juego (9 de 9 innings) tienen
  `pitcher_anon_id` nulo de principio a fin — el pitcher del equipo local nunca
  quedó mapeado en ningún momento del juego.
- Las medias entradas "Top" también son nulas en los innings 1-5 (el
  abridor visitante tampoco está mapeado), pero **sí** aparecen mapeados 3
  relevistas distintos (`pitcher_00450`, `pitcher_00850`, `pitcher_01110`) en
  los innings 6, 7 y 8 — es decir, esos relevistas específicos SÍ tienen ID
  asignado (probablemente porque también lanzaron en otros juegos ya
  procesados), pero el abridor de este juego en particular no.

**Conclusión:** el nulo está ligado a la **identidad de pitchers específicos**
(probablemente abridores/pitchers nuevos de la cola de la temporada 2026 que
aún no pasaron por el proceso de anonimización del organizador al momento de
exportar el dataset), no a un problema de tracking aleatorio pitch por pitch.
Es un patrón de **missingness no aleatorio (MNAR)**: un pitcher entero (todos
sus pitches, en todos los juegos donde apareció) queda invisible.

## 2. ¿Rompió la reconstrucción de run_value? No

La simulación de estados base-out y plate appearances en `src/features/targets.py`
(`_assign_plate_appearances`, `_resolve_plate_appearance`) agrupa y ordena
exclusivamente por `game_anon_id`, `Inning`, `Top/Bottom` y el timestamp
embebido en `PitchUID` — y decide transiciones de estado con `strikeout`,
`base_on_balls`, `is_hit_by_pitch`, `is_ball_in_play`, `RunsScored`,
`OutsOnPlay`. **Ninguna de esas columnas depende de `pitcher_anon_id` ni de
`batter_anon_id`.**

Verificación directa sobre `game_000486` (el juego con más nulos, 85.1%) en
`data/processed/pitches_with_targets.parquet`:

- `run_value` nulo: **0** filas de 282
- `is_out_in_pa` nulo: **0** filas de 282
- La secuencia cronológica de innings/medias-entradas es correcta de principio
  a fin, incluyendo un Home Run con `RunsScored=2` correctamente reflejado en
  `run_value=1.658` en la fila correspondiente.

Cada fila se trata independientemente para el encadenamiento cronológico; el
pitcher nulo no rompe nada en `targets.py`.

## 3. Decisión: excluir estas filas de TODO el pipeline de entrenamiento

**Confirmado: Location+ no depende de `pitcher_anon_id` como feature
obligatoria.** Sus features/target (`PlateLocHeight`, `PlateLocSide`,
`in_strike_zone`, count, zona) no requieren identidad del pitcher — así que,
en aislamiento, técnicamente sí se podrían conservar estas 3,008 filas para
entrenar Location+.

**Aun así, recomiendo excluirlas de los tres sub-modelos (Stuff+, Pitching+ y
Location+), no solo de los que usan arsenal/holdout por pitcher**, por tres
razones:

1. **Consistencia del split y de la comparación entre sub-modelos.** Si
   `pitcher_holdout` se usa como esquema de evaluación compartido (para poder
   reportar Stuff+/Location+/Pitching+ del mismo pitcher sobre el mismo
   conjunto de pitches), los tres modelos necesitan operar sobre el mismo
   universo de filas. Meter estas 3,008 filas solo en Location+ rompe esa
   comparabilidad.
2. **El missingness no es aleatorio, es un sesgo real de composición.** Estas
   filas sobre-representan Extreme Altitude (52.7% vs 28.9% del total) y están
   concentradas casi enteramente en el tramo final de la temporada 2026. Un
   Location+ entrenado con esas filas de más estaría viendo una mezcla de
   datos distinta (más pesada hacia altitud extrema y hacia 2026) que
   Stuff+/Pitching+ — un sesgo de selección real, no ruido inofensivo.
3. **Simplicidad operativa.** Mantener un universo de filas distinto por
   sub-modelo es una fuente directa de bugs (joins que no cuadran entre
   scores de distintos sub-modelos, reportes que no suman el mismo total de
   pitches).

**Decisión aplicada:** excluir las filas con `pitcher_anon_id` nulo (3,008,
0.47% del dataset limpio) de los tres sub-modelos por igual, en el paso de
train/test split (no en `src/data/clean.py`, que las deja intactas y solo las
reporta — ver punto 4). Documentado aquí para que quede explícito y auditable.

**Nota aparte:** `batter_anon_id` (2,013 nulos, 0.32%) y `catcher_anon_id`
(2,317 nulos, 0.37%) tienen un patrón de nulos del mismo orden de magnitud,
probablemente por la misma causa raíz (mapeo de anonimización incompleto para
jugadores nuevos de 2026). No se investigaron a fondo aquí porque el pedido
era sobre `pitcher_anon_id`, pero cualquier feature futura que agrupe por
`batter_anon_id` o `catcher_anon_id` debería revisar esto antes de asumir
cobertura completa.

## 4. `src/data/clean.py` actualizado

Se agregó `validate_grouping_id_columns()` (sección "6. Nulos en columnas de
ID de agrupación" del reporte), que ahora reporta explícitamente los nulos en
`game_anon_id`, `pitcher_anon_id`, `batter_anon_id` y `catcher_anon_id` en
cada corrida — sin eliminar filas ahí (la decisión de exclusión es específica
de cada sub-modelo y se documenta en este archivo, no en el paso de limpieza
genérico). Esto evita que un nulo de este tipo pase desapercibido de nuevo en
otras columnas de ID.

**Actualización:** esta función se renombró a `validate_key_dependency_columns()`
y ahora también incluye `altitude_category` (ver sección siguiente) — quedó
fuera del chequeo original y se descubrió recién al construir
`src/validation/splits.py`.

---

# Investigación adicional: altitude_category nulo (3,019 filas)

Al construir `altitude_holdout` en `src/validation/splits.py` aparecieron
2,703 filas con `altitude_category` nulo que no estaban cubiertas por la
exclusión de `pitcher_anon_id` nulo. Esto se investigó con la misma
metodología que la sección 1.

## 1. ¿Es la misma causa raíz que pitcher_anon_id nulo? No — es un problema distinto

De las 3,019 filas con `altitude_category` nulo:

- **316 (10.5%)** también tienen `pitcher_anon_id` nulo — overlap real, pero chico.
- **2,703 (89.5%)** tienen `pitcher_anon_id` VÁLIDO — es decir, la gran mayoría
  del problema de `altitude_category` nulo es un problema **separado**, no la
  misma fila afectada por dos síntomas del mismo bug.

## 2. Patrón de las 2,703 filas con pitcher válido

**Concentración por juego:** vienen de solo **10 de 2,126 juegos** — y en
**los 10, el 100% de los pitches del juego tienen `altitude_category` nulo**
(no hay ningún bloque parcial dentro de un juego, a diferencia del patrón de
`pitcher_anon_id` nulo que sí tenía mezcla parcial dentro de un mismo juego).
Verificado directamente en `game_001996` y `game_001638`: un solo bloque
contiguo cubriendo el 100% de sus 397 y 375 pitches respectivamente.

**Solapamiento de juegos con el problema de pitcher_anon_id:** solo **4 de
estos 10 juegos** coinciden con los 34(33) juegos del problema de
`pitcher_anon_id` — la mayoría (6 de 10) son juegos donde `pitcher_anon_id`
está perfectamente poblado pero `altitude_category` no.

**Concentración por año:** distinta a la de `pitcher_anon_id` nulo (que era
95.4% de 2026). Aquí es **65.0% de 2026 y 35.0% de 2025**, sin nada de 2024.

**Conclusión:** es un problema de **mapeo estadio → altitude_category**
faltante para un lote específico de 10 juegos (probablemente porque esos
juegos se jugaron en una sede que aún no estaba en la tabla de buckets de
altitud del organizador al momento del export), distinto del mapeo de
pitcher_anon_id (identidad de jugador). Ambos son fallas de "tabla de lookup
incompleta para datos recientes", pero en tablas de lookup DIFERENTES —
consistente con que solo 4 de 10 juegos se solapen entre ambos problemas.

## 3. Decisión: excluir también de los tres sub-modelos, por consistencia

Igual que con `pitcher_anon_id` nulo, y por el mismo razonamiento: Stuff+
necesita `altitude_category` como feature obligatoria (paso 3.2), así que
aunque `pitcher_holdout` y `season_holdout` no la necesiten para construirse,
mantener un universo de filas distinto por sub-modelo generaría el mismo
problema de comparabilidad y sesgo de composición ya documentado arriba.

**Decisión aplicada:** `src/validation/splits.py` ahora excluye la UNIÓN de
`pitcher_anon_id` nulo y `altitude_category` nulo (5,711 filas, 0.90% del
dataset — 3,008 + 3,019 − 316 de overlap) de los tres esquemas de split por
igual. Antes de este cambio, `pitcher_holdout` y `season_holdout` sí incluían
las 2,703 filas de `altitude_category` nulo; ahora quedan fuera de los tres,
igual que ya estaba `altitude_holdout` desde el principio (que sí dependía
directamente de la columna).

## 4. `src/data/clean.py` actualizado (otra vez)

`validate_key_dependency_columns()` (antes `validate_grouping_id_columns()`)
ahora también reporta nulos de `altitude_category` junto con las columnas de
ID, en la misma sección del reporte de limpieza — para que la próxima columna
de la que dependa un sub-modelo río abajo no se descubra tarde otra vez.
