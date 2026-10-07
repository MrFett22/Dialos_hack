# Notas metodológicas — Stuff+ LMB

Notas de diseño que no pertenecen a un reporte de resultados específico, para
no repetir errores de diseño en fases futuras del proyecto.

## 1. El efecto de altitud sobre is_whiff está mediado por el movimiento — el contrafactual de Semana 5 debe encadenar Magnus antes de Stuff+

**Contexto:** el ablation de `reports/stuff_plus_baseline.md` (sección "Aporte
de features de altitud/física") entrenó 3 versiones del modelo de `is_whiff`
sobre el mismo split `pitcher_holdout`:

| versión | AUC test |
|---|---:|
| Completa (con altitude_level + magnus_residual) | 0.7297 |
| Sin altitud ni magnus | 0.7306 |
| Solo altitud (sin magnus) | 0.7306 |

Las tres son estadísticamente indistinguibles (diferencias de ~0.0009 AUC,
ruido). Quitar `altitude_level` del modelo **no le cuesta nada** una vez que
`InducedVertBreak`/`HorzBreak` (el movimiento observado) ya están como
features.

**Conclusión metodológica:** el efecto de la altitud sobre `is_whiff` está
**completamente mediado por el movimiento físico observado**. No hay un
efecto directo adicional de "lanzar en Extreme Altitude" sobre la
probabilidad de whiff más allá de lo que ya explica el movimiento del pitch
en sí. En términos de un diagrama causal:

```
altitude_category -> densidad de aire -> movimiento (IVB/HB) -> is_whiff
                                              ^
                              (esta es la UNICA via — no hay flecha directa
                               altitude_category -> is_whiff)
```

### Por qué esto rompe el diseño ingenuo de un contrafactual de altitud

Un contrafactual "ingenuo" para responder "¿cómo cambiaría el whiff rate de
este pitch si se hubiera lanzado en otra altitud?" sería: tomar el pitch con
su `InducedVertBreak`/`HorzBreak` **observados tal cual**, cambiar solo la
etiqueta de `altitude_category` (o `altitude_level`), y volver a predecir con
el modelo de Stuff+.

**Ese diseño está estructuralmente condenado a no mostrar ningún efecto**,
precisamente por el hallazgo de arriba: si el modelo ya no usa
`altitude_level` como señal relevante una vez que ve el movimiento, cambiar
esa columna sola (dejando el movimiento fijo) va a mover la predicción
~0, sin importar qué tan real sea el efecto de la altitud en el mundo real.
No es que el contrafactual "demuestre" que la altitud no importa — es que el
diseño mantiene fijo precisamente el canal (movimiento) por el que la altitud
sí actúa.

### Diseño correcto: encadenar Magnus -> Stuff+, no fijar el movimiento

El contrafactual de altitud de Semana 5 **debe hacerse en dos pasos**:

1. **Paso físico (Magnus):** usando `src/features/physics.py`
   (`add_theoretical_break`, `add_air_density`), recalcular el movimiento
   teórico del pitch bajo la densidad de aire de la altitud contrafactual,
   manteniendo fijos `SpinRate`, `SpinAxis`, `RelSpeed` y `Extension` (el
   "diseño" del pitch — lo que el pitcher intenta hacer, que no depende de la
   altitud). Esto da `ivb_theoretical_contrafactual` / `hb_theoretical_contrafactual`.

2. **Ajustar el movimiento observado, no reemplazarlo por el teórico puro:**
   el modelo de Magnus sobreestima sistemáticamente la magnitud absoluta
   (spin util vs. giroscópico, ver `reports/magnus_analysis.md`), así que no
   hay que usar el movimiento teórico directamente. Hay que preservar el
   `magnus_residual_vert`/`magnus_residual_horz` del pitch (el componente de
   movimiento que es habilidad/agarre del pitcher, no explicado por la
   física simple) y sumarlo al nuevo teórico:

   ```
   IVB_contrafactual = magnus_residual_vert_observado + ivb_theoretical_contrafactual
   HB_contrafactual  = magnus_residual_horz_observado + hb_theoretical_contrafactual
   ```

   Esto es exactamente la misma descomposición ya usada en
   `src/features/physics.py` (`magnus_residual = observado - teórico`),
   despejada para "observado" bajo una densidad de aire distinta.

3. **Paso predictivo (Stuff+):** alimentar `IVB_contrafactual`/`HB_contrafactual`
   (junto con el resto de features sin cambiar) al modelo de `is_whiff` de
   `src/models/stuff_plus.py` para obtener la probabilidad de whiff
   contrafactual.

Recién ahí el contrafactual mide lo que se quiere medir: "si este pitcher
hubiera lanzado el mismo pitch (mismo spin/velocidad de diseño) en otra
altitud, ¿cómo habría cambiado su movimiento, y qué tanto habría cambiado su
probabilidad de whiff por ese cambio de movimiento?" — en vez de la pregunta
mal planteada "¿cambia el whiff si le miento al modelo sobre la altitud pero
le dejo el movimiento intacto?".

**Acción para Semana 5:** implementar el contrafactual como pipeline de 2
etapas (Magnus -> Stuff+), no como una sustitución directa de
`altitude_category` con el resto de las features fijas.
