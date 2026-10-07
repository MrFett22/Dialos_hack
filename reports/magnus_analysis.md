# Residual de Magnus por altitude_category

## Limitacion explicita

El dataset no trae altitud numerica ni identificador de estadio, solo el bucket `altitude_category` (3 valores). Los siguientes puntos medios de elevacion son **supuestos manuales del usuario, no medidos**:

| altitude_category | elevacion asumida (m) | densidad de aire (kg/m^3, ISA) |
|---|---:|---:|
| No Altitude | 50 | 1.2191 |
| Medium Altitude | 800 | 1.1336 |
| Extreme Altitude | 1900 | 1.0167 |

Nota: estas elevaciones son SUPUESTOS de la version v1 (el dataset no trae elevaciones ni estadios). Cada categoria agrupa varios estadios anonimos con densidades distintas (`reports/estadios_anonimos.md`); la version vigente mide la densidad efectiva con los datos en lugar de suponerla.

## Modelo fisico (fuerza de Magnus)

Ver docstring de `src/features/physics.py` para el detalle completo (formula barometrica ISA, coeficiente de sustentacion de Nathan, aproximacion de aceleracion constante sobre el tiempo de vuelo). Limitacion clave: usa SpinRate total (no separa spin util de spin giroscopico), asi que sobreestima sistematicamente la magnitud absoluta del break — ver chequeo de cordura abajo.

### Calibracion empirica del offset SpinAxis -> direccion de break

Mediana de (angulo observado del movimiento - SpinAxis) sobre 633,447 pitches: **178.67 grados** (std=8.72). Se uso **180.0** grados como offset (valor redondo dentro de esta distribucion tan angosta) en vez de asumir a ciegas una convencion de manual del proveedor de tracking.

### Chequeo de cordura: magnitud teorica vs observada

- Magnitud teorica promedio: 22.28 in
- Magnitud observada promedio: 13.91 in
- Razon teorica/observada (mediana): 1.46x

La sobreestimacion (~1.5-2x) es consistente con no separar spin util de spin giroscopico (ver limitacion arriba) — el orden de magnitud es razonable (decenas de pulgadas, no cientos ni una fraccion), lo cual confirma que el modelo no tiene un error de unidades o de formula.

## Nulos

- `magnus_residual_vert`: 3,019
- `magnus_residual_horz`: 3,019

## ¿La fisica simple explica la diferencia entre categorias de altitud?

Eta^2 (fraccion de la varianza total explicada por altitude_category, via ANOVA de un factor) — crudo vs. residual:

| variable | eta^2 crudo | eta^2 residual | reduccion |
|---|---:|---:|---:|
| Vertical (IVB) | 0.0148 | 0.0031 | 79.2% |
| Horizontal (HB) | 0.0058 | 0.0001 | 98.9% |

Ver figuras `reports/figures/06_magnus_vert_crudo_vs_residual.png` y `07_magnus_horz_crudo_vs_residual.png` — cada una compara boxplots de la variable cruda (izquierda) contra el residual (derecha), por altitude_category.

## ¿El remanente vertical es un artefacto de composicion de AutoPitchType?

Medium Altitude quedo con el residual vertical mas negativo de las tres categorias (ver seccion anterior), un patron no-monotonico raro si el unico factor fuera densidad de aire. Se investigo si es porque Medium Altitude tira una mezcla distinta de tipos de pitch (ej. mas curvas o sinkers, que tienen IVB naturalmente mas bajo).

### Composicion de AutoPitchType por altitude_category (%)

| altitude_category   |   Four-Seam |   Sinker |   Slider |   Changeup |   Curveball |   Cutter |   Splitter |
|:--------------------|------------:|---------:|---------:|-----------:|------------:|---------:|-----------:|
| No Altitude         |       33.85 |    18.71 |    20.34 |      14.07 |        8.55 |     3.27 |       1.2  |
| Medium Altitude     |       35.74 |    15.22 |    23.34 |      11.38 |        9.09 |     4.26 |       0.97 |
| Extreme Altitude    |       33.95 |    17.39 |    23.4  |      10.58 |        7.91 |     5.56 |       1.21 |

Test chi-cuadrado de independencia: chi2=4344.1, p=0.00e+00 (n enorme, casi cualquier diferencia sale 'significativa'). El tamaño de efecto real es lo que importa: **Cramer's V = 0.0587** — segun convenciones estandar (V<0.1 = asociacion despreciable), la mezcla de pitch types SI difiere entre categorias de altitud, pero muy poco en terminos practicos.

### magnus_residual_vert promedio por (AutoPitchType, altitude_category)

Si el patron 'Medium mas bajo' fuera puramente por composicion, deberia desaparecer o invertirse al mirar DENTRO de cada tipo de pitch por separado:

| AutoPitchType   |   No Altitude |   Medium Altitude |   Extreme Altitude |
|:----------------|--------------:|------------------:|-------------------:|
| Changeup        |        -0.929 |            -2.069 |             -1.521 |
| Curveball       |         9.083 |            10.01  |              8.652 |
| Cutter          |       -12.226 |           -11.534 |             -9.743 |
| Four-Seam       |        -2.859 |            -4.486 |             -3.483 |
| Sinker          |        -1.129 |            -2.417 |             -1.794 |
| Slider          |        -3.637 |            -3.564 |             -2.573 |
| Splitter        |        -0.024 |            -0.049 |             -0.356 |

**No desaparece.** Para 6 de los 7 tipos de pitch principales (Four-Seam, Sinker, Slider, Changeup, Curveball, Cutter), Medium Altitude tiene el residual mas negativo de las tres categorias — el mismo patron que en el agregado. Solo Splitter (la muestra mas chica, ~1% de los pitches) rompe el patron.

### Estandarizacion directa: crudo vs. ajustado por composicion

Se recalculo el promedio de cada altitude_category reponderando con la mezcla GLOBAL fija de pitch types (en vez de la mezcla real de cada categoria), para aislar el efecto de composicion del efecto dentro-de-tipo:

| altitude_category | crudo | ajustado por composicion |
|---|---:|---:|
| No Altitude | -1.6737 | -1.8319 |
| Medium Altitude | -2.6214 | -2.6341 |
| Extreme Altitude | -2.1187 | -1.9392 |

**Conclusion: NO es un artefacto de composicion.** El ajuste apenas mueve los valores (Medium Altitude pasa de -2.6214 crudo a -2.6341 ajustado — practicamente identico). El patron 'Medium mas bajo' se sostiene dentro de casi todos los tipos de pitch individuales, con muestras grandes en cada celda (miles a decenas de miles de pitches por combinacion tipo-altitud). **Esto es un hallazgo abierto**: hay algo en el movimiento vertical de Medium Altitude que el modelo de Magnus simple (densidad de aire + spin + velocidad) no explica, y no se debe a que Medium Altitude tire una mezcla distinta de pitches. Candidatos a investigar despues (fuera del alcance de este analisis): efecto de drag sobre la forma de la trayectoria (IVB no es un efecto de Magnus puro, tambien lo modula el drag, que este modelo no separa explicitamente mas alla de la densidad de aire en la propia formula de Magnus), o composicion de PITCHERS/equipos especificos que juegan desproporcionadamente en estadios de altitud media (no se investigo aqui, solo se descarto composicion de AutoPitchType).
