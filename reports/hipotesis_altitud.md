# Hipótesis del analista: ¿en altitud extrema conviene evitar el contacto?

Generado por `src/analysis/altitude_hypotheses.py` (625,972 pitches). Hipótesis, variables y criterio fijados en el docstring del módulo **antes** de correr. *Respaldada* = el IC 95% excluye 0 en la dirección de la hipótesis; *contradicha* = lo excluye en la dirección contraria; *no concluyente* = incluye 0.

**Cómo leer β_E.** Diferencia promedio Extreme − No Altitude comparando al **mismo pitcher contra el mismo bateador** (efectos fijos de pitcher, bateador, temporada y conteo). La columna *crudo* es la diferencia simple, sin controles.

## H1 y H2

| hipótesis | variable | media en No Altitude | crudo E − N | β_E con controles [IC 95%] | β_M (Medium) | veredicto |
|---|---|---:|---:|---|---:|---|
| H1 | whiff por swing | 0.2493 | -0.0172 | -0.0290 [-0.0341, -0.0239] | -0.0126 | **RESPALDADA** |
| H1 | bola en juego por swing | 0.3784 | +0.0291 | +0.0396 [+0.0344, +0.0449] | +0.0247 | **RESPALDADA** |
| H1 | ponche por turno | 0.2026 | -0.0359 | -0.0466 [-0.0522, -0.0410] | -0.0289 | **RESPALDADA** |
| H2 | run value RE24 por bola en juego | 0.0145 | +0.0707 | +0.0678 [+0.0576, +0.0780] | +0.0583 | **RESPALDADA** |
| H2 | hit por bola en juego | 0.3308 | +0.0556 | +0.0532 [+0.0449, +0.0615] | +0.0462 | **RESPALDADA** |
| H2 | extra base por bola en juego | 0.1002 | +0.0250 | +0.0225 [+0.0170, +0.0279] | +0.0208 | **RESPALDADA** |
| H2 | HR por bola en juego | 0.0365 | +0.0061 | +0.0042 [+0.0009, +0.0076] | +0.0057 | **RESPALDADA** |
| H2 | velocidad de salida (mph) | 83.0288 | +0.9869 | +0.9503 [+0.6597, +1.2410] | +1.4544 | **RESPALDADA** |
| H2 | velocidad de salida ≥ 95 mph | 0.2239 | +0.0351 | +0.0273 [+0.0196, +0.0350] | +0.0330 | **RESPALDADA** |
| H2 | barrel | 0.0256 | +0.0023 | -0.0002 [-0.0030, +0.0025] | +0.0038 | **NO CONCLUYENTE** |

### Consistencia entre estadios de Extreme

Con ~4–5 estadios por temporada, los IC por juego no capturan la variación entre estadios. Aquí se estima β por cada estadio anónimo de Extreme (`reports/estadios_anonimos.md`; IDs estables entre temporadas) contra No Altitude, con los mismos controles. Si el efecto fuera de un solo estadio, la mayoría de las columnas tendría el signo contrario o cero.

| variable | estadios en la dirección de la hipótesis | β por estadio (ordenados) |
|---|---:|---|
| whiff por swing | 8/8 | -0.041, -0.040, -0.031, -0.030, -0.025, -0.023, -0.021, -0.002 |
| bola en juego por swing | 8/8 | +0.011, +0.021, +0.031, +0.033, +0.035, +0.049, +0.052, +0.066 |
| ponche por turno | 8/8 | -0.059, -0.058, -0.050, -0.047, -0.040, -0.035, -0.035, -0.013 |
| run value RE24 por bola en juego | 8/8 | +0.032, +0.044, +0.044, +0.061, +0.063, +0.067, +0.074, +0.089 |
| hit por bola en juego | 8/8 | +0.030, +0.033, +0.035, +0.038, +0.048, +0.049, +0.062, +0.082 |
| extra base por bola en juego | 8/8 | +0.002, +0.007, +0.007, +0.026, +0.032, +0.035, +0.036, +0.040 |
| HR por bola en juego | 6/8 | -0.007, -0.002, +0.002, +0.002, +0.006, +0.008, +0.017, +0.017 |
| velocidad de salida (mph) | 7/8 | -0.637, +0.644, +0.983, +1.313, +1.334, +1.408, +1.773, +3.041 |
| velocidad de salida ≥ 95 mph | 7/8 | -0.008, +0.015, +0.025, +0.029, +0.037, +0.041, +0.044, +0.060 |
| barrel | 4/8 | -0.007, -0.005, -0.000, -0.000, +0.002, +0.002, +0.004, +0.005 |

Tamaños: whiff por swing: n=293,020, juegos=2,116; bola en juego por swing: n=293,020, juegos=2,116; ponche por turno: n=162,576, juegos=2,115; run value RE24 por bola en juego: n=114,538, juegos=2,114; hit por bola en juego: n=114,538, juegos=2,114; extra base por bola en juego: n=114,538, juegos=2,114; HR por bola en juego: n=114,538, juegos=2,114; velocidad de salida (mph): n=106,805, juegos=2,112; velocidad de salida ≥ 95 mph: n=106,805, juegos=2,112; barrel: n=106,804, juegos=2,112

## H3: ¿cuánto más cuesta una bola en juego que un whiff?

G = valor esperado (en carreras, para el bateador) de una bola en juego menos el de un whiff en el mismo conteo, con V(b,s) y Δ recalculados en cada categoría y promediados con la mezcla de conteos de los swings de toda la liga. IC 95% por bootstrap de juegos.

| categoría | Δ(whiff) | E[Δ \| bola en juego] | G [IC 95%] |
|---|---:|---:|---|
| No Altitude | -0.1371 | +0.0352 | +0.1723 [+0.1677, +0.1764] |
| Medium Altitude | -0.1548 | +0.0574 | +0.2122 [+0.2063, +0.2175] |
| Extreme Altitude | -0.1646 | +0.0535 | +0.2181 [+0.2122, +0.2243] |

**G_Extreme − G_NoAltitude = +0.0457 carreras [+0.0382, +0.0533] → RESPALDADA.** (Medium − No: +0.0399 [+0.0327, +0.0464].)

Por estadio anónimo de Extreme: G > G_NoAltitude en **7/7** estadios (G por estadio: +0.189, +0.192, +0.220, +0.222, +0.228, +0.230, +0.231).

## Limitaciones

- *Extreme* agrupa varios estadios anónimos distintos (`reports/estadios_anonimos.md`): es el efecto promedio de esos estadios, no de un estadio en particular.
- Con tan pocos estadios por categoría, la altitud no se puede separar de otras características del parque (dimensiones, clima, calibración de Trackman de la velocidad de salida). Los IC por juego son condicionales a esos estadios.
- Es asociación con controles por pitcher y bateador, no un efecto causal aislado de la densidad del aire.
