# Location+ v2 — valor de la ubicación dado el tipo de pitch y el contexto

Generado por `src/models/location_plus_v2.py`. Diseño fijado en el docstring antes de entrenar.

## 1. Qué mide

Dado el tipo de pitch, el conteo y el enfrentamiento de manos, ¿cuántas carreras esperadas vale haber terminado en **esta** ubicación en lugar de en una ubicación típica para ese mismo contexto? Usa la misma cadena de probabilidades de Pitching+ con solo ubicación y contexto (sin velocidad ni movimiento).

Features: `loc_in` (lado relativo al bateador; positivo = adentro, signo verificado con los golpeados), `PlateLocHeight`, `Balls`, `Strikes`, `AutoPitchType`, `same_side`, `bat_side`.

## 2. Sub-modelos (OOF por pitcher)

| sub-modelo | n | AUC | log-loss |
|---|---:|---:|---:|
| swing | 627,185 | 0.8588 | 0.4631 |
| take | 333,626 | multiclase | 0.2079 |
| whiff | 293,559 | 0.7426 | 0.4730 |
| foul | 222,412 | 0.6104 | 0.6727 |
| bip | 114,759 | multiclase | 0.9707 |

Las probabilidades de los 10 eventos suman 1 en cada pitch (máx. |Σp − 1| = 5.6e-16).

## 3. Validación por pitcher

| mín. pitches | pares | estabilidad | BB% t+1 | K% t+1 | carreras evitadas t+1 | (BB% propio → BB% t+1) |
|---|---:|---|---|---|---|---|
| 100 | 437 | +0.542 [+0.479, +0.600] | -0.385 [-0.462, -0.304] | -0.084 [-0.178, +0.024] | +0.133 [+0.043, +0.224] | +0.354 [+0.266, +0.441] |
| 300 | 245 | +0.580 [+0.490, +0.648] | -0.400 [-0.499, -0.289] | -0.058 [-0.191, +0.080] | +0.150 [+0.023, +0.267] | +0.413 [+0.300, +0.503] |

- Location+ y BB% del mismo año: r = -0.685 (signo esperado: negativo, mejor ubicación = menos bases por bolas).
- Location+ y Stuff+ Physical: r = -0.042 → miden cosas casi independientes.

## 4. Valor por región de la zona y ambiente

Δ realizado (carreras por 100 pitches, positivo = malo para el pitcher) con efectos fijos de pitcher, bateador, temporada y conteo. Regiones aproximadas: corazón, borde, persecución, desperdicio.

| región | % de pitches | Δ medio en No Altitude | Medium − No [±1.96 EE] | Extreme − No [±1.96 EE] |
|---|---:|---:|---|---|
| corazón | 25.3% | -3.21 | +2.38 [+1.86, +2.91] | +2.92 [+2.43, +3.41] |
| borde | 40.8% | -2.82 | +1.20 [+0.88, +1.52] | +1.55 [+1.27, +1.83] |
| persecución | 23.4% | +2.74 | +0.24 [+0.02, +0.46] | +0.46 [+0.25, +0.66] |
| desperdicio | 10.5% | +5.93 | +0.05 [-0.18, +0.28] | +0.16 [-0.05, +0.36] |

## 5. Mapas de calor

Un archivo por tipo de pitch (pitcher derecho; filas = mismo lado / lado contrario; columnas = pitcher adelante, conteo parejo, pitcher atrás). Azul = ubicación que **suele** ser más favorable al pitcher que la típica de ese contexto; rojo = menos favorable. Se ocultan las celdas con menos de 20 pitches reales en 0.25 × 0.25 pies. Un mapa describe promedios: **ninguna zona garantiza un out**.

- `reports/figures/p2/location_fourseam.png`
- `reports/figures/p2/location_sinker.png`
- `reports/figures/p2/location_cutter.png`
- `reports/figures/p2/location_slider.png`
- `reports/figures/p2/location_curveball.png`
- `reports/figures/p2/location_changeup.png`
- `reports/figures/p2/location_splitter.png`

Distribución de Location+ (pitcher-temporadas): media 98.3, sd 7.7.
