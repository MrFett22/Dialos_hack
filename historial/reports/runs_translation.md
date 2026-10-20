# Traduccion de Pitching+ a carreras por temporada — impacto de altitud

Convierte los deltas de Pitching+ por altitud de `historial/reports/altitude_counterfactual.md` (seccion 1) a un estimado de carreras por temporada, para que el hallazgo sea legible sin tener que interpretar puntos de un indice abstracto. Pensado para usarse directamente en el entregable de Recomendaciones Operativas.

## 1. Pendiente Pitching+ -> run_value

Regresion lineal ponderada por `n_pitches` de `run_value` (carreras esperadas por pitch, prediccion OOF del modelo — mayor = peor para el pitcher) sobre `pitching_plus`, usando las 1,278 pitcher-temporadas con n_pitches>=100 (de 1,628 totales — mismo filtro de tamaño minimo que `reports/pitching_plus_score_oof.md`).

| estadistico | valor |
|---|---:|
| pendiente (runs por pitch, por punto de Pitching+) | -0.001551 |
| intercepto | 0.162144 |
| R² (ponderado) | 0.7177 |
| correlacion ponderada | -0.8472 |
| correlacion simple (sin ponderar, referencia) | -0.8616 |

**Lectura de la pendiente:** cada punto de Pitching+ perdido equivale a ~0.00155 carreras esperadas ADICIONALES por pitch (pendiente negativa: mas Pitching+ -> menos run_value/menos carreras permitidas, como se espera de un indice donde mas alto es mejor).

**Nota sobre R²:** run_value es solo 1 de los 8 componentes del score combinado (peso 20%, ver `historial/reports/pitching_plus_score.md`) — un R² notablemente menor a 1 es esperado y no invalida la pendiente como conversion de escala; refleja que Pitching+ tambien captura comando/engaño y calidad de contacto que no se traducen 1-a-1 en run_value pitch a pitch, sino en agregado.

**Nota sobre rango de validez:** los deltas de Pitching+ por altitud (seccion 2) son pequeños (entre -1.3 y +0.1 puntos) frente al rango real de pitcher-temporadas usado para ajustar la pendiente (std=6.32, ver `reports/pitching_plus_score_oof.md`) — la extrapolacion lineal es local, sobre una porcion muy chica del rango donde la relacion fue estimada, no hace falta asumir linealidad global.

## 2. Impacto en carreras por temporada, por tipo de pitch y rol

Runs por temporada = pendiente x delta_pitching_plus x volumen de pitches del rol. Volumen tipico: **abridor 2,500-3,000 pitches/temporada** (punto medio 2,750), **relevista 800-1,000** (punto medio 900). Convencion de signo: **positivo = mas carreras permitidas (el pitcher PIERDE valor en altitud)**; **negativo = menos carreras permitidas (el pitcher GANA valor en altitud)**.

| AutoPitchType | delta Pitching+ | Abridor (2500-3000) | Relevista (800-1000) |
|---|---:|---|---|
| Sinker | -1.059 | +4.10 a +4.93 (≈+4.52) | +1.31 a +1.64 (≈+1.48) |
| Changeup | -0.951 | +3.69 a +4.43 (≈+4.06) | +1.18 a +1.48 (≈+1.33) |
| Cutter | -0.904 | +3.50 a +4.21 (≈+3.86) | +1.12 a +1.40 (≈+1.26) |
| Slider | -0.843 | +3.27 a +3.92 (≈+3.60) | +1.05 a +1.31 (≈+1.18) |
| Four-Seam | -0.839 | +3.25 a +3.91 (≈+3.58) | +1.04 a +1.30 (≈+1.17) |
| Curveball | -0.829 | +3.21 a +3.86 (≈+3.54) | +1.03 a +1.29 (≈+1.16) |
| Splitter | -0.817 | +3.17 a +3.80 (≈+3.49) | +1.01 a +1.27 (≈+1.14) |

## 3. Lectura en lenguaje llano

- Un **abridor sinker-heavy** (repertorio dominado por Sinker) que se muda a jugar en altitud extrema (el estadio de altitud extrema) pierde aproximadamente **4.5 carreras por temporada** (rango 4.1-4.9, segun volumen de pitches) frente a jugar a nivel del mar — el mayor impacto de los tipos de pitch analizados.
- El mismo pitch type, en un **relevista**, representa ~1.5 carreras por temporada (1.3-1.6) — menor en magnitud absoluta solo porque lanza muchos menos pitches en la temporada, no porque el efecto por pitch sea distinto.

### Caso especial — Curveball: correccion de encuadre

**El delta_pitching_plus TOTAL de Curveball tambien es NEGATIVO (-0.829, ver `historial/reports/altitude_counterfactual.md` seccion 1) — un abridor curveball-heavy tambien PIERDE runs en el agregado al jugar en altitud, no los gana.** Lo que la seccion 1 de ese reporte muestra es que Curveball pierde MENOS que Sinker/Changeup/Cutter (3.5 carreras/temporada para un abridor, vs. 4.5 de Sinker), no que gane valor.

**La direccion opuesta real esta confinada a UN componente dentro de la descomposicion de 8 targets** (`historial/reports/altitude_counterfactual.md` seccion 4): `is_whiff` (y, en menor medida, `is_weak_contact`) SI mejoran para Curveball en altitud — pero ese beneficio queda sobre-compensado por perdidas en `is_out_in_pa` e `is_called_strike`, que pesan mas en el score combinado. Traduciendo SOLO ese componente aislado a carreras (mismo metodo de la seccion 1, aplicado a `contrib_is_whiff`=+0.063 puntos de Pitching+ en vez de al delta total):

- El mecanismo especifico de 'mas swing-and-miss con la curva en altitud' vale, AISLADO, aproximadamente **0.27 carreras ganadas por temporada** para un abridor (0.24 a 0.29) — pero es un efecto pequeño que NO revierte el resultado neto (que sigue siendo una perdida de ~3.5 carreras/temporada, no una ganancia).
- Recordar la nota de cautela de `historial/reports/altitude_whiff_stratified.md`: el patron de is_whiff en Curveball tiene p=0.036 (vs. p<0.001 del efecto agregado) — es un patron real confirmado por dos metodos independientes, pero con mayor incertidumbre relativa que el resto de las cifras de este reporte. No usar esta cifra especifica con la misma confianza que las demas de la tabla de la seccion 2.

**Para el entregable de Recomendaciones Operativas: no presentar a Curveball como un tipo de pitch que 'gana' valor en altitud.** Todos los 7 tipos de pitch analizados pierden Pitching+/runs en altitud extrema; Curveball, Four-Seam y Splitter son los que MENOS pierden, no los que ganan.

**Limitacion explicita:** estas cifras asumen que el 100% de los pitches del pitcher son del tipo indicado (el escenario 'heavy' de cada fila) — un pitcher real mezcla varios tipos, asi que el impacto real sobre un pitcher especifico es un promedio ponderado por su propio `pitch_usage_pct` (ver `reports/arsenal_summary.md`), no directamente uno de estos valores. Estas cifras son el efecto marginal por tipo de pitch, utiles para comparar arquetipos de repertorio, no una prediccion puntual para un pitcher individual sin ponderar por su arsenal real.
