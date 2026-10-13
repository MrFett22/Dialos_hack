# ¿Es el mismo un buen lanzamiento en cualquier ambiente?

Generado por `src/analysis/pitch_environment.py`. Extensión exploratoria: no cambia ningún índice oficial.

## 1. Respuesta corta

- **Con el costo de los resultados de Extreme (capa validada), el lanzamiento óptimo prácticamente no cambia**: en 21 de 21 combinaciones tipo × conteo el cambio del perfil óptimo es menor que el ruido de muestreo; la correlación de rangos entre candidatos es ≥ 0.992 y el mejor lanzamiento general siempre queda en el top 0.05% en Extreme.
- **Si además se simula la física del aire de Extreme (capa aproximada)**, el orden de los candidatos se mueve más (correlación de rangos mediana 0.87; mínimo 0.76), pero el perfil óptimo sigue dentro del ruido en 17 de 21 combinaciones.
- **"Mejor Stuff" y "mejor pitch" no son lo mismo**: los lanzamientos con mayor Stuff+ Physical y los de mayor Pitching+ casi no se traslapan (sección 5). Esa diferencia es mucho más grande que la que produce el ambiente.

Conclusión: **un gran lanzamiento sigue siendo un gran lanzamiento**. El ambiente cambia el margen, no la definición.

## 2. Método

- **Candidatos = lanzamientos reales** de cada tipo lanzados en No Altitude (muestra de hasta 4,000 por tipo, marco de un derecho). No hay optimización libre: el óptimo no puede ser 110 mph ni una combinación nunca observada, y la correlación entre velocidad, movimiento, spin y salida es la real.
- **Objetivos:** Stuff+ Physical; Pitching+ general (valores de la liga); Pitching+ Extreme (mismo pitch, valores de Extreme, capa validada); Pitching+ Extreme con física aproximada (el mismo pitch re-simulado en el aire de Extreme con el contrafactual, que sobreestima ≈ 20% el cambio de IVB). Pitching+ en escala de lanzamiento (100 = promedio de la liga).
- **Contextos (documentados, no elegidos a la vista del resultado):** conteo 0-0 (neutral), 0-2 (dos strikes) y 2-0 (bateador adelante); mismo lado y lado contrario ponderados por la mezcla real de cada tipo.
- **Experimento 1 (física sola):** la ubicación se fija en la ubicación mediana real del tipo para cada enfrentamiento.
- **Experimento 2 (física + ubicación):** una rejilla de 0.2 pies, solo celdas con al menos 20 lanzamientos reales del tipo en el mismo enfrentamiento.
- **El óptimo** se reporta de dos formas: el **mejor lanzamiento observado** y el **perfil del top 1%** (mediana y p10–p90), puntuado con sus 25 vecinos reales más cercanos ("mejor plausible"). Si el mejor observado está rodeado de vecinos que ni siquiera están en el top 10%, se marca como inestable.
- **Incertidumbre:** bootstrap por pitcher (200 réplicas) del perfil óptimo. **Distancia** entre óptimos = distancia en desviaciones estándar del tipo en 7 variables. Poco: distancia ≤ ruido q95; Moderado: ≤ 2× ruido q95; Mucho: > 2× ruido q95 (ruido = distancia entre dos réplicas bootstrap por pitcher del mismo objetivo). No se reentrenaron los modelos en cada réplica (5 modelos × 200 réplicas no es razonable aquí): la incertidumbre del modelo no está incluida.

## 3. Resultado global

| tipo | conteo | costos de Extreme: correlación de rangos | top 1% compartido | cambio del perfil | física aprox.: correlación | top 1% compartido | cambio del perfil |
|---|---|---:|---:|---|---:|---:|---|
| Four-Seam | 0-0 | 0.999 | 67% | Poco (0.24 vs. ruido 1.12) | 0.83 | 31% | Poco (0.60) |
| Four-Seam | 0-2 | 1.000 | 95% | Poco (0.10 vs. ruido 1.30) | 0.85 | 48% | Poco (0.33) |
| Four-Seam | 2-0 | 0.996 | 70% | Poco (0.33 vs. ruido 1.12) | 0.84 | 19% | Poco (0.80) |
| Sinker | 0-0 | 0.999 | 78% | Poco (0.63 vs. ruido 1.35) | 0.80 | 16% | Moderado (1.85) |
| Sinker | 0-2 | 1.000 | 95% | Poco (0.24 vs. ruido 1.57) | 0.80 | 16% | Moderado (1.84) |
| Sinker | 2-0 | 0.999 | 90% | Poco (0.18 vs. ruido 1.04) | 0.85 | 29% | Poco (0.80) |
| Cutter | 0-0 | 0.999 | 82% | Poco (0.53 vs. ruido 3.44) | 0.89 | 33% | Poco (2.45) |
| Cutter | 0-2 | 1.000 | 95% | Poco (0.32 vs. ruido 5.36) | 0.91 | 25% | Poco (3.61) |
| Cutter | 2-0 | 0.998 | 86% | Poco (0.21 vs. ruido 1.30) | 0.92 | 43% | Poco (0.28) |
| Slider | 0-0 | 0.995 | 74% | Poco (0.21 vs. ruido 3.55) | 0.89 | 43% | Poco (0.33) |
| Slider | 0-2 | 1.000 | 90% | Poco (0.14 vs. ruido 4.11) | 0.93 | 67% | Poco (0.29) |
| Slider | 2-0 | 0.992 | 82% | Poco (0.28 vs. ruido 0.85) | 0.87 | 25% | Poco (0.63) |
| Curveball | 0-0 | 0.997 | 82% | Poco (0.14 vs. ruido 4.23) | 0.91 | 51% | Poco (0.78) |
| Curveball | 0-2 | 1.000 | 95% | Poco (0.20 vs. ruido 6.20) | 0.89 | 45% | Poco (2.69) |
| Curveball | 2-0 | 0.994 | 70% | Poco (0.43 vs. ruido 2.45) | 0.89 | 21% | Moderado (2.83) |
| Changeup | 0-0 | 0.997 | 78% | Poco (0.12 vs. ruido 1.05) | 0.76 | 7% | Poco (0.54) |
| Changeup | 0-2 | 0.999 | 90% | Poco (0.25 vs. ruido 1.61) | 0.83 | 19% | Moderado (2.08) |
| Changeup | 2-0 | 0.996 | 86% | Poco (0.17 vs. ruido 0.99) | 0.79 | 18% | Poco (0.47) |
| Splitter | 0-0 | 0.997 | 56% | Poco (0.64 vs. ruido 2.75) | 0.87 | 19% | Poco (0.85) |
| Splitter | 0-2 | 1.000 | 100% | Poco (0.00 vs. ruido 1.93) | 0.89 | 49% | Poco (0.40) |
| Splitter | 2-0 | 0.994 | 89% | Poco (0.06 vs. ruido 1.20) | 0.83 | 25% | Poco (0.60) |

"Top 1% compartido" = índice de Jaccard entre los conjuntos top 1% con cada objetivo. Casos con cambio mayor al ruido (física aproximada): Sinker 0-0 (Moderado, sobre todo IVB); Sinker 0-2 (Moderado, sobre todo IVB); Curveball 2-0 (Moderado, sobre todo salida lateral); Changeup 0-2 (Moderado, sobre todo IVB).

## 4. Tipo por tipo (experimento 1, conteo 0-0)

### Four-Seam

4,000 candidatos reales de 657 pitchers.

| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |
|---|---|---|---|---|
| velocidad | 94.1 mph [93.0 mph–96.1 mph] | 93.9 mph | 94.2 mph | 93.8 mph |
| IVB | 21.2 pulg. [19.1 pulg.–24.0 pulg.] | 21.2 pulg. | 22.6 pulg. | 22.4 pulg. |
| HB | 9.9 pulg. [5.9 pulg.–15.6 pulg.] | 9.1 pulg. | 8.8 pulg. | 8.8 pulg. |
| spin | 2478 rpm [2334 rpm–2622 rpm] | 2473 rpm | 2440 rpm | 2434 rpm |
| extensión | 6.14 pies [5.46 pies–6.49 pies] | 6.14 pies | 6.09 pies | 5.69 pies |
| Pitching+ del perfil (vecinos reales) | 108.6 | 108.2 | 105.5 | Stuff+ 112.2 |
| mejor lanzamiento observado | 109.6 (estable) | 109.3 (inestable) | 107.8 | 142.0 (estable) |

- Costos de Extreme: cambio **poco** (distancia 0.24 contra ruido 1.12); el mejor lanzamiento general queda en el percentil 99.95 en Extreme.
- Física aproximada: cambio **poco** (distancia 0.60); la variable que más se mueve es **IVB** (+0.49 sd).
- Mejor Stuff vs. mejor pitch: correlación de rangos 0.84; top 1% compartido 13%.

### Sinker

4,000 candidatos reales de 570 pitchers.

| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |
|---|---|---|---|---|
| velocidad | 93.8 mph [91.7 mph–95.8 mph] | 93.8 mph | 93.9 mph | 87.0 mph |
| IVB | 15.1 pulg. [6.4 pulg.–18.2 pulg.] | 12.5 pulg. | 8.1 pulg. | -3.5 pulg. |
| HB | 18.1 pulg. [15.7 pulg.–21.4 pulg.] | 18.0 pulg. | 21.1 pulg. | 16.1 pulg. |
| spin | 2419 rpm [2222 rpm–2613 rpm] | 2411 rpm | 2384 rpm | 2004 rpm |
| extensión | 6.03 pies [5.33 pies–6.49 pies] | 6.03 pies | 5.82 pies | 5.61 pies |
| Pitching+ del perfil (vecinos reales) | 106.9 | 106.8 | 103.5 | Stuff+ 95.2 |
| mejor lanzamiento observado | 109.6 (inestable) | 109.8 (inestable) | 107.4 | 153.8 (inestable) |

- Costos de Extreme: cambio **poco** (distancia 0.63 contra ruido 1.35); el mejor lanzamiento general queda en el percentil 99.98 en Extreme.
- Física aproximada: cambio **moderado** (distancia 1.85); la variable que más se mueve es **IVB** (-1.51 sd).
- Mejor Stuff vs. mejor pitch: correlación de rangos 0.45; top 1% compartido 4%.

### Cutter

4,000 candidatos reales de 392 pitchers.

| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |
|---|---|---|---|---|
| velocidad | 91.4 mph [88.8 mph–92.7 mph] | 91.2 mph | 90.4 mph | 88.4 mph |
| IVB | 17.4 pulg. [7.6 pulg.–20.6 pulg.] | 15.8 pulg. | 9.7 pulg. | 8.9 pulg. |
| HB | -3.5 pulg. [-6.8 pulg.–-0.6 pulg.] | -4.3 pulg. | -4.6 pulg. | -5.7 pulg. |
| spin | 2670 rpm [2493 rpm–2770 rpm] | 2666 rpm | 2641 rpm | 2612 rpm |
| extensión | 6.31 pies [5.71 pies–6.50 pies] | 6.28 pies | 6.14 pies | 5.61 pies |
| Pitching+ del perfil (vecinos reales) | 108.3 | 107.7 | 106.2 | Stuff+ 111.4 |
| mejor lanzamiento observado | 109.5 (estable) | 109.1 (estable) | 108.8 | 133.6 (estable) |

- Costos de Extreme: cambio **poco** (distancia 0.53 contra ruido 3.44); el mejor lanzamiento general queda en el percentil 99.98 en Extreme.
- Física aproximada: cambio **poco** (distancia 2.45); la variable que más se mueve es **IVB** (-2.27 sd).
- Mejor Stuff vs. mejor pitch: correlación de rangos 0.60; top 1% compartido 3%.

### Slider

4,000 candidatos reales de 662 pitchers.

| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |
|---|---|---|---|---|
| velocidad | 78.0 mph [76.3 mph–83.0 mph] | 77.9 mph | 78.4 mph | 85.5 mph |
| IVB | 5.8 pulg. [2.6 pulg.–10.2 pulg.] | 5.2 pulg. | 5.5 pulg. | -3.4 pulg. |
| HB | -14.9 pulg. [-18.6 pulg.–-9.2 pulg.] | -14.5 pulg. | -15.3 pulg. | -10.7 pulg. |
| spin | 2464 rpm [2230 rpm–2765 rpm] | 2464 rpm | 2519 rpm | 2597 rpm |
| extensión | 5.14 pies [4.68 pies–5.67 pies] | 5.11 pies | 5.14 pies | 5.54 pies |
| Pitching+ del perfil (vecinos reales) | 108.1 | 107.6 | 107.0 | Stuff+ 113.0 |
| mejor lanzamiento observado | 109.2 (estable) | 108.7 (inestable) | 108.7 | 147.3 (estable) |

- Costos de Extreme: cambio **poco** (distancia 0.21 contra ruido 3.55); el mejor lanzamiento general queda en el percentil 99.95 en Extreme.
- Física aproximada: cambio **poco** (distancia 0.33); la variable que más se mueve es **altura de salida** (+0.22 sd).
- Mejor Stuff vs. mejor pitch: correlación de rangos 0.31; top 1% compartido 1%.

### Curveball

4,000 candidatos reales de 509 pitchers.

| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |
|---|---|---|---|---|
| velocidad | 74.4 mph [72.4 mph–78.0 mph] | 74.3 mph | 73.4 mph | 81.6 mph |
| IVB | 4.8 pulg. [-0.6 pulg.–9.3 pulg.] | 4.8 pulg. | 5.9 pulg. | -14.9 pulg. |
| HB | -15.9 pulg. [-19.9 pulg.–-12.2 pulg.] | -15.8 pulg. | -15.9 pulg. | -14.5 pulg. |
| spin | 2436 rpm [2132 rpm–2628 rpm] | 2428 rpm | 2498 rpm | 2632 rpm |
| extensión | 5.47 pies [4.85 pies–5.76 pies] | 5.44 pies | 5.54 pies | 5.66 pies |
| Pitching+ del perfil (vecinos reales) | 108.6 | 108.1 | 107.7 | Stuff+ 119.0 |
| mejor lanzamiento observado | 109.0 (estable) | 108.5 (estable) | 108.2 | 152.1 (estable) |

- Costos de Extreme: cambio **poco** (distancia 0.14 contra ruido 4.23); el mejor lanzamiento general queda en el percentil 99.95 en Extreme.
- Física aproximada: cambio **poco** (distancia 0.78); la variable que más se mueve es **altura de salida** (-0.48 sd).
- Mejor Stuff vs. mejor pitch: correlación de rangos 0.45; top 1% compartido 0%.
- **Frontera del soporte:** perfil extreme_fisica (0-0) fuera de p1–p99 de los candidatos en altura de salida, salida lateral; perfil extreme_fisica (0-2) fuera de p1–p99 de los candidatos en altura de salida, salida lateral. Sigue dentro del rango real, pero en el borde: leerlo con cautela.

### Changeup

4,000 candidatos reales de 580 pitchers.

| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |
|---|---|---|---|---|
| velocidad | 86.9 mph [84.4 mph–89.5 mph] | 86.8 mph | 86.5 mph | 83.9 mph |
| IVB | 4.9 pulg. [3.6 pulg.–18.8 pulg.] | 4.8 pulg. | 5.8 pulg. | 2.6 pulg. |
| HB | 16.8 pulg. [12.9 pulg.–17.9 pulg.] | 17.0 pulg. | 17.3 pulg. | 16.1 pulg. |
| spin | 1837 rpm [1519 rpm–2328 rpm] | 1825 rpm | 1707 rpm | 1699 rpm |
| extensión | 6.01 pies [5.48 pies–6.49 pies] | 5.96 pies | 6.16 pies | 5.98 pies |
| Pitching+ del perfil (vecinos reales) | 106.8 | 107.2 | 103.3 | Stuff+ 109.5 |
| mejor lanzamiento observado | 109.9 (estable) | 109.9 (estable) | 108.3 | 141.3 (inestable) |

- Costos de Extreme: cambio **poco** (distancia 0.12 contra ruido 1.05); el mejor lanzamiento general queda en el percentil 99.98 en Extreme.
- Física aproximada: cambio **poco** (distancia 0.54); la variable que más se mueve es **spin** (-0.38 sd).
- Mejor Stuff vs. mejor pitch: correlación de rangos 0.15; top 1% compartido 3%.

### Splitter

3,524 candidatos reales de 307 pitchers.

| variable | perfil óptimo General (mediana [p10–p90]) | perfil óptimo Extreme (costos) | Extreme con física aprox. | mejor Stuff+ |
|---|---|---|---|---|
| velocidad | 86.7 mph [82.3 mph–88.4 mph] | 86.8 mph | 87.5 mph | 87.0 mph |
| IVB | 3.5 pulg. [-2.8 pulg.–21.4 pulg.] | 3.5 pulg. | 0.5 pulg. | 1.0 pulg. |
| HB | 13.0 pulg. [6.5 pulg.–16.7 pulg.] | 12.8 pulg. | 10.2 pulg. | 13.3 pulg. |
| spin | 1047 rpm [817 rpm–2192 rpm] | 1189 rpm | 1001 rpm | 987 rpm |
| extensión | 6.18 pies [5.46 pies–6.62 pies] | 6.16 pies | 6.33 pies | 5.90 pies |
| Pitching+ del perfil (vecinos reales) | 107.9 | 107.3 | 106.6 | Stuff+ 128.5 |
| mejor lanzamiento observado | 109.2 (estable) | 108.9 (estable) | 108.7 | 157.5 (estable) |

- Costos de Extreme: cambio **poco** (distancia 0.64 contra ruido 2.75); el mejor lanzamiento general queda en el percentil 99.97 en Extreme.
- Física aproximada: cambio **poco** (distancia 0.85); la variable que más se mueve es **HB** (-0.59 sd).
- Mejor Stuff vs. mejor pitch: correlación de rangos 0.33; top 1% compartido 9%.

## 5. "Mejor Stuff" no es "mejor pitch"

| tipo | correlación de rangos Stuff+ vs. Pitching+ general (0-0) | top 1% compartido | (0-2) | top 1% compartido |
|---|---:|---:|---:|---:|
| Four-Seam | 0.84 | 13% | 0.93 | 43% |
| Sinker | 0.45 | 4% | 0.62 | 7% |
| Cutter | 0.60 | 3% | 0.76 | 10% |
| Slider | 0.31 | 1% | 0.42 | 5% |
| Curveball | 0.45 | 0% | 0.60 | 10% |
| Changeup | 0.15 | 3% | 0.40 | 7% |
| Splitter | 0.33 | 9% | 0.43 | 13% |

Stuff+ premia lo que hace fallar swings; Pitching+ premia el valor total, que en 0-0 incluye strikes cantados, bolas y contacto débil. Con dos strikes ambos se acercan, porque ahí el swing fallado es el resultado más valioso.

## 6. Experimento 2: física + ubicación (conteo 0-0)

| tipo | enfrentamiento | ubicación óptima General (lado, altura) | Extreme (costos) | Extreme (física aprox.) | distancia General–Extreme | correlación de rangos de combinaciones |
|---|---|---|---|---|---:|---:|
| Four-Seam | mismo lado | (+0.6, 3.2) | (+0.6, 2.8) | (-0.4, 3.2) | 0.40 pies | 1.000 |
| Four-Seam | lado contrario | (+0.6, 2.4) | (+0.6, 2.4) | (-0.2, 3.2) | 0.00 pies | 1.000 |
| Sinker | mismo lado | (+0.6, 3.2) | (+0.6, 2.4) | (+0.8, 2.4) | 0.80 pies | 0.999 |
| Sinker | lado contrario | (+0.8, 2.2) | (+0.8, 2.2) | (+0.6, 2.2) | 0.00 pies | 1.000 |
| Cutter | mismo lado | (+0.6, 3.0) | (+0.6, 3.0) | (-0.6, 2.2) | 0.00 pies | 0.999 |
| Cutter | lado contrario | (-0.6, 3.0) | (-0.6, 3.0) | (-0.6, 2.8) | 0.00 pies | 0.999 |
| Slider | mismo lado | (-0.4, 2.6) | (-0.6, 2.2) | (-0.6, 2.6) | 0.45 pies | 1.000 |
| Slider | lado contrario | (+0.4, 2.0) | (-0.6, 2.6) | (-0.6, 2.8) | 1.17 pies | 1.000 |
| Curveball | mismo lado | (-0.4, 2.6) | (-0.4, 2.6) | (-0.6, 2.2) | 0.00 pies | 1.000 |
| Curveball | lado contrario | (-0.4, 2.6) | (-0.4, 2.6) | (-0.4, 2.8) | 0.00 pies | 1.000 |
| Changeup | mismo lado | (+0.8, 2.4) | (+0.8, 2.4) | (+0.8, 2.4) | 0.00 pies | 0.999 |
| Changeup | lado contrario | (+0.8, 2.2) | (+0.8, 2.2) | (+0.8, 2.2) | 0.00 pies | 1.000 |
| Splitter | mismo lado | (-0.4, 2.0) | (+0.6, 2.2) | (-0.4, 2.0) | 1.02 pies | 0.996 |
| Splitter | lado contrario | (+0.8, 2.2) | (+0.8, 2.2) | (+0.6, 2.0) | 0.00 pies | 0.998 |

Ubicación en pies, marco de un derecho (lado + = lado del brazo del pitcher; con el bateador del mismo lado, + es adentro).

- **Mejor combinación pitch + ubicación:** la celda óptima con costos de Extreme coincide con la general en 9 de 14 casos. Cuando se mueve (Four-Seam mismo lado: 0.40 pies, Sinker mismo lado: 0.80 pies, Slider mismo lado: 0.45 pies, Slider lado contrario: 1.17 pies, Splitter mismo lado: 1.02 pies), el orden de todas las combinaciones sigue correlacionando ≥ 0.993: hay varias ubicaciones prácticamente empatadas y el máximo salta entre ellas.
- **Lanzamiento fijo con muy buen Stuff+** (el real más cercano al perfil de mejor Stuff+ de cada tipo): el orden de las ubicaciones por Pitching+ general y Pitching+ Extreme correlaciona al menos 0.975, y su mejor celda es la misma en 13 de 14 casos con los costos de Extreme y en 4 de 14 con la física aproximada.

## 7. Relación con los hallazgos de pitch mix (comparación posterior, no usada en la búsqueda)

- La sección 6 de la página muestra que, en promedio, la recta pierde más whiff y el slider menos valor en Extreme. Eso describe cómo cambian los lanzamientos reales entre ambientes (incluye la física del aire y el vuelo de la pelota).
- Este análisis pregunta otra cosa: **cuál sería la mejor versión de cada tipo**. Con los costos de Extreme el óptimo no cambia. Con la física aproximada, los tipos que más reordenan sus candidatos en 0-0 son Changeup (correlación de rangos 0.76), Sinker (correlación de rangos 0.80), Four-Seam (correlación de rangos 0.83). Es coherente con la sección 6 (sinker y changeup son los que más movimiento horizontal pierden y la recta la que más IVB y whiff pierde), pero aun así el perfil óptimo de cada tipo queda cerca del general.

## 8. Lo que no decimos

- No es una receta: "todo pitcher debe lanzar este slider". El óptimo del modelo es una combinación favorable según los datos y no considera biomecánica, salud, tunneling, secuencias, repertorio ni fatiga.
- Los perfiles de tipos con poca muestra (Splitter) y los de ruido alto (Cutter, Slider y Curveball tienen umbrales de ruido grandes) solo detectarían cambios grandes.
- La capa física es aproximada (sobreestima el cambio de IVB) y no incluye el vuelo de la pelota bateada.
- Los modelos se entrenaron con todos los datos; el bootstrap es de candidatos (pitchers), no de modelos.
