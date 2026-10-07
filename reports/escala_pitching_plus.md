# Escala de Pitching+: ¿por qué el promedio no es exactamente 100?

Generado por `src/validation/scale_check.py` (definición en el docstring).

|                                                          |   2024 |   2025 |   2026 |
|:---------------------------------------------------------|-------:|-------:|-------:|
| sin shrinkage, ponderada por pitches, n≥100 (definición) | 100    | 100    | 100    |
| sin shrinkage, por pitcher, n≥100                        |  98.3  |  98.06 |  98.85 |
| con shrinkage, ponderada por pitches, n≥100              | 100.14 | 100.21 | 100.14 |
| con shrinkage, por pitcher, n≥100                        |  98.85 |  98.74 |  99.31 |
| con shrinkage, por pitcher, todos                        |  98    |  97.87 |  98.79 |
| con shrinkage, por pitcher, n<100                        |  94.45 |  93.83 |  97.45 |
| sd con shrinkage, n≥100                                  |   8.85 |   8.66 |   8.58 |
| sd sin shrinkage, n≥100                                  |  10.93 |  11.2  |  11.27 |
| corr(volumen, score) n≥100                               |   0.24 |   0.26 |   0.22 |

## Lectura

- **100 = el pitch promedio de la liga**: la media de E[Δ] ponderada por pitches entre pitchers con ≥100 pitches. Con esa ponderación y sin shrinkage el promedio es exactamente 100 (primera fila).
- El promedio **por pitcher** queda por debajo de 100 porque los pitchers con más volumen son mejores (correlación de rangos volumen–score positiva, última fila) y porque los de menos de 100 pitches son peores en promedio.
- El **shrinkage** casi no mueve el promedio ponderado; lo que hace es acercar a 100 a los de poca muestra.
- La **neutralización** no es la causa: el índice con altitud tiene la misma construcción.
- La sd del índice es menor que 10 **por diseño**: 10 puntos = 1 τ (la desviación de la habilidad real), y θ̂ está encogido, así que su dispersión es menor que τ. La versión sin shrinkage tiene sd > 10 porque incluye ruido de muestreo.

## Decisión

**Se conserva la escala (opción A).** La definición es explícita y coincide con la convención de los índices "+" (100 = promedio de la liga por pitch). Recalibrar para que el promedio por pitcher sea 100 sería cosmético y haría que el valor 100 dependiera de cuántos pitchers de poco volumen hay en la temporada. Lo que cambia es la documentación: "100 = pitch promedio de la liga", no "100 = pitcher promedio".
