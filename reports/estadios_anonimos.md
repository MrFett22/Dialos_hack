# Grupos de estadio anónimos (ANON_STADIUM_xx)

Generado por `src/data/parks.py`; método y supuestos en el docstring del módulo. **Se respeta la anonimización del dataset:** no se usa información externa y no se intenta identificar equipos, estadios ni pitchers reales. Un grupo = los juegos que un mismo equipo anónimo jugó como local. La categoría de altitud solo se usa para validar.

## 1. Reconstrucción por temporada

El número de grupos se elige con un criterio interno (mínimo de juegos con local = visitante; empate → menos grupos).

| temporada | k | elegido | juegos con local = visitante | pureza de categoría por grupo |
|---|---:|:---:|---:|---:|
| 2024 | 16 |  | 2.8% | 90.0% |
| 2024 | 18 |  | 0.7% | 93.5% |
| 2024 | 20 | ✔ | 0.0% | 100.0% |
| 2024 | 22 |  | 0.0% | 100.0% |
| 2024 | 24 |  | 0.0% | 99.8% |
| 2025 | 16 |  | 5.0% | 73.9% |
| 2025 | 18 |  | 2.6% | 81.7% |
| 2025 | 20 |  | 1.2% | 91.0% |
| 2025 | 22 |  | 0.4% | 95.7% |
| 2025 | 24 | ✔ | 0.1% | 99.9% |
| 2026 | 16 |  | 4.4% | 87.3% |
| 2026 | 18 |  | 2.0% | 86.6% |
| 2026 | 20 |  | 2.2% | 87.3% |
| 2026 | 22 | ✔ | 0.2% | 99.6% |
| 2026 | 24 |  | 0.2% | 99.6% |

*Pureza* = fracción de los juegos locales de cada grupo que cae en una sola categoría de altitud (no se usa para elegir).

## 2. Estabilidad del clustering

Índice de Rand ajustado (ARI; 1 = misma partición, 0 = azar) de la asignación juego → grupo local contra la reconstrucción oficial.

| temporada | ARI con otras semillas (mín / media) | ARI con 80% de los juegos (mín / media) |
|---|---|---|
| 2024 | 1.000 / 1.000 | 1.000 / 1.000 |
| 2025 | 0.962 / 0.970 | 0.997 / 0.998 |
| 2026 | 0.983 / 0.990 | 0.916 / 0.941 |

## 3. IDs estables entre temporadas

Enlace por similitud de roster (Jaccard ≥ 0.08 y ≥ 2.5× la segunda mejor). En las temporadas posteriores a la primera: Jaccard mediano del enlace 0.18 (segundo mejor: 0.03); clusters sin enlace claro (reciben ID nuevo): 10 de 44.

| ID | 2024 | 2025 | 2026 |
|---|---|---|---|
| ANON_STADIUM_01 | No · 53 | No · 52 | No · 23 |
| ANON_STADIUM_02 | Medium · 51 | Medium · 47 | Medium · 10 (mixto) |
| ANON_STADIUM_03 | No · 51 | No · 48 | No · 20 |
| ANON_STADIUM_04 | Extreme · 51 | Extreme · 50 | Extreme · 28 |
| ANON_STADIUM_05 | Medium · 51 | Medium · 51 | Medium · 26 |
| ANON_STADIUM_06 | No · 50 | No · 38 |  |
| ANON_STADIUM_07 | No · 48 | No · 42 | No · 20 |
| ANON_STADIUM_08 | No · 45 | No · 37 | No · 31 |
| ANON_STADIUM_09 | No · 45 | No · 45 | No · 28 |
| ANON_STADIUM_10 | Extreme · 44 | Extreme · 48 |  |
| ANON_STADIUM_11 | Medium · 43 | Medium · 41 | Medium · 22 |
| ANON_STADIUM_12 | Extreme · 43 | Extreme · 43 | Extreme · 30 |
| ANON_STADIUM_13 | Medium · 43 | Medium · 35 | Medium · 17 |
| ANON_STADIUM_14 | Extreme · 41 | Extreme · 44 | Extreme · 27 |
| ANON_STADIUM_15 | No · 41 |  |  |
| ANON_STADIUM_16 | Extreme · 41 |  |  |
| ANON_STADIUM_17 | No · 38 | No · 32 | No · 28 |
| ANON_STADIUM_18 | No · 35 | No · 42 | No · 20 |
| ANON_STADIUM_19 | Medium · 16 |  |  |
| ANON_STADIUM_20 | Extreme · 3 (mixto) | Extreme · 14 (mixto) |  |
| ANON_STADIUM_21 |  | No · 49 | No · 19 |
| ANON_STADIUM_22 |  | Medium · 40 | Medium · 17 |
| ANON_STADIUM_23 |  | Extreme · 38 | Extreme · 21 |
| ANON_STADIUM_26 |  |  | No · 25 |
| ANON_STADIUM_27 |  |  | Extreme · 23 |
| ANON_STADIUM_28 |  |  | Extreme · 16 |

Se omiten 4 grupos diminutos (menos de 15 juegos locales y sin categoría de altitud: juegos sin `altitude_category` en el dataset).

Celdas: categoría · juegos locales. *(mixto)* = el grupo mezcla categorías (probablemente dos equipos fusionados); sus juegos se usan solo a nivel de categoría.

**Consistencia de categoría entre temporadas:** 19 de 19 grupos resueltos que aparecen en ≥ 2 temporadas conservan la misma categoría de altitud. La categoría no se usó para enlazar, así que esto valida el enlace.

## 4. Diferencias físicas entre grupos

Ratio de drag/v² y Magnus/v² del mismo pitcher, tipo y temporada en el grupo contra los OTROS estadios de No Altitude (todas las temporadas juntas; IC 95% por bootstrap de celdas pitcher × temporada × tipo). 1 = igual que el promedio de No Altitude; menos de 1 = menos fuerza del aire.

| ID | categoría | celdas | ratio drag [IC 95%] | ratio Magnus [IC 95%] |
|---|---|---:|---|---|
| ANON_STADIUM_12 | Extreme Altitude | 656 | 0.828 [0.824, 0.832] | 0.767 [0.759, 0.772] |
| ANON_STADIUM_28 | Extreme Altitude | 92 | 0.832 [0.825, 0.843] | 0.769 [0.743, 0.789] |
| ANON_STADIUM_04 | Extreme Altitude | 758 | 0.840 [0.835, 0.845] | 0.717 [0.710, 0.725] |
| ANON_STADIUM_10 | Extreme Altitude | 615 | 0.857 [0.854, 0.864] | 0.792 [0.779, 0.801] |
| ANON_STADIUM_14 | Extreme Altitude | 685 | 0.861 [0.858, 0.866] | 0.781 [0.775, 0.787] |
| ANON_STADIUM_27 | Extreme Altitude | 120 | 0.863 [0.848, 0.869] | 0.806 [0.787, 0.821] |
| ANON_STADIUM_23 | Extreme Altitude | 352 | 0.888 [0.880, 0.894] | 0.808 [0.799, 0.820] |
| ANON_STADIUM_16 | Extreme Altitude | 232 | 0.902 [0.893, 0.908] | 0.837 [0.821, 0.851] |
| ANON_STADIUM_02 | Medium Altitude | 615 | 0.870 [0.865, 0.875] | 0.815 [0.808, 0.821] |
| ANON_STADIUM_11 | Medium Altitude | 656 | 0.893 [0.888, 0.897] | 0.833 [0.827, 0.842] |
| ANON_STADIUM_19 | Medium Altitude | 117 | 0.905 [0.895, 0.914] | 0.839 [0.820, 0.853] |
| ANON_STADIUM_13 | Medium Altitude | 587 | 0.905 [0.900, 0.910] | 0.825 [0.817, 0.834] |
| ANON_STADIUM_22 | Medium Altitude | 334 | 0.906 [0.897, 0.911] | 0.848 [0.839, 0.861] |
| ANON_STADIUM_05 | Medium Altitude | 704 | 0.949 [0.946, 0.952] | 0.859 [0.852, 0.867] |
| ANON_STADIUM_07 | No Altitude | 579 | 0.971 [0.967, 0.975] | 0.910 [0.902, 0.919] |
| ANON_STADIUM_26 | No Altitude | 141 | 0.977 [0.961, 0.986] | 0.993 [0.960, 1.030] |
| ANON_STADIUM_03 | No Altitude | 646 | 0.979 [0.975, 0.983] | 1.016 [1.007, 1.025] |
| ANON_STADIUM_08 | No Altitude | 624 | 0.988 [0.984, 0.992] | 0.967 [0.960, 0.974] |
| ANON_STADIUM_01 | No Altitude | 653 | 0.990 [0.984, 0.996] | 0.966 [0.958, 0.973] |
| ANON_STADIUM_18 | No Altitude | 536 | 0.994 [0.990, 0.999] | 1.031 [1.021, 1.040] |
| ANON_STADIUM_06 | No Altitude | 509 | 0.997 [0.992, 1.000] | 1.007 [0.996, 1.014] |
| ANON_STADIUM_17 | No Altitude | 552 | 1.010 [1.006, 1.014] | 1.005 [0.995, 1.016] |
| ANON_STADIUM_21 | No Altitude | 378 | 1.012 [1.006, 1.018] | 1.021 [1.013, 1.032] |
| ANON_STADIUM_15 | No Altitude | 207 | 1.025 [1.021, 1.033] | 1.036 [1.022, 1.060] |
| ANON_STADIUM_09 | No Altitude | 614 | 1.064 [1.061, 1.068] | 1.082 [1.072, 1.092] |

**Heterogeneidad dentro de cada categoría.** sd real = √max(Var observada − error² medio, 0); I² = fracción de la variación entre grupos que no es ruido de muestreo.

| componente | categoría | grupos | rango | sd observada | error de muestreo | sd real entre estadios | I² |
|---|---|---:|---|---:|---:|---:|---:|
| drag | Extreme Altitude | 8 | 0.828–0.902 | 0.026 | 0.004 | 0.026 | 98% |
| drag | Medium Altitude | 6 | 0.870–0.949 | 0.026 | 0.003 | 0.026 | 99% |
| drag | No Altitude | 11 | 0.971–1.064 | 0.027 | 0.003 | 0.026 | 99% |
| Magnus | Extreme Altitude | 8 | 0.717–0.837 | 0.036 | 0.008 | 0.035 | 95% |
| Magnus | Medium Altitude | 6 | 0.815–0.859 | 0.016 | 0.005 | 0.015 | 90% |
| Magnus | No Altitude | 11 | 0.910–1.082 | 0.045 | 0.007 | 0.044 | 98% |

**Test–retest físico** (¿el ratio de un grupo en una temporada predice el de la siguiente?):

- Todos los grupos: corr = 0.96 (drag), 0.99 (Magnus); 32 pares grupo-temporada.
- Solo Extreme: corr = 0.68 (drag), 0.94 (Magnus); 8 pares.

## 5. Qué se puede y qué no se puede afirmar

- **Se puede:** cada categoría de altitud agrupa varios estadios anónimos; dentro de una misma categoría los estadios pueden diferir en densidad efectiva más allá del ruido de muestreo (I²). "Extreme Altitude" no es un ambiente físico homogéneo.
- **No se puede (por diseño):** asignar nombres reales a los grupos. Los resultados se expresan por categoría de altitud y por grupo anónimo.
- **Identificabilidad:** cada grupo pertenece a una sola categoría. En un modelo con efectos de grupo, el efecto de la categoría queda absorbido por los grupos: solo se identifica como el promedio de los efectos de sus pocos grupos (4–6), no por separado.
- **Supuesto:** un equipo juega como local siempre en el mismo estadio. Un cambio de sede aparecería como un cambio de categoría entre temporadas (sección 3).
