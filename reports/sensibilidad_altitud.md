# Calidad neutral vs. sensibilidad a la altitud extrema

Generado por `src/analysis/sensitivity.py`. Definiciones en el docstring. Positivo en sensibilidad = peor en Extreme.

## 1. Sensibilidad implicada por el modelo (S_mod)

- Promedio de la liga: +1.09 carreras/100 pitches (Extreme − No Altitude, mismo pitch proyectado a ambas).
- Dispersión entre pitchers con ≥300 pitches: sd = 0.24 carreras/100.
- **R² de S_mod explicado solo por la mezcla de tipos de pitch: 0.44.** Lo que queda depende de las características de sus pitches (velocidad, movimiento, ubicación), no de una reacción individual a la altitud: todos los pitchers reciben los mismos ratios de densidad.

## 2. Sensibilidad empírica (S_emp)

- Pitchers con ≥150 pitches en Extreme y en No Altitude: 294.
- sd observada de S_emp: 1.99 carreras/100; error estándar típico: 1.88.
- Correlación entre mitades (juegos pares vs. impares): +0.059 → confiabilidad (Spearman–Brown) **+0.111**.
- sd "real" estimada entre pitchers (τ, momentos): 0.66 carreras/100.

## 3. ¿El modelo predice la sensibilidad observada?

- corr(S_mod relativo, S_emp) = **+0.057** (Spearman +0.024; 294 pitchers).

## 4. Calidad neutral vs. sensibilidad del modelo (pitchers con ≥300 pitches)

- corr(Pitching+ neutral, S_mod relativo) = +0.328.
- Cuadrantes (cortes en las medianas):

| row_0        |   sensibilidad alta |   sensibilidad baja |
|:-------------|--------------------:|--------------------:|
| calidad alta |                 185 |                 116 |
| calidad baja |                 115 |                 185 |

## 5. Lectura

- La sensibilidad del modelo es un **perfil de pitches**: dice qué tipo de pitches pierde más valor cuando el aire es menos denso. No es una habilidad individual de adaptación medida en los datos.
- La sensibilidad empírica es la que podría revelar una habilidad individual, pero su confiabilidad indica cuánto de ella es señal. Con confiabilidad cercana a 0, rankear pitchers por "resistencia a la altitud" observada no es defendible.
- El cuadrante "calidad alta + sensibilidad baja" se puede mostrar, pero debe leerse como "buen pitcher con un repertorio que el modelo espera que pierda poco en altura", no como "pitcher demostradamente resistente".
