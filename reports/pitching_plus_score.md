# Pitching+ v1 — score compuesto por pitcher-temporada

## Pesos de combinacion (documentados)

| target | peso | nivel | justificacion |
|---|---:|---|---|
| run_value | 0.20 | alto | metrica mas directa de prevencion de carreras |
| is_out_in_pa | 0.20 | alto | ~82% del efecto de altitud en run_value (Semana 2) estaba mediado por tasa de outs, no por contacto |
| is_whiff | 0.12 | medio | comando/engaño — capacidad de generar swing-and-miss |
| is_chase | 0.12 | medio | comando/engaño — capacidad de generar persecucion fuera de zona |
| is_called_strike | 0.12 | medio | comando/engaño — capacidad de robarse strikes cantados |
| is_weak_contact | 0.08 | bajo | calidad de contacto — señal secundaria |
| is_groundball | 0.08 | bajo | calidad de contacto — señal secundaria |
| is_hard_contact | 0.08 | bajo | is_hard_contact NO mostro patron claro por altitud (Semana 2/3); el efecto dominante de altitud es via outs, no via contacto duro — coherente con pesarlo menos aqui tambien |

Suma de pesos: 1.00.

## Metodologia

1. Se usan las PREDICCIONES de los 8 modelos de `src/models/pitching_plus.py` (no los outcomes 0/1 observados crudos), promediadas por (pitcher_anon_id, year) — asi se suaviza el ruido pitch a pitch con la expectativa modelada, el mismo principio detras de cualquier indice tipo '+'.
2. `run_value` e `is_hard_contact` se invierten de signo antes de combinar (mayor valor = peor para el pitcher en esos dos; en el resto, mayor valor = mejor).
3. Cada target se estandariza (z-score) DENTRO de cada `year` por separado, no agrupando las 3 temporadas — para no confundir diferencias de nivel entre temporadas (ej. 2026 parcial) con diferencias reales de habilidad.
4. Los 8 z-scores se combinan con los pesos de la tabla de arriba (normalizados a sumar 1).
5. `Pitching+ = 100 + 10 * z_combinado`.

**Limitacion explicita:** las predicciones se generan sobre TODOS los pitches validos (train+test de pitcher_holdout combinados), no solo sobre out-of-fold. Los pitchers que estuvieron en el TRAIN del modelo pueden tener predicciones ligeramente optimistas (el modelo vio sus propios pitches al ajustarse) frente a los pitchers de TEST, que son genuinamente out-of-sample. Para una version de produccion, lo correcto es generar predicciones out-of-fold para todos los pitchers (K-fold o similar) antes de construir el ranking — queda pendiente, no se implemento en este v1.

## Distribucion del Pitching+ final

n pitcher-temporadas: 1,628

| estadistico | valor |
|---|---:|
| mean | 100.00 |
| std | 6.41 |
| min | 61.98 |
| 1% | 77.56 |
| 5% | 88.77 |
| 10% | 92.87 |
| 25% | 97.27 |
| 50% | 100.77 |
| 75% | 103.76 |
| 90% | 106.46 |
| 95% | 107.96 |
| 99% | 112.75 |
| max | 150.09 |

## Ranking — top 10 y bottom 10 (filtro: n_pitches >= 100 en la temporada)

De 1,628 pitcher-temporadas, 1,278 cumplen el filtro de tamaño minimo. El filtro es una regla simple de corte, NO shrinkage bayesiano completo (pendiente de fases futuras) — pitcher-temporadas con muestra chica quedan fuera del ranking en vez de regularizadas hacia la media.

### Top 10 (mejor Pitching+)

| pitcher_anon_id   |   year |   n_pitches |   pitching_plus |   z_is_whiff |   z_is_chase |   z_is_called_strike |   z_is_weak_contact |   z_is_groundball |   z_is_hard_contact |   z_is_out_in_pa |   z_run_value |
|:------------------|-------:|------------:|----------------:|-------------:|-------------:|---------------------:|--------------------:|------------------:|--------------------:|-----------------:|--------------:|
| pitcher_00469     |   2026 |         283 |         116.532 |        2.315 |        0.827 |                1.79  |               0.693 |             1.446 |               0.652 |            2.27  |         1.92  |
| pitcher_01058     |   2025 |         471 |         114.927 |        1.15  |        0.658 |                1.795 |               2.503 |             1.237 |               0.821 |            1.952 |         1.526 |
| pitcher_00456     |   2026 |         220 |         114.405 |        2.164 |        1.383 |                0.158 |               0.654 |             1.183 |              -0.127 |            1.854 |         2.442 |
| pitcher_00402     |   2026 |         101 |         113.546 |        1.174 |        0.626 |                0.962 |               0.453 |            -0.357 |               0.639 |            2.406 |         2.415 |
| pitcher_01058     |   2026 |         301 |         113.007 |        0.426 |        0.597 |                1.462 |               1.223 |             1.648 |               0.884 |            1.89  |         1.62  |
| pitcher_00222     |   2025 |         528 |         112.958 |        1.81  |       -0.007 |                2.317 |               0.53  |             0.707 |               0.82  |            1.439 |         1.746 |
| pitcher_00875     |   2026 |         349 |         112.842 |        0.738 |        2.041 |                0.474 |               1.268 |             0.847 |               0.24  |            1.996 |         1.53  |
| pitcher_00054     |   2024 |         691 |         112.487 |        2.024 |        0.928 |               -0.044 |              -0.081 |             0.411 |               0.885 |            2.016 |         1.997 |
| pitcher_00263     |   2026 |         102 |         111.634 |        1.819 |        1.558 |                1.136 |              -0.035 |            -0.274 |               0.479 |            1.497 |         1.544 |
| pitcher_00548     |   2025 |         542 |         111.284 |        1.863 |        0.031 |                1.082 |               0.89  |            -0.267 |               1.094 |            1.503 |         1.667 |

### Bottom 10 (peor Pitching+)

| pitcher_anon_id   |   year |   n_pitches |   pitching_plus |   z_is_whiff |   z_is_chase |   z_is_called_strike |   z_is_weak_contact |   z_is_groundball |   z_is_hard_contact |   z_is_out_in_pa |   z_run_value |
|:------------------|-------:|------------:|----------------:|-------------:|-------------:|---------------------:|--------------------:|------------------:|--------------------:|-----------------:|--------------:|
| pitcher_01048     |   2024 |         106 |          86.998 |       -1.312 |       -1.559 |                0.239 |              -1.447 |            -1.136 |              -0.172 |           -2.281 |        -1.539 |
| pitcher_00369     |   2024 |         103 |          87.136 |       -1.082 |       -2.4   |                0.26  |              -1.072 |            -1.131 |              -0.67  |           -1.947 |        -1.402 |
| pitcher_00309     |   2026 |         107 |          87.326 |       -1.19  |       -0.869 |               -0.348 |              -1.541 |            -1.755 |               0.166 |           -2.052 |        -1.589 |
| pitcher_00777     |   2024 |         206 |          87.395 |       -1.778 |       -1.759 |                0.023 |              -1.391 |            -0.668 |               0.245 |           -1.841 |        -1.628 |
| pitcher_00089     |   2025 |         190 |          87.985 |       -1.902 |       -1.893 |                0.334 |              -1.322 |            -0.869 |              -0.257 |           -1.276 |        -1.675 |
| pitcher_00120     |   2025 |         267 |          88.614 |       -1.115 |       -1.252 |                0.622 |              -1.747 |            -0.694 |               0.757 |           -2.266 |        -1.706 |
| pitcher_00687     |   2024 |         137 |          88.762 |       -2.189 |       -1.183 |                0.359 |              -0.772 |            -0.361 |              -1.258 |           -1.382 |        -1.472 |
| pitcher_00992     |   2025 |         111 |          88.774 |       -1.539 |       -0.724 |               -0.568 |              -0.315 |            -0.109 |               1.061 |           -1.874 |        -2.295 |
| pitcher_00003     |   2026 |         112 |          89.156 |       -0.705 |       -1.215 |               -0.48  |              -1.236 |            -1.218 |              -0.542 |           -1.33  |        -1.454 |
| pitcher_00941     |   2025 |         260 |          89.194 |       -1.319 |       -1.103 |                0.058 |              -1.074 |            -0.118 |              -0.418 |           -1.574 |        -1.766 |

## Chequeo de cordura

Los pitchers estan anonimizados (`pitcher_anon_id`), asi que este chequeo es de **consistencia interna de los componentes**, no de reputacion real de jugadores: ¿el pitcher #1 domina en la mayoria de los z-components (no solo en uno, arrastrado por un peso alto), y el ultimo lugar es consistentemente malo, no solo un outlier en una sola metrica con signo mal invertido?

- **#1 (pitcher_00469, 2026, Pitching+=116.5)**: 8/8 componentes con z positivo (favorable) — consistente en todos los frentes, no solo en las metricas de mayor peso.
- **Ultimo lugar (pitcher_01048, 2024, Pitching+=87.0)**: 7/8 componentes con z negativo (desfavorable) — consistentemente malo, no un outlier de una sola metrica.

Revisar las tablas de arriba: si algun pitcher del top-10 tiene un `z_run_value` o `z_is_hard_contact` MUY positivo mientras el resto de sus componentes son negativos, es señal de un posible error de signo (recordar: ambos se invierten antes de z-scorear) y debe investigarse antes de usar el score para el analisis contrafactual.
