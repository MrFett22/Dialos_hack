# Reporte de arsenal — pitches_with_arsenal.parquet

## Pitchers-temporada sin fastball identificable

- Total de combinaciones pitcher-year: 1,718
- Sin Four-Seam ni Sinker en toda la temporada: **24** (1.4%)
- Pitches de esos 24 pitcher-year (quedan con velo_diff_vs_fb/ivb_diff_vs_fb/hb_diff_vs_fb/tunnel_similarity nulos): **440**

Aparte, hay **3008** pitches con `pitcher_anon_id` nulo en el dato crudo (no introducido por este pipeline) — para esos no hay ningun pitcher a quien buscarle fastball, asi que tambien quedan con estas 4 columnas nulas. El total de nulos en velo_diff_vs_fb/ivb_diff_vs_fb/hb_diff_vs_fb/tunnel_similarity (3448) es la suma de ambas causas.

Tamaño de muestra (n pitches en la temporada) de cada pitcher-year sin fastball — todos son casos de muestra muy chica, no especialistas sin fastball real:

- pitcher_00788 (2026): 68 pitches en la temporada
- pitcher_00368 (2024): 57 pitches en la temporada
- pitcher_00316 (2025): 33 pitches en la temporada
- pitcher_00228 (2025): 22 pitches en la temporada
- pitcher_00400 (2026): 21 pitches en la temporada
- pitcher_00947 (2025): 18 pitches en la temporada
- pitcher_01081 (2026): 18 pitches en la temporada
- pitcher_00022 (2025): 17 pitches en la temporada
- pitcher_00808 (2025): 17 pitches en la temporada
- pitcher_00556 (2025): 15 pitches en la temporada
- pitcher_00585 (2026): 15 pitches en la temporada
- pitcher_00491 (2025): 14 pitches en la temporada
- pitcher_00896 (2025): 13 pitches en la temporada
- pitcher_00639 (2025): 13 pitches en la temporada
- pitcher_00223 (2025): 13 pitches en la temporada
- pitcher_00007 (2025): 12 pitches en la temporada
- pitcher_00613 (2025): 12 pitches en la temporada
- pitcher_00249 (2025): 12 pitches en la temporada
- pitcher_00735 (2025): 11 pitches en la temporada
- pitcher_00272 (2025): 10 pitches en la temporada
- pitcher_00867 (2026): 10 pitches en la temporada
- pitcher_01026 (2024): 10 pitches en la temporada
- pitcher_00652 (2026): 6 pitches en la temporada
- pitcher_00655 (2026): 3 pitches en la temporada

## Nulos por columna nueva (todo el dataset)

- `velo_diff_vs_fb`: 3,448
- `ivb_diff_vs_fb`: 3,448
- `hb_diff_vs_fb`: 3,448
- `tunnel_similarity`: 3,448
- `pitch_usage_pct`: 3,008
- `prev_pitch_type`: 165,170

## tunnel_similarity: normalizacion usada

Desviacion estandar global (todo el dataset) por dimension, usada para convertir pies y grados a una escala comparable antes de la distancia euclidiana:

- `RelHeight`: std=0.4917
- `RelSide`: std=1.7513
- `VertApprAngle`: std=1.7421
- `HorzApprAngle`: std=2.0912

## Validacion manual: pitcher_00938, temporada 2025

Repertorio de esta temporada (conteo y % de uso):

| AutoPitchType   |   n |   pitch_usage_pct |
|:----------------|----:|------------------:|
| Four-Seam       | 945 |            0.4854 |
| Slider          | 389 |            0.1998 |
| Changeup        | 245 |            0.1258 |
| Sinker          | 216 |            0.1109 |
| Splitter        | 130 |            0.0668 |
| Curveball       |  20 |            0.0103 |
| Cutter          |   2 |            0.001  |

Promedios por tipo de pitch (para verificar que velo/ivb/hb_diff_vs_fb tengan sentido: el fastball deberia dar ~0, offspeed mas lento/con mas break deberia dar diffs negativos de velocidad y de break segun el pitch):

| AutoPitchType   |   RelSpeed |   velo_diff_vs_fb |   InducedVertBreak |   ivb_diff_vs_fb |   HorzBreak |   hb_diff_vs_fb |   tunnel_similarity |
|:----------------|-----------:|------------------:|-------------------:|-----------------:|------------:|----------------:|--------------------:|
| Changeup        |      83.34 |             -7.49 |               7.93 |            -7.69 |        4.32 |           -5.83 |                0.38 |
| Curveball       |      81.38 |             -9.45 |              -4.99 |           -20.62 |       -3.78 |          -13.93 |                0.34 |
| Cutter          |      87.22 |             -3.61 |              15.11 |            -0.52 |        1.11 |           -9.04 |                0.39 |
| Four-Seam       |      90.84 |              0.02 |              16.42 |             0.8  |        9.36 |           -0.79 |                0.59 |
| Sinker          |      90.75 |             -0.08 |              12.14 |            -3.48 |       13.61 |            3.45 |                0.58 |
| Slider          |      83.49 |             -7.34 |               1.22 |           -14.41 |       -2.77 |          -12.92 |                0.37 |
| Splitter        |      83.41 |             -7.41 |               6.71 |            -8.92 |        2.92 |           -7.23 |                0.37 |

Primeros 15 pitches (orden cronologico) con prev_pitch_type, para revisar manualmente el encadenamiento dentro de la plate appearance:

| PitchUID                             | AutoPitchType   | prev_pitch_type   |   RelSpeed |   velo_diff_vs_fb |   tunnel_similarity |
|:-------------------------------------|:----------------|:------------------|-----------:|------------------:|--------------------:|
| 1c9da3a0-4980-11f0-bfdb-6f84104f962d | Four-Seam       | nan               |     89.891 |            -0.932 |               0.612 |
| 2896fa30-4980-11f0-bfdb-6f84104f962d | Four-Seam       | Four-Seam         |     90.915 |             0.092 |               0.434 |
| 324ffdb0-4980-11f0-bfdb-6f84104f962d | Four-Seam       | Four-Seam         |     91.18  |             0.357 |               0.569 |
| 40718b70-4980-11f0-bfdb-6f84104f962d | Splitter        | Four-Seam         |     83.507 |            -7.316 |               0.418 |
| 4a9537a0-4980-11f0-bfdb-6f84104f962d | Changeup        | Splitter          |     82.837 |            -7.986 |               0.279 |
| 5a239bd0-4980-11f0-bfdb-6f84104f962d | Changeup        | Changeup          |     82.677 |            -8.146 |               0.402 |
| 68ffee10-4980-11f0-bfdb-6f84104f962d | Four-Seam       | Changeup          |     90.214 |            -0.609 |               0.461 |
| 7a2ee2e0-4980-11f0-bfdb-6f84104f962d | Four-Seam       | nan               |     91.335 |             0.512 |               0.411 |
| 833a1a30-4980-11f0-bfdb-6f84104f962d | Four-Seam       | Four-Seam         |     90.298 |            -0.525 |               0.683 |
| 8d39c3a0-4980-11f0-bfdb-6f84104f962d | Sinker          | Four-Seam         |     90.374 |            -0.449 |               0.549 |
| 9e027c90-4980-11f0-bfdb-6f84104f962d | Changeup        | Sinker            |     84.913 |            -5.91  |               0.34  |
| ac9c95b0-4980-11f0-bfdb-6f84104f962d | Splitter        | Changeup          |     82.389 |            -8.434 |               0.278 |
| be4cef80-4980-11f0-bfdb-6f84104f962d | Four-Seam       | Splitter          |     91.866 |             1.043 |               0.508 |
| c85e2520-4980-11f0-bfdb-6f84104f962d | Changeup        | Four-Seam         |     82.513 |            -8.31  |               0.256 |
| d84e7f70-4980-11f0-bfdb-6f84104f962d | Slider          | Changeup          |     83.444 |            -7.379 |               0.354 |
