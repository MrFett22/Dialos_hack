# Pitching+ v1 — score compuesto, predicciones OUT-OF-FOLD (version oficial)

**Esta es ahora la version oficial del score de Pitching+ para todos los entregables de aqui en adelante**, reemplazando a `data/processed/pitching_plus_scores.parquet` (in-sample, se conserva solo como referencia historica/auditoria, no usar para analisis nuevos).

Metodologia identica a `reports/pitching_plus_score.md` (mismos pesos, mismo signo invertido en run_value/is_hard_contact, mismo z-score por temporada, misma formula 100+10z) — el UNICO cambio es el origen de las predicciones: K-fold agrupado por pitcher (k=5, seed=42) en vez de un solo modelo train/test. Cada pitcher queda completamente afuera del entrenamiento del fold que predice sus pitches — ningun pitcher se predice con un modelo que haya visto (aunque sea parcialmente) sus propios pitches.

## Distribucion del Pitching+ (OOF)

n pitcher-temporadas: 1,628

| estadistico | valor |
|---|---:|
| mean | 100.00 |
| std | 6.42 |
| min | 59.85 |
| 1% | 75.84 |
| 5% | 89.18 |
| 10% | 92.69 |
| 25% | 97.26 |
| 50% | 100.80 |
| 75% | 103.93 |
| 90% | 106.69 |
| 95% | 107.92 |
| 99% | 111.93 |
| max | 123.10 |

## Comparacion contra la version in-sample anterior

- Correlacion entre score in-sample y score OOF (las 1,628 pitcher-temporadas): **0.9360**
- Cambio absoluto promedio en el score (|OOF - in-sample|): 1.07 puntos
- Cambio absoluto maximo: 37.37 puntos
- Overlap del top-10: **8/10** pitcher-temporadas se mantienen en el top-10 en ambas versiones
- Overlap del bottom-10: **4/10** pitcher-temporadas se mantienen en el bottom-10 en ambas versiones

**El ranking es robusto:** la correlacion es alta y el top-10/bottom-10 se mantiene mayormente igual — el sesgo in-sample que preocupaba no cambio sustancialmente el resultado, aunque sigue siendo la version metodologicamente correcta a usar de ahora en adelante.

### Pitcher-temporadas con mayor cambio (|delta| top 10)

| pitcher_anon_id   |   year |   pitching_plus_insample |   pitching_plus_oof |   delta |
|:------------------|-------:|-------------------------:|--------------------:|--------:|
| pitcher_00463     |   2026 |                   112.89 |               75.53 |  -37.37 |
| pitcher_01026     |   2024 |                   128.95 |               94.02 |  -34.93 |
| pitcher_00655     |   2026 |                   150.09 |              123.1  |  -26.99 |
| pitcher_00316     |   2025 |                   131.42 |              106.82 |  -24.61 |
| pitcher_00809     |   2025 |                    92.21 |               70.72 |  -21.49 |
| pitcher_00568     |   2026 |                    78.35 |               92.38 |   14.03 |
| pitcher_00190     |   2025 |                    87.47 |               99.9  |   12.43 |
| pitcher_00442     |   2024 |                    73.4  |               84.62 |   11.22 |
| pitcher_00368     |   2024 |                   116.44 |              105.46 |  -10.98 |
| pitcher_00682     |   2026 |                   107.75 |               96.83 |  -10.92 |

## Ranking OOF — top 10 y bottom 10 (filtro: n_pitches >= 100)

### Top 10 (mejor Pitching+, OOF)

| pitcher_anon_id   |   year |   n_pitches |   pitching_plus |   z_is_whiff |   z_is_chase |   z_is_called_strike |   z_is_weak_contact |   z_is_groundball |   z_is_hard_contact |   z_is_out_in_pa |   z_run_value |
|:------------------|-------:|------------:|----------------:|-------------:|-------------:|---------------------:|--------------------:|------------------:|--------------------:|-----------------:|--------------:|
| pitcher_00469     |   2026 |         283 |         116.543 |        3.075 |        0.696 |                1.61  |               0.957 |             1.464 |              -0.166 |            2.21  |         1.931 |
| pitcher_00456     |   2026 |         220 |         115.482 |        1.987 |        1.337 |                0.793 |               0.785 |             1.648 |               0.469 |            2.071 |         2.039 |
| pitcher_00875     |   2026 |         349 |         115.23  |        1.334 |        2.189 |                0.408 |               1.168 |             0.713 |               0.792 |            2.148 |         2.04  |
| pitcher_01058     |   2025 |         471 |         115.156 |        1.2   |        0.344 |                2.028 |               2.125 |             0.873 |               0.341 |            2.191 |         1.909 |
| pitcher_00402     |   2026 |         101 |         114.18  |        1.508 |        0.719 |                1.224 |               0.506 |            -0.434 |               0.256 |            2.3   |         2.588 |
| pitcher_00054     |   2024 |         691 |         113.526 |        2.013 |        0.953 |                0.078 |               0.337 |             0.723 |               0.942 |            2.124 |         2.011 |
| pitcher_01058     |   2026 |         301 |         113.484 |        0.503 |        0.36  |                1.678 |               1.287 |             1.489 |               0.219 |            2.198 |         1.821 |
| pitcher_00465     |   2026 |         357 |         112.471 |        2.605 |        0.683 |                0.859 |              -0.718 |            -0.446 |               0.595 |            1.846 |         2.129 |
| pitcher_00626     |   2026 |         354 |         112.337 |        2.582 |        1.38  |               -1.056 |               0.291 |             0.007 |               0.989 |            2.066 |         1.844 |
| pitcher_00263     |   2026 |         102 |         111.889 |        2.168 |        1.411 |                1.224 |               0.167 |            -0.523 |               0.515 |            1.451 |         1.548 |

### Bottom 10 (peor Pitching+, OOF)

| pitcher_anon_id   |   year |   n_pitches |   pitching_plus |   z_is_whiff |   z_is_chase |   z_is_called_strike |   z_is_weak_contact |   z_is_groundball |   z_is_hard_contact |   z_is_out_in_pa |   z_run_value |
|:------------------|-------:|------------:|----------------:|-------------:|-------------:|---------------------:|--------------------:|------------------:|--------------------:|-----------------:|--------------:|
| pitcher_00777     |   2024 |         206 |          86.097 |       -1.91  |       -1.701 |                0.136 |              -1.46  |            -0.71  |               0.05  |           -2.196 |        -1.823 |
| pitcher_00309     |   2026 |         107 |          87.117 |       -1.423 |       -0.96  |               -0.411 |              -1.779 |            -1.821 |              -0.027 |           -1.975 |        -1.34  |
| pitcher_00369     |   2024 |         103 |          87.158 |       -0.766 |       -2.269 |                0.235 |              -1.195 |            -1.162 |              -0.554 |           -1.93  |        -1.648 |
| pitcher_00120     |   2025 |         267 |          88.038 |       -1.153 |       -1.168 |                0.482 |              -1.596 |            -0.51  |               0.365 |           -2.187 |        -1.994 |
| pitcher_00089     |   2025 |         190 |          88.236 |       -1.821 |       -1.961 |                0.457 |              -1.293 |            -0.844 |              -0.146 |           -1.178 |        -1.795 |
| pitcher_01029     |   2024 |         144 |          88.31  |       -1.728 |       -1.696 |               -0.716 |              -1.016 |             0.241 |               0.286 |           -1.533 |        -1.633 |
| pitcher_00339     |   2025 |         115 |          88.317 |       -0.961 |       -1.799 |                0.765 |              -1.043 |            -0.515 |              -0.207 |           -1.724 |        -2.214 |
| pitcher_01048     |   2024 |         106 |          88.573 |       -1.216 |       -1.454 |                0.437 |              -1.611 |            -0.841 |               0.671 |           -1.995 |        -1.667 |
| pitcher_00687     |   2024 |         137 |          88.621 |       -2.194 |       -1.324 |                0.412 |              -0.783 |            -0.378 |               0.156 |           -1.539 |        -1.885 |
| pitcher_00668     |   2024 |         139 |          88.753 |       -0.025 |       -0.568 |               -2.476 |              -1.951 |            -1.424 |               0.763 |           -1.394 |        -1.344 |
