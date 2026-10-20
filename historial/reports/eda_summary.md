# EDA — Resumen de hallazgos (Stuff+ LMB)

Dataset: `data/raw/stuff_model_df.parquet` — **635,002 pitches**, **1134 pitchers unicos**, temporadas [2024, 2025, 2026].

## 1. Cobertura y balance de datos

- `altitude_category` tiene 3020 valores nulos (0.48% del total), a excluir o imputar antes de modelar.
- **Extreme Altitude** (asumido el estadio de altitud extrema) representa **28.9%** de los pitches totales — desbalance esperado dado que es 1 de N estadios, pero suficiente volumen (183,267 pitches) para modelar su efecto con confianza.
- Distribucion por year: {2024: 243254, 2025: 256144, 2026: 135604}.
- `AutoPitchType` esta dominado por Four-Seam, Slider y Sinker; hay clases residuales con muestra casi nula (OneSeamFastBall, TwoSeamFastBall, Knuckleball con 1 pitch c/u, Sweeper con 588) que deben agruparse en 'Other' o excluirse del modelo por falta de muestra.

## 2. Estadisticas descriptivas

Ver tablas completas en `historial/reports/desc_stats_by_altitude.csv` y `historial/reports/desc_stats_by_pitchtype.csv` (mean, std, p1/p25/p50/p75/p99 de RelSpeed, SpinRate, InducedVertBreak, HorzBreak, Extension, RelHeight).

## 3. Efecto de altitud sobre movimiento (el estadio de altitud extrema vs resto)

Comparacion de medias por tipo de pitch (grupos con n>=200 en ambos lados):

| pitch_type   |   ivb_extreme |   ivb_resto |   hb_extreme |   hb_resto |   velo_extreme |   velo_resto |
|:-------------|-----------:|------------:|----------:|-----------:|------------:|-------------:|
| Four-Seam    |      13.07 |       16.08 |      7.22 |       8.77 |       91.98 |        91.68 |
| Slider       |       2.34 |        2.78 |      4.04 |       5.28 |       83.99 |        83.66 |
| Sinker       |       7.98 |        9.67 |     11.77 |      14.66 |       91.37 |        91.06 |
| Changeup     |       5.2  |        6.87 |      9.47 |      12.47 |       83.03 |        83.26 |


- El patron esperado por fisica de la altitud (menor densidad de aire -> menos movimiento vertical inducido y menos movimiento horizontal, mayor velocidad efectiva) debe leerse directamente de la tabla anterior antes de asumir causalidad: esto es correlacional, confunde altitud con plantilla de pitchers que juegan en el estadio de altitud extrema, y requiere el modelado causal (PyMC/econml) planeado en fases posteriores.
- Ver boxplots `reports/figures/02_box_inducedvertbreak_extreme_vs_resto.png` y `03_box_horzbreak_extreme_vs_resto.png` para la distribucion completa por pitch type.

## 4. Muestra por pitcher (shrinkage bayesiano)

- 1134 pitchers unicos, con una distribucion de pitches por pitcher muy sesgada a la derecha: mediana de 331 pitches, pero p10=22 y p1=8, contra un maximo de 4222.
- **309 pitchers (27.2%) tienen menos de 100 pitches registrados.** Esto confirma la necesidad de shrinkage bayesiano (partial pooling) hacia la media de la liga/tipo de pitch para evitar sobreajustar las metricas de pitchers con poca muestra.
- Ver histograma en `reports/figures/04_pitches_por_pitcher_hist.png`.

## 5. Distribucion de resultados del pitch

Ver `reports/figures/05_pct_pitchcall_hittype_playresult.png` y tablas impresas en el notebook. Puntos relevantes:
- `hit_type` es NaN en 71.0% de los casos (pitches que no resultan en bola en juego bateada), por lo que solo tiene sentido analizarlo condicionado a `is_ball_in_play`/`is_batted`.
- `PitchCall` esta dominado por BallCalled (35.6%), InPlay (18.3%) y FoulBallNotFieldable (16.4%).
- Eventos raros como HomeRun, Triple y Error en `play_result` tienen muestra muy baja y deben tratarse con cuidado en cualquier modelo run-value-based.

## Siguientes pasos sugeridos

- Confirmar con el equipo local que `Extreme Altitude` == el estadio de altitud extrema (asuncion usada en este EDA).
- Definir tratamiento de `altitude_category` NaN antes de features.
- Excluir o agrupar `AutoPitchType` con muestra insuficiente (<50 pitches).
- Diseñar el nivel de partial pooling (por pitcher x pitch type) para el modelo bayesiano, dado el fuerte sesgo en pitches-por-pitcher.
