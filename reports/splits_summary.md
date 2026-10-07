# Reporte de splits — data/processed/splits/splits.parquet

## Filtro base (aplicado a los 3 esquemas por igual)

- Pitches totales en pitches_with_physics.parquet: 633,447
- Excluidos por `pitcher_anon_id` nulo: 3,008 (0.47%)
- Excluidos por `altitude_category` nulo: 3,019 (0.48%)
- (de los cuales 316 tienen AMBOS nulos — se cuentan una sola vez)
- Excluidos en total (union de ambos): 5,711 (0.90%) — ver reports/pitcher_id_null_investigation.md
- Pitches validos usados para los 3 splits: 627,736
- Pitchers unicos validos: 1,045

## 1. pitcher_holdout

- Pitchers en train: 837 | en test: 208 (19.9% de 1,045)
- Cuartiles de pitches-por-pitcher usados para estratificar: p25=129, p50=388, p75=806

Balance de altitude_category modal del pitcher (%), train vs test — deberia verse parecido si la estratificacion funciono:

| altitude_mode    |   test |   train |
|:-----------------|-------:|--------:|
| Extreme Altitude |  31.73 |   31.66 |
| Medium Altitude  |  21.63 |   21.74 |
| No Altitude      |  46.63 |   46.59 |

Balance de cuartil de volumen de pitches del pitcher (%), train vs test:

| pitch_count_quartile   |   test |   train |
|:-----------------------|-------:|--------:|
| Q1                     |     25 |   25.21 |
| Q2                     |     25 |   24.85 |
| Q3                     |     25 |   24.97 |
| Q4                     |     25 |   24.97 |

## 2. season_holdout (year=2026 a test)

- Pitches train: 497,596 | test: 130,140
- Pitchers train: 834 | test: 511

### Cobertura parcial de 2026 por el problema de anonimizacion

- Pitchers con datos validos en 2026: 511
- De esos, sin historial previo (solo aparecen en 2026): 211
- Con historial en 2024 y/o 2025 tambien: 300
- **Pitchers flageados con cobertura 2026 sospechosamente baja** (volumen 2026 < 20% de su promedio historico 2024-2025): **37**

Nota importante: un pitch con `pitcher_anon_id` nulo NO se puede reasignar a un pitcher especifico a posteriori — ya esta excluido de todos los splits (ver filtro base arriba). Lo que se reporta aqui es un sintoma indirecto: pitchers con historial 2024/2025 solido cuyo conteo 2026 cae muy por debajo de lo esperado, consistente con haber perdido parte de sus apariciones de 2026 en alguno de los 34 juegos con el problema de anonimizacion. **No se excluyen de season_holdout** (sus pitches validos restantes son datos reales, no ruido), pero su senal en 2026 debe tratarse con mas cautela. Como season_holdout parte estrictamente por `year`, todos los pitches validos de 2026 de un pitcher quedan enteros en test (no hay fragmentacion adicional dentro del split).

Top 10 pitchers mas flageados (2026 vs. promedio historico):

| pitcher_anon_id | ratio 2026/historico |
|---|---:|
| pitcher_01117 | 0.024 |
| pitcher_00450 | 0.033 |
| pitcher_00558 | 0.037 |
| pitcher_00774 | 0.040 |
| pitcher_00647 | 0.041 |
| pitcher_00163 | 0.045 |
| pitcher_00071 | 0.054 |
| pitcher_01059 | 0.062 |
| pitcher_00279 | 0.064 |
| pitcher_01038 | 0.065 |

## 3. altitude_holdout (Extreme Altitude a test)

- Pitches train (No Altitude + Medium Altitude): 446,932
- Pitches test (Extreme Altitude): 180,804
- Excluidos por altitude_category nulo: 0
- Pitchers train: 972 | test: 891

Nota: un pitcher puede aparecer en ambos lados de este split (lanza tanto en Extreme Altitude como en otros estadios) — es esperado, altitude_holdout particiona por PITCH, no por pitcher.

## Reproducibilidad

Seed fijo: `42`. Splits guardados en `data/processed/splits/splits.parquet` (una fila por PitchUID, columnas pitcher_holdout/season_holdout/altitude_holdout con valores train/test/excluded).
