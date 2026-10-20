# Reporte de targets — pitches_with_targets.parquet

## 1. Validacion de `is_whiff` (columna existente)

- Nulos: 0
- Filas con is_whiff==1: 71,742
- Coincide con is_swinging_strike en 100% de las filas
- Inconsistencias is_whiff==1 pero is_swing==0: 0
- **Inconsistencias is_whiff==1 y is_contact==1 al mismo tiempo (contradiccion logica): 447** — revisar con el organizador antes de usarla sin filtrar.

## 3. Validacion de `is_called_strike` (columna existente)

- Nulos: 0
- Filas con is_called_strike==1: 97,579
- Discrepancias vs PitchCall=='StrikeCalled': 0

## 3b. `is_hit_by_pitch` (reconstruida desde PitchCall)

- La columna original vale 1 en 0 filas (está vacía en el dataset).
- Reconstruida como PitchCall == 'HitByPitch': 2,276 golpeados (2,254 con play_result == 'HitByPitch'; 22 marcados además como base por bolas).
- Cierra el turno y pone al bateador en primera (avance forzado, igual que la base por bolas).

## 2, 4, 5, 6. Targets derivados

- `is_chase` = swung_outside_strike_zone (copia directa).
- `is_weak_contact`: 1 si ExitSpeed < 85.0 mph e is_ball_in_play==1; nulo si is_ball_in_play==1 pero ExitSpeed no se midio.
- `is_groundball`: 1 si hit_type == 'GroundBall'.
- `is_hard_contact` (barrel): ExitSpeed >= 98.0 mph con Angle dentro del rango [26.0,30.0] grados a 98.0 mph, ensanchando linealmente hasta [8.0,50.0] grados a 116.0+ mph (aproximacion estandar del Barrel de Statcast; MLB no publica la formula exacta).

## 7. `run_value` (matriz de 24 estados base-out)

- Plate appearances reconstruidas: 164,289 en 36,995 medias-entradas.
- Pitches sin estado RE valido (bordes de datos / Outs invalido): 3,092
- Metodologia: se ordenan los pitches cronologicamente usando el timestamp embebido en PitchUID (UUID version 1), se agrupan en plate appearances y se simula la ocupacion de bases con heuristicas de avance estandar (ver docstring del modulo). RunsScored y OutsOnPlay reales se usan como fuente de verdad; solo la ocupacion de bases es aproximada.
- run_value es 0 para pitches que no terminan la plate appearance (el estado base-out no cambia hasta el ultimo pitch).

Matriz de run expectancy (estado = corredor en 1B, 2B, 3B, outs):

| on1 | on2 | on3 | outs | RE |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.632 |
| 0 | 0 | 0 | 1 | 0.356 |
| 0 | 0 | 0 | 2 | 0.15 |
| 0 | 0 | 0 | 3 | 0.0 |
| 0 | 0 | 1 | 0 | 1.517 |
| 0 | 0 | 1 | 1 | 0.906 |
| 0 | 0 | 1 | 2 | 0.316 |
| 0 | 0 | 1 | 3 | 0.0 |
| 0 | 1 | 0 | 0 | 1.294 |
| 0 | 1 | 0 | 1 | 0.879 |
| 0 | 1 | 0 | 2 | 0.396 |
| 0 | 1 | 0 | 3 | 0.0 |
| 0 | 1 | 1 | 0 | 2.139 |
| 0 | 1 | 1 | 1 | 1.405 |
| 0 | 1 | 1 | 2 | 0.59 |
| 0 | 1 | 1 | 3 | 0.0 |
| 1 | 0 | 0 | 0 | 1.072 |
| 1 | 0 | 0 | 1 | 0.69 |
| 1 | 0 | 0 | 2 | 0.304 |
| 1 | 0 | 0 | 3 | 0.0 |
| 1 | 0 | 1 | 0 | 1.931 |
| 1 | 0 | 1 | 1 | 1.281 |
| 1 | 0 | 1 | 2 | 0.485 |
| 1 | 0 | 1 | 3 | 0.0 |
| 1 | 1 | 0 | 0 | 1.787 |
| 1 | 1 | 0 | 1 | 1.166 |
| 1 | 1 | 0 | 2 | 0.512 |
| 1 | 1 | 0 | 3 | 0.0 |
| 1 | 1 | 1 | 0 | 2.6 |
| 1 | 1 | 1 | 1 | 1.651 |
| 1 | 1 | 1 | 2 | 0.763 |
| 1 | 1 | 1 | 3 | 0.0 |

## 8. `is_out_in_pa`

1 si la plate appearance termino en strikeout o en bola en juego que no fue hit (is_ball_in_play==1 e is_hit==0); 0 si termino en hit, base por bolas o hit by pitch. Se transmite (broadcast) a todos los pitches de la PA. **Limitacion:** no distingue embasarse por error de embasarse por fielder's choice; ambos casos quedan como 'out'.

## Verificacion de fuga de informacion (features vs targets)

- Columnas `stuff_feature` que tambien son `target_only` en el diccionario: ninguna
- Columnas `stuff_feature` que coinciden con los nuevos targets creados aqui: ninguna
- Columnas `stuff_feature` que en realidad son datos de resultado/ubicacion (PlateLoc*, in/outside_strike_zone, target_only): ninguna

Correlaciones |r| >= 0.30 entre una stuff_feature numerica y un target binario (a revisar manualmente, no implican fuga por si solas):

Ninguna encontrada.

## Distribucion (proporcion de 1s) por altitude_category

|                  |   Extreme Altitude |   Medium Altitude |   No Altitude |    nan |
|:-----------------|-------------------:|------------------:|--------------:|-------:|
| is_whiff         |             0.108  |            0.1124 |        0.117  | 0.1047 |
| is_chase         |             0.2171 |            0.2197 |        0.2268 | 0.1997 |
| is_called_strike |             0.1507 |            0.1501 |        0.1583 | 0.1484 |
| is_weak_contact  |             0.0811 |            0.0763 |        0.0792 | 0.0847 |
| is_groundball    |             0.1093 |            0.1012 |        0.0948 | 0.1126 |
| is_hard_contact  |             0.005  |            0.0052 |        0.0043 | 0.0033 |
| is_out_in_pa     |             0.6007 |            0.6152 |        0.6531 | 0.5745 |

## `run_value` (solo pitches terminales de PA) por altitude_category

| altitude_category   |    mean |    std |   count |
|:--------------------|--------:|-------:|--------:|
| Extreme Altitude    |  0.0472 | 0.5696 |   47965 |
| Medium Altitude     |  0.0326 | 0.5599 |   40021 |
| No Altitude         | -0.0153 | 0.5144 |   75538 |
| nan                 |  0.06   | 0.5927 |     765 |
