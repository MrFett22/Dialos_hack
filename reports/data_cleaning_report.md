# Reporte de limpieza — pitches_clean.parquet

Filas de entrada: **635,002**  
Filas de salida: **633,447**  
Filas eliminadas en total: **1,555**

## 1. Nulos en stuff_feature criticas

Filas eliminadas por tener al menos un nulo en RelSpeed, SpinRate, InducedVertBreak, HorzBreak, Extension, RelHeight, RelSide: **1,555**

Nulos por columna (antes de eliminar):

- `RelSpeed`: 0
- `SpinRate`: 80
- `InducedVertBreak`: 702
- `HorzBreak`: 702
- `Extension`: 773
- `RelHeight`: 0
- `RelSide`: 0

## 2. Estandarizacion de `count` a partir de Balls/Strikes

- Filas con `count` inconsistente respecto a Balls-Strikes, corregidas: **0**
- Filas con Balls o Strikes nulos (no se pudieron validar/corregir): 0

## 3. Unicidad de PitchUID

- PitchUID nulos: 0
- Filas involucradas en algun PitchUID duplicado: 0
- De esas, duplicados exactos (fila identica) eliminados: 0
- Duplicados con datos en conflicto (NO eliminados, requieren revision manual): 0

## 4. Validacion de altitude_category

- Valores inesperados encontrados: ninguno
- Filas con valor inesperado: 0
- Filas con altitude_category nulo (quedan con altitude_level nulo, no se eliminan): 3,019

## 5. Columna ordinal `altitude_level`

Creada a partir de altitude_category: `No Altitude` -> 0, `Medium Altitude` -> 1, `Extreme Altitude` -> 2, valores nulos o inesperados -> nulo (dtype `Int8`, usable como categorica u ordinal).

## 6. Nulos en columnas clave (IDs de agrupacion + altitude_category)

No se eliminan filas aqui — distintos sub-modelos/splits toleran distinto nivel de nulo en estas columnas (ej. Location+ no depende de pitcher_anon_id, pero Stuff+/Pitching+ si; altitude_category la necesitan los 3, ver Stuff+ paso 3.2). La decision de exclusion se documenta en `reports/pitcher_id_null_investigation.md` y se aplica rio abajo, en `src/validation/splits.py`. `altitude_category` se agrego a esta validacion despues de que su nulo (2,703 filas con pitcher valido, un problema DISTINTO al de pitcher_anon_id) se descubriera recien al construir los splits, no en la limpieza original — de ahi la necesidad de este chequeo explicito para no repetir el mismo descuido con otra columna.

- `game_anon_id`: 0 nulos (0.00% del dataset limpio)
- `pitcher_anon_id`: 3,008 nulos (0.47% del dataset limpio)
- `batter_anon_id`: 2,013 nulos (0.32% del dataset limpio)
- `catcher_anon_id`: 2,317 nulos (0.37% del dataset limpio)
- `altitude_category`: 3,019 nulos (0.48% del dataset limpio)
