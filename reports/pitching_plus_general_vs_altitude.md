# Pitching+ general vs. Pitching+ para ambientes de altitud

Generado por `src/analysis/pitch_environment.py` y `src/reports/pitch_environment_report.py`. No cambia ningún índice oficial:
Pitching+, Stuff+ Physical, Location+, AEPP y el ranking quedan igual. Es una extensión para explicar qué cambia en el valor de un
lanzamiento cuando cambia el costo de los resultados.

## 1. Qué es "normal" (decisión del baseline)

Se revisaron tres candidatos:

| candidato | qué es | ¿sirve como baseline? |
|---|---|---|
| Pitching+ simple ajustado por entorno | el índice oficial por pitcher: E[Δ] menos la media de su temporada × estadio anónimo | no: el ajuste resta el nivel de cada ambiente, pero NO cambia lo que vale cada resultado; compararlo con otra cosa mezclaría el ajuste con el costo de los resultados |
| valores RE24 de No Altitude | la recursión por conteo con solo los pitches de No Altitude | sirve como contraste extremo (se reporta), pero no es lo que usa la página |
| **índice general de liga** | **la recursión por conteo con TODOS los pitches: exactamente los Δ(evento, conteo) del Pitching+ oficial** | **sí: es el Pitching+ que el usuario ya ve en el simulador** |

**General = valores de toda la liga.** Se verificó contra el artefacto final: diferencia máxima en Δ(evento, conteo) = 0.0e+00; en E[Δ] por pitch = 0.0e+00.

**Pitching+ para Extreme = las mismas probabilidades del modelo** (mismo pitch, misma física, misma ubicación, mismo conteo) valuadas con lo que vale cada resultado en Extreme:

1. Valores terminales de Extreme: promedio del run value RE24 de cada resultado en los pitches de Extreme.
2. Recursión por conteo de Extreme: V(conteo) y Δ(evento, conteo) con las frecuencias de transición observadas en Extreme.
3. **Calibración del costo de contacto.** El modelo no ve el ambiente: su reparto out / sencillo / doble / triple / jonrón dada una bola en juego es el de la liga, y P2 §7.2 mostró que acierta P(BIP) pero **subestima el daño por bola en juego en altitud**. Por eso a la rama de bola en juego se le suma, en cada ambiente, la diferencia entre el Δ observado de sus bolas en juego y el que predice el modelo. Es un valor observado, no un peso elegido a mano.

No hay ningún "bono de altitud" manual. Escala: 100 + 10·(media − E[Δ])/sd con media y sd de todos los lanzamientos reales de la liga valuados con cada juego de valores, sin desplazamiento por ambiente, para que la única diferencia sea el costo de los resultados.

## 2. Lo que vale cada resultado

| resultado (valor terminal, carreras) | General | No Altitude | Extreme |
|---|---:|---:|---:|
| base por bolas | +0.378 | +0.378 | +0.385 |
| ponche | -0.334 | -0.326 | -0.347 |
| golpeado | +0.397 | +0.389 | +0.395 |
| out en juego | -0.307 | -0.304 | -0.309 |
| sencillo | +0.512 | +0.491 | +0.534 |
| doble | +0.836 | +0.810 | +0.851 |
| triple | +1.041 | +1.005 | +1.066 |
| jonrón | +1.472 | +1.438 | +1.490 |

| V(conteo): carreras esperadas del resto del turno | General | No Altitude | Extreme |
|---|---:|---:|---:|
| 0-0 | +0.015 | -0.015 | +0.047 |
| 0-2 | -0.107 | -0.127 | -0.083 |
| 2-0 | +0.128 | +0.104 | +0.159 |
| 3-2 | +0.075 | +0.064 | +0.097 |

| calibración del costo de contacto (Δ medio de una bola en juego) | predicho por el modelo | observado | ajuste |
|---|---:|---:|---:|
| General (no se ajusta; control) | +0.0458 | +0.0457 | +0.0000 |
| No Altitude | +0.0478 | +0.0352 | -0.0127 |
| Medium | +0.0433 | +0.0556 | +0.0123 |
| Extreme | +0.0421 | +0.0516 | +0.0095 |

**Lectura.** En Extreme casi todo resultado vale un poco más para la ofensiva, y **el turno completo empieza siendo más caro** (V(0-0) = +0.047 contra -0.015 en No Altitude). Eso es importante: un strike o una bola dejan al pitcher en un conteo que en Extreme también cuesta más.

## 3. ¿Cambia el valor de un mismo lanzamiento?

Sobre 627,185 lanzamientos reales:

- Correlación de rangos entre Pitching+ general y Pitching+ Extreme: **0.997** (No Altitude vs. Extreme: 0.994).
- Cambio absoluto mediano: **0.54 puntos** de Pitching+ (p90: 1.21); 14% cambia más de 1 punto y 0.1% más de 3.

El orden de los lanzamientos casi no cambia. Es coherente con P2: el ranking de pitchers con pesos de altitud y sin ellos correlaciona 0.986.

### 3.1 ¿Quién gana valor relativo? Comparando con el lanzamiento típico del MISMO conteo

Comparar contra el promedio de toda la liga mezcla conteos (con dos strikes hay más contacto y los valores de cada conteo cambian con el ambiente). Por eso la ganancia se mide frente al lanzamiento típico del mismo conteo: ganancia = −100·[(E[Δ] − media del conteo) en Extreme − (lo mismo) en el baseline], en carreras por 100 pitches (positivo = el pitch gana valor relativo en Extreme).

| fase | quintil de P(whiff) relativo | P(whiff) relativo | P(BIP) relativo | ganancia vs. General | ganancia vs. No Altitude |
|---|---|---:|---:|---:|---:|
| 0 o 1 strike | Q1 | -9.3 pp | -9.3 pp | -0.12 | -0.16 |
| 0 o 1 strike | Q2 | -5.5 pp | +1.4 pp | +0.01 | -0.00 |
| 0 o 1 strike | Q3 | -2.6 pp | +6.4 pp | +0.07 | +0.09 |
| 0 o 1 strike | Q4 | +2.0 pp | +4.3 pp | +0.05 | +0.07 |
| 0 o 1 strike | Q5 | +15.4 pp | -2.7 pp | -0.01 | +0.00 |
| dos strikes | Q1 | -11.2 pp | -14.4 pp | -0.40 | -0.56 |
| dos strikes | Q2 | -6.7 pp | +8.0 pp | -0.13 | -0.30 |
| dos strikes | Q3 | -2.5 pp | +9.2 pp | -0.03 | -0.15 |
| dos strikes | Q4 | +3.7 pp | +3.7 pp | +0.11 | +0.15 |
| dos strikes | Q5 | +16.7 pp | -6.6 pp | +0.46 | +0.85 |

- **Con dos strikes** (donde un whiff es ponche), la ganancia sigue al swing-and-miss: correlación con P(whiff) relativo +0.70 contra No Altitude (+0.70 contra General).
- **Con 0 o 1 strike**, el swing-and-miss casi no mueve el valor relativo (+0.08); la ganancia se asocia más bien con P(BIP) (+0.46). Antes de dos strikes, un pitch que no termina en contacto lleva a otro conteo, y en Extreme el resto del turno también es más caro: evitar el contacto temprano no rinde más.

**Conclusión.** El hallazgo de P1 (evitar una bola en juego vale ≈ 27% más por swing y cambiar una bola en juego por un ponche ≈ 27% más por turno) se ve en el valor por lanzamiento **donde el whiff termina el turno**: con dos strikes. Lanzamiento por lanzamiento, el Pitching+ para Extreme es casi el mismo que el general; la diferencia está en el margen y premia el swing-and-miss en el momento en que produce ponches.

### 3.2 Por tipo de pitch (promedios; Pitching+ en escala de lanzamiento)

| tipo | lanzamientos | Pitching+ general | Pitching+ Extreme | cambio |
|---|---:|---:|---:|---:|
| Changeup | 77,730 | 99.68 | 99.70 | +0.02 |
| Curveball | 53,269 | 100.56 | 100.64 | +0.08 |
| Cutter | 26,122 | 100.21 | 100.18 | -0.03 |
| Four-Seam | 215,235 | 99.63 | 99.58 | -0.05 |
| Sinker | 109,470 | 99.48 | 99.44 | -0.03 |
| Slider | 137,513 | 100.87 | 100.92 | +0.06 |
| Splitter | 7,212 | 101.05 | 101.23 | +0.18 |
| Sweeper | 571 | 100.55 | 100.74 | +0.19 |

Los promedios por tipo casi no cambian: el costo de los resultados no favorece en bloque a un tipo de pitch. (Los efectos por tipo de la sección 6 de la página son otra cosa: cambios observados de whiff, movimiento y valor realizado entre ambientes.)

## 4. Cómo se explica en la página

Para el lanzamiento del simulador se muestra la tabla General / Extreme (Stuff+, Location+, P(whiff), P(BIP), E[RV | BIP], Pitching+) y una explicación automática a partir de la descomposición Σ (p − p̄)·(Δ_Extreme − Δ_General) por grupo de resultados (bola en juego, ponche / strike, base por bolas, golpeado), donde p̄ es la probabilidad media de la liga en el mismo conteo. Stuff+ y Location+ no cambian: es el mismo lanzamiento. Lo que cambia es lo que vale cada resultado.

## 5. Lo que no decimos

- No es un efecto causal de la altitud: son los valores observados en esos ambientes (P1/P2).
- No reemplaza al Pitching+ oficial ni al AEPP: el AEPP mide al pitcher por turno y es la herramienta para elegir pitchers.
- La calibración del costo de contacto es un promedio por ambiente: no distingue estadios dentro de Extreme.
- Hereda las limitaciones del run value (por ejemplo, la reconstrucción de los outs de cada jugada).
