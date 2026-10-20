# AUDITORÍA INICIAL — CDMX Como Variable

**Fecha:** 2026-10-05 · **Commit auditado:** `db1eed6` (rama `main`)

**Alcance.** Inspección del proyecto, reproducción de resultados y auditoría matemática, estadística, de leakage, de unidades y de signos. **No se modificó código del proyecto.** Los scripts de verificación están en el scratchpad de la sesión (`audit_*.py`), fuera del repositorio.

**Documento complementario:** [reports/auditoria_matematica.md](reports/auditoria_matematica.md). Es la auditoría matemática detallada (tablas componente por componente, 10 tests numéricos, experimento de leakage temporal).

**Convención de este documento:**
- ✅ = verificado con código y datos en esta auditoría;
- ⚠️ = verificado, con problemas;
- ❌ = no se cumple o no existe;
- 🔎 = hipótesis o pendiente de validación.

Ninguna cifra de este documento es inventada: todas salen de correr el código actual sobre los datos actuales.

---

## Resumen ejecutivo (lo que hay que saber primero)

1. **"Extreme Altitude" NO es el estadio de altitud extrema.** El dataset no trae estadio, equipo, clima ni elevación.
   - Reconstruí los equipos anónimos a partir de qué pitchers lanzan juntos del mismo lado.
   - Resultado: **5–6 equipos distintos juegan de local en "Extreme"** (§15).
   - Esos estadios tienen **densidades efectivas distintas**: ratio de drag de 0.81 a 0.90 contra No Altitude, con intervalos que no se traslapan.
   - Todo resultado actual de "Extreme" es un **promedio de varios estadios de altura**.
2. **La maquinaria matemática está bien implementada:**
   - la cadena de probabilidades suma 1;
   - la recursión del conteo es correcta;
   - la física reproduce la trayectoria;
   - el shrinkage se comporta como debe.
3. **Hay un error de datos con impacto real.** El golpeado (HBP) vale ≈ 0 carreras, porque `targets.py` usa una columna que siempre vale 0. Mueve hasta 10 puntos de Pitching+ a algunos pitchers.
4. **La conclusión predictiva principal debe matizarse.** El índice neutral predice mejor las carreras futuras ajustadas por entorno, y eso sobrevive al control de leakage temporal. Pero un ajuste trivial por categoría de altitud, sin física, logra lo mismo.
5. **La hipótesis del analista del equipo local es comprobable con estos datos.** Un análisis descriptivo preliminar es consistente con ella (§6.4), pero todavía no está demostrada.
6. **No existe página web.** Existe un dashboard Streamlit de 2 pestañas. Faltan, entre otros: modelo de K, Location+ integrado, sección de bateadores, sensibilidad por pitcher validada e identificación del equipo local.

---

## 1. Estructura del proyecto

```
Hackathon_diablos/
├── data/
│   ├── raw/stuff_model_df.parquet          ← dataset original (635,002 pitches, 90 columnas)
│   ├── raw/stuff_plus_data_dictionary.csv  ← diccionario oficial del hackathon
│   └── processed/                          ← 26 parquet intermedios (limpieza → targets → física → modelos)
├── src/
│   ├── data/clean.py                       ← limpieza
│   ├── features/  targets.py · physics.py · arsenal.py · trajectory.py
│   ├── models/    stuff_plus*.py · pitching_plus*.py · run_value_chain.py · location_plus.py · tuning.py
│   │              bayesian_hierarchical.py · causal_forest.py · shap_analysis.py   (legado v1)
│   ├── analysis/  altitude_counterfactual*.py · runs_translation.py · altitude_*   (v1 y v2)
│   ├── validation/ benchmark.py · splits.py
│   └── reports/   scouting_report_v2.py
├── dashboard/  app.py · scoring.py          ← Streamlit (2 pestañas)
├── tests/      test_trajectory.py           ← 6 pruebas (pasan)
├── notebooks/  01_eda.ipynb
├── reports/    ~35 reportes .md, figuras, benchmark/*.json, scouting/
├── README.md · pyproject.toml · requirements.txt · requirements-lock.txt
└── .venv (Python 3.14)
```

**Dependencias.**
- Fijadas en `requirements-lock.txt`: lightgbm 4.7, pandas, numpy 2.5, scikit-learn, optuna, pygam, pymc/arviz, econml, streamlit, plotly, matplotlib.
- ⚠️ **Restricción del entorno:** el control de aplicaciones de Windows bloquea las DLL de `sklearn.cluster`. Los clustering de esta auditoría se hicieron con `scipy`.
- `networkx` no está instalado.

## 2. Archivos importantes (pipeline vigente, v3)

| orden | archivo | entra | sale | estado |
|---|---|---|---|---|
| 1 | `src/data/clean.py` | raw | `pitches_clean.parquet` (633,447) | ✅ |
| 2 | `src/features/targets.py` | clean | `pitches_with_targets.parquet` (targets + run_value RE24 reconstruido) | ⚠️ error HBP (§11) |
| 3 | `src/features/physics.py`, `arsenal.py` | targets | physics/arsenal parquet | ✅ |
| 4 | `src/validation/splits.py` | physics | `splits.parquet` (excluye pitcher o altitud nulos → 627,736) | ✅ |
| 5 | `src/validation/benchmark.py` | todo lo anterior | benchmark año-contra-año, Stuff+ v1 OOF | ✅ |
| 6 | `src/models/stuff_plus_v2.py` | physics + arsenal | features v2 (espejo, VAA/HAA ajustados) | ✅ |
| 7 | `src/models/run_value_chain.py` | v2 + targets | cadena de 5 modelos, V(b,s), Δ, Pitching+ v3 | ✅ (salvo HBP) |
| 8 | `src/features/trajectory.py` | 9 parámetros Trackman | drag/Magnus, ρ̂, contrafactual | ✅ |
| 9 | `src/analysis/altitude_counterfactual_v2.py` | 7 + 8 | efecto de altitud por pitch | ✅ (modelos in-sample) |
| 10 | `src/models/pitching_plus_neutral.py` | 7 + 8 | **índice oficial** `pitching_plus_official.parquet` | ✅ / ⚠️ interpretación |
| 11 | `src/models/stuff_plus_v4.py` | 7 + 8 | P(whiff \| swing) neutral (no adoptado) | ✅ |
| 12 | `src/models/tuning.py` | 7 | hiperparámetros (activos) | ✅ |
| 13 | `src/reports/scouting_report_v2.py` | 10 | fichas | ✅ (heredan el error HBP) |
| 14 | `historial/dashboard/app.py`, `scoring.py` | artefactos | Streamlit | ✅ |

**Legado v1 (no forma parte del índice vigente):** `pitching_plus.py`, `pitching_plus_oof.py`, `pitching_plus_score.py`, `stuff_plus.py` (sigue siendo la base de Stuff+ v1 oficial), `location_plus.py` (GAM), `bayesian_hierarchical.py`, `causal_forest.py`, `shap_analysis.py`, `altitude_counterfactual.py`, `altitude_counterfactual_ood.py`, `runs_translation.py`, `altitude_whiff_stratified.py`, `altitude_h2_changeup.py`.

## 3. Dataset utilizado (verificado)

| nivel | pitches | pitchers | nota |
|---|---:|---:|---|
| raw | 635,002 | 1,134 | 2,127 juegos, 864 bateadores. 2024: 243,254 · 2025: 256,144 · 2026: 135,604 (temporada parcial) |
| limpio / targets | 633,447 | 1,134 | |
| splits válidos (sin pitcher ni altitud nulos) | **627,736** | **1,045** | ✅ coincide con lo documentado |
| cadena v3 (PitchCall clasificable, conteo válido) | 627,185 | 1,045 | 840 PitchCall "Undefined" excluidos |
| tabla oficial | 1,628 pitcher-temporadas | 1,045 | 1,278 con ≥100 pitches |

- **Fechas:** 2024-04-13 → 2026-06-13. Se infieren del timestamp del `PitchUID` (UUID v1; 325 PitchUID son v4, sin fecha).
- **Valores faltantes:** pitcher nulo en 3,272 filas; altitud nula en 3,020.
- **Altitud por juego:** constante dentro de cada juego (0 juegos con más de una categoría).

| juegos | Extreme | Medium | No Altitude |
|---|---:|---:|---:|
| 2024 | 223 | 204 | 406 |
| 2025 | 237 | 213 | 386 |
| 2026 | 144 | 92 | 212 |

## 4. Variables disponibles (verificado en `stuff_model_df.parquet`)

| grupo | variables | cobertura / nota |
|---|---|---|
| Identificadores | `PitchUID`, `game_anon_id`, `pitcher_anon_id`, `batter_anon_id`, `catcher_anon_id` | anónimos. **Sin equipo, sin estadio** |
| Contexto | `year`, `altitude_category` (3 niveles), `Inning`, `Top/Bottom`, `Outs`, `Balls`, `Strikes`, `count`, `PitcherThrows`, `BatterSide` | sin fecha explícita (se infiere del UUID), sin clima, sin elevación |
| Pitch | `AutoPitchType`, `RelSpeed`, `EffectiveVelo`, `ZoneSpeed`, `SpinRate`, `SpinAxis`, `Tilt`, `RelHeight`, `RelSide`, `Extension`, `VertBreak`, `InducedVertBreak`, `HorzBreak`, `VertRelAngle`, `HorzRelAngle`, `VertApprAngle`, `HorzApprAngle`, `SpeedDrop`, `ZoneTime`, `pfxx`, `pfxz` | ~100% |
| Trayectoria | `x0..az0` (9 parámetros en y = 50 ft), `PitchTrajectory{X,Y,Z}c{0,1,2}` (polinomios) | ~100% |
| Ubicación | `PlateLocHeight`, `PlateLocSide`, `in_strike_zone`, `outside_strike_zone`, `swung_outside_strike_zone` | ~100% |
| Resultado del pitch | `PitchCall`, `KorBB`, `play_result`, `is_swing`, `is_whiff`, `is_contact`, `is_called_strike`, … | `is_hit_by_pitch` **= 0 en todo el dataset** (inconsistente con PitchCall) |
| Contacto | `ExitSpeed`, `Angle`, `Direction` (93.3% de las bolas en juego), `Distance` (88.7%), `hit_type` (94.4%) | **sí permite estudiar calidad del contacto** |
| Resultado de la jugada | `single`, `double`, `triple`, `home_run`, `RunsScored`, `OutsOnPlay` | 100% |
| **No existen** | estadio, equipo, local/visitante explícito, elevación, dimensiones, temperatura, humedad, presión, viento, umpire, corredores en base | corredores reconstruidos por heurística en `targets.py` |

## 5. Modelos existentes

| modelo | qué estima | universo | features | validación | estado |
|---|---|---|---|---|---|
| **Stuff+ v1 (oficial)** | P(whiff) por pitch → 100 + 10·z por temporada | todos los pitches | 20 numéricas: física, **ángulos de salida y llegada**, **`altitude_level`**, residuales de Magnus, mano/lado | OOF por pitcher (k = 5) | ⚠️ incluye ubicación **y altitud** |
| Stuff+ v2a–d | igual, con features corregidas | todos | espejo, VAA/HAA ajustados, arsenal | OOF | diagnóstico |
| Stuff+ v3 | cadena sin ubicación ni conteo | todos | v2d | OOF | descartado |
| Stuff+ v4 | P(whiff \| swing), neutral, con shrinkage | entrena con swings | v2d | OOF | no adoptado; se usa dentro de cada tipo de pitch en fichas |
| **Pitching+ v3 neutral (oficial)** | E[Δ carreras] con 5 LightGBM condicionales, proyectado a No Altitude, shrinkage EB | todos | v2d + PlateLoc + conteo | OOF por pitcher, benchmark año-contra-año | ✅ / ⚠️ |
| Location+ v1 | GAM: P(strike cantado), P(chase) por ubicación + conteo | pitcher_holdout | PlateLoc, conteo, lado, tipo | holdout | legado, **no integrado** al índice vigente |
| Pitching+ v1 | 8 modelos con pesos a mano | | | OOF | reemplazado |
| Bayesiano jerárquico, causal forest, SHAP | efecto de altitud (v1) | | | | legado |
| **Modelo de K** | — | — | — | — | ❌ no existe |
| **Contacto / hard contact por altitud** | — | — | — | — | ❌ no existe como modelo (`is_hard_contact` existe como target v1) |

## 6. Resultados existentes y su reproducción

### 6.1 Se reproducen exactamente ✅

| resultado | documentado | reproducido |
|---|---|---|
| Stuff+ v1, AUC OOF | 0.732 | **0.7315** (627,736 pitches) |
| Stuff+ v2d, AUC | 0.640 | **0.6396** |
| Stuff+ v4, AUC sobre swings | 0.651 | **0.6507** (293,559 swings) |
| V(0,0) | +0.0142 | **+0.01417** |
| ρ̂ drag/Magnus Medium | 0.906 / 0.834 | **0.9059 / 0.8340** |
| ρ̂ drag/Magnus Extreme | 0.854 / 0.770 | **0.8536 / 0.7701** |
| corr(score con altitud, %Extreme) | −0.327 | **−0.3274** |
| corr(score neutral oficial, %Extreme) | −0.054 | **−0.0541** |
| r_pred carreras ajustadas t+1, mín. 300: con altitud → neutral | 0.208 → 0.276 | **0.2084 → 0.2756** |
| Δr, P(Δr > 0) | +0.067, 99% | **+0.0672, 98.7%** |
| τ por temporada | 0.0056–0.0063 | **0.00626 / 0.00561 / 0.00627** |
| n½ | 65–85 | **66 / 84 / 66** |

Las cifras que circulaban como "−0.333 → −0.074" y "0.208 → 0.267, Δ = +0.059" son las **previas al ajuste de hiperparámetros** (bitácora §7.7). Las vigentes son las de esta tabla.

### 6.2 Se reproducen, pero la interpretación no se sostiene ⚠️

- **"La neutralización física mejora la predicción."** Un control trivial resta a E[Δ] observado su media por (temporada, categoría), sin física. Obtiene 0.2747 contra 0.2756 del índice oficial (P = 54%), y con mín. 100 le gana al oficial (0.174 vs. 0.164).
- **"Cumple el criterio de §7.2."** El criterio pre-registrado (carreras observadas t+1, mín. 100, contra v1) no se cumple: neutral 0.134 vs. v1 0.169, P = 21%.
- **"Techo ≈ √r_stab."** No es una cota general (ver §14).

### 6.3 Leakage temporal (experimento nuevo) ✅

Reentrené todo (ubicación, ρ̂, valores por conteo, modelos) solo con temporadas ≤ t:

| | r con mín. 300 | Δr vs. con altitud | P |
|---|---:|---:|---:|
| pipeline oficial | 0.2756 | +0.0672 | 98.7% |
| temporalmente limpio | 0.2646 | +0.0611 | 98.6% |

El leakage existe, pero es pequeño (≈ 0.01) y la conclusión relativa sobrevive.

### 6.4 Primer vistazo a la hipótesis del analista 🔎 PRELIMINAR

Comparación descriptiva por categoría. Sin controles por pitcher ni bateador, y con el error HBP sin corregir.

| categoría | pitches | carreras por bola en juego | H/BIP | HR/BIP | % EV ≥ 95 mph | whiff/swing | Δ(whiff, 0-2) | Δ medio bola en juego (0-2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Extreme | 180,794 | +0.090 | 0.387 | 0.0425 | 24.4% | 0.232 | −0.258 | +0.172 |
| Medium | 154,048 | +0.078 | 0.379 | 0.0425 | 24.5% | 0.241 | −0.243 | +0.168 |
| No Altitude | 292,343 | +0.019 | 0.331 | 0.0365 | 20.7% | 0.249 | −0.193 | +0.146 |

**Lectura (con cautela):**
- En las categorías de altura, la bola en juego vale más carreras.
- La diferencia en carreras entre un whiff y una bola en juego con 0-2 es mayor en Extreme (0.43) que en No Altitude (0.34).
- Esto es **consistente** con la hipótesis, pero **no la demuestra**. Confunde altitud con equipos, dimensiones y calibración de Trackman por estadio, y mezcla estadios distintos.

## 7. Qué resultados pueden reproducirse

Todo el pipeline v3 corre de punta a punta con los comandos de la bitácora §7.11, y `pytest` pasa 6/6. Todas las cifras de §6.1 se reproducen al dígito.

## 8. Qué resultados no pueden reproducirse o no deben mostrarse todavía

| resultado | por qué |
|---|---|
| Cualquier afirmación "en el estadio de altitud extrema" | El dataset no identifica el estadio; Extreme son 5–6 estadios (§15) |
| Rankings y fichas de pitchers con nombres o percentiles | Error HBP (hasta 10 puntos) |
| "La física/neutralización mejora la predicción" | Un control trivial empata (§6.2) |
| "Calidad del pitch" del efecto de altitud (+0.41/100 en Four-Seam) | Omite el término de tasa de contacto (subestima ~14%) |
| Efecto de altitud con IC 95% | Calculado con modelos in-sample; el IC solo refleja el muestreo de pitchers |
| "Elevación equivalente ≈ 1,600 m" (documento técnico) | Convierte un ratio de **drag** en densidad, cuando drag y Magnus dan ratios distintos, y promedia estadios distintos |
| Cifras de EDA tituladas "el estadio de altitud extrema vs resto" (`eda_summary.md`) | Suponen Extreme = el estadio de altitud extrema |

## 9. Partes del nuevo proyecto que ya existen

| pieza del proyecto "CDMX Como Variable" | existe | dónde | estado |
|---|---|---|---|
| Física de trayectoria, gravedad, drag/Magnus efectivos | ✅ | `trajectory.py` | validada |
| Densidad efectiva medida (ρ̂) | ✅ | `estimate_density_ratios` | válida, pero por categoría, no por estadio |
| Contrafactual (mismo pitch, otra altitud, mismo objetivo) | ✅ | `project_to_altitude` | validado; split-half sobrepredice el cambio de IVB ~20% |
| Stuff+ | ✅ | v1 oficial; v4 | ver §5 |
| Location+ | ⚠️ | `location_plus.py` (GAM v1) | no integrado; targets distintos (strike cantado, chase) |
| Pitching+ con cadena de probabilidades y valor esperado | ✅ | `run_value_chain.py` | correcto salvo HBP |
| Score neutral con shrinkage e intervalo | ✅ | `pitching_plus_neutral.py` | correcto; renombrar el intervalo |
| Benchmark de estabilidad/predicción con bootstrap | ✅ | `benchmark.py` | correcto |
| Costo de altitud por pitcher (E − N) | ⚠️ | `costo_altitud_total_por_100` | in-sample; es **sensibilidad implicada por el modelo**, no observada |
| Efecto de altitud por tipo de pitch | ✅ | `altitude_counterfactual_v2.md` | ver §8 |
| Dashboard interactivo | ⚠️ | Streamlit, 2 pestañas (pitch hipotético / pitcher) | no es la página pedida |
| Fichas de scouting | ✅ | `reports/scouting/` | heredan HBP |
| Heatmaps de ubicación | ⚠️ | `reports/figures/location_heatmaps` | de Location+ v1 |

## 10. Partes que faltan

1. **Identificación de estadios** (al menos reconstruida) y separación de los estadios dentro de Extreme.
2. **Cadena de altitud por niveles:** física → pitch → bateador → contacto → juego, cada eslabón con su prueba (hoy el proyecto salta de física a E[Δ]).
3. **Modelo de whiff interpretable** (qué características lo aumentan), con efectos por tipo de pitch y altitud.
4. **Modelo o análisis de K** a nivel turno (universo y variables documentados; K no es solo del pitcher).
5. **Contacto y calidad del contacto** (EV, ángulo, hard contact) por altitud, controlando por pitcher y bateador.
6. **Prueba formal de la hipótesis del analista** (valor relativo del whiff contra el contacto por entorno).
7. **Sensibilidad a la altitud por pitcher validada empíricamente**, no solo la implicada por el modelo, con su incertidumbre.
8. **Gráfico calidad neutral vs. sensibilidad** (cuadrantes).
9. **Sección el equipo local:** no es posible sin identificar al equipo (§15).
10. **Sección de bateadores** (heatmaps de P(hit), P(HR), P(hard contact)).
11. **Antecedentes de Coors Field** (revisión bibliográfica separada de nuestra evidencia).
12. **Página web** con los tres niveles (simple / visual / técnico).
13. **Tests automáticos:** hoy hay 6 (física/shrinkage). Faltan probabilidades, recursión, signos, escala, HBP e integración.
14. **Comentarios didácticos en el código** y README actualizado: el README todavía describe Extreme como "el estadio de altitud extrema".

## 11. Posibles errores (y su clasificación)

| id | tipo | descripción | impacto |
|---|---|---|---|
| E1 | **programación → error matemático** | `targets.py` cierra el turno con `is_hit_by_pitch` (siempre 0). Resultado: v_HBP = +0.004; Δ(hbp) < 0 en 7 de 12 conteos (3-0: −0.25, "mejor que un strike"); turnos fusionados y estados base-out mal reconstruidos después de cada HBP | Hasta 10.2 puntos de Pitching+; corr con el índice corregido = 0.991 |
| E2 | supuesto incorrecto en la documentación | README, EDA y documento técnico tratan Extreme = el estadio de altitud extrema | Interpretación |
| E3 | modelado | Un solo ρ̂ para Extreme, cuando sus estadios difieren (drag de 0.81 a 0.90) | El contrafactual "Extreme" es un promedio; para el estadio de altitud extrema el efecto probablemente es **mayor** que el reportado (🔎) |
| E4 | documentación | El intervalo "IC 80%" es un intervalo de credibilidad EB plug-in; no incluye la incertidumbre de los modelos ni de ρ̂ | Interpretación |
| E5 | definición | La media de Pitching+ es 98.0–98.8, no 100 (μ = media ponderada por n de pitchers con ≥100 pitches); sd = 8.4–8.7 por diseño (10 puntos = 1 τ) | Interpretación |
| E6 | programación menor | Sin guardia si τ² ≤ 0; NaN que se propaga sin aviso; Extension imposible (−6 a 14 ft) no filtrada; recta base del arsenal recalculada con otro universo (3,334 pitches distintos) | Bajo |
| E7 | decisión discutible | Stuff+ v1 "oficial" usa `altitude_level` como feature | No es una medida de stuff neutral |

## 12. Posibles problemas de leakage

| fuente | estado |
|---|---|
| Resultados futuros del mismo pitcher | **Descartado** (folds agrupados por pitcher) |
| Datos futuros de otros pitchers en los modelos | **Confirmado, efecto ≈ 0.01** en r; la conclusión sobrevive (§6.3) |
| ρ̂, coeficientes de VAA/HAA, RE24 y Δ con todas las temporadas | Confirmado, efecto incluido en el 0.01 |
| Hiperparámetros elegidos con las 3 temporadas y adopción condicionada al benchmark | Posible, no medido |
| Variables del resultado como features | **Descartado** |
| Decisiones tomadas después de ver resultados (prueba ajustada, mín. 300, efecto total) | **Posible** (sesgo de selección del análisis) |
| Selección de pares que exige volumen en t+1 | Sesgo de selección, no leakage |

## 13. Posibles problemas de unidades

| variable | unidad esperada | en el código | conversión | ¿correcta? |
|---|---|---|---|---|
| posición (x0, z0, RelHeight, PlateLoc) | ft | ft | — | ✅ (x(T) vs −PlateLocSide: MAE 0.002 ft) |
| velocidad Trackman (vx0…) | ft/s | ft/s | ×3600/5280 → mph | ✅ (ZoneSpeed: sesgo −0.10 mph) |
| RelSpeed, ZoneSpeed | mph | mph | — | ✅ |
| aceleración | ft/s² | ft/s² | — | ✅ |
| gravedad | 32.174 ft/s² | 32.174 | — | ✅ |
| tiempo (ZoneTime, T) | s | s | — | ✅ (T de la simulación = ZoneTime, MAE < 1e−4 s) |
| IVB, HB, VB | pulg. | ft × 12 | ½aT²·12 | ✅ unidades; ⚠️ sesgo de definición +0.67 pulg. contra Trackman (lo absorbe el método de deltas) |
| ángulos (VAA, HAA) | grados | `np.degrees(arctan2)` | rad → ° | ✅ |
| SpinAxis | grados [0, 360) | grados | 360 − x en zurdos | ✅ (rango 0.003–359.997) |
| SpinRate | rpm | rpm | — | ✅ (no se usa en la física) |
| densidad | kg/m³ | **no existe** | solo ratios adimensionales ρ̂ | ✅ (y debe decirse así) |
| k_D, k_M | 1/ft | 1/ft | se cancelan en el ratio | ✅ |
| "elevación equivalente" (reporte v2) | m | 8434.5·(−ln ρ) + 50 | escala de altura isoterma | ⚠️ aproximada; usa el ratio de drag como densidad |
| run value, E[Δ] | carreras/pitch | carreras/pitch | ×100 → carreras/100 pitches | ✅ |
| Pitching+ | puntos | 100 + 10·(μ − θ̂)/τ | — | ✅ (escala en unidades de τ) |

No se encontraron mezclas mph/ft·s⁻¹, pulgadas/pies, grados/radianes ni metros/pies en los cálculos.

## 14. Posibles problemas matemáticos (resumen)

1. **HBP** (E1): el único error matemático con impacto numérico.
2. **"Techo √r_stab".** Con o = T + e: r_pred ≤ √rel(o) y r_stab(o) = ρ_T·rel(o), así que √rel = √(r_stab/ρ_T) ≥ √r_stab. Es cota solo si el talento no cambia entre años (ρ_T = 1), con errores independientes e igual confiabilidad. Es una heurística, no una ley.
3. **"Calidad vs entorno" del efecto de altitud.** Algebraicamente, δ_bip = ΔP(bip)·v̄_N + P_E(bip)·(v̄_E − v̄_N). El primer término (+0.044/100) es efecto del pitch y hoy se cuenta como "entorno".
4. **D y M son una descomposición geométrica efectiva**, no el drag y el Magnus físicos verdaderos.
   - Además, ρ̂_M < ρ̂_D en todas las categorías. Con solo densidad, ambos deberían escalar igual, así que los ratios mezclan densidad, calibración de Trackman por estadio, clima y pelota.
   - El propio proyecto ya lo reconoce. **La página debe decirlo así.**
5. **Signos verificados:**
   - Δ(bola) > 0, Δ(strike) < 0, Δ(out) < 0 y Δ(hits) > 0 en los 12 conteos;
   - ejemplo 1-1: strike −0.080, bola +0.058;
   - corr(Pitching+ sin shrinkage, x̄) = −1.000: menos carreras esperadas = mejor score ✅.

**Signos y convenciones (verificados):**

| variable | convención | verificación |
|---|---|---|
| eje x | x = −PlateLocSide (vista Trackman) | r = 0.99999 |
| eje y | distancia al plato; vy0 < 0 (hacia el plato) | vy0 medio −127 ft/s |
| eje z | arriba; gravedad = −g ẑ | az0 medio −23.8 (= −32.2 + sustentación) |
| drag | opuesto a v (a_ng·v̂ < 0) | 100% de los pitches |
| HB | −½·a_x·T²·12; recta de derecho +8.4 pulg., de zurdo −8.3 | espejo → simétrico ✅ |
| IVB | ½(a_z + g)T²·12 | ✅ |
| VAA | negativo = hacia abajo | media −6.3° |
| HAA | −atan(v_x/−v_y) | MAE 0.04° |
| SpinAxis | derecho ≈ 211°, zurdo ≈ 150° | espejo 360 − x ✅ |
| RelSide | derecho +1.72 ft, zurdo −1.80 ft | inferencia de mano por signo ✅ |

## 15. Información que falta para identificar el estadio de altitud extrema

> **Decisión del proyecto (2026-10-05):** no se usará información externa (calendarios, nombres) ni se intentará identificar equipos, estadios o pitchers reales. Se respeta por completo la anonimización; los grupos se nombran `ANON_STADIUM_xx` (`reports/estadios_anonimos.md`). Las opciones de identificación de abajo quedan **descartadas**.

**Lo que el dataset NO trae:** estadio, equipo local/visitante, elevación, dimensiones, clima (temperatura, humedad, presión, viento) y fecha explícita.

**Lo que sí se puede reconstruir sin datos externos:**
- **Fecha de cada juego:** se obtiene del UUID v1 del PitchUID (2024-04-13 → 2026-06-13).
- **Equipos anónimos:** clustering espectral del grafo de "compañeros" (pitchers que lanzan del mismo lado en el mismo juego). Con k = 20 en 2024 (los 20 equipos de la LMB), el 97% de los juegos locales de cada equipo caen en una sola categoría, y solo 0.5% de los juegos quedan con local = visitante.
- **Resultado:**

| temporada | equipos con ≥80% de juegos locales en Extreme | juegos locales de cada uno |
|---|---:|---|
| 2024 | 5 | 51, 41, 41, 39, 37 |
| 2025 | 5–6 | 46, 46, 44, 43 (+1 mezclado) |
| 2026 | 5–6 | 30, 25, 21, 21, 16 (+1 mezclado) |

- **Ratio de drag dentro de pitcher, por estadio reconstruido de Extreme** (vs. el mismo pitcher y tipo en No Altitude): va de **0.81 a 0.90**, con intervalos de bootstrap que **no se traslapan** entre extremos. HR/BIP por estadio: 2.7%–6.6%.

**Conclusiones:**
1. "Extreme" agrupa ~5 estadios locales **físicamente distintos**.
2. 🔎 **Hipótesis (no verificable con el dataset):** el estadio de altitud extrema (~2,240 m, el más alto de la LMB) debería ser el de menor ratio de drag (~0.81–0.83). Pero los estadios más altos pueden quedar muy cerca entre sí, y el clima y la calibración de Trackman también mueven el ratio, así que el drag **no basta para identificarlo** con seguridad.
3. Para identificar a el equipo local y al el estadio de altitud extrema hace falta una de dos cosas, ambas **decisiones tuyas** por la regla del hackathon:
   - (a) aceptar el **calendario público** de la LMB (fechas de juegos locales del México) para emparejarlo con las fechas reconstruidas (es información pública, no pedida a el equipo local);
   - (b) trabajar solo con "estadios de altitud extrema" reconstruidos y anónimos, y decir explícitamente que el estadio de altitud extrema es uno de ellos.

## 16. Propuesta de arquitectura nueva

```
src/
├── data/           clean.py · targets.py (HBP corregido) · parks.py (NUEVO: reconstrucción de equipos/estadios, fechas)
├── physics/        trajectory.py · density.py (ρ̂ por categoría y POR ESTADIO reconstruido) · counterfactual.py
├── models/
│   ├── chain.py            cadena de eventos + V(b,s) + Δ (una sola fuente de verdad)
│   ├── stuff.py            Stuff+ (versión documentada: con/sin ubicación, con/sin altitud)
│   ├── location.py         Location+ integrado (misma cadena con solo ubicación + conteo)
│   ├── pitching.py         Pitching+ neutral + shrinkage
│   ├── whiff_k.py          (NUEVO) P(whiff | x), P(K | turno), efectos interpretables
│   └── contact.py          (NUEVO) P(contacto), EV/ángulo, P(hard), P(hit), P(HR) por entorno
├── analysis/
│   ├── altitude_chain.py   (NUEVO) los 5 niveles: física → pitch → bateador → contacto → juego
│   ├── sensitivity.py      (NUEVO) calidad neutral vs sensibilidad (modelo + empírica, con IC)
│   └── analyst_hypothesis.py (NUEVO) prueba formal de la hipótesis whiff > contacto en altura
├── validation/     benchmark.py · leakage.py (prueba temporal como rutina) · controls.py (control trivial obligatorio)
└── reports/        exportación de tablas → JSON para la web (cada cifra con su fuente)
web/                (NUEVO) página estática (HTML + JS, gráficos interactivos) que lee los JSON
tests/              probabilidades · recursión (2 strikes + foul) · física · D+M · contrafactual · shrinkage · signos · escala · HBP
```

**Principio rector:**
1. Cada número de la web viene de un JSON generado por un script.
2. Cada JSON registra el dataset, el commit y la función que lo produjo.
3. Nada se escribe a mano.

## 17. Lista de tareas

**P0 — bloquean todo lo demás**
1. Corregir HBP en `targets.py` y regenerar el pipeline completo. Agregar un test: Δ(hbp) > 0 en los 12 conteos.
2. Decidir cómo tratar el estadio de altitud extrema: (a) calendario público o (b) estadios reconstruidos anónimos. Mientras tanto, reemplazar "el estadio de altitud extrema" por "condiciones de altitud extrema" en README, EDA y documento técnico.
3. Reconstrucción de estadios (`parks.py`) como módulo con pruebas de calidad (pureza, local ≠ visitante) y ρ̂ por estadio.
4. Agregar el control trivial al benchmark como baseline obligatorio y reformular la conclusión predictiva. Declarar que el criterio pre-registrado no se cumplió.

**P1 — física y Pitching+ defendibles**
5. Tests automáticos: Σ P = 1, recursión con 2 strikes + foul, D + M = a_ng, contrafactual sin NaN y físicamente razonable, B ∈ (0, 1) y monotonía, signos, escala.
6. Renombrar el intervalo (credibilidad EB) y documentar media ≠ 100, sd < 10 y la cota √r_stab con sus supuestos.
7. Corregir la descomposición calidad/entorno (término de tasa de contacto).
8. Efecto de altitud con predicciones OOF; ρ̂ por estadio; investigar la sobrepredicción de IVB (~20%).
9. Documentar que Stuff+ v1 incluye altitud y ubicación (o reentrenar una versión sin `altitude_level`).

**P2 — las preguntas nuevas del proyecto**
10. Cadena de altitud por niveles (física → pitch → bateador → contacto → juego), con pruebas por eslabón y control por pitcher y bateador (efectos dentro de pitcher).
11. Whiff/K: P(whiff | x) interpretable (efectos por característica y tipo de pitch); K a nivel turno con universo documentado.
12. Contacto: EV, ángulo, hard contact, hit y HR por entorno.
13. Prueba formal de la hipótesis del analista (§6.4 es solo descriptivo).
14. Sensibilidad por pitcher: implicada por el modelo vs. empírica (dentro de pitcher, E vs N), con IC. Después, el gráfico de cuadrantes.

**P3 — producto**
15. Rankings con incertidumbre; sección el equipo local (si P0-2 lo permite).
16. Sección de bateadores (heatmaps de probabilidad, nunca "garantía").
17. Antecedentes de Coors Field (revisión separada de nuestra evidencia).
18. Página web (3 niveles por sección), README y comentarios didácticos en el código.

---

### Conflictos que necesitan tu decisión (no los elegí en silencio)

1. **el estadio de altitud extrema vs. "altitud extrema".**
   - El nombre del proyecto y el objetivo piden el estadio de altitud extrema; los datos solo permiten estadios anónimos.
   - Opción (a): el calendario público da precisión, pero usa información externa (¿está permitida?).
   - Opción (b): es 100% interna, pero no permite afirmar nada específico del estadio de altitud extrema.
2. **Sensibilidad "Pitching+_neutral − Pitching+_extreme".**
   - Con el contrafactual actual, esa diferencia la **implica el modelo**: todos los pitchers reciben los mismos ratios ρ̂, y la sensibilidad sale casi solo de su mezcla de pitches (Four-Seam +1.45 contra Slider +0.66 carreras/100).
   - La sensibilidad **empírica** (rendimiento real del mismo pitcher en E contra N) es la que de verdad distingue pitchers, pero es muy ruidosa.
   - Propuesta: mostrar ambas y no llamar "adaptación" a la primera.
3. **Stuff+ oficial.**
   - v1 predice mejor, pero incluye ubicación y altitud.
   - v4 es más "puro", pero predice peor.
   - Para una página que separa Stuff de Location, v1 contradice la definición. La decisión es tuya; la auditoría solo expone las consecuencias.
