# Auditoría matemática y de implementación (2026-10-05)

**Alcance.** Se contrastó `historial/reports/resumen_matematico.md` contra el código (commit `db1eed6`) y se ejecutaron pruebas numéricas sobre los datos y artefactos existentes. **No se modificó código del proyecto.** Los scripts de la auditoría están fuera del repositorio (scratchpad): `audit_data.py`, `audit_tests.py`, `audit_t8.py` y `audit_temporal.py`.

**Cifras del encargo vs. cifras vigentes.** Algunas cifras del encargo son las previas al ajuste de hiperparámetros (§7.7 de la bitácora):

| cifra | en el encargo | vigente (§7.11, resumen) | reproducida |
|---|---:|---:|---:|
| corr(con altitud, %Extreme) | −0.333 | −0.327 | **−0.3274** |
| corr(neutral oficial, %Extreme) | −0.074 | −0.054 | **−0.0541** |
| r_pred neutral, mín. 300 | 0.267 | 0.276 | **0.2756** |
| Δr neutral − con altitud | +0.059 | +0.067 | **+0.0672 (P = 98.7%)** |

Todas las cifras vigentes se reproducen exactamente.

---

## A. Veredicto general

| # | componente | veredicto |
|---|---|---|
| 1 | Benchmark (emparejamiento, bootstrap, Δr pareado) | 🟢 Correcto |
| 2 | "Techo" √r_stab | 🟠 Necesita revisión: es una heurística, no una cota |
| 3 | Espejo de zurdos | 🟢 Correcto |
| 4 | VAA/HAA ajustados por ubicación | 🟡 Correcto con observaciones |
| 5 | Cadena de probabilidades (Σ = 1, universos) | 🟢 Correcto |
| 6 | Recursión V(b, s) | 🟢 Correcto |
| 7 | Valores terminales / RE24 (golpeado) | 🔴 **Incorrecto** |
| 8 | Δ y E[Δ] (signo, estado previo/posterior, sin fuga del resultado) | 🟢 Correcto (salvo HBP, ver #7) |
| 9 | Cinemática de 9 parámetros | 🟡 Correcto con observaciones |
| 10 | Descomposición drag / Magnus | 🟢 Correcto |
| 11 | Densidad efectiva ρ̂ | 🟡 Correcto con observaciones |
| 12 | Reproyección (T, v₀, mismo objetivo) | 🟢 Correcto |
| 13 | Coherencia física de las features proyectadas | 🟢 Correcto (sin features congeladas en el modelo oficial) |
| 14 | Efecto de altitud δ y su descomposición | 🟡 Correcto con observaciones |
| 15 | Pitching+ neutral (usa features proyectadas) | 🟢 Correcto |
| 16 | Shrinkage empírico-bayesiano | 🟡 Correcto con observaciones |
| 17 | Intervalos de 80% | 🟠 Necesita revisión (nombre y alcance) |
| 18 | Escala del índice (100 / 10) | 🟡 Correcto con observaciones (la media no es 100) |
| 19 | Validación de neutralización | 🟡 Correcto con observaciones |
| 20 | Validación predictiva del índice neutral | 🟠 **Necesita revisión** (la ganancia no se atribuye a la física) |
| 21 | Stuff+ v1 (oficial) | 🟠 Necesita revisión (incluye altitud, no solo ubicación) |
| 22 | Stuff+ v4 | 🟡 Correcto con observaciones |
| 23 | Ajuste de hiperparámetros | 🟡 Correcto con observaciones |

**En una frase:** la maquinaria matemática (cadena, recursión, física, shrinkage) está implementada fielmente y pasa todas las pruebas numéricas. Hay **un error de datos que afecta los valores** (el golpeado vale ≈ 0 carreras) y **un problema de interpretación de la conclusión principal**: la ventaja predictiva del índice neutral es real y sobrevive al control de leakage temporal, pero un ajuste trivial por parque logra lo mismo. La física no es la que aporta esa ventaja.

---

## B. Tabla de auditoría

| # | metodología | archivo · función | qué hace el código | qué debería hacer | ¿coincide? | severidad | impacto | recomendación |
|---|---|---|---|---|---|---|---|---|
| 1 | r_stab, r_pred por pares (p, y→y+1); bootstrap por pitcher B=2,000; Δr pareado; P(Δr>0) ≥ 0.90 | `src/validation/benchmark.py` · `build_pairs`, `_bootstrap_indices`, `correlation_suite` | Une t con t+1 por `pitcher_anon_id`, exigiendo n ≥ n_min en ambas. Remuestrea pitchers (todas sus filas, incluso si aparece en los dos pares). Calcula todas las métricas sobre las mismas filas (casos completos) y en la misma réplica. P = mean(boot_m > boot_ref) | Lo mismo | Sí | — | — | Documentar que la selección de pares condiciona en el volumen de t+1 (n ≥ n_min) |
| 2 | "r_pred ≲ √r_stab(o)" | `resumen_matematico.md` §2 (no hay código) | — | Una cota válida solo bajo supuestos explícitos | No es teorema | Media (documental) | Presentada como cota dura | Reescribir como heurística (ver C2) |
| 3 | Espejo: HB, RelSide, ángulos horizontales, PlateLocSide → −; SpinAxis → 360° − SpinAxis | `src/models/stuff_plus_v2.py` · `build_v2_features` | Multiplica 7 columnas por s = ±1 según la mano del **pitcher** (inferida de RelSide si es nula). SpinAxis → 360 − x solo en zurdos. Se aplica una sola vez por DataFrame, siempre sobre datos crudos (`cf_v2.load_data` no espeja) | Lo mismo | Sí | — | — | Ninguna. SpinAxis ∈ (0.003, 359.997) ⇒ la imagen queda en (0, 360); no hay 0 que se convierta en 360 |
| 4 | VAA_adj = VAA − (â_t + b̂_t·PlateLocHeight), por tipo | `stuff_plus_v2.py` · `fit_location_adjustment` / `apply_location_adjustment` | OLS (`np.polyfit`) por AutoPitchType (≥100 pitches; si no, global), ajustada **con todos los pitches** y después del espejo (HAA ~ PlateLocSide en el marco de un derecho). Unidades: grados ~ pies | Lo mismo; idealmente ajustada solo con el fold de entrenamiento | Sí (fuga despreciable) | Baja | 2 parámetros por tipo sobre 627k pitches | Opcional: ajustar dentro del fold |
| 5 | P(e\|x) por cadena condicional | `run_value_chain.py` · `load_data`, `universe_masks`, `oof_submodel`, `event_probabilities` | Universos verificados: take ⟂ swing; contacto = swing ∧ ¬whiff; en juego = contacto ∧ ¬foul. Eventos desde PitchCall (1,118 filas con is_swing inconsistente se resuelven por PitchCall). Σ = 1 algebraicamente | Lo mismo | Sí | — | — | Agregar una aserción de NaN (hoy un NaN se propaga sin aviso) |
| 6 | V(b,s) hacia atrás; auto-lazo del foul con 2 strikes | `run_value_chain.py` · `count_values` | Orden total = b + s de 5 a 0 (cada estado depende solo de estados con total + 1). Con s = 2: V = Σ_{e≠foul} p_e v_e / (1 − p_foul) | Lo mismo | Sí | — | — | — |
| 7 | v_T = E[rv \| el turno termina con T] | `run_value_chain.py` · `terminal_values`; causa en `src/features/targets.py` · `_assign_plate_appearances` | `_pa_end` usa `is_hit_by_pitch`, que vale 0 en todo el dataset ⇒ el golpeado no termina el turno: rv = 0 en 2,228 de 2,249 HBP y v_HBP = +0.004 | v_HBP ≈ valor de una base gratis (≈ +0.33 a +0.38) | **No** | **Alta** | Δ(hbp) < 0 en 7 de 12 conteos; hasta **10.2 puntos** de Pitching+ para algunos pitchers; RE24 de los turnos siguientes contaminado | Usar PitchCall == "HitByPitch" en `_pa_end` y en el avance de bases; regenerar todo |
| 8 | Δ = v_next − V(b,s); E[Δ] = Σ P̂ Δ | `count_values`, `expected_run_value`, `delta_matrix_for` | V(b,s) = estado **antes** del pitch (Balls/Strikes del pitch); v_next = estado posterior. Ninguna feature deriva del resultado; `realized_delta` solo se usa como resultado | Lo mismo | Sí | — | — | — |
| 9 | r(t) = r₀ + v₀t + ½at² | `src/features/trajectory.py` · `_release_state`, `_fly` | Raíz cuadrática para y(t); ft, ft/s, ft/s², s; x = −PlateLocSide | Lo mismo | Sí | Baja | IVB simulado vs. Trackman: sesgo +0.67 pulg. (lo absorbe el método de deltas) | Documentar que IVB/HB de Trackman no coinciden exactamente con ½aT² |
| 10 | a_ng = a + gẑ; D = (a_ng·v̂)v̂; M = a_ng − D | `trajectory.py` · `decompose_acceleration` | Exactamente eso, con v̂ = v(y = 50)/\|v\| | Lo mismo | Sí | — | — | — |
| 11 | ρ̂_X(c) = median_g[k̃_X(g,c) / k̃_X(g,N)] | `trajectory.py` · `estimate_density_ratios` | g = (pitcher, temporada, AutoPitchType); ≥15 pitches por celda en ambas categorías (inner join con N); mediana dentro de la celda y mediana entre grupos; ratio por tipo si hay ≥50 grupos, si no global | Lo mismo | Sí | Baja | Ratios estimados con todos los datos (ver E) | Reportar la dispersión entre grupos (p10–p90 Magnus E: 0.66–0.90) |
| 12 | a′ escalado; T de la ecuación en y; v_x = (x_p − x_r − ½a′_xT²)/T | `trajectory.py` · `project_to_altitude`, `_fly` | T depende solo de (y_r, v_ry, a′_y). v_ry se conserva y a′_y no depende de v_x, v_z ⇒ **no hay circularidad** v₀ → T → v₀ ni iteración. Llega al objetivo con error 1.8e−15 | Lo mismo | Sí | — | — | — |
| 13 | f′ = f_obs + [f̂(a′) − f̂(a)] | `project_to_altitude`, `recompute_fastball_diffs` | Actualiza ZoneSpeed, ZoneTime, SpeedDrop, VAA, HAA, IVB, HB, VB y después vaa/haa_loc_adj e ivb/hb_diff_vs_fb. No toca RelSpeed, SpinRate, SpinAxis, Extension, EffectiveVelo, ángulos de salida, pfxx/pfxz ni Tilt | Lo mismo; de las no tocadas, ninguna es feature del modelo oficial salvo RelSpeed/Spin/Extension (físicamente invariantes) | Sí | Baja | RelSpeed implícita cambia +0.018 mph (despreciable) | — |
| 14 | δᵢ = E[Δ\|E] − E[Δ\|N]; calidad = take + whiff + foul | `src/analysis/altitude_counterfactual_v2.py` · `run`, `predict_scenario` | Con los modelos **finales** (in-sample). Σ componentes = δ (error 1e−16) | Igual; con OOF para no subestimar la incertidumbre | Sí | Media | "Calidad" omite el cambio en P(en juego) (+0.044 de 0.786/100) | Ver D-M7 |
| 15 | x̄_py = media de E[Δ \| proyectado a N] | `pitching_plus_neutral.py` · `neutral_features`, `generate_oof`, `build_official_table` | `xrv_neutral` se calcula con `neutral[cols]` (features proyectadas) y el shrinkage oficial se aplica a `xrv_neutral` | Lo mismo | Sí | — | — | — |
| 16 | τ² = Var(x̄) − E[σ²/n]; B; θ̂ | `pitching_plus_neutral.py` · `empirical_bayes` | σ² = media ponderada por n de la var. dentro de pitcher (n ≥ 100); μ ponderada por n; Var(x̄) **sin ponderar**; sin protección si τ² ≤ 0 | Igual + estimador coherente + guardia | Sí (con observaciones) | Baja | — | Guardia para τ² ≤ 0; documentar el estimador |
| 17 | IC 80% = ±1.2816·10·sd_post/τ | `empirical_bayes` | Fórmula correcta, pero es un **intervalo de credibilidad posterior plug-in** | Llamarlo así | Nombre incorrecto | Media (documental) | Se lee como IC frecuentista con toda la incertidumbre | Renombrar y declarar qué excluye |
| 18 | 100 = media, 10 = τ | `empirical_bayes` | 100 ↔ μ = media ponderada por n (n ≥ 100); 10 ↔ 1 τ | — | Sí, pero "media = 100" es falso empíricamente (98–99) | Baja | Interpretación | Documentar |
| 19 | corr(score, %E) antes/después | `neutralization_check` | n ≥ 100, temporadas agrupadas | — | Sí | Baja | Heterogénea por año (+0.08 / −0.10 / −0.15) | Reportar por año |
| 20 | Δr neutral − con altitud vs. carreras ajustadas t+1 | `park_adjusted_test` | Correcto como cálculo. Resultado post-hoc; un control trivial empata | — | Cálculo sí; conclusión no | **Alta** (conclusión) | Ver H | Ver P0-2 |
| 21 | Stuff+ v1 = 100 + 10·z(p̄ whiff) | `benchmark.py` · `stuff_plus_v1_oof_pitches`, `scale_plus` | Target is_whiff sobre todos los pitches; features con `altitude_level`, residuales de Magnus, VAA/HAA y ángulos de salida; z por año sobre **todas** las pitcher-temporadas, sin shrinkage | — | Sí, pero el nombre es incompleto | Media | No es neutral por altitud y su escala no es comparable con la de Pitching+ | Renombrar o documentar |
| 22 | Stuff+ v4 = P(whiff \| swing), neutral + EB; uso por tipo | `stuff_plus_v4.py`; `scouting_report_v2.py` · `arsenal_table` | Entrena con swings y predice todos los pitches (proyectados); EB sobre −p; dentro del tipo: 100·(π̄_p,t − π̄_t) en pp, **sin shrinkage** | — | Sí | Baja | Diferencias por tipo ruidosas con pocos pitches | Shrinkage por pitcher × tipo |
| 23 | Optuna por log-loss; adopción con 3 criterios | `src/models/tuning.py` | Partición por pitcher 80/20 con las 3 temporadas; el criterio 3 usa el benchmark como barrera de no-inferioridad | — | Sí | Baja | Selección parcial con la métrica de evaluación | Documentar |

---

## C. Errores matemáticos demostrables

### C1. El golpeado (HBP) tiene un valor terminal ≈ 0 y por eso Δ(hbp) tiene el signo equivocado

- **Hecho.** En `pitches_with_targets.parquet`, el run_value de los 2,249 pitches HBP es 0 en 2,228 casos (media +0.0038). Su valor terminal debería parecerse al de la base por bolas (+0.381), porque ambos ponen al bateador en primera con avance forzado.
- **Consecuencia matemática.** Como Δ(hbp, b, s) = v_HBP − V(b, s) ≈ −V(b, s), el golpeado se valúa como **favorable al pitcher** en todo conteo con V > 0:

  | conteo | 0-0 | 1-0 | 2-0 | 2-1 | 3-0 | 3-1 | 3-2 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Δ(hbp) | −0.010 | −0.055 | −0.126 | −0.053 | −0.246 | −0.170 | −0.074 |

  En 3-0, golpear al bateador "vale" −0.25 carreras: mejor para el pitcher que un strike (−0.076). Viola el orden Δ(hbp) ≥ Δ(ball) > 0.
- **Sensibilidad medida.** Con v_HBP = v_BB y las mismas probabilidades OOF:
  - V(b, s) sube entre +0.0015 y +0.0052;
  - corr(x̄ original, x̄ corregido) = 0.991 en pitcher-temporadas con n ≥ 100;
  - el desplazamiento relativo tiene sd = 1.65 puntos de Pitching+ y **máximo = 10.2 puntos**.
- La causa está en D1.

### C2. "Ninguna métrica puede superar √r_stab(o)" no es una cota válida tal como está escrita

- **Derivación correcta.** Teoría clásica de la medición: o_y = T_y + e_y, con errores independientes entre sí y de la métrica m.
  - corr(m_y, o_{y+1}) = corr(m_y, T_{y+1}) · √rel(o_{y+1}) ≤ √rel(o), con rel = Var(T)/Var(o).
  - Además, r_stab(o) = corr(o_y, o_{y+1}) = ρ_T · rel(o) si la confiabilidad es igual en ambos años, donde ρ_T = corr(T_y, T_{y+1}).
  - Entonces √rel(o) = √(r_stab(o)/ρ_T) ≥ √r_stab(o).
- **Lectura.** √r_stab(o) solo es cota si el talento no cambia entre años (ρ_T = 1), los errores son independientes y la confiabilidad es igual en los dos años. Si el talento deriva (ρ_T < 1), la cota real es **más alta**.
- **Problema adicional.** La cifra citada (r_stab = 0.085–0.155) es del resultado RE24 sin ajustar con n ≥ 100. La conclusión principal usa otro resultado (carreras ajustadas por entorno, n ≥ 300), cuya autocorrelación año contra año es ≈ 0.27, así que el "techo" correspondiente sería ≈ 0.52.
- **Corrección.** Presentarlo como orden de magnitud bajo supuestos, no como cota.

### C3. La descomposición "calidad = take + whiff + foul" no separa calidad de entorno

- **Algebraicamente:** δ_bip = ΔP(bip)·v̄_N + P_E(bip)·(v̄_E − v̄_N), con v̄ = Σ_k π_bip(k)·Δ_k. El primer término (más o menos contacto en juego porque el pitch cambió) es efecto del pitch, pero queda contado en "bip/entorno".
- **Medido** con 120k pitches (carreras/100):

  | | total | "calidad" documentada | término de tasa de contacto | término de valor del contacto |
  |---|---:|---:|---:|---:|
  | todos | +1.104 | +0.319 | +0.044 | +0.742 |

  La "calidad" se subestima ~14% (0.319 vs. 0.363).
- **Severidad.** Cuantitativamente menor, pero la etiqueta es formalmente incorrecta. Además, el término de valor mezcla calidad del contacto (que también depende del pitch) con el parque. El §8.2 del resumen ya lo reconoce como supuesto de ignorabilidad.

**No se encontraron otros errores matemáticos.** Se verificaron numéricamente:
- Σ P = 1 (máx. 5.6e−16);
- la ecuación de Bellman con auto-lazo (residuo 1.4e−17);
- la monotonía de V;
- el signo de Δ;
- D + M = a_ng (3.6e−15) y D ⟂ M;
- la llegada al objetivo (1.8e−15);
- las identidades internas de las features proyectadas;
- la fórmula de la escala (1.4e−14).

---

## D. Errores de implementación y decisiones (por categoría)

### D-I. Errores de programación

| id | dónde | qué | impacto |
|---|---|---|---|
| D1 | `targets.py` · `_assign_plate_appearances`, `_resolve_plate_appearance` | El fin del turno y el avance de bases usan `is_hit_by_pitch`, que vale 0 en todo el dataset (el propio `benchmark.py` lo documenta). (1) El turno del HBP se fusiona con el siguiente. (2) El corredor no se coloca en primera y el estado base-out queda mal en el resto de la media entrada. (3) La matriz RE24 se estima con esos estados. | Causa de C1. También afecta `obs_rv_prevented` del benchmark (HBP = 0) y `realized_delta` |
| D2 | `pitching_plus_neutral.py` · `empirical_bayes` | Sin guardia para τ² ≤ 0: `np.sqrt` daría NaN en toda la temporada | Latente (hoy τ² > 0) |
| D3 | `trajectory.py` · `recompute_fastball_diffs` vs. `arsenal.py` | Sin proyectar, recalcular ivb/hb_diff_vs_fb difiere del original en 3,334 pitches (hasta 1.4/2.6 pulg.), porque las rectas base se promedian sobre universos distintos | Cambio espurio menor en el escenario "neutral" |
| D4 | `trajectory.py` · `_release_state` | Solo se descartan trayectorias con T no finito. Extension tiene valores imposibles (mín. −6.1 ft, máx. 14.4 ft; 17 de 50k fuera de [3, 9]) que producen puntos de salida no físicos pero "válidos" | Despreciable |
| D5 | `benchmark.build_metric_table` vs. `park_adjusted_test` | n_pitches viene de universos distintos (17 pitcher-temporadas difieren, hasta 157 pitches), así que los pares de las dos pruebas pueden no ser idénticos | Bajo |
| D6 | `build_official_table` | La tabla oficial mezcla el índice OOF con `costo_altitud_*` de los modelos finales in-sample. Además, "ajuste_altitud" (obs − N, OOF) y "costo_altitud" (E − N, in-sample) son dos cantidades distintas con nombres parecidos | Confusión de interpretación |

### D-II. Decisiones metodológicas discutibles

| id | decisión | por qué es discutible |
|---|---|---|
| M1 | Aceptar el índice neutral con la prueba de "carreras ajustadas por entorno, n ≥ 300" | La prueba se definió **después** de ver resultados (admitido en §7.7). El criterio pre-registrado (§7.2) era carreras observadas t+1, n ≥ 100, contra v1, y **no se cumple**: neutral 0.134 vs. v1 0.169 (P = 21%) vs. v3 con altitud 0.160. De las 4 combinaciones (observadas/ajustadas × 100/300) solo una pasa el umbral. La frase "cumple el criterio de 7.2" en la bitácora es inexacta |
| M2 | Atribuir la mejora predictiva a la neutralización física | Control: restar a E[Δ] observado su media por (temporada, categoría), con el mismo shrinkage y sin física. Obtiene 0.2747 vs. 0.2756 del oficial (Δ = +0.0008, P = 54%) con n ≥ 300, y **le gana** con n ≥ 100 (0.174 vs. 0.164). corr(oficial, control) = 0.979. La prueba premia cualquier ajuste por parque porque el resultado fue ajustado por la misma partición |
| M3 | μ y τ estimados con n ≥ 100 y usados como destino del shrinkage para todos | Los pitchers con poco volumen son peores en promedio (media 98.0 en todos vs. 98.9 con n ≥ 100): su θ̂ queda sesgado hacia arriba |
| M4 | σ² común para todos los pitchers | Ignora diferencias de rol/mezcla de conteos. El split-half da σ² ≈ 0.00277 vs. 0.00260 usado (+6%, leve dependencia entre pitches) |
| M5 | Stuff+ oficial = v1 | Incluye `altitude_level` y residuales de Magnus como features: **no es neutral y ve la altitud explícitamente**. Además usa VAA/HAA y ángulos de salida (ubicación) y se escala sin shrinkage sobre todas las pitcher-temporadas (sd = 10 incluyendo ruido). No es comparable con Pitching+ |
| M6 | δ con modelos finales in-sample | Subestima la incertidumbre (ya declarado en §8.3) |
| M7 | Etiqueta "calidad del pitch" | Ver C3 |
| M8 | Hiperparámetros con las 3 temporadas; criterio 3 de adopción usa el benchmark | Selección parcial sobre la métrica de evaluación (como barrera de no-inferioridad, riesgo bajo) |
| M9 | Meta de neutralización: corr(score, %E) → 0 | Supone que el talento es independiente del parque. Si los equipos de altura tienen pitchers peores, la correlación verdadera no es 0 |

### D-III. Supuestos no demostrados

| id | supuesto | estado |
|---|---|---|
| S1 | Aceleración constante | Validado (posición en el plato con MAE de 0.002–0.007 ft, r ≥ 0.998). IVB/HB simulados difieren de Trackman (sesgo +0.67 pulg., MAE de 0.72/0.93): Trackman los define con otra ventana; el método de deltas absorbe el sesgo aditivo, no el de escala |
| S2 | Un ρ̂ por (categoría, tipo) vale para todos los pitches | La dispersión entre grupos es amplia (Magnus E: p10 = 0.66, p90 = 0.90). Split-half: el cambio de IVB se **sobrepredice ~20%** (−2.64 vs. −2.20 pulg.), mientras ZoneSpeed y VAA se predicen bien |
| S3 | Comando: mismo objetivo en el plato | No verificable con los datos |
| S4 | Ignorabilidad: f(trayectoria) → resultado es invariante a la altitud | No demostrado; M2 sugiere que, a nivel índice, la física equivale a un ajuste por categoría |
| S5 | Conteo markoviano, frecuencias de eventos comunes a parques y años | Estándar, no verificado por parque |
| S6 | Normal-normal con pitches iid dentro de pitcher | Aproximado (+6% en σ²) |
| S7 | √r_stab como techo | Ver C2 |

---

## E. Leakage

| fuente | tratamiento en el código | estado |
|---|---|---|
| Resultados t+1 **del mismo pitcher** al puntuarlo en t | GroupKFold por `pitcher_anon_id` con todas sus temporadas en el mismo fold | **Descartada** |
| Datos t+1 de **otros** pitchers en los modelos que puntúan t | Los folds mezclan temporadas | **Confirmada, efecto pequeño.** Reentrenando todo con temporadas ≤ t: r = 0.2646 vs. 0.2756 (n ≥ 300) y 0.1665 vs. 0.164 (n ≥ 100). Δ neutral − con altitud = +0.061, P = 98.6%: **la conclusión sobrevive** |
| ρ̂ con todas las temporadas y pitchers | `estimate_density_ratios(raw)` antes del split | Confirmada, pequeña: ρ_E con ≤ 2024 = 0.864/0.780 vs. 0.854/0.770 con todo. Incluida en la prueba temporal |
| Coeficientes de VAA/HAA ajustados | Con todos los pitches | Posible, despreciable (incluida en la prueba temporal) |
| Valores terminales, V(b,s), Δ | Con las 3 temporadas | Posible, despreciable (incluida en la prueba temporal) |
| Shrinkage (μ, τ, σ por temporada) | Solo datos de la temporada t para puntuar t | **Descartada** |
| z-scores (`scale_plus`) | Por temporada | **Descartada** |
| Hiperparámetros | Optuna con las 3 temporadas; adopción condicionada al benchmark | **Posible** (no cubierta por la prueba temporal, que usó los parámetros ajustados) |
| Variables del resultado como features | Features = física + ubicación + conteo previo; sin PitchCall, rv, eventos | **Descartada** |
| `realized_delta` como feature | Solo se usa como resultado | **Descartada** |
| Ajuste del resultado por (año, categoría) | Se calcula sobre el año t+1 (lado del resultado) | **Descartada** |
| Selección de pares por n ≥ n_min en t+1 | Condiciona en el volumen futuro | No es leakage del predictor; sí es **sesgo de selección** (restringe el rango) |
| Decisiones de diseño tras ver resultados (prueba ajustada, n ≥ 300, efecto total, DIPS) | — | **Posible** (jardín de senderos que se bifurcan, ver M1) |
| "Mejor baseline" elegido por estimación puntual | — | Posible, menor |

---

## F. Resultados de los tests

| test | resultado | valor esperado | diferencia | severidad |
|---|---|---|---|---|
| 1. Σ P(e) = 1 (627,185 pitches OOF ×2 + 3×40k con el artefacto final; casos 0/1) | máx. 5.6e−16, media 8.3e−17, **0 violaciones**; P ∈ [1.7e−9, 0.9994]. Un NaN se propaga (no se recorta) | 1 ± 1e−9 | ninguna | 🟢 |
| 2. Recursión V(b,s), 12 conteos | Bellman 1.4e−17; V decrece con strikes y crece con bolas en todos los casos; V(0,0) = 0.01417 vs. rv/turno = 0.01418 | — | Chequeo de V(0,0) **tautológico** (misma muestra) | 🟢 recursión / 🔴 Δ(hbp) |
| 3. Cinemática r(T) | x: sesgo −0.0007, MAE 0.0023 ft; z: −0.0065, MAE 0.0069 ft; ZoneSpeed −0.10 mph; VAA −0.033°; HAA +0.012°; RelHeight/RelSide MAE 0.003 ft; T = ZoneTime (MAE 0.0000 s); IVB +0.67 pulg.; HB MAE 0.93 pulg. | ≈ 0 | IVB/HB con sesgo de definición | 🟡 |
| 4. D + M = a_ng | 3.6e−15; M·v̂ = 2.5e−14; drag ≤ 0 en 0% | 0 | ninguna | 🟢 |
| 5. Contrafactual E → N (30k) | Objetivo exacto (1.8e−15). FB: IVB +4.0 pulg., VAA +0.29°, ZoneSpeed −1.38 mph, ZoneTime +0.003 s (signos correctos). Identidades SpeedDrop, VB−IVB y ZoneTime exactas. Identidad E→E = 1e−5. Ninguna feature del modelo queda congelada indebidamente | coherente | RelSpeed fija (+0.018 mph implícitos); IVB sobrepredicho ~20% en split-half | 🟢/🟡 |
| 6. Neutralización | −0.3274 → −0.0541 (Spearman −0.333 → −0.076). Por año: 2024 −0.196 → +0.078; 2025 −0.389 → −0.096; 2026 −0.400 → −0.150 | ≈ 0 | heterogeneidad entre años | 🟡 |
| 7. Shrinkage | B medio: n ≤ 30: 0.81; 30–100: 0.54; 100–300: 0.28; 300–1000: 0.13; >1000: 0.06. Monótono en n. B(n=1) ≈ 0.985; B(n=10⁶) ≈ 7e−5. Ancho 80%: 23 → 6 puntos | B decreciente | ninguna | 🟢 |
| 8. Benchmark | 0.2084 → 0.2756, Δ = +0.0672, P = 98.7% (reproducido). Temporal: Δ = +0.061, P = 98.6%. **Control trivial: 0.2747** (Δ oficial − control = +0.0008, P = 54%). Criterio pre-registrado: neutral 0.134 vs. v1 0.169, P = 21% | ganancia atribuible a la física | la física no supera al control | 🟠 |
| 9. Signos | Δ(bola) > 0, Δ(strike) < 0, Δ(out) < 0, Δ(hits) > 0 en los 12 conteos; 1-1: strike −0.080, bola +0.058; 3-2: K −0.407, BB +0.303; corr(Pitching+ sin shrinkage, x̄) = −1.000000. **Δ(hbp) < 0 en 7 conteos** | Δ(hbp) > 0 | 7/12 | 🔴 (HBP) |
| 10. Escala | Pitching+ = 100 + 10z exacto (1.4e−14). Media: 98.0–98.8 (todos), 98.9–99.3 (n ≥ 100). sd: 8.4–8.7. P5–P95 ≈ 83.5–112. Sin shrinkage con n ≥ 100: sd ≈ 11 | "media 100" | −1 a −2 puntos | 🟡 |
| Extra: AUC 0.732 → 0.640 | v1 0.7315, v2d 0.6396 (reproducido). La caída ocurre en v2a → v2b (0.736 → 0.634), paso que **a la vez** residualiza VAA/HAA y quita los ángulos de salida | — | la atribución a "ubicación" mezcla dos cambios | 🟡 |
| Extra: pytest | 6/6 pasan | — | — | 🟢 |

---

## G. Resultados que se mantienen

1. **La cadena de probabilidades está bien construida:** Σ P = 1, universos correctos, sin fuga del resultado.
2. **La recursión de V(b,s) y la tabla de Δ** son correctas para todos los eventos, salvo HBP.
3. **La física del contrafactual es coherente:** reproducción de la trayectoria, descomposición exacta, reproyección analítica sin circularidad e identidades conjuntas entre features.
4. **Los ratios ρ̂ medidos** (0.906/0.834; 0.854/0.770) se reproducen exactamente, y son robustos a estimarse solo con ≤ 2024.
5. **El índice neutral elimina la mayor parte de la relación con %Extreme** (−0.33 → −0.05).
6. **El índice neutral predice mejor que el índice con altitud las carreras ajustadas por entorno con n ≥ 300** (Δ ≈ +0.06–0.07, P ≈ 98–99%), y esto **sobrevive al control de leakage temporal**.
7. **El shrinkage está bien implementado** y se comporta según la teoría (límites n → 0 y n → ∞).
8. **El diagnóstico DIPS, el rechazo de Stuff+ v4 y la equivalencia v3 ≈ v1** no dependen de los errores encontrados (HBP los mueve poco: corr 0.991).

## H. Resultados que no deben publicarse todavía (o deben reformularse)

1. **"La neutralización física mejora el poder predictivo."** Un ajuste trivial por (temporada, categoría) logra lo mismo. Lo defendible es: *ajustar por entorno mejora la predicción del rendimiento ajustado por entorno; el contrafactual físico lo logra tan bien como un ajuste por parque y además da una explicación mecánica.*
2. **"Cumple el criterio de 7.2."** El criterio pre-registrado (carreras observadas, n ≥ 100, contra v1) no se cumple. Debe presentarse como análisis exploratorio post-hoc.
3. **Rankings individuales y fichas.** El error de HBP mueve hasta 10 puntos a algunos pitchers. Hay que regenerarlos antes de publicar nombres o percentiles.
4. **"Techo ≈ 0.3–0.4".** Reformular como heurística (C2).
5. **"IC 80%".** Renombrar como intervalo de credibilidad empírico-bayesiano que excluye la incertidumbre de los modelos ML, de ρ̂ y de los hiperparámetros.
6. **"Calidad del pitch" del efecto de altitud** (+0.41 Four-Seam, etc.). Agregar el término de tasa de contacto o renombrarlo como "componentes antes del contacto".
7. **"Stuff+ v1 (incluye ubicación)".** También incluye altitud; no debe presentarse junto a Pitching+ neutral como si fuera comparable.
8. **Cifras absolutas en carreras** (efecto total +1.10/100, V(b,s), valores Δ). Cambian levemente al corregir HBP (V sube 0.002–0.005).

---

## I. Plan de corrección priorizado

**P0 — bloquean las conclusiones principales**
1. **Corregir HBP en `targets.py`** (fin de turno y avance de bases con PitchCall == "HitByPitch") y regenerar en orden: targets → RE24 → cadena → contrafactual → índice neutral → tablas, fichas y reportes. Verificar Δ(hbp) > 0 en los 12 conteos.
2. **Reformular la conclusión predictiva.** Agregar el control trivial al reporte oficial como baseline obligatorio. Decidir si el claim es "neutralización ≈ ajuste por parque, con mecanismo" o buscar una prueba donde la física aporte por encima del control (p. ej., pitchers que cambian de equipo/parque entre temporadas, o la predicción de whiff/K t+1 neutral).
3. **Declarar explícitamente el resultado del criterio pre-registrado** (no cumplido) y etiquetar la prueba ajustada como post-hoc.

**P1 — importantes**
4. Reescribir el "techo" √r_stab con sus supuestos (C2).
5. Renombrar el intervalo como intervalo de credibilidad EB plug-in y listar lo que excluye.
6. Corregir la descomposición "calidad vs. entorno" (término de tasa de contacto) o renombrarla.
7. Stuff+ v1: documentar que incluye `altitude_level` y residuales de Magnus, y que su escala no tiene shrinkage; o reentrenarlo sin altitud.
8. Documentar que la media de Pitching+ no es 100 (μ ponderado, n ≥ 100) y que la sd del índice es < 10 por diseño.
9. Reportar la neutralización por temporada (2026 queda en −0.15).
10. Calcular δ con predicciones OOF (como el índice) para que su IC no sea in-sample.

**P2 — mejoras metodológicas**
11. Destino del shrinkage dependiente del volumen/rol (μ(n)) en vez de μ de n ≥ 100.
12. Estimador de momentos coherente (μ y Var con la misma ponderación) y σ² por pitcher o por rol.
13. Investigar la sobrepredicción de ~20% del cambio de IVB (ventana de Trackman, ρ por tipo/pitcher).
14. Ajustar los coeficientes de ubicación y ρ̂ dentro de cada fold (pulcritud; impacto medido despreciable).
15. Ablation que separe "residualizar VAA/HAA" de "quitar ángulos de salida" para sostener la atribución a ubicación.
16. Shrinkage para el Stuff+ v4 por tipo de pitch.

**P3 — limpieza e ingeniería**
17. Guardia para τ² ≤ 0 en `empirical_bayes`; aserción de NaN en `event_probabilities`.
18. Unificar la recta base de `recompute_fastball_diffs` con `arsenal.py`.
19. Filtro de Extension físicamente plausible.
20. Usar el mismo universo de n_pitches en el benchmark y en la prueba ajustada.
21. Separar en la tabla oficial las columnas OOF de las in-sample y aclarar la diferencia entre `ajuste_altitud` y `costo_altitud`.
22. Documentar que `synthesize_pitch` (dashboard) es una segunda implementación de la proyección (iterativa, con dirección de velocidad en línea recta), válida solo para pitches hipotéticos.
