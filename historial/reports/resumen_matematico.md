# Resumen matemático del proyecto (v1 → v3 neutral)

Base para reescribir el documento técnico. Las cifras vienen de `reports/resultados_finales.md` §7 y de los reportes que se citan ahí.

> **Versión post-auditoría (2026-10-05).** Corregida según `reports/auditoria_matematica.md`; las cifras salen del pipeline regenerado desde los datos crudos después de corregir el golpeado (`historial/reports/validacion_post_auditoria.md`). Cambios principales:
> - valor del golpeado (§4);
> - el "techo" deja de presentarse como cota (§2);
> - atribución de Shapley del efecto de altitud (§5);
> - intervalo de credibilidad y escala (§6);
> - el índice neutral no supera al ajuste trivial (§6);
> - nombre de Stuff+ v1 (§7);
> - nueva §9.

## 0. Notación

| símbolo | significado |
|---|---|
| i | pitch |
| p, y | pitcher, temporada (2024, 2025, 2026) |
| xᵢ | features del pitch |
| cᵢ ∈ {N, M, E} | categoría de altitud (No, Medium, Extreme) |
| (bᵢ, sᵢ) | conteo (bolas, strikes) antes del pitch |
| eᵢ ∈ 𝓔 | evento observado del pitch |
| rvᵢ | run value RE24 del pitch (≠ 0 solo en el pitch que termina el turno) |
| n_py | pitches del pitcher p en la temporada y |

Universo: 627,736 pitches válidos, 1,045 pitchers (627,185 en la cadena de eventos).

---

## 1. Línea base v1

**Stuff+ v1.** Un solo modelo: p̂ᵢ = P̂(whiffᵢ = 1 | xᵢ^stuff), con LightGBM y 23 features físicas. Se entrena sobre **todos** los pitches.

**Pitching+ v1.** Ocho modelos independientes, uno por target k ∈ {whiff, chase, called strike, weak contact, groundball, hard contact, out en PA, rv}:

  m̄_py,k = (1/n_py) Σᵢ ŷᵢ,k
  z_py,k = ± (m̄_py,k − mean_y) / sd_y        (signo − para rv y hard contact)
  Pitching+ = 100 + 10 · Σₖ wₖ · z_py,k

Los pesos wₖ se eligieron a mano (0.20 / 0.12 / 0.08). Resultado: sd = 6.32, no 10, porque la suma ponderada de z-scores no se vuelve a estandarizar.

---

## 2. Benchmark año-contra-año (Paso 1)

Para cada métrica m, sobre pares (p, y → y+1) con n ≥ n_min en ambas temporadas:

  Estabilidad:          r_stab(m)   = corr(m_p,y , m_p,y+1)
  Poder predictivo:     r_pred(m,o) = corr(m_p,y , o_p,y+1)

Resultado principal: o = carreras evitadas por 100 pitches = −100 · Σᵢ rvᵢ / n_py.

- **Inferencia:** bootstrap por pitcher, B = 2,000. Se remuestrean pitchers con reemplazo, no filas.
- **Comparación entre métricas:** Δr = r(m₁) − r(m₂), con bootstrap pareado (mismo remuestreo para ambas).
- **Criterio de aceptación** (fijado antes de ver resultados): P̂(Δr > 0) ≥ 0.90.
- **Pares disponibles:** 437 con n_min = 100; 245 con n_min = 300.

Línea base (n_min = 100):

| métrica | r_stab | r_pred (carreras t+1) | r_pred (whiff% t+1) |
|---|---:|---:|---:|
| Stuff+ v1 | 0.668 | 0.181 | 0.362 |
| Pitching+ v1 | 0.545 | 0.169 | 0.172 |
| CSW% observado | 0.349 | 0.205 | 0.357 |
| Whiff% observado | 0.487 | 0.196 | 0.487 |

**Relación entre estabilidad y poder predictivo (no es un techo universal).** Modelo de medición clásico: o_p,y = T_p,y + e_p,y, con errores independientes entre años y de la métrica m, y la misma confiabilidad rel(o) = Var(T)/Var(o) en ambos años. Entonces:

  corr(m_y, o_y+1) = corr(m_y, T_y+1) · √rel(o) ≤ √rel(o)
  r_stab(o) = corr(o_y, o_y+1) = ρ_T · rel(o),    ρ_T = corr(T_y, T_y+1)
  ⇒ r_pred ≤ √(r_stab(o) / ρ_T)

La cota es √r_stab(o) **solo** si el talento no cambia entre años (ρ_T = 1). Si cambia (ρ_T < 1), la cota real es más alta. Valores observados:
- carreras observadas: r_stab(o) = 0.08 (n ≥ 100) y 0.16 (n ≥ 300);
- carreras ajustadas por entorno: 0.17 y 0.27.

Se usan como orientación del orden de magnitud de r_pred alcanzable, no como cota.

---

## 3. Features v2 (Paso 2)

**Espejo de zurdos.** Sea sᵢ = −1 si el pitcher es zurdo y +1 si es derecho:

  HB, RelSide, HorzRelAngle, HorzApprAngle, res_Magnus,h, hb_diff, PlateLocSide  ←  sᵢ · (·)
  SpinAxis ← 360° − SpinAxis   (solo zurdos)

**Ubicación escondida.** corr(VAA, PlateLocHeight) = 0.78 y corr(HAA, PlateLocSide) = 0.58. Para cada tipo de pitch t se ajusta por OLS:

  VAA = a_t + b_t · PlateLocHeight + ε     ⇒     VAA_adj = VAA − (â_t + b̂_t · PlateLocHeight)

(y lo mismo para HAA con PlateLocSide). VAA_adj es el ángulo de llegada a ubicación constante.

**Resultado.** El AUC de P(whiff) baja de 0.732 a 0.640 al quitar la ubicación escondida. La mayor parte de la señal de Stuff+ v1 era ubicación.

---

## 4. Pitching+ v3: cadena de probabilidades valuada en carreras (Paso 3)

**Eventos.** 𝓔 = {ball, called strike, hbp, whiff, foul, out, 1B, 2B, 3B, HR}. Su probabilidad se factoriza con 5 modelos condicionales, cada uno entrenado en su propio universo:

  P(ball | x)  = (1 − π_sw) · π_take(ball)              (igual para called strike y hbp)
  P(whiff | x) = π_sw · π_wh
  P(foul | x)  = π_sw · (1 − π_wh) · π_f
  P(k | x)     = π_sw · (1 − π_wh) · (1 − π_f) · π_bip(k),    k ∈ {out, 1B, 2B, 3B, HR}

donde π_sw = P(swing | x), π_wh = P(whiff | swing, x) y π_f = P(foul | contacto, x). Por construcción Σ_e P(e | x) = 1.

**Valores terminales.** v_T = E[rv | el turno termina con T]:

| evento | carreras |
|---|---:|
| BB | +0.378 |
| HBP | +0.397 |
| K | −0.334 |
| out | −0.308 |
| 1B | +0.512 |
| 2B | +0.836 |
| 3B | +1.041 |
| HR | +1.472 |

Antes de la auditoría, HBP valía +0.004: `is_hit_by_pitch` venía en 0 en todo el dataset, así que el golpeado no cerraba el turno. Ahora se reconstruye desde PitchCall, y Δ(hbp, b, s) > 0 en los 12 conteos (`historial/reports/hbp_antes_despues.md`).

**Valor del conteo.** Recursión hacia atrás sobre los 12 conteos, con π_bs(e) = frecuencia empírica del evento e en el conteo (b, s):

  V(b, s) = Σₑ π_bs(e) · v_next(e, b, s)

Con 2 strikes, el foul deja el conteo igual (auto-lazo), así que:

  V(b, 2) = Σ_{e ≠ foul} π_b2(e) · v_next(e) / (1 − π_b2(foul))

Chequeo: V(0, 0) = +0.0147. Coincide con el promedio observado de rv por turno, pero eso es casi una **identidad**: con frecuencias empíricas por conteo y valores terminales promedio de la misma muestra, la recursión reproduce la media por turno por construcción. No es una validación independiente. Las pruebas reales son la ecuación de Bellman en cada conteo (residuo ≈ 1e−17), la monotonía (V baja con cada strike y sube con cada bola) y los signos de Δ en los 12 conteos (`tests/test_math.py`, `tests/test_targets.py`).

**Valor de cada evento.**

  Δ(e, b, s) = v_next(e, b, s) − V(b, s)

**Valor esperado de un pitch y score.**

  E[Δᵢ] = Σₑ P̂(e | xᵢ) · Δ(e, bᵢ, sᵢ)
  Pitching+ v3 = 100 + 10 · z_y(−x̄_py),    con x̄_py = promedio de E[Δᵢ] de los pitches de p en y

Para un índice neutral al conteo se usa Δ̄(e) = Σ_bs w_bs · Δ(e, b, s), donde w_bs es la frecuencia del conteo (b, s).

**Descomposición de la varianza.** Con Var(x̄) = Σ_g Cov(x̄_g, x̄), el componente de bola en juego explica el 66% de la varianza de Stuff+ v3. Ese componente tiene r_stab = 0.71 pero r_pred = 0.04: es el fenómeno DIPS.

**Resultado.** Pitching+ v3 tiene poder predictivo equivalente a v1 (0.154 vs. 0.169 con n ≥ 100; 0.227 vs. 0.194 con n ≥ 300). Se adoptó por metodología, no por mejora demostrada.

---

## 5. Contrafactual de trayectoria completa (Paso 4)

**Modelo cinemático.** Trackman da posición, velocidad y aceleración en y = 50 pies (9 parámetros). Se supone aceleración constante:

  r(t) = r₀ + v₀ t + ½ a t²

Validación: reproduce PlateLoc, ZoneSpeed, ZoneTime, VAA, HAA, RelHeight y RelSide con r ≥ 0.997, e IVB, HB y VB con r ≥ 0.995.

**Descomposición drag / Magnus.** Con g = 32.174 ft/s² y v̂ = v₀ / |v₀|:

  a_ng = a + g ẑ
  D = (a_ng · v̂) v̂          (drag, paralelo a la velocidad)
  M = a_ng − D              (Magnus, perpendicular)

**Densidad efectiva medida.** Como |D| ∝ ρ C_D |v|² y |M| ∝ ρ C_L |v|², se definen k_D = |D| / |v|² y k_M = |M| / |v|². Para cada grupo g = (pitcher, temporada, tipo de pitch) con ≥ 15 pitches en ambas categorías:

  ρ̂_X(c) = median_g [ k̃_X(g, c) / k̃_X(g, N) ],    X ∈ {D, M},   k̃ = mediana dentro de la celda

| | ρ̂_D | ρ̂_M | supuesto barométrico en v1 |
|---|---:|---:|---:|
| M vs. N | 0.906 | 0.834 | 0.930 |
| E vs. N | 0.854 | 0.770 | 0.834 |

Los IC son de ±0.003. Que ρ̂_M < ρ̂_D contradice "solo densidad", porque ambos deberían escalar igual con ρ.

**Proyección de la categoría c a la categoría c′.**

  a′ = D · ρ̂_D(c′) / ρ̂_D(c)  +  M · ρ̂_M(c′) / ρ̂_M(c)  −  g ẑ

1. Se calcula el estado de salida (r_r, v_r) en y_r = 60.5 − Extension.
2. El tiempo de vuelo T resuelve y_r + v_ry T + ½ a′_y T² = 17/12.
3. Supuesto de comando (mismo objetivo en el plato):

  v_rx = (x_p − x_r − ½ a′_x T²) / T        (igual para z)

4. Las features se recalculan y se aplican como diferencias sobre lo observado:

  f′ = f_obs + [ f̂(a′) − f̂(a) ],    con IVB = ½ (a_z + g) T² · 12,   HB = −½ a_x T² · 12

Identidad: si c′ = c, max |f′ − f| = 1 × 10⁻⁵.

**Validación split-half.** Los ratios se estiman con la mitad de los pitchers y se predice el cambio N → E en la otra mitad (1,061 grupos):

| feature | cambio observado | cambio predicho |
|---|---:|---:|
| IVB (pulg.) | −2.20 | −2.64 |
| ZoneSpeed (mph) | +1.24 | +1.20 |
| VAA (°) | −0.165 | −0.156 |

**Efecto de la altitud.** Para cada pitch:

  δᵢ = E[Δᵢ | proyectado a E] − E[Δᵢ | proyectado a N]

**Atribución por eslabón (Shapley).** E[Δ] = f(π_sw, π_take, π_wh, π_f, π_bip) es multilineal en las probabilidades de los 5 sub-modelos. El cambio δ se reparte con

  φ_j = Σ_{S ⊆ F∖{j}} |S|!(5 − |S| − 1)!/5! · [f(S ∪ {j}) − f(S)],    Σ_j φ_j = δ

donde f(S) evalúa E[Δ] con los eslabones de S en Extreme y el resto en No Altitude. Es una atribución contable entre dos predicciones del modelo, **no causal**. La versión anterior repartía por grupo de eventos y llamaba "calidad del pitch" a take + whiff + foul. Eso omitía que menos whiffs implican más pelotas en juego: δ_bip = ΔP(bip)·v̄_N + P_E(bip)·(v̄_E − v̄_N), y el primer término lo causa el pitch.

En carreras por 100 pitches:

| | total | antes del contacto (swing + take + whiff + foul) | resultado de la bola en juego |
|---|---:|---:|---:|
| Four-Seam | +1.45 | +0.52 | +0.93 |
| Slider | +0.65 | +0.27 | +0.38 |
| Todos | +1.10 | +0.39 | +0.71 |

El resultado de la bola en juego mezcla calidad del contacto con el entorno del parque; con este modelo no se separan. La brecha realizada E − N es +1.48; la que atribuye el modelo, +1.10 (`reports/altitude_counterfactual_v2.md`).

---

## 6. Índice oficial: Pitching+ v3 neutral con shrinkage (Paso 5)

**Score neutral por pitcher-temporada** (predicción OOF, GroupKFold por pitcher con k = 5):

  x̄_py = (1/n_py) Σᵢ E[Δᵢ | proyectado a N]

**Shrinkage empírico-bayesiano** (modelo normal-normal, por temporada). Se usan pitcher-temporadas con n ≥ 100 para estimar los parámetros:

  σ² = varianza ponderada de E[Δ] dentro de cada pitcher
  μ_y = Σ n · x̄ / Σ n
  τ²_y = Var(x̄) − E[σ² / n]

  B_py = (σ² / n_py) / (σ² / n_py + τ²)
  θ̂_py = μ + (1 − B_py)(x̄_py − μ)
  sd_post = √((1 − B_py) · σ² / n_py)

**Escala e intervalo.**

  Pitching+ = 100 + 10 · (μ − θ̂_py) / τ
  intervalo de credibilidad EB de 80% = Pitching+ ± 1.2816 · 10 · sd_post / τ

**Escala.** 100 = μ, la media de E[Δ] **ponderada por pitches** de los pitchers con n ≥ 100 (el pitch promedio de la liga). El promedio simple por pitcher es ≈ 98–99, porque los pitchers de más volumen son mejores y los de n < 100 peores. La sd del índice (≈ 8.7) es < 10 por diseño: 10 puntos = 1 τ y θ̂ está encogido (`reports/escala_pitching_plus.md`).

**Intervalo.** Es un intervalo de **credibilidad** empírico-bayesiano *plug-in*: la posterior de θ con μ, τ y σ fijos en sus estimaciones. Refleja el tamaño de muestra del pitcher. **No incluye** la incertidumbre de μ, τ y σ, de los modelos ML, de ρ̂, del contrafactual ni de los hiperparámetros, ni la dependencia entre pitches. No es un intervalo de confianza.

Valores: τ = 0.0065 / 0.0059 / 0.0065, σ ≈ 0.054, y n_½ = σ² / τ² = 67 / 86 / 69 pitches (2024 / 2025 / 2026).

**Prueba de neutralización.** corr(score, % de pitches en E) pasa de −0.324 (con altitud) a −0.062 (neutral).

**Resultado ajustado por entorno.** Para cada pitch:

  ỹᵢ = Δᵢ^realizado − mean(Δ^realizado | y, cᵢ)

y o_py = −100 · ȳ̃_py.

Esta prueba es **secundaria y post-hoc** (se definió después de ver la preespecificada). Incluye el baseline obligatorio B1: E[Δ] observado menos su media por (temporada, categoría), sin física (`reports/baselines.md`).

| n_min | con altitud (B0) | ajuste trivial (B1) | neutral físico (A) | A − B0 (P) | A − B1 (P) |
|---|---:|---:|---:|---|---|
| 100 | 0.165 | 0.191 | 0.181 | +0.016 (78%) | −0.010 (20%) |
| 300 | 0.215 | 0.279 | 0.279 | +0.065 (99%) | **+0.000 (53%)** |

**La ventaja del índice neutral sobre el índice con altitud la reproduce un ajuste trivial por entorno.** El contrafactual físico no aporta valor predictivo adicional. Con entrenamiento temporal estricto (todo con temporadas < t, pareja 2025→2026) tampoco lo supera (`reports/validacion_temporal.md`).

**Validación preespecificada (§7.2: carreras observadas t+1, n ≥ 100, contra Pitching+ v1, P ≥ 0.90):** ningún score la cumple. Pitching+ v1 0.188; con altitud 0.177; trivial 0.169; neutral 0.153 (Δ = −0.035, P = 20%).

---

## 7. Stuff+

**Stuff+ v4.** π_wh = P(whiff | swing, x^v2d), sin ubicación ni conteo, con AUC = 0.651 sobre swings. Se proyecta a N y lleva el mismo shrinkage que Pitching+.

| | Stuff+ v4 | Stuff+ v1 |
|---|---:|---:|
| Carreras ajustadas t+1 | 0.149 | 0.184 |
| P(v4 mejor) | 21% | — |
| r_stab | 0.769 | 0.668 |

No se adoptó. Su sesgo estructural: E[π_wh | sinker] = 0.15 vs. E[π_wh | curva] = 0.34.

**Stuff+ v1 (`stuff_plus_v1`, modelo predictivo de whiff).** Sus features incluyen ángulos de salida y llegada (ubicación implícita), `altitude_level` (la categoría de altitud) y residuales de Magnus. **No es independiente de la ubicación ni de la altitud.** Se escala sin shrinkage, sobre todas las pitcher-temporadas:

  100 + 10 · z_y(p̄_py),    con p̄_py = promedio OOF de P̂(whiff)

Stuff+ v4 se usa solo dentro de cada tipo de pitch, como 100 · (π̄_wh,p,t − π̄_wh,t) en puntos porcentuales.

---

## 8. Supuestos y límites (forma matemática)

1. **Aceleración constante.** Los ratios ρ̂ son una mezcla de densidad, calibración por estadio, clima y pelota. Son efectivos, no físicos.
2. **Ignorabilidad.** δᵢ supone que el modelo es invariante a la altitud una vez que se condiciona en la trayectoria. Pero el modelo detecta c a través de (D, M), así que δ_bip mezcla calidad del contacto con entorno del parque.
3. **Incertidumbre subestimada.** Los IC del efecto de altitud solo capturan el muestreo de pitchers: el modelo es uno solo, con predicciones in-sample.
4. **Potencia limitada.** Con unos 440 pares y r_stab(o) ≈ 0.1, los IC de Δr son de alrededor de ±0.07.

---

## 9. Validación post-auditoría (2026-10-05)

Detalle en `historial/reports/validacion_post_auditoria.md`.

1. **HBP corregido.** v_HBP = +0.397, Δ(hbp) > 0 en los 12 conteos. Pitching+ cambia con correlación de 0.988 y como máximo 5.9 puntos.
2. **Baseline obligatorio.** El neutral físico (A) no supera al ajuste trivial temporada × categoría (B1): A − B1 = 0.000 (P = 53%) con n ≥ 300.
3. **Criterio preespecificado (§7.2):** no se cumple.
4. **Leakage temporal.** Con todo entrenado en temporadas < t (incluida la matriz RE24), no hay señal de inflación. Única excepción documentada: los hiperparámetros son globales.
5. **Estadios.** "Extreme" agrupa 4–5 estadios locales reconstruidos (`src/data/parks.py`) con ratios de drag de 0.81 a 0.90. El el estadio de altitud extrema no es identificable sin información externa.
6. **Hipótesis del analista** (efectos fijos de pitcher, bateador, temporada y conteo, en Extreme vs. No Altitude):
   - whiff por swing −0.029, bola en juego por swing +0.040, ponche por turno −0.047;
   - run value por bola en juego +0.068;
   - G_E − G_N = +0.046 carreras (valor de un whiff frente a una bola en juego);
   - consistente en 13/13 estadio-temporada.
   Es asociación condicional a esos estadios; Medium se comporta casi igual.
7. **Sensibilidad.** La sensibilidad individual a la altitud observada tiene confiabilidad 0.11. La que implica el modelo no la predice (r = 0.06) y depende en un 44% de la mezcla de pitches.
