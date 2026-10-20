# Analisis SHAP — Stuff+ v1 (is_whiff)

Metodologia: `shap.TreeExplainer` sobre TODO el set de test de `pitcher_holdout` (124,596 pitches, pitchers nunca vistos en train). Valor base (log-odds, clase whiff=1): -2.3537.

## 1. Importancia global (mean |SHAP value|)

Figura: `reports/figures/shap/summary_global.png`

| # | feature | mean |SHAP| |
|---:|---|---:|
| 1 | VertApprAngle | 0.2985 |
| 2 | HorzRelAngle | 0.2861 |
| 3 | VertRelAngle | 0.2709 |
| 4 | RelSide | 0.1840 |
| 5 | RelHeight | 0.1208 |
| 6 | AutoPitchType | 0.1028 |
| 7 | VertBreak | 0.0878 |
| 8 | HorzApprAngle | 0.0679 |
| 9 | SpinRate | 0.0517 |
| 10 | magnus_residual_horz | 0.0386 |
| 11 | SpeedDrop | 0.0340 |
| 12 | ZoneSpeed | 0.0282 |
| 13 | InducedVertBreak | 0.0280 |
| 14 | HorzBreak | 0.0274 |
| 15 | magnus_residual_vert | 0.0261 |
| 16 | SpinAxis | 0.0261 |
| 17 | BatterSide | 0.0225 |
| 18 | RelSpeed | 0.0215 |
| 19 | EffectiveVelo | 0.0158 |
| 20 | Extension | 0.0143 |
| 21 | ZoneTime | 0.0121 |
| 22 | PitcherThrows | 0.0112 |
| 23 | altitude_level | 0.0076 |

Top 5 global: VertApprAngle, HorzRelAngle, VertRelAngle, RelSide, RelHeight. Consistente con el ranking de `feature_importances_` (split count) de LightGBM reportado en `historial/reports/stuff_plus_baseline.md` — dominan angulos de release/aproximacion y posicion de release, no solo velocidad/spin crudos.

## 2. Summary plots por AutoPitchType

Tipos con n>=1,000 en test: Four-Seam, Slider, Sinker, Changeup, Curveball, Cutter, Splitter.

- Four-Seam (n=42,598): `reports/figures/shap/summary_by_pitchtype/Four-Seam.png`
- Slider (n=28,453): `reports/figures/shap/summary_by_pitchtype/Slider.png`
- Sinker (n=21,426): `reports/figures/shap/summary_by_pitchtype/Sinker.png`
- Changeup (n=14,125): `reports/figures/shap/summary_by_pitchtype/Changeup.png`
- Curveball (n=10,263): `reports/figures/shap/summary_by_pitchtype/Curveball.png`
- Cutter (n=5,941): `reports/figures/shap/summary_by_pitchtype/Cutter.png`
- Splitter (n=1,686): `reports/figures/shap/summary_by_pitchtype/Splitter.png`

Top 5 features por tipo de pitch (mean |SHAP| dentro de ese subgrupo):

| AutoPitchType | top 5 features |
|---|---|
| Four-Seam | HorzRelAngle, VertApprAngle, VertRelAngle, RelSide, VertBreak |
| Slider | VertApprAngle, VertRelAngle, HorzRelAngle, RelSide, RelHeight |
| Sinker | HorzRelAngle, VertApprAngle, VertRelAngle, RelSide, AutoPitchType |
| Changeup | HorzRelAngle, VertApprAngle, VertRelAngle, RelSide, RelHeight |
| Curveball | VertApprAngle, VertRelAngle, HorzRelAngle, RelHeight, RelSide |
| Cutter | HorzRelAngle, VertApprAngle, VertRelAngle, AutoPitchType, RelSide |
| Splitter | VertApprAngle, VertRelAngle, HorzRelAngle, RelSide, RelHeight |

El top-5 global se mantiene parcialmente por tipo de pitch (overlap con el top-5 de cada tipo: {'Four-Seam': 4, 'Slider': 5, 'Sinker': 4, 'Changeup': 5, 'Curveball': 5, 'Cutter': 4, 'Splitter': 5}). El tipo con menos coincidencia es **Four-Seam** (4/5 features en comun con el top-5 global) — revisar su summary plot individual para ver que feature(s) distinta(s) domina(n) ahi.

## 3. Dependence plots

### InducedVertBreak

Figura: `reports/figures/shap/dependence_InducedVertBreak.png`

mean |SHAP| = 0.0280. Correlacion (valor de la feature vs. su SHAP value) = -0.004 — relacion debil/no-lineal, en promedio a mayor InducedVertBreak el modelo disminuye la probabilidad predicha de whiff (la correlacion es un resumen lineal; el dependence plot puede mostrar no-linealidades que esto no captura).

### HorzBreak

Figura: `reports/figures/shap/dependence_HorzBreak.png`

mean |SHAP| = 0.0274. Correlacion (valor de la feature vs. su SHAP value) = -0.137 — relacion debil/no-lineal, en promedio a mayor HorzBreak el modelo disminuye la probabilidad predicha de whiff (la correlacion es un resumen lineal; el dependence plot puede mostrar no-linealidades que esto no captura).

### SpinRate

Figura: `reports/figures/shap/dependence_SpinRate.png`

mean |SHAP| = 0.0517. Correlacion (valor de la feature vs. su SHAP value) = 0.424 — relacion moderada, en promedio a mayor SpinRate el modelo aumenta la probabilidad predicha de whiff (la correlacion es un resumen lineal; el dependence plot puede mostrar no-linealidades que esto no captura).

### RelSpeed

Figura: `reports/figures/shap/dependence_RelSpeed.png`

mean |SHAP| = 0.0215. Correlacion (valor de la feature vs. su SHAP value) = 0.391 — relacion moderada, en promedio a mayor RelSpeed el modelo aumenta la probabilidad predicha de whiff (la correlacion es un resumen lineal; el dependence plot puede mostrar no-linealidades que esto no captura).

### HorzRelAngle

Figura: `reports/figures/shap/dependence_HorzRelAngle.png`

mean |SHAP| = 0.2861. Correlacion (valor de la feature vs. su SHAP value) = 0.089 — relacion debil/no-lineal, en promedio a mayor HorzRelAngle el modelo aumenta la probabilidad predicha de whiff (la correlacion es un resumen lineal; el dependence plot puede mostrar no-linealidades que esto no captura).

### RelSide

Figura: `reports/figures/shap/dependence_RelSide.png`

mean |SHAP| = 0.1840. Correlacion (valor de la feature vs. su SHAP value) = -0.210 — relacion moderada, en promedio a mayor RelSide el modelo disminuye la probabilidad predicha de whiff (la correlacion es un resumen lineal; el dependence plot puede mostrar no-linealidades que esto no captura).

### VertApprAngle

Figura: `reports/figures/shap/dependence_VertApprAngle.png`

mean |SHAP| = 0.2985. Correlacion (valor de la feature vs. su SHAP value) = -0.537 — relacion fuerte, en promedio a mayor VertApprAngle el modelo disminuye la probabilidad predicha de whiff (la correlacion es un resumen lineal; el dependence plot puede mostrar no-linealidades que esto no captura).

## 4. Verificacion cruzada contra el ablation (altitude_level, magnus_residual)

El ablation de Stuff+ (historial/reports/stuff_plus_baseline.md) mostro que quitar `altitude_level` y `magnus_residual_vert/horz` no le cuesta nada de AUC al modelo. SHAP deberia mostrar estas 3 features con importancia baja de forma consistente:

| feature | mean |SHAP| | ranking (de 23) |
|---|---:|---:|
| altitude_level | 0.0076 | 23 |
| magnus_residual_vert | 0.0261 | 15 |
| magnus_residual_horz | 0.0386 | 10 |

**DISCREPANCIA A INVESTIGAR:** magnus_residual_horz aparece en la mitad superior del ranking de SHAP (#10 de 23) a pesar de que el ablation no mostro perdida de AUC al quitarla. Investigacion de seguimiento (no generada automaticamente por este script — corrida manual):

- Correlacion de `magnus_residual_horz` con sus componentes fisicos, en el mismo set de test:
  `SpinAxis` = **-0.809**, `HorzBreak` = -0.625, `SpinRate` = 0.206, `RelSpeed` = -0.195.
- `mean|SHAP|` individual: magnus_residual_horz = 0.0386, HorzBreak = 0.0274. Pero la
  magnitud de la SUMA de sus SHAP values fila por fila (`|shap_mrh + shap_hb|`, promedio
  0.0559) es menor que la suma de sus magnitudes individuales (0.0660) — señal de
  cancelacion parcial tipica de credito repartido entre features correlacionadas, no de
  dos señales independientes que se refuercen.

**Conclusion: es redundancia, no una interaccion genuina que el ablation simple se haya
perdido.** `magnus_residual_horz` se construye (en `src/features/physics.py`) como
`HorzBreak - hb_theoretical`, y `hb_theoretical` es una funcion determinista de
`SpinAxis`, `SpinRate`, `RelSpeed`, `Extension` y la densidad de aire — TODAS ya presentes
en el modelo. La correlacion de -0.81 con `SpinAxis` confirma que gran parte de lo que
`magnus_residual_horz` le "dice" al arbol es una recombinacion suave de columnas que el
modelo ya puede partir directamente. LightGBM aprovecha esa combinacion pre-calculada
cuando esta disponible (de ahi el SHAP individual no trivial), pero al quitarla los
arboles simplemente vuelven a reconstruir splits equivalentes combinando `SpinAxis` +
`HorzBreak` + `SpinRate` + `RelSpeed` — de ahi que el AUC no se mueva en el ablation.
Esto es consistente con, no contradictorio con, la conclusion original: `magnus_residual`
no aporta señal NUEVA para predecir is_whiff, solo la re-empaqueta.
