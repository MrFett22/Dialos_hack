# Historial (material de fases anteriores)

Material que ya no forma parte de la entrega pero que documenta cómo se llegó a ella. No se borró nada: se movió aquí con `git mv`
para conservar la historia. La bitácora que explica cada decisión está en `reports/resultados_finales.md` §7.

| ruta | qué es | por qué se archivó |
|---|---|---|
| `AUDITORIA_INICIAL.md` | inventario y reproducción del proyecto antes de la reformulación | reemplazado por `reports/auditoria_matematica.md` y el documento técnico |
| `notebooks/01_eda.ipynb` | exploración inicial de los datos | el pipeline reproducible (`src/`) la reemplaza |
| `dashboard/app.py` | dashboard v3 (índice neutral físico) | reemplazado por la página `web/app.py`; `dashboard/scoring.py` sigue en uso |
| `reports/documento_tecnico_v3.md` | documento técnico v3.2 (antes de P0–P2) | reemplazado por `reports/documento_tecnico.md` / `.pdf` |
| `reports/*_baseline.md`, `pitching_plus_score.md`, `shap_findings.md` | modelos v1 (Stuff+, Location+, Pitching+) | superados por Stuff+ Physical, Location+ v2 y la cadena de Pitching+ |
| `reports/altitude_counterfactual*.md`, `altitude_whiff_stratified.md`, `runs_translation.md` | contrafactual v1 y análisis derivados | reemplazados por el contrafactual v2 y P0 |
| `reports/bayesian_hierarchical.md`, `causal_forest.md` | modelos causales exploratorios | no forman parte de las conclusiones finales |
| `reports/eda_summary.md`, `desc_stats_*.csv`, `notas_metodologicas.md`, `resumen_matematico.md`, `run_value_validation.md`, `validacion_post_auditoria.md`, `hbp_antes_despues.md`, `dashboard_sanity_check.md` | reportes intermedios | su contenido vigente está en los reportes de `reports/` |
| `reports/scouting/v1/` | fichas de scouting de la primera versión | reemplazadas por la ficha del pitcher de la página |
| `reports/figures/` | figuras de los reportes anteriores | acompañan a esos reportes |

Los módulos de `src/` que generaban estos reportes (por ejemplo `src/models/stuff_plus.py`, `bayesian_hierarchical.py`,
`causal_forest.py`) se conservan porque otros módulos los importan o porque documentan el proceso; no forman parte de
`src/pipeline.py`. Si se ejecutan a mano, escriben sus reportes en `reports/`.

Los nombres reales de equipos y estadios que aparecían en estos documentos se reemplazaron por descripciones genéricas, según la
regla de anonimización del proyecto.
