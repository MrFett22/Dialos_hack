"""Ejecución reproducible del pipeline completo, de los datos crudos a la validación.

    raw → limpieza → targets → física → arsenal → splits → estadios reconstruidos
        → Pitching+ v1 OOF (referencia del benchmark) → benchmark / Stuff+ v1 OOF
        → Stuff+ v2 (features) → cadena v3 (OOF) → modelos finales v3
        → contrafactual de altitud → índice neutral + shrinkage
        → baselines y validación preespecificada → validación temporal estricta
        → hipótesis del analista (H1–H3) → sensibilidad a la altitud
        → Stuff+ v4 → fichas de scouting → pruebas automáticas

Cada paso se corre como `python -m <módulo>` en un proceso nuevo y con
`--refresh` cuando el módulo tiene caché, así que ningún resultado intermedio
viejo se reutiliza. Se detiene en el primer paso que falle.

Excepción documentada: los hiperparámetros ajustados de Pitching+ v3
(src/models/artifacts/pitching_plus_v3_tuned_params.json) no se vuelven a
buscar en cada corrida. La búsqueda (src/models/tuning.py) optimiza el
log-loss de las etiquetas de eventos, que salen de PitchCall y no dependen de
run_value; volver a buscarlos es un paso aparte y opcional.

Salida: reports/pipeline_run.md (commit, hora, duración y estado de cada paso).

Uso:
    python -m src.pipeline                 # todo
    python -m src.pipeline --from chain    # retoma desde un paso
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "pipeline_run.md"

# (nombre, módulo, argumentos)
STEPS = [
    ("clean", "src.data.clean", []),
    ("targets", "src.features.targets", []),
    ("physics", "src.features.physics", []),
    ("arsenal", "src.features.arsenal", []),
    ("splits", "src.validation.splits", []),
    ("parks", "src.data.parks", []),
    ("pitching_v1_oof", "src.models.pitching_plus_oof", []),
    ("benchmark", "src.validation.benchmark", ["--refresh"]),
    ("stuff_v2", "src.models.stuff_plus_v2", ["--refresh"]),
    ("chain", "src.models.run_value_chain", ["--refresh"]),
    ("chain_final", "src.models.run_value_chain", ["--train-final"]),
    ("counterfactual", "src.analysis.altitude_counterfactual_v2", []),
    ("neutral", "src.models.pitching_plus_neutral", ["--refresh"]),
    ("scale", "src.validation.scale_check", []),
    ("baselines", "src.validation.baselines", []),
    ("temporal", "src.validation.temporal", []),
    ("hypotheses", "src.analysis.altitude_hypotheses", []),
    ("sensitivity", "src.analysis.sensitivity", []),
    ("stuff_v4", "src.models.stuff_plus_v4", ["--refresh"]),
    ("scouting", "src.reports.scouting_report_v2", []),
    ("stuff_physical", "src.models.stuff_plus_physical", ["--refresh"]),
    ("location_v2", "src.models.location_plus_v2", ["--refresh"]),
    ("profiles", "src.analysis.pitcher_profiles", []),
    ("p2_report", "src.reports.p2_report", []),
    ("web_data", "src.reports.web_data", []),
    ("pitch_environment", "src.analysis.pitch_environment", []),
    ("tests", "pytest", ["-q", "tests"]),
]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + (" (con cambios sin commit)" if dirty else "")
    except OSError:
        return "desconocido"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s[0] for s in STEPS])
    args = parser.parse_args()
    names = [s[0] for s in STEPS]
    steps = STEPS[names.index(args.start):]

    started = datetime.now()
    rows = []
    ok = True
    for name, module, extra in steps:
        cmd = [sys.executable, "-m", module, *extra]
        print(f"\n=== [{name}] {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        dt = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FALLÓ (código {result.returncode})"
        rows.append((name, " ".join(cmd[2:]), dt, status))
        print(f"=== [{name}] {status} en {dt:.0f}s", flush=True)
        if result.returncode != 0:
            ok = False
            break

    lines = [
        "# Ejecución del pipeline",
        "",
        f"- Inicio: {started:%Y-%m-%d %H:%M} · commit `{git_commit()}` · Python {sys.version.split()[0]}",
        f"- Desde el paso: `{args.start}` · resultado: **{'completo' if ok else 'interrumpido'}**",
        "",
        "| paso | comando | duración | estado |",
        "|---|---|---:|---|",
    ]
    lines += [f"| {n} | `{c}` | {d / 60:.1f} min | {s} |" for n, c, d, s in rows]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
