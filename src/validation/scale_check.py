"""¿Por qué la media de Pitching+ no es 100? Descomposición de la escala (auditoría P1.5).

Definición implementada (src/models/pitching_plus_neutral.py · empirical_bayes):

    Pitching+_p = 100 + 10·(μ − θ̂_p)/τ
    μ  = Σ n_p x̄_p / Σ n_p        sobre pitchers con n_p ≥ 100 (media POR PITCH)
    θ̂_p = μ + (1 − B_p)(x̄_p − μ)   (shrinkage)

Por construcción, la media de la versión sin shrinkage, PONDERADA por pitches
y sobre pitchers con n ≥ 100, es exactamente 100. Cualquier otro promedio
puede diferir por:
  (a) ponderar por pitcher en vez de por pitch (los de más volumen son mejores);
  (b) incluir a los pitchers con n < 100 (peores en promedio);
  (c) el shrinkage (mueve θ̂ hacia μ, más a los de poco volumen).
La neutralización NO cambia la definición: el índice con altitud usa la misma
construcción, con su propio μ y τ.

Salida: reports/escala_pitching_plus.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import pitching_plus_neutral as pn

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "escala_pitching_plus.md"


def run() -> None:
    off = pd.read_parquet(pn.OFFICIAL_PATH)
    rows = []
    for y, g in off.groupby("year"):
        big = g[g["n_pitches"] >= 100]
        rows.append({
            "temporada": int(y),
            "sin shrinkage, ponderada por pitches, n≥100 (definición)": np.average(big["pitching_plus_sin_shrinkage"], weights=big["n_pitches"]),
            "sin shrinkage, por pitcher, n≥100": big["pitching_plus_sin_shrinkage"].mean(),
            "con shrinkage, ponderada por pitches, n≥100": np.average(big["pitching_plus"], weights=big["n_pitches"]),
            "con shrinkage, por pitcher, n≥100": big["pitching_plus"].mean(),
            "con shrinkage, por pitcher, todos": g["pitching_plus"].mean(),
            "con shrinkage, por pitcher, n<100": g.loc[g["n_pitches"] < 100, "pitching_plus"].mean(),
            "sd con shrinkage, n≥100": big["pitching_plus"].std(),
            "sd sin shrinkage, n≥100": big["pitching_plus_sin_shrinkage"].std(),
            "corr(volumen, score) n≥100": big["n_pitches"].corr(big["pitching_plus_sin_shrinkage"], method="spearman"),
        })
    t = pd.DataFrame(rows).set_index("temporada").T
    L = [
        "# Escala de Pitching+: ¿por qué el promedio no es exactamente 100?",
        "",
        "Generado por `src/validation/scale_check.py` (definición en el docstring).",
        "",
        t.round(2).to_markdown(),
        "",
        "## Lectura",
        "",
        "- **100 = el pitch promedio de la liga**: la media de E[Δ] ponderada por pitches entre pitchers con ≥100 pitches. "
        "Con esa ponderación y sin shrinkage el promedio es exactamente 100 (primera fila).",
        "- El promedio **por pitcher** queda por debajo de 100 porque los pitchers con más volumen son mejores "
        "(correlación de rangos volumen–score positiva, última fila) y porque los de menos de 100 pitches son peores en promedio.",
        "- El **shrinkage** casi no mueve el promedio ponderado; lo que hace es acercar a 100 a los de poca muestra.",
        "- La **neutralización** no es la causa: el índice con altitud tiene la misma construcción.",
        "- La sd del índice es menor que 10 **por diseño**: 10 puntos = 1 τ (la desviación de la habilidad real), y θ̂ está "
        "encogido, así que su dispersión es menor que τ. La versión sin shrinkage tiene sd > 10 porque incluye ruido de muestreo.",
        "",
        "## Decisión",
        "",
        "**Se conserva la escala (opción A).** La definición es explícita y coincide con la convención de los índices \"+\" "
        "(100 = promedio de la liga por pitch). Recalibrar para que el promedio por pitcher sea 100 sería cosmético y haría que "
        "el valor 100 dependiera de cuántos pitchers de poco volumen hay en la temporada. Lo que cambia es la documentación: "
        "\"100 = pitch promedio de la liga\", no \"100 = pitcher promedio\".",
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    run()
