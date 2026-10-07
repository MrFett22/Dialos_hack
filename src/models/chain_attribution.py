"""Atribución del cambio en E[Δ] entre dos escenarios a los eslabones de la cadena.

Problema que resuelve. El efecto de la altitud sobre un pitch es
δ = E[Δ | escenario B] − E[Δ | escenario A]. La versión anterior lo repartía
por GRUPO DE EVENTOS (take, whiff, foul, bola en juego) y llamaba "calidad del
pitch" a take + whiff + foul. Eso es incorrecto: si el pitch provoca menos
whiffs, aumenta la probabilidad de que la pelota quede en juego, y ese cambio
aparecía en "bola en juego" aunque lo causó el pitch (auditoría 2026-10-05, C3).

Aquí se reparte por ESLABÓN de la cadena. E[Δ] es una función multilineal de
las probabilidades condicionales de los 5 sub-modelos:

    E[Δ] = f(π_sw, π_take, π_wh, π_f, π_bip)
         = (1 − π_sw)·Σ_t π_take(t)·Δ_t
           + π_sw·[π_wh·Δ_whiff + (1 − π_wh)·(π_f·Δ_foul + (1 − π_f)·Σ_k π_bip(k)·Δ_k)]

Para pasar de A a B cambian los 5 eslabones a la vez, y el orden en que se
"cambien" altera cuánto se le atribuye a cada uno. El valor de Shapley
promedia sobre todos los órdenes:

    φ_j = Σ_{S ⊆ F∖{j}} |S|!·(5 − |S| − 1)!/5! · [f(S ∪ {j}) − f(S)]

donde f(S) evalúa E[Δ] con los eslabones de S en el escenario B y el resto
en A. Propiedades: Σ_j φ_j = δ exactamente (eficiencia), y un eslabón que no
cambia recibe 0.

Lectura de cada componente:
- swing  : cambio en la decisión del bateador de tirarle;
- take   : cambio en bola / strike cantado / golpeado cuando no tira;
- whiff  : cambio en P(whiff | swing);
- foul   : cambio en P(foul | contacto);
- bip    : cambio en el resultado de la bola en juego (out, 1B, 2B, 3B, HR).
  Mezcla calidad del contacto y entorno del parque; los datos no permiten
  separarlos con este modelo.

Es una ATRIBUCIÓN contable del cambio en el valor esperado entre dos
predicciones del modelo, no una descomposición causal.
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from src.models import run_value_chain as chain

LINKS = ["swing", "take", "whiff", "foul", "bip"]
LINK_LABELS = {
    "swing": "decisión de swing",
    "take": "resultado del take (bola / strike cantado / golpeado)",
    "whiff": "whiff dado swing",
    "foul": "foul dado contacto",
    "bip": "resultado de la bola en juego (calidad del contacto + parque)",
}
PRE_CONTACT = ["swing", "take", "whiff", "foul"]


def submodel_probabilities(artifact: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Probabilidades de los 5 sub-modelos de un artefacto final (misma forma que en run_value_chain)."""
    X = df[artifact["numeric_features"] + artifact["categorical_features"]]
    out = {}
    for name, _, _, k in chain.SUBMODELS:
        p = artifact["models"][name].predict_proba(X)
        out[name] = p if k > 2 else p[:, 1:2]
    return out


def expected_value(preds: dict[str, np.ndarray], delta_matrix: np.ndarray) -> np.ndarray:
    return chain.expected_run_value(chain.event_probabilities(preds), delta_matrix)


def shapley(pa: dict[str, np.ndarray], pb: dict[str, np.ndarray], delta_matrix: np.ndarray) -> pd.DataFrame:
    """φ_j por pitch al pasar del escenario A (pa) al B (pb). Las columnas suman E[Δ|B] − E[Δ|A]."""
    n = len(LINKS)
    cache: dict[frozenset, np.ndarray] = {}

    def f(S: frozenset) -> np.ndarray:
        if S not in cache:
            cache[S] = expected_value({k: (pb[k] if k in S else pa[k]) for k in LINKS}, delta_matrix)
        return cache[S]

    out = {}
    for j in LINKS:
        others = [k for k in LINKS if k != j]
        acc = np.zeros(len(delta_matrix))
        for size in range(n):
            w = factorial(size) * factorial(n - size - 1) / factorial(n)
            for S in combinations(others, size):
                S = frozenset(S)
                acc += w * (f(S | {j}) - f(S))
        out[j] = acc
    return pd.DataFrame(out)[LINKS]
