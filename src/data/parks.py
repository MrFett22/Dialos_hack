"""Grupos de estadio ANÓNIMOS (ANON_STADIUM_xx) reconstruidos solo con el dataset.

Contexto. El dataset viene anonimizado: no trae estadio, equipo, local/visitante
explícito, elevación ni clima, solo `altitude_category` (No / Medium / Extreme)
por juego. Por decisión del proyecto (2026-10-05) se respeta por completo la
anonimización: NO se usa información externa (calendarios, nombres, número de
equipos de la liga) y NO se intenta identificar equipos, estadios ni pitchers
reales. Los grupos se nombran ANON_STADIUM_01, ANON_STADIUM_02, ...

Qué es un grupo: el conjunto de juegos que un mismo equipo (anónimo) jugó
como LOCAL. Si el equipo no cambió de casa, es su estadio. Se reconstruye sin
usar la altitud ni ningún resultado:

1. Equipos por temporada. Los compañeros de equipo lanzan y catchean del
   mismo lado del mismo juego. Grafo de jugadores (pitchers y catchers) con
   peso = log(1 + mitades de juego compartidas), agrupado con clustering
   espectral normalizado. El número de grupos k ∈ K_CANDIDATES se elige
   por temporada con un criterio interno: el que minimiza los juegos con
   local = visitante (empate → el k más chico). En la parte alta de cada
   entrada lanza el local, así que el equipo local de un juego es el cluster
   mayoritario entre los pitchers/catchers de la alta.
2. Enlace entre temporadas. Los clusters con juegos de local de cada
   temporada se asignan uno a uno (algoritmo húngaro) a los IDs existentes
   según el índice de Jaccard de sus rosters contra el roster más reciente
   de cada ID. Un enlace se acepta si es claro (Jaccard ≥ JACCARD_MIN y ≥
   JACCARD_MARGIN × el segundo mejor candidato); si no, el cluster recibe un
   ID nuevo. Los umbrales se fijaron mirando la distribución del Jaccard
   (enlaces reales 0.12–0.35 con segundo mejor ≈ 0.02–0.05; sin pareja ≈ 0),
   no la altitud.
3. Calidad. Un grupo-temporada es "resuelto" si ≥ 80% de sus juegos locales
   caen en una sola categoría de altitud y tiene ≥ 15 juegos locales; si no,
   es "mixto" (probablemente dos equipos fusionados) y sus juegos no se usan
   a nivel de estadio (solo a nivel de categoría).

La categoría de altitud se usa SOLO para validar (pureza, consistencia entre
temporadas), nunca para construir ni enlazar los grupos.

Validaciones de estabilidad:
- semilla: el mismo clustering con otras semillas de k-means (ARI);
- submuestreo: se reconstruye con el 80% de los juegos (ARI en los comunes);
- entre temporadas: ¿un grupo enlazado conserva su categoría de altitud?;
- física test–retest: ¿el ratio de drag de un grupo en una temporada predice
  el de la siguiente?

Física por grupo. Ratio de drag/v² y Magnus/v² del mismo pitcher, tipo y
temporada en el grupo contra el mismo pitcher en los OTROS estadios de No
Altitude (dejando fuera al propio grupo), mediana entre celdas (pitcher,
temporada, tipo) con ≥ 10 pitches en el grupo y ≥ 15 en la referencia.
Heterogeneidad dentro de cada categoría: sd entre grupos menos el error de
muestreo (momentos) e I².

Salidas:
- data/processed/anon_stadiums.parquet (un renglón por juego)
- data/processed/anon_stadium_ratios.parquet (ratios por grupo: total y por temporada)
- reports/estadios_anonimos.md
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

from src.features import trajectory as tr

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "stuff_model_df.parquet"
OUT_PATH = ROOT / "data" / "processed" / "anon_stadiums.parquet"
RATIOS_PATH = ROOT / "data" / "processed" / "anon_stadium_ratios.parquet"
REPORT_PATH = ROOT / "reports" / "estadios_anonimos.md"
JSON_PATH = ROOT / "reports" / "benchmark" / "estadios_anonimos.json"
K_CANDIDATES = (16, 18, 20, 22, 24)
MIN_HOME_GAMES = 15
PURITY_MIN = 0.8
JACCARD_MIN = 0.08
JACCARD_MARGIN = 2.5
MIN_CELL_STADIUM = 10
MIN_CELLS_RATIO = 20
N_BOOT = 500
SEED = 0
N_STABILITY_REPS = 5

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Equipos por temporada
# ---------------------------------------------------------------------------


def _spectral(A, k: int, seed: int = SEED) -> np.ndarray:
    """Clustering espectral normalizado (Ng–Jordan–Weiss) con k-means repetido; determinista por semilla."""
    d = np.asarray(A.sum(1)).ravel()
    dm = diags(1 / np.sqrt(d))
    _, vecs = eigsh(dm @ A @ dm, k=k, which="LA")
    X = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    best = None
    for s in range(10):
        cen, lab = kmeans2(X, k, seed=seed * 100 + s, minit="++")
        inertia = ((X - cen[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def _players(d: pd.DataFrame) -> pd.DataFrame:
    side = np.where(d["Top/Bottom"] == "Top", "home", "away")
    return pd.concat([
        pd.DataFrame({"game": d["game_anon_id"].to_numpy(), "side": side, "player": d["pitcher_anon_id"].to_numpy()}),
        pd.DataFrame({"game": d["game_anon_id"].to_numpy(), "side": side, "player": d["catcher_anon_id"].to_numpy()}),
    ]).dropna().drop_duplicates()


def reconstruct_season(d: pd.DataFrame, k: int, seed: int = SEED) -> tuple[pd.DataFrame, float, dict]:
    """(juego → cluster local/visitante, % de juegos con local = visitante, jugador → cluster)."""
    players = _players(d)
    pid = {p: i for i, p in enumerate(sorted(players["player"].unique()))}
    rows, cols = [], []
    for ps in players.groupby(["game", "side"])["player"].agg(list):
        ix = [pid[p] for p in ps]
        for i in ix:
            for j in ix:
                if i != j:
                    rows.append(i)
                    cols.append(j)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pid), len(pid))).tocsr()
    A.data = np.log1p(A.data)
    keep = np.asarray(A.sum(1)).ravel() > 0
    names = np.array(sorted(pid))[keep]
    team = dict(zip(names, _spectral(A[keep][:, keep], k, seed)))
    players["team"] = players["player"].map(team)
    major = players.dropna(subset=["team"]).groupby(["game", "side"])["team"].agg(lambda s: Counter(s).most_common(1)[0][0]).unstack("side")
    major = major.rename(columns={"home": "home_team", "away": "away_team"}).reset_index().rename(columns={"game": "game_anon_id"})
    same = float((major["home_team"] == major["away_team"]).mean())
    return major, same, team


def ari(a: pd.Series, b: pd.Series) -> float:
    """Índice de Rand ajustado entre dos particiones alineadas por índice (filas con NaN se descartan)."""
    ok = a.notna() & b.notna()
    c = pd.crosstab(a[ok], b[ok]).to_numpy().astype(float)
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    sum_ij, sum_a, sum_b, n = comb(c).sum(), comb(c.sum(1)).sum(), comb(c.sum(0)).sum(), comb(c.sum())
    expected = sum_a * sum_b / n
    return float((sum_ij - expected) / (0.5 * (sum_a + sum_b) - expected))


# ---------------------------------------------------------------------------
# 2. Enlace entre temporadas → ANON_STADIUM_xx
# ---------------------------------------------------------------------------


def link_seasons(seasons: dict[int, tuple[pd.DataFrame, dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asigna IDs estables por similitud de roster. Devuelve (juegos con stadium_id, tabla de enlaces).

    Cada temporada se compara contra el roster MÁS RECIENTE de cada ID (la unión acumulada diluiría el Jaccard,
    porque los rosters rotan mucho). La asignación es uno a uno (algoritmo húngaro sobre el Jaccard) y un enlace
    se acepta solo si es claro: Jaccard ≥ JACCARD_MIN y ≥ JACCARD_MARGIN × el segundo mejor candidato de ese cluster.
    Solo los clusters con juegos de local reciben ID.
    """
    latest_roster: dict[str, set] = {}
    links, frames = [], []
    next_id = 1
    for y in sorted(seasons):
        games, team = seasons[y]
        team_s = pd.Series(team)
        rosters = {t: set(g.index) for t, g in team_s.groupby(team_s)}
        home_clusters = games["home_team"].value_counts().index.tolist()  # orden determinista: más juegos locales primero
        ids = sorted(latest_roster)
        mapping = {}
        if ids:
            J = np.array([[len(rosters.get(t, set()) & latest_roster[i]) / max(len(rosters.get(t, set()) | latest_roster[i]), 1)
                           for i in ids] for t in home_clusters])
            rows, cols = linear_sum_assignment(-J)
            for r, c in zip(rows, cols):
                second = np.sort(J[r])[-2] if J.shape[1] > 1 else 0.0
                if J[r, c] >= JACCARD_MIN and J[r, c] >= JACCARD_MARGIN * second:
                    mapping[home_clusters[r]] = (ids[c], float(J[r, c]), float(second))
        for t in home_clusters:
            if t not in mapping:
                mapping[t] = (f"ANON_STADIUM_{next_id:02d}", np.nan, np.nan)
                next_id += 1
            sid, jac, second = mapping[t]
            links.append({"year": int(y), "cluster": int(t), "stadium_id": sid, "jaccard": jac, "jaccard_segundo": second})
            latest_roster[sid] = rosters.get(t, set())
        sid_of = {t: v[0] for t, v in mapping.items()}
        g = games.copy()
        g["year"] = int(y)
        g["stadium_id"] = g["home_team"].map(sid_of)
        g["away_stadium_id"] = g["away_team"].map(sid_of)
        frames.append(g)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(links)


# ---------------------------------------------------------------------------
# 3. Física por grupo
# ---------------------------------------------------------------------------


def _cells(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return d.groupby(keys, observed=True).agg(n=("kd", "size"), kd=("kd", "median"), km=("km", "median"))


def physics_frame(raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """k_drag y k_Magnus por pitch, con el grupo anónimo del juego (solo grupos resueltos)."""
    comp = tr.decompose_acceleration(raw)
    d = raw[["game_anon_id", "pitcher_anon_id", "year", "AutoPitchType", "altitude_category"]].copy()
    d["AutoPitchType"] = d["AutoPitchType"].astype(object)
    d["kd"], d["km"] = comp["drag"] / comp["speed"] ** 2, comp["magnus"] / comp["speed"] ** 2
    res = games.loc[games["status"] == "resuelto", ["game_anon_id", "stadium_id", "stadium_category"]]
    return d.merge(res, on="game_anon_id", how="left")


def stadium_ratios(d: pd.DataFrame, by_season: bool = False, seed: int = SEED) -> pd.DataFrame:
    """Ratio drag/Magnus de cada grupo resuelto contra el mismo pitcher × tipo × temporada en los OTROS estadios de No Altitude."""
    keys = ["pitcher_anon_id", "year", "AutoPitchType"]
    no_alt = d[d["altitude_category"] == "No Altitude"]
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in d.dropna(subset=["stadium_id"]).groupby(["stadium_id", "year"] if by_season else ["stadium_id"]):
        sid = key[0] if isinstance(key, tuple) else key
        ref = _cells(no_alt[no_alt["stadium_id"].ne(sid) | no_alt["stadium_id"].isna()], keys)
        ref = ref[ref["n"] >= tr.MIN_PITCHES_PER_CELL]
        m = _cells(g, keys)
        m = m[m["n"] >= MIN_CELL_STADIUM].join(ref, rsuffix="_ref", how="inner")
        if len(m) < MIN_CELLS_RATIO:
            continue
        rd, rm = (m["kd"] / m["kd_ref"]).to_numpy(), (m["km"] / m["km_ref"]).to_numpy()
        bd = np.array([np.median(rd[rng.integers(0, len(rd), len(rd))]) for _ in range(N_BOOT)])
        bm = np.array([np.median(rm[rng.integers(0, len(rm), len(rm))]) for _ in range(N_BOOT)])
        row = {"stadium_id": sid, "categoria": g["stadium_category"].mode().iat[0], "celdas": len(m),
               "drag_ratio": float(np.median(rd)), "drag_se": float(bd.std()),
               "drag_lo": float(np.percentile(bd, 2.5)), "drag_hi": float(np.percentile(bd, 97.5)),
               "magnus_ratio": float(np.median(rm)), "magnus_se": float(bm.std()),
               "magnus_lo": float(np.percentile(bm, 2.5)), "magnus_hi": float(np.percentile(bm, 97.5))}
        if by_season:
            row["year"] = int(key[1])
        rows.append(row)
    return pd.DataFrame(rows)


def heterogeneity(r: pd.DataFrame, col: str) -> pd.DataFrame:
    """sd entre grupos dentro de cada categoría, descontando el error de muestreo (momentos), e I²."""
    out = []
    se_col = col.replace("_ratio", "_se")
    for cat, g in r.groupby("categoria"):
        if len(g) < 2:
            continue
        v_obs = g[col].var()
        se2 = (g[se_col] ** 2).mean()
        tau2 = max(v_obs - se2, 0.0)
        out.append({"categoria": cat, "grupos": len(g), "min": g[col].min(), "max": g[col].max(), "sd_observada": np.sqrt(v_obs),
                    "error_muestreo": np.sqrt(se2), "sd_real": np.sqrt(tau2), "I2": tau2 / v_obs if v_obs > 0 else 0.0})
    return pd.DataFrame(out)


def load_stadium_ratios(path: Path = RATIOS_PATH) -> pd.DataFrame:
    r = pd.read_parquet(path)
    return r[r["nivel"] == "total"].set_index("stadium_id")


def pitch_ratios(df: pd.DataFrame, game_stadium: pd.Series, cat_ratios: pd.DataFrame,
                 st_ratios: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ratios (drag, Magnus) de ORIGEN de cada pitch usando su estadio anónimo cuando está resuelto.

    ρ_s,t = ρ_s · (ρ_c,t / ρ_c): el ratio del estadio (todas las temporadas y tipos) por el ajuste de tipo de pitch de
    su categoría. Si el juego no tiene estadio resuelto o el estadio no tiene ratio, se usa el de la categoría × tipo.
    `game_stadium`: juego → stadium_id (solo grupos resueltos), alineado con `df` por índice vía df["game_anon_id"].
    Devuelve (drag, Magnus, máscara de pitches con ratio de estadio).
    """
    cd, cm = tr.lookup_ratios(cat_ratios, df["altitude_category"], df["AutoPitchType"])
    cd_all, cm_all = tr.lookup_ratios(cat_ratios, df["altitude_category"], pd.Series("__all__", index=df.index))
    sid = df["game_anon_id"].map(game_stadium)
    sd = sid.map(st_ratios["drag_ratio"]).to_numpy(float)
    sm = sid.map(st_ratios["magnus_ratio"]).to_numpy(float)
    has = np.isfinite(sd) & np.isfinite(sm)
    drag = np.where(has, sd * cd / cd_all, cd)
    mag = np.where(has, sm * cm / cm_all, cm)
    return drag, mag, has


# ---------------------------------------------------------------------------
# Construcción y validación
# ---------------------------------------------------------------------------


def _group_status(s: pd.Series) -> pd.Series:
    vc = s.dropna().value_counts(normalize=True)
    ok = len(vc) > 0 and vc.iloc[0] >= PURITY_MIN and s.size >= MIN_HOME_GAMES
    return pd.Series({"stadium_category": vc.index[0] if len(vc) else np.nan, "pureza": vc.iloc[0] if len(vc) else np.nan,
                      "juegos_locales": s.size, "status": "resuelto" if ok else "mixto"})


def build(raw: pd.DataFrame):
    gcat = raw.groupby("game_anon_id")["altitude_category"].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    choice, seasons, stability = [], {}, []
    for y in sorted(raw["year"].unique()):
        d = raw[raw["year"] == y]
        cands = {k: reconstruct_season(d, k) for k in K_CANDIDATES}
        k_best = min(K_CANDIDATES, key=lambda k: (round(cands[k][1], 3), k))
        games, _, team = cands[k_best]
        seasons[int(y)] = (games, team)
        for k, (g, s, _) in cands.items():
            g = g.assign(cat=g["game_anon_id"].map(gcat))
            tab = pd.crosstab(g["home_team"], g["cat"])
            choice.append({"year": int(y), "k": k, "elegido": k == k_best, "local_igual_visitante": s,
                           "pureza": float(tab.max(axis=1).sum() / tab.values.sum())})
        base = games.set_index("game_anon_id")["home_team"]
        all_games = pd.Series(d["game_anon_id"].unique())
        for rep in range(1, N_STABILITY_REPS + 1):
            alt, _, _ = reconstruct_season(d, k_best, seed=rep)
            a_seed = ari(base, alt.set_index("game_anon_id")["home_team"].reindex(base.index))
            keep_games = all_games.sample(frac=0.8, random_state=rep)
            sub, _, _ = reconstruct_season(d[d["game_anon_id"].isin(keep_games)], k_best, seed=SEED)
            sub = sub.set_index("game_anon_id")["home_team"]
            stability.append({"year": int(y), "rep": rep, "ari_semilla": a_seed, "ari_submuestra_80": ari(base.reindex(sub.index), sub)})
    games, links = link_seasons(seasons)
    games["altitude_category"] = games["game_anon_id"].map(gcat)
    gs = games.groupby(["stadium_id", "year"])["altitude_category"].apply(_group_status).unstack().reset_index()
    games = games.merge(gs[["stadium_id", "year", "stadium_category", "status"]], on=["stadium_id", "year"], how="left")
    return games, gs, links, pd.DataFrame(choice), pd.DataFrame(stability)


def run() -> dict:
    raw = pd.read_parquet(RAW_PATH)
    for c in tr.TRAJECTORY_COLUMNS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    games, gs, links, choice, st = build(raw)
    games.to_parquet(OUT_PATH, index=False)

    phys = physics_frame(raw.dropna(subset=["pitcher_anon_id"] + tr.TRAJECTORY_COLUMNS), games)
    ratios = stadium_ratios(phys)
    ratios_y = stadium_ratios(phys, by_season=True)
    pd.concat([ratios.assign(year=np.nan, nivel="total"), ratios_y.assign(nivel="temporada")], ignore_index=True).to_parquet(RATIOS_PATH, index=False)
    het_d, het_m = heterogeneity(ratios, "drag_ratio"), heterogeneity(ratios, "magnus_ratio")

    res = gs[gs["status"] == "resuelto"]
    multi = res.groupby("stadium_id").filter(lambda g: g["year"].nunique() >= 2)
    consistent = multi.groupby("stadium_id")["stadium_category"].nunique().eq(1)
    pairs = []
    for y0, y1 in [(2024, 2025), (2025, 2026)]:
        a = ratios_y[ratios_y["year"] == y0].set_index("stadium_id")
        b = ratios_y[ratios_y["year"] == y1].set_index("stadium_id")
        for s in a.index.intersection(b.index):
            pairs.append({"stadium_id": s, "par": f"{y0}->{y1}", "categoria": a.loc[s, "categoria"],
                          "drag_t": a.loc[s, "drag_ratio"], "drag_t1": b.loc[s, "drag_ratio"],
                          "magnus_t": a.loc[s, "magnus_ratio"], "magnus_t1": b.loc[s, "magnus_ratio"]})
    pairs = pd.DataFrame(pairs)
    ext_pairs = pairs[pairs["categoria"] == "Extreme Altitude"]
    # dentro de Extreme, ¿el orden de los estadios se repite? (correlación de rangos entre temporadas)
    lk = links

    L = [
        "# Grupos de estadio anónimos (ANON_STADIUM_xx)",
        "",
        "Generado por `src/data/parks.py`; método y supuestos en el docstring del módulo. **Se respeta la anonimización del "
        "dataset:** no se usa información externa y no se intenta identificar equipos, estadios ni pitchers reales. Un grupo = "
        "los juegos que un mismo equipo anónimo jugó como local. La categoría de altitud solo se usa para validar.",
        "",
        "## 1. Reconstrucción por temporada",
        "",
        "El número de grupos se elige con un criterio interno (mínimo de juegos con local = visitante; empate → menos grupos).",
        "",
        "| temporada | k | elegido | juegos con local = visitante | pureza de categoría por grupo |",
        "|---|---:|:---:|---:|---:|",
    ]
    for c in choice.itertuples():
        L.append(f"| {c.year} | {c.k} | {'✔' if c.elegido else ''} | {c.local_igual_visitante:.1%} | {c.pureza:.1%} |")
    L += [
        "",
        "*Pureza* = fracción de los juegos locales de cada grupo que cae en una sola categoría de altitud (no se usa para elegir).",
        "",
        "## 2. Estabilidad del clustering",
        "",
        "Índice de Rand ajustado (ARI; 1 = misma partición, 0 = azar) de la asignación juego → grupo local contra la reconstrucción oficial.",
        "",
        "| temporada | ARI con otras semillas (mín / media) | ARI con 80% de los juegos (mín / media) |",
        "|---|---|---|",
    ]
    for y, g in st.groupby("year"):
        L.append(f"| {y} | {g['ari_semilla'].min():.3f} / {g['ari_semilla'].mean():.3f} | {g['ari_submuestra_80'].min():.3f} / {g['ari_submuestra_80'].mean():.3f} |")
    later = lk[lk["year"] > lk["year"].min()]
    L += [
        "",
        "## 3. IDs estables entre temporadas",
        "",
        f"Enlace por similitud de roster (Jaccard ≥ {JACCARD_MIN} y ≥ {JACCARD_MARGIN}× la segunda mejor). En las temporadas posteriores "
        f"a la primera: Jaccard mediano del enlace {later['jaccard'].median():.2f} (segundo mejor: {later['jaccard_segundo'].median():.2f}); "
        f"clusters sin enlace claro (reciben ID nuevo): {int(later['jaccard'].isna().sum())} de {len(later)}.",
        "",
        "| ID | " + " | ".join(str(y) for y in sorted(gs["year"].unique())) + " |",
        "|---|" + "---|" * gs["year"].nunique(),
    ]
    txt = gs.assign(txt=gs["stadium_category"].astype(str).str.replace(" Altitude", "").str.replace("nan", "—")
                    + " · " + gs["juegos_locales"].astype(int).astype(str) + np.where(gs["status"] == "mixto", " (mixto)", ""))
    tiny = gs.groupby("stadium_id").apply(lambda g: g["stadium_category"].isna().all() and g["juegos_locales"].max() < MIN_HOME_GAMES)
    piv = txt[~txt["stadium_id"].map(tiny)].pivot(index="stadium_id", columns="year", values="txt").fillna("")
    for sid, r in piv.sort_index().iterrows():
        L.append(f"| {sid} | " + " | ".join(r.tolist()) + " |")
    L += ["", f"Se omiten {int(tiny.sum())} grupos diminutos (menos de {MIN_HOME_GAMES} juegos locales y sin categoría de altitud: "
          "juegos sin `altitude_category` en el dataset)."]
    L += [
        "",
        "Celdas: categoría · juegos locales. *(mixto)* = el grupo mezcla categorías (probablemente dos equipos fusionados); "
        "sus juegos se usan solo a nivel de categoría.",
        "",
        f"**Consistencia de categoría entre temporadas:** {int(consistent.sum())} de {len(consistent)} grupos resueltos que aparecen en "
        "≥ 2 temporadas conservan la misma categoría de altitud. La categoría no se usó para enlazar, así que esto valida el enlace.",
        "",
        "## 4. Diferencias físicas entre grupos",
        "",
        "Ratio de drag/v² y Magnus/v² del mismo pitcher, tipo y temporada en el grupo contra los OTROS estadios de No Altitude "
        "(todas las temporadas juntas; IC 95% por bootstrap de celdas pitcher × temporada × tipo). 1 = igual que el promedio de "
        "No Altitude; menos de 1 = menos fuerza del aire.",
        "",
        "| ID | categoría | celdas | ratio drag [IC 95%] | ratio Magnus [IC 95%] |",
        "|---|---|---:|---|---|",
    ]
    for r in ratios.sort_values(["categoria", "drag_ratio"]).itertuples():
        L.append(f"| {r.stadium_id} | {r.categoria} | {r.celdas} | {r.drag_ratio:.3f} [{r.drag_lo:.3f}, {r.drag_hi:.3f}] | "
                 f"{r.magnus_ratio:.3f} [{r.magnus_lo:.3f}, {r.magnus_hi:.3f}] |")
    L += ["", "**Heterogeneidad dentro de cada categoría.** sd real = √max(Var observada − error² medio, 0); I² = fracción de la "
          "variación entre grupos que no es ruido de muestreo.", "",
          "| componente | categoría | grupos | rango | sd observada | error de muestreo | sd real entre estadios | I² |",
          "|---|---|---:|---|---:|---:|---:|---:|"]
    for name, het in (("drag", het_d), ("Magnus", het_m)):
        for r in het.itertuples():
            L.append(f"| {name} | {r.categoria} | {r.grupos} | {r.min:.3f}–{r.max:.3f} | {r.sd_observada:.3f} | {r.error_muestreo:.3f} | "
                     f"{r.sd_real:.3f} | {r.I2:.0%} |")
    L += [
        "",
        "**Test–retest físico** (¿el ratio de un grupo en una temporada predice el de la siguiente?):",
        "",
        f"- Todos los grupos: corr = {pairs['drag_t'].corr(pairs['drag_t1']):.2f} (drag), {pairs['magnus_t'].corr(pairs['magnus_t1']):.2f} "
        f"(Magnus); {len(pairs)} pares grupo-temporada.",
    ]
    if len(ext_pairs) >= 3:
        L.append(f"- Solo Extreme: corr = {ext_pairs['drag_t'].corr(ext_pairs['drag_t1']):.2f} (drag), "
                 f"{ext_pairs['magnus_t'].corr(ext_pairs['magnus_t1']):.2f} (Magnus); {len(ext_pairs)} pares.")
    L += [
        "",
        "## 5. Qué se puede y qué no se puede afirmar",
        "",
        "- **Se puede:** cada categoría de altitud agrupa varios estadios anónimos; dentro de una misma categoría los estadios pueden "
        "diferir en densidad efectiva más allá del ruido de muestreo (I²). \"Extreme Altitude\" no es un ambiente físico homogéneo.",
        "- **No se puede (por diseño):** asignar nombres reales a los grupos. Los resultados se expresan por categoría de altitud y "
        "por grupo anónimo.",
        "- **Identificabilidad:** cada grupo pertenece a una sola categoría. En un modelo con efectos de grupo, el efecto de la "
        "categoría queda absorbido por los grupos: solo se identifica como el promedio de los efectos de sus pocos grupos (4–6), "
        "no por separado.",
        "- **Supuesto:** un equipo juega como local siempre en el mismo estadio. Un cambio de sede aparecería como un cambio de "
        "categoría entre temporadas (sección 3).",
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    logger.info("\n".join(L))
    summary = {
        "juegos": int(len(games)), "pct_juegos_resueltos": float((games["status"] == "resuelto").mean()),
        "ari_semilla_min": float(st["ari_semilla"].min()), "ari_submuestra_80_min": float(st["ari_submuestra_80"].min()),
        "grupos_multitemporada": int(len(consistent)), "grupos_misma_categoria": int(consistent.sum()),
        "test_retest_drag": float(pairs["drag_t"].corr(pairs["drag_t1"])),
        "test_retest_magnus": float(pairs["magnus_t"].corr(pairs["magnus_t1"])),
        "test_retest_drag_extreme": float(ext_pairs["drag_t"].corr(ext_pairs["drag_t1"])) if len(ext_pairs) >= 3 else None,
        "test_retest_magnus_extreme": float(ext_pairs["magnus_t"].corr(ext_pairs["magnus_t1"])) if len(ext_pairs) >= 3 else None,
        "heterogeneidad": {name: het.to_dict("records") for name, het in (("drag", het_d), ("magnus", het_m))},
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return {"games": games, "ratios": ratios}


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    run()


if __name__ == "__main__":
    main()
