"""Features de arsenal por pitcher para el modelo Stuff+ LMB.

Toma data/processed/pitches_with_targets.parquet (salida de src/features/targets.py)
y agrega, agrupando por pitcher_anon_id y year (para no mezclar temporadas), seis
columnas que describen cada pitch en relacion al resto del repertorio de ESE
pitcher en ESA temporada:

1. velo_diff_vs_fb   : RelSpeed - promedio de RelSpeed del fastball del pitcher/year
2. ivb_diff_vs_fb    : InducedVertBreak - promedio de IVB del fastball del pitcher/year
3. hb_diff_vs_fb     : HorzBreak - promedio de HB del fastball del pitcher/year
4. tunnel_similarity : similitud (0,1] entre (RelHeight, RelSide, VertApprAngle,
                        HorzApprAngle) de este pitch y los promedios del fastball
                        del pitcher/year (ver `add_tunnel_similarity`)
5. pitch_usage_pct   : frecuencia relativa del AutoPitchType de este pitch dentro
                        del repertorio del pitcher esa temporada
6. prev_pitch_type   : AutoPitchType del pitch anterior en la misma plate
                        appearance (orden cronologico real via PitchUID, ver
                        src/features/targets.py)

"Fastball" = AutoPitchType en {Four-Seam, Sinker}. Si un pitcher no tiro ningun
fastball en una temporada dada, las columnas 1-4 quedan nulas para TODOS sus
pitches de esa temporada (no hay baseline contra que comparar) — ver
`count_missing_fastball` y la seccion correspondiente del reporte.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.targets import _add_uuid_time, _assign_plate_appearances

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "pitches_with_targets.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pitches_with_arsenal.parquet"
DEFAULT_REPORT = ROOT / "reports" / "arsenal_summary.md"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FASTBALL_TYPES = {"Four-Seam", "Sinker"}
TUNNEL_DIMS = ["RelHeight", "RelSide", "VertApprAngle", "HorzApprAngle"]


# ---------------------------------------------------------------------------
# 1-3. Diferencias vs el fastball del propio pitcher esa temporada
# ---------------------------------------------------------------------------


def _fastball_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Promedios del fastball (Four-Seam/Sinker) por (pitcher_anon_id, year).

    Un pitcher-year sin fastball simplemente no aparece en esta tabla; el merge
    posterior deja NaN para todas sus filas en las columnas derivadas de esto.
    """
    fb = df.loc[df["AutoPitchType"].isin(FASTBALL_TYPES)]
    agg_cols = ["RelSpeed", "InducedVertBreak", "HorzBreak", *TUNNEL_DIMS]
    baseline = fb.groupby(["pitcher_anon_id", "year"])[agg_cols].mean()
    baseline["fb_n"] = fb.groupby(["pitcher_anon_id", "year"]).size()
    baseline = baseline.add_prefix("fb_")
    baseline = baseline.rename(columns={"fb_fb_n": "fb_n"})
    return baseline.reset_index()


def add_fastball_diffs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    baseline = _fastball_baseline_table(df)
    df = df.merge(baseline, on=["pitcher_anon_id", "year"], how="left")

    df["velo_diff_vs_fb"] = df["RelSpeed"] - df["fb_RelSpeed"]
    df["ivb_diff_vs_fb"] = df["InducedVertBreak"] - df["fb_InducedVertBreak"]
    df["hb_diff_vs_fb"] = df["HorzBreak"] - df["fb_HorzBreak"]
    return df


# ---------------------------------------------------------------------------
# 4. tunnel_similarity
# ---------------------------------------------------------------------------


def add_tunnel_similarity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Similitud de tunneling entre este pitch y el fastball del pitcher/year.

    Cada dimension (RelHeight, RelSide en pies; VertApprAngle, HorzApprAngle en
    grados) se normaliza dividiendo entre su desviacion estandar GLOBAL (todo el
    dataset), para que pies y grados sean comparables antes de combinarlos en una
    sola distancia euclidiana (equivalente a una distancia euclidiana sobre
    z-scores). tunnel_similarity = 1 / (1 + distancia): 1.0 = identico al
    fastball del pitcher esa temporada, decreciendo hacia 0 mientras mas distinto
    es el release point / angulo de aproximacion.
    """
    df = df.copy()
    global_std = {dim: float(df[dim].std()) for dim in TUNNEL_DIMS}

    squared_terms = []
    for dim in TUNNEL_DIMS:
        z_diff = (df[dim] - df[f"fb_{dim}"]) / global_std[dim]
        squared_terms.append(z_diff**2)

    distance = np.sqrt(sum(squared_terms))
    df["tunnel_similarity"] = 1 / (1 + distance)

    return df, {"global_std": global_std}


# ---------------------------------------------------------------------------
# 5. pitch_usage_pct
# ---------------------------------------------------------------------------


def add_pitch_usage_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    usage = (
        df.groupby(["pitcher_anon_id", "year", "AutoPitchType"]).size()
        / df.groupby(["pitcher_anon_id", "year"]).size()
    )
    usage.name = "pitch_usage_pct"
    df = df.merge(
        usage.reset_index(), on=["pitcher_anon_id", "year", "AutoPitchType"], how="left"
    )
    return df


# ---------------------------------------------------------------------------
# 6. prev_pitch_type
# ---------------------------------------------------------------------------


def add_prev_pitch_type(df: pd.DataFrame) -> pd.DataFrame:
    """AutoPitchType del pitch anterior en la misma plate appearance.

    Reutiliza el orden cronologico real (timestamp embebido en PitchUID, ver
    src/features/targets.py) y la asignacion de plate appearances por
    media-entrada, para no depender del orden de fila del archivo.
    """
    original_index = df.index
    work = _add_uuid_time(df)
    work = _assign_plate_appearances(work)
    work["prev_pitch_type"] = work.groupby(["_half_inning_id", "_pa_num"])[
        "AutoPitchType"
    ].shift(1)

    df = df.copy()
    df["prev_pitch_type"] = work["prev_pitch_type"].reindex(original_index)
    return df


# ---------------------------------------------------------------------------
# Fastballs faltantes por pitcher-year
# ---------------------------------------------------------------------------


def count_missing_fastball(df: pd.DataFrame) -> dict:
    groups = df.groupby(["pitcher_anon_id", "year"])
    has_fastball = groups.apply(
        lambda g: g["AutoPitchType"].isin(FASTBALL_TYPES).any(), include_groups=False
    )
    missing = has_fastball[~has_fastball]
    n_pitches_affected = int(
        df.set_index(["pitcher_anon_id", "year"]).index.isin(missing.index).sum()
    )
    sample_sizes = groups.size().loc[missing.index].sort_values(ascending=False)

    # df.groupby(["pitcher_anon_id","year"]) descarta por default las filas con
    # pitcher_anon_id nulo (dato crudo preexistente, no introducido por este modulo)
    # asi que se cuentan aparte: para esas filas no hay NINGUN pitcher a quien
    # buscarle fastball, es una causa de nulo distinta a "pitcher conocido sin fastball"
    n_null_pitcher_id = int(df["pitcher_anon_id"].isna().sum())

    return {
        "n_pitcher_year_groups": int(groups.ngroups),
        "n_pitcher_year_without_fastball": int(len(missing)),
        "n_pitches_affected": n_pitches_affected,
        "affected_group_sample_sizes": sample_sizes.to_dict(),
        "n_pitches_null_pitcher_id": n_null_pitcher_id,
    }


# ---------------------------------------------------------------------------
# Validacion manual con un pitcher de ejemplo
# ---------------------------------------------------------------------------


def validate_example_pitcher(df: pd.DataFrame, pitcher_id: str, year: int) -> pd.DataFrame:
    cols = [
        "PitchUID",
        "AutoPitchType",
        "RelSpeed",
        "fb_RelSpeed",
        "velo_diff_vs_fb",
        "InducedVertBreak",
        "fb_InducedVertBreak",
        "ivb_diff_vs_fb",
        "HorzBreak",
        "fb_HorzBreak",
        "hb_diff_vs_fb",
        "tunnel_similarity",
        "pitch_usage_pct",
        "prev_pitch_type",
    ]
    sub = df.loc[(df["pitcher_anon_id"] == pitcher_id) & (df["year"] == year), cols]
    return sub


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_arsenal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    report["missing_fastball"] = count_missing_fastball(df)

    df = add_fastball_diffs(df)
    df, tunnel_report = add_tunnel_similarity(df)
    report["tunnel_similarity"] = tunnel_report
    df = add_pitch_usage_pct(df)
    df = add_prev_pitch_type(df)

    new_cols = [
        "velo_diff_vs_fb",
        "ivb_diff_vs_fb",
        "hb_diff_vs_fb",
        "tunnel_similarity",
        "pitch_usage_pct",
        "prev_pitch_type",
    ]
    report["null_counts"] = {c: int(df[c].isna().sum()) for c in new_cols}

    df = df.drop(columns=[c for c in df.columns if c.startswith("fb_")])
    return df, report


def format_report(report: dict, example: pd.DataFrame, example_pitcher: str, example_year: int) -> str:
    mf = report["missing_fastball"]
    ts = report["tunnel_similarity"]
    nulls = report["null_counts"]

    lines = [
        "# Reporte de arsenal — pitches_with_arsenal.parquet",
        "",
        "## Pitchers-temporada sin fastball identificable",
        "",
        f"- Total de combinaciones pitcher-year: {mf['n_pitcher_year_groups']:,}",
        f"- Sin Four-Seam ni Sinker en toda la temporada: **{mf['n_pitcher_year_without_fastball']}** "
        f"({100*mf['n_pitcher_year_without_fastball']/mf['n_pitcher_year_groups']:.1f}%)",
        f"- Pitches de esos {mf['n_pitcher_year_without_fastball']} pitcher-year (quedan con "
        f"velo_diff_vs_fb/ivb_diff_vs_fb/hb_diff_vs_fb/tunnel_similarity nulos): "
        f"**{mf['n_pitches_affected']}**",
        "",
        f"Aparte, hay **{mf['n_pitches_null_pitcher_id']}** pitches con `pitcher_anon_id` nulo en "
        "el dato crudo (no introducido por este pipeline) — para esos no hay ningun pitcher a "
        "quien buscarle fastball, asi que tambien quedan con estas 4 columnas nulas. El total de "
        f"nulos en velo_diff_vs_fb/ivb_diff_vs_fb/hb_diff_vs_fb/tunnel_similarity "
        f"({mf['n_pitches_affected'] + mf['n_pitches_null_pitcher_id']}) es la suma de ambas causas.",
        "",
        "Tamaño de muestra (n pitches en la temporada) de cada pitcher-year sin fastball "
        "— todos son casos de muestra muy chica, no especialistas sin fastball real:",
        "",
    ]
    for (pid, yr), n in mf["affected_group_sample_sizes"].items():
        lines.append(f"- {pid} ({int(yr)}): {n} pitches en la temporada")

    lines += [
        "",
        "## Nulos por columna nueva (todo el dataset)",
        "",
    ]
    for c, n in nulls.items():
        lines.append(f"- `{c}`: {n:,}")

    lines += [
        "",
        "## tunnel_similarity: normalizacion usada",
        "",
        "Desviacion estandar global (todo el dataset) por dimension, usada para "
        "convertir pies y grados a una escala comparable antes de la distancia euclidiana:",
        "",
    ]
    for dim, std in ts["global_std"].items():
        lines.append(f"- `{dim}`: std={std:.4f}")

    lines += [
        "",
        f"## Validacion manual: {example_pitcher}, temporada {example_year}",
        "",
        "Repertorio de esta temporada (conteo y % de uso):",
        "",
    ]
    usage_table = (
        example.groupby("AutoPitchType")
        .agg(n=("PitchUID", "count"), pitch_usage_pct=("pitch_usage_pct", "first"))
        .sort_values("n", ascending=False)
    )
    lines.append(usage_table.round(4).to_markdown())

    lines += [
        "",
        "Promedios por tipo de pitch (para verificar que velo/ivb/hb_diff_vs_fb "
        "tengan sentido: el fastball deberia dar ~0, offspeed mas lento/con mas "
        "break deberia dar diffs negativos de velocidad y de break segun el pitch):",
        "",
    ]
    means_table = example.groupby("AutoPitchType")[
        ["RelSpeed", "velo_diff_vs_fb", "InducedVertBreak", "ivb_diff_vs_fb", "HorzBreak", "hb_diff_vs_fb", "tunnel_similarity"]
    ].mean()
    lines.append(means_table.round(2).to_markdown())

    lines += [
        "",
        "Primeros 15 pitches (orden cronologico) con prev_pitch_type, para revisar "
        "manualmente el encadenamiento dentro de la plate appearance:",
        "",
    ]
    preview = example.head(15)[["PitchUID", "AutoPitchType", "prev_pitch_type", "RelSpeed", "velo_diff_vs_fb", "tunnel_similarity"]]
    lines.append(preview.round(3).to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--example-pitcher", type=str, default="pitcher_00938")
    parser.add_argument("--example-year", type=int, default=2025)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    logger.info("Cargado %s: %s filas", args.input.name, f"{len(df):,}")

    df_out, report = build_arsenal_features(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    logger.info("Guardado %s (%s filas, %s columnas)", args.output, f"{len(df_out):,}", df_out.shape[1])

    example = validate_example_pitcher(
        df_out.merge(
            _fastball_baseline_table(df), on=["pitcher_anon_id", "year"], how="left"
        ),
        args.example_pitcher,
        args.example_year,
    )
    report_text = format_report(report, example, args.example_pitcher, args.example_year)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    logger.info("Guardado reporte en %s", args.report)


if __name__ == "__main__":
    main()
