#!/usr/bin/env python3
"""
evaluate_csv.py
---------------
Liest alle *_evaluation.csv Dateien im aktuellen Ordner und schreibt
eine aggregierte Summary-CSV: mase_checkpoint_summary.csv

Die Spalte `trained_on` ist standardmäßig False und kann manuell
in der CSV nachträglich auf True gesetzt werden.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

MASE_COL_CANDIDATES = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MASE",
    "MASE",
]

DATASET_COL = "dataset"
FINETUNE_COL = "finetuning_place"
NUM_SERIES_COL_CANDIDATES = ["num_variates", "num_series"]
CKPTS = [1, 2, 3, 4, 5]

OUTPUT_CSV = "mase_checkpoint_summary.csv"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip('"')
    return df


def infer_context_length(dataset_name: str) -> str:
    if not isinstance(dataset_name, str):
        return ""
    parts = dataset_name.split("/")
    return parts[-1] if parts else ""


# ---------------------------------------------------------------------------
# CSV laden & aggregieren
# ---------------------------------------------------------------------------

def load_single_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df = clean_string_columns(df)

    mase_col = find_first_existing_column(df, MASE_COL_CANDIDATES)
    if mase_col is None:
        raise ValueError(f"Keine MASE-Spalte in {path.name} gefunden.")

    num_series_col = find_first_existing_column(df, NUM_SERIES_COL_CANDIDATES)
    if num_series_col is None:
        df["num_variates"] = np.nan
        num_series_col = "num_variates"

    required = [DATASET_COL, FINETUNE_COL, mase_col, num_series_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {path.name}: {missing}")

    df = df[[DATASET_COL, FINETUNE_COL, mase_col, num_series_col]].copy()
    df.columns = ["dataset", "finetuning_place", "mase", "num_series"]

    df["finetuning_place"] = pd.to_numeric(df["finetuning_place"], errors="coerce")
    df["mase"] = pd.to_numeric(df["mase"], errors="coerce")
    df["num_series"] = pd.to_numeric(df["num_series"], errors="coerce")

    df = df.dropna(subset=["dataset", "finetuning_place", "mase"])
    return df


def aggregate_all_csvs(folder: str = ".") -> pd.DataFrame:
    files = sorted(Path(folder).glob("*_evaluation.csv"))
    if not files:
        raise FileNotFoundError(
            "Keine Dateien im Format '*_evaluation.csv' im aktuellen Ordner gefunden."
        )

    frames = [load_single_csv(p) for p in files]
    full = pd.concat(frames, ignore_index=True)

    grouped = (
        full.groupby(["dataset", "finetuning_place"], as_index=False)
        .agg(
            mase=("mase", "mean"),
            num_series=("num_series", "mean"),
        )
    )
    return grouped


# ---------------------------------------------------------------------------
# Summary-Tabelle bauen
# ---------------------------------------------------------------------------

def build_summary_table(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in sorted(grouped["dataset"].unique()):
        sub = grouped[grouped["dataset"] == dataset].copy()

        base_row = sub[sub["finetuning_place"] == -1]
        if base_row.empty:
            continue

        base_mase = float(base_row["mase"].iloc[0])
        if not np.isfinite(base_mase) or base_mase == 0:
            continue

        num_series = base_row["num_series"].iloc[0]
        if pd.isna(num_series):
            valid = sub["num_series"].dropna()
            num_series = valid.iloc[0] if len(valid) > 0 else np.nan

        deltas: dict[str, float] = {}
        available: list[float] = []

        for ckpt in CKPTS:
            ckpt_row = sub[sub["finetuning_place"] == ckpt]
            if ckpt_row.empty:
                delta_pct = np.nan
            else:
                ckpt_mase = float(ckpt_row["mase"].iloc[0])
                delta_pct = ((ckpt_mase - base_mase) / base_mase) * 100.0
                available.append(delta_pct)
            deltas[f"ckpt_{ckpt}_delta_pct"] = delta_pct

        mean_delta = float(np.nanmean(available)) if available else np.nan
        min_delta = float(np.nanmin(available)) if available else np.nan
        max_delta = float(np.nanmax(available)) if available else np.nan
        improved = bool(mean_delta < 0) if not np.isnan(mean_delta) else False

        rows.append(
            {
                "dataset": dataset,
                "context_length": infer_context_length(dataset),
                "num_series": int(num_series) if not pd.isna(num_series) else np.nan,
                # ── NEU: manuell befüllbare Spalte ──────────────────────────
                "trained_on": False,
                # ────────────────────────────────────────────────────────────
                "ckpt_1_delta_pct": deltas["ckpt_1_delta_pct"],
                "ckpt_2_delta_pct": deltas["ckpt_2_delta_pct"],
                "ckpt_3_delta_pct": deltas["ckpt_3_delta_pct"],
                "ckpt_4_delta_pct": deltas["ckpt_4_delta_pct"],
                "ckpt_5_delta_pct": deltas["ckpt_5_delta_pct"],
                "mean_delta_pct": mean_delta,
                "min_delta_pct": min_delta,
                "max_delta_pct": max_delta,
                "improved": improved,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("mean_delta_pct", ascending=True).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def generate_csv(folder: str = ".") -> pd.DataFrame:
    """
    Aggregiert alle *_evaluation.csv Dateien und schreibt die Summary-CSV.

    Returns:
        Das Summary-DataFrame (wird zusätzlich als CSV gespeichert).
    """
    grouped = aggregate_all_csvs(folder)
    summary = build_summary_table(grouped)

    if summary.empty:
        raise RuntimeError("Keine auswertbaren Datensätze gefunden.")

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"[CSV] Geschrieben: {OUTPUT_CSV}")
    print(
        f"      {len(summary)} Datensätze · "
        f"{summary['improved'].sum()} verbessert · "
        f"Spalte 'trained_on' ist standardmäßig False – bitte manuell befüllen."
    )
    return summary


if __name__ == "__main__":
    generate_csv()
