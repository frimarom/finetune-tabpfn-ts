"""
analyze_finetuning_summary.py
------------------------------
Liest alle single_finetune/<dataset>/finetuning_summary.csv Dateien,
sortiert pro Dataset nach Best Validation Loss (aufsteigend),
nimmt die Top-5 Best Validation Losses sowie den Initial Validation Loss
und schreibt sie als je eine Zeile in eine Output-CSV.

Ausfuehrung (vom Projektroot):
    python analyze_finetuning_summary.py [--root single_finetune] [--out top5_results.csv]
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd


# =============================================================================
# 1. CSVs finden und laden
# =============================================================================

def find_summary_csvs(root: Path) -> list[Path]:
    root = root.resolve()
    found = sorted(root.rglob("finetuning_summary.csv"))
    print("Durchsuche: " + str(root))
    if not found:
        print("  -> Keine finetuning_summary.csv gefunden.")
    else:
        for p in found:
            print("  OK " + str(p.relative_to(root)))
    return found


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# =============================================================================
# 2. Pro Dataset: Top-5 nach Best Validation Loss
# =============================================================================

TERM_NAMES = {"short", "medium", "long"}

def dataset_name_from_path(csv_path: Path, root: Path) -> str:
    parts = csv_path.relative_to(root).parts[:-1]
    meaningful = [p for p in parts if p.lower() not in TERM_NAMES]
    return "/".join(meaningful)


def build_top5_records(root: Path) -> pd.DataFrame:
    csvs = find_summary_csvs(root)
    if not csvs:
        raise FileNotFoundError("Keine finetuning_summary.csv unter " + str(root) + " gefunden.")

    print("\nGefundene finetuning_summary.csv Dateien: " + str(len(csvs)) + "\n")

    all_rows = []

    for csv_path in csvs:
        try:
            df = load_csv(csv_path)
        except Exception as e:
            warnings.warn("Fehler beim Lesen von " + str(csv_path) + ": " + str(e))
            continue

        dataset_name = dataset_name_from_path(csv_path, root)
        print("Verarbeite: " + dataset_name)

        for col in ("Best Validation Loss", "Initial Validation Loss"):
            if col not in df.columns:
                warnings.warn("Spalte '{}' fehlt in {} – uebersprungen.".format(col, csv_path))
                continue

        # Initial Validation Loss: Minimum ueber alle Eintraege
        initial_val_loss = df["Initial Validation Loss"].min()

        # Top-5 aufsteigend nach Best Validation Loss
        top5 = (
            df.sort_values("Best Validation Loss", ascending=True)
            .head(5)["Best Validation Loss"]
            .reset_index(drop=True)
        )

        row: dict = {"dataset": dataset_name, "initial_validation_loss": initial_val_loss}
        for rank, val in enumerate(top5, start=1):
            row["top{}_best_validation_loss".format(rank)] = val

        all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows).sort_values("dataset").reset_index(drop=True)


# =============================================================================
# 3. Ausgabe
# =============================================================================

def export_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print("\nCSV gespeichert: " + str(path))
    print("Datasets verarbeitet: " + str(len(df)))
    print("\nVorschau:")
    print(df.to_string(index=False))


# =============================================================================
# 4. Entrypoint
# =============================================================================

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Finetuning Summary Top-5 Analyse")
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_DIR / "single_finetune",
        help="Wurzelordner mit den finetuning_summary.csv Dateien (default: <script-dir>/single_finetune)",
    )
    parser.add_argument(
        "--out", type=Path, default=SCRIPT_DIR / "top5_results.csv",
        help="Pfad fuer die Output-CSV (default: <script-dir>/top5_results.csv)",
    )
    args = parser.parse_args()

    df = build_top5_records(args.root)

    if df.empty:
        print("Keine Daten geladen. Bitte Pfad und CSV-Struktur pruefen.")
        return

    export_csv(df, args.out)


if __name__ == "__main__":
    main()