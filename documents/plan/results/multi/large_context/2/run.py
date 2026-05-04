#!/usr/bin/env python3
"""
run.py
------
CLI-Einstiegspunkt für die MASE-Checkpoint-Auswertung.

Verwendung
----------
# 1) Nur die Summary-CSV aus den *_evaluation.csv Dateien erzeugen:
python run.py --csv

# 2) Nur die Plots aus einer bereits vorhandenen CSV erzeugen:
python run.py --plot

# 3) Nur die Übersichtstabelle (als PNG/PDF) aus einer CSV erzeugen:
python run.py --table

# 4) Alles auf einmal (CSV → Plots → Tabelle):
python run.py --all

# Optional: anderen CSV-Pfad angeben (für --plot / --table):
python run.py --plot --csv-path meine_summary.csv
python run.py --table --csv-path meine_summary.csv

# Optional: anderen Eingabeordner für --csv angeben:
python run.py --csv --folder /pfad/zu/eval/dateien
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Imports aus den beiden Modulen
# ---------------------------------------------------------------------------

from evaluate_csv import generate_csv, OUTPUT_CSV as DEFAULT_CSV
from plot_results import generate_plots


# ---------------------------------------------------------------------------
# Tabellen-Plot
# ---------------------------------------------------------------------------

DELTA_COLS = [
    "ckpt_1_delta_pct",
    "ckpt_2_delta_pct",
    "ckpt_3_delta_pct",
    "ckpt_4_delta_pct",
    "ckpt_5_delta_pct",
    "mean_delta_pct",
]

COL_LABELS = ["Ckpt 1", "Ckpt 2", "Ckpt 3", "Ckpt 4", "Ckpt 5", "Mean"]


def _fmt(val) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:+.1f}%"


def _cell_color(val) -> str:
    """Hintergrundfarbe einer Delta-Zelle – nur blau/rot, kein trained_on."""
    if pd.isna(val):
        return "#f0f0f0"
    return "#aec7e8" if val < 0 else "#f4a6a6"   # hellblau / hellrot


def generate_table(csv_path: str = DEFAULT_CSV) -> None:
    """
    Erstellt eine Tabellen-Grafik (PNG + PDF) aus der Summary-CSV.
    Zeilen mit trained_on=True werden farbig hervorgehoben.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV nicht gefunden: {csv_path}\n"
            "Bitte zuerst mit --csv die Summary-CSV generieren."
        )

    df = pd.read_csv(csv_path)

    if "trained_on" not in df.columns:
        print("[Warnung] Spalte 'trained_on' fehlt – wird als False angenommen.")
        df["trained_on"] = False
    else:
        df["trained_on"] = df["trained_on"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    df = df.sort_values("mean_delta_pct", ascending=True).reset_index(drop=True)

    n_rows = len(df)
    n_cols = 2 + len(DELTA_COLS)   # dataset | num_series | ckpt_1..5 | mean

    col_labels = ["Dataset", "# Series"] + COL_LABELS

    # Zelleninhalte
    cell_text = []
    cell_colors = []
    trained_row_indices: list[int] = []   # 1-based tbl row indices (trained_on=True)

    for i, (_, row) in enumerate(df.iterrows()):
        trained = bool(row["trained_on"])

        short_name = row["dataset"].replace("/short", "")
        if len(short_name) > 45:
            short_name = short_name[:42] + "..."
        if trained:
            short_name = f"★  {short_name}"
            trained_row_indices.append(i + 1)   # +1 because row 0 is the header

        ns = row.get("num_series", np.nan)
        ns_str = str(int(ns)) if not pd.isna(ns) else "—"

        cell_text.append([short_name, ns_str] + [_fmt(row.get(c, np.nan)) for c in DELTA_COLS])
        cell_colors.append(
            ["white", "white"] + [_cell_color(row.get(c, np.nan)) for c in DELTA_COLS]
        )

    # Grafik
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    row_height  = 0.38   # cm → figure units
    fig_height  = max(3.5, n_rows * row_height + 1.8)
    fig_width   = 14.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    col_widths = [0.30] + [0.08] + [0.10] * len(DELTA_COLS)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)

    # Kopfzeile dunkler
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Dataset-Spalte linksbündig; trained_on Zeilen → fett + dunklere Schrift
    for row_idx in range(1, n_rows + 1):
        cell = tbl[row_idx, 0]
        cell.set_text_props(ha="left")
        cell._loc = "left"
        if row_idx in trained_row_indices:
            cell.set_text_props(ha="left", fontweight="bold", color="#1a1a1a")

    # Legende
    patches = [
        mpatches.Patch(color="#aec7e8", label="Improved"),
        mpatches.Patch(color="#f4a6a6", label="Worsened"),
    ]
    from matplotlib.lines import Line2D
    star_handle = Line2D(
        [], [], marker="*", color="w", markerfacecolor="#1a1a1a",
        markersize=9, label="★  Trained on this dataset",
    )
    ax.legend(
        handles=[star_handle, *patches],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=3,
        frameon=True,
        fontsize=8,
    )

    ax.set_title(
        "MASE checkpoint delta summary",
        fontsize=11,
        fontweight="bold",
        pad=14,
        loc="left",
    )

    out = "mase_summary_table"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Tabelle] {out}.pdf / .png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MASE Checkpoint Auswertung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_argument_group("Modus (mindestens eine Option wählen)")
    mode.add_argument(
        "--csv",
        action="store_true",
        help="Summary-CSV aus *_evaluation.csv Dateien generieren.",
    )
    mode.add_argument(
        "--plot",
        action="store_true",
        help="Alle Barplots aus der Summary-CSV generieren.",
    )
    mode.add_argument(
        "--table",
        action="store_true",
        help="Nur die Übersichtstabelle (PNG/PDF) generieren.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="CSV + Plots + Tabelle generieren.",
    )

    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV,
        metavar="PATH",
        help=f"Pfad zur Summary-CSV (Standard: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--folder",
        default=".",
        metavar="DIR",
        help="Ordner mit den *_evaluation.csv Dateien (Standard: aktuelles Verzeichnis).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    do_csv   = args.csv   or args.all
    do_plot  = args.plot  or args.all
    do_table = args.table or args.all

    if not any([do_csv, do_plot, do_table]):
        print("Kein Modus angegeben. Nutze --help für Informationen.")
        sys.exit(1)

    if do_csv:
        generate_csv(folder=args.folder)

    if do_plot:
        generate_plots(csv_path=args.csv_path)

    if do_table:
        generate_table(csv_path=args.csv_path)

    print("\nFertig.")


if __name__ == "__main__":
    main()