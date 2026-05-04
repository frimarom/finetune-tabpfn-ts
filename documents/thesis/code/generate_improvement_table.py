#!/usr/bin/env python3
"""
Generates a paper-ready LaTeX table from multiple CSV files.

Usage:
    python generate_latex_table.py file1.csv file2.csv ... [options]

Options:
    --names NAME1 NAME2 ...   Column names for each CSV (default: filename without extension)
    --output FILE             Output .tex file (default: table.tex)
    --caption TEXT            Table caption (default: "Results")
    --label TEXT              LaTeX label (default: "tab:results")
    --metric COLUMN           Which metric column to use (default: mean_delta_pct)

Example:
    python generate_latex_table.py model_a.csv model_b.csv \\
        --names "Model A" "Model B" \\
        --caption "Mean delta percentage per dataset" \\
        --output results_table.tex
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def normalize_dataset_name(name: str) -> str:
    """
    Normalize a dataset name for robust joining across CSV files:
      - Strip leading/trailing whitespace
      - Remove ALL occurrences of '/short' anywhere in the name (case-insensitive)
        e.g. 'ett1/15T/short' -> 'ett1/15T'
             'LOOP_SEATTLE/D/short' -> 'LOOP_SEATTLE/D'
      - Lowercase for join key  (e.g. 'LOOP_SEATTLE/D' == 'loop_seattle/d')
      - Original casing (minus /short) preserved for display
    Returns (display_name, join_key).
    """
    import re
    name = name.strip()
    name = re.sub(r"/short", "", name, flags=re.IGNORECASE).strip()
    display = name        # keep original casing for display
    key = name.lower()
    return display, key


def clean_column_name(name: str) -> str:
    """Clean a column name: remove '/short' (case-insensitive) and apply title case."""
    import re
    name = re.sub(r"/short", "", name, flags=re.IGNORECASE).strip()
    return name.title()


def load_csv(filepath: str, metric: str) -> tuple[dict[str, float], dict[str, str]]:
    """
    Load a CSV and return:
      - data:    {join_key -> metric_value}
      - display: {join_key -> display_name}
    """
    data = {}
    display = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["dataset"]
            disp, key = normalize_dataset_name(raw)
            try:
                value = float(row[metric])
            except (KeyError, ValueError):
                print(f"  Warning: Could not read '{metric}' for dataset '{raw}' in {filepath}", file=sys.stderr)
                value = float("nan")
            data[key] = value
            display[key] = disp
    return data, display


def render_png(
    dataset_keys: list[str],
    all_data: list[dict[str, float]],
    all_display: dict[str, str],
    column_names: list[str],
    caption: str,
    decimals: int,
    png_path: str,
) -> None:
    """Render the table as a PNG using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import math

    # Build cell data
    rows = []
    bold_mask = []  # parallel bool grid
    for key in dataset_keys:
        values = [d.get(key, float("nan")) for d in all_data]
        valid = [(i, v) for i, v in enumerate(values) if not math.isnan(v)]
        best_idx = min(valid, key=lambda x: x[1])[0] if valid else -1
        display = all_display.get(key, key)
        row = [display]
        mask = [False]
        for i, v in enumerate(values):
            row.append(f"{v:.{decimals}f}" if not math.isnan(v) else "--")
            mask.append(i == best_idx)
        rows.append(row)
        bold_mask.append(mask)

    col_labels = ["Dataset"] + column_names
    n_rows = len(rows)
    n_cols = len(col_labels)

    # Figure sizing
    row_h = 0.38
    fig_h = max(2.0, row_h * (n_rows + 2) + 0.6)
    col_w = 1.6
    fig_w = col_w * n_cols + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Style header
    header_color = "#2c3e50"
    alt_color    = "#f2f4f5"
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # Style data rows + bold best values
    for row_idx, (row_data, mask) in enumerate(zip(rows, bold_mask)):
        bg = alt_color if row_idx % 2 == 0 else "white"
        for col_idx in range(n_cols):
            cell = tbl[row_idx + 1, col_idx]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#cccccc")
            if mask[col_idx]:
                cell.set_text_props(fontweight="bold", color="#1a252f")
            # Left-align dataset name column
            if col_idx == 0:
                cell.set_text_props(ha="left")
                cell._loc = "left"

    if caption:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color="#555555", style="italic")

    plt.tight_layout(pad=0.3)
    plt.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"PNG written to:   {png_path}")


def format_value(value: float, is_best: bool, decimals: int = 2) -> str:
    """Format a float value, bold if best."""
    if value != value:  # NaN check
        return "--"
    formatted = f"{value:.{decimals}f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_latex_table(
    csv_files: list[str],
    column_names: list[str],
    metric: str,
    caption: str,
    label: str,
    decimals: int,
) -> tuple:
    """Generate LaTeX string + raw data for PNG rendering."""

    # Load all data; keyed by normalized join_key
    all_data: list[dict[str, float]] = []
    all_display: dict[str, str] = {}  # join_key -> display name
    for filepath in csv_files:
        print(f"Loading {filepath} ...")
        data, display = load_csv(filepath, metric)
        all_data.append(data)
        all_display.update(display)

    # Collect all unique join keys
    keys_seen: set[str] = set()
    dataset_keys: list[str] = []
    for d in all_data:
        for k in d.keys():
            if k not in keys_seen:
                dataset_keys.append(k)
                keys_seen.add(k)
    dataset_keys.sort()  # alphabetical — remove to keep original order

    num_models = len(csv_files)

    # Build LaTeX
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("    \\centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")

    # Column spec: left-aligned dataset name + right-aligned numeric columns
    col_spec = "l" + "r" * num_models
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("        \\toprule")

    # Header row
    header_cols = ["Dataset"] + column_names
    lines.append("        " + " & ".join(header_cols) + " \\\\")
    lines.append("        \\midrule")

    # Data rows
    for key in dataset_keys:
        values = [d.get(key, float("nan")) for d in all_data]

        # Find best (minimum) among valid values
        valid_values = [(i, v) for i, v in enumerate(values) if v == v]
        best_idx = min(valid_values, key=lambda x: x[1])[0] if valid_values else -1

        display_name = all_display.get(key, key).replace("_", "\\_")
        cells = [display_name]
        for i, v in enumerate(values):
            cells.append(format_value(v, is_best=(i == best_idx), decimals=decimals))

        lines.append("        " + " & ".join(cells) + " \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines), dataset_keys, all_data, all_display


def main():
    parser = argparse.ArgumentParser(
        description="Generate a paper-ready LaTeX table from multiple CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to process")
    parser.add_argument(
        "--names",
        nargs="+",
        metavar="NAME",
        help="Column names for each CSV (default: filename stem)",
    )
    parser.add_argument("--output", default="table.tex", help="Output .tex file (default: table.tex)")
    parser.add_argument("--png", default=None, metavar="FILE", help="Also render a PNG (e.g. table.png)")
    parser.add_argument("--caption", default="Results", help="Table caption")
    parser.add_argument("--label", default="tab:results", help="LaTeX label")
    parser.add_argument(
        "--metric",
        default="mean_delta_pct",
        help="Metric column to use (default: mean_delta_pct)",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for values (default: 2)",
    )

    args = parser.parse_args()

    # Validate files
    for f in args.csv_files:
        if not os.path.isfile(f):
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    # Column names default to filename stems
    if args.names:
        if len(args.names) != len(args.csv_files):
            print(
                f"Error: --names count ({len(args.names)}) must match number of CSV files ({len(args.csv_files)})",
                file=sys.stderr,
            )
            sys.exit(1)
        column_names = [clean_column_name(n) for n in args.names]
    else:
        column_names = [clean_column_name(Path(f).stem) for f in args.csv_files]

    latex, dataset_keys, all_data, all_display = generate_latex_table(
        csv_files=args.csv_files,
        column_names=column_names,
        metric=args.metric,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(latex)
        f.write("\n")

    print(f"\nTable written to: {args.output}")

    if args.png:
        render_png(
            dataset_keys=dataset_keys,
            all_data=all_data,
            all_display=all_display,
            column_names=column_names,
            caption=args.caption,
            decimals=args.decimals,
            png_path=args.png,
        )

    print("\nPreview:\n")
    print(latex)
    print(
        "\nNote: Add '\\usepackage{booktabs}' to your LaTeX preamble for \\toprule/\\midrule/\\bottomrule."
    )


if __name__ == "__main__":
    main()