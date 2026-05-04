"""
plot_finetuning.py
------------------
Generates three paper-quality PDF scatter plots from a finetuning results CSV.

  plot1_score_series.pdf   — context × num_series  vs. mean_delta_pct
  plot2_context_vs_series.pdf — context_length vs. num_series  (2-D view)
  plot3_score_windows.pdf  — context × total_windows vs. mean_delta_pct
                             (total_windows = num_series × dataset.windows,
                              loaded via finetune_tabpfn_ts.evaluation.data.Dataset)

Usage:
    python plot_finetuning.py results.csv
    python plot_finetuning.py results.csv --output-dir ./figures
    python plot_finetuning.py results.csv --score-threshold 50000
    python plot_finetuning.py results.csv --fig-width 6.5   # single-column / full-width

Required CSV columns:
    dataset, context_length, num_series, mean_delta_pct, improved
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.use("pdf")   # vector PDF output — no screen needed

# ---------------------------------------------------------------------------
# Paper-quality rc settings
# Uses Type-42 (TrueType-embedded) fonts so PDF passes publisher checks.
# Switch font.family to "sans-serif" and font.sans-serif to ["Helvetica"] for
# NeurIPS / ICLR style.
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "legend.borderpad":   0.4,
    "lines.linewidth":    0.8,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "axes.grid":          True,
    "grid.linewidth":     0.35,
    "grid.alpha":         0.45,
    "grid.color":         "#aaaaaa",
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
COLOR_IMPROVED  = "#1a6b3c"   # dark green  — distinguishable in greyscale
COLOR_WORSENED  = "#b03a2e"   # dark red
MARKER_IMPROVED = "o"
MARKER_WORSENED = "s"
MARKER_SIZE     = 26          # scatter s= (pt²)
ALPHA           = 0.85

# Default figure size: two-column paper column width (3.3 in).
# Pass --fig-width 6.5 for single-column / full-width figures.
FIG_W = 3.3
FIG_H = 2.8

# Datasets to label directly on each plot
LABELS_P1 = {"hospital", "electricity/W", "saugeenday/D"}
LABELS_P2 = {"hospital", "electricity/W", "saugeenday/D", "covid_deaths"}
LABELS_P3 = {"hospital", "electricity/W", "saugeenday/D"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_log(val, _):
    """Compact log-axis tick labels (1k, 10k, 1M …)."""
    if val >= 1_000_000:
        return f"{val / 1_000_000:.0f}M"
    if val >= 1_000:
        return f"{int(val / 1_000)}k"
    return str(int(val)) if val >= 1 else ""


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def _scatter(ax, df, xcol, ycol):
    """Two-group scatter: improved vs. worsened."""
    for improved, grp in df.groupby("improved"):
        ax.scatter(
            grp[xcol], grp[ycol],
            c=COLOR_IMPROVED if improved else COLOR_WORSENED,
            marker=MARKER_IMPROVED if improved else MARKER_WORSENED,
            s=MARKER_SIZE, alpha=ALPHA,
            linewidths=0.3, edgecolors="white",
            label=(r"improved ($\Delta < 0$)" if improved
                   else r"worsened ($\Delta \geq 0$)"),
            zorder=3,
        )


def _annotate(ax, df, xcol, ycol, label_set):
    for _, row in df.iterrows():
        if row["dataset"] not in label_set:
            continue
        ax.annotate(
            row["dataset"],
            xy=(row[xcol], row[ycol]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=COLOR_IMPROVED if row["improved"] else COLOR_WORSENED,
            va="bottom",
        )


def _threshold(ax, xval):
    ax.axvline(xval, color="#888888", linestyle="--", linewidth=0.65, zorder=1)


def _zero_line(ax):
    ax.axhline(0, color="#999999", linewidth=0.5, linestyle=":", zorder=1)


def _save(fig, path: Path):
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 1 — context × num_series vs mean_delta_pct
# ---------------------------------------------------------------------------

def plot1(df: pd.DataFrame, threshold: float, out: Path):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style_ax(ax)
    _scatter(ax, df, "score_series", "mean_delta_pct")
    _threshold(ax, threshold)
    _zero_line(ax)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{context\_length} \times \mathrm{num\_series}$")
    ax.set_ylabel(r"Mean $\Delta$ [\%]")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_log))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), markerscale=0.85,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 2 — context_length vs num_series  (2-D view, colour = improved)
# ---------------------------------------------------------------------------

def plot2(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style_ax(ax)
    _scatter(ax, df, "context_length", "num_series")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{context\_length}$ per series")
    ax.set_ylabel(r"$\mathrm{num\_series}$")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_log))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_log))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), markerscale=0.85,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 3 — context × total_windows vs mean_delta_pct
# ---------------------------------------------------------------------------

def plot3(df: pd.DataFrame, threshold: float, out: Path):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style_ax(ax)
    _scatter(ax, df, "score_windows", "mean_delta_pct")
    _threshold(ax, threshold)
    _zero_line(ax)

    ax.set_xscale("log")
    ax.set_xlabel(
        r"$\mathrm{context\_length} \times \mathrm{total\_windows}$" + "\n"
        r"$(\mathrm{total\_windows} = \mathrm{num\_series} \times \mathrm{windows})$",
        labelpad=4,
    )
    ax.set_ylabel(r"Mean $\Delta$ [\%]")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_log))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), markerscale=0.85,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 4 — total_windows vs. window_length  (colour = improved)
# ---------------------------------------------------------------------------

def plot4(df: pd.DataFrame, out: Path):
    """Plot 4: total_windows (X) vs. window_length (Y), colour = improved."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style_ax(ax)
    _scatter(ax, df, "window_length", "total_windows")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"$\mathrm{window\_length}$" + "\n"
        r"$(= \mathrm{context\_length} - (\mathrm{windows}{-}1) \times \mathrm{pred\_length})$",
        labelpad=4,
    )
    ax.set_ylabel(r"$\mathrm{total\_windows}$ ($= \mathrm{num\_series} \times \mathrm{windows}$)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_log))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_log))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), markerscale=0.85,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Window loading
# ---------------------------------------------------------------------------

def load_dataset_attrs(dataset_names: list) -> tuple[dict, dict]:
    """
    Load `windows` and `prediction_length` for each dataset via Dataset().
    Returns two dicts: (windows_map, pred_length_map).
    Falls back to windows=1, pred_length=1 on import/instantiation failure.
    """
    try:
        from finetune_tabpfn_ts.evaluation.data import Dataset
    except Exception as import_exc:
        warnings.warn(
            f"Cannot import finetune_tabpfn_ts.evaluation.data.Dataset "
            f"({type(import_exc).__name__}: {import_exc}) — "
            "falling back to windows=1 and pred_length=1 for all datasets. "
            "Tip: run with PYTHONPATH=/path/to/project python plot_finetuning.py ...",
            stacklevel=2,
        )
        return {n: 1 for n in dataset_names}, {n: 1 for n in dataset_names}

    windows_map, pred_map = {}, {}
    for name in dataset_names:
        try:
            ds = Dataset(name=name)
            windows_map[name] = int(ds.windows)
            pred_map[name]    = int(ds.prediction_length)
        except Exception as exc:
            warnings.warn(f"Dataset('{name}') failed ({exc}) — using defaults.")
            windows_map[name] = 1
            pred_map[name]    = 1
    return windows_map, pred_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global FIG_W, FIG_H

    parser = argparse.ArgumentParser(
        description="Four paper-quality PDF scatter plots for finetuning results."
    )
    parser.add_argument("csv",
                        help="Path to the finetuning results CSV")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for the three PDFs (default: .)")
    parser.add_argument("--score-threshold", type=float, default=50_000,
                        help="Dashed vertical line on plots 1 & 3 (default: 50000)")
    parser.add_argument("--fig-width", type=float, default=FIG_W,
                        help="Figure width in inches (default: 3.3 — two-column)")
    parser.add_argument("--fig-height", type=float, default=FIG_H,
                        help="Figure height in inches (default: 2.8)")
    parser.add_argument("--max-context-length", type=int, default=None,
                        help="If set, clip context_length and window_length to this value "
                             "(e.g. 4096). When omitted, raw values are used.")
    args = parser.parse_args()

    FIG_W = args.fig_width
    FIG_H = args.fig_height

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load & validate CSV ───────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        sys.exit(f"Error: file not found — {args.csv}")

    required = {"dataset", "context_length", "num_series", "mean_delta_pct", "improved"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: CSV is missing columns: {missing}")

    if df["improved"].dtype == object:
        df["improved"] = (
            df["improved"].str.strip().str.lower()
              .map({"true": True, "false": False, "1": True, "0": False})
        )

    # ── load windows + pred_length ───────────────────────────────────────────
    print("Loading dataset attributes from Dataset …")
    wmap, pmap = load_dataset_attrs(df["dataset"].tolist())
    df["windows"]     = df["dataset"].map(wmap)
    df["pred_length"] = df["dataset"].map(pmap)
    df["total_windows"] = df["num_series"] * df["windows"]

    # window_length before any capping
    df["window_length"] = df["context_length"] - (df["windows"] - 1) * df["pred_length"]

    # optional cap
    cap = args.max_context_length
    if cap is not None:
        print(f"Capping context_length and window_length to {cap}")
        df["context_length"] = df["context_length"].clip(upper=cap)
        df["window_length"]  = df["window_length"].clip(upper=cap)

    # derived scores (always use the (possibly capped) context_length)
    df["score_series"]  = df["context_length"] * df["num_series"]
    df["score_windows"] = df["context_length"] * df["total_windows"]

    print(df[["dataset", "num_series", "windows", "pred_length",
              "total_windows", "window_length"]].to_string(index=False))

    # ── generate PDFs ─────────────────────────────────────────────────────────
    print("\nGenerating PDFs …")
    plot1(df, args.score_threshold, out_dir / "plot1_score_series.pdf")
    plot2(df,                       out_dir / "plot2_context_vs_series.pdf")
    plot3(df, args.score_threshold, out_dir / "plot3_score_windows.pdf")
    plot4(df,                       out_dir / "plot4_windows_vs_windowlength.pdf")

    print(f"\nDone — four PDFs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()