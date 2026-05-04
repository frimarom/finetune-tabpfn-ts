#!/usr/bin/env python3
"""
plot_results.py
---------------
Liest mase_checkpoint_summary.csv und erstellt alle Plots.

Datensätze bei denen `trained_on == True` werden in den Plots
mit einem eigenen Farbton hervorgehoben.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

CKPTS = [1, 2, 3, 4, 5]
INPUT_CSV = "mase_checkpoint_summary.csv"

# Farben für Balken – klassisch blau/rot
COLOR_IMPROVED = "#2166ac"
COLOR_WORSENED = "#b2182b"


# ---------------------------------------------------------------------------
# Stil
# ---------------------------------------------------------------------------

def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def shorten_dataset_name(name: str, max_len: int = 42) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Farb-Hilfsfunktion
# ---------------------------------------------------------------------------

def bar_colors(values: np.ndarray) -> list[str]:
    """Blau = verbessert, Rot = verschlechtert."""
    return [COLOR_IMPROVED if v < 0 else COLOR_WORSENED for v in values]


# ---------------------------------------------------------------------------
# Einzel-Barplot
# ---------------------------------------------------------------------------

def make_single_diverging_barplot(
    df: pd.DataFrame,
    value_col: str,
    output_base: str,
    title: str,
    x_label: str = "Δ MASE [%] vs. original checkpoint",
) -> None:
    needed = ["dataset", value_col]
    if "trained_on" in df.columns:
        needed.append("trained_on")
    plot_df = df[needed].dropna(subset=[value_col]).copy()
    if plot_df.empty:
        return

    if "trained_on" not in plot_df.columns:
        plot_df["trained_on"] = False
    else:
        plot_df["trained_on"] = plot_df["trained_on"].astype(str).str.lower().isin(["true", "1", "yes"])

    plot_df = plot_df.sort_values(value_col, ascending=True).reset_index(drop=True)

    def _label(name: str, trained: bool) -> str:
        s = shorten_dataset_name(name)
        return f"[T]  {s}" if trained else s

    labels  = [_label(r["dataset"], r["trained_on"]) for _, r in plot_df.iterrows()]
    values  = plot_df[value_col].to_numpy(dtype=float)

    set_paper_style()

    height = max(4.5, 0.32 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, height))

    colors = bar_colors(values)
    y = np.arange(len(plot_df))

    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6, height=0.75)
    ax.axvline(0, color="black", linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_title(title, pad=10)

    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    max_abs = np.nanmax(np.abs(values))
    if np.isfinite(max_abs) and max_abs > 0:
        ax.set_xlim(-max_abs * 1.12, max_abs * 1.12)

    legend_patches = [
        mpatches.Patch(color=COLOR_IMPROVED, label="improved"),
        mpatches.Patch(color=COLOR_WORSENED, label="worsened"),
    ]
    ax.legend(handles=[*legend_patches], loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {output_base}.pdf / .png")


# ---------------------------------------------------------------------------
# Kombinierter Barplot
# ---------------------------------------------------------------------------

def make_combined_barplot(
    df: pd.DataFrame,
    output_base: str = "mase_all_combined_plot",
    title: str = "Relative change in MASE across all checkpoints",
    x_label: str = "Δ MASE [%] vs. original checkpoint",
) -> None:
    cols = [
        "mean_delta_pct",
        "ckpt_1_delta_pct",
        "ckpt_2_delta_pct",
        "ckpt_3_delta_pct",
        "ckpt_4_delta_pct",
        "ckpt_5_delta_pct",
    ]
    legend_labels = ["mean", "ckpt 1", "ckpt 2", "ckpt 3", "ckpt 4", "ckpt 5"]

    plot_df = df[["dataset", "trained_on"] + cols].copy() if "trained_on" in df.columns else df[["dataset"] + cols].copy()
    if "trained_on" not in plot_df.columns:
        plot_df["trained_on"] = False
    else:
        plot_df["trained_on"] = plot_df["trained_on"].astype(str).str.lower().isin(["true", "1", "yes"])

    if plot_df.empty:
        return

    plot_df  = plot_df.sort_values("mean_delta_pct", ascending=True).reset_index(drop=True)
    d_labels = [
        f"[T]  {shorten_dataset_name(r['dataset'])}" if r["trained_on"] else shorten_dataset_name(r["dataset"])
        for _, r in plot_df.iterrows()
    ]

    set_paper_style()

    n_datasets = len(plot_df)
    n_series   = len(cols)
    group_height = 0.82
    bar_h = group_height / n_series

    height = max(5.0, 0.45 * n_datasets + 2.0)
    fig, ax = plt.subplots(figsize=(10.0, height))

    y_base = np.arange(n_datasets)

    for i, col in enumerate(cols):
        offset = (i - (n_series - 1) / 2) * bar_h
        y      = y_base + offset
        values = plot_df[col].to_numpy(dtype=float)
        valid  = np.isfinite(values)

        colors = bar_colors(values[valid])

        ax.barh(
            y[valid],
            values[valid],
            height=bar_h * 0.92,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            label=legend_labels[i],
        )

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y_base)
    ax.set_yticklabels(d_labels)
    ax.set_xlabel(x_label)
    ax.set_title(title, pad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    all_values = plot_df[cols].to_numpy(dtype=float).ravel()
    all_values = all_values[np.isfinite(all_values)]
    if len(all_values) > 0:
        lim = np.max(np.abs(all_values)) * 1.12
        if lim > 0:
            ax.set_xlim(-lim, lim)

    # Legende: Checkpoint-Labels
    ckpt_handles = [
        mpatches.Patch(color="grey", label=lbl) for lbl in legend_labels
    ]
    color_handles = [
        mpatches.Patch(color=COLOR_IMPROVED, label="improved"),
        mpatches.Patch(color=COLOR_WORSENED, label="worsened"),
    ]
    from matplotlib.lines import Line2D
    trained_handle = Line2D(
        [], [], marker="s", color="w", markerfacecolor="#1a1a1a",
        markersize=7, label="[T]  trained on",
    )
    legend1 = ax.legend(
        handles=ckpt_handles,
        loc="lower right",
        frameon=False,
        ncol=2,
        title="Checkpoint",
    )
    ax.add_artist(legend1)
    ax.legend(handles=[trained_handle, *color_handles], loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {output_base}.pdf / .png")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def generate_plots(csv_path: str = INPUT_CSV) -> None:
    """
    Liest die Summary-CSV und erstellt alle Plots.

    Args:
        csv_path: Pfad zur mase_checkpoint_summary.csv
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV nicht gefunden: {csv_path}\n"
            "Bitte zuerst mit --csv die Summary-CSV generieren."
        )

    summary = pd.read_csv(csv_path)

    print(f"[Plot] {len(summary)} Datensätze geladen.")

    make_single_diverging_barplot(
        df=summary,
        value_col="mean_delta_pct",
        output_base="mase_mean_delta_plot",
        title="Mean relative change in MASE across checkpoints 1–5",
    )

    for ckpt in CKPTS:
        col = f"ckpt_{ckpt}_delta_pct"
        make_single_diverging_barplot(
            df=summary.sort_values(col, ascending=True).reset_index(drop=True),
            value_col=col,
            output_base=f"mase_ckpt_{ckpt}_delta_plot",
            title=f"Relative change in MASE – checkpoint {ckpt}",
        )

    make_combined_barplot(
        df=summary,
        output_base="mase_all_combined_plot",
        title="Relative change in MASE across checkpoints 1–5 and mean",
    )

    print("[Plot] Alle Plots erstellt.")


if __name__ == "__main__":
    generate_plots()