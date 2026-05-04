#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MASE_COL_CANDIDATES = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MASE",
    "MASE",
]

DATASET_COL = "dataset"
FINETUNE_COL = "finetuning_place"
NUM_SERIES_COL_CANDIDATES = ["num_variates", "num_series"]
CKPTS = [1, 2, 3, 4, 5]


def find_first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_string_columns(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip('"')
    return df


def infer_context_length(dataset_name):
    if not isinstance(dataset_name, str):
        return ""
    parts = dataset_name.split("/")
    if len(parts) == 0:
        return ""
    return parts[-1]


def shorten_dataset_name(name, max_len=42):
    name = name.replace("/short", "")
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def load_single_csv(path):
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


def aggregate_all_csvs():
    files = sorted(Path(".").glob("*_evaluation.csv"))
    if not files:
        raise FileNotFoundError("Keine Dateien im Format '*_evaluation.csv' im aktuellen Ordner gefunden.")

    frames = []
    for path in files:
        frames.append(load_single_csv(path))

    full = pd.concat(frames, ignore_index=True)

    grouped = (
        full.groupby(["dataset", "finetuning_place"], as_index=False)
        .agg(
            mase=("mase", "mean"),
            num_series=("num_series", "mean"),
        )
    )

    return grouped


def build_summary_table(grouped):
    rows = []
    dataset_names = sorted(grouped["dataset"].unique())

    for dataset in dataset_names:
        sub = grouped[grouped["dataset"] == dataset].copy()

        base_row = sub[sub["finetuning_place"] == -1]
        if base_row.empty:
            continue

        base_mase = float(base_row["mase"].iloc[0])
        if not np.isfinite(base_mase) or base_mase == 0:
            continue

        num_series = base_row["num_series"].iloc[0]
        if pd.isna(num_series):
            valid_num_series = sub["num_series"].dropna()
            num_series = valid_num_series.iloc[0] if len(valid_num_series) > 0 else np.nan

        deltas = {}
        available = []

        for ckpt in CKPTS:
            ckpt_row = sub[sub["finetuning_place"] == ckpt]
            if ckpt_row.empty:
                delta_pct = np.nan
            else:
                ckpt_mase = float(ckpt_row["mase"].iloc[0])
                delta_pct = ((ckpt_mase - base_mase) / base_mase) * 100.0
                available.append(delta_pct)
            deltas[f"ckpt_{ckpt}_delta_pct"] = delta_pct

        mean_delta = float(np.nanmean(available)) if len(available) > 0 else np.nan
        min_delta = float(np.nanmin(available)) if len(available) > 0 else np.nan
        max_delta = float(np.nanmax(available)) if len(available) > 0 else np.nan
        improved = bool(mean_delta < 0) if not np.isnan(mean_delta) else False

        rows.append(
            {
                "dataset": dataset,
                "context_length": infer_context_length(dataset),
                "num_series": int(num_series) if not pd.isna(num_series) else np.nan,
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


def set_paper_style():
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


def make_single_diverging_barplot(
    df,
    value_col,
    output_base,
    title,
    x_label="Δ MASE [%] vs. original checkpoint",
):
    plot_df = df[["dataset", value_col]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(value_col, ascending=True).reset_index(drop=True)

    labels = [shorten_dataset_name(x) for x in plot_df["dataset"].tolist()]
    values = plot_df[value_col].to_numpy()

    set_paper_style()

    height = max(4.5, 0.32 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, height))

    colors = ["#2166ac" if v < 0 else "#b2182b" for v in values]
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
        lim = max_abs * 1.12
        ax.set_xlim(-lim, lim)

    fig.tight_layout()
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_combined_barplot(
    df,
    output_base="mase_all_combined_plot",
    title="Relative change in MASE across all checkpoints",
    x_label="Δ MASE [%] vs. original checkpoint",
):
    cols = [
        "mean_delta_pct",
        "ckpt_1_delta_pct",
        "ckpt_2_delta_pct",
        "ckpt_3_delta_pct",
        "ckpt_4_delta_pct",
        "ckpt_5_delta_pct",
    ]
    labels_for_legend = [
        "mean",
        "ckpt 1",
        "ckpt 2",
        "ckpt 3",
        "ckpt 4",
        "ckpt 5",
    ]

    plot_df = df[["dataset"] + cols].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("mean_delta_pct", ascending=True).reset_index(drop=True)
    dataset_labels = [shorten_dataset_name(x) for x in plot_df["dataset"].tolist()]

    set_paper_style()

    n_datasets = len(plot_df)
    n_series = len(cols)
    group_height = 0.82
    bar_h = group_height / n_series

    height = max(5.0, 0.45 * n_datasets + 2.0)
    fig, ax = plt.subplots(figsize=(10.0, height))

    y_base = np.arange(n_datasets)

    legend_handles = []

    for i, col in enumerate(cols):
        offset = (i - (n_series - 1) / 2) * bar_h
        y = y_base + offset
        values = plot_df[col].to_numpy(dtype=float)

        valid = np.isfinite(values)
        bars = ax.barh(
            y[valid],
            values[valid],
            height=bar_h * 0.92,
            edgecolor="black",
            linewidth=0.5,
            label=labels_for_legend[i],
        )
        legend_handles.append(bars)

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y_base)
    ax.set_yticklabels(dataset_labels)
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

    ax.legend(loc="best", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    grouped = aggregate_all_csvs()
    summary = build_summary_table(grouped)

    if summary.empty:
        print("Keine auswertbaren Datensätze gefunden.")
        sys.exit(1)

    summary.to_csv("mase_checkpoint_summary.csv", index=False)

    make_single_diverging_barplot(
        df=summary,
        value_col="mean_delta_pct",
        output_base="mase_mean_delta_plot",
        title="Mean relative change in MASE across checkpoints 1-5",
    )

    for ckpt in CKPTS:
        col = f"ckpt_{ckpt}_delta_pct"
        make_single_diverging_barplot(
            df=summary.sort_values(col, ascending=True).reset_index(drop=True),
            value_col=col,
            output_base=f"mase_ckpt_{ckpt}_delta_plot",
            title=f"Relative change in MASE for checkpoint {ckpt}",
        )

    make_combined_barplot(
        df=summary,
        output_base="mase_all_combined_plot",
        title="Relative change in MASE across checkpoints 1-5 and mean",
    )

    print("Fertig.")
    print("Erstellt:")
    print("  - mase_checkpoint_summary.csv")
    print("  - mase_mean_delta_plot.pdf/.png")
    for ckpt in CKPTS:
        print(f"  - mase_ckpt_{ckpt}_delta_plot.pdf/.png")
    print("  - mase_all_combined_plot.pdf/.png")


if __name__ == "__main__":
    main()