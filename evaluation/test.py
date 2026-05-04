"""
Visualize train / validation / test splits using the Dataset class methods:
    - dataset.training_dataset
    - dataset.validation_dataset
    - dataset.test_data

Usage:
    python visualize_splits.py monash_covid_deaths
    python visualize_splits.py electricity --index 3
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from gluonts.dataset.split import split
from finetune_tabpfn_ts.evaluation.data import Dataset, Term


COLORS = {
    "train": "#4C72B0",
    "val":   "#DD8452",
    "test":  "#55A868",
}


def get_nth_entry(iterable, index: int) -> np.ndarray:
    """Return the target array of the n-th entry from any iterable dataset."""
    for i, entry in enumerate(iterable):
        if i == index:
            t = entry["target"]
            return t[0] if t.ndim > 1 else t
    raise IndexError(f"Series index {index} out of range.")


def collect_windows(test_data, series_index: int, n_windows: int):
    """
    Collect all (past_target, label_target) windows for a given series index.
    GluonTS ordering: window 0 of series 0, window 1 of series 0, …
    """
    windows = []
    for i, (inp, label) in enumerate(test_data):
        if i // n_windows == series_index:
            t_inp = inp["target"]
            t_lbl = label["target"]
            if t_inp.ndim > 1: t_inp = t_inp[0]
            if t_lbl.ndim > 1: t_lbl = t_lbl[0]
            windows.append((t_inp, t_lbl))
        elif i // n_windows > series_index:
            break
    return windows


def plot_splits(name: str, index: int = 0):
    ds = Dataset(name, term=Term.SHORT)
    pred = ds.prediction_length
    wins = ds.windows

    # ── fetch data from each split ───────────────────────────────────────────
    train_series = get_nth_entry(ds.training_dataset, index)
    val_series   = get_nth_entry(ds.validation_dataset, index)
    test_windows = collect_windows(ds.test_data, index, wins)

    T_train = len(train_series)
    T_val   = len(val_series)
    T_total = T_val + wins * pred

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)
    fig.suptitle(
        f"Dataset: {name}  |  Series index: {index}  |  "
        f"Freq: {ds.freq}  |  Pred length: {pred}  |  Windows: {wins}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── helpers ───────────────────────────────────────────────────────────────
    def shade(ax, start, end, color, alpha=0.13):
        if end > start:
            ax.axvspan(start, end - 1, color=color, alpha=alpha)

    # ── Plot 1: training_dataset ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(np.arange(T_train), train_series,
            color=COLORS["train"], linewidth=1.5, label="Training")
    shade(ax, 0, T_train, COLORS["train"], alpha=0.15)
    ax.set_title(f"training_dataset  —  length: {T_train}", fontsize=11)
    ax.set_ylabel("Value"); ax.legend(loc="upper left")
    ax.set_xlim(0, T_train - 1)

    # ── Plot 2: validation_dataset ────────────────────────────────────────────
    ax = axes[1]
    t_val = np.arange(T_val)
    ax.plot(t_val[:T_train], val_series[:T_train],
            color="lightgrey", linewidth=1, label="Train context")
    ax.plot(t_val[T_train:], val_series[T_train:],
            color=COLORS["val"], linewidth=2,
            label=f"Validation window (+{T_val - T_train} = 1 × pred)")
    shade(ax, 0, T_train,    COLORS["train"], alpha=0.07)
    shade(ax, T_train, T_val, COLORS["val"],  alpha=0.2)
    ax.axvline(T_train, color=COLORS["val"], linewidth=1.2, linestyle="--")
    ax.set_title(
        f"validation_dataset  —  length: {T_val}  "
        f"(train {T_train} + val window {T_val - T_train})",
        fontsize=11,
    )
    ax.set_ylabel("Value"); ax.legend(loc="upper left")
    ax.set_xlim(0, T_val - 1)

    # ── Plot 3: test_data ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(np.arange(T_val), val_series,
            color="lightgrey", linewidth=1)
    shade(ax, 0, T_train, COLORS["train"], alpha=0.05)
    shade(ax, T_train, T_val, COLORS["val"], alpha=0.07)

    for w, (inp, lbl) in enumerate(test_windows):
        ctx_len   = len(inp)
        lbl_start = T_val + w * pred
        lbl_end   = lbl_start + pred
        ctx_start = lbl_start - ctx_len

        t_ctx = np.arange(ctx_start, lbl_start)
        t_lbl = np.arange(lbl_start, lbl_end)

        ax.plot(t_ctx, inp,
                color=COLORS["train"], linewidth=1, alpha=0.3,
                label="Context (past_target)" if w == 0 else None)
        ax.plot(t_lbl, lbl,
                color=COLORS["test"], linewidth=2,
                label="Forecast label" if w == 0 else None)
        shade(ax, lbl_start, lbl_end, COLORS["test"], alpha=0.18)
        ax.axvline(lbl_start, color=COLORS["test"], linewidth=0.8,
                   linestyle="--", alpha=0.6)
        ax.text((lbl_start + lbl_end) / 2,
                ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(val_series),
                f"w{w+1}", ha="center", va="top",
                fontsize=8, color=COLORS["test"], clip_on=True)

    ax.axvline(T_val, color="black", linewidth=1.2, linestyle=":",
               label="Val / Test boundary")
    ax.set_title(
        f"test_data  —  {wins} window(s) × pred={pred}  "
        f"=  {wins * pred} forecast steps",
        fontsize=11,
    )
    ax.set_ylabel("Value"); ax.set_xlabel("Time step")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, T_total - 1)

    # ── shared legend ─────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=COLORS["train"], alpha=0.5, label="Train region"),
        mpatches.Patch(color=COLORS["val"],   alpha=0.5, label="Validation window"),
        mpatches.Patch(color=COLORS["test"],  alpha=0.5, label="Test forecast windows"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()

    out = f"splits_{name.replace('/', '_')}_idx{index}.png"
    plt.savefig("finetune_tabpfn_ts/evaluation/" + out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize dataset splits.")
    parser.add_argument("--dataset", help="Dataset name, e.g. monash_covid_deaths")
    parser.add_argument("--index", type=int, default=0,
                        help="Series index to visualize (default: 0)")
    args = parser.parse_args()
    plot_splits(args.dataset, args.index)