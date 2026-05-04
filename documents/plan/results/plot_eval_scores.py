"""
eval/plot_eval_scores.py

Reads evaluation.csv + finetuning_summary.csv from:
  eval/single_finetune/<dataset_name>/<optional_frequency>/

Uses MASE[0.5] as the metric.
  finetuning_place == -1  →  pre score (original model, used as x-coordinate)
  finetuning_place >= 1   →  post scores (ordered by place, 1 = biggest circle)

Hyperparameter heatmaps use top-5 configs by Best Validation Loss from
finetuning_summary.csv, joined on Dataset_1 == dataset folder name.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.join(SCRIPT_DIR, "single_finetune")
METRIC_COL    = "eval_metrics/MASE[0.5]"

MAX_POST_SIZE = 320
MIN_POST_SIZE = 50

BASE_COLORS = [
    "#E63946", "#2196F3", "#2DC653", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#E91E63", "#607D8B",
]
ALPHA_POST = 0.82

CLIP_PERCENTILE = 95   # top X% treated as outliers in combined plot

HPARAM_COLS = ["Learning Rate", "Batch Size", "L2-SP Lambda"]
TOP_N       = 5        # top configs by Best Validation Loss
# ─────────────────────────────────────────────────────────────────────────────


def find_csv_files(base_dir: str) -> list:
    entries = []
    dataset_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    for ds_dir in dataset_dirs:
        dataset_name = os.path.basename(ds_dir)
        direct_csv   = os.path.join(ds_dir, "evaluation.csv")
        direct_sum   = os.path.join(ds_dir, "finetuning_summary.csv")
        if os.path.isfile(direct_csv):
            entries.append({
                "dataset": dataset_name, "frequency": None,
                "path": direct_csv,
                "summary_path": direct_sum if os.path.isfile(direct_sum) else None,
            })
        else:
            for freq_dir in glob.glob(os.path.join(ds_dir, "*")):
                if not os.path.isdir(freq_dir):
                    continue
                freq_csv = os.path.join(freq_dir, "evaluation.csv")
                freq_sum = os.path.join(freq_dir, "finetuning_summary.csv")
                if os.path.isfile(freq_csv):
                    entries.append({
                        "dataset":   dataset_name,
                        "frequency": os.path.basename(freq_dir),
                        "path":      freq_csv,
                        "summary_path": freq_sum if os.path.isfile(freq_sum) else None,
                    })
    return entries


def load_entry(entry: dict):
    df = pd.read_csv(entry["path"])
    df.columns = df.columns.str.strip()
    if METRIC_COL not in df.columns or "finetuning_place" not in df.columns:
        print(f"  [SKIP] Missing columns in {entry['path']}")
        return None
    df["finetuning_place"] = pd.to_numeric(df["finetuning_place"], errors="coerce")
    df[METRIC_COL]         = pd.to_numeric(df[METRIC_COL],         errors="coerce")
    df = df.dropna(subset=["finetuning_place", METRIC_COL])
    pre_rows  = df[df["finetuning_place"] == -1]
    post_rows = df[df["finetuning_place"] >= 1].sort_values("finetuning_place")
    if pre_rows.empty:
        print(f"  [SKIP] No finetuning_place==-1 row in {entry['path']}")
        return None

    # Load hyperparameters
    hparams_df = None
    if entry.get("summary_path"):
        try:
            sdf = pd.read_csv(entry["summary_path"])
            sdf.columns = sdf.columns.str.strip()
            # Filter to this dataset and take top N by Best Validation Loss
            mask = sdf["Dataset_1"].astype(str).str.strip() == entry["dataset"]
            sdf  = sdf[mask].copy()
            if not sdf.empty:
                sdf = sdf.sort_values("Best Validation Loss").head(TOP_N)
                avail = [c for c in HPARAM_COLS if c in sdf.columns]
                for c in avail:
                    sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
                hparams_df = sdf[avail].reset_index(drop=True)
        except Exception as e:
            print(f"  [WARN] Could not load summary for {entry['dataset']}: {e}")

    label = entry["dataset"]
    if entry["frequency"]:
        label += f" ({entry['frequency']})"

    return {
        "label":       label,
        "dataset":     entry["dataset"],
        "pre_score":   pre_rows[METRIC_COL].iloc[0],
        "post_scores": post_rows[METRIC_COL].tolist(),
        "post_places": post_rows["finetuning_place"].tolist(),
        "hparams_df":  hparams_df,   # DataFrame of top-N configs x hparam cols
    }


def compute_sizes(n: int) -> list:
    if n == 1:
        return [MAX_POST_SIZE]
    return list(np.linspace(MAX_POST_SIZE, MIN_POST_SIZE, n))


# ── SCATTER PLOTS ─────────────────────────────────────────────────────────────

def plot_dataset_axes(ax, entry_data: dict, color: str, title: str = ""):
    pre   = entry_data["pre_score"]
    posts = entry_data["post_scores"]
    sizes = compute_sizes(len(posts))
    all_vals = [pre] + posts
    lo, hi   = min(all_vals), max(all_vals)
    margin   = (hi - lo) * 0.15 if hi != lo else max(abs(pre) * 0.1, 0.05)
    lim      = (lo - margin, hi + margin)
    ax.set_xlim(lim); ax.set_ylim(lim)
    diag = np.linspace(lim[0], lim[1], 300)
    ax.plot(diag, diag, color="#888888", linewidth=1.2, linestyle="--", zorder=1, alpha=0.55)
    ax.fill_between(diag, lim[0], diag, color="#2DC653", alpha=0.07, zorder=0)
    ax.fill_between(diag, diag, lim[1], color="#E63946", alpha=0.07, zorder=0)
    for val, sz in zip(posts, sizes):
        ax.scatter(pre, val, s=sz, color=color, alpha=ALPHA_POST,
                   edgecolors="white", linewidths=0.8, zorder=3)
    ax.text(lim[0]+margin*0.3, lim[0]+margin*0.5, "▼ better", fontsize=7, color="#2DC653", alpha=0.8)
    ax.text(lim[0]+margin*0.3, lim[1]-margin*0.6, "▲ worse",  fontsize=7, color="#E63946", alpha=0.8)
    ax.set_xlabel("MASE – original model", fontsize=9)
    ax.set_ylabel("MASE – finetuned",      fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


def plot_combined_single(ax, entries: list, color_map: dict):
    all_x    = [e["pre_score"] for e in entries]
    all_y    = [v for e in entries for v in e["post_scores"]]
    all_vals = all_x + all_y
    clip_max = np.percentile(all_vals, CLIP_PERCENTILE)
    lo_raw   = min(all_vals)
    margin   = (clip_max - max(0, lo_raw)) * 0.08
    lo = max(0, lo_raw) - margin
    hi = clip_max + margin
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    diag = np.linspace(lo, hi, 300)
    ax.plot(diag, diag, color="#888888", linewidth=1.2, linestyle="--", zorder=1, alpha=0.6)
    ax.fill_between(diag, lo, diag, color="#2DC653", alpha=0.05, zorder=0)
    ax.fill_between(diag, diag, hi,  color="#E63946", alpha=0.05, zorder=0)
    for entry_data in entries:
        pre   = entry_data["pre_score"]
        posts = entry_data["post_scores"]
        sizes = compute_sizes(len(posts))
        color = color_map[entry_data["label"]]
        x_clipped = pre > hi
        x_plot    = min(pre, hi - margin * 0.3)
        for val, sz, place in zip(posts, sizes, entry_data["post_places"]):
            y_clipped = val > hi
            y_plot    = min(val, hi - margin * 0.3)
            ax.scatter(x_plot, y_plot, s=sz, color=color, alpha=ALPHA_POST,
                       edgecolors="white", linewidths=0.9, zorder=4)
            if y_clipped:
                ax.annotate(f"{val:.2f}", xy=(x_plot, hi-margin*0.3),
                            xytext=(x_plot, hi-margin*1.8),
                            fontsize=6.5, color=color, ha="center",
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=8), zorder=6)
            if x_clipped:
                ax.annotate(f"x={pre:.2f}", xy=(hi-margin*0.3, y_plot),
                            xytext=(hi-margin*2.5, y_plot),
                            fontsize=6.5, color=color, ha="right",
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=8), zorder=6)
    ax.text(lo+margin, lo+margin*1.5, "▼ better", fontsize=8, color="#2DC653", alpha=0.85)
    ax.text(lo+margin, hi-margin*1.5, "▲ worse",  fontsize=8, color="#E63946", alpha=0.85)
    if max(all_vals) > clip_max:
        ax.text(hi-margin*0.2, lo+margin*0.5,
                f"clip @ {clip_max:.2f} ({CLIP_PERCENTILE}th pct)",
                fontsize=7, color="#888", ha="right", style="italic")
    ax.set_xlabel("MASE – original model", fontsize=10)
    ax.set_ylabel("MASE – finetuned",      fontsize=10)
    ax.set_title("All datasets – MASE comparison", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)


# ── HEATMAP HELPERS ───────────────────────────────────────────────────────────

def _draw_heatmap(ax, matrix: pd.DataFrame, title: str, fmt: str = ".3f",
                  cmap_center: float = 0.0, is_delta: bool = True):
    """Draw a single annotated heatmap on ax."""
    vals = matrix.values.astype(float)

    if is_delta:
        # diverging: negative delta = better (green), positive = worse (red)
        vmax = np.nanmax(np.abs(vals)) or 1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = "RdYlGn_r"
    else:
        norm = None
        cmap = "YlOrRd"

    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)

    # Annotate cells
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            txt = format(v, fmt)
            brightness = im.norm(v)
            text_color = "white" if (brightness < 0.35 or brightness > 0.75) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")


def build_heatmap_matrix(entry_data: dict, metric: str) -> pd.DataFrame | None:
    """
    Returns a DataFrame: rows = top-N config index, cols = hyperparams,
    with cell values = chosen metric (delta MASE or absolute MASE).
    """
    hdf = entry_data.get("hparams_df")
    if hdf is None or hdf.empty:
        return None

    avail_hp = [c for c in HPARAM_COLS if c in hdf.columns]
    if not avail_hp:
        return None

    pre    = entry_data["pre_score"]
    posts  = entry_data["post_scores"]
    places = entry_data["post_places"]
    n_rows = min(len(hdf), len(posts))

    rows = []
    for i in range(n_rows):
        row = {}
        for hp in avail_hp:
            row[hp] = hdf[hp].iloc[i]
        if metric == "delta":
            row["_val"] = posts[i] - pre          # negative = improvement
        else:
            row["_val"] = posts[i]
        rows.append(row)

    if not rows:
        return None

    df_out = pd.DataFrame(rows)

    # For heatmap: rows = config rank, cols = hparams, color = metric value
    # We create one row per config showing its hparam values coloured by metric
    # Better: pivot so each column is a hparam, value is the hparam value,
    # but colour by metric → use a single-column metric column beside hparam cols.
    # Practical approach: rows = configs, cols = hparams, annotate hparam value,
    # colour = metric _val (same shade across all hp cols per row)
    result = df_out[avail_hp].copy()
    result.index = [f"cfg {int(places[i])}" for i in range(n_rows)]
    result["MASE" if metric == "abs" else "ΔMASE"] = df_out["_val"].values
    return result


def plot_heatmap_for_entry(entry_data: dict, out_dir: str):
    """Save delta + absolute heatmap for one dataset."""
    label = entry_data["label"]
    safe  = label.replace(" ","_").replace("/","-").replace("(","").replace(")","")

    for metric, is_delta, suffix, fmt in [
        ("delta", True,  "delta",  "+.3f"),
        ("abs",   False, "abs",    ".3f"),
    ]:
        mat = build_heatmap_matrix(entry_data, metric)
        if mat is None:
            continue

        fig, ax = plt.subplots(figsize=(max(5, len(mat.columns) * 1.4 + 1),
                                        max(3, len(mat) * 0.7 + 1.5)))
        fig.patch.set_facecolor("#F7F7F7")
        ax.set_facecolor("#FFFFFF")
        title = (f"{label} – ΔMASE per config & hyperparameter\n"
                 f"(negative = improvement)" if is_delta else
                 f"{label} – MASE (absolute) per config & hyperparameter")
        _draw_heatmap(ax, mat, title, fmt=fmt, is_delta=is_delta)
        fig.tight_layout()
        path = os.path.join(out_dir, f"heatmap_{safe}_{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


def plot_heatmap_aggregated(entries: list, out_dir: str):
    """
    Aggregated heatmap over all datasets.
    For each hyperparameter bucket (binned), show mean delta / mean abs MASE.
    Rows = datasets, cols = hyperparams (annotated with mean value of that hp
    for top-N configs), colour = mean ΔMASE or mean MASE of those configs.
    """
    rows_delta = {}
    rows_abs   = {}

    for entry_data in entries:
        label = entry_data["label"]
        pre   = entry_data["pre_score"]
        posts = entry_data["post_scores"]
        hdf   = entry_data.get("hparams_df")
        if hdf is None or hdf.empty or not posts:
            continue

        avail_hp = [c for c in HPARAM_COLS if c in hdf.columns]
        n        = min(len(hdf), len(posts))

        row_d = {}
        row_a = {}
        for hp in avail_hp:
            # mean hparam value of top-N configs (annotated in cell)
            hp_vals   = hdf[hp].iloc[:n].values.astype(float)
            mase_vals = np.array(posts[:n])
            delta_vals = mase_vals - pre
            # store tuple (mean_hp_val, mean_metric) → display hp, colour metric
            row_d[hp] = (np.nanmean(hp_vals), np.nanmean(delta_vals))
            row_a[hp] = (np.nanmean(hp_vals), np.nanmean(mase_vals))

        rows_delta[label] = row_d
        rows_abs[label]   = row_a

    if not rows_delta:
        print("  [WARN] No data for aggregated heatmap.")
        return

    all_hp = [c for c in HPARAM_COLS if any(c in r for r in rows_delta.values())]

    for rows, is_delta, suffix, fmt in [
        (rows_delta, True,  "delta", "+.3f"),
        (rows_abs,   False, "abs",   ".3f"),
    ]:
        # Build two matrices: annotation (hp value) and colour (metric)
        datasets = list(rows.keys())
        annot_mat = pd.DataFrame(index=datasets, columns=all_hp, dtype=float)
        color_mat = pd.DataFrame(index=datasets, columns=all_hp, dtype=float)

        for ds in datasets:
            for hp in all_hp:
                if hp in rows[ds]:
                    hp_val, metric_val = rows[ds][hp]
                    annot_mat.loc[ds, hp] = hp_val
                    color_mat.loc[ds, hp] = metric_val

        fig, ax = plt.subplots(figsize=(max(5, len(all_hp) * 2.2 + 1),
                                        max(4, len(datasets) * 0.75 + 1.5)))
        fig.patch.set_facecolor("#F7F7F7")

        vals = color_mat.values.astype(float)
        if is_delta:
            vmax = np.nanmax(np.abs(vals)) or 1
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = "RdYlGn_r"
        else:
            norm = None
            cmap = "YlOrRd"

        im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03,
                     label="ΔMASE (mean)" if is_delta else "MASE (mean)")

        ax.set_xticks(range(len(all_hp)))
        ax.set_xticklabels(all_hp, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)

        title = ("All datasets – mean ΔMASE coloured, mean HP value annotated\n"
                 "(green = improvement)" if is_delta else
                 "All datasets – mean MASE coloured, mean HP value annotated")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

        annot_vals = annot_mat.values.astype(float)
        for i in range(len(datasets)):
            for j in range(len(all_hp)):
                v_col  = vals[i, j]
                v_ann  = annot_vals[i, j]
                if np.isnan(v_ann):
                    continue
                # Format annotation: scientific for very small/large
                if abs(v_ann) < 0.001 or abs(v_ann) > 9999:
                    ann_txt = f"{v_ann:.2e}"
                else:
                    ann_txt = f"{v_ann:.4g}"
                brightness = im.norm(v_col)
                text_color = "white" if (brightness < 0.35 or brightness > 0.75) else "black"
                ax.text(j, i, ann_txt, ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

        fig.tight_layout()
        path = os.path.join(out_dir, f"heatmap_ALL_{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ── LEGEND ────────────────────────────────────────────────────────────────────

def make_legend_handles(color_map: dict, max_posts: int) -> list:
    handles = []
    for label, col in color_map.items():
        handles.append(mpatches.Patch(facecolor=col, edgecolor="white",
                                      linewidth=0.5, label=label))
    handles.append(Line2D([0], [0], linestyle="None", label=""))
    for i, sz in enumerate(compute_sizes(max_posts), 1):
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                   markersize=np.sqrt(sz)*0.55, label=f"finetuning_place={i}")
        )
    handles.append(Line2D([0], [0], color="#888", linewidth=1.2,
                          linestyle="--", label="no change"))
    return handles


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    entries_raw = find_csv_files(BASE_DIR)
    if not entries_raw:
        print(f"No evaluation.csv files found under {BASE_DIR}")
        return

    entries = [e for e in (load_entry(r) for r in entries_raw) if e is not None]
    if not entries:
        print("No valid data could be loaded.")
        return

    print(f"Loaded {len(entries)} dataset(s).")
    color_map = {e["label"]: BASE_COLORS[i % len(BASE_COLORS)]
                 for i, e in enumerate(entries)}
    max_posts = max(len(e["post_scores"]) for e in entries)

    out_dir = os.path.join(SCRIPT_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # ── Individual scatter plots ──────────────────────────────────────────────
    for entry_data in entries:
        lbl   = entry_data["label"]
        color = color_map[lbl]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#F7F7F7"); ax.set_facecolor("#FFFFFF")
        plot_dataset_axes(ax, entry_data, color, title=lbl)
        sizes = compute_sizes(len(entry_data["post_scores"]))
        leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=color,
                      markersize=np.sqrt(sz)*0.55, label=f"place={int(p)}")
               for sz, p in zip(sizes, entry_data["post_places"])]
        leg.append(Line2D([0],[0], color="#888", linewidth=1.2,
                          linestyle="--", label="no change"))
        ax.legend(handles=leg, fontsize=8, loc="upper left", framealpha=0.85)
        fig.tight_layout()
        safe = lbl.replace(" ","_").replace("/","-").replace("(","").replace(")","")
        path = os.path.join(out_dir, f"eval_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)

    # ── Grid overview ─────────────────────────────────────────────────────────
    n    = len(entries)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig_grid, axes = plt.subplots(rows, cols,
                                  figsize=(5.5*cols, 5.2*rows+1.8), squeeze=False)
    fig_grid.patch.set_facecolor("#F0F0F0")
    axes_flat = axes.flatten()
    for idx, entry_data in enumerate(entries):
        plot_dataset_axes(axes_flat[idx], entry_data,
                          color_map[entry_data["label"]], title=entry_data["label"])
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig_grid.legend(handles=make_legend_handles(color_map, max_posts),
                    loc="lower center", ncol=min(5, 2+max_posts+2),
                    fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
    fig_grid.suptitle("MASE – Finetuned vs Original Model (all datasets)",
                      fontsize=14, fontweight="bold", y=1.01)
    fig_grid.tight_layout(rect=[0, 0.08, 1, 1])
    fig_grid.savefig(os.path.join(out_dir, "eval_GRID.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved → eval_GRID.png")
    plt.close(fig_grid)

    # ── Single combined scatter ───────────────────────────────────────────────
    fig_comb, ax_comb = plt.subplots(figsize=(8, 7))
    fig_comb.patch.set_facecolor("#F7F7F7"); ax_comb.set_facecolor("#FFFFFF")
    plot_combined_single(ax_comb, entries, color_map)
    fig_comb.legend(handles=make_legend_handles(color_map, max_posts),
                    loc="lower center", ncol=min(4, len(color_map)+3),
                    fontsize=8.5, framealpha=0.92, bbox_to_anchor=(0.5, 0.0))
    fig_comb.tight_layout(rect=[0, 0.10, 1, 1])
    fig_comb.savefig(os.path.join(out_dir, "eval_COMBINED.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved → eval_COMBINED.png")
    plt.close(fig_comb)

    # ── Heatmaps per dataset ──────────────────────────────────────────────────
    for entry_data in entries:
        plot_heatmap_for_entry(entry_data, out_dir)

    # ── Aggregated heatmaps ───────────────────────────────────────────────────
    plot_heatmap_aggregated(entries, out_dir)


if __name__ == "__main__":
    main()