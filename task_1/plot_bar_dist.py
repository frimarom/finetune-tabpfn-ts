"""
Publication-quality Bar Distribution plot for TabPFN.

Shows a single time series with:
  - Left:  Bar Distribution (density) over normalized bins for a chosen timestep
  - Right: Context + ground truth + predicted mean with 80%/95% credible intervals

Usage:
    python plot_bar_dist.py \
        --checkpoint path/to/model.ckpt \
        --dataset covid_deaths \
        --series_idx 0 \
        --pred_step -1 \
        --save figure.pdf
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from functools import partial
from pathlib import Path

# ── publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
})

BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#4dac26"
GRAY   = "#888888"


def compute_dist(logits_1d: np.ndarray, bin_widths: np.ndarray):
    """Softmax → density."""
    lg = logits_1d - logits_1d.max()
    pr = np.exp(lg)
    pr /= (pr * bin_widths).sum()
    return pr


def credible_interval(probs: np.ndarray, bin_centers: np.ndarray,
                      bin_widths: np.ndarray, level: float):
    """Return (lo, hi) of a symmetric credible interval at `level` (e.g. 0.80)."""
    cdf = np.cumsum(probs * bin_widths)
    alpha = (1 - level) / 2
    lo = bin_centers[np.searchsorted(cdf, alpha)]
    hi = bin_centers[np.searchsorted(cdf, 1 - alpha)]
    return lo, hi


def plot_publication(
    logits:     torch.Tensor,   # (pred_len, batch_size, n_bins)
    borders:    torch.Tensor,   # (n_bins + 1,)
    mean:       torch.Tensor,   # (1, batch_size, 1)
    std:        torch.Tensor,   # (1, batch_size, 1)
    y_train:    torch.Tensor,   # (ctx_len, batch_size, 1)
    y_test:     torch.Tensor,   # (pred_len, batch_size, 1)
    series_idx: int  = 0,
    pred_step:  int  = -1,       # which prediction timestep to show distribution for
    dataset_name: str = "",
    save_path:  str  = "bar_dist.pdf",
):
    borders_np   = borders.cpu().float().numpy()
    bin_centers  = (borders_np[:-1] + borders_np[1:]) / 2.0
    bin_widths   = borders_np[1:] - borders_np[:-1]

    b   = series_idx
    m   = mean[0, b, 0].item()
    s   = std[0, b, 0].item()

    pred_len = logits.shape[0]
    ctx_len  = y_train.shape[0]
    t        = pred_step % pred_len   # support negative indexing

    # ── denormalized predictions + intervals over all timesteps ─────────────
    pred_means, lo80, hi80, lo95, hi95 = [], [], [], [], []
    for ti in range(pred_len):
        pr = compute_dist(logits[ti, b, :].cpu().float().numpy(), bin_widths)
        mu_norm = (pr * bin_centers * bin_widths).sum()
        pred_means.append(mu_norm * s + m)

        l80, h80 = credible_interval(pr, bin_centers, bin_widths, 0.80)
        l95, h95 = credible_interval(pr, bin_centers, bin_widths, 0.95)
        lo80.append(l80 * s + m);  hi80.append(h80 * s + m)
        lo95.append(l95 * s + m);  hi95.append(h95 * s + m)

    pred_means = np.array(pred_means)
    lo80, hi80 = np.array(lo80), np.array(hi80)
    lo95, hi95 = np.array(lo95), np.array(hi95)

    # clip CI to sensible range based on actual data
    y_train_np_tmp = y_train[:, b, 0].cpu().numpy()
    y_test_np_tmp  = y_test[:,  b, 0].cpu().numpy()
    y_all_tmp      = np.concatenate([y_train_np_tmp, y_test_np_tmp, pred_means])
    y_margin       = (y_all_tmp.max() - y_all_tmp.min()) * 0.2
    y_min_clip     = y_all_tmp.min() - y_margin * 3
    y_max_clip     = y_all_tmp.max() + y_margin * 3
    lo95 = np.clip(lo95, y_min_clip, y_max_clip)
    hi95 = np.clip(hi95, y_min_clip, y_max_clip)
    lo80 = np.clip(lo80, y_min_clip, y_max_clip)
    hi80 = np.clip(hi80, y_min_clip, y_max_clip)

    # ── distribution at chosen timestep ─────────────────────────────────────
    probs_t    = compute_dist(logits[t, b, :].cpu().float().numpy(), bin_widths)
    true_norm  = (y_test[t, b, 0].item() - m) / s
    mu_norm_t  = (probs_t * bin_centers * bin_widths).sum()

    y_train_np = y_train[:, b, 0].cpu().numpy()
    y_test_np  = y_test[:,  b, 0].cpu().numpy()
    time_ctx   = np.arange(ctx_len)
    time_pred  = np.arange(ctx_len, ctx_len + pred_len)

    # ── layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 3.8), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 2, width_ratios=[1, 2], wspace=0.32,
        left=0.07, right=0.97, top=0.88, bottom=0.15,
    )
    ax_dist = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1])

    # ════════════════════════════════════════════════════════════════════════
    # LEFT – Bar Distribution
    # ════════════════════════════════════════════════════════════════════════
    # clip overflow bins for display so interior is visible
    interior = (bin_centers > borders_np[1]) & (bin_centers < borders_np[-2])
    overflow = ~interior

    ax_dist.bar(bin_centers[interior], probs_t[interior],
                width=bin_widths[interior] * 0.92,
                color=BLUE, alpha=0.75, linewidth=0, label="_nolegend_")

    # overflow bins in lighter color with hatching
    if overflow.any():
        ax_dist.bar(bin_centers[overflow], probs_t[overflow],
                    width=bin_widths[overflow] * 0.92,
                    color=BLUE, alpha=0.30, linewidth=0.5,
                    edgecolor=BLUE, hatch="///", label="overflow bins")

    # vertical lines
    ax_dist.axvline(mu_norm_t,  color=ORANGE, lw=1.5, ls="--",
                    label=f"predicted mean ({mu_norm_t:.2f})")
    ax_dist.axvline(true_norm,  color=GREEN,  lw=1.5, ls="-.",
                    label=f"ground truth ({true_norm:.2f})")

    ax_dist.set_xlabel("Normalized value")
    ax_dist.set_ylabel("Density")
    step_label = f"t+{t+1}" if t >= 0 else f"t{t}"
    ax_dist.set_title(f"Predictive Distribution at {step_label}")
    ax_dist.legend(framealpha=0.9, edgecolor="#cccccc")
    ax_dist.set_facecolor("white")

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT – Forecast
    # ════════════════════════════════════════════════════════════════════════
    ax_pred.plot(time_ctx,  y_train_np, color=GRAY,  lw=1.2,
                 label="Context", zorder=3)
    ax_pred.plot(time_pred, y_test_np,  color=GREEN, lw=1.5,
                 label="Ground truth", zorder=4)
    ax_pred.plot(time_pred, pred_means, color=ORANGE, lw=1.5, ls="--",
                 label="Predicted mean", zorder=5)

    # credible intervals
    ax_pred.fill_between(time_pred, lo95, hi95,
                         color=ORANGE, alpha=0.12, label="95% CI")
    ax_pred.fill_between(time_pred, lo80, hi80,
                         color=ORANGE, alpha=0.22, label="80% CI")

    # mark the timestep shown in the dist plot
    marker_x = time_pred[t]
    ax_pred.axvline(marker_x, color="#999999", lw=0.8, ls=":")
    ax_pred.scatter([marker_x], [pred_means[t]],
                    color=ORANGE, s=30, zorder=6)

    # vertical separator between context and forecast
    ax_pred.axvline(ctx_len - 0.5, color="#aaaaaa", lw=0.8, ls="-")
    ax_pred.text(ctx_len - 0.5 + 0.3,
                 ax_pred.get_ylim()[0] if ax_pred.get_ylim()[0] != 0 else y_train_np.min(),
                 "forecast →", fontsize=7.5, color="#666666", va="bottom")

    # fix Y-axis to data range so CI shading doesn't blow up the scale
    y_all    = np.concatenate([y_train_np, y_test_np, pred_means])
    y_margin = (y_all.max() - y_all.min()) * 0.2
    ax_pred.set_ylim(y_all.min() - y_margin, y_all.max() + y_margin)

    ax_pred.set_xlabel("Timestep")
    ax_pred.set_ylabel("Value")
    ds_str = f" — {dataset_name}" if dataset_name else ""
    ax_pred.set_title(f"TabPFN Forecast{ds_str}")
    ax_pred.legend(framealpha=0.9, edgecolor="#cccccc", ncol=2)
    ax_pred.set_facecolor("white")

    # ── figure title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "TabPFN Predictive Bar Distribution",
        fontsize=12, fontweight="bold", y=0.97,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {save_path}")
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str,  default="auto")
    parser.add_argument("--dataset",      type=str,  default="covid_deaths")
    parser.add_argument("--ts_amount",    type=int,  default=8)
    parser.add_argument("--max_context",  type=int,  default=512)
    parser.add_argument("--series_idx",   type=int,  default=0,
                        help="Which time series in the batch to plot (0-indexed)")
    parser.add_argument("--pred_step",    type=int,  default=-1,
                        help="Which prediction timestep to show the distribution for (-1 = last)")
    parser.add_argument("--device",       type=str,  default="cpu")
    parser.add_argument("--save",         type=str,  default="bar_dist.pdf")
    args = parser.parse_args()

    device = args.device

    # ── model ────────────────────────────────────────────────────────────────
    from tabpfn.base import load_model_criterion_config
    from finetune_tabpfn_ts.edits.finetune_tabpfn_main import _model_forward
    from finetune_tabpfn_ts.task_1.dataset_utils import create_train_val_split

    print("Loading model …")
    model_path = None if args.checkpoint == "auto" else Path(args.checkpoint)
    models, criterion, _, _ = load_model_criterion_config(
        model_path=model_path,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="regressor", version="v2", download_if_not_exists=True,
    )
    model = models[0]
    model.criterion = criterion
    model.to(device).eval()
    borders = criterion.borders.cpu()

    # ── data ─────────────────────────────────────────────────────────────────
    print(f"Loading '{args.dataset}' …")
    X_t, y_t, _, _, _, pred_length = create_train_val_split(
        args.dataset,
        max_training_ts_amount=args.ts_amount,
        max_context_length=args.max_context,
        max_validation_ts_amount=2,
    )

    context_length = X_t.shape[0] - pred_length
    n_ts           = X_t.shape[2]
    b              = min(args.series_idx, n_ts - 1)

    X_ctx = X_t[:context_length, :, b].unsqueeze(1).to(device).float()   # (ctx, 1, F)
    X_tgt = X_t[context_length:, :, b].unsqueeze(1).to(device).float()   # (pred, 1, F)
    y_ctx = y_t[:context_length, :, b].unsqueeze(1).to(device).float()   # (ctx, 1, 1)
    y_tgt = y_t[context_length:, :, b].unsqueeze(1).cpu().float()

    mean = y_ctx.mean(dim=0, keepdim=True)
    std  = y_ctx.std(dim=0,  keepdim=True)
    std  = torch.where(std < 0.01, torch.ones_like(std), std)

    # ── forward ──────────────────────────────────────────────────────────────
    print("Forward pass …")
    model_forward_fn = partial(
        _model_forward,
        n_classes=None, categorical_features_index=None,
        use_autocast=False, device=device,
        is_data_parallel=False, forward_for_validation=False,
    )
    with torch.no_grad():
        logits = model_forward_fn(
            model=model, X_train=X_ctx, y_train=y_ctx, X_test=X_tgt,
        )   # (pred_len, 1, n_bins)
    print(f"logits: {logits.shape}")

    # ── plot ─────────────────────────────────────────────────────────────────
    save = args.save
    if not save.endswith((".pdf", ".png", ".svg")):
        save = str(Path(save) / f"bar_dist_{args.dataset}_s{b}.pdf")

    plot_publication(
        logits=logits.cpu(),
        borders=borders,
        mean=mean.cpu(),
        std=std.cpu(),
        y_train=y_ctx.cpu(),
        y_test=y_tgt,
        series_idx=0,
        pred_step=args.pred_step,
        dataset_name=args.dataset,
        save_path=save,
    )


if __name__ == "__main__":
    main()