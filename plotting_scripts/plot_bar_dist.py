"""
Plot the Bar Distribution (logits) from a TabPFN model forward pass.
Usage:
    python plot_bar_distribution.py --checkpoint path/to/model.ckpt --dataset covid_deaths
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import partial
from pathlib import Path

from tabpfn.base import load_model_criterion_config
from finetune_tabpfn_ts.edits.finetune_tabpfn_main import _model_forward
from finetune_tabpfn_ts.task_1.dataset_utils import create_train_val_split, DatasetAttributes


def plot_bar_distribution(
    logits: torch.Tensor,        # (pred_len, batch_size, n_bins)
    borders: torch.Tensor,       # (n_bins + 1,)
    mean: torch.Tensor,          # (1, batch_size, 1)  – train mean used for denorm
    std: torch.Tensor,           # (1, batch_size, 1)  – train std  used for denorm
    y_train: torch.Tensor,       # (ctx_len, batch_size, 1)
    y_test: torch.Tensor,        # (pred_len, batch_size, 1)
    save_path: str = "bar_distribution.png",
    max_series: int = 4,
):
    """
    For each of the first `max_series` batch entries, plot:
      - the bar distribution (softmax over bins) in normalized space
      - the predicted mean (denormalized) vs. ground truth
    """
    n_pred, batch_size, n_bins = logits.shape
    n_show = min(batch_size, max_series)

    # bin centers in normalized space
    borders_np = borders.cpu().float().numpy()
    bin_centers = (borders_np[:-1] + borders_np[1:]) / 2.0   # (n_bins,)
    bin_widths  = borders_np[1:] - borders_np[:-1]

    fig, axes = plt.subplots(
        n_show, 2,
        figsize=(14, 4 * n_show),
        facecolor="#0f0f14",
    )
    if n_show == 1:
        axes = axes[np.newaxis, :]

    colors = cm.plasma(np.linspace(0.2, 0.9, n_pred))

    for b in range(n_show):
        ax_dist = axes[b, 0]
        ax_pred = axes[b, 1]

        m = mean[0, b, 0].item()
        s = std[0, b, 0].item()

        # ── left: bar distribution for last predicted timestep ──────────────
        last_logits = logits[-1, b, :].cpu().float().numpy()   # (n_bins,)
        probs = np.exp(last_logits - last_logits.max())
        probs /= (probs * bin_widths).sum()                    # density

        ax_dist.bar(
            bin_centers, probs, width=bin_widths * 0.9,
            color="#7c3aed", alpha=0.85, edgecolor="none",
        )
        # mark predicted mean (normalized)
        pred_mean_norm = (probs * bin_centers * bin_widths).sum()
        ax_dist.axvline(pred_mean_norm, color="#f59e0b", lw=1.5, ls="--", label=f"pred mean (norm) = {pred_mean_norm:.2f}")
        # mark true value (normalized)
        true_norm = (y_test[-1, b, 0].item() - m) / s
        ax_dist.axvline(true_norm, color="#10b981", lw=1.5, ls="--", label=f"true (norm) = {true_norm:.2f}")

        ax_dist.set_facecolor("#1a1a24")
        ax_dist.tick_params(colors="white")
        ax_dist.spines[:].set_color("#333")
        ax_dist.set_xlabel("normalized value", color="white", fontsize=9)
        ax_dist.set_ylabel("density", color="white", fontsize=9)
        ax_dist.set_title(f"Bar Distribution  –  series {b}  (last pred step)", color="white", fontsize=10)
        ax_dist.legend(fontsize=8, labelcolor="white", facecolor="#1a1a24", edgecolor="#444")

        # ── right: predicted mean trajectory vs ground truth ────────────────
        ctx_len  = y_train.shape[0]
        pred_len = y_test.shape[0]

        y_train_np = y_train[:, b, 0].cpu().numpy()
        y_test_np  = y_test[:,  b, 0].cpu().numpy()

        # predicted mean per timestep (denormalized)
        pred_means = []
        for t in range(pred_len):
            lg = logits[t, b, :].cpu().float().numpy()
            pr = np.exp(lg - lg.max())
            pr /= (pr * bin_widths).sum()
            pred_mean_norm_t = (pr * bin_centers * bin_widths).sum()
            pred_means.append(pred_mean_norm_t * s + m)

        time_ctx  = np.arange(ctx_len)
        time_pred = np.arange(ctx_len, ctx_len + pred_len)

        ax_pred.plot(time_ctx,  y_train_np, color="#60a5fa", lw=1.5, label="context (train)")
        ax_pred.plot(time_pred, y_test_np,  color="#10b981", lw=1.5, label="ground truth")
        ax_pred.plot(time_pred, pred_means, color="#f59e0b", lw=1.5, ls="--", label="predicted mean")

        ax_pred.set_facecolor("#1a1a24")
        ax_pred.tick_params(colors="white")
        ax_pred.spines[:].set_color("#333")
        ax_pred.set_xlabel("timestep", color="white", fontsize=9)
        ax_pred.set_ylabel("value", color="white", fontsize=9)
        ax_pred.set_title(f"Prediction vs. Ground Truth  –  series {b}", color="white", fontsize=10)
        ax_pred.legend(fontsize=8, labelcolor="white", facecolor="#1a1a24", edgecolor="#444")

    fig.suptitle("TabPFN Bar Distribution Inspection", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, default="auto",        help="Path to .ckpt or 'auto'")
    parser.add_argument("--dataset",      type=str, default="covid_deaths", help="Dataset name")
    parser.add_argument("--ts_amount",    type=int, default=8,             help="Number of training time series to load")
    parser.add_argument("--max_context",  type=int, default=512,           help="Max context length")
    parser.add_argument("--batch_idx",    type=int, default=0,             help="Which batch element to use (0-indexed)")
    parser.add_argument("--max_series",   type=int, default=4,             help="How many series to plot")
    parser.add_argument("--device",       type=str, default="cpu",         help="cpu or cuda")
    parser.add_argument("--save",         type=str, default="bar_distribution.png")
    args = parser.parse_args()

    device = args.device

    # ── load model ───────────────────────────────────────────────────────────
    print("Loading model …")
    model_path = None if args.checkpoint == "auto" else Path(args.checkpoint)
    models, criterion, checkpoint_configs, _ = load_model_criterion_config(
        model_path=model_path,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="regressor",
        version="v2",
        download_if_not_exists=True,
    )
    model = models[0]
    model.criterion = criterion
    model.to(device)
    model.eval()

    borders = criterion.borders.cpu()   # (n_bins+1,)

    # ── load data ────────────────────────────────────────────────────────────
    print(f"Loading dataset '{args.dataset}' …")
    X_train_list, y_train_list, X_val_list, y_val_list, ts_length, pred_length = create_train_val_split(
        args.dataset,
        max_training_ts_amount=args.ts_amount,
        max_context_length=args.max_context,
        max_validation_ts_amount=2,
    )
    # X_train_list[0]: (time, features, n_ts)
    X_t = X_train_list[0]   # (T, F, N)
    y_t = y_train_list[0]   # (T, 1, N)

    forecast_horizon = pred_length
    context_length   = X_t.shape[0] - forecast_horizon

    # pick first `max_series` time series as a batch
    n_show = min(args.max_series, X_t.shape[2])
    ts_indices = list(range(n_show))

    # shape needed by _model_forward: (seq_len, batch_size, features)
    X_ctx  = X_t[:context_length,   :, ts_indices].permute(0, 2, 1)   # (ctx, n_show, F)
    X_tgt  = X_t[context_length:,   :, ts_indices].permute(0, 2, 1)   # (pred, n_show, F)
    y_ctx  = y_t[:context_length,   :, ts_indices].permute(0, 2, 1)   # (ctx, n_show, 1)
    y_tgt  = y_t[context_length:,   :, ts_indices].permute(0, 2, 1)   # (pred, n_show, 1)

    X_ctx  = X_ctx.to(device).float()
    X_tgt  = X_tgt.to(device).float()
    y_ctx  = y_ctx.to(device).float()

    # compute mean/std for denormalization (same as in _model_forward)
    mean = y_ctx.mean(dim=0, keepdim=True)
    std  = y_ctx.std(dim=0,  keepdim=True)
    std  = torch.where(std < 0.01, torch.ones_like(std), std)

    # ── forward pass ─────────────────────────────────────────────────────────
    print("Running forward pass …")
    model_forward_fn = partial(
        _model_forward,
        n_classes=None,
        categorical_features_index=None,
        use_autocast=False,
        device=device,
        is_data_parallel=False,
        forward_for_validation=False,
    )

    with torch.no_grad():
        logits = model_forward_fn(
            model=model,
            X_train=X_ctx,
            y_train=y_ctx,
            X_test=X_tgt,
        )
    # logits: (pred_len, n_show, n_bins)
    print(f"logits shape: {logits.shape}")

    # ── plot ─────────────────────────────────────────────────────────────────
    plot_bar_distribution(
        logits=logits,
        borders=borders,
        mean=mean.cpu(),
        std=std.cpu(),
        y_train=y_ctx.cpu(),
        y_test=y_tgt.cpu(),
        save_path=args.save,
        max_series=n_show,
    )


if __name__ == "__main__":
    main()