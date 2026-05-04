"""
plot_forecasts.py
=================
Paper-conform forecast plotting for TabPFNTS predictions across multiple datasets.

Usage
-----
# Single dataset:
    python plot_forecasts.py monash_covid_deaths

# Multiple datasets:
    python plot_forecasts.py monash_covid_deaths electricity_hourly m4_monthly

# With optional flags:
    python plot_forecasts.py m4_monthly --series 4 --context 80 --out figures/my_plot.pdf
    python plot_forecasts.py m4_monthly --no-png
    python plot_forecasts.py m4_monthly --no-forecast

All datasets are evaluated with term=short (fixed).
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tabpfn_time_series import DEFAULT_QUANTILE_CONFIG
from finetune_tabpfn_ts.evaluation.data import Dataset, Term
from finetune_tabpfn_ts.evaluation.tabpfn_ts_wrapper import TabPFNTSPredictor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FIXED DEFAULTS  (override via CLI flags where applicable)
# ─────────────────────────────────────────────────────────────────────────────

TERM = "short"           # always short — not exposed as a CLI flag

DEFAULT_SERIES_PER_DS = 1    # --series
DEFAULT_CONTEXT       = 4096   # --context
DEFAULT_OUTPUT        = Path("./figures/forecasts_overview.pdf")   # --out
DPI                   = 300

# Quantile bands: (lower_q_str, upper_q_str, fill_alpha)
QUANTILE_BANDS = [
    ("0.1", "0.9", 0.15),
    ("0.2", "0.8", 0.35),
    ("0.3", "0.7", 0.60),
]
MEDIAN_Q = "0.5"

# ─────────────────────────────────────────────────────────────────────────────
# PAPER STYLE
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_COLOR      = "#2d2d2d"
ACCENT_COLORS      = ["#c0392b", "#2471a3", "#1e8449", "#7d3c98", "#d35400", "#117a65"]
VLINE_COLOR        = "#888888"
GROUND_TRUTH_COLOR = "#444444"

RC = {
    # Font
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6.5,
    "figure.titlesize":  10,
    # Lines
    "lines.linewidth":   1.2,
    "axes.linewidth":    0.6,
    "grid.linewidth":    0.4,
    # Grid
    "axes.grid":         True,
    "grid.color":        "#cccccc",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "axes.axisbelow":    True,
    # Spines
    "axes.spines.top":   False,
    "axes.spines.right": False,
    # Ticks
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # Layout
    "figure.constrained_layout.use": False,
    "savefig.bbox":      "tight",
    "savefig.dpi":       DPI,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_quantile_index(q_str: str) -> int:
    """Return the column index for a quantile string like '0.5'."""
    q_keys = list(map(str, DEFAULT_QUANTILE_CONFIG))
    return q_keys.index(q_str)


def _blend_with_white(hex_color: str, alpha: float) -> tuple:
    """
    Alpha-composite hex_color over white at the given alpha.

    Returns an opaque RGB tuple so that legend Patch colours visually match
    the fill_between bands drawn in the axes (which also blend over white).
    """
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - alpha) + alpha * r, (1 - alpha) + alpha * g, (1 - alpha) + alpha * b


def run_prediction(dataset_name: str, n_series: int, context_to_show: int,
                   run_forecast: bool = True):
    """
    Load a dataset (term=short, fixed), run TabPFNTS prediction on the first
    n_series test windows, and return collected arrays + forecast objects.

    Parameters
    ----------
    run_forecast : bool
        If False, skip model inference entirely and return None placeholders
        for each forecast. Use together with --no-forecast to plot only the
        historical time series.

    Returns
    -------
    histories     : list of 1-D np.ndarray  (historical target values shown)
    ground_truths : list of 1-D np.ndarray  (actual future values, if available)
    forecasts     : list of QuantileForecast or list of None
    freq          : str
    pred_len      : int
    """
    ds = Dataset(name=dataset_name, term=TERM)
    predictor = TabPFNTSPredictor(
        ds_prediction_length=ds.prediction_length,
        ds_freq=ds.freq,
    )

    histories, ground_truths = [], []

    for i, (input_entry, label_entry) in enumerate(ds.test_data):
        if i >= n_series:
            break

        # Collect history — for multivariate data take only the first dimension
        target = np.asarray(input_entry["target"], dtype=float)
        if target.ndim > 1:
            target = target[0]
        histories.append(target[-context_to_show:] if len(target) > context_to_show else target)

        # Collect ground truth — same first-dimension rule
        if label_entry is not None:
            gt = np.asarray(label_entry["target"], dtype=float)
            if gt.ndim > 1:
                gt = gt[0]
            ground_truths.append(gt)
        else:
            ground_truths.append(None)

    if run_forecast:
        # Run predictor on the first n_series inputs
        test_inputs = [inp for inp, _ in list(ds.test_data)[:n_series]]
        forecast_list = predictor.predict(test_inputs)
        forecasts = forecast_list[:n_series]
    else:
        forecasts = [None] * n_series

    return histories, ground_truths, forecasts, ds.freq, ds.prediction_length


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_series(
    ax: plt.Axes,
    history: np.ndarray,
    ground_truth: Optional[np.ndarray],
    forecast,           # QuantileForecast or None
    accent_color: str,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    ds_name: str = "",
):
    """Render one time series panel. If forecast is None, only the history is drawn."""
    n_hist = len(history)
    hist_x = np.arange(n_hist)

    # ── history ──────────────────────────────────────────────────────────────
    ax.plot(hist_x, history, color=HISTORY_COLOR, lw=1.1, zorder=3)

    if forecast is not None:
        pred_len = forecast.forecast_array.shape[1]
        fore_x = np.arange(n_hist, n_hist + pred_len)

        # vertical divider
        ax.axvline(n_hist - 0.5, color=VLINE_COLOR, lw=0.8, ls="--", zorder=2)

        # ── quantile bands ────────────────────────────────────────────────────
        for z, (lo_q, hi_q, alpha) in enumerate(QUANTILE_BANDS):
            lo = forecast.forecast_array[get_quantile_index(lo_q)]
            hi = forecast.forecast_array[get_quantile_index(hi_q)]
            blended = _blend_with_white(accent_color, alpha)
            ax.fill_between(fore_x, lo, hi, color=blended, zorder=1 + z)

        # ── median ────────────────────────────────────────────────────────────
        median = forecast.forecast_array[get_quantile_index(MEDIAN_Q)]
        ax.plot(fore_x, median, color=accent_color, lw=1.4, zorder=4)

        # ── ground truth ──────────────────────────────────────────────────────
        if ground_truth is not None:
            gt_x = fore_x[: len(ground_truth)]
            ax.plot(gt_x, ground_truth[: pred_len], color=GROUND_TRUTH_COLOR,
                    lw=1.0, ls=":", zorder=5)

    # ── cosmetics ─────────────────────────────────────────────────────────────
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis="both", which="both", length=3)

    if show_xlabel:
        ax.set_xlabel("Timestamp")
    if show_ylabel:
        ax.set_ylabel("Value")


def build_figure(
    all_histories,
    all_ground_truths,
    all_forecasts,
    dataset_names,
    accent_colors,
    n_series: int,
):
    """
    Build the full multi-dataset figure.

    Layout: rows = datasets, cols = series_per_dataset
    The legend sits below all subplots and uses the real accent colours so
    that CI patch colours match the shaded bands in the plot.
    """
    n_datasets = len(dataset_names)
    fig_w = 2.8 * n_series
    fig_h = 2.0 * n_datasets + 0.5  # +0.5 inch reserved for the figure legend

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_datasets, n_series,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        bottom=0.28,   # leave space below subplots for the legend
    )

    has_ground_truth = False
    has_forecast = False

    for row, (ds_name, histories, gts, forecasts, color) in enumerate(
        zip(dataset_names, all_histories, all_ground_truths, all_forecasts, accent_colors)
    ):
        for col in range(n_series):
            ax = fig.add_subplot(gs[row, col])

            show_xlabel = (row == n_datasets - 1)
            show_ylabel = (col == 0)

            if any(gt is not None for gt in gts):
                has_ground_truth = True

            if any(fc is not None for fc in forecasts):
                has_forecast = True

            plot_single_series(
                ax=ax,
                history=histories[col],
                ground_truth=gts[col],
                forecast=forecasts[col],
                accent_color=color,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                ds_name=ds_name,
            )

            # Column header on first row
            if row == 0:
                ax.set_title(f"Series {col + 1}", pad=4)

    # ── shared figure-level legend below all subplots ────────────────────────
    representative_color = accent_colors[0]

    legend_handles = [
        Line2D([0], [0], color=HISTORY_COLOR, lw=1.1, label="History"),
    ]

    if has_forecast:
        if n_datasets == 1:
            legend_handles += [
                Line2D([0], [0], color=representative_color, lw=1.4, label="Median forecast"),
            ]
        else:
            for ds_name, color in zip(dataset_names, accent_colors):
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=1.4, label=f"Median ({ds_name})")
                )

        legend_handles += [
            Patch(
                facecolor=_blend_with_white(representative_color, QUANTILE_BANDS[0][2]),
                label="80% CI",
            ),
            Patch(
                facecolor=_blend_with_white(representative_color, QUANTILE_BANDS[1][2]),
                label="60% CI",
            ),
            Patch(
                facecolor=_blend_with_white(representative_color, QUANTILE_BANDS[2][2]),
                label="40% CI",
            ),
        ]

    if has_ground_truth and has_forecast:
        legend_handles.append(
            Line2D([0], [0], color=GROUND_TRUTH_COLOR, lw=1.0, ls=":", label="Ground truth")
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=7,
        handlelength=1.8,
        handleheight=0.9,
        columnspacing=1.2,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI + MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="plot_forecasts.py",
        description=(
            "Plot paper-conform TabPFNTS short-term forecasts for one or more datasets.\n"
            "\n"
            "Examples:\n"
            "  python plot_forecasts.py monash_covid_deaths\n"
            "  python plot_forecasts.py monash_covid_deaths electricity_hourly m4_monthly\n"
            "  python plot_forecasts.py m4_monthly --series 4 --context 80\n"
            "  python plot_forecasts.py m4_monthly --no-forecast\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="DATASET",
        help="One or more dataset names (e.g. monash_covid_deaths m4_monthly).",
    )
    parser.add_argument(
        "--series", "-s",
        type=int,
        default=DEFAULT_SERIES_PER_DS,
        metavar="N",
        help=f"Number of time series to plot per dataset (default: {DEFAULT_SERIES_PER_DS}).",
    )
    parser.add_argument(
        "--context", "-c",
        type=int,
        default=DEFAULT_CONTEXT,
        metavar="STEPS",
        help=f"Historical steps shown left of the forecast window (default: {DEFAULT_CONTEXT}).",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Output PDF path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip saving a PNG preview alongside the PDF.",
    )
    parser.add_argument(
        "--no-forecast",
        action="store_true",
        help="Plot only the historical time series, skip prediction and forecast overlay.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    matplotlib.rcParams.update(RC)

    output_file: Path = args.out
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_histories, all_gts, all_forecasts, dataset_names = [], [], [], []

    for ds_name in args.datasets:
        if args.no_forecast:
            logger.info(f"Loading data for '{ds_name}'  [term=short, no forecast] …")
        else:
            logger.info(f"Running predictions for '{ds_name}'  [term=short] …")

        histories, gts, forecasts, freq, pred_len = run_prediction(
            ds_name, args.series, args.context,
            run_forecast=not args.no_forecast,
        )
        logger.info(f"  freq={freq}  pred_length={pred_len}  n_series={len(histories)}")

        all_histories.append(histories)
        all_gts.append(gts)
        all_forecasts.append(forecasts)
        dataset_names.append(ds_name)

    accent_colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(args.datasets))]

    fig = build_figure(
        all_histories=all_histories,
        all_ground_truths=all_gts,
        all_forecasts=all_forecasts,
        dataset_names=dataset_names,
        accent_colors=accent_colors,
        n_series=args.series,
    )

    fig.savefig(output_file, format="pdf")
    logger.info(f"Figure saved → {output_file}")

    if not args.no_png:
        png_path = output_file.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=DPI)
        logger.info(f"PNG preview  → {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()