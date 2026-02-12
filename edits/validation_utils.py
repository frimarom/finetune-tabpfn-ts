from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from matplotlib import pyplot as plt
from finetuning_scripts.constant_utils import SupportedDevice, TaskType
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from finetuning_scripts.metric_utils.ag_metrics import Scorer
    from tabpfn.model.transformer import PerFeatureTransformer
    from tabpfn import TabPFNClassifier, TabPFNRegressor

# Split training data into training and validation sets
# Only take full time series into account for splitting
# Use multiple time series for validation and create mean over all time series
def create_val_data(
    *,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    rng: np.random.RandomState,
    n_time_series: int,
    windows_per_series: int,
    val_time_series: Int = None
) -> tuple[
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.Series | np.ndarray,
    pd.Series | np.ndarray,
]:
    # Split data ourselves
    if val_time_series is None:
        dataset_window_amount = n_time_series * windows_per_series
        if dataset_window_amount < 1000:
            test_size = 0.33
        elif dataset_window_amount < 5000:
            test_size = 0.2
        elif dataset_window_amount < 10000:
            test_size = 0.1
        else:
            test_size = 0.05
        val_time_series = n_time_series * test_size

    random.setstate = rng.get_state()

    # create true/false mask array with ntime_series * test_size true values
    mask = np.zeros(n_time_series, dtype=bool)
    mask_indices = random.sample(range(n_time_series), int(val_time_series))
    print("Val indices", mask_indices)
    mask[mask_indices] = True

    X_t = X_train
    y_t = y_train
    X_train = X_train[:, :, ~mask]
    y_train = y_train[:, :, ~mask] # TODO maybe error here because shape maybe ony two dimensional
    X_val = X_t[:, :, mask]
    y_val = y_t[:, :, mask]

    return X_train, X_val, y_train, y_val


# TODO validate over windows and average results
def validate_tabpfn_fixed_context(
    *,
    X_val: torch.Tensor,  # (n_samples, feature_count, time_series_length)
    y_val: torch.Tensor,  # (n_samples, 1, time_series_length)
    dataset_attributes: DatasetAttributes,
    validation_metric: Scorer,
    model: PerFeatureTransformer,
    model_forward_fn: Callable,
    task_type: TaskType,
    plotting: bool = False,
    plot_save_path: str = "./validation_plots",
    iteration: int = 0,
    device: str,
) -> float:
    """
    Validate the TabPFN model on multiple time series and multiple windows
    with a fixed context length (History length per Window).
    """
    n_samples = X_val.shape[2]
    forecast_horizon = dataset_attributes.forecast_horizon
    windows = dataset_attributes.windows

    all_window_scores = []

    X_val = X_val.to(device)
    y_val = y_val.to(device)

    for ts_idx in range(n_samples):
        x_series = X_val[:, :, ts_idx]  # keep batch dim
        y_series = y_val[:, :, ts_idx]

        x_series = x_series

        series_length = x_series.shape[0]
        context_length = (series_length - forecast_horizon) // windows

        # Berechne Origins vom Ende der Serie aus
        origins = [
            series_length - forecast_horizon - w * context_length
            for w in reversed(range(windows))
        ]
        print("origins", origins)
        print("context_length", context_length)
        window_scores = []

        for index, origin in enumerate(origins):
            origin = max(0, origin)

            # Slice History bis zum origin
            start_idx = max(0, origin - context_length)
            X_window_train = x_series[start_idx:origin,...].clone().unsqueeze(1)
            y_window_train = y_series[start_idx:origin,...].clone().unsqueeze(1)

            # Slice Test
            X_window_test = x_series[origin:origin + forecast_horizon, ...].clone().unsqueeze(1)
            y_window_true = y_series[origin:origin + forecast_horizon, ...].clone().unsqueeze(1)

            # Vorhersage
            pred_logits = model_forward_fn(
                model=model,
                X_train=X_window_train,
                y_train=y_window_train,
                X_test=X_window_test,
                forward_for_validation=True,
            )

            if task_type == TaskType.REGRESSION:
                y_pred = pred_logits.float().flatten().cpu().detach().numpy()
                y_true = y_window_true.float().flatten().cpu().detach().numpy()
                print("y_pred Validation", y_pred)
                print("y_true Validation", y_true)
            else:
                raise ValueError(f"Task type {task_type} not supported.")

            if plotting:
                x_train_plot = X_window_train[:, 0, 0].detach().cpu().numpy()
                y_train_plot = y_window_train[:, 0, 0].detach().cpu().numpy()

                x_test_plot = X_window_test[:, 0, 0].detach().cpu().numpy()
                y_test_plot = y_window_true[:, 0, 0].detach().cpu().numpy()

                plt.plot(x_train_plot, y_train_plot, color="blue")
                plt.plot(x_test_plot, y_test_plot, color="green")
                plt.plot(x_test_plot, y_pred, color="red")

                plt.savefig(f"{plot_save_path}/validation_pred_{dataset_attributes.name.replace('/', '_')}_iter_{iteration}_{ts_idx}_windowidx_{index}.png")
                plt.clf()

            score = validation_metric(y_true=y_true, y_pred=y_pred)
            window_scores.append(score)

        # Mittelwert über Windows der Serie
        all_window_scores.append(sum(window_scores) / len(window_scores))

    # Mittelwert über alle Serien
    avg_score = sum(all_window_scores) / len(all_window_scores)

    X_val.cpu()
    y_val.cpu()

    return validation_metric.convert_score_to_error(avg_score)


