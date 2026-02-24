from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from matplotlib import pyplot as plt
from finetuning_scripts.constant_utils import SupportedDevice, TaskType
from sympy import floor

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from finetuning_scripts.metric_utils.ag_metrics import Scorer
    from tabpfn.model.transformer import PerFeatureTransformer

def create_val_data(
    *,
    X_train: list[pd.DataFrame | np.ndarray],
    y_train: list[pd.Series | np.ndarray],
    rng: np.random.RandomState,
    dataset_attributes: list[DatasetAttributes],
    val_time_series: Int
) -> tuple[
    list[pd.DataFrame | np.ndarray],
    list[pd.DataFrame | np.ndarray],
    list[pd.Series | np.ndarray],
    list[pd.Series | np.ndarray],
]:
    # Split data ourselves
    if val_time_series is None:
        raise ValueError("val_time_series must be provided for validation.")

    val_amount_per_ds = floor(val_time_series / len(X_train))
    X_val_series_list = []
    y_val_series_list = []

    for i in range(len(X_train)):
        n_time_series = dataset_attributes[i].ts_amount
        # create true/false mask array with ntime_series * test_size true values
        mask = np.zeros(n_time_series, dtype=bool)
        print("time_series", n_time_series)
        mask_indices = rng.choice(n_time_series, int(val_amount_per_ds), replace=False).tolist()
        print("Val indices", mask_indices)
        mask[mask_indices] = True
        print("Xtrain_shape", X_train.shape)
        X_t = X_train[i]
        y_t = y_train[i]
        X_train[i] = X_train[i][:, :, ~mask]
        y_train[i] = y_train[i][:, :, ~mask]
        X_val = X_t[:, :, mask]
        y_val = y_t[:, :, mask]
        X_val_series_list.append(X_val)
        y_val_series_list.append(y_val)

    return X_train, X_val_series_list, y_train, y_val_series_list


# TODO validate over windows and average results
def validate_tabpfn_fixed_context(
    *,
    X_val_list: list[torch.Tensor],  # list[(n_samples, feature_count, time_series_length)]
    y_val_list: list[torch.Tensor],  # list[(n_samples, 1, time_series_length)]
    dataset_attributes_list: list[DatasetAttributes],
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

    scores_per_ds = []

    for i in range(len(X_val_list)):
        X_val = X_val_list[i].to(device)
        y_val = y_val_list[i].to(device)
        n_samples = X_val.shape[2]
        forecast_horizon = dataset_attributes_list[i].forecast_horizon
        windows = dataset_attributes_list[i].windows

        all_window_scores = []
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

                    plt.savefig(f"{plot_save_path}/validation_pred_{dataset_attributes_list[i].name.replace('/', '_')}_iter_{iteration}_{ts_idx}_windowidx_{index}.png")
                    plt.clf()

                score = validation_metric(y_true=y_true, y_pred=y_pred)
                window_scores.append(score)

            # Mittelwert über Windows der Serie
            all_window_scores.append(sum(window_scores) / len(window_scores))

        # Mittelwert über alle Serien
        avg_score = sum(all_window_scores) / len(all_window_scores)
        scores_per_ds.append(avg_score)

        X_val = X_val.cpu()
        y_val = y_val.cpu()
        torch.cuda.empty_cache()



    return validation_metric.convert_score_to_error(np.mean(scores_per_ds)) #TODO


