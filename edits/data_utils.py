from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from finetune_tabpfn_ts.prior.generate_series import generate
from torch.utils.data import DataLoader, Dataset

from finetune_tabpfn_ts.task_1.dataset_utils import transform_data, to_x_y
from tabpfn_time_series import TimeSeriesDataFrame

if TYPE_CHECKING:
    import pandas as pd

RANDOM_SEED = 4213

# 0 minute
# 1 hour
# 2 daily
# 3 weekly
# 4 monthly
# 5 yearly
FREQUENCY_MAP = {
    0: "min",
    1: "H",
    2: "D",
    3: "W",
    4: "MS",
    5: "Y",
}

PRED_LENGTH_MAP = {
    "min": 60,
    "H": 48,
    "D": 30,
    "W": 8,
    "MS": 12,
    "Y": 6,
}

MIN_CONTEXT_BY_FREQ = {
    0: 150,  # minute
    1: 150,  # hour
    2: 150,  # day
    3: 64,   # week
    4: 24,   # month
    5: 12,   # year
}

MAX_CONTEXT_BY_FREQ = {
    0: 4096,
    1: 4096,
    2: 4096,
    3: 1040,
    4: 240,
    5: 100,
}

class TimeSeriesDataset(Dataset):
    """Tabular dataset.

    This class is used to load tabular data.

    Here one sample is equal to one split of the data.

    Arguments:
    ----------
    X_train: torch.Tensor (n_samples, n_features)
        Input features.
    y_train: torch.Tensor (n_samples, 1)
        Target labels.
    max_steps: int
        Maximum number of steps (splits of the data).
    """


    def __init__(
        self,
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        max_steps: int,
        ts_amount_for_ds: int,
        dataset_attributes: DatasetAttributes,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.max_steps = max_steps
        self.dataset_attributes = dataset_attributes
        self._rng = np.random.RandomState(RANDOM_SEED)
        self.time_series_window_count = [np.zeros(X_t.shape[2]) for X_t in X_train]
        self.ts_amount_for_ds = ts_amount_for_ds
        self.ts_left_for_ds = ts_amount_for_ds

        ts_amounts = np.array(
            [max(1, int(attr.ts_amount)) for attr in self.dataset_attributes],
            dtype=float,
        )

        self.dataset_sampling_weights = np.sqrt(ts_amounts)
        self.dataset_sampling_probs = (
                self.dataset_sampling_weights / self.dataset_sampling_weights.sum()
        )
        self.current_ds = self._sample_dataset_id()

    def __len__(self):
        return self.max_steps # amount of batches. Each batch contains batch_size amount of splits returned by get_item.

    def _sample_dataset_id(self) -> int:
        return int(self._rng.choice(len(self.X_train), p=self.dataset_sampling_probs))

        # splits_generator komplett ersetzen:
    def _get_next_split(self):
        current_ds = self.current_ds
        windows = self.dataset_attributes[current_ds].windows
        forecast_horizon = self.dataset_attributes[current_ds].forecast_horizon
        z_len = self.X_train[current_ds].shape[2]
        series_length = self.X_train[current_ds].shape[0]

        if np.all(self.time_series_window_count[current_ds][:z_len] == windows):
            self.time_series_window_count[current_ds] = np.zeros(z_len)

        time_series = self._rng.randint(0, z_len)
        while self.time_series_window_count[current_ds][time_series] >= windows:
            time_series = self._rng.randint(0, z_len)

        window_count = self.time_series_window_count[current_ds][time_series]
        context_length = (series_length - forecast_horizon) // windows

        start_idx = max(0, series_length - forecast_horizon - (windows - window_count) * context_length)
        origin = start_idx + context_length
        end_idx = origin + forecast_horizon

        self.time_series_window_count[current_ds][time_series] += 1

        return int(current_ds), int(time_series), int(start_idx), int(origin), int(end_idx)

    def get_splits(self):
        return self._get_next_split()

    def create_data(self, dataset_id, time_series, start_idx, train_test_bound, end_idx):
        sample_X_train = self.X_train[dataset_id][start_idx:train_test_bound, :, time_series]
        sample_X_test = self.X_train[dataset_id][train_test_bound:end_idx, :, time_series]
        sample_y_train = self.y_train[dataset_id][start_idx:train_test_bound, :, time_series]
        sample_y_test = self.y_train[dataset_id][train_test_bound:end_idx, :, time_series]
        return sample_X_train, sample_X_test, sample_y_train, sample_y_test

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.ts_left_for_ds <= 0:
            self.current_ds = self._sample_dataset_id()
            self.ts_left_for_ds = self.ts_amount_for_ds

        dataset_id, time_series, start_idx, train_test_bound, end_idx = self.get_splits()
        s_X_train, s_X_test, s_y_train, s_y_test = self.create_data(
            dataset_id, time_series, start_idx, train_test_bound, end_idx
        )

        self.ts_left_for_ds -= 1

        return dict(
            X_train=s_X_train,
            X_test=s_X_test,
            y_train=s_y_train,
            y_test=s_y_test,
        )

class ArtificalTimeSeriesDataset(Dataset):
    def __init__(
        self,
        *,
        context_lengths: list[int],
        frequencies: list[int],
        max_steps: int,
        batch_size: int,
    ):
        self.context_lengths = list(context_lengths)
        self.frequencies = list(frequencies)
        self.current_context_length = self.context_lengths[0]
        self.current_frequency = self.frequencies[0]
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.ts_left_for_current_attributes = batch_size
        self._rng = np.random.RandomState(RANDOM_SEED)

    def __len__(self):
        return self.max_steps

    def _sample_attributes(self):
        self.current_frequency = int(self._rng.choice(self.frequencies))

        min_len = MIN_CONTEXT_BY_FREQ[self.current_frequency]
        max_len = MAX_CONTEXT_BY_FREQ[self.current_frequency]

        valid_contexts = [
            c for c in self.context_lengths
            if min_len <= c <= max_len
        ]

        if not valid_contexts:
            raise ValueError(
                f"No valid context lengths for frequency {self.current_frequency}. "
                f"Allowed range: [{min_len}, {max_len}], "
                f"provided context_lengths range: "
                f"[{min(self.context_lengths)}, {max(self.context_lengths)}]"
            )

        self.current_context_length = int(self._rng.choice(valid_contexts))
    def create_data(self):
        if self.current_frequency is None:
            raise ValueError(f"current_frequency is None. frequencies={self.frequencies}")
        if self.current_context_length is None:
            raise ValueError(f"current_context_length is None. context_lengths={self.context_lengths}")

        print("frequency:", self.current_frequency, "context_length:", self.current_context_length)

        cfg, sample = generate(
            self.current_context_length,
            freq_index=self.current_frequency,
            start=None,
            options={},
        )

        dataframe = sample[["series_values", "noise"]].copy()
        dataframe["target"] = dataframe["series_values"] + dataframe["noise"]
        dataframe = dataframe[["target"]]
        dataframe["item_id"] = range(len(dataframe))
        dataframe = dataframe.reset_index()
        dataframe = dataframe.rename(columns={"index": "timestamp"})

        train_part_ts = TimeSeriesDataFrame(dataframe)
        transformed_data = transform_data(train_part_ts)
        X, y = to_x_y("prior", transformed_data)

        X_np = X.values.astype(float)
        y_np = y.values.astype(float).reshape(-1, 1)

        border = len(X) - PRED_LENGTH_MAP[FREQUENCY_MAP[self.current_frequency]]

        sample_X_train = torch.tensor(X_np[:border]).float()
        sample_y_train = torch.tensor(y_np[:border]).float()
        sample_X_test = torch.tensor(X_np[border:]).float()
        sample_y_test = torch.tensor(y_np[border:]).float()

        return sample_X_train, sample_X_test, sample_y_train, sample_y_test

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.ts_left_for_current_attributes <= 0:
            self._sample_attributes()
            self.ts_left_for_current_attributes = self.batch_size

        s_X_train, s_X_test, s_y_train, s_y_test = self.create_data()
        self.ts_left_for_current_attributes -= 1

        return dict(
            X_train=s_X_train,
            X_test=s_X_test,
            y_train=s_y_train,
            y_test=s_y_test,
        )


def get_data_loader(
    *,
    prior: bool,
    context_lengths: list[int],
    frequencies: list[int],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    max_steps: int,
    torch_rng: torch.Generator,
    batch_size: int,
    dataset_attributes: DatasetAttributes,
    num_workers: int,
) -> DataLoader:
    """Get data loader.

    This function is used to get data loader.

    Arguments:
    ----------
    X_train: pd.DataFrame
        Input features.
    y_train: pd.Series
        Target labels.
    max_steps: int
        Maximum number of steps (splits of the data).
    torch_rng: torch.Generator
        Torch random number generator for splits and similar.
    batch_size: int
        Batch size. How many splits to load at a time.
    is_classification: bool
        Whether the task is classification or regression.
    num_workers: int
        Number of workers for data loader.

    Returns:
    --------
    DataLoader
        Data loader.
    """
    # TODO move to finetuning code
    # X_train = torch.tensor(X_train.copy().values).float()
    # y_train = torch.tensor(y_train.copy().values).reshape(-1, 1).float()
    if not prior:
        dataset = TimeSeriesDataset(
            X_train=X_train,
            y_train=y_train,
            max_steps=max_steps * batch_size,
            ts_amount_for_ds = batch_size,
            dataset_attributes = dataset_attributes,
        )
    else:
        dataset = ArtificalTimeSeriesDataset(
            context_lengths=context_lengths,
            frequencies=frequencies,
            max_steps=max_steps * batch_size,
            batch_size=batch_size,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        generator=torch_rng,
        persistent_workers=False,
    )



