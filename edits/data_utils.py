from __future__ import annotations

from typing import TYPE_CHECKING

from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import pandas as pd

RANDOM_SEED = 4213

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
    time_series_window_count = []

    def __init__(
        self,
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        max_steps: int,
        windows: int,
        forecast_horizon: int,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.max_steps = max_steps
        self._splits_generator = self.splits_generator(
            X_train=X_train,
            y_train=y_train,
            windows = windows,
            forecast_horizon=forecast_horizon,
            seed=RANDOM_SEED,
        )
        self._rng = np.random.RandomState(RANDOM_SEED)
        TimeSeriesDataset.time_series_window_count = np.zeros(X_train.shape[2])

    @staticmethod
    def splits_generator(
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor, # TODO obsolete?
        windows: int,
        forecast_horizon: int,
        seed: int,
    ):
        """Endless generator for splits of a time series sliding window."""
        series_length = X_train.shape[0]
        z_len = X_train.shape[2]

        rng = np.random.RandomState(seed)

        while True:
            time_series = rng.randint(0, z_len)
            while TimeSeriesDataset.time_series_window_count[time_series] >= windows :
                time_series = rng.randint(0, z_len)

            window_count = TimeSeriesDataset.time_series_window_count[time_series]
            context_length = (series_length - forecast_horizon) // windows

            start_idx = max(0, series_length - forecast_horizon - (windows-window_count) * context_length)
            origin = start_idx+context_length
            end_idx = origin+forecast_horizon

            TimeSeriesDataset.time_series_window_count[time_series] += 1

            yield int(time_series), int(start_idx), int(origin), int(end_idx)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_splits_generator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._splits_generator = self.splits_generator(
            X_train=self.X_train,
            y_train=self.y_train,
            context_size=self.context_size,
            forecast_horizon=self.forecast_horizon,
            seed=self.RANDOM_SEED,
        )

    def __len__(self):
        return self.max_steps # amount of batches. Each batch contains batch_size amount of splits returned by get_item.

    def get_splits(self) -> tuple[int, int, int, int]:
        """Get train and test indices for next batch."""
        time_series, start_idx, train_test_bound, end_idx = next(self._splits_generator)
        return time_series, start_idx, train_test_bound, end_idx

    def create_data(self, time_series, start_idx, train_test_bound, end_idx):
        sample_X_train = self.X_train[start_idx:train_test_bound, :, time_series]
        sample_X_test = self.X_train[train_test_bound:end_idx, :, time_series]
        sample_y_train = self.y_train[start_idx:train_test_bound, :, time_series]
        sample_y_test = self.y_train[train_test_bound:end_idx, :, time_series]
        return sample_X_train, sample_X_test, sample_y_train, sample_y_test

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        time_series, start_idx, train_test_bound, end_idx = self.get_splits()
        s_X_train, s_X_test, s_y_train, s_y_test = self.create_data(
            time_series, start_idx, train_test_bound, end_idx
        )
        # filter out padding filled time series
        while (s_X_train[:, 28].numpy() == 0).sum() >= s_X_train.shape[0]*0.5:
            time_series, start_idx, train_test_bound, end_idx = self.get_splits()
            s_X_train, s_X_test, s_y_train, s_y_test = self.create_data(
                time_series, start_idx, train_test_bound, end_idx
            )

        return dict(
            X_train=s_X_train,
            X_test=s_X_test,
            y_train=s_y_train,
            y_test=s_y_test,
        )


def get_data_loader(
    *,
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
    dataset = TimeSeriesDataset(
        X_train=X_train,
        y_train=y_train,
        max_steps=max_steps * batch_size,
        windows=dataset_attributes.windows,
        forecast_horizon=dataset_attributes.forecast_horizon,
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

def create_batch(dataset: TimeSeriesDataset, batch_size: int) -> dict[str, torch.Tensor]:
    batch = {
        "X_train": [],
        "X_test": [],
        "y_train": [],
        "y_test": [],
    }
    for _ in range(batch_size):
        sample = dataset.__getitem__(0)  # idx is not used in __getitem__
        for key in batch:
            batch[key].append(sample[key])
    for key in batch:
        batch[key] = torch.stack(batch[key], dim=0)
    return batch


def get_batches_with_variable_forecast_horizon(
    *,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    max_steps: int,
    torch_rng: torch.Generator,
    batch_size: int,
    forecast_horizon,
    sample_offset: int,
):
    forecast_horizons = []
    for i in range(max_steps):
        fh = forecast_horizon()
        forecast_horizons.append(fh)

    counts = Counter(forecast_horizons)

    batches = []
    for forecast_horizon_element, amount in counts.items():
        forecast_horizon_element = int(forecast_horizon_element)
        context_size = 2 * forecast_horizon_element # TODO maybe find a better way by doing

        dataset = TimeSeriesDataset(
            X_train=X_train,
            y_train=y_train,
            max_steps=max_steps * batch_size,
            context_size=context_size,
            forecast_horizon=forecast_horizon_element,
            sample_offset=sample_offset
        )
        for _ in range(amount):
            batch = create_batch(dataset, batch_size)
            batches.append(batch)

    indices = torch.randperm(len(batches), generator=torch_rng).numpy()
    shuffled_batches = [batches[i] for i in indices]

    return shuffled_batches

def get_batches_with_whole_ts(
    *,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    max_steps: int,
    torch_rng: torch.Generator,
    batch_size: int,
    forecast_horizon: int,
    sample_offset: int,
):
    batches = []
    for _ in range(amount):
        dataset = TimeSeriesDataset(
            X_train=X_train,
            y_train=y_train,
            max_steps=max_steps * batch_size,
            context_size=context_size,
            forecast_horizon=forecast_horizon,
            sample_offset=sample_offset
        )
        batch = create_batch(dataset, batch_size)
        batches.append(batch)

    indices = torch.randperm(len(batches), generator=torch_rng).numpy()
    shuffled_batches = [batches[i] for i in indices]

    return shuffled_batches

