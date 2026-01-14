from __future__ import annotations

from typing import TYPE_CHECKING

from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import pandas as pd


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
        context_size: int,
        forecast_horizon: int,
        sample_offset: int
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.max_steps = max_steps
        self._splits_generator = self.splits_generator(
            X_train=X_train,
            y_train=y_train,
            context_size=context_size,
            forecast_horizon=forecast_horizon,
            seed=RANDOM_SEED,
        )
        self._rng = np.random.RandomState(RANDOM_SEED)

    @staticmethod
    def splits_generator(
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor, # TODO obsolete?
        context_size: int,
        forecast_horizon: int,
        seed: int,
    ):
        """Endless generator for splits of a time series sliding window."""
        x_len = X_train.shape[0]
        z_len = X_train.shape[2]

        rng = np.random.RandomState(seed)

        while True:
            # TODO Outsource like in finetuning data_utils so that samples can also be created in a linear deterministic way
            time_series = rng.randint(0, z_len)
            # TODO start index with sample_offset to avoid starting at same point mutiple times
            start_idx = rng.randint(0, x_len - context_size - forecast_horizon + 1)
            train_test_bound = start_idx + context_size
            end_idx = start_idx + context_size + forecast_horizon

            yield from (time_series, start_idx, train_test_bound, end_idx)

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        time_series, start_idx, train_test_bound, end_idx = self.get_splits()

        # Correct for equal batch size
        sample_X_train = self.X_train[start_idx:train_test_bound, :, time_series]
        sample_X_test = self.X_train[train_test_bound:end_idx, :, time_series]
        sample_y_train = self.y_train[start_idx:train_test_bound, :, time_series]
        sample_y_test = self.y_train[train_test_bound:end_idx, :, time_series]

        return dict(
            X_train=sample_X_train,
            X_test=sample_X_test,
            y_train=sample_y_train,
            y_test=sample_y_test,
        )


def get_data_loader(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_steps: int,
    torch_rng: torch.Generator,
    batch_size: int,
    sample_offset: int,
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
    X_train = torch.tensor(X_train.copy().values).float()
    y_train = torch.tensor(y_train.copy().values).reshape(-1, 1).float()
    dataset = TimeSeriesDataset(
        X_train=X_train,
        y_train=y_train,
        max_steps=max_steps * batch_size,
        context_size=context_size,
        forecast_horizon=forecast_horizon,
        sample_offset=sample_offset
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
    X_train: pd.DataFrame,
    y_train: pd.Series,
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

    counts = Counter(lst)
    X_train = torch.tensor(X_train.copy().values).float()
    y_train = torch.tensor(y_train.copy().values).reshape(-1, 1).float()

    batches = []
    for forecast_horizon_element, amount in counts.items():
        forecast_horizon_element = int(forecast_horizon_element)
        context_size = 2 * forecast_horizon_element

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

    indices = torch.randperm(len(lst), generator=torch_rng).numpy()
    shuffled_batches = [batches[i] for i in indices]

    return shuffled_batches

