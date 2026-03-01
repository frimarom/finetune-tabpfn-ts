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
        ts_amount_for_ds: int,
        dataset_attributes: DatasetAttributes,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.max_steps = max_steps
        self.dataset_attributes = dataset_attributes
        #next(self._splits_generator)
        self._rng = np.random.RandomState(RANDOM_SEED)
        TimeSeriesDataset.time_series_window_count = np.zeros(X_train[0].shape[2])
        self.current_ds = 0
        self.ts_amount_for_ds = ts_amount_for_ds
        self.ts_left_for_ds = ts_amount_for_ds

    @staticmethod
    def splits_generator(
            *,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            dataset_attributes: DatasetAttributes,
            seed: int,
    ):
        rng = np.random.RandomState(seed)
        while True:
            current_ds = yield  # <- Generator wartet hier auf current_ds
            if current_ds is None:
                current_ds = rng.randint(0, len(X_train))
            windows = dataset_attributes[current_ds].windows
            forecast_horizon = dataset_attributes[current_ds].forecast_horizon
            z_len = X_train[current_ds].shape[2]
            series_length = X_train[current_ds].shape[1]

            if np.all(TimeSeriesDataset.time_series_window_count == windows):
                TimeSeriesDataset.time_series_window_count = np.zeros(z_len)

            time_series = rng.randint(0, z_len)
            while TimeSeriesDataset.time_series_window_count[time_series] >= windows:
                time_series = rng.randint(0, z_len)

            window_count = TimeSeriesDataset.time_series_window_count[time_series]
            context_length = (series_length - forecast_horizon) // windows

            start_idx = max(0, series_length - forecast_horizon - (windows - window_count) * context_length)
            origin = start_idx + context_length
            end_idx = origin + forecast_horizon

            TimeSeriesDataset.time_series_window_count[time_series] += 1

            yield int(current_ds), int(time_series), int(start_idx), int(origin), int(end_idx)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_splits_generator"]
        return state

    def __setstate__(self, state):
        seed = RANDOM_SEED
        self.__dict__.update(state)
        self._splits_generator = self.splits_generator(
            X_train=self.X_train,
            y_train=self.y_train,
            dataset_attributes=self.dataset_attributes,
            seed=seed,
        )
        next(self._splits_generator)

    def __len__(self):
        return self.max_steps # amount of batches. Each batch contains batch_size amount of splits returned by get_item.

    # splits_generator komplett ersetzen:
    def _get_next_split(self):
        current_ds = self.current_ds
        windows = self.dataset_attributes[current_ds].windows
        forecast_horizon = self.dataset_attributes[current_ds].forecast_horizon
        z_len = self.X_train[current_ds].shape[2]
        series_length = self.X_train[current_ds].shape[1]

        if np.all(TimeSeriesDataset.time_series_window_count[:z_len] == windows):
            TimeSeriesDataset.time_series_window_count = np.zeros(z_len)

        time_series = self._rng.randint(0, z_len)
        while TimeSeriesDataset.time_series_window_count[time_series] >= windows:
            time_series = self._rng.randint(0, z_len)

        window_count = TimeSeriesDataset.time_series_window_count[time_series]
        context_length = (series_length - forecast_horizon) // windows

        start_idx = max(0, series_length - forecast_horizon - (windows - window_count) * context_length)
        origin = start_idx + context_length
        end_idx = origin + forecast_horizon

        TimeSeriesDataset.time_series_window_count[time_series] += 1

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
            self.current_ds = self._rng.randint(0, len(self.X_train))
            self.ts_left_for_ds = self.ts_amount_for_ds

        dataset_id, time_series, start_idx, train_test_bound, end_idx = self.get_splits()
        s_X_train, s_X_test, s_y_train, s_y_test = self.create_data(
            dataset_id, time_series, start_idx, train_test_bound, end_idx
        )

        self.ts_left_for_ds -= 1


        #TODO overwork this part
        """
        # filter out padding filled time series
        while (s_X_train[:, 28].numpy() == 0).sum() >= s_X_train.shape[0]*0.5:
            TimeSeriesDataset.time_series_window_count[time_series] += 1
            time_series, start_idx, train_test_bound, end_idx = self.get_splits()
            s_X_train, s_X_test, s_y_train, s_y_test = self.create_data(
                time_series, start_idx, train_test_bound, end_idx
            )
        """

        return dict(
            X_train=s_X_train,
            X_test=s_X_test,
            y_train=s_y_train,
            y_test=s_y_test,
        )

class ArtificalTimeSeriesDataset(Dataset):
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
        self._splits_generator = self.splits_generator(
            X_train=X_train,
            y_train=y_train,
            dataset_attributes=dataset_attributes,
            seed=RANDOM_SEED,
        )
        self.dataset_attributes = dataset_attributes
        next(self._splits_generator)
        self._rng = np.random.RandomState(RANDOM_SEED)
        TimeSeriesDataset.time_series_window_count = np.zeros(X_train[0].shape[2])
        self.current_ds = 0
        self.ts_amount_for_ds = ts_amount_for_ds
        self.ts_left_for_ds = ts_amount_for_ds

    @staticmethod
    def splits_generator(
            *,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            dataset_attributes: DatasetAttributes,
            seed: int,
    ):
        while True:

            yield int(current_ds), int(time_series), int(start_idx), int(origin), int(end_idx)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_splits_generator"]
        return state

    def __setstate__(self, state):
        seed = RANDOM_SEED
        self.__dict__.update(state)
        self._splits_generator = self.splits_generator(
            X_train=self.X_train,
            y_train=self.y_train,
            dataset_attributes=self.dataset_attributes,
            seed=seed,
        )
        next(self._splits_generator)

    def __len__(self):
        return self.max_steps # amount of batches. Each batch contains batch_size amount of splits returned by get_item.

    def get_splits(self):
        result = self._splits_generator.send(self.current_ds)
        return result

    def create_data(self, dataset_id, time_series, start_idx, train_test_bound, end_idx):

        return sample_X_train, sample_X_test, sample_y_train, sample_y_test

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.ts_left_for_ds <= 0:
            self.current_ds = self._rng.randint(0, len(self.X_train))
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
        ts_amount_for_ds = batch_size,
        dataset_attributes = dataset_attributes,
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



