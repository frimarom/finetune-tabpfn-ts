# split data into batches
# params: sample_size, batch_size, sample offset
# fun split into samples
# datasets into samples by first going vertically trough each time series of dataset and then moving by offset horizontally and repeating

# fun split into batches
# construct batches by grouping samples together which either are in the same time series but far apart or are from different time series
from tabpfn_time_series import TimeSeriesDataFrame
from dataclasses import dataclass


from gift_eval.data import Dataset, itemize_start
from enum import Enum

import torch
import pandas as pd
import numpy as np
from finetune_tabpfn_ts.edits.data_utils import get_data_loader, create_batch, get_batches_with_variable_forecast_horizon
from finetune_tabpfn_ts.edits.feature_transformer import FeatureTransformer
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

from pathlib import Path
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt

short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass
class DatasetAttributes:

    name: str

    time_series_length: int

    ts_amount: str

    forecast_horizon: int

    context_size: int # maybe useless since it can be calculated with forecast horizon and offset and windows

    offset: int

    windows: int

    @property
    def report_str(self):
        return f"""
        === Learning HPs ===
            \tDataset Name: {self.name}
            \tTime series amount: {self.ts_amount}
            \tContext Size: {self.context_size} | Forecast horizon: {self.forecast_horizon} | Offset: {self.offset}
            \tWindows: {self.windows}
        """

def load_dataset(name: str) -> Dataset:
    term = ""
    if name in short_datasets.split():
        term = "short"
    elif name in med_long_datasets.split():
        term = "medium"

    dataset = None
    if term != "":
        to_univariate = (
            False
            if Dataset(name=name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=name, term=term, to_univariate=to_univariate)
    return dataset

def transform_data(train_data: TimeSeriesDataFrame):
    selected_features = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]
    feature_transformer = FeatureTransformer(selected_features)

    return feature_transformer.transform_one_dataframe(train_data)

def load_and_transform_dataset(name: str):
    dataset = load_dataset(name)

    records = []
    for time_series in dataset.gluonts_dataset:
        pandas_ts = to_pandas(time_series)

        dataframe = pandas_ts.to_frame().reset_index()
        dataframe.columns = ["timestamp", "target"]

        dataframe["timestamp"] = dataframe["timestamp"].dt.to_timestamp()

        dataframe["item_id"] = time_series["item_id"]

        train_part_ts = TimeSeriesDataFrame(dataframe)
        transformed_data = transform_data(train_part_ts)
        X, y = to_x_y(transformed_data)

        records.append({
            "X": X,
            "y": y,
        })

    return records

def to_x_y(data: TimeSeriesDataFrame):
    y = data["target"]
    X = data.drop("target", axis=1)
    return X, y

def stack_records_along_z(records):
    X_list = []
    y_list = []

    for record in records:
        X = record["X"]
        y = record["y"]

        X_tensor = torch.tensor(X.copy().values).float()
        y_tensor = torch.tensor(y.copy().values).reshape(-1, 1).float()

        X_list.append(X_tensor.numpy())
        y_list.append(y_tensor.numpy())

    X_stacked = np.array(X_list, dtype=object)
    y_stacked = np.array(y_list, dtype=object)
    #X_stacked = torch.stack(X_list, dim=-1)
    #y_stacked = torch.stack(y_list, dim=-1)

    return X_stacked, y_stacked

def get_transformed_stacked_dataset(dataset: str):
    data = load_and_transform_dataset(dataset)
    X_stacked, y_stacked = stack_records_along_z(data)
    return X_stacked, y_stacked

def create_homgenous_ts_dataset(
    dataset_name: str,
    target_length: int,
):
    records = load_and_transform_dataset(dataset_name)

    X_out = []
    y_out = []

    for record in records:
        X_df = record["X"]          # DataFrame (T_i, F)
        y_series = record["y"]      # Series (T_i,)

        X_np = X_df.values.astype(float)
        y_np = y_series.values.astype(float).reshape(-1, 1)

        T_i, F = X_np.shape

        # -------------------------------
        # Truncate from left if too long
        # -------------------------------
        if T_i > target_length:
            X_np = X_np[-target_length:]
            y_np = y_np[-target_length:]
            padding_mask = np.ones((target_length, 1), dtype=float)

        # -------------------------------
        # Pad from left if too short
        # -------------------------------
        elif T_i < target_length:
            pad_len = target_length - T_i

            X_pad = np.zeros((pad_len, F), dtype=float)
            y_pad = np.zeros((pad_len, 1), dtype=float)

            X_np = np.concatenate([X_pad, X_np], axis=0)
            y_np = np.concatenate([y_pad, y_np], axis=0)

            padding_mask = np.concatenate(
                [np.zeros((pad_len, 1), dtype=float),
                 np.ones((T_i, 1), dtype=float)],
                axis=0
            )

        # -------------------------------
        # Exact length → no pad, but mask still needed
        # -------------------------------
        else:
            padding_mask = np.ones((target_length, 1), dtype=float)

        # -------------------------------
        # Append padding feature
        # -------------------------------
        X_with_padding = np.concatenate([X_np, padding_mask], axis=1)

        # -------------------------------
        # Convert to torch + reshape
        # -------------------------------
        X_tensor = torch.tensor(X_with_padding).float().transpose(0, 1)
        y_tensor = torch.tensor(y_np).float().transpose(0, 1)

        X_out.append(X_tensor)
        y_out.append(y_tensor)

    X_out = torch.stack(X_out, dim=0)  # (N_ts, F+1, L)
    y_out = torch.stack(y_out, dim=0)  # (N_ts, 1, L)

    X_out = X_out.permute(2, 1, 0)  # (L, F+1, N_ts)
    y_out = y_out.permute(2, 1, 0)  # (L, 1, N_ts)

    return X_out, y_out


if __name__ == "__main__":
    dataset = Dataset("SZ_TAXI/H")
    print(len(dataset.gluonts_dataset))

    X, y = create_homgenous_ts_dataset("SZ_TAXI/H", 1035)
    print("X_dim", X.shape)
    print("Y_dim", y.shape)
    plt.plot(X[0, 0, :], y[0, 0, :])
    plt.plot(X[1, 0, :], y[1, 0, :])
    plt.plot(X[2, 0, :], y[2, 0, :])
    plt.plot(X[3, 0, :], y[3, 0, :])
    plt.savefig("finetune_tabpfn_ts/m4_weekly.png")
