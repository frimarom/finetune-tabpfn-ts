# split data into batches
# params: sample_size, batch_size, sample offset
# fun split into samples
# datasets into samples by first going vertically trough each time series of dataset and then moving by offset horizontally and repeating

# fun split into batches
# construct batches by grouping samples together which either are in the same time series but far apart or are from different time series
from tabpfn_time_series import TimeSeriesDataFrame
from dataclasses import dataclass


from finetune_tabpfn_ts.evaluation.data import Dataset
from enum import Enum

import torch
import numpy as np
from finetune_tabpfn_ts.edits.data_utils import get_data_loader
from finetune_tabpfn_ts.edits.feature_transformer import FeatureTransformer
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

import datasets
import random
from gluonts.dataset.util import to_pandas
from tabpfn_time_series.data_preparation import to_gluonts_univariate
import matplotlib.pyplot as plt

short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

gift_eval_datasets = short_datasets.split() + med_long_datasets.split()
autogluon_chronos_datasets = "weatherbench_daily wiki_daily_100k solar_1h"

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
}


dataset_metadata = {
    "weatherbench_daily": {
        "prediction_length": 30,
        "frequency": "D",
    },
    "wiki_daily_100k": {
        "prediction_length": 30,
        "frequency": "D",
    },
    "solar_1h": {
        "prediction_length": 48,
        "frequency": "H",
        "target_column": "power_mw"
    },
    "monash_tourism_monthly": {
        "prediction_length": 24,
        "frequency": "M",
    },
    "taxi_1h": {
        "prediction_length": 48,
        "frequency": "H",
    },
    "taxi_30min": {
        "prediction_length": 48,
        "frequency": "30min",
    },
    "uber_tlc_daily": {
        "prediction_length": 30,
        "frequency": "D",
    },
    "uber_tlc_hourly": {
        "prediction_length": 48,
        "frequency": "H",
    },
    "monash_traffic": {
        "prediction_length": 48,
        "frequency": "H",
    },
    "monash_electricity_hourly": {
        "prediction_length": 48,
        "frequency": "H",
    },
    "monash_electricity_weekly": {
        "prediction_length": 13,
        "frequency": "W",
    },
    "monash_fred_md": {
        "prediction_length": 12,
        "frequency": "M",
    },
    "monash_covid_deaths": {
        "prediction_length": 30,
        "frequency": "D",
    },
    "monash_hospital": {
        "prediction_length": 12,
        "frequency": "M",
    },
    "monash_london_smart_meters": {
        "prediction_length": 48,
        "frequency": "30min",
    },
    "weatherbench_hourly_10m_v_component_of_wind": {
        "prediction_length": 48,
        "frequency": "H",
    },
    "weatherbench_hourly_temperature": {
        "prediction_length": 48,
        "frequency": "H",
    },
}

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

def transform_autogluon_dataset(dataset_choice, dataset, ts_amount_limit: int = None):

    tsdf = TimeSeriesDataFrame(dataset)

    #limit number of ts so lido cluster doesnt oom
    if ts_amount_limit and tsdf.index.get_level_values("item_id").nunique() > ts_amount_limit:
        sampled_ids = tsdf.item_ids.to_series().sample(n=ts_amount_limit, random_state=42)
        tsdf = tsdf[tsdf.index.get_level_values("item_id").isin(sampled_ids)]

    print(tsdf)
    record = []
    for item_id, ts in tsdf.groupby(level="item_id"):
        time_series = ts.sort_index()
        transformed_time_series = transform_data(time_series)
        X, y = to_x_y(dataset_choice, transformed_time_series)
        record.append({"X": X, "y": y})
    return record


def transform_gift_eval_dataset(dataset_choice, dataset, ts_amount_limit: int = None):
    records = []

    def get_items(dataset):
        for item in dataset:
            if isinstance(item, tuple):
                input_entry, _ = item
                yield input_entry
            else:
                yield item

    all_items = list(get_items(dataset))
    if ts_amount_limit and len(all_items) > ts_amount_limit:
        indices = np.random.default_rng(seed=42).choice(len(all_items), size=int(ts_amount_limit), replace=False)
        all_items = [all_items[i] for i in indices]

    for time_series in all_items:
        target_key = "target" if "target" in time_series else "past_target"
        pandas_ts = to_pandas({**time_series, "target": time_series[target_key]})

        dataframe = pandas_ts.to_frame().reset_index()
        dataframe.columns = ["timestamp", "target"]

        dataframe["timestamp"] = dataframe["timestamp"].dt.to_timestamp()

        dataframe["item_id"] = time_series["item_id"]

        valid_count = dataframe["target"].notna().sum()
        if valid_count < 10:
            print(f"Skipping {time_series.get('item_id')}: only {valid_count} valid values")
            continue

        train_part_ts = TimeSeriesDataFrame(dataframe)
        transformed_data = transform_data(train_part_ts)
        X, y = to_x_y(dataset_choice, transformed_data)

        records.append({
            "X": X,
            "y": y,
        })

    return records

def to_x_y(name: str, data: TimeSeriesDataFrame):
    if "target" in data.columns:
        y = data["target"]
    else:
        y = data[dataset_metadata[name]["target_column"]]
    X = data.drop("target", axis=1)
    return X, y

def create_homogenous_ts_dataset(
    dataset_name: str,
    dataset: Dataset,
    ts_amount_limit: int = None,
    ts_length_limit: int = None,
    windows: int = 1,
):
    """
    Parameters
    ----------
    windows : int
        Number of sliding windows per time series.
        Each window w (0-indexed) covers:
            X[w] = padded_series[w*pred : T - (windows-1-w)*pred]
        All windows have the same length = window_length.
        Output shape: (window_length, F+1, N_ts * windows)

    Output
    ------
    X_out        : (window_length, F+1, N_ts * windows)
    y_out        : (window_length, 1,   N_ts * windows)
    window_length: int
    pred_length  : int
    """
    if dataset_name in gift_eval_datasets:
        records = transform_gift_eval_dataset(dataset_name, dataset, ts_amount_limit)
    elif dataset_name in dataset_metadata.keys():
        records = transform_autogluon_dataset(dataset_name, dataset, ts_amount_limit)
    else:
        raise ValueError(
            f"Dataset {dataset_name} not found in either "
            "gift_eval_datasets or autogluon_chronos_datasets"
        )

    pred_length = get_prediction_length(dataset_name)

    # ── target length = length of ONE window ─────────────────────────────────
    # raw series length after optional truncation, then subtract the parts that
    # belong to the other windows so every window has the same length.
    raw_lengths   = [len(ts["y"]) for ts in records]
    median_length = int(np.median(raw_lengths))
    if ts_length_limit is not None:
        median_length = min(ts_length_limit, median_length)

    # window_length: the full padded series is split into `windows` chunks of
    # equal size. Chunk w covers [w*pred_length, median_length - (windows-1-w)*pred_length).
    # All chunks have the same length = median_length - (windows-1)*pred_length.
    window_length = median_length - (windows - 1) * pred_length
    if window_length <= 0:
        raise ValueError(
            f"window_length={window_length} <= 0. "
            f"Reduce `windows` or increase `ts_length_limit`."
        )

    X_out = []
    y_out = []

    for record in records:
        X_df     = record["X"]
        y_series = record["y"]

        X_np = X_df.values.astype(float)
        y_np = y_series.values.astype(float).reshape(-1, 1)

        T_i, F = X_np.shape

        # ── pad / truncate to median_length ──────────────────────────────────
        if T_i > median_length:
            X_np = X_np[-median_length:]
            y_np = y_np[-median_length:]
            padding_mask = np.ones((median_length, 1), dtype=float)

        elif T_i < median_length:
            pad_len  = median_length - T_i
            X_np     = np.concatenate([np.zeros((pad_len, F), dtype=float), X_np], axis=0)
            y_np     = np.concatenate([np.zeros((pad_len, 1), dtype=float), y_np], axis=0)
            padding_mask = np.concatenate(
                [np.zeros((pad_len, 1), dtype=float),
                 np.ones((T_i,    1), dtype=float)],
                axis=0,
            )

        else:
            padding_mask = np.ones((median_length, 1), dtype=float)

        X_full = np.concatenate([X_np, padding_mask], axis=1)  # (median_length, F+1)

        # ── slice into windows ────────────────────────────────────────────────
        # window w: rows [w*pred_length : w*pred_length + window_length]
        for w in range(windows):
            start = w * pred_length
            end   = start + window_length          # = median_length - (windows-1-w)*pred

            X_win = X_full[start:end]              # (window_length, F+1)
            y_win = y_np[start:end]                # (window_length, 1)

            X_tensor = torch.tensor(X_win).float().transpose(0, 1)  # (F+1, window_length)
            y_tensor = torch.tensor(y_win).float().transpose(0, 1)  # (1,   window_length)

            X_out.append(X_tensor)
            y_out.append(y_tensor)

    X_out = torch.stack(X_out, dim=0)   # (N_ts*windows, F+1, window_length)
    y_out = torch.stack(y_out, dim=0)   # (N_ts*windows, 1,   window_length)

    X_out = X_out.permute(2, 1, 0)      # (window_length, F+1, N_ts*windows)
    y_out = y_out.permute(2, 1, 0)      # (window_length, 1,   N_ts*windows)

    return X_out, y_out, window_length, pred_length

def get_prediction_length(dataset_name: str):
    if dataset_name in gift_eval_datasets:
        dataset = load_dataset(dataset_name)
        return dataset.prediction_length
    elif dataset_name in dataset_metadata.keys():
        return dataset_metadata[dataset_name]["prediction_length"]
    else:
        raise ValueError(f"Dataset {dataset_name} not found in either gift_eval_datasets or autogluon_chronos_datasets")


def create_train_val_split(
        dataset_name: str,
        max_training_ts_amount: int = None,
        max_context_length: int = None,
        max_validation_ts_amount: int = None):
    dataset = Dataset(name = dataset_name)
    print(list(dataset.training_dataset))
    #switch between if windows >1 or not for windowed or not

    X_train, y_train, target_length, prediction_length = create_homogenous_ts_dataset(
        dataset_name,
        dataset.training_dataset,
        ts_amount_limit=max_training_ts_amount,
        ts_length_limit=max_context_length,
        windows=dataset.windows)
    X_val, y_val, _, _ = create_homogenous_ts_dataset(
        dataset_name,
        dataset.validation_dataset,
        ts_amount_limit=max_validation_ts_amount,
        ts_length_limit=max_context_length,
        windows=1)
    return X_train, y_train, X_val, y_val, target_length, prediction_length

if __name__ == "__main__":
    #TODO select random datasets from it when it comes to validation data
    X_train, y_train, X_val, y_val, target_length, prediction_length = create_train_val_split("us_births/D", max_training_ts_amount=1, max_context_length=4096, max_validation_ts_amount=1)
    print(X_train.shape)
    print(X_val.shape)
    print(prediction_length)

    #dataset = Dataset(name = "hierarchical_sales/D")
    #print(list(dataset.validation_dataset)[0])
    """
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
    """
