from tabpfn_time_series import TimeSeriesDataFrame

from gift_eval.data import Dataset, itemize_start
from enum import Enum

import torch
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

def get_dataset_by_term(term: Term):
    if term == Term.SHORT:
        return short_datasets.split()
    elif term == Term.MEDIUM:
        return med_long_datasets.split()
    else:
        return []

def to_timeseries_dataframe_list(dataset: Dataset, maxdata=None):
    records = []
    testdata_iterator = iter(dataset.test_data)
    count = 0

    for test_batch in testdata_iterator:
        train_part = to_pandas(test_batch[0])
        test_part = to_pandas(test_batch[1])

        train_dataframe = train_part.to_frame().reset_index()
        train_dataframe.columns = ["timestamp", "target"]

        test_dataframe = test_part.to_frame().reset_index()
        test_dataframe.columns = ["timestamp", "target"]

        train_dataframe["timestamp"] = train_dataframe["timestamp"].dt.to_timestamp()
        test_dataframe["timestamp"] = test_dataframe["timestamp"].dt.to_timestamp()

        train_dataframe["item_id"] = test_batch[0]["item_id"]
        test_dataframe["item_id"] = test_batch[1]["item_id"]

        train_part_ts = TimeSeriesDataFrame(train_dataframe)
        test_part_ts = TimeSeriesDataFrame(test_dataframe)

        record = {
            "train": train_part_ts,
            "test": test_part_ts,
        }
        records.append(record)
        count += 1
        if maxdata is not None and count >= maxdata:
            break


    return records

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

        records.append(transformed_data)

    return records

def to_x_y(data: TimeSeriesDataFrame):
    y = data["target"]
    X = data.drop("target", axis=1)
    return X, y

if __name__ == "__main__":
    ds_name = "m4_monthly"  # Name of the dataset
    dataset = load_dataset(ds_name)

    print("Dataset sum series length", dataset.sum_series_length)
    print("Dataset frequency: ", dataset.freq)
    print("Prediction length: ", dataset.prediction_length)
    print("Number of windows in the rolling evaluation: ", dataset.windows)

    print("covid_deaths")
    data = load_and_transform_dataset("covid_deaths")
    print(data)

    X_1, y_1 = to_x_y(data[0])
    X_2, y_2 = to_x_y(data[1])

    X_train = torch.tensor(X_1.copy().values).float()
    y_train = torch.tensor(y_1.copy().values).reshape(-1, 1).float()
    X_train_1 = torch.tensor(X_2.copy().values).float()
    y_train_1 = torch.tensor(y_2.copy().values).reshape(-1, 1).float()

    X_stacked = torch.stack([X_train, X_train_1], dim=-1)  # dim=-1 = neue z-Achse
    y_stacked = torch.stack([y_train, y_train_1], dim=-1)


    print(X_train, y_train)
    x_len = X_stacked.shape[0]
    y_len = X_stacked.shape[1]
    z_len = X_stacked.shape[2]
    print("X_stacked shape:", X_stacked.shape)
