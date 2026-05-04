"""
Module to generate synthetic dataset for pre training
a time series forecasting model
"""
import yaml
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from finetune_tabpfn_ts.prior.torch_generate_series import (
    save_torch_records,
    torch_generate_n,
    convert_torch_to_rows,
    load_torch_dataset,
    generate_product_input,
)
from finetune_tabpfn_ts.prior.config_variables import Config
from finetune_tabpfn_ts.task_1.dataset_utils import transform_data, to_x_y
from tabpfn_time_series import TimeSeriesDataFrame
from finetune_tabpfn_ts.edits.data_utils import PRED_LENGTH_MAP, FREQUENCY_MAP

def save_torch_dataset(prefix: str, version: str, options: dict, num_series: int = 10_000):
    """
    Generate dataset and save as tf records
    """
    for freq, freq_index in Config.freq_and_index:
        print("Frequency: " + freq)
        save_torch_records(
            prefix,
            f"{version}/{freq}.pt",
            torch_generate_n(
                N=num_series,
                freq_index=freq_index,
                # start=pd.Timestamp("2020-01-01"),
                options=options,
            ),
        )


def generate_product_input_dataset(prefix, version):
    """
    Load dataset from tf records and save as avro files
    """
    for freq in Config.frequency_names:
        print("Frequency: " + freq)
        generate_product_input(
            prefix,
            f"{version}/{freq}.avro",
            convert_torch_to_rows(
                load_torch_dataset(prefix, f"{version}/{freq}.tfrecords").as_numpy_iterator()
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    Config.set_freq_variables(config["sub_day"])
    if "transition" in config:
        Config.set_transition(config["transition"])


    save_torch_dataset(config["prefix"], config["version"], config["options"], config["num_series"])
    print(test)
    generate_product_input_dataset(config["prefix"], config["version"])

from finetune_tabpfn_ts.prior.generate_series import generate

if __name__ == "__main__":
    Config.set_freq_variables(True)
    Config.set_transition(False)
    """
    sample = torch_generate_n(
        N=1,
        freq_index=1,
        # start=pd.Timestamp("2020-01-01"),
        options={

        },
    ),
    #"id": id_,
    #"ts": torch.tensor(sample.index.astype(np.int64).values, dtype=torch.int64),
    #"y": torch.tensor(sample.series_values.values, dtype=torch.float32),
    #"noise": torch.tensor(sample.noise.values, dtype=torch.float32),
    print(list(sample[0]))
    """
    #main()
    # freq_index
        # 0 daily
        # 1 weekly
        # 2 monthly
    # is_subday(config option)
        # 0 minute
        # 1 hour
        # 2 daily
        # 3 weekly
        # 4 monthly
        # 5 yearly
    """
    cfg, sample = generate(
        100, #number of observations per time series
        freq_index=1,
        start=None,
        options={},
    )
    print(sample)
    """
