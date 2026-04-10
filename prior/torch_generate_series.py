"""
Module to process synthetic series using PyTorch
"""

import fastavro
import numpy as np
import pandas as pd
import torch
from datetime import date
from finetune_tabpfn_ts.prior.generate_series import generate
from finetune_tabpfn_ts.prior.constants import *
from finetune_tabpfn_ts.prior.series_config import *


def torch_generate_n(
    N=100,
    size=CONTEXT_LENGTH,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
):
    """
    Generate time series as dicts of tensors
    """
    for i in range(N):
        if i % 1000 == 0:
            print(f"Completed: {i}")

        if i < N * options.get("linear_random_walk_frac", 0):
            cfg, sample = generate(
                size,
                freq_index=freq_index,
                start=start,
                options=options,
                random_walk=True,
            )
        else:
            cfg, sample = generate(
                size, freq_index=freq_index, start=start, options=options
            )

        id_ = str(cfg)
        yield {
            "id": id_,
            "ts": torch.tensor(sample.index.astype(np.int64).values, dtype=torch.int64),
            "y": torch.tensor(sample.series_values.values, dtype=torch.float32),
            "noise": torch.tensor(sample.noise.values, dtype=torch.float32),
        }


def save_torch_records(prefix:str, dest: str, it):
    """
    Save records locally using torch.save
    """
    records = list(it)
    torch.save(records, prefix+"/"+dest)


def decode_fn(record):
    # Validates expected keys and tensor shapes
    assert set(record.keys()) == {"id", "ts", "y", "noise"}
    assert record["ts"].shape == (CONTEXT_LENGTH,)
    assert record["y"].shape == (CONTEXT_LENGTH,)
    assert record["noise"].shape == (CONTEXT_LENGTH,)
    return record


def load_torch_dataset(prefix: str, src: str):
    records = torch.load(prefix+"/"+src)

    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.records = [decode_fn(r) for r in records]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            return self.records[idx]

    return TimeSeriesDataset(records)


def convert_torch_to_rows(records):
    for i, r in enumerate(records):
        if i % 1000 == 0:
            print(f"Completed: {i}")
        id_ = r["id"]
        for ts, y, noise in zip(
            (date.fromtimestamp(v / 1_000_000_000) for v in r["ts"].tolist()),
            r["y"].tolist(),
            r["noise"].tolist(),
        ):
            yield {"id": id_, "ts": ts, "y": y, "noise": noise}


def generate_product_input(prefix:str, dest: str, it):
    """
    Write generated dataset into avro files
    """
    with open(prefix+"/"+dest, "wb") as file:
        fastavro.writer(file, PRODUCT_SCHEMA, it, codec="deflate")