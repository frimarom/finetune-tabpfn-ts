"""
Module containing constants for synthetic data generation
"""

from datetime import date
import torch

BASE_START = date.fromisoformat("1885-01-01").toordinal()
BASE_END = date.fromisoformat("2023-12-31").toordinal() + 1

PRODUCT_SCHEMA = {
    "doc": "Timeseries sample",
    "name": "TimeseriesSample",
    "type": "record",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "ts", "type": {"type": "int", "logicalType": "date"}},
        {"name": "y", "type": ["null", "float"]},
        {"name": "noise", "type": ["float"]}
    ],
}

CONTEXT_LENGTH = 1_000

TORCH_SCHEMA = {
    "id": {"dtype": torch.int64, "shape": []},
    "ts": {"dtype": torch.int64, "shape": [CONTEXT_LENGTH]},
    "y": {"dtype": torch.float32, "shape": [CONTEXT_LENGTH]},
    "noise": {"dtype": torch.float32, "shape": [CONTEXT_LENGTH]}
}