#!/usr/bin/env python3
# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate dataset statistics CSV from a list of dataset names.

Usage:
    python generate_dataset_stats.py dataset1 dataset2 dataset3
    python generate_dataset_stats.py --file datasets.txt
    python generate_dataset_stats.py --output my_stats.csv dataset1 dataset2

Output CSV columns:
    Dataset, Source, Frequency, # Series, Avg Length, Min Length, Max Length,
    # Obs, # Target Variates, Pred Length (Short), Windows (Short)
"""

import argparse
import csv
import sys
from pathlib import Path

import pyarrow.compute as pc
from finetune_tabpfn_ts.evaluation.data import (
    Dataset, Term, itemize_start, MultivariateToUnivariate,
    M4_PRED_LENGTH_MAP, PRED_LENGTH_MAP, maybe_reconvert_freq,
    MAX_WINDOW, TEST_SPLIT,
)
# ── local imports (same as dataset.py) ──────────────────────────────────────
from finetune_tabpfn_ts.evaluation.dataset_definition import CHRONOS_DATASETS_METADATA


# ── helpers ──────────────────────────────────────────────────────────────────

def avg_series_length(hf_dataset) -> float:
    if hf_dataset[0]["target"].ndim > 1:
        lengths = pc.list_value_length(
            pc.list_flatten(hf_dataset.data.column("target"))
        )
    else:
        lengths = pc.list_value_length(hf_dataset.data.column("target"))
    arr = lengths.to_numpy()
    return round(float(arr.mean()), 2)


def max_series_length(hf_dataset) -> int:
    if hf_dataset[0]["target"].ndim > 1:
        lengths = pc.list_value_length(
            pc.list_flatten(
                pc.list_slice(hf_dataset.data.column("target"), 0, 1)
            )
        )
    else:
        lengths = pc.list_value_length(hf_dataset.data.column("target"))
    return int(lengths.to_numpy().max())


def num_series(hf_dataset, target_dim: int) -> int:
    """Number of univariate series (accounts for multivariate datasets)."""
    return len(hf_dataset) * target_dim


def get_stats(name: str) -> dict:
    """Load a dataset by name and return a flat stats dict."""
    print(f"  Loading '{name}' …", flush=True)

    ds = Dataset(name, term=Term.SHORT)

    hf = ds.hf_dataset
    target_dim = ds.target_dim
    freq = ds.freq
    pred_len = ds.prediction_length   # already multiplied by SHORT (×1)
    windows = ds.windows
    min_len = ds._min_series_length
    max_len = max_series_length(hf)
    avg_len = avg_series_length(hf)
    n_series = num_series(hf, target_dim)
    n_obs = ds.sum_series_length
    source = "Chronos" if ds.is_chronos else "GIFT-Eval"

    return {
        "Dataset": name,
        "Source": source,
        "Frequency": freq,
        "# Series": n_series,
        "Avg Length": avg_len,
        "Min Length": min_len,
        "Max Length": max_len,
        "# Obs": n_obs,
        "# Target Variates": target_dim,
        "Pred Length (Short)": pred_len,
        "Windows (Short)": windows,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics and write them to a CSV file."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to process (space-separated).",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a text file with one dataset name per line.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset_stats.csv",
        help="Output CSV file path (default: dataset_stats.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect dataset names
    names: list[str] = list(args.datasets)
    if args.file:
        with open(args.file) as fh:
            file_names = [line.strip() for line in fh if line.strip()]
        names = file_names + names   # file first, then CLI extras

    if not names:
        print("ERROR: No dataset names provided. "
              "Pass them as arguments or via --file.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(names)} dataset(s) …\n")

    fieldnames = [
        "Dataset", "Source", "Frequency",
        "# Series", "Avg Length", "Min Length", "Max Length",
        "# Obs", "# Target Variates",
        "Pred Length (Short)", "Windows (Short)",
    ]

    rows = []
    errors = []
    for name in names:
        try:
            rows.append(get_stats(name))
            print(f"  ✓ {name}")
        except Exception as exc:
            print(f"  ✗ {name}: {exc}", file=sys.stderr)
            errors.append({"Dataset": name, "Source": "ERROR",
                           "Frequency": str(exc)})

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        if errors:
            # Append error rows with empty numeric fields
            for err in errors:
                writer.writerow(err)

    print(f"\nDone. Results written to: {output_path}")
    if errors:
        print(f"  {len(errors)} dataset(s) failed – see above for details.")


if __name__ == "__main__":
    main()