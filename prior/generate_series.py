"""
Module to generate synthetic series
"""

import numpy as np
import pandas as pd
from datetime import date

from pandas.tseries.frequencies import to_offset
from finetune_tabpfn_ts.prior.constants import *
from finetune_tabpfn_ts.prior.config_variables import Config
from finetune_tabpfn_ts.prior.generate_series_components import make_series
from finetune_tabpfn_ts.prior.utils import sample_scale, get_transition_coefficients
from finetune_tabpfn_ts.prior.series_config import ComponentScale, SeriesConfig, ComponentNoise
from scipy.stats import beta

def __generate(
    n=100,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
    random_walk: bool = False,
):
    if n is None or int(n) <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    if freq_index is None:
        freq_index = np.random.choice(len(Config.frequencies))

    freq, timescale = Config.frequencies[freq_index]

    a, m, w, h, minute = 0.0, 0.0, 0.0, 0.0, 0.0
    if freq == "min":
        minute = np.random.uniform(0.0, 1.0)
        h = np.random.uniform(0.0, 0.2)
    elif freq == "H":
        minute = np.random.uniform(0.0, 0.2)
        h = np.random.uniform(0.0, 1)
    elif freq == "D":
        w = np.random.uniform(0.0, 1.0)
        m = np.random.uniform(0.0, 0.2)
    elif freq == "W":
        m = np.random.uniform(0.0, 0.3)
        a = np.random.uniform(0.0, 0.3)
    elif freq == "MS":
        w = np.random.uniform(0.0, 0.1)
        a = np.random.uniform(0.0, 0.5)
    elif freq == "Y":
        w = np.random.uniform(0.0, 0.2)
        a = np.random.uniform(0.0, 1)
    else:
        raise NotImplementedError(f"Unsupported frequency: {freq}")

    if start is None:
        last_error = None
        for _ in range(100):
            sampled_ord = int((BASE_END - BASE_START) * beta.rvs(5, 1) + BASE_START)
            candidate_start = pd.Timestamp(date.fromordinal(sampled_ord))

            try:
                test_dates = pd.date_range(start=candidate_start, periods=int(n), freq=to_offset(freq))
                if len(test_dates) == int(n):
                    start = candidate_start
                    break
            except Exception as e:
                last_error = e
                continue

        if start is None:
            raise ValueError(
                f"Could not sample valid start date for freq={freq}, n={n}. "
                f"Last error: {last_error}"
            )

    scale_config = ComponentScale(
        1.0,
        np.random.normal(0, 0.01),
        np.random.normal(1, 0.005 / timescale),
        a=a,
        m=m,
        w=w,
        minute=minute,
        h=h
    )

    offset_config = ComponentScale(
        0,
        np.random.uniform(-0.1, 0.5),
        np.random.uniform(-0.1, 0.5),
        a=np.random.uniform(0.0, 1.0),
        m=np.random.uniform(0.0, 1.0),
        w=np.random.uniform(0.0, 1.0),
    )

    noise_config = ComponentNoise(
        k=np.random.uniform(1, 5),
        median=1,
        scale=sample_scale()
    )

    cfg = SeriesConfig(scale_config, offset_config, noise_config)

    return cfg, make_series(cfg, to_offset(freq), int(n), start, options, random_walk)

def generate(
    n = 100,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
    random_walk: bool = False,
):
    """
    Function to generate a synthetic series for a given config
    """

    cfg1, series1 = __generate(n, freq_index, start, options, random_walk)
    cfg2, series2 = __generate(n, freq_index, start, options, random_walk)

    if Config.transition:
        coeff = get_transition_coefficients(CONTEXT_LENGTH)
        values = coeff * series1['values'] + (1 - coeff) * series2['values']
    else:
        values = series1['values']

    dataframe_data = {
        'series_values': values,
        'noise': series1['noise']
    }

    return cfg1, pd.DataFrame(data=dataframe_data, index=series1['dates'])#.clip(lower=0.0)

