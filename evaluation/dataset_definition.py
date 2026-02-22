import json
import os
from pathlib import Path

from dotenv import load_dotenv

# short und medium long datasets heißt nur wie groß der forecast horizon ist
# „Macht ein längerer Forecast-Horizont für dieses Dataset fachlich Sinn?“

CHRONOS_DATASETS_METADATA = {
            "weatherbench_daily": {
                "prediction_length": 30,
                "frequency": "D",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "wiki_daily_100k": {
                "prediction_length": 30,
                "frequency": "D",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "solar_1h": {
                "prediction_length": 48,
                "frequency": "H",
                "target_column": "power_mw",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_tourism_monthly": {
                "prediction_length": 24,
                "frequency": "M",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "taxi_1h": {
                "prediction_length": 48,
                "frequency": "H",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "taxi_30min": {
                "prediction_length": 48,
                "frequency": "30min",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "uber_tlc_daily": {
                "prediction_length": 30,
                "frequency": "D",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "uber_tlc_hourly": {
                "prediction_length": 48,
                "frequency": "H",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_traffic": {
                "prediction_length": 48,
                "frequency": "H",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_electricity_hourly": {
                "prediction_length": 48,
                "frequency": "H",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_electricity_weekly": {
                "prediction_length": 13,
                "frequency": "W",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_fred_md": {
                "prediction_length": 12,
                "frequency": "M",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_covid_deaths": {
                "prediction_length": 30,
                "frequency": "D",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_hospital": {
                "prediction_length": 12,
                "frequency": "M",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "monash_london_smart_meters": {
                "prediction_length": 48,
                "frequency": "30min",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "weatherbench_hourly_10m_v_component_of_wind": {
                "prediction_length": 48,
                "frequency": "H",
                "domain": "placeholder",
                "num_variates": 1,
            },
            "weatherbench_hourly_temperature": {
                "prediction_length": 48,
                "frequency": "H",
                "domain": "placeholder",
                "num_variates": 1,
            },
        }

SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H".split() + list(CHRONOS_DATASETS_METADATA.keys())
MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H".split()

load_dotenv()
storage_env_var = "GIFT_EVAL"
# Get union of short and med_long datasets
ALL_DATASETS = list(set(SHORT_DATASETS + MED_LONG_DATASETS))
DATASET_PROPERTIES_MAP = json.load(
    open(Path(os.getenv(storage_env_var)) / "dataset_properties.json")
)
DATASET_PROPERTIES_MAP.update(CHRONOS_DATASETS_METADATA)
