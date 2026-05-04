from datasets import get_dataset_config_names
from huggingface_hub import HfFileSystem
import duckdb, pandas as pd

fs = HfFileSystem()
configs = get_dataset_config_names("autogluon/chronos_datasets")

print(configs)

results = []
for config in configs:
    try:
        files = fs.glob(f"datasets/autogluon/chronos_datasets/{config}/train-*.parquet")
        paths = [f"hf://{f}" for f in files]

        stats = duckdb.sql(f"""
            SELECT 
                '{config}' as dataset,
                COUNT(*) as n_series,
                MIN(len(target)) as min_len,
                MAX(len(target)) as max_len,
                AVG(len(target)) as avg_len
            FROM read_parquet({paths})
        """).df()
        results.append(stats)
        print(f"✓ {config}: {stats['n_series'].iloc[0]} Zeitreihen")
    except Exception as e:
        print(f"✗ {config}: {e}")

summary = pd.concat(results, ignore_index=True)
summary["avg_len"] = summary["avg_len"].round(1)

output_path = "chronos_datasets_summary.csv"
summary.to_csv(output_path, index=False)

print(f"\nGespeichert: {output_path}")
print(summary.to_string(index=False))