# create different finetuning config for different datasets
# for the parameters that have to be adjusted fpr every config for hyperparameter search make parameter a list with all values to try and create config for every combination of those parameters
import yaml
import os
from itertools import product
import json
import argparse

def create_configs(base_config_path, output_dir, param_grid, base_dataset_config):
    with open(base_config_path) as config_file:
        base_config = yaml.safe_load(config_file)

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    print(combinations)
    combinations = [combo for combo in combinations if combo["l2_sp_lambda"] == 0.0 or combo["weight_decay"] == 0.0]

    print(len(combinations))
    #get number of files in output dir to use as offset for naming new config files
    print(os.listdir(output_dir))
    offset = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])
    print(offset)

    for i, combo in enumerate(combinations):
        config_copy = base_config.copy()
        config_copy["finetuning"].update(base_dataset_config["finetuning"])
        config_copy["finetuning"].update(combo)
        config_copy["dataset"].update(base_dataset_config["dataset"])

        output_path = os.path.join(output_dir, f"{base_dataset_config["dataset"]["name"]}_{offset+i+1}.yml")
        with open(output_path, 'w') as output_file:
            yaml.dump(config_copy, output_file)

def create_config_main():
    base_config_path = "config_template.yml"  # Path to your base config file
    dataset_name = "hospital"  # Directory to save generated config files

    base_dataset_config = {
        "dataset": {
            "name": dataset_name,
            "prediction_length": -1,
            "windows": 1
        },
        "finetuning":{
            "update_every_n_steps": 2,
            "validation": {
                "early_stopping": {
                    "min_patience": 100,
                    "max_patience": 200,
                },
                "validate_every_n_steps": 10,
                "ts_val_amount": 30
            },
            "checkpoint_to_save": "finetuned_model",
        }
    }

    # Define the parameter grid for hyperparameter search
    param_grid = {
        "time_limit": [2400],
        "learning_rate": [0.000005],
        "batch_size": [32, 64],
        "l2_sp_lambda": [0.0, 0.0001],
        "weight_decay": [0.0],
    }

    os.makedirs(dataset_name, exist_ok=True)
    create_configs(base_config_path, dataset_name, param_grid, base_dataset_config)

# wait for jobs to finish
# last sbatch submission: sbatch --time=0:15:00 --account my-account-abc --wait --dependency afterok:12345678:12345679 /dev/stdin <<< '#!/bin/bash \n sleep 1'

"""
    finetuning_json_report = {
        "dataset": {
            "name": dataset_attributes.name,
            "forecast_horizon": dataset_attributes.forecast_horizon,
            "ts_amount": dataset_attributes.ts_amount,
            "windows": dataset_attributes.windows,
        },
        "hyperparameters": finetuning_config,
        "finetuning_stats": {
            "total_time_spent": time.time() - st_time,
            "initial_validation_loss": step_results_over_time[0].validation_loss,
            "best_validation_loss": step_results_over_time[-1].best_validation_loss,
            "last_validation_loss": step_results_over_time[len(step_results_over_time) - 1].validation_loss,
            "total_steps": len(step_results_over_time),
            "best_step": np.argmin([x.validation_loss for x in step_results_over_time]),
            "early_stopping_reason": es_reason,
            "avg_time_per_step": (time.time() - st_time) / len(step_results_over_time),
            "avg_device_utilization": np.mean([step.device_utilization for step in step_results_over_time]),
            "training_loss": [step.training_loss for step in step_results_over_time],
            "validation_loss": [step.validation_loss for step in step_results_over_time],
        },
    }
"""

def create_csv_from_results(pids, result_folder, output_csv):
    import pandas as pd

    results = []
    for pid in pids:
        # Load results from a file named after the PID
        with open(f"{result_folder}/finetuning.{pid}/finetuning_report.json") as f:
            json_result = json.load(f)
            result = {
                "PID": pid,
                "Dataset": json_result["dataset"]["name"],
                "Forecast Horizon": json_result["dataset"]["forecast_horizon"],
                "TS Amount": json_result["dataset"]["ts_amount"],
                "Windows": json_result["dataset"]["windows"],
                "Time Limit": json_result["hyperparameters"]["time_limit"],
                "Learning Rate": json_result["hyperparameters"]["learning_rate"],
                "Batch Size": json_result["hyperparameters"]["batch_size"],
                "L2-SP Lambda": json_result["hyperparameters"]["l2_sp_lambda"],
                "Weight Decay": json_result["hyperparameters"]["weight_decay"],
                "Total Time Spent": json_result["finetuning_stats"]["total_time_spent"],
                "Initial Validation Loss": json_result["finetuning_stats"]["initial_validation_loss"],
                "Best Validation Loss": json_result["finetuning_stats"]["best_validation_loss"],
                "Last Validation Loss": json_result["finetuning_stats"]["last_validation_loss"],
                "Total Steps": json_result["finetuning_stats"]["total_steps"],
                "Best Step": json_result["finetuning_stats"]["best_step"],
                "Early Stopping Reason": json_result["finetuning_stats"]["early_stopping_reason"],
                "Avg Time Per Step": json_result["finetuning_stats"]["avg_time_per_step"],
                "Avg Device Utilization": json_result["finetuning_stats"]["avg_device_utilization"],
            }

            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="finetuning_graphs",
                        help="Path to a directory to save the output files")
    parser.add_argument("--job_ids", type=str, nargs='+', default=[],
                        help="Space-separated list of job IDs (PIDs) to include in the CSV report")
    args = parser.parse_args()

    print("test", args.job_ids, args.output_dir)

    #create_csv_from_results(args.job_ids, result_folder=args.output_dir, output_csv=f"{args.output_dir}/finetuning_summary.csv")
