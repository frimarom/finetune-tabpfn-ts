# create different finetuning config for different datasets
# for the parameters that have to be adjusted fpr every config for hyperparameter search make parameter a list with all values to try and create config for every combination of those parameters
import yaml
import os
from itertools import product



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
        print(f"Created config file: {output_path}")

if __name__ == "__main__":
    base_config_path = "config_template.yml"  # Path to your base config file
    output_dir = "hospital"  # Directory to save generated config files

    base_dataset_config = {
        "dataset": {
            "name": 'hospital',
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
            "checkpoint_to_save": "test"
        }
    }

    # Define the parameter grid for hyperparameter search
    param_grid = {
        "time_limit": [1800],
        "learning_rate": [0.00005, 0.000005, 0.0000005],
        "batch_size": [16, 32, 64],
        "l2_sp_lambda": [0.0001],
        "weight_decay": [0.0],
    }

    os.makedirs(output_dir, exist_ok=True)
    create_configs(base_config_path, output_dir, param_grid, base_dataset_config)