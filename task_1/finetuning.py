from finetune_tabpfn_ts.task_1.load_datasets import load_dataset
from finetune_tabpfn_ts.task_1.load_datasets import get_transformed_stacked_dataset
from finetune_tabpfn_ts.task_1.load_datasets import transform_data
from finetune_tabpfn_ts.task_1.load_datasets import to_x_y
from finetune_tabpfn_ts.task_1.load_datasets import transform_data
from finetune_tabpfn_ts.task_1.load_datasets import stack_records_along_z
from finetune_tabpfn_ts.edits.finetune_tabpfn_main import fine_tune_tabpfn
from finetune_tabpfn_ts.task_1.dataset_utils import create_homgenous_ts_dataset
from finetune_tabpfn_ts.task_1.dataset_utils import DatasetAttributes
import argparse
import logging
import torch.cuda
import os
from gift_eval.data import Dataset
import yaml

logger = logging.getLogger(__name__)

def ensure_correct_config(config: dict):
    required_keys = ["dataset",
                     "learning_rate",
                     "batch_size",
                     "l2_sp_lambda",
                     "weight_decay",
                     "min_patience",
                     "max_patience",
                     "validate_every_n_steps",
                     "ts_val_amount"
                     "update_every_n_steps"
                     "checkpoint_to_save",
                     "path_to_save_all"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required fine-tuning config key: {key}")

    if config["l2_sp_lambda"] != 0.0 and config["weight_decay"] != 0.0:
        raise ValueError("Both l2_sp_lambda and weight_decay cannot be non-zero at the same time.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuning_config", type=str, default=None, help="Path to a yml file containing the fine-tuning config")
    parser.add_argument("--path_to_save_all" , type=str, default="finetuning_graphs", help="Path to save the training graphs")
    """
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use for fine-tuning")
    parser.add_argument("--checkpoint_name", type=str, default="finetune_tabpfn", help="Name of the checkpoint to save the fine-tuned model")
    parser.add_argument("--time_limit", type=int, default=600, help="Time limit for fine-tuning in seconds")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning")
    parser.add_argument("--time_series_val_amount", type=int, default=-1, help="Number of time series to use for validation")
    parser.add_argument("--update_every_n_steps", type=int, default=1, help="Number of steps to update the model before validation")
    parser.add_argument("--validate_every_n_steps", type=int, default=1, help="Number of steps to validate the model")
    """ # TODO add condition wether to use config file or command line arguments e.g. for scripts
    args = parser.parse_args()

    with open(args.finetuning_config) as config_file:
        finetuning_config = yaml.safe_load(config_file)

    #ensure_correct_config(finetuning_config)

    dataset = Dataset(finetuning_config["dataset"]["name"]) # TODO make for every type of dataset
    dataset_attributes = DatasetAttributes(name = finetuning_config["dataset"]["name"],
                                           time_series_length = len(next(iter(dataset.gluonts_dataset))["target"]), # TODO make variable
                                           ts_amount = len(dataset.gluonts_dataset),
                                           forecast_horizon = dataset.prediction_length
                                           if finetuning_config["dataset"]["prediction_length"] == -1
                                           else finetuning_config["dataset"]["prediction_length"],
                                           context_size = 0,
                                           offset = 0,
                                           windows = finetuning_config["dataset"]["windows"],)

    train_X, train_y = create_homgenous_ts_dataset(dataset_attributes.name, dataset_attributes.time_series_length)
    print("shapes", train_X.shape, train_y.shape)
    print(dataset_attributes.report_str)

    if args.path_to_save_all is not None and not os.path.exists(args.path_to_save_all):
        os.makedirs(args.path_to_save_all)

    fine_tune_tabpfn(
        path_to_base_model="./tabpfn-v2-regressor-2noar4o2.ckpt",
        save_path_to_fine_tuned_model=f"./{args.path_to_save_all}/{finetuning_config['finetuning']['checkpoint_to_save']}.ckpt",
        # Finetuning HPs
        time_limit=finetuning_config["finetuning"]["time_limit"],
        finetuning_config={"learning_rate": finetuning_config["finetuning"]["learning_rate"],
                           "batch_size": finetuning_config["finetuning"]["batch_size"],
                           "min_patience": finetuning_config["finetuning"]["validation"]["early_stopping"]["min_patience"],
                           "max_patience": finetuning_config["finetuning"]["validation"]["early_stopping"]["max_patience"],
                           "validate_every_n_steps": finetuning_config["finetuning"]["validation"]["validate_every_n_steps"],
                           "update_every_n_steps": finetuning_config["finetuning"]["update_every_n_steps"],
                           "l2_sp_lambda": finetuning_config["finetuning"]["l2_sp_lambda"],
                           },
        validation_metric="mean_absolute_error",
        dataset_attributes = dataset_attributes,
        X_train=train_X,
        y_train=train_y,
        categorical_features_index=None,
        device="cuda" if torch.cuda.is_available() else "cpu",  # use "cpu" if you don't have a GPU
        task_type="regression",
        X_val = None, # hier keine Daten angeben
        y_val = None,
        val_time_series_amount = None if finetuning_config["finetuning"]["validation"]["ts_val_amount"] == -1
        else finetuning_config["finetuning"]["validation"]["ts_val_amount"],
        # Optional
        path_for_all = args.path_to_save_all,
        show_training_curve=True,  # Shows a final report after finetuning.
        logger_level=0,  # Shows all logs, higher values shows less
        use_wandb=False,  # Init wandb yourself, and set to True
    )