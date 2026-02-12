from finetune_tabpfn_ts.task_1.load_datasets import load_dataset
from finetune_tabpfn_ts.task_1.load_datasets import get_transformed_stacked_dataset
from finetune_tabpfn_ts.task_1.load_datasets import transform_data
from finetune_tabpfn_ts.edits.finetune_tabpfn_main import fine_tune_tabpfn
from finetune_tabpfn_ts.task_1.dataset_utils import create_homgenous_ts_dataset
from finetune_tabpfn_ts.task_1.dataset_utils import DatasetAttributes
import argparse
import logging
import torch.cuda
import os

from gift_eval.data import Dataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use for fine-tuning")
    parser.add_argument("--checkpoint_name", type=str, default="finetune_tabpfn", help="Name of the checkpoint to save the fine-tuned model")
    parser.add_argument("--time_limit", type=int, default=600, help="Time limit for fine-tuning in seconds")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning")
    parser.add_argument("--time_series_val_amount", type=int, default=-1, help="Number of time series to use for validation")
    parser.add_argument("--update_every_n_steps", type=int, default=1, help="Number of steps to update the model before validation")
    parser.add_argument("--validate_every_n_steps", type=int, default=1, help="Number of steps to validate the model")
    parser.add_argument("--path_for_graphs" , type=str, default="finetuning_graphs", help="Path to save the training graphs")
    args = parser.parse_args()

    ds_name = args.dataset
    dataset = Dataset(ds_name)
    dataset_attributes = DatasetAttributes(name = dataset.name,
                                           time_series_length = len(next(iter(dataset.gluonts_dataset))["target"]), # TODO make variable
                                           ts_amount = len(dataset.gluonts_dataset),
                                           forecast_horizon = dataset.prediction_length,
                                           context_size = 0,
                                           offset = 0,
                                           windows = dataset.windows)
    print("Dataset:", ds_name)
    print("Prediction length:", dataset.prediction_length)
    print("Windows:", dataset.windows)
    train_X, train_y = create_homgenous_ts_dataset(ds_name, dataset_attributes.time_series_length)
    print("shapes", train_X.shape, train_y.shape)
    print(dataset_attributes.report_str)

    if args.path_for_graphs is not None and not os.path.exists(args.path_for_graphs):
        os.makedirs(args.path_for_graphs)

    fine_tune_tabpfn(
        path_to_base_model="./tabpfn-v2-regressor-2noar4o2.ckpt",
        save_path_to_fine_tuned_model=f"./{args.checkpoint_name}.ckpt",
        # Finetuning HPs
        time_limit=args.time_limit,
        finetuning_config={"learning_rate": args.learning_rate,
                           "batch_size": args.batch_size,
                           "min_patience": 20,
                           "max_patience": 100,
                           "validate_every_n_steps": args.validate_every_n_steps,
                           "update_every_n_steps": args.update_every_n_steps},
        validation_metric="mean_absolute_error",
        dataset_attributes = dataset_attributes,
        X_train=train_X,
        y_train=train_y,
        categorical_features_index=None,
        device="cuda" if torch.cuda.is_available() else "cpu",  # use "cpu" if you don't have a GPU
        task_type="regression",
        X_val = None, # hier keine Daten angeben
        y_val = None,
        val_time_series_amount = None if args.time_series_val_amount == -1 else args.time_series_val_amount,
        # Optional
        path_for_graphs = args.path_for_graphs,
        show_training_curve=True,  # Shows a final report after finetuning.
        logger_level=0,  # Shows all logs, higher values shows less
        use_wandb=False,  # Init wandb yourself, and set to True
    )