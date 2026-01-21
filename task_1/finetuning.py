from finetune_tabpfn_ts.task_1.load_datasets import load_dataset
from finetune_tabpfn_ts.task_1.load_datasets import get_transformed_stacked_dataset
from finetune_tabpfn_ts.task_1.load_datasets import transform_data
from finetune_tabpfn_ts.edits.finetune_tabpfn_main import fine_tune_tabpfn
import torch
import random
import argparse
import logging

from gift_eval.data import Dataset

logger = logging.getLogger(__name__)

def remove_random_z_series(X, y):
    N = X.shape[-1]
    idx = random.randrange(N)

    X_val = X[..., idx]
    y_val = y[..., idx]

    mask = torch.ones(N, dtype=torch.bool)
    mask[idx] = False

    X_train = X[..., mask]
    y_train = y[..., mask]

    return X_train, y_train, X_val, y_val, idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use for fine-tuning")
    parser.add_argument("--checkpoint_name", type=str, default="finetune_tabpfn", help="Name of the checkpoint to save the fine-tuned model")
    parser.add_argument("--pred_length", type=int, default=-1, help="How many points should be predicted/How many points are hidden for training")
    parser.add_argument("--time_limit", type=int, default=600, help="Time limit for fine-tuning in seconds")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Command Line Arguments: {vars(args)}")

    ds_name = args.dataset
    dataset = Dataset(ds_name)
    print("Dataset:", ds_name)
    print("Prediction length:", dataset.prediction_length)
    train_X, train_y = get_transformed_stacked_dataset(ds_name)
    train_X, train_y, val_X, val_y, removed_idx = remove_random_z_series(train_X, train_y)

    fine_tune_tabpfn(
        path_to_base_model="./tabpfn-v2-regressor-2noar4o2.ckpt",
        save_path_to_fine_tuned_model=f"./{args.checkpoint_name}.ckpt",
        # Finetuning HPs
        time_limit=args.time_limit,
        finetuning_config={"learning_rate": 0.00001, "batch_size": 16},
        validation_metric="mean_absolute_error",
        X_train=train_X,
        y_train=train_y,
        categorical_features_index=None,
        device="cpu",  # use "cpu" if you don't have a GPU
        task_type="regression",
        X_val = val_X, # hier keine Daten angeben
        y_val = val_y,
        pred_length = dataset.prediction_length if args.pred_length < 0 else args.pred_length,
        # Optional
        show_training_curve=True,  # Shows a final report after finetuning.
        logger_level=0,  # Shows all logs, higher values shows less
        use_wandb=False,  # Init wandb yourself, and set to True
    )