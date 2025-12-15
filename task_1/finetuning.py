import load_datasets

import edits
import load_datasets
from edits.finetune_tabpfn_main import fine_tune_tabpfn

if __name__ == "__main__":
    dataset = load_datasets.load_dataset("m4_hourly")
    record = load_datasets.to_timeseries_dataframe_list(dataset, maxdata=2)

    train = record[0]["train"]
    test = record[0]["test"]
    print("test", test)

    test_y = test["target"]
    test["target"] = None
    # test = test.to_frame(name="target").reset_index()
    # test = test.drop('target', axis=1)

    train_data, test_data = load_datasets.transform_data(train, test)
    train_y = train_data["target"]
    train_X = train_data.drop("target", axis=1)
    test_X = test_data.drop("target", axis=1)


    fine_tune_tabpfn(
        path_to_base_model="./tabpfn-v2-regressor-2noar4o2.ckpt",
        save_path_to_fine_tuned_model="./fine_tuned_model_timeseries.ckpt",
        # Finetuning HPs
        time_limit=600,
        finetuning_config={"learning_rate": 0.00001, "batch_size": 16},
        validation_metric="mean_absolute_error",
        X_train=train_X,
        y_train=train_y,
        categorical_features_index=None,
        device="cpu",  # use "cpu" if you don't have a GPU
        task_type="regression",
        X_val = test_X, # hier keine Daten angeben
        y_val = test_y,
        # Optional
        show_training_curve=True,  # Shows a final report after finetuning.
        logger_level=0,  # Shows all logs, higher values shows less
        use_wandb=False,  # Init wandb yourself, and set to True
    )