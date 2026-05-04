# Plan for evaluation
- for evaluation metric to plot we take one of the following metrics: NRMSE, mean_weighted_sum_quantile_loss, Normalized deviation
- evaluation not very costly -> evaluate as much as possible
## single finetune(finetuning on one dataset)
### What data
- only data from gift eval to stay in this system and look if it really improves performance with gift eval
- other datasets can be used but for now only stick to gift eval datasets
- preferably all datasets that have single variety
- also datasets with multiple varieties can be evaluated 
- maybe all datasets from gift eval(???)
- no m4 datasets but maybe finetune and look if the act like a prior 
### Validation
- validation also only on the same dataset
- take n validation time series -> gift eval provides with data so it does not overlap with train or test
- early stopping if performance does not improve(what value for early stopping? small/high?)
### How to
- use `summarize_finetuning_scripts.sh` to finetune on vast amount of hyperparameters
- take the checkpoint with the best configuration(maybe for later find pattern for hyperparameters)

### Evaluation
- evaluation before and after on the dataset we use to finetune
- evaluate on other datasets with similar structure(num observations per ts, frequency etc.)
- plot all results in graph(x: score before finetuning, y: score after finetuning)(diagonal)
- `sbatch evaluate_tabpfn_ts_local_one_ds.sh <dataset> finetuning_results/finetuning.<pid>/finetuned_model.ckpt`
## multi finetune(finetuning on multiple datasets)
### What data
- data from multiple datasets
- here maybe also datasets from autogluon/chronos-datasets
- for every batch the dataset from which is samples stays the same
### Validation
- take n time series from every dataset to validate(same principle as above with gift eval)
- how to calculate early stopping/track validation loss?
- Normalized error and add together and then mean
### How to
### Evaluation

## continued pretraining
### What data
- data generated from the prior
- different attributes maybe(?) and one for all attributes(context_length,forecast_horizon,etc.)
- for every batch prediction length, context length and frequency stays the same -> then changes
- different variants(only one frequency, multiple frequencies, one context length area etc.)
### Validation
- validate on multiple real world datasets
- 
### How to
### Evaluation