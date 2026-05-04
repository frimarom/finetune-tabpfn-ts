# This repository contains all code and results from the bachelor thesis: "Continued Pretraining of TabPFN with time series data"
## Code
- Code is available under `task_1`, `edits` and `prior`
- Evaluation code is available under `evaluation`
## Results
- all results for all experiments are available under `documents/plan/results`

## Utils
### Start different project files
```
cd ..
python -m finetune_tabpfn_ts.task_1.<file_to_start>
```
### Start Batch Scripts on Cluster
#### `finetune_tabpfn_ts_one_ds.sh` 
```
sbatch finetune_tabpfn_ts_one_ds.sh <dataset_name> <checkpoint_name> <pred_length> <time_limit> <learning_rate> <batch_size> <debug>
```
- if pred_length is -1, it will use the default prediction length of the dataset

### Download GIFT-Eval dataset folder
#### Local
```
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir ./data
```
#### On Cluster
```
python -m venv .venv
source .venv/bin/activate
pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Salesforce/GiftEval', repo_type='dataset', local_dir='./data', local_dir_use_symlinks=False)"
```