## Start different project files
```
cd ..
python -m finetune_tabpfn_ts.task_1.<file_to_start>
```
## Download GIFT-Eval dataset folder
### Local
```
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir ./data
```
### On Cluster
```
python -m venv .venv
source .venv/bin/activate
pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Salesforce/GiftEval', repo_type='dataset', local_dir='./data', local_dir_use_symlinks=False)"
```