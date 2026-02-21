# Explanation of SLURM on Lido Cluster

## Information
- Only available nodes for this project are from partition `gpu_short` and `gpu_med` since the other ones are too old

## Broad time estimation for evaluation jobs
- On `gpu_med` partition with 1 GPU and 8 CPUs:
- 1,5 it/s
- dataset-size/1024(batchsize) * 12 min = total time in min
    
### Cuda Version 11.8
```
module add nvidia/cuda/11.8
```
### Python Version 3.11.7
```
module add python/3.11.7-gcc114-base
```

## Helpful commands

### Interactive Slurm Test Scripts if script runs
```
srun --partition=short --nodes=1 --cpus-per-task=4 --gres=gpu:tesla:1 --time=00:10:00 --pty bash
```

### See status of own jobs
```
squeue -u $USER
```

### Cancel own job
```
scancel <job_id>
```

### Submit batch job
```
sbatch <script_name>.sh
```

### See Datasets data in job
```
cat tabpfn.<job-id>.err | grep "Dataset"
```

### Check nvidia driver on node
```
nvidia-smi
```

## Common Errors

Check the python version now and then create a conda environment without a specific python version
