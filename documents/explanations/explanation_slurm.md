# Explanation of SLURM on Lido Cluster

## Information

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

### Check nvidia driver on node
```
nvidia-smi
```

## Common Errors

Check the python version now and then create a conda environment without a specific python version
