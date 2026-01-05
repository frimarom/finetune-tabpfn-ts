# Explanation of SLURM on Lido Cluster

## Information

### Cuda Version 11.8
```
module add nvidia/cuda/11.8
```
### Python Version 3.11.7
```
module add 
```

## Helpful commands

### Interactive Slurm Test Scripts if script runs
```
srun --partition=short --nodes=1 --cpus-per-task=4 --gres=gpu:tesla:1 --time=00:10:00 --pty bash
```

### Create Conda environment
```
conda create -n finetune-tabpfn-ts -y
```

### Move conda dirs in shared file folder for to be accessed on nodes
```
export CONDA_PKGS_DIRS=/work/$USER/conda_env/conda_pkgs
export CONDA_ENVS_DIRS=/work/$USER/conda_env/conda_envs
mkdir -p $CONDA_PKGS_DIRS $CONDA_ENVS_DIRS

rm -rf ~/.conda/pkgs/*
```

### Conda reset
```
mv ~/.condarc ~/.condarc.bak
unset CONDA_PKGS_DIRS
unset CONDA_SOLVER
unset CONDA_CHANNELS
unset CONDA_SUBDIR
unset CONDA_OVERRIDE_CUDA
exit
ssh lido-cluster
```

### Check nvidia driver on node
```
nvidia-smi
```

## Common Errors

### Checksum Mismatch 
When an error like this occurs while creating a conda environment with a designated python version
```
ChecksumMismatchError: Conda detected a mismatch between the expected content and downloaded content                                                                                    
for url 'https://repo.anaconda.com/pkgs/main/linux-64/python-3.11.5-h955ad1f_0.conda'.                                                                                                  
  download saved to: /work/smfrromb/conda_env/conda_pkgs/python-3.11.5-h955ad1f_0.conda                                                                                                 
  expected sha256: 47c0168daefabdd932f41ef3ed92af467b03843864b85177e313f26c93bf1000                                                                                                     
  actual sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855                                                                                                       
                                                                                                                                                                                        
ChecksumMismatchError: Conda detected a mismatch between the expected content and downloaded content                                                                                    
for url 'https://repo.anaconda.com/pkgs/main/linux-64/python-3.11.5-h955ad1f_0.conda'.                                                                                                  
  download saved to: /work/smfrromb/conda_env/conda_pkgs/python-3.11.5-h955ad1f_0.conda                                                                                                 
  expected sha256: 47c0168daefabdd932f41ef3ed92af467b03843864b85177e313f26c93bf1000                                                                                                     
  actual sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 
```
you probably use the wrong python version. First do
```
module avail python
```
and then add the right python module
```
module add <python-module>
```
Check the python version now and then create a conda environment without a specific python version
