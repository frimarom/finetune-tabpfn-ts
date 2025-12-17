#!/bin/bash -l
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --job-name=tabpfn-local-eval
#SBATCH --output=/work/smfrromb/tabpfn.%j.out
#SBATCH --error=/work/smfrromb/tabpfn.%j_err.out

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DATASETS=(
  m4_yearly
  m4_quarterly
  solar/W
  hospital
  covid_deaths
  bizitobs_l2c/H
  SZ_TAXI/15T
  M_DENSE/H
  ett1/15T
)

for ds in "${DATASETS[@]}"; do
  srun --exclusive -n1 -c1 --gres=gpu:tesla:1\
    python /home/smfrromb/finetune-tabpfn-ts/task_1/evaluate_local_tabpfn_gift_eval.py \
    --dataset "$ds" &
done
wait