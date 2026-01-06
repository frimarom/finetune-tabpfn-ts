#!/bin/bash -l
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=04:00:00
#SBATCH --job-name=tabpfn-local-eval
#SBATCH --output=/work/smfrromb/sbatch_log/tabpfn.%j.out
#SBATCH --error=/work/smfrromb/sbatch_log/tabpfn.%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /work/smfrromb/finetune_tabpfn_ts/.venv/bin/activate
cd /work/smfrromb || exit

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
  srun -n1 -c4 \
    python -m finetune_tabpfn_ts.task_1.evaluate_local_tabpfn_gift_eval \
    --dataset "$ds"
done
wait
