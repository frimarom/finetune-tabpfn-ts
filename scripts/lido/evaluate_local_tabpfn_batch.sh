#!/bin/bash -l

#!/bin/bash -l
#SBATCH --partition=med
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=tabpfn-local-eval
#SBATCH --output=/work/smfrromb/tabpfn.%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
  srun --exclusive --nodes=1 \
    python /home/smfrromb/finetune-tabpfn-ts/task_1/evaluate_local_tabpfn_gift_eval.py \
    --dataset "$ds" &
done

wait