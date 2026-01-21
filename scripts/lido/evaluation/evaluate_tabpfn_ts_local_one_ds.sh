#!/bin/bash -l
#SBATCH --partition=gpu_med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=tabpfn-local-eval
#SBATCH --output=/work/smfrromb/sbatch_log/tabpfn.%j.out
#SBATCH --error=/work/smfrromb/sbatch_log/tabpfn.%j.err

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source /work/smfrromb/finetune_tabpfn_ts/.venv/bin/activate
cd /work/smfrromb || exit

DATASETS="$1"

if [ -z "$DATASETS" ]; then
    echo "Usage: sbatch run.sh <dataset>"
    exit 1
fi

srun -n1 -c8 \
    python -m finetune_tabpfn_ts.task_1.evaluate_local_tabpfn_gift_eval \
    --dataset "$DATASETS"

wait