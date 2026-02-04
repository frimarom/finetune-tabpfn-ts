#!/bin/bash -l
#SBATCH --partition=ext_vwl_norm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=tabpfn-ts-finetuning
#SBATCH --output=/work/smfrromb/sbatch_log/tabpfn.%j.out
#SBATCH --error=/work/smfrromb/sbatch_log/tabpfn.%j.err

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

export GIFT_EVAL=/work/smfrromb/finetune_tabpfn_ts/data

source /work/smfrromb/finetune_tabpfn_ts/.venv/bin/activate
cd /work/smfrromb || exit

DATASET="$1"
CHECKPOINT_NAME="$2"
TIME_LIMIT=$3
LEARNING_RATE=$4
BATCH_SIZE=$5
VAL_AMOUNT=$6

if [ "$#" -lt 6 ]; then
    echo "Usage:"
    echo "  sbatch run.sh <dataset> <checkpoint_name> <time_limit> <learning_rate> <batch_size> <time_series_val_amount>"
    exit 1
fi

srun -n1 -c8 \
    python -m finetune_tabpfn_ts.task_1.finetuning \
    --dataset "$DATASET" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --time_limit "$TIME_LIMIT" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --time_series_val_amount "$VAL_AMOUNT"

wait