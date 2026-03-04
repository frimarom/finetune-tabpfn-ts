#!/bin/bash -l
#SBATCH --partition=ext_chem2_norm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --job-name=tabpfn-local-eval
#SBATCH --output=/work/smfrromb/sbatch_log/tabpfn.%j.out
#SBATCH --error=/work/smfrromb/sbatch_log/tabpfn.%j.err

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

source /work/smfrromb/finetune_tabpfn_ts/.venv/bin/activate
cd /work/smfrromb || exit

DATASETS="$1"
PATH_TO_MODEL_CHECKPOINT="$2"

if [ -z "$DATASETS" ]; then
    echo "Usage: sbatch run.sh <dataset>"
    exit 1
fi

ARGS=(python -m finetune_tabpfn_ts.evaluation.evaluate_local_tabpfn \
  --dataset "$DATASETS" \
  --model_name tabpfn_ts_local \
  --mode local)
if [ -n "$PATH_TO_MODEL_CHECKPOINT" ]; then
  ARGS+=(--path_to_model_checkpoint "$PATH_TO_MODEL_CHECKPOINT")
fi
srun -n1 -c8 "${ARGS[@]}"

wait