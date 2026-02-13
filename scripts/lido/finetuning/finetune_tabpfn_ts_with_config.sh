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

RESULT_DIR="finetuning_results/finetuning.${SLURM_JOB_ID}"
mkdir -p "${RESULT_DIR}"

CONFIG="$1"

if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  sbatch finetune_tabpfn_ts_with_config.sh <path_to_config>"
    exit 1
fi

srun -n1 -c8 \
    python -m finetune_tabpfn_ts.task_1.finetuning \
    --finetuning_config "${CONFIG}" \
    --path_to_save_all "finetuning.${SLURM_JOB_ID}"

wait