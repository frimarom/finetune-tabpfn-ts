#!/bin/bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <pid1> [pid2 ...]"
    exit 1
fi

DATASET="$1"
shift
PIDS=("$@")

SBATCH_SCRIPT="evaluate_tabpfn_ts_local_one_ds_vwl.sh"

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "SBATCH script not found: $SBATCH_SCRIPT"
    exit 1
fi

JOB_IDS=()

for pid in "${PIDS[@]}"; do
    CHECKPOINT_DIR="../../../../finetuning_results/finetuning.${pid}"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Skipping pid=$pid: directory not found ($CHECKPOINT_DIR)"
        continue
    fi

    shopt -s nullglob
    CHECKPOINTS=("$CHECKPOINT_DIR"/*.ckpt)
    shopt -u nullglob

    if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
        echo "Skipping pid=$pid: no checkpoints found in $CHECKPOINT_DIR"
        continue
    fi

    for checkpoint in "${CHECKPOINTS[@]}"; do
        sbatch_output=$(sbatch "$SBATCH_SCRIPT" "$DATASET" "$checkpoint")
        job_id=$(echo "$sbatch_output" | awk '{print $4}')
        JOB_IDS+=("$job_id")
        echo "dataset=$DATASET pid=$pid checkpoint=$checkpoint job_id=$job_id"
    done
done

echo
echo "All job IDs:"
printf '%s\n' "${JOB_IDS[@]}"