#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=tabpfn-ts-finetuning
#SBATCH --output=/work/smfrromb/sbatch_log/tabpfn.%j.out
#SBATCH --error=/work/smfrromb/sbatch_log/tabpfn.%j.err

erster="$1"
shift

declare -a JOB_IDS=()

for i in "$@"; do
    echo "$i"
    JOB_ID=$(sbatch ./finetune_tabpfn_ts_with_config.sh "finetune_tabpfn_ts/finetuning_configs/$erster/${erster}_${i}.yml" | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")
done

echo "Gestartete Job-IDs: ${JOB_IDS[*]}"

while true; do
    # Alle noch laufenden/ausstehenden Job-IDs abfragen
    ACTIVE=$(squeue -j "$(IFS=','; echo "${JOB_IDS[*]}")" -h 2>/dev/null | wc -l)

    if [[ "$ACTIVE" -eq 0 ]]; then
        break
    fi

    printf "\rNoch aktiv: %d / %d Jobs..." "$ACTIVE" "${#JOB_IDS[@]}"
    sleep 15
done

cd /work/smfrromb/finetune_tabpfn_ts/finetuning_configs || exit

RESULT_DIR="../../finetuning_results/finetuning.${erster}.$(date '+%Y%m%d_%H%M%S')"
mkdir -p "${RESULT_DIR}"

python -m config_util --output_dir "${RESULT_DIR}" --job_ids "${JOB_IDS[@]}"

mkdir -p "${RESULT_DIR}/graphs"

for i in "${JOB_IDS[@]}"; do
  cp "../../finetuning_results/finetuning.${i}/fine_tuning_loss_plot_*" "${RESULT_DIR}/graphs/" 2>/dev/null || echo "Keine Graphen für Job $i"
done