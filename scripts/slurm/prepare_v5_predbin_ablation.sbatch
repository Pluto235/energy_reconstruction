#!/bin/bash
#SBATCH -J prep_v5_predbin
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH -t 24:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

STRATEGY=${STRATEGY:-all}
SOURCE_BINNED_ROOT=${SOURCE_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v1}
TARGET_ROOT_TEMPLATE=${TARGET_ROOT_TEMPLATE:-/mnt/mydisk/WCDA_simulation_binned_response_v5_predbin_{strategy}}
MAX_FILES_PER_SOURCE_BIN=${MAX_FILES_PER_SOURCE_BIN:-}

CMD=(
  python apply/stages/00_prepare_v5_predbin_ablation.py
  --strategy "${STRATEGY}"
  --source-binned-root "${SOURCE_BINNED_ROOT}"
  --target-root-template "${TARGET_ROOT_TEMPLATE}"
  --write-configs
  --write-diagnostics
  --prepare-cache
  --overwrite-filtered-cache
  --overwrite-merged-cache
  --workers "${SLURM_CPUS_PER_TASK:-18}"
  --progress-every 500
)
if [[ -n "${MAX_FILES_PER_SOURCE_BIN}" ]]; then
  CMD+=(--max-files-per-source-bin "${MAX_FILES_PER_SOURCE_BIN}")
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
