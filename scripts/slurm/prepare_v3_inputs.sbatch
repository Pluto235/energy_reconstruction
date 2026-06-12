#!/bin/bash
#SBATCH -J prep_v3_inputs
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

SOURCE_BINNED_ROOT=${SOURCE_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v1}
TARGET_BINNED_ROOT=${TARGET_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v3_candidate}
MAX_FILES_PER_SOURCE_BIN=${MAX_FILES_PER_SOURCE_BIN:-}

CMD=(
  python apply/stages/00_prepare_v3_inputs.py
  --source-binned-root "${SOURCE_BINNED_ROOT}"
  --target-binned-root "${TARGET_BINNED_ROOT}"
  --write-configs
  --write-diagnostics
  --prepare-cache
  --workers "${SLURM_CPUS_PER_TASK:-18}"
  --progress-every 500
)
if [[ -n "${MAX_FILES_PER_SOURCE_BIN}" ]]; then
  CMD+=(--max-files-per-source-bin "${MAX_FILES_PER_SOURCE_BIN}")
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
