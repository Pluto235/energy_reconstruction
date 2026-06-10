#!/bin/bash
#SBATCH -J prep_v2_inputs
#SBATCH -p main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 04:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

echo "==== Job Info ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "=================="

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

SOURCE_BINNED_ROOT=${SOURCE_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v1}
TARGET_BINNED_ROOT=${TARGET_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v2_raw65}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
OVERWRITE_HIGH_CACHE=${OVERWRITE_HIGH_CACHE:-}

CMD=(
  python apply/stages/00_prepare_v2_inputs.py
  --write-configs
  --prepare-cache
  --source-binned-root "${SOURCE_BINNED_ROOT}"
  --target-binned-root "${TARGET_BINNED_ROOT}"
  --workers "${WORKERS}"
  --progress-every 500
)

if [[ -n "${OVERWRITE_HIGH_CACHE}" ]]; then
  CMD+=(--overwrite-high-cache)
fi

echo "===== Start v2 input preparation ====="
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo "v2 input preparation finished"
