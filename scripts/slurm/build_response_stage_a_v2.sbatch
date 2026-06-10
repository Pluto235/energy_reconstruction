#!/bin/bash
#SBATCH -J build_resp_a_v2
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH -t 24:00:00

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

BINNED_ROOT=${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v2_raw65}
CELL_SELECTION=${CELL_SELECTION:-apply/config/cell_ledger_v2_raw65.csv}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_a_v2_raw65}
NUMERATOR_WORKERS=${NUMERATOR_WORKERS:-${SLURM_CPUS_PER_TASK:-18}}
DENOMINATOR_WORKERS=${DENOMINATOR_WORKERS:-1}

echo "===== Start Stage A v2 raw65 response ====="
echo "Binned root: ${BINNED_ROOT}"
echo "Cell selection: ${CELL_SELECTION}"
echo "Output dir: ${OUTPUT_DIR}"

python apply/stages/01_build_response.py \
  --binned-root "${BINNED_ROOT}" \
  --cell-selection-csv "${CELL_SELECTION}" \
  --output-dir "${OUTPUT_DIR}" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --denominator-workers "${DENOMINATOR_WORKERS}" \
  --numerator-workers "${NUMERATOR_WORKERS}" \
  --numerator-files-per-task 250 \
  --numerator-progress-every 20 \
  --npz-name response_2d_v2_raw65.npz \
  --metadata-name response_2d_v2_raw65_metadata.json

echo "Stage A v2 raw65 response finished"
