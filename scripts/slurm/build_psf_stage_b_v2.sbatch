#!/bin/bash
#SBATCH -J build_psf_b_v2
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
STAGE_A_METADATA=${STAGE_A_METADATA:-apply/output/stage_a_v2_raw65/response_2d_v2_raw65_metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_b_v2_raw65}
RUN_ID=${RUN_ID:-slurm_${SLURM_JOB_ID}}

echo "===== Start Stage B v2 raw65 PSF build ====="
echo "Binned root: ${BINNED_ROOT}"
echo "Cell selection: ${CELL_SELECTION}"
echo "Stage A metadata: ${STAGE_A_METADATA}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Run id: ${RUN_ID}"

python apply/stages/02_build_psf.py \
  --binned-root "${BINNED_ROOT}" \
  --cell-selection-csv "${CELL_SELECTION}" \
  --stage-a-metadata "${STAGE_A_METADATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --lhaaso-lat-deg 29.45 \
  --source-dec-deg 22.01 \
  --theta-min-deg 0.0 \
  --theta-max-deg 50.0 \
  --theta-step-deg 1.0 \
  --hour-angle-samples 200000 \
  --allow-incomplete-theta-support \
  --min-events-per-cell 1000 \
  --min-effective-events 200 \
  --core-fit-max-deg 3.0 \
  --theta-missing-mass-fail-threshold 0.10 \
  --containment-warning-tolerance 0.12 \
  --angle-check-max-events 20000 \
  --file-progress-every 5000 \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --npz-name psf_v2_raw65.npz \
  --metadata-name psf_v2_raw65_metadata.json \
  --summary-csv-name psf_v2_raw65_summary.csv \
  --summary-md-name psf_v2_raw65_summary.md

echo "Stage B v2 raw65 PSF build finished"
