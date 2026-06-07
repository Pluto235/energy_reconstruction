#!/bin/bash
#SBATCH -J build_psf_b
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
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

python -V
which python

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v1
CELL_SELECTION=apply/config/cell_selection_v1.csv
STAGE_A_METADATA=apply/output/stage_a/response_2d_metadata.json
OUTPUT_DIR=apply/output/stage_b
RUN_ID=slurm_${SLURM_JOB_ID}

echo "===== Start Stage B Crab PSF build ====="
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
  --workers "${SLURM_CPUS_PER_TASK:-18}"

echo "Stage B Crab PSF build finished"
