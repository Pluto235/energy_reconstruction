#!/bin/bash
#SBATCH -J build_bkg_d
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

python -V
which python

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

STAGE_C_DIR=${STAGE_C_DIR:-apply/output/stage_c/current}
PSF_NPZ=${PSF_NPZ:-apply/output/stage_b/current/psf_v1.npz}
CELL_SELECTION=${CELL_SELECTION:-apply/config/cell_selection_v1.csv}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_d}
RUN_ID=${RUN_ID:-slurm_${SLURM_JOB_ID}}
MAX_BATCHES=${MAX_BATCHES:-}
NO_PLOTS=${NO_PLOTS:-}

echo "===== Start Stage D direct-integration background ====="
echo "Stage C dir: ${STAGE_C_DIR}"
echo "PSF NPZ: ${PSF_NPZ}"
echo "Cell selection: ${CELL_SELECTION}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Run id: ${RUN_ID}"
echo "Max batches: ${MAX_BATCHES:-<all>}"

CMD=(
  python apply/stages/04_background.py
  --stage-c-dir "${STAGE_C_DIR}"
  --psf-npz "${PSF_NPZ}"
  --cell-selection-csv "${CELL_SELECTION}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --time-bin-minutes 20
  --ha-min-deg -65
  --ha-max-deg 65
  --dec-min-deg -25
  --dec-max-deg 85
  --grid-step-deg 0.25
  --theta-max-deg 50
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --workers "${SLURM_CPUS_PER_TASK:-18}"
  --print-every 10
)

if [[ -n "${MAX_BATCHES}" ]]; then
  CMD+=(--max-batches "${MAX_BATCHES}")
fi

if [[ -n "${NO_PLOTS}" ]]; then
  CMD+=(--no-plots)
fi

"${CMD[@]}"

echo "Stage D direct-integration background finished"
