#!/bin/bash
#SBATCH -J psf_b_v5_predbin
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

STRATEGY=${STRATEGY:?Set STRATEGY=gap025 or STRATEGY=gap1}
BINNED_ROOT=${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v5_predbin_${STRATEGY}}
CELL_LEDGER=${CELL_LEDGER:-apply/config/cell_ledger_v5_predbin_${STRATEGY}.csv}
STAGE_A_DIR=${STAGE_A_DIR:-apply/output/stage_a_v5_predbin_${STRATEGY}}
STAGE_A_METADATA=${STAGE_A_METADATA:-${STAGE_A_DIR}/response_2d_v5_predbin_${STRATEGY}_metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_b_v5_predbin_${STRATEGY}}
RUN_ID=${RUN_ID:-v5_predbin_${STRATEGY}_stage_b_slurm_${SLURM_JOB_ID}}
MAX_FILES_PER_CELL=${MAX_FILES_PER_CELL:-}

CMD=(
  python apply/stages/02_build_psf.py
  --binned-root "${BINNED_ROOT}"
  --cell-selection-csv "${CELL_LEDGER}"
  --stage-a-metadata "${STAGE_A_METADATA}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --tree-name t_eventout
  --weight-branch mc_weight
  --lhaaso-lat-deg 29.45
  --source-dec-deg 22.01
  --theta-min-deg 0.0
  --theta-max-deg 50.0
  --theta-step-deg 1.0
  --hour-angle-samples 200000
  --allow-incomplete-theta-support
  --allow-missing-cell-dirs
  --allow-low-stat-psf-fallback
  --min-events-per-cell 1000
  --min-effective-events 200
  --core-fit-max-deg 3.0
  --theta-missing-mass-fail-threshold 0.10
  --containment-warning-tolerance 0.12
  --angle-check-max-events 20000
  --file-progress-every 5000
  --workers "${SLURM_CPUS_PER_TASK:-18}"
  --npz-name psf_v5_predbin_${STRATEGY}.npz
  --metadata-name psf_v5_predbin_${STRATEGY}_metadata.json
  --summary-csv-name psf_v5_predbin_${STRATEGY}_summary.csv
  --summary-md-name psf_v5_predbin_${STRATEGY}_summary.md
)
if [[ -n "${MAX_FILES_PER_CELL}" ]]; then
  CMD+=(--max-files-per-cell "${MAX_FILES_PER_CELL}")
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
