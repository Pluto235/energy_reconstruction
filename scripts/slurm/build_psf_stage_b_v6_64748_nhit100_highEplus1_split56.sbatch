#!/bin/bash
#SBATCH -J psf_b_v6_64748_n100_hi1
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 06:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

RUN_ID=v6_64748_nhit100_highEplus1_split56
BINNED_ROOT=${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate}
CELL_SELECTION=${CELL_SELECTION:-apply/config/cell_ledger_${RUN_ID}_candidate.csv}
STAGE_A_METADATA=${STAGE_A_METADATA:-apply/output/stage_a_${RUN_ID}/response_2d_${RUN_ID}_metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_b_${RUN_ID}}
RUN_STAGE_B=${RUN_STAGE_B:-${RUN_ID}_stage_b_psf}

python apply/stages/02_build_psf.py \
  --binned-root "${BINNED_ROOT}" \
  --cell-selection-csv "${CELL_SELECTION}" \
  --stage-a-metadata "${STAGE_A_METADATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_STAGE_B}" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --lhaaso-lat-deg 29.45 \
  --source-dec-deg 22.01 \
  --theta-min-deg 0.0 \
  --theta-max-deg 50.0 \
  --theta-step-deg 1.0 \
  --hour-angle-samples 200000 \
  --allow-missing-cell-dirs \
  --allow-incomplete-theta-support \
  --allow-low-stat-psf-fallback \
  --min-events-per-cell 1000 \
  --min-effective-events 200 \
  --core-fit-max-deg 3.0 \
  --theta-missing-mass-fail-threshold 0.10 \
  --containment-warning-tolerance 0.12 \
  --angle-check-max-events 20000 \
  --file-progress-every 5000 \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --npz-name "psf_${RUN_ID}.npz" \
  --metadata-name "psf_${RUN_ID}_metadata.json" \
  --summary-csv-name "psf_${RUN_ID}_summary.csv" \
  --summary-md-name "psf_${RUN_ID}_summary.md" \
  --overwrite-run-dir
