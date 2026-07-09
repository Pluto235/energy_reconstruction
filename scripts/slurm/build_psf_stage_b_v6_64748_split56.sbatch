#!/bin/bash
#SBATCH -J psf_b_v6_64748
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

BINNED_ROOT=${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_split56_candidate}

python apply/stages/02_build_psf.py \
  --binned-root "${BINNED_ROOT}" \
  --cell-selection-csv apply/config/cell_ledger_v6_64748_split56_candidate.csv \
  --stage-a-metadata apply/output/stage_a_v6_64748_split56/response_2d_v6_64748_split56_metadata.json \
  --output-dir apply/output/stage_b_v6_64748_split56 \
  --run-id v6_64748_split56_stage_b_psf \
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
  --npz-name psf_v6_64748_split56.npz \
  --metadata-name psf_v6_64748_split56_metadata.json \
  --summary-csv-name psf_v6_64748_split56_summary.csv \
  --summary-md-name psf_v6_64748_split56_summary.md \
  --overwrite-run-dir
