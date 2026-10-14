#!/bin/bash
#SBATCH -J v6_202201_psf
#SBATCH -p main
#SBATCH --qos=default-safe
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH -e /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export PYTHONPATH=/mnt/newdisk/shijie/energy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/newdisk/shijie/conda/envs/py310/bin/python
TAG=v6_64748_202201_scheme_R_double_rayleigh_analytic_bon
BASE=v6_64748_nhit100_reselect44_split56_miss030
SOURCE=v6_64748_nhit100_highEplus1_split56
APPLY=/mnt/newdisk/shijie/energy/apply
SOURCE_DIR="${APPLY}/output/stage_b_${BASE}/runs/${BASE}_stage_b_psf"
# Saved-profile mode returns before opening this historical documentary MC path.
"$PY" stages/02e_build_psf_v5_compare.py \
  --psf-method double_rayleigh_mixture \
  --binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate \
  --cell-selection-csv "${APPLY}/config/cell_ledger_${SOURCE}_candidate.csv" \
  --stage-a-metadata "${APPLY}/output/stage_a_${SOURCE}/response_2d_${SOURCE}_metadata.json" \
  --double-rayleigh-stage-b-source-npz "${SOURCE_DIR}/psf_${BASE}.npz" \
  --double-rayleigh-stage-b-source-metadata "${SOURCE_DIR}/psf_${BASE}_metadata.json" \
  --output-dir "${APPLY}/output/$TAG/psf" --run-id "${TAG}_stage_b_psf" \
  --containment-warning-tolerance 0.12 --no-borrow-v3-fallback-psf --no-promote-current \
  --npz-name "psf_${TAG}.npz" --metadata-name "psf_${TAG}_metadata.json" \
  --summary-csv-name "psf_${TAG}_summary.csv" --summary-md-name "psf_${TAG}_summary.md"
