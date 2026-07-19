#!/bin/bash
#SBATCH -J v6_empirical_ropt_i0
#SBATCH -p main
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
PROJECT_ROOT=${PROJECT_ROOT:-/home/server/projects/energy_reconstruction}
cd "${PROJECT_ROOT}"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

BASE=v6_64748_nhit100_reselect44_split56_miss030_empirical_cdf_asimov_ropt
SOURCE=v6_64748_nhit100_highEplus1_split56
DOUBLE_RAYLEIGH=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh
ITER0=${BASE}_iter0
PSF_DIR=apply/output/stage_b_${ITER0}/runs/${ITER0}_stage_b_psf
FULL_RESPONSE=apply/output/stage_a_${SOURCE}/response_2d_${SOURCE}.npz
BASE_PSF_DIR=apply/output/stage_b_${DOUBLE_RAYLEIGH}/runs/${DOUBLE_RAYLEIGH}_stage_b_psf
BASE_PSF=${BASE_PSF_DIR}/psf_${DOUBLE_RAYLEIGH}.npz
BASE_PSF_META=${BASE_PSF_DIR}/psf_${DOUBLE_RAYLEIGH}_metadata.json
BASE_EXPERIMENT=${DOUBLE_RAYLEIGH}_scheme_R_fixed712979_poisson_unbinned
BASE_BRANCH=apply/output/${BASE_EXPERIMENT}/branches/h010_x0_y0
BASE_BACKGROUND=${BASE_BRANCH}/stage_d/runs/stage_d/background_h010_x0_y0.npz
BASE_FIT_META=${BASE_BRANCH}/stage_f/runs/stage_f/fit_h010_x0_y0_metadata.json
BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate
SOURCE_TABLE=apply/config/cell_ledger_${SOURCE}_candidate.csv
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv

for path in "${FULL_RESPONSE}" "${BASE_PSF}" "${BASE_PSF_META}" "${BASE_BACKGROUND}" "${BASE_FIT_META}" "${SOURCE_TABLE}" "${SELECTOR}"; do
  test -s "${path}"
done
test -d "${BINNED_ROOT}"
test ! -e "${PSF_DIR}"

python apply/stages/02e_optimize_empirical_cdf_asimov.py \
  --full-response-npz "${FULL_RESPONSE}" \
  --binned-root "${BINNED_ROOT}" \
  --source-table "${SOURCE_TABLE}" \
  --selector "${SELECTOR}" \
  --baseline-psf-npz "${BASE_PSF}" \
  --baseline-psf-metadata "${BASE_PSF_META}" \
  --background-npz "${BASE_BACKGROUND}" \
  --spectrum-metadata "${BASE_FIT_META}" \
  --output-dir "${PSF_DIR}" \
  --run-id "${ITER0}" \
  --scan-cache-output "${PSF_DIR}/cumulative_response_scan_cache.npz" \
  --radius-min-deg 0.20 --radius-max-deg 2.00 --radius-step-deg 0.01 \
  --near-max-fraction 0.99 --workers "${SLURM_CPUS_PER_TASK:-24}"

echo "Iteration-0 aperture optimization: ${PSF_DIR}"
