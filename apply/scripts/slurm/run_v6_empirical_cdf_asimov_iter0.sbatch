#!/bin/bash
#SBATCH -J v6_empirical_pipe_i0
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH -t 24:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
PROJECT_ROOT=${PROJECT_ROOT:-/home/server/projects/energy_reconstruction}
cd "${PROJECT_ROOT}"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:-527af0380b1cf44de4a0c20d642174e210ca9485}

BASE=v6_64748_nhit100_reselect44_split56_miss030_empirical_cdf_asimov_ropt
SOURCE=v6_64748_nhit100_highEplus1_split56
ITER0=${BASE}_iter0
PSF_DIR=apply/output/stage_b_${ITER0}/runs/${ITER0}_stage_b_psf
PSF=${PSF_DIR}/psf_${ITER0}.npz
RESPONSE_DIR=apply/output/stage_a_${ITER0}_aperture_conditioned
RESPONSE=${RESPONSE_DIR}/response_2d_${ITER0}_aperture_conditioned.npz
RESPONSE_META=${RESPONSE_DIR}/response_2d_${ITER0}_aperture_conditioned_metadata.json
ROOT=apply/output/${BASE}_poisson_unbinned/iterations/iter0
D_ROOT=${ROOT}/stage_d
E_ROOT=${ROOT}/stage_e
F_ROOT=${ROOT}/stage_f
D_RUN=${D_ROOT}/runs/stage_d
E_RUN=${E_ROOT}/runs/stage_e
F_RUN=${F_ROOT}/runs/stage_f
STAGE_C=apply/output/stage_c_${SOURCE}/runs/${SOURCE}_stage_c_halfyear
SOURCE_TABLE=apply/config/cell_ledger_${SOURCE}_candidate.csv
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
MANIFEST=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate
RUN_DIR=runs/theta_recoxy_position_embed_noemax_no_core_cut_64748
PRIMARY_DENOM=/mnt/mydisk/WCDA_simulation_primary_hist/primary_denominator_stage_a.npz

for path in "${PSF}" "${STAGE_C}/obs_events_metadata.json" "${SOURCE_TABLE}" "${SELECTOR}" "${MANIFEST}" "${PRIMARY_DENOM}"; do
  test -s "${path}"
done
test ! -e "${RESPONSE_DIR}"
test ! -e "${ROOT}"

python apply/stages/01_build_response.py \
  --binned-root "${BINNED_ROOT}" --cell-selection-csv "${SOURCE_TABLE}" \
  --run-dir "${RUN_DIR}" --primary-denominator-npz "${PRIMARY_DENOM}" \
  --output-dir "${RESPONSE_DIR}" --aperture-psf-npz "${PSF}" \
  --numerator-workers "${SLURM_CPUS_PER_TASK:-32}" --numerator-files-per-task 250 \
  --npz-name "$(basename "${RESPONSE}")" --metadata-name "$(basename "${RESPONSE_META}")"

python apply/stages/04_background.py \
  --stage-c-dir "${STAGE_C}" --psf-npz "${PSF}" --cell-selection-csv "${SELECTOR}" \
  --output-dir "${D_ROOT}" --run-id stage_d --no-promote-current \
  --background-mode crab_roi_local --roi-background-method annulus-quadratic \
  --roi-fit-statistic poisson --pooling-manifest "${MANIFEST}" --poisson-quadratic-fit unbinned \
  --allow-annulus-count-mismatch --roi-fiducial-deg 6 --roi-edge-diagnostic-deg 8 \
  --roi-grid-step-deg 0.10 --roi-grid-offset-x-fraction 0 --roi-grid-offset-y-fraction 0 \
  --on-aperture-integration analytic-quadratic --annulus-normalize-surface \
  --annulus-default-inner-deg 1.5 --annulus-width-deg 2.0 --annulus-max-inner-deg 4.5 \
  --batch-size 500000 --workers "${SLURM_CPUS_PER_TASK:-32}" --print-every 10 --no-plots \
  --npz-name background_iter0.npz --metadata-name background_iter0_metadata.json \
  --summary-csv-name background_iter0_summary.csv --summary-md-name background_iter0_summary.md

set +e
python apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C}" --background-npz "${D_RUN}/background_iter0.npz" \
  --background-metadata "${D_RUN}/background_iter0_metadata.json" --cell-selection-csv "${SELECTOR}" --included-only \
  --output-dir "${E_ROOT}" --run-id stage_e --no-promote-current \
  --report-html "${E_RUN}/stage_e_report.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --batch-size 500000 --quality-min-total-sigma 50 --quality-max-total-sigma 300 \
  --containment-override fixed1 --print-every 10 --no-plots \
  --npz-name signal_iter0.npz --metadata-name signal_iter0_metadata.json \
  --summary-csv-name signal_iter0_summary.csv --summary-md-name signal_iter0_summary.md
stage_e_status=$?
set -e
test -s "${E_RUN}/signal_iter0.npz" || exit "${stage_e_status}"

python apply/stages/06_fit.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_RUN}/signal_iter0.npz" --signal-metadata "${E_RUN}/signal_iter0_metadata.json" \
  --stage-c-dir "${STAGE_C}" --cell-subset-csv "${SELECTOR}" \
  --output-dir "${F_ROOT}" --run-id stage_f --no-promote-current \
  --report-html "${F_RUN}/stage_f_report.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --lhaaso-lat-deg 29.45 --lhaaso-lon-deg 100.14 --exposure-sample-step-sec 60 \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 --no-plots \
  --npz-name fit_iter0.npz --metadata-name fit_iter0_metadata.json \
  --summary-csv-name fit_iter0_summary.csv --summary-md-name fit_iter0_summary.md

echo "Iteration-0 Stage A-F pipeline: ${ROOT}"
