#!/bin/bash
#SBATCH -J v6_empirical_final
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G
#SBATCH -t 24:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:-527af0380b1cf44de4a0c20d642174e210ca9485}

BASE=v6_64748_nhit100_reselect44_split56_miss030_empirical_cdf_asimov_ropt
SOURCE=v6_64748_nhit100_highEplus1_split56
DOUBLE_RAYLEIGH=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh
ITER0=${BASE}_iter0
ITER0_ROOT=apply/output/${BASE}_poisson_unbinned/iterations/iter0
ITER0_PSF_DIR=apply/output/stage_b_${ITER0}/runs/${ITER0}_stage_b_psf
ITER0_CACHE=${ITER0_PSF_DIR}/cumulative_response_scan_cache.npz
ITER0_BACKGROUND=${ITER0_ROOT}/stage_d/runs/stage_d/background_iter0.npz
ITER0_FIT_META=${ITER0_ROOT}/stage_f/runs/stage_f/fit_iter0_metadata.json
PSF_DIR=apply/output/stage_b_${BASE}/runs/${BASE}_stage_b_psf
PSF=${PSF_DIR}/psf_${BASE}.npz
PSF_META=${PSF_DIR}/psf_${BASE}_metadata.json
FULL_RESPONSE=apply/output/stage_a_${SOURCE}/response_2d_${SOURCE}.npz
BASE_PSF_DIR=apply/output/stage_b_${DOUBLE_RAYLEIGH}/runs/${DOUBLE_RAYLEIGH}_stage_b_psf
BASE_PSF=${BASE_PSF_DIR}/psf_${DOUBLE_RAYLEIGH}.npz
BASE_PSF_META=${BASE_PSF_DIR}/psf_${DOUBLE_RAYLEIGH}_metadata.json
RESPONSE_DIR=apply/output/stage_a_${BASE}_aperture_conditioned
RESPONSE=${RESPONSE_DIR}/response_2d_${BASE}_aperture_conditioned.npz
RESPONSE_META=${RESPONSE_DIR}/response_2d_${BASE}_aperture_conditioned_metadata.json
ROOT=apply/output/${BASE}_poisson_unbinned/final
D_ROOT=${ROOT}/stage_d
E_ROOT=${ROOT}/stage_e
F_ROOT=${ROOT}/stage_f
G_ROOT=${ROOT}/stage_g
D_RUN=${D_ROOT}/runs/stage_d
E_RUN=${E_ROOT}/runs/stage_e
F_RUN=${F_ROOT}/runs/stage_f
G_RUN=${G_ROOT}/runs/stage_g
STAGE_C=apply/output/stage_c_${SOURCE}/runs/${SOURCE}_stage_c_halfyear
SOURCE_TABLE=apply/config/cell_ledger_${SOURCE}_candidate.csv
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
MANIFEST=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate
RUN_DIR=runs/theta_recoxy_position_embed_noemax_no_core_cut_64748
PRIMARY_DENOM=/mnt/mydisk/WCDA_simulation_primary_hist/primary_denominator_stage_a.npz
REPORT=apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_empirical_cdf_asimov_ropt_poisson_unbinned_report.html
ASSETS=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-empirical-cdf-asimov-ropt

for path in "${ITER0_CACHE}" "${ITER0_BACKGROUND}" "${ITER0_FIT_META}" "${FULL_RESPONSE}" "${BASE_PSF}" "${BASE_PSF_META}" "${STAGE_C}/obs_events_metadata.json" "${SOURCE_TABLE}" "${SELECTOR}" "${MANIFEST}" "${PRIMARY_DENOM}"; do
  test -s "${path}"
done
test ! -e "${PSF_DIR}"
test ! -e "${RESPONSE_DIR}"
test ! -e "${ROOT}"
test ! -e "${REPORT}"
test ! -e "${ASSETS}"

python apply/stages/02e_optimize_empirical_cdf_asimov.py \
  --full-response-npz "${FULL_RESPONSE}" --binned-root "${BINNED_ROOT}" \
  --source-table "${SOURCE_TABLE}" --selector "${SELECTOR}" \
  --baseline-psf-npz "${BASE_PSF}" --baseline-psf-metadata "${BASE_PSF_META}" \
  --background-npz "${ITER0_BACKGROUND}" --spectrum-metadata "${ITER0_FIT_META}" \
  --output-dir "${PSF_DIR}" --run-id "${BASE}" --scan-cache-input "${ITER0_CACHE}" \
  --radius-min-deg 0.20 --radius-max-deg 2.00 --radius-step-deg 0.01 --near-max-fraction 0.99

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
  --npz-name background_final.npz --metadata-name background_final_metadata.json \
  --summary-csv-name background_final_summary.csv --summary-md-name background_final_summary.md

set +e
python apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C}" --background-npz "${D_RUN}/background_final.npz" \
  --background-metadata "${D_RUN}/background_final_metadata.json" --cell-selection-csv "${SELECTOR}" --included-only \
  --output-dir "${E_ROOT}" --run-id stage_e --no-promote-current \
  --report-html "${E_RUN}/stage_e_report.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --batch-size 500000 --quality-min-total-sigma 50 --quality-max-total-sigma 300 \
  --containment-override fixed1 --print-every 10 --no-plots \
  --npz-name signal_final.npz --metadata-name signal_final_metadata.json \
  --summary-csv-name signal_final_summary.csv --summary-md-name signal_final_summary.md
stage_e_status=$?
set -e
test -s "${E_RUN}/signal_final.npz" || exit "${stage_e_status}"

python apply/stages/06_fit.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_RUN}/signal_final.npz" --signal-metadata "${E_RUN}/signal_final_metadata.json" \
  --stage-c-dir "${STAGE_C}" --cell-subset-csv "${SELECTOR}" \
  --output-dir "${F_ROOT}" --run-id stage_f --no-promote-current \
  --report-html "${F_RUN}/stage_f_report.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --lhaaso-lat-deg 29.45 --lhaaso-lon-deg 100.14 --exposure-sample-step-sec 60 \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 --no-plots \
  --npz-name fit_final.npz --metadata-name fit_final_metadata.json \
  --summary-csv-name fit_final_summary.csv --summary-md-name fit_final_summary.md

included=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] == "1"))' "${SELECTOR}")
excluded=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] != "1"))' "${SELECTOR}")
python apply/stages/07_sed_points.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_RUN}/signal_final.npz" --signal-metadata "${E_RUN}/signal_final_metadata.json" \
  --stage-f-npz "${F_RUN}/fit_final.npz" --stage-f-metadata "${F_RUN}/fit_final_metadata.json" \
  --output-dir "${G_ROOT}" --run-id stage_g --no-promote-current \
  --report-html "${G_RUN}/stage_g_report.html" --baseline-name "${BASE}_poisson_unbinned" \
  --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 --skip-expected-stage-f-validation \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name sed_points_final.npz --metadata-name sed_points_final_metadata.json \
  --summary-csv-name sed_points_final_summary.csv --summary-json-name sed_points_final_summary.json \
  --summary-md-name sed_points_final_summary.md

mkdir -p "${ASSETS}"
python apply/report/replot_v6_display_cell_ids.py \
  --stage-d-npz "${D_RUN}/background_final.npz" \
  --stage-f-npz "${F_RUN}/fit_final.npz" --stage-f-metadata "${F_RUN}/fit_final_metadata.json" \
  --stage-d-output-dir "${D_RUN}" --stage-f-output-dir "${F_RUN}" --asset-dir "${ASSETS}"

python - "${E_RUN}/signal_final.npz" "${E_RUN}" <<'PY'
from importlib import import_module
from pathlib import Path
import sys
import numpy as np

signal_path, output_dir = map(Path, sys.argv[1:])
stage_e = import_module("apply.stages.05_signal")
with np.load(signal_path, allow_pickle=False) as handle:
    arrays = {name: np.asarray(handle[name]) for name in handle.files}
cells = [
    stage_e.CellSpec(index=index, cell_id=int(cell_id), nhit_bin=str(nhit), predE_bin=str(prede),
                     mc_count=0, selection_version="empirical_cdf_asimov", selection_reason="final")
    for index, (cell_id, nhit, prede) in enumerate(zip(arrays["cell_id"], arrays["nhit_bin"], arrays["predE_bin"]))
]
stage_e.plot_heatmap_grid(arrays["formal_sigma"], cells, output_dir / "formal_sigma_grid.png",
                          title="Stage E formal significance", colorbar_label="formal sigma",
                          cmap_name="RdBu_r", symmetric=True)
stage_e.plot_on_background_grid(arrays["N_on"], arrays["B_on"], cells, output_dir / "on_background_grid.png",
                                title="Stage E Crab on-region counts and unbinned background")
PY

IMPLEMENTATION_SHA=$(git rev-parse HEAD)
export V6_REPORT_RUN_ID="${BASE}"
export V6_REPORT_SOURCE_RUN_ID="${SOURCE}"
export V6_REPORT_TITLE="Crab SED v6 64748 reselect44 - Scheme R - Empirical-CDF Asimov-rOpt Poisson-Unbinned Background"
export V6_REPORT_SCHEME=R
export V6_REPORT_SCHEME_LABEL="Scheme R - Empirical-CDF Asimov-rOpt with Grid-Free Poisson Background"
export V6_REPORT_SCHEME_CONTRACT="The 44 selected cells use full cumulative MC response and the actual Stage D background to choose the smallest radius within 99% of maximum Asimov significance. The remaining diagnostic cells retain their prior apertures. Stage A is rebuilt with mc_dangle <= r_opt; Stage E containment is 1, so aperture efficiency enters the SED exactly once."
export V6_REPORT_PATH="${REPORT}"
export V6_REPORT_ASSET_DIR="${ASSETS}"
export V6_REPORT_EXPERIMENT_ID="${BASE}_poisson_unbinned"
export V6_REPORT_INPUT_COMMIT="${IMPLEMENTATION_SHA}"
export V6_REPORT_TRUE_ENERGY_GRID=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030/true-energy-cell-grid/v6_64748_reselect44_true_energy_cell_grid.png
export V6_REPORT_RESPONSE_META="${RESPONSE_META}"
export V6_REPORT_FIT_SELECTOR="${SELECTOR}"
export V6_REPORT_SELECTOR_META=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit_metadata.json
export V6_REPORT_PREFIT_SELECTOR=apply/config/cell_selector_v6_64748_nhit100_highEplus1_split56_prefit.csv
export V6_REPORT_HIGH_E_DECISIONS=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_highEplus1_decisions.csv
export V6_REPORT_STAGE_B_UNFILTERED_DIAGNOSTIC=apply/output/stage_b_v6_64748_nhit100_reselect44_split56_miss030/runs/v6_64748_nhit100_reselect44_split56_miss030_stage_b_psf/psf_v6_64748_nhit100_reselect44_split56_miss030_unfiltered_diagnostic.npz
export V6_REPORT_STAGE_D_RUN_DIR="${D_RUN}"
export V6_REPORT_STAGE_E_RUN_DIR="${E_RUN}"
export V6_REPORT_STAGE_F_RUN_DIR="${F_RUN}"
export V6_REPORT_STAGE_G_RUN_DIR="${G_RUN}"
export V6_REPORT_STAGE_D_STEM=background_final
export V6_REPORT_STAGE_E_STEM=signal_final
export V6_REPORT_STAGE_F_STEM=fit_final
export V6_REPORT_STAGE_G_STEM=sed_points_final
unset V6_REPORT_EXPECTED_FIXED_CONTAINMENT
export V6_REPORT_RATIO_HIDDEN_PREDE_POINTS=2
export V6_REPORT_SED_OVERLAY_TITLE="Crab SED - empirical-CDF Asimov rOpt"
export PIPELINE_JOB_IDS="opt0:${OPT0_JOB_ID:-n/a};iter0:${ITER0_JOB_ID:-n/a};final:${SLURM_JOB_ID:-n/a}"

python apply/report/build_v6_64748_nhit100_highEplus1_report.py
OVERLAY=${ASSETS}/${BASE}_stage_g_external_overlay_with_predE.png
test -s "${OVERLAY}"
python apply/report/prepare_v6_empirical_cdf_asimov_ropt_report.py \
  --report "${REPORT}" --asset-dir "${ASSETS}" \
  --iteration0-psf-metadata "${ITER0_PSF_DIR}/psf_${ITER0}_metadata.json" \
  --iteration0-summary-csv "${ITER0_PSF_DIR}/psf_${ITER0}_summary.csv" \
  --final-psf-npz "${PSF}" --final-psf-metadata "${PSF_META}" \
  --final-summary-csv "${PSF_DIR}/psf_${BASE}_summary.csv" \
  --response-npz "${RESPONSE}" --stage-d-npz "${D_RUN}/background_final.npz" \
  --stage-e-npz "${E_RUN}/signal_final.npz" --stage-f-metadata "${F_RUN}/fit_final_metadata.json" \
  --stage-g-npz "${G_RUN}/sed_points_final.npz" --stage-g-overlay "${OVERLAY}" \
  --implementation-sha "${IMPLEMENTATION_SHA}"

echo "Final report: ${REPORT}"
echo "Final Stage G overlay: ${OVERLAY}"

