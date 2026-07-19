#!/bin/bash
set -eo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/server/projects/energy_reconstruction}
cd "${PROJECT_ROOT}"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

BASE=v6_64748_nhit100_reselect44_split56_miss030_empirical_cdf_asimov_ropt
SOURCE=v6_64748_nhit100_highEplus1_split56
ITER0=${BASE}_iter0
ITER0_PSF_DIR=apply/output/stage_b_${ITER0}/runs/${ITER0}_stage_b_psf
PSF_DIR=apply/output/stage_b_${BASE}/runs/${BASE}_stage_b_psf
PSF=${PSF_DIR}/psf_${BASE}.npz
PSF_META=${PSF_DIR}/psf_${BASE}_metadata.json
RESPONSE_DIR=apply/output/stage_a_${BASE}_aperture_conditioned
RESPONSE=${RESPONSE_DIR}/response_2d_${BASE}_aperture_conditioned.npz
RESPONSE_META=${RESPONSE_DIR}/response_2d_${BASE}_aperture_conditioned_metadata.json
ROOT=apply/output/${BASE}_poisson_unbinned/final
D_RUN=${ROOT}/stage_d/runs/stage_d
E_RUN=${ROOT}/stage_e/runs/stage_e
F_RUN=${ROOT}/stage_f/runs/stage_f
G_RUN=${ROOT}/stage_g/runs/stage_g
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
REPORT=apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_empirical_cdf_asimov_ropt_poisson_unbinned_report.html
ASSETS=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-empirical-cdf-asimov-ropt

for path in \
  "${ITER0_PSF_DIR}/psf_${ITER0}_metadata.json" \
  "${ITER0_PSF_DIR}/psf_${ITER0}_summary.csv" \
  "${PSF}" "${PSF_META}" "${RESPONSE}" "${RESPONSE_META}" \
  "${D_RUN}/background_final.npz" "${E_RUN}/signal_final.npz" \
  "${F_RUN}/fit_final.npz" "${F_RUN}/fit_final_metadata.json" \
  "${G_RUN}/sed_points_final.npz" "${SELECTOR}"; do
  test -s "${path}"
done

mkdir -p "${ASSETS}"
python apply/report/replot_v6_display_cell_ids.py \
  --stage-d-npz "${D_RUN}/background_final.npz" \
  --stage-f-npz "${F_RUN}/fit_final.npz" --stage-f-metadata "${F_RUN}/fit_final_metadata.json" \
  --stage-d-output-dir "${D_RUN}" --stage-f-output-dir "${F_RUN}" --asset-dir "${ASSETS}"

python - "${F_RUN}/fit_final.npz" "${F_RUN}" <<'PY'
from importlib import import_module
from pathlib import Path
import sys
import numpy as np

fit_path, output_dir = map(Path, sys.argv[1:])
stage_f = import_module("apply.stages.06_fit")
with np.load(fit_path, allow_pickle=False) as handle:
    arrays = {name: np.asarray(handle[name]) for name in handle.files}
stage_f.plot_heatmap(
    arrays["pl_conservative_pull"], arrays, output_dir / "pull_grid_pl.png",
    title="Stage F PL conservative-error pulls", colorbar_label="pull",
)
stage_f.plot_heatmap(
    arrays["logpar_conservative_pull"], arrays, output_dir / "pull_grid_logpar.png",
    title="Stage F LogPar conservative-error pulls", colorbar_label="pull",
)
PY

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
export PIPELINE_JOB_IDS="opt0:${OPT0_JOB_ID:-65329};iter0:${ITER0_JOB_ID:-65334};final:${FINAL_JOB_ID:-65335};report:${SLURM_JOB_ID:-n/a}"

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

