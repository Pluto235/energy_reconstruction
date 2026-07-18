#!/bin/bash
#SBATCH -J v6_64748_unbinned_report
#SBATCH -p main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

EXPERIMENT=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_unbinned
POOLED_EXPERIMENT=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled
RUN_ID=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh
BRANCH=h010_x0_y0
ROOT=apply/output/${EXPERIMENT}/branches/${BRANCH}
POOLED_ROOT=apply/output/${POOLED_EXPERIMENT}/branches/${BRANCH}
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
STAGE_C=apply/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear
RESPONSE_ROOT=apply/output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh_annnorm
RESPONSE=${RESPONSE_ROOT}/response_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh.npz
RESPONSE_META=${RESPONSE_ROOT}/response_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh_metadata.json
D_RUN=${ROOT}/stage_d/runs/stage_d
E_RUN=${ROOT}/stage_e/runs/stage_e
F_RUN=${ROOT}/stage_f/runs/stage_f
G_ROOT=${ROOT}/stage_g
G_RUN=${G_ROOT}/runs/stage_g
POOLED_D=${POOLED_ROOT}/stage_d/runs/stage_d/background_${BRANCH}.npz
POOLED_F_META=${POOLED_ROOT}/stage_f/runs/stage_f/fit_${BRANCH}_metadata.json
POOLED_G=${POOLED_ROOT}/stage_g/runs/stage_g/sed_points_${BRANCH}.npz
D_STEM=background_${BRANCH}
E_STEM=signal_${BRANCH}
F_STEM=fit_${BRANCH}
G_STEM=sed_points_${BRANCH}
REPORT=apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_poisson_unbinned_report.html
REPORT_TMP=apply/report/.crab_sed_v6_64748_unbinned_full_report.tmp.html
ASSETS=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-unbinned
COV_NPZ=scheme_R_double_rayleigh_poisson_unbinned_background_covariance.npz
COV_JSON=scheme_R_double_rayleigh_poisson_unbinned_background_covariance.json
TIME_SPLIT=scheme_R_double_rayleigh_poisson_unbinned_time_split.json
IMPLEMENTATION_SHA=018d5568994c6406299b74a945df0f6dca9f7f15

for path in \
  "${D_RUN}/${D_STEM}.npz" "${E_RUN}/${E_STEM}.npz" \
  "${F_RUN}/${F_STEM}.npz" "${F_RUN}/${F_STEM}_metadata.json" \
  "${POOLED_D}" "${POOLED_F_META}" "${POOLED_G}" \
  "${COV_NPZ}" "${COV_JSON}" "${TIME_SPLIT}"; do
  test -s "${path}"
done

included=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] == "1"))' "${SELECTOR}")
excluded=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] != "1"))' "${SELECTOR}")
python apply/stages/07_sed_points.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_RUN}/${E_STEM}.npz" --signal-metadata "${E_RUN}/${E_STEM}_metadata.json" \
  --stage-f-npz "${F_RUN}/${F_STEM}.npz" --stage-f-metadata "${F_RUN}/${F_STEM}_metadata.json" \
  --output-dir "${G_ROOT}" --run-id stage_g --no-promote-current --overwrite-run-dir \
  --report-html "${G_RUN}/stage_g_report.html" --baseline-name "${EXPERIMENT}" \
  --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 --skip-expected-stage-f-validation \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${G_STEM}.npz" --metadata-name "${G_STEM}_metadata.json" \
  --summary-csv-name "${G_STEM}_summary.csv" --summary-json-name "${G_STEM}_summary.json" \
  --summary-md-name "${G_STEM}_summary.md"

mkdir -p "${ASSETS}"
python apply/report/replot_v6_display_cell_ids.py \
  --stage-d-npz "${D_RUN}/${D_STEM}.npz" \
  --stage-f-npz "${F_RUN}/${F_STEM}.npz" \
  --stage-f-metadata "${F_RUN}/${F_STEM}_metadata.json" \
  --stage-d-output-dir "${D_RUN}" --stage-f-output-dir "${F_RUN}" \
  --asset-dir "${ASSETS}"

python - "${E_RUN}/${E_STEM}.npz" "${E_RUN}" <<'PY'
from importlib import import_module
from pathlib import Path
import sys

import numpy as np

signal_path, output_dir = map(Path, sys.argv[1:])
stage_e = import_module("apply.stages.05_signal")
with np.load(signal_path, allow_pickle=False) as handle:
    arrays = {name: np.asarray(handle[name]) for name in handle.files}
cells = [
    stage_e.CellSpec(
        index=index,
        cell_id=int(cell_id),
        nhit_bin=str(nhit_bin),
        predE_bin=str(pred_bin),
        mc_count=0,
        selection_version="poisson_unbinned_report",
        selection_reason="nominal_report_plot",
    )
    for index, (cell_id, nhit_bin, pred_bin) in enumerate(
        zip(arrays["cell_id"], arrays["nhit_bin"], arrays["predE_bin"])
    )
]
stage_e.plot_heatmap_grid(
    arrays["formal_sigma"], cells, output_dir / "formal_sigma_grid.png",
    title="Stage E formal significance", colorbar_label="formal sigma",
    cmap_name="RdBu_r", symmetric=True,
)
stage_e.plot_on_background_grid(
    arrays["N_on"], arrays["B_on"], cells, output_dir / "on_background_grid.png",
    title="Stage E Crab on-region counts and unbinned background",
)
PY

export V6_REPORT_RUN_ID="${RUN_ID}"
export V6_REPORT_SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
export V6_REPORT_TITLE="Crab SED v6 64748 reselect44 - Scheme R - Double-Rayleigh Poisson-Unbinned Background"
export V6_REPORT_SCHEME=R
export V6_REPORT_SCHEME_LABEL="Scheme R - Double-Rayleigh PSF with Grid-Free Poisson Background"
export V6_REPORT_SCHEME_CONTRACT="Stage B uses the double-Rayleigh PSF with F(r_opt)=0.7129790300890827. The 7 order-2 Stage D surfaces use continuous-event profiled-Poisson curvature and analytic circular B_on; the other 37 target cells retain their frozen lower-order contracts. Stage E containment is 1; Stage F/G use Aeff_R=0.7129790300890827*Aeff_nominal exactly once."
export V6_REPORT_PATH="${REPORT_TMP}"
export V6_REPORT_ASSET_DIR="${ASSETS}"
export V6_REPORT_EXPERIMENT_ID="${EXPERIMENT}"
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
export V6_REPORT_STAGE_D_STEM="${D_STEM}"
export V6_REPORT_STAGE_E_STEM="${E_STEM}"
export V6_REPORT_STAGE_F_STEM="${F_STEM}"
export V6_REPORT_STAGE_G_STEM="${G_STEM}"
export V6_REPORT_EXPECTED_FIXED_CONTAINMENT=0.7129790300890827
export V6_REPORT_RATIO_HIDDEN_PREDE_POINTS=2
export V6_REPORT_SED_OVERLAY_TITLE="Crab SED - R/2R Poisson-unbinned"
export PIPELINE_JOB_IDS="grid:65261;bootstrap:65262;report:${SLURM_JOB_ID:-n/a}"

python apply/report/build_v6_64748_nhit100_highEplus1_report.py
python apply/report/prepare_v6_scheme_r_double_rayleigh_poisson_unbinned_report.py \
  --report "${REPORT_TMP}" --published-report "${REPORT}" --asset-dir "${ASSETS}" \
  --unbinned-stage-d "${D_RUN}/${D_STEM}.npz" --pooled-stage-d "${POOLED_D}" \
  --unbinned-stage-f-metadata "${F_RUN}/${F_STEM}_metadata.json" \
  --pooled-stage-f-metadata "${POOLED_F_META}" \
  --unbinned-stage-g "${G_RUN}/${G_STEM}.npz" --pooled-stage-g "${POOLED_G}" \
  --time-split-json "${TIME_SPLIT}" --covariance-npz "${COV_NPZ}" \
  --covariance-json "${COV_JSON}" \
  --shared-model-tier-figure apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-pooled/model_tier_donor_map.png \
  --implementation-sha "${IMPLEMENTATION_SHA}" --grid-job-id 65261 --bootstrap-job-id 65262 \
  --finalizer-job-id "${SLURM_JOB_ID:-n/a}"
mv "${REPORT_TMP}" "${REPORT}"

echo "Full Poisson-unbinned report: ${REPORT}"
