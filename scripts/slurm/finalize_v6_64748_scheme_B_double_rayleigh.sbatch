#!/bin/bash
#SBATCH -J v6_64748_r44_dr_report
#SBATCH -p main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

BASE_RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
RUN_ID=${BASE_RUN_ID}_double_rayleigh
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
ASSET_DIR=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh
OLD_ASSET_DIR=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030

python apply/report/validate_v6_scheme_b_double_rayleigh.py

STAGE_D_RUN_DIR=apply/output/stage_d_${RUN_ID}_annnorm/runs/${RUN_ID}_stage_d_annnorm
STAGE_F_RUN_DIR=apply/output/stage_f_${RUN_ID}/runs/${RUN_ID}_stage_f
python apply/report/replot_v6_display_cell_ids.py \
  --stage-d-npz "${STAGE_D_RUN_DIR}/background_${RUN_ID}_annnorm.npz" \
  --stage-f-npz "${STAGE_F_RUN_DIR}/fit_${RUN_ID}.npz" \
  --stage-f-metadata "${STAGE_F_RUN_DIR}/fit_${RUN_ID}_metadata.json" \
  --stage-d-output-dir "${STAGE_D_RUN_DIR}" \
  --stage-f-output-dir "${STAGE_F_RUN_DIR}" \
  --asset-dir "${ASSET_DIR}" \
  --profile-half-width-deg 1.0

export V6_REPORT_RUN_ID="${RUN_ID}"
export V6_REPORT_SOURCE_RUN_ID="${SOURCE_RUN_ID}"
export V6_REPORT_TITLE="Crab SED v6 64748 reselect44 - Scheme B - Double-Rayleigh MC Aperture"
export V6_REPORT_SCHEME=B
export V6_REPORT_SCHEME_LABEL="Scheme B - Double-Rayleigh MC Aperture-conditioned Response"
export V6_REPORT_SCHEME_CONTRACT="Stage B fits the Crab-theta-weighted MC radial profile with a constrained double-Rayleigh mixture and solves F(r_opt)=0.7129790300890827; sigma_eq=r_opt/1.58 is diagnostic only. Stage A is rebuilt inside r_opt, Stage C is reused, Stage E containment is 1, and no additional containment factor is applied in Stage F/G."
export V6_REPORT_PATH="apply/report/crab_sed_${RUN_ID}_stage_a_to_g_report.html"
export V6_REPORT_ASSET_DIR="${ASSET_DIR}"
export V6_REPORT_TRUE_ENERGY_GRID="${OLD_ASSET_DIR}/true-energy-cell-grid/v6_64748_reselect44_true_energy_cell_grid.png"
export V6_REPORT_RESPONSE_META="apply/output/stage_a_${RUN_ID}_aperture_conditioned/response_2d_${RUN_ID}_aperture_conditioned_metadata.json"
export V6_REPORT_FIT_SELECTOR="apply/config/cell_selector_${BASE_RUN_ID}_fit.csv"
export V6_REPORT_SELECTOR_META="apply/config/cell_selector_${BASE_RUN_ID}_fit_metadata.json"
export V6_REPORT_PREFIT_SELECTOR="apply/config/cell_selector_${SOURCE_RUN_ID}_prefit.csv"
export V6_REPORT_HIGH_E_DECISIONS="apply/config/cell_selector_${BASE_RUN_ID}_highEplus1_decisions.csv"
export V6_REPORT_STAGE_B_UNFILTERED_DIAGNOSTIC="apply/output/stage_b_${BASE_RUN_ID}/runs/${BASE_RUN_ID}_stage_b_psf/psf_${BASE_RUN_ID}_unfiltered_diagnostic.npz"
export V6_REPORT_INPUT_COMMIT="$(git rev-parse HEAD)"
if [[ "${INCLUDE_CURRENT_REPORT_JOB:-1}" != "0" ]]; then
  export PIPELINE_JOB_IDS="${PIPELINE_JOB_IDS:+${PIPELINE_JOB_IDS};}report:${SLURM_JOB_ID}"
fi

python apply/report/build_v6_64748_nhit100_highEplus1_report.py
export V6_SCHEME_B_RUN_ID="${RUN_ID}"
export V6_SCHEME_B_SOURCE_REPORT="${V6_REPORT_PATH}"
python apply/report/prepare_v6_scheme_b_report.py
python apply/report/validate_v6_scheme_b_double_rayleigh.py --require-report

echo "Final double-Rayleigh Scheme B report: apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_B_report.html"
