#!/bin/bash
#SBATCH -J v6_64748_R_2R_report
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
PSF_RUN_ID=${BASE_RUN_ID}_double_rayleigh
EXPERIMENT_ID=${PSF_RUN_ID}_scheme_R_fixed712979
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
ASSET_DIR=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R
OLD_ASSET_DIR=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030

RESPONSE_ROOT=apply/output/stage_e_${BASE_RUN_ID}_fixed712979_rayleigh_annnorm
STAGE_E=apply/output/stage_e_${PSF_RUN_ID}_containment1_annnorm/runs/${PSF_RUN_ID}_stage_e_containment1_annnorm
STAGE_F=apply/output/stage_f_${EXPERIMENT_ID}/runs/${EXPERIMENT_ID}_stage_f
STAGE_G=apply/output/stage_g_${EXPERIMENT_ID}/runs/${EXPERIMENT_ID}_stage_g

python apply/report/validate_v6_scheme_r_double_rayleigh.py

python apply/report/replot_v6_display_cell_ids.py \
  --stage-f-only \
  --stage-f-npz "${STAGE_F}/fit_${EXPERIMENT_ID}.npz" \
  --stage-f-metadata "${STAGE_F}/fit_${EXPERIMENT_ID}_metadata.json" \
  --stage-f-output-dir "${STAGE_F}" \
  --asset-dir "${ASSET_DIR}"

export V6_REPORT_RUN_ID="${PSF_RUN_ID}"
export V6_REPORT_EXPERIMENT_ID="${EXPERIMENT_ID}"
export V6_REPORT_SOURCE_RUN_ID="${SOURCE_RUN_ID}"
export V6_REPORT_TITLE="Crab SED v6 64748 reselect44 - Scheme R - Double-Rayleigh PSF"
export V6_REPORT_SCHEME=R
export V6_REPORT_SCHEME_LABEL="Scheme R - Double-Rayleigh PSF with Fixed 1.58-sigma Response"
export V6_REPORT_SCHEME_CONTRACT="Stage B uses the constrained double-Rayleigh PSF with F(r_opt)=0.7129790300890827. Stage D/E are reused byte-for-byte from Scheme B double-Rayleigh. Stage F/G use Aeff_R=0.7129790300890827*Aeff_nominal exactly once; downstream containment is 1 and the aperture-conditioned response is not used."
export V6_REPORT_PATH="apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_report.html"
export V6_REPORT_ASSET_DIR="${ASSET_DIR}"
export V6_REPORT_TRUE_ENERGY_GRID="${OLD_ASSET_DIR}/true-energy-cell-grid/v6_64748_reselect44_true_energy_cell_grid.png"
export V6_REPORT_RESPONSE_META="${RESPONSE_ROOT}/response_${BASE_RUN_ID}_fixed712979_rayleigh_metadata.json"
export V6_REPORT_FIT_SELECTOR="apply/config/cell_selector_${BASE_RUN_ID}_fit.csv"
export V6_REPORT_SELECTOR_META="apply/config/cell_selector_${BASE_RUN_ID}_fit_metadata.json"
export V6_REPORT_PREFIT_SELECTOR="apply/config/cell_selector_${SOURCE_RUN_ID}_prefit.csv"
export V6_REPORT_HIGH_E_DECISIONS="apply/config/cell_selector_${BASE_RUN_ID}_highEplus1_decisions.csv"
export V6_REPORT_STAGE_B_UNFILTERED_DIAGNOSTIC="apply/output/stage_b_${BASE_RUN_ID}/runs/${BASE_RUN_ID}_stage_b_psf/psf_${BASE_RUN_ID}_unfiltered_diagnostic.npz"
export V6_REPORT_STAGE_E_RUN_DIR="${STAGE_E}"
export V6_REPORT_STAGE_F_RUN_DIR="${STAGE_F}"
export V6_REPORT_STAGE_G_RUN_DIR="${STAGE_G}"
export V6_REPORT_STAGE_E_STEM="signal_${PSF_RUN_ID}_containment1_annnorm"
export V6_REPORT_STAGE_F_STEM="fit_${EXPERIMENT_ID}"
export V6_REPORT_STAGE_G_STEM="sed_points_${EXPERIMENT_ID}"
export V6_REPORT_EXPECTED_FIXED_CONTAINMENT=0.7129790300890827
export V6_REPORT_RATIO_HIDDEN_PREDE_POINTS=2
export V6_REPORT_INPUT_COMMIT="$(git rev-parse HEAD)"
export PIPELINE_JOB_IDS="scheme_R_2R_stage_f_g:${SCHEME_R_2R_JOB_ID:?SCHEME_R_2R_JOB_ID is required};report:${SLURM_JOB_ID}"

python apply/report/build_v6_64748_nhit100_highEplus1_report.py
python apply/report/prepare_v6_scheme_r_double_rayleigh_report.py
python apply/report/validate_v6_scheme_r_double_rayleigh.py --require-report

echo "Final Scheme R double-Rayleigh report: ${V6_REPORT_PATH}"
