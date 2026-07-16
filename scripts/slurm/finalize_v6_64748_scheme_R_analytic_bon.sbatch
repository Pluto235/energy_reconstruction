#!/bin/bash
#SBATCH -J v6_64748_R_analytic_report
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

BASE=v6_64748_nhit100_reselect44_split56_miss030
PSF2=${BASE}_double_rayleigh
SOURCE=v6_64748_nhit100_highEplus1_split56
A2=${PSF2}_analytic_bon
E2=${PSF2}_scheme_R_fixed712979_analytic_bon
ASSET_DIR=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-analytic-bon
OLD_ASSET_DIR=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030
RESPONSE_ROOT=apply/output/stage_e_${BASE}_fixed712979_rayleigh_annnorm
D_DIR=apply/output/stage_d_${A2}_annnorm/runs/${A2}_stage_d_annnorm
E_DIR=apply/output/stage_e_${A2}_containment1_annnorm/runs/${A2}_stage_e_containment1_annnorm
F_DIR=apply/output/stage_f_${E2}/runs/${E2}_stage_f
G_DIR=apply/output/stage_g_${E2}/runs/${E2}_stage_g
D_STEM=background_${A2}_annnorm
E_STEM=signal_${A2}_containment1_annnorm
F_STEM=fit_${E2}
G_STEM=sed_points_${E2}

export ANALYTIC_R1_JOB_ID=${ANALYTIC_R1_JOB_ID:?ANALYTIC_R1_JOB_ID is required}
export ANALYTIC_R2_JOB_ID=${ANALYTIC_R2_JOB_ID:?ANALYTIC_R2_JOB_ID is required}
python apply/report/validate_v6_scheme_r_double_rayleigh_analytic_bon.py

python apply/report/replot_v6_display_cell_ids.py \
  --stage-f-only \
  --stage-f-npz "${F_DIR}/${F_STEM}.npz" \
  --stage-f-metadata "${F_DIR}/${F_STEM}_metadata.json" \
  --stage-f-output-dir "${F_DIR}" \
  --asset-dir "${ASSET_DIR}"

export V6_REPORT_RUN_ID="${PSF2}"
export V6_REPORT_EXPERIMENT_ID="${E2}"
export V6_REPORT_SOURCE_RUN_ID="${SOURCE}"
export V6_REPORT_TITLE="Crab SED v6 64748 reselect44 - Scheme R - Double-Rayleigh PSF - Analytic B_on"
export V6_REPORT_SCHEME=R
export V6_REPORT_SCHEME_LABEL="Scheme R - Double-Rayleigh PSF with Analytic B_on"
export V6_REPORT_SCHEME_CONTRACT="Stage B uses the constrained double-Rayleigh PSF with F(r_opt)=0.7129790300890827. Stage D keeps the 0.1-degree annulus-quadratic fit but integrates the fitted surface analytically over the true circular aperture. Stage E uses event-level spherical N_on and containment=1. Stage F/G use Aeff_R=0.7129790300890827*Aeff_nominal exactly once."
export V6_REPORT_PATH="apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_analytic_bon_report.html"
export V6_REPORT_ASSET_DIR="${ASSET_DIR}"
export V6_REPORT_TRUE_ENERGY_GRID="${OLD_ASSET_DIR}/true-energy-cell-grid/v6_64748_reselect44_true_energy_cell_grid.png"
export V6_REPORT_RESPONSE_META="${RESPONSE_ROOT}/response_${BASE}_fixed712979_rayleigh_metadata.json"
export V6_REPORT_FIT_SELECTOR="apply/config/cell_selector_${BASE}_fit.csv"
export V6_REPORT_SELECTOR_META="apply/config/cell_selector_${BASE}_fit_metadata.json"
export V6_REPORT_PREFIT_SELECTOR="apply/config/cell_selector_${SOURCE}_prefit.csv"
export V6_REPORT_HIGH_E_DECISIONS="apply/config/cell_selector_${BASE}_highEplus1_decisions.csv"
export V6_REPORT_STAGE_B_UNFILTERED_DIAGNOSTIC="apply/output/stage_b_${BASE}/runs/${BASE}_stage_b_psf/psf_${BASE}_unfiltered_diagnostic.npz"
export V6_REPORT_STAGE_D_RUN_DIR="${D_DIR}"
export V6_REPORT_STAGE_E_RUN_DIR="${E_DIR}"
export V6_REPORT_STAGE_F_RUN_DIR="${F_DIR}"
export V6_REPORT_STAGE_G_RUN_DIR="${G_DIR}"
export V6_REPORT_STAGE_D_STEM="${D_STEM}"
export V6_REPORT_STAGE_E_STEM="${E_STEM}"
export V6_REPORT_STAGE_F_STEM="${F_STEM}"
export V6_REPORT_STAGE_G_STEM="${G_STEM}"
export V6_REPORT_EXPECTED_FIXED_CONTAINMENT=0.7129790300890827
export V6_REPORT_RATIO_HIDDEN_PREDE_POINTS=2
export V6_REPORT_INPUT_COMMIT="$(git rev-parse HEAD)"
export PIPELINE_JOB_IDS="analytic_R1_stage_d_g:${ANALYTIC_R1_JOB_ID};analytic_R2_stage_d_g:${ANALYTIC_R2_JOB_ID};report:${SLURM_JOB_ID}"

python apply/report/build_v6_64748_nhit100_highEplus1_report.py
python apply/report/prepare_v6_scheme_r_double_rayleigh_analytic_bon_report.py
python apply/report/validate_v6_scheme_r_double_rayleigh_analytic_bon.py --require-report

echo "Final analytic B_on report: ${V6_REPORT_PATH}"
