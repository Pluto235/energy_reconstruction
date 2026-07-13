#!/bin/bash
#SBATCH -J v6_scheme_reports
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

RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
E_ROOT=apply/output/stage_e_${RUN_ID}_fixed715_rayleigh_annnorm
E_RUN=${RUN_ID}_stage_e_fixed715_rayleigh_annnorm
F_ROOT=apply/output/stage_f_${RUN_ID}_scheme_R
F_RUN=${RUN_ID}_stage_f_scheme_R
G_ROOT=apply/output/stage_g_${RUN_ID}_scheme_R
G_RUN=${RUN_ID}_stage_g_scheme_R

python apply/report/prepare_v6_scheme_b_report.py

export V6_REPORT_RUN_ID="${RUN_ID}"
export V6_REPORT_SOURCE_RUN_ID="${SOURCE_RUN_ID}"
export V6_REPORT_TITLE="Crab SED v6 64748 reselect44 - Scheme R - Fixed 71.5% Rayleigh Containment"
export V6_REPORT_SCHEME=R
export V6_REPORT_SCHEME_LABEL="Scheme R - Fixed 71.5% Rayleigh Containment"
export V6_REPORT_SCHEME_CONTRACT="Uses a derived nominal response with Aeff multiplied by 0.715 exactly once during fixed715 response preparation; Stage E containment is 1, Stage F/G apply no further containment, and the aperture-conditioned response is not used."
export V6_REPORT_PATH="apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_report.html"
export V6_REPORT_ASSET_DIR="apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030/scheme-R"
export V6_REPORT_TRUE_ENERGY_GRID="apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030/true-energy-cell-grid/v6_64748_reselect44_true_energy_cell_grid.png"
export V6_REPORT_RESPONSE_META="${E_ROOT}/response_${RUN_ID}_fixed715_rayleigh_metadata.json"
export V6_REPORT_STAGE_E_RUN_DIR="${E_ROOT}/runs/${E_RUN}"
export V6_REPORT_STAGE_F_RUN_DIR="${F_ROOT}/runs/${F_RUN}"
export V6_REPORT_STAGE_G_RUN_DIR="${G_ROOT}/runs/${G_RUN}"
export V6_REPORT_STAGE_E_STEM="signal_${RUN_ID}_fixed715_rayleigh_annnorm"
export V6_REPORT_STAGE_F_STEM="fit_${RUN_ID}_scheme_R"
export V6_REPORT_STAGE_G_STEM="sed_points_${RUN_ID}_scheme_R"
export PIPELINE_JOB_IDS="scheme_R_stage_e_f_g:${SCHEME_R_JOB_ID:?SCHEME_R_JOB_ID is required}"
export V6_REPORT_INPUT_COMMIT="$(git rev-parse HEAD)"
python apply/report/build_v6_64748_nhit100_highEplus1_report.py

echo "Scheme reports finalized: ${SLURM_JOB_ID}"
