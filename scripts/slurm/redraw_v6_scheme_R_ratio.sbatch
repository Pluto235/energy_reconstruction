#!/bin/bash
#SBATCH -J v6_R_ratio_trim2
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
SELECTOR=apply/config/cell_selector_${RUN_ID}_fit.csv
E_ROOT=apply/output/stage_e_${RUN_ID}_fixed715_rayleigh_annnorm
E_DIR=${E_ROOT}/runs/${RUN_ID}_stage_e_fixed715_rayleigh_annnorm
E_STEM=signal_${RUN_ID}_fixed715_rayleigh_annnorm
F_ROOT=apply/output/stage_f_${RUN_ID}_scheme_R
F_DIR=${F_ROOT}/runs/${RUN_ID}_stage_f_scheme_R
F_STEM=fit_${RUN_ID}_scheme_R
G_ROOT=apply/output/stage_g_${RUN_ID}_scheme_R
G_RUN=${RUN_ID}_stage_g_scheme_R
G_STEM=sed_points_${RUN_ID}_scheme_R
RESPONSE=${E_ROOT}/response_${RUN_ID}_fixed715_rayleigh.npz
RESPONSE_META=${E_ROOT}/response_${RUN_ID}_fixed715_rayleigh_metadata.json

included=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] == "1"))' "${SELECTOR}")
excluded=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] != "1"))' "${SELECTOR}")

python apply/stages/07_sed_points.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_DIR}/${E_STEM}.npz" --signal-metadata "${E_DIR}/${E_STEM}_metadata.json" \
  --stage-f-npz "${F_DIR}/${F_STEM}.npz" --stage-f-metadata "${F_DIR}/${F_STEM}_metadata.json" \
  --output-dir "${G_ROOT}" --run-id "${G_RUN}" --no-promote-current --overwrite-run-dir \
  --report-html "apply/report/stage_g_${RUN_ID}_scheme_R_report.html" \
  --baseline-name "${RUN_ID}_scheme_R" --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 --skip-expected-stage-f-validation \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${G_STEM}.npz" --metadata-name "${G_STEM}_metadata.json" \
  --summary-csv-name "${G_STEM}_summary.csv" --summary-json-name "${G_STEM}_summary.json" \
  --summary-md-name "${G_STEM}_summary.md"

echo "Scheme R ratio redrawn with final two predE points hidden: ${SLURM_JOB_ID}"
