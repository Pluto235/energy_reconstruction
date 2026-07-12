#!/bin/bash
#SBATCH -J v6_64748_r44_m030_dg
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 18:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
CELL_LEDGER=apply/config/cell_ledger_${SOURCE_RUN_ID}_candidate.csv
PREFIT_SELECTOR=apply/config/cell_selector_${SOURCE_RUN_ID}_prefit.csv
FIT_SELECTOR=apply/config/cell_selector_${RUN_ID}_fit.csv
HIGH_E_DECISIONS=apply/config/cell_selector_${RUN_ID}_highEplus1_decisions.csv
SELECTOR_META=apply/config/cell_selector_${RUN_ID}_fit_metadata.json

STAGE_A_AP_DIR=apply/output/stage_a_${RUN_ID}_aperture_conditioned
STAGE_B_DIR=apply/output/stage_b_${RUN_ID}
STAGE_B_RUN=${RUN_ID}_stage_b_psf
STAGE_C_RUN_DIR=apply/output/stage_c_${SOURCE_RUN_ID}/runs/${SOURCE_RUN_ID}_stage_c_halfyear
STAGE_D_DIR=apply/output/stage_d_${RUN_ID}_annnorm
STAGE_E_DIR=apply/output/stage_e_${RUN_ID}_containment1_annnorm
STAGE_F_DIR=apply/output/stage_f_${RUN_ID}
STAGE_G_DIR=apply/output/stage_g_${RUN_ID}

RUN_D=${RUN_ID}_stage_d_annnorm
RUN_E=${RUN_ID}_stage_e_containment1_annnorm
RUN_F=${RUN_ID}_stage_f
RUN_G=${RUN_ID}_stage_g
STAGE_D_RUN_DIR=${STAGE_D_DIR}/runs/${RUN_D}
STAGE_E_RUN_DIR=${STAGE_E_DIR}/runs/${RUN_E}
STAGE_F_RUN_DIR=${STAGE_F_DIR}/runs/${RUN_F}

echo "===== Build exact 44-cell selector ====="
python apply/stages/00e_build_highEplus1_selector.py \
  --prefit-selector-csv "${PREFIT_SELECTOR}" \
  --candidate-ledger-csv "${CELL_LEDGER}" \
  --stage-b-summary-csv "${STAGE_B_DIR}/runs/${STAGE_B_RUN}/psf_${RUN_ID}_summary.csv" \
  --output-selector-csv "${FIT_SELECTOR}" \
  --decision-csv "${HIGH_E_DECISIONS}" \
  --metadata-json "${SELECTOR_META}" \
  --subset-version "${RUN_ID}_fit" \
  --min-mc-count 1000 \
  --min-psf-effective-events 200 \
  --min-core-fit-effective-events 200 \
  --max-theta-missing-mass 0.30 \
  --force-exclude-cell-ids 37,50,51,63,90 \
  --force-include-cell-ids 5,6,20,21,34,47,48,61,62,75 \
  --expected-fit-cells 44

echo "===== Stage D annulus-normalized quadratic background ====="
python apply/stages/04_background.py \
  --stage-c-dir "${STAGE_C_RUN_DIR}" \
  --psf-npz "${STAGE_B_DIR}/runs/${STAGE_B_RUN}/psf_${RUN_ID}.npz" \
  --cell-selection-csv "${CELL_LEDGER}" \
  --output-dir "${STAGE_D_DIR}" \
  --run-id "${RUN_D}" \
  --background-mode crab_roi_local \
  --roi-background-method annulus-quadratic \
  --roi-fiducial-deg 6 \
  --roi-edge-diagnostic-deg 8 \
  --roi-grid-step-deg 0.1 \
  --annulus-default-inner-deg 1.5 \
  --annulus-width-deg 2.0 \
  --annulus-max-inner-deg 4.5 \
  --annulus-normalize-surface \
  --roi-surface-order 2 \
  --batch-size 500000 \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --print-every 10 \
  --npz-name "background_${RUN_ID}_annnorm.npz" \
  --metadata-name "background_${RUN_ID}_annnorm_metadata.json" \
  --summary-csv-name "background_${RUN_ID}_annnorm_summary.csv" \
  --summary-md-name "background_${RUN_ID}_annnorm_summary.md" \
  --overwrite-run-dir

echo "===== Stage E containment-1 signal ====="
set +e
python apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C_RUN_DIR}" \
  --background-npz "${STAGE_D_RUN_DIR}/background_${RUN_ID}_annnorm.npz" \
  --background-metadata "${STAGE_D_RUN_DIR}/background_${RUN_ID}_annnorm_metadata.json" \
  --cell-selection-csv "${CELL_LEDGER}" \
  --output-dir "${STAGE_E_DIR}" \
  --report-html "apply/report/stage_e_${RUN_ID}_containment1_annnorm_report.html" \
  --run-id "${RUN_E}" \
  --source-ra-deg 83.63 \
  --source-dec-deg 22.01 \
  --batch-size 500000 \
  --quality-min-total-sigma 50 \
  --quality-max-total-sigma 300 \
  --containment-override fixed1 \
  --print-every 10 \
  --npz-name "signal_${RUN_ID}_containment1_annnorm.npz" \
  --metadata-name "signal_${RUN_ID}_containment1_annnorm_metadata.json" \
  --summary-csv-name "signal_${RUN_ID}_containment1_annnorm_summary.csv" \
  --summary-md-name "signal_${RUN_ID}_containment1_annnorm_summary.md" \
  --overwrite-run-dir
stage_e_status=$?
set -e
if [[ ! -f "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm.npz" ]]; then
  exit "${stage_e_status}"
fi

echo "===== Stage F aperture-conditioned fit ====="
set +e
python apply/stages/06_fit.py \
  --response-npz "${STAGE_A_AP_DIR}/response_2d_${RUN_ID}_aperture_conditioned.npz" \
  --response-metadata "${STAGE_A_AP_DIR}/response_2d_${RUN_ID}_aperture_conditioned_metadata.json" \
  --signal-npz "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm.npz" \
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm_metadata.json" \
  --stage-c-dir "${STAGE_C_RUN_DIR}" \
  --cell-subset-csv "${FIT_SELECTOR}" \
  --output-dir "${STAGE_F_DIR}" \
  --report-html "apply/report/stage_f_${RUN_ID}_report.html" \
  --run-id "${RUN_F}" \
  --source-ra-deg 83.63 \
  --source-dec-deg 22.01 \
  --lhaaso-lat-deg 29.45 \
  --lhaaso-lon-deg 100.14 \
  --exposure-sample-step-sec 60 \
  --pivot-tev 3 \
  --reference-phi0 2.114e-12 \
  --reference-gamma 2.69 \
  --npz-name "fit_${RUN_ID}.npz" \
  --metadata-name "fit_${RUN_ID}_metadata.json" \
  --summary-csv-name "fit_${RUN_ID}_summary.csv" \
  --summary-md-name "fit_${RUN_ID}_summary.md" \
  --overwrite-run-dir
stage_f_status=$?
set -e
if [[ ! -f "${STAGE_F_RUN_DIR}/fit_${RUN_ID}.npz" ]]; then
  exit "${stage_f_status}"
fi

REQUIRED_CELL_IDS="$(python - <<PY
import csv
rows = list(csv.DictReader(open("${FIT_SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if row.get("include") == "1"))
PY
)"
EXCLUDED_CELL_IDS="$(python - <<PY
import csv
rows = list(csv.DictReader(open("${FIT_SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if row.get("include") != "1"))
PY
)"

echo "===== Stage G SED points ====="
python apply/stages/07_sed_points.py \
  --response-npz "${STAGE_A_AP_DIR}/response_2d_${RUN_ID}_aperture_conditioned.npz" \
  --response-metadata "${STAGE_A_AP_DIR}/response_2d_${RUN_ID}_aperture_conditioned_metadata.json" \
  --signal-npz "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm.npz" \
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm_metadata.json" \
  --stage-f-npz "${STAGE_F_RUN_DIR}/fit_${RUN_ID}.npz" \
  --stage-f-metadata "${STAGE_F_RUN_DIR}/fit_${RUN_ID}_metadata.json" \
  --output-dir "${STAGE_G_DIR}" \
  --report-html "apply/report/stage_g_${RUN_ID}_report.html" \
  --run-id "${RUN_G}" \
  --baseline-name "${RUN_ID}" \
  --required-cell-ids "${REQUIRED_CELL_IDS}" \
  --excluded-cell-ids "${EXCLUDED_CELL_IDS}" \
  --skip-expected-stage-f-validation \
  --pivot-tev 3 \
  --reference-phi0 2.114e-12 \
  --reference-gamma 2.69 \
  --npz-name "sed_points_${RUN_ID}.npz" \
  --metadata-name "sed_points_${RUN_ID}_metadata.json" \
  --summary-csv-name "sed_points_${RUN_ID}_summary.csv" \
  --summary-json-name "sed_points_${RUN_ID}_summary.json" \
  --summary-md-name "sed_points_${RUN_ID}_summary.md" \
  --overwrite-run-dir

echo "===== Build integrated report ====="
export V6_REPORT_RUN_ID="${RUN_ID}"
export V6_REPORT_SOURCE_RUN_ID="${SOURCE_RUN_ID}"
export V6_REPORT_TITLE="Crab SED v6 64748 nhit100 reselect44 miss030 Stage A-G"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export PIPELINE_JOB_IDS="${PIPELINE_JOB_IDS:+${PIPELINE_JOB_IDS};}stage_d_to_g:${SLURM_JOB_ID}"
fi
python apply/report/build_v6_64748_nhit100_highEplus1_report.py

echo "${RUN_ID} Stage D-G chain finished"
