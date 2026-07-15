#!/bin/bash
#SBATCH -J v6_64748_r44_dr_dg
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
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

BASE_RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
RUN_ID=${BASE_RUN_ID}_double_rayleigh
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
CELL_LEDGER=apply/config/cell_ledger_${SOURCE_RUN_ID}_candidate.csv
FIT_SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
EXPECTED_SELECTOR_SHA=c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6

actual_selector_sha=$(sha256sum "${FIT_SELECTOR}" | awk '{print $1}')
if [[ "${actual_selector_sha}" != "${EXPECTED_SELECTOR_SHA}" ]]; then
  echo "Selector changed: ${actual_selector_sha}" >&2
  exit 2
fi

STAGE_A_DIR=apply/output/stage_a_${RUN_ID}_aperture_conditioned
STAGE_B_DIR=apply/output/stage_b_${RUN_ID}/runs/${RUN_ID}_stage_b_psf
STAGE_C_DIR=apply/output/stage_c_${SOURCE_RUN_ID}/runs/${SOURCE_RUN_ID}_stage_c_halfyear
STAGE_D_ROOT=apply/output/stage_d_${RUN_ID}_annnorm
STAGE_E_ROOT=apply/output/stage_e_${RUN_ID}_containment1_annnorm
STAGE_F_ROOT=apply/output/stage_f_${RUN_ID}
STAGE_G_ROOT=apply/output/stage_g_${RUN_ID}
RUN_D=${RUN_ID}_stage_d_annnorm
RUN_E=${RUN_ID}_stage_e_containment1_annnorm
RUN_F=${RUN_ID}_stage_f
RUN_G=${RUN_ID}_stage_g
STAGE_D_DIR=${STAGE_D_ROOT}/runs/${RUN_D}
STAGE_E_DIR=${STAGE_E_ROOT}/runs/${RUN_E}
STAGE_F_DIR=${STAGE_F_ROOT}/runs/${RUN_F}

echo "===== Stage D: unchanged annnorm background with double-Rayleigh aperture ====="
python apply/stages/04_background.py \
  --stage-c-dir "${STAGE_C_DIR}" \
  --psf-npz "${STAGE_B_DIR}/psf_${RUN_ID}.npz" \
  --cell-selection-csv "${CELL_LEDGER}" \
  --output-dir "${STAGE_D_ROOT}" \
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

echo "===== Stage E: containment=1 ====="
set +e
python apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C_DIR}" \
  --background-npz "${STAGE_D_DIR}/background_${RUN_ID}_annnorm.npz" \
  --background-metadata "${STAGE_D_DIR}/background_${RUN_ID}_annnorm_metadata.json" \
  --cell-selection-csv "${CELL_LEDGER}" \
  --output-dir "${STAGE_E_ROOT}" \
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
[[ -f "${STAGE_E_DIR}/signal_${RUN_ID}_containment1_annnorm.npz" ]] || exit "${stage_e_status}"

echo "===== Stage F: unchanged LogPar/PL fitting and exposure ====="
set +e
python apply/stages/06_fit.py \
  --response-npz "${STAGE_A_DIR}/response_2d_${RUN_ID}_aperture_conditioned.npz" \
  --response-metadata "${STAGE_A_DIR}/response_2d_${RUN_ID}_aperture_conditioned_metadata.json" \
  --signal-npz "${STAGE_E_DIR}/signal_${RUN_ID}_containment1_annnorm.npz" \
  --signal-metadata "${STAGE_E_DIR}/signal_${RUN_ID}_containment1_annnorm_metadata.json" \
  --stage-c-dir "${STAGE_C_DIR}" \
  --cell-subset-csv "${FIT_SELECTOR}" \
  --output-dir "${STAGE_F_ROOT}" \
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
[[ -f "${STAGE_F_DIR}/fit_${RUN_ID}.npz" ]] || exit "${stage_f_status}"

REQUIRED_CELL_IDS=$(python - <<PY
import csv
rows = list(csv.DictReader(open("${FIT_SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if row.get("include") == "1"))
PY
)
EXCLUDED_CELL_IDS=$(python - <<PY
import csv
rows = list(csv.DictReader(open("${FIT_SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if row.get("include") != "1"))
PY
)

echo "===== Stage G: unchanged Nhit/predE SED grouping ====="
python apply/stages/07_sed_points.py \
  --response-npz "${STAGE_A_DIR}/response_2d_${RUN_ID}_aperture_conditioned.npz" \
  --response-metadata "${STAGE_A_DIR}/response_2d_${RUN_ID}_aperture_conditioned_metadata.json" \
  --signal-npz "${STAGE_E_DIR}/signal_${RUN_ID}_containment1_annnorm.npz" \
  --signal-metadata "${STAGE_E_DIR}/signal_${RUN_ID}_containment1_annnorm_metadata.json" \
  --stage-f-npz "${STAGE_F_DIR}/fit_${RUN_ID}.npz" \
  --stage-f-metadata "${STAGE_F_DIR}/fit_${RUN_ID}_metadata.json" \
  --output-dir "${STAGE_G_ROOT}" \
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

python apply/report/validate_v6_scheme_b_double_rayleigh.py
