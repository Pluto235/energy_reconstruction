#!/bin/bash
#SBATCH -J v6_64748_n100_hi1_cg
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

RUN_ID=v6_64748_nhit100_highEplus1_split56
CELL_LEDGER=apply/config/cell_ledger_${RUN_ID}_candidate.csv
PREFIT_SELECTOR=apply/config/cell_selector_${RUN_ID}_prefit.csv
FIT_SELECTOR=apply/config/cell_selector_${RUN_ID}_fit.csv
HIGH_E_DECISIONS=apply/config/cell_selector_${RUN_ID}_highEplus1_decisions.csv
SELECTOR_META=apply/config/cell_selector_${RUN_ID}_fit_metadata.json

STAGE_A_AP_DIR=apply/output/stage_a_${RUN_ID}_aperture_conditioned
STAGE_B_DIR=apply/output/stage_b_${RUN_ID}
STAGE_B_RUN=${RUN_ID}_stage_b_psf
STAGE_C_DIR=apply/output/stage_c_${RUN_ID}
STAGE_D_DIR=apply/output/stage_d_${RUN_ID}_annnorm
STAGE_E_DIR=apply/output/stage_e_${RUN_ID}_containment1_annnorm
STAGE_F_DIR=apply/output/stage_f_${RUN_ID}
STAGE_G_DIR=apply/output/stage_g_${RUN_ID}

RUN_C=${RUN_ID}_stage_c_halfyear
RUN_D=${RUN_ID}_stage_d_annnorm
RUN_E=${RUN_ID}_stage_e_containment1_annnorm
RUN_F=${RUN_ID}_stage_f
RUN_G=${RUN_ID}_stage_g

STAGE_C_RUN_DIR="${STAGE_C_DIR}/runs/${RUN_C}"
STAGE_D_RUN_DIR="${STAGE_D_DIR}/runs/${RUN_D}"
STAGE_E_RUN_DIR="${STAGE_E_DIR}/runs/${RUN_E}"
STAGE_F_RUN_DIR="${STAGE_F_DIR}/runs/${RUN_F}"

MAX_FILES=${MAX_FILES:-}
MAX_BATCHES=${MAX_BATCHES:-}
OBS_ROOT=${OBS_ROOT:-/mnt/mydisk/WCDA_observation_eval_64748}
TIME_ROOT=${TIME_ROOT:-${OBS_ROOT}/recovered_time}

echo "===== Build ${RUN_ID} highEplus1 fit selector from Stage B PSF quality ====="
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
  --max-theta-missing-mass 0.10

echo "===== Stage C ${RUN_ID} half-year observation reduction ====="
CMD_C=(
  python apply/stages/03_reduce_obs.py
  --obs-root "${OBS_ROOT}"
  --time-root "${TIME_ROOT}"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_C_DIR}"
  --run-id "${RUN_C}"
  --tree-name t_eventout
  --time-tree-name t_recovered_time
  --entries-per-chunk 200000
  --workers "${SLURM_CPUS_PER_TASK:-18}"
  --match-status-equals 0
  --cut-pinc-max 1.1
  --cut-fitstat-equals 0
  --cut-theta-max-deg 50
  --cut-dcedge-min 20
  --gap-threshold-sec 60
  --allow-missing-time
  --allow-entry-mismatch
  --print-every 25
  --overwrite-run-dir
)
if [[ -n "${MAX_FILES}" ]]; then
  CMD_C+=(--max-files "${MAX_FILES}")
fi
printf '%q ' "${CMD_C[@]}"
echo
"${CMD_C[@]}"

echo "===== Stage D ${RUN_ID} annulus-normalized quadratic background ====="
CMD_D=(
  python apply/stages/04_background.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --psf-npz "${STAGE_B_DIR}/runs/${STAGE_B_RUN}/psf_${RUN_ID}.npz"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_D_DIR}"
  --run-id "${RUN_D}"
  --background-mode crab_roi_local
  --roi-background-method annulus-quadratic
  --roi-fiducial-deg 6
  --roi-edge-diagnostic-deg 8
  --roi-grid-step-deg 0.1
  --annulus-default-inner-deg 1.5
  --annulus-width-deg 2.0
  --annulus-max-inner-deg 4.5
  --annulus-normalize-surface
  --roi-surface-order 2
  --batch-size 500000
  --workers "${SLURM_CPUS_PER_TASK:-18}"
  --print-every 10
  --npz-name "background_${RUN_ID}_annnorm.npz"
  --metadata-name "background_${RUN_ID}_annnorm_metadata.json"
  --summary-csv-name "background_${RUN_ID}_annnorm_summary.csv"
  --summary-md-name "background_${RUN_ID}_annnorm_summary.md"
  --overwrite-run-dir
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_D+=(--max-batches "${MAX_BATCHES}")
fi
printf '%q ' "${CMD_D[@]}"
echo
"${CMD_D[@]}"

echo "===== Stage E ${RUN_ID} containment-1 signal ====="
CMD_E=(
  python apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --background-npz "${STAGE_D_RUN_DIR}/background_${RUN_ID}_annnorm.npz"
  --background-metadata "${STAGE_D_RUN_DIR}/background_${RUN_ID}_annnorm_metadata.json"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_E_DIR}"
  --report-html "apply/report/stage_e_${RUN_ID}_containment1_annnorm_report.html"
  --run-id "${RUN_E}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma 50
  --quality-max-total-sigma 300
  --containment-override fixed1
  --print-every 10
  --npz-name "signal_${RUN_ID}_containment1_annnorm.npz"
  --metadata-name "signal_${RUN_ID}_containment1_annnorm_metadata.json"
  --summary-csv-name "signal_${RUN_ID}_containment1_annnorm_summary.csv"
  --summary-md-name "signal_${RUN_ID}_containment1_annnorm_summary.md"
  --overwrite-run-dir
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_E+=(--max-batches "${MAX_BATCHES}")
fi
printf '%q ' "${CMD_E[@]}"
echo
set +e
"${CMD_E[@]}"
stage_e_status=$?
set -e
if [[ ! -f "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm.npz" ]]; then
  exit "${stage_e_status}"
fi

echo "===== Stage F ${RUN_ID} aperture-conditioned fit ====="
CMD_F=(
  python apply/stages/06_fit.py
  --response-npz "${STAGE_A_AP_DIR}/response_2d_${RUN_ID}_aperture_conditioned.npz"
  --response-metadata "${STAGE_A_AP_DIR}/response_2d_${RUN_ID}_aperture_conditioned_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_${RUN_ID}_containment1_annnorm_metadata.json"
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --cell-subset-csv "${FIT_SELECTOR}"
  --output-dir "${STAGE_F_DIR}"
  --report-html "apply/report/stage_f_${RUN_ID}_report.html"
  --run-id "${RUN_F}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name "fit_${RUN_ID}.npz"
  --metadata-name "fit_${RUN_ID}_metadata.json"
  --summary-csv-name "fit_${RUN_ID}_summary.csv"
  --summary-md-name "fit_${RUN_ID}_summary.md"
  --overwrite-run-dir
)
printf '%q ' "${CMD_F[@]}"
echo
set +e
"${CMD_F[@]}"
stage_f_status=$?
set -e
if [[ ! -f "${STAGE_F_RUN_DIR}/fit_${RUN_ID}.npz" ]]; then
  exit "${stage_f_status}"
fi

REQUIRED_CELL_IDS="$(python - <<PY
import csv
truthy={"1","true","yes","y"}
rows=list(csv.DictReader(open("${FIT_SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if str(row.get("include","")).strip().lower() in truthy))
PY
)"
EXCLUDED_CELL_IDS="$(python - <<PY
import csv
truthy={"1","true","yes","y"}
rows=list(csv.DictReader(open("${FIT_SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if str(row.get("include","")).strip().lower() not in truthy))
PY
)"
echo "Required Stage F cells: ${REQUIRED_CELL_IDS}"
echo "Excluded Stage F cells: ${EXCLUDED_CELL_IDS}"

echo "===== Stage G ${RUN_ID} SED points ====="
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

echo "===== Build final ${RUN_ID} report ====="
python apply/report/build_v6_64748_nhit100_highEplus1_report.py

echo "${RUN_ID} Stage C-G chain finished"
