#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x /home/server/anaconda3/envs/py310/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/server/anaconda3/envs/py310/bin/python}"
elif [[ -x /home/server/anaconda3/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/server/anaconda3/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

RUN_ID="${RUN_ID:-v4_split56_ridge}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-18}}"
DENOMINATOR_WORKERS="${DENOMINATOR_WORKERS:-1}"
MAX_FILES_PER_SOURCE_BIN="${MAX_FILES_PER_SOURCE_BIN:-}"
MAX_FILES_PER_CELL="${MAX_FILES_PER_CELL:-}"
MAX_OBS_FILES="${MAX_OBS_FILES:-}"
MAX_BATCHES="${MAX_BATCHES:-}"
RESUME="${RESUME:-1}"
QUALITY_MIN_TOTAL_SIGMA="${QUALITY_MIN_TOTAL_SIGMA:--300}"
QUALITY_MAX_TOTAL_SIGMA="${QUALITY_MAX_TOTAL_SIGMA:-300}"

SOURCE_BINNED_ROOT="${SOURCE_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v1}"
SOURCE_BIN_COUNTS="${SOURCE_BIN_COUNTS:-/mnt/mydisk/WCDA_simulation_binned_response_v1/summary/bin_counts.csv}"
GAP025_BINNED_ROOT="${GAP025_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v5_predbin_gap025}"
BINNED_ROOT="${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v4_split56_ridge}"
OBS_ROOT="${OBS_ROOT:-/mnt/mydisk/WCDA_observation_eval}"
TIME_ROOT="${TIME_ROOT:-/mnt/mydisk/WCDA_observation_eval/recovered_time}"

CELL_LEDGER="${CELL_LEDGER:-apply/config/cell_ledger_v5_predbin_split56.csv}"
SELECTOR="${SELECTOR:-apply/config/cell_selector_v5_predbin_split56.csv}"
PSF_DIR="${PSF_DIR:-apply/output/stage_b_v4_split56_ridge}"
PSF_RUN="${PSF_RUN:-${RUN_ID}_stage_b_rayleigh}"
PSF_NPZ_NAME="psf_${RUN_ID}.npz"
PSF_META_NAME="psf_${RUN_ID}_metadata.json"
PSF_NPZ="${PSF_DIR}/runs/${PSF_RUN}/${PSF_NPZ_NAME}"
PSF_META="${PSF_DIR}/runs/${PSF_RUN}/${PSF_META_NAME}"

STAGE_A_DIR="${STAGE_A_DIR:-apply/output/stage_a_v4_split56_ridge}"
RESPONSE_NPZ_NAME="response_2d_${RUN_ID}.npz"
RESPONSE_META_NAME="response_2d_${RUN_ID}_metadata.json"
RESPONSE_NPZ="${STAGE_A_DIR}/${RESPONSE_NPZ_NAME}"
RESPONSE_META="${STAGE_A_DIR}/${RESPONSE_META_NAME}"

STAGE_C_DIR_ROOT="${STAGE_C_DIR_ROOT:-apply/output/stage_c_v4_split56_ridge}"
STAGE_C_RUN="${STAGE_C_RUN:-${RUN_ID}_stage_c}"
STAGE_C_DIR="${STAGE_C_DIR_ROOT}/runs/${STAGE_C_RUN}"

STAGE_D_DIR="${STAGE_D_DIR:-apply/output/stage_d_v4_split56_ridge}"
STAGE_D_RUN="${STAGE_D_RUN:-${RUN_ID}_stage_d_annnorm}"
STAGE_D_NPZ_NAME="background_${RUN_ID}.npz"
STAGE_D_META_NAME="background_${RUN_ID}_metadata.json"
STAGE_D_NPZ="${STAGE_D_DIR}/runs/${STAGE_D_RUN}/${STAGE_D_NPZ_NAME}"
STAGE_D_META="${STAGE_D_DIR}/runs/${STAGE_D_RUN}/${STAGE_D_META_NAME}"

STAGE_E_DIR="${STAGE_E_DIR:-apply/output/stage_e_v4_split56_ridge}"
STAGE_E_RUN="${STAGE_E_RUN:-${RUN_ID}_stage_e_containment1}"
STAGE_E_NPZ_NAME="signal_${RUN_ID}.npz"
STAGE_E_META_NAME="signal_${RUN_ID}_metadata.json"
STAGE_E_NPZ="${STAGE_E_DIR}/runs/${STAGE_E_RUN}/${STAGE_E_NPZ_NAME}"
STAGE_E_META="${STAGE_E_DIR}/runs/${STAGE_E_RUN}/${STAGE_E_META_NAME}"

STAGE_F_DIR="${STAGE_F_DIR:-apply/output/stage_f_v4_split56_ridge}"
STAGE_F_RUN="${STAGE_F_RUN:-${RUN_ID}_stage_f}"
STAGE_F_NPZ_NAME="fit_${RUN_ID}.npz"
STAGE_F_META_NAME="fit_${RUN_ID}_metadata.json"
STAGE_F_NPZ="${STAGE_F_DIR}/runs/${STAGE_F_RUN}/${STAGE_F_NPZ_NAME}"
STAGE_F_META="${STAGE_F_DIR}/runs/${STAGE_F_RUN}/${STAGE_F_META_NAME}"

STAGE_G_DIR="${STAGE_G_DIR:-apply/output/stage_g_v4_split56_ridge}"
STAGE_G_RUN="${STAGE_G_RUN:-${RUN_ID}_stage_g}"
STAGE_G_NPZ_NAME="sed_points_${RUN_ID}.npz"
STAGE_G_META_NAME="sed_points_${RUN_ID}_metadata.json"
STAGE_G_META="${STAGE_G_DIR}/runs/${STAGE_G_RUN}/${STAGE_G_META_NAME}"

for required in "${SOURCE_BINNED_ROOT}" "${SOURCE_BIN_COUNTS}" "${OBS_ROOT}" "${TIME_ROOT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

run_or_skip() {
  local label="$1"
  local sentinel="$2"
  shift 2
  if [[ "${RESUME}" == "1" && -s "${sentinel}" ]]; then
    echo "===== Skip ${label}; found ${sentinel} ====="
    return 0
  fi
  echo "===== ${label} ====="
  printf '%q ' "$@"; echo
  "$@"
}

echo "Python: ${PYTHON_BIN}"
echo "Run id: ${RUN_ID}"
echo "Workers: ${WORKERS}"
echo "Resume: ${RESUME}"

CMD_CACHE=(
  "${PYTHON_BIN}" apply/stages/00a_build_split56_cache_from_gap025.py
  --source-root "${GAP025_BINNED_ROOT}"
  --target-root "${BINNED_ROOT}"
  --overwrite
)
run_or_skip "Build split56 MC cache from gap025 cache" "${BINNED_ROOT}/summary/v4_split56_from_gap025_manifest.json" "${CMD_CACHE[@]}"

CMD_PREP=(
  "${PYTHON_BIN}" apply/stages/00_prepare_v5_predbin_ablation.py
  --strategy split56
  --source-binned-root "${SOURCE_BINNED_ROOT}"
  --source-bin-counts-csv "${SOURCE_BIN_COUNTS}"
  --target-root-template "${BINNED_ROOT}"
  --config-dir apply/config
  --diagnostics-html apply/report/v4_split56_ridge_cell_selection_diagnostics.html
  --write-configs
  --use-existing-target-cache-counts
  --workers "${WORKERS}"
  --progress-every 500
  --min-baseline-mc-count 1000
  --ridge-min-peak-fraction 0.10
  --write-diagnostics
)
if [[ -n "${MAX_FILES_PER_SOURCE_BIN}" ]]; then
  CMD_PREP+=(--max-files-per-source-bin "${MAX_FILES_PER_SOURCE_BIN}")
fi
run_or_skip "Prepare split56 MC cache and configs" "apply/config/v5_predbin_split56_strategy_manifest.json" "${CMD_PREP[@]}"

CMD_B=(
  "${PYTHON_BIN}" apply/stages/02e_build_psf_v5_compare.py
  --psf-method rayleigh_baseline
  --binned-root "${BINNED_ROOT}"
  --cell-selection-csv "${CELL_LEDGER}"
  --stage-a-metadata "${RESPONSE_META}"
  --allow-missing-stage-a-metadata
  --logE-min 2
  --logE-max 6
  --output-dir "${PSF_DIR}"
  --run-id "${PSF_RUN}"
  --tree-name t_eventout
  --weight-branch mc_weight
  --lhaaso-lat-deg 29.45
  --source-dec-deg 22.01
  --theta-min-deg 0.0
  --theta-max-deg 50.0
  --theta-step-deg 1.0
  --hour-angle-samples 200000
  --allow-incomplete-theta-support
  --allow-missing-cell-dirs
  --allow-low-stat-psf-fallback
  --min-events-per-cell 1000
  --min-effective-events 200
  --core-fit-max-deg 3.0
  --theta-missing-mass-fail-threshold 0.10
  --containment-warning-tolerance 0.12
  --angle-check-max-events 20000
  --file-progress-every 5000
  --workers "${WORKERS}"
  --npz-name "${PSF_NPZ_NAME}"
  --metadata-name "${PSF_META_NAME}"
  --summary-csv-name "psf_${RUN_ID}_summary.csv"
  --summary-md-name "psf_${RUN_ID}_summary.md"
  --overwrite-run-dir
  --no-promote-current
)
if [[ -n "${MAX_FILES_PER_CELL}" ]]; then
  CMD_B+=(--max-files-per-cell "${MAX_FILES_PER_CELL}")
fi
run_or_skip "Stage B rayleigh PSF" "${PSF_META}" "${CMD_B[@]}"

CMD_A=(
  "${PYTHON_BIN}" apply/stages/01_build_response.py
  --binned-root "${BINNED_ROOT}"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_A_DIR}"
  --tree-name t_eventout
  --weight-branch mc_weight
  --allow-missing-cell-dirs
  --denominator-workers "${DENOMINATOR_WORKERS}"
  --numerator-workers "${WORKERS}"
  --numerator-files-per-task 250
  --numerator-progress-every 20
  --aperture-psf-npz "${PSF_NPZ}"
  --npz-name "${RESPONSE_NPZ_NAME}"
  --metadata-name "${RESPONSE_META_NAME}"
)
run_or_skip "Stage A aperture-conditioned response" "${RESPONSE_META}" "${CMD_A[@]}"

CMD_C=(
  "${PYTHON_BIN}" apply/stages/03_reduce_obs.py
  --obs-root "${OBS_ROOT}"
  --time-root "${TIME_ROOT}"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_C_DIR_ROOT}"
  --run-id "${STAGE_C_RUN}"
  --tree-name t_eventout
  --time-tree-name t_recovered_time
  --workers "${WORKERS}"
  --entries-per-chunk 200000
  --print-every 25
  --allow-entry-mismatch
  --overwrite-run-dir
  --no-promote-current
)
if [[ -n "${MAX_OBS_FILES}" ]]; then
  CMD_C+=(--max-files "${MAX_OBS_FILES}")
fi
run_or_skip "Stage C observation reduction" "${STAGE_C_DIR}/obs_events_metadata.json" "${CMD_C[@]}"

CMD_D=(
  "${PYTHON_BIN}" apply/stages/04_background.py
  --stage-c-dir "${STAGE_C_DIR}"
  --psf-npz "${PSF_NPZ}"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_D_DIR}"
  --run-id "${STAGE_D_RUN}"
  --background-mode crab_roi_local
  --roi-background-method annulus-quadratic
  --roi-fiducial-deg 6.0
  --annulus-normalize-surface
  --npz-name "${STAGE_D_NPZ_NAME}"
  --metadata-name "${STAGE_D_META_NAME}"
  --summary-csv-name "background_${RUN_ID}_summary.csv"
  --summary-md-name "background_${RUN_ID}_summary.md"
  --overwrite-run-dir
  --no-promote-current
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_D+=(--max-batches "${MAX_BATCHES}")
fi
run_or_skip "Stage D annulus-normalized background" "${STAGE_D_META}" "${CMD_D[@]}"

CMD_E=(
  "${PYTHON_BIN}" apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_DIR}"
  --background-npz "${STAGE_D_NPZ}"
  --background-metadata "${STAGE_D_META}"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_E_DIR}"
  --run-id "${STAGE_E_RUN}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma "${QUALITY_MIN_TOTAL_SIGMA}"
  --quality-max-total-sigma "${QUALITY_MAX_TOTAL_SIGMA}"
  --containment-override fixed1
  --print-every 10
  --npz-name "${STAGE_E_NPZ_NAME}"
  --metadata-name "${STAGE_E_META_NAME}"
  --summary-csv-name "signal_${RUN_ID}_summary.csv"
  --summary-md-name "signal_${RUN_ID}_summary.md"
  --report-html "apply/report/stage_e_${RUN_ID}_report.html"
  --overwrite-run-dir
  --no-promote-current
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_E+=(--max-batches "${MAX_BATCHES}")
fi
run_or_skip "Stage E signal" "${STAGE_E_META}" "${CMD_E[@]}"

REQUIRED_CELL_IDS="$("${PYTHON_BIN}" - <<PY
import csv
truthy={"1","true","yes","y","include"}
rows=list(csv.DictReader(open("${SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if str(row.get("include","")).strip().lower() in truthy))
PY
)"
EXCLUDED_CELL_IDS="$("${PYTHON_BIN}" - <<PY
import csv
truthy={"1","true","yes","y","include"}
rows=list(csv.DictReader(open("${SELECTOR}", newline="")))
print(",".join(row["cell_id"] for row in rows if str(row.get("include","")).strip().lower() not in truthy))
PY
)"

echo "Required Stage F cells: ${REQUIRED_CELL_IDS}"
echo "Excluded Stage F cells: ${EXCLUDED_CELL_IDS}"

CMD_F=(
  "${PYTHON_BIN}" apply/stages/06_fit.py
  --response-npz "${RESPONSE_NPZ}"
  --response-metadata "${RESPONSE_META}"
  --signal-npz "${STAGE_E_NPZ}"
  --signal-metadata "${STAGE_E_META}"
  --stage-c-dir "${STAGE_C_DIR}"
  --cell-subset-csv "${SELECTOR}"
  --output-dir "${STAGE_F_DIR}"
  --run-id "${STAGE_F_RUN}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name "${STAGE_F_NPZ_NAME}"
  --metadata-name "${STAGE_F_META_NAME}"
  --summary-csv-name "fit_${RUN_ID}_summary.csv"
  --summary-md-name "fit_${RUN_ID}_summary.md"
  --report-html "apply/report/stage_f_${RUN_ID}_report.html"
  --overwrite-run-dir
  --no-promote-current
)
run_or_skip "Stage F fit" "${STAGE_F_META}" "${CMD_F[@]}"

CMD_G=(
  "${PYTHON_BIN}" apply/stages/07_sed_points.py
  --response-npz "${RESPONSE_NPZ}"
  --response-metadata "${RESPONSE_META}"
  --signal-npz "${STAGE_E_NPZ}"
  --signal-metadata "${STAGE_E_META}"
  --stage-f-npz "${STAGE_F_NPZ}"
  --stage-f-metadata "${STAGE_F_META}"
  --output-dir "${STAGE_G_DIR}"
  --run-id "${STAGE_G_RUN}"
  --baseline-name "${RUN_ID}"
  --required-cell-ids "${REQUIRED_CELL_IDS}"
  --excluded-cell-ids "${EXCLUDED_CELL_IDS}"
  --skip-expected-stage-f-validation
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name "${STAGE_G_NPZ_NAME}"
  --metadata-name "${STAGE_G_META_NAME}"
  --summary-csv-name "sed_points_${RUN_ID}_summary.csv"
  --summary-json-name "sed_points_${RUN_ID}_summary.json"
  --summary-md-name "sed_points_${RUN_ID}_summary.md"
  --report-html "apply/report/stage_g_${RUN_ID}_report.html"
  --overwrite-run-dir
  --no-promote-current
)
run_or_skip "Stage G SED" "${STAGE_G_META}" "${CMD_G[@]}"

echo "===== Build v4 baseline report ====="
"${PYTHON_BIN}" apply/report/build_v4_annnorm_report.py

echo "v4 split56 ridge pipeline finished"
