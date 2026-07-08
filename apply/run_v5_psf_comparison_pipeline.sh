#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x /home/server/anaconda3/envs/py310/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/server/anaconda3/envs/py310/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

METHODS="${METHODS:-rayleigh_baseline two_1d_gaussian mc_quantile_715 observed_data double_rayleigh_mixture}"
STAGE_B_METHOD="${STAGE_B_METHOD:-double_rayleigh_mixture}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-18}}"
DENOMINATOR_WORKERS="${DENOMINATOR_WORKERS:-1}"
MAX_FILES_PER_CELL="${MAX_FILES_PER_CELL:-}"
MAX_BATCHES="${MAX_BATCHES:-}"
RESUME="${RESUME:-1}"
QUALITY_MIN_TOTAL_SIGMA="${QUALITY_MIN_TOTAL_SIGMA:--300}"
QUALITY_MAX_TOTAL_SIGMA="${QUALITY_MAX_TOTAL_SIGMA:-300}"
OBSERVED_PEDESTAL_MIN_DEG="${OBSERVED_PEDESTAL_MIN_DEG:-2.5}"
OBSERVED_MAX_R_OPT_OVER_RAYLEIGH="${OBSERVED_MAX_R_OPT_OVER_RAYLEIGH:-2.5}"
OBSERVED_MAX_R_OPT_OVER_MC_QUANTILE="${OBSERVED_MAX_R_OPT_OVER_MC_QUANTILE:-2.0}"
OBSERVED_MAX_R_OPT_DEG="${OBSERVED_MAX_R_OPT_DEG:-2.0}"
OBSERVED_MIN_POSITIVE_TOTAL="${OBSERVED_MIN_POSITIVE_TOTAL:-100}"
OBSERVED_REQUIRE_RELIABLE="${OBSERVED_REQUIRE_RELIABLE:-1}"

BINNED_ROOT="${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v3_candidate}"
CELL_LEDGER="${CELL_LEDGER:-apply/config/cell_ledger_v3_candidate.csv}"
DROP4_SELECTOR="${DROP4_SELECTOR:-apply/config/cell_selector_v4_drop4_psfborrow.csv}"
STAGE_A_BASE_METADATA="${STAGE_A_BASE_METADATA:-apply/output/stage_a_v3_candidate/response_2d_v3_candidate_metadata.json}"
STAGE_C_DIR="${STAGE_C_DIR:-apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024}"

SOURCE_STAGE_D_NPZ="${SOURCE_STAGE_D_NPZ:-apply/output/stage_d_v3_candidate_psfborrow/runs/v3_stage_d_psfborrow_slurm_42029/background_v3_candidate_psfborrow.npz}"
SOURCE_STAGE_D_META="${SOURCE_STAGE_D_META:-apply/output/stage_d_v3_candidate_psfborrow/runs/v3_stage_d_psfborrow_slurm_42029/background_v3_candidate_psfborrow_metadata.json}"
SOURCE_STAGE_E_NPZ="${SOURCE_STAGE_E_NPZ:-apply/output/stage_e_v3_candidate_annnorm/runs/v3_stage_e_annnorm_from_psfborrow/signal_v3_candidate_annnorm.npz}"
SOURCE_STAGE_E_META="${SOURCE_STAGE_E_META:-apply/output/stage_e_v3_candidate_annnorm/runs/v3_stage_e_annnorm_from_psfborrow/signal_v3_candidate_annnorm_metadata.json}"

STAGE_B_DIR="${STAGE_B_DIR:-apply/output/stage_b_v5_psf_compare}"
STAGE_A_DIR="${STAGE_A_DIR:-apply/output/stage_a_v5_psf_compare}"
STAGE_D_DIR="${STAGE_D_DIR:-apply/output/stage_d_v5_psf_compare}"
STAGE_E_DIR="${STAGE_E_DIR:-apply/output/stage_e_v5_psf_compare}"
STAGE_F_DIR="${STAGE_F_DIR:-apply/output/stage_f_v5_psf_compare}"
STAGE_G_DIR="${STAGE_G_DIR:-apply/output/stage_g_v5_psf_compare}"
DERIVED_UNUSED_E_DIR="${DERIVED_UNUSED_E_DIR:-apply/output/stage_e_v5_psf_compare_derived_unused}"

for required in \
  "${BINNED_ROOT}" \
  "${CELL_LEDGER}" \
  "${DROP4_SELECTOR}" \
  "${STAGE_A_BASE_METADATA}" \
  "${STAGE_C_DIR}" \
  "${SOURCE_STAGE_D_NPZ}" \
  "${SOURCE_STAGE_D_META}" \
  "${SOURCE_STAGE_E_NPZ}" \
  "${SOURCE_STAGE_E_META}"
do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

REQUIRED_CELL_IDS="$("${PYTHON_BIN}" - <<PY
import csv
rows=list(csv.DictReader(open("${DROP4_SELECTOR}", newline="")))
truthy={"1","true","yes","y","include"}
print(",".join(row["cell_id"] for row in rows if str(row.get("include","")).strip().lower() in truthy))
PY
)"
EXCLUDED_CELL_IDS="$("${PYTHON_BIN}" - <<PY
import csv
rows=list(csv.DictReader(open("${DROP4_SELECTOR}", newline="")))
truthy={"1","true","yes","y","include"}
print(",".join(row["cell_id"] for row in rows if str(row.get("include","")).strip().lower() not in truthy))
PY
)"

echo "Python: ${PYTHON_BIN}"
echo "Methods: ${METHODS}"
echo "Stage B method: ${STAGE_B_METHOD}"
echo "Workers: ${WORKERS}"
echo "Resume: ${RESUME}"
echo "Stage E total-sigma quality window: [${QUALITY_MIN_TOTAL_SIGMA}, ${QUALITY_MAX_TOTAL_SIGMA}]"
if [[ "${STAGE_B_METHOD}" == "observed_data" || "${METHODS}" == *"observed_data"* ]]; then
  echo "Observed-data gates: pedestal_min=${OBSERVED_PEDESTAL_MIN_DEG} deg, max_r/rayleigh=${OBSERVED_MAX_R_OPT_OVER_RAYLEIGH}, max_r/mc=${OBSERVED_MAX_R_OPT_OVER_MC_QUANTILE}, max_r=${OBSERVED_MAX_R_OPT_DEG} deg, min_positive=${OBSERVED_MIN_POSITIVE_TOTAL}, require_reliable=${OBSERVED_REQUIRE_RELIABLE}"
fi
echo "Required Stage F cells: ${REQUIRED_CELL_IDS}"
echo "Excluded Stage F cells: ${EXCLUDED_CELL_IDS}"

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

if [[ "${STAGE_B_METHOD}" == "all" ]]; then
  STAGE_B_RUN_ID="v5_psf_all_drop4"
  STAGE_B_NPZ_NAME="psf_v5_psf_all_drop4.npz"
  STAGE_B_METADATA_NAME="psf_v5_psf_all_drop4_metadata.json"
  STAGE_B_SUMMARY_CSV_NAME="psf_v5_psf_all_drop4_summary.csv"
  STAGE_B_SUMMARY_MD_NAME="psf_v5_psf_all_drop4_summary.md"
  STAGE_B_SENTINEL="${STAGE_B_DIR}/runs/v5_psf_double_rayleigh_mixture_drop4/psf_v5_psf_double_rayleigh_mixture_drop4_metadata.json"
  STAGE_B_LABEL="Stage B v5 PSF all methods"
else
  STAGE_B_RUN_ID="v5_psf_${STAGE_B_METHOD}_drop4"
  STAGE_B_NPZ_NAME="psf_${STAGE_B_RUN_ID}.npz"
  STAGE_B_METADATA_NAME="psf_${STAGE_B_RUN_ID}_metadata.json"
  STAGE_B_SUMMARY_CSV_NAME="psf_${STAGE_B_RUN_ID}_summary.csv"
  STAGE_B_SUMMARY_MD_NAME="psf_${STAGE_B_RUN_ID}_summary.md"
  STAGE_B_SENTINEL="${STAGE_B_DIR}/runs/${STAGE_B_RUN_ID}/${STAGE_B_METADATA_NAME}"
  STAGE_B_LABEL="Stage B v5 PSF ${STAGE_B_METHOD}"
fi
OBSERVED_STAGE_B_SOURCE_NPZ="${OBSERVED_STAGE_B_SOURCE_NPZ:-${STAGE_B_DIR}/runs/v5_psf_rayleigh_baseline_drop4/psf_v5_psf_rayleigh_baseline_drop4.npz}"
OBSERVED_STAGE_B_SOURCE_METADATA="${OBSERVED_STAGE_B_SOURCE_METADATA:-${STAGE_B_DIR}/runs/v5_psf_rayleigh_baseline_drop4/psf_v5_psf_rayleigh_baseline_drop4_metadata.json}"
DOUBLE_RAYLEIGH_STAGE_B_SOURCE_NPZ="${DOUBLE_RAYLEIGH_STAGE_B_SOURCE_NPZ:-${STAGE_B_DIR}/runs/v5_psf_rayleigh_baseline_drop4/psf_v5_psf_rayleigh_baseline_drop4.npz}"
DOUBLE_RAYLEIGH_STAGE_B_SOURCE_METADATA="${DOUBLE_RAYLEIGH_STAGE_B_SOURCE_METADATA:-${STAGE_B_DIR}/runs/v5_psf_rayleigh_baseline_drop4/psf_v5_psf_rayleigh_baseline_drop4_metadata.json}"

CMD_B=(
  "${PYTHON_BIN}" apply/stages/02e_build_psf_v5_compare.py
  --psf-method "${STAGE_B_METHOD}"
  --binned-root "${BINNED_ROOT}"
  --cell-selection-csv "${CELL_LEDGER}"
  --stage-a-metadata "${STAGE_A_BASE_METADATA}"
  --output-dir "${STAGE_B_DIR}"
  --run-id "${STAGE_B_RUN_ID}"
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
  --observed-pedestal-min-deg "${OBSERVED_PEDESTAL_MIN_DEG}"
  --observed-max-r-opt-over-rayleigh "${OBSERVED_MAX_R_OPT_OVER_RAYLEIGH}"
  --observed-max-r-opt-over-mc-quantile "${OBSERVED_MAX_R_OPT_OVER_MC_QUANTILE}"
  --observed-max-r-opt-deg "${OBSERVED_MAX_R_OPT_DEG}"
  --observed-min-positive-total "${OBSERVED_MIN_POSITIVE_TOTAL}"
  --workers "${WORKERS}"
  --npz-name "${STAGE_B_NPZ_NAME}"
  --metadata-name "${STAGE_B_METADATA_NAME}"
  --summary-csv-name "${STAGE_B_SUMMARY_CSV_NAME}"
  --summary-md-name "${STAGE_B_SUMMARY_MD_NAME}"
  --no-plots
  --overwrite-run-dir
  --no-promote-current
)
if [[ "${OBSERVED_REQUIRE_RELIABLE}" =~ ^(0|false|False|no|NO)$ ]]; then
  CMD_B+=(--no-observed-require-reliable)
fi
if [[ -n "${MAX_FILES_PER_CELL}" ]]; then
  CMD_B+=(--max-files-per-cell "${MAX_FILES_PER_CELL}")
fi
if [[ "${STAGE_B_METHOD}" == "observed_data" ]]; then
  CMD_B+=(
    --observed-stage-b-source-npz "${OBSERVED_STAGE_B_SOURCE_NPZ}"
    --observed-stage-b-source-metadata "${OBSERVED_STAGE_B_SOURCE_METADATA}"
  )
fi
if [[ "${STAGE_B_METHOD}" == "double_rayleigh_mixture" ]]; then
  CMD_B+=(
    --double-rayleigh-stage-b-source-npz "${DOUBLE_RAYLEIGH_STAGE_B_SOURCE_NPZ}"
    --double-rayleigh-stage-b-source-metadata "${DOUBLE_RAYLEIGH_STAGE_B_SOURCE_METADATA}"
  )
fi
run_or_skip "${STAGE_B_LABEL}" "${STAGE_B_SENTINEL}" "${CMD_B[@]}"

for METHOD in ${METHODS}; do
  RUN_ID="v5_psf_${METHOD}_drop4"
  PSF_NPZ_NAME="psf_${RUN_ID}.npz"
  PSF_META_NAME="psf_${RUN_ID}_metadata.json"
  PSF_NPZ="${STAGE_B_DIR}/runs/${RUN_ID}/${PSF_NPZ_NAME}"
  PSF_META="${STAGE_B_DIR}/runs/${RUN_ID}/${PSF_META_NAME}"

  STAGE_A_METHOD_DIR="${STAGE_A_DIR}/${METHOD}"
  RESPONSE_NPZ_NAME="response_2d_${RUN_ID}.npz"
  RESPONSE_META_NAME="response_2d_${RUN_ID}_metadata.json"
  RESPONSE_NPZ="${STAGE_A_METHOD_DIR}/${RESPONSE_NPZ_NAME}"
  RESPONSE_META="${STAGE_A_METHOD_DIR}/${RESPONSE_META_NAME}"

  STAGE_D_NPZ_NAME="background_${RUN_ID}.npz"
  STAGE_D_META_NAME="background_${RUN_ID}_metadata.json"
  STAGE_D_NPZ="${STAGE_D_DIR}/runs/${RUN_ID}/${STAGE_D_NPZ_NAME}"
  STAGE_D_META="${STAGE_D_DIR}/runs/${RUN_ID}/${STAGE_D_META_NAME}"

  STAGE_E_NPZ_NAME="signal_${RUN_ID}.npz"
  STAGE_E_META_NAME="signal_${RUN_ID}_metadata.json"
  STAGE_E_NPZ="${STAGE_E_DIR}/runs/${RUN_ID}/${STAGE_E_NPZ_NAME}"
  STAGE_E_META="${STAGE_E_DIR}/runs/${RUN_ID}/${STAGE_E_META_NAME}"

  STAGE_F_NPZ_NAME="fit_${RUN_ID}.npz"
  STAGE_F_META_NAME="fit_${RUN_ID}_metadata.json"
  STAGE_F_NPZ="${STAGE_F_DIR}/runs/${RUN_ID}/${STAGE_F_NPZ_NAME}"
  STAGE_F_META="${STAGE_F_DIR}/runs/${RUN_ID}/${STAGE_F_META_NAME}"

  STAGE_G_NPZ_NAME="sed_points_${RUN_ID}.npz"
  STAGE_G_META_NAME="sed_points_${RUN_ID}_metadata.json"

  CMD_A=(
    "${PYTHON_BIN}" apply/stages/01_build_response.py
    --binned-root "${BINNED_ROOT}"
    --cell-selection-csv "${CELL_LEDGER}"
    --output-dir "${STAGE_A_METHOD_DIR}"
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
  run_or_skip "Stage A aperture-conditioned response ${METHOD}" "${RESPONSE_META}" "${CMD_A[@]}"

  if [[ "${METHOD}" == "double_rayleigh_mixture" ]]; then
    CMD_A_FILL=(
      "${PYTHON_BIN}" apply/stages/01b_fill_v5_double_rayleigh_response.py
      --target-response-npz "${RESPONSE_NPZ}"
      --target-response-metadata "${RESPONSE_META}"
      --source-response-npz "${STAGE_A_DIR}/rayleigh_baseline/response_2d_v5_psf_rayleigh_baseline_drop4.npz"
      --source-response-metadata "${STAGE_A_DIR}/rayleigh_baseline/response_2d_v5_psf_rayleigh_baseline_drop4_metadata.json"
      --target-psf-npz "${PSF_NPZ}"
      --source-psf-npz "${STAGE_B_DIR}/runs/v5_psf_rayleigh_baseline_drop4/psf_v5_psf_rayleigh_baseline_drop4.npz"
    )
    echo "===== Fill zero Stage A response cells ${METHOD} ====="
    printf '%q ' "${CMD_A_FILL[@]}"; echo
    "${CMD_A_FILL[@]}"
  fi

  CMD_D=(
    "${PYTHON_BIN}" apply/report/build_v3_annnorm_from_stage_d.py
    --source-stage-d-npz "${SOURCE_STAGE_D_NPZ}"
    --source-stage-d-metadata "${SOURCE_STAGE_D_META}"
    --source-stage-e-npz "${SOURCE_STAGE_E_NPZ}"
    --source-stage-e-metadata "${SOURCE_STAGE_E_META}"
    --psf-npz "${PSF_NPZ}"
    --psf-metadata "${PSF_META}"
    --cell-selection-csv "${CELL_LEDGER}"
    --baseline-selector-csv "${DROP4_SELECTOR}"
    --stage-d-output-dir "${STAGE_D_DIR}"
    --stage-d-run-id "${RUN_ID}"
    --stage-e-output-dir "${DERIVED_UNUSED_E_DIR}"
    --stage-e-run-id "${RUN_ID}_derived_unused"
    --stage-d-npz-name "${STAGE_D_NPZ_NAME}"
    --stage-d-metadata-name "${STAGE_D_META_NAME}"
    --stage-d-summary-csv-name "background_${RUN_ID}_summary.csv"
    --stage-d-summary-md-name "background_${RUN_ID}_summary.md"
    --stage-e-npz-name "signal_${RUN_ID}_derived_unused.npz"
    --stage-e-metadata-name "signal_${RUN_ID}_derived_unused_metadata.json"
    --stage-e-summary-csv-name "signal_${RUN_ID}_derived_unused_summary.csv"
    --stage-e-summary-md-name "signal_${RUN_ID}_derived_unused_summary.md"
    --overwrite-run-dir
    --no-promote-current
    --no-plots
  )
  run_or_skip "Stage D annulus-normalized aperture background ${METHOD}" "${STAGE_D_META}" "${CMD_D[@]}"

  CMD_E=(
    "${PYTHON_BIN}" apply/stages/05_signal.py
    --stage-c-dir "${STAGE_C_DIR}"
    --background-npz "${STAGE_D_NPZ}"
    --background-metadata "${STAGE_D_META}"
    --cell-selection-csv "${CELL_LEDGER}"
    --output-dir "${STAGE_E_DIR}"
    --run-id "${RUN_ID}"
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
  run_or_skip "Stage E signal ${METHOD}" "${STAGE_E_META}" "${CMD_E[@]}"

  CMD_F=(
    "${PYTHON_BIN}" apply/stages/06_fit.py
    --response-npz "${RESPONSE_NPZ}"
    --response-metadata "${RESPONSE_META}"
    --signal-npz "${STAGE_E_NPZ}"
    --signal-metadata "${STAGE_E_META}"
    --stage-c-dir "${STAGE_C_DIR}"
    --cell-subset-csv "${DROP4_SELECTOR}"
    --output-dir "${STAGE_F_DIR}"
    --run-id "${RUN_ID}"
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
  run_or_skip "Stage F fit ${METHOD}" "${STAGE_F_META}" "${CMD_F[@]}"

  CMD_G=(
    "${PYTHON_BIN}" apply/stages/07_sed_points.py
    --response-npz "${RESPONSE_NPZ}"
    --response-metadata "${RESPONSE_META}"
    --signal-npz "${STAGE_E_NPZ}"
    --signal-metadata "${STAGE_E_META}"
    --stage-f-npz "${STAGE_F_NPZ}"
    --stage-f-metadata "${STAGE_F_META}"
    --output-dir "${STAGE_G_DIR}"
    --run-id "${RUN_ID}"
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
  run_or_skip "Stage G SED ${METHOD}" "${STAGE_G_DIR}/runs/${RUN_ID}/${STAGE_G_META_NAME}" "${CMD_G[@]}"
done

echo "===== Build v5 PSF comparison report ====="
"${PYTHON_BIN}" apply/report/build_v5_psf_comparison_report.py

echo "v5 PSF comparison pipeline finished"
