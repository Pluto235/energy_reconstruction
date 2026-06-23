#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE_C_DIR="${STAGE_C_DIR:-apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024}"
DROP4_SELECTOR="${DROP4_SELECTOR:-apply/config/cell_selector_v4_drop4_psfborrow.csv}"
APERTURE_RESPONSE_DIR="${APERTURE_RESPONSE_DIR:-apply/output/stage_a_v4_aperture_conditioned}"
APERTURE_RESPONSE_NPZ="${APERTURE_RESPONSE_DIR}/response_2d_v4_aperture_conditioned.npz"
APERTURE_RESPONSE_META="${APERTURE_RESPONSE_DIR}/response_2d_v4_aperture_conditioned_metadata.json"
CONTAINMENT1_SIGNAL_DIR="${CONTAINMENT1_SIGNAL_DIR:-apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow}"
CONTAINMENT1_SIGNAL_NPZ="${CONTAINMENT1_SIGNAL_DIR}/signal_v4_containment1_annnorm.npz"
CONTAINMENT1_SIGNAL_META="${CONTAINMENT1_SIGNAL_DIR}/signal_v4_containment1_annnorm_metadata.json"

STAGE_F_OUT_DIR="${STAGE_F_OUT_DIR:-apply/output/stage_f_v4_aperture_conditioned}"
STAGE_F_RUN_ID="${STAGE_F_RUN_ID:-v4_stage_f_aperture_conditioned_drop4}"
STAGE_F_NPZ="${STAGE_F_OUT_DIR}/runs/${STAGE_F_RUN_ID}/fit_v4_aperture_conditioned_drop4.npz"
STAGE_F_META="${STAGE_F_OUT_DIR}/runs/${STAGE_F_RUN_ID}/fit_v4_aperture_conditioned_drop4_metadata.json"

STAGE_G_OUT_DIR="${STAGE_G_OUT_DIR:-apply/output/stage_g_v4_aperture_conditioned}"
STAGE_G_RUN_ID="${STAGE_G_RUN_ID:-v4_stage_g_aperture_conditioned_drop4}"

for required in \
  "${APERTURE_RESPONSE_NPZ}" \
  "${APERTURE_RESPONSE_META}" \
  "${CONTAINMENT1_SIGNAL_NPZ}" \
  "${CONTAINMENT1_SIGNAL_META}" \
  "${DROP4_SELECTOR}" \
  "${STAGE_C_DIR}"
do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" apply/stages/06_fit.py \
  --response-npz "${APERTURE_RESPONSE_NPZ}" \
  --response-metadata "${APERTURE_RESPONSE_META}" \
  --signal-npz "${CONTAINMENT1_SIGNAL_NPZ}" \
  --signal-metadata "${CONTAINMENT1_SIGNAL_META}" \
  --stage-c-dir "${STAGE_C_DIR}" \
  --cell-subset-csv "${DROP4_SELECTOR}" \
  --output-dir "${STAGE_F_OUT_DIR}" \
  --run-id "${STAGE_F_RUN_ID}" \
  --npz-name fit_v4_aperture_conditioned_drop4.npz \
  --metadata-name fit_v4_aperture_conditioned_drop4_metadata.json \
  --summary-csv-name fit_v4_aperture_conditioned_drop4_summary.csv \
  --summary-md-name fit_v4_aperture_conditioned_drop4_summary.md \
  --report-html apply/report/stage_f_v4_aperture_conditioned_drop4_report.html \
  --overwrite-run-dir

"${PYTHON_BIN}" apply/stages/07_sed_points.py \
  --response-npz "${APERTURE_RESPONSE_NPZ}" \
  --response-metadata "${APERTURE_RESPONSE_META}" \
  --signal-npz "${CONTAINMENT1_SIGNAL_NPZ}" \
  --signal-metadata "${CONTAINMENT1_SIGNAL_META}" \
  --stage-f-npz "${STAGE_F_NPZ}" \
  --stage-f-metadata "${STAGE_F_META}" \
  --output-dir "${STAGE_G_OUT_DIR}" \
  --run-id "${STAGE_G_RUN_ID}" \
  --baseline-name v4_aperture_conditioned_drop4 \
  --required-cell-ids "" \
  --excluded-cell-ids "4,17,39,43" \
  --skip-expected-stage-f-validation \
  --npz-name sed_points_v4_aperture_conditioned_drop4.npz \
  --metadata-name sed_points_v4_aperture_conditioned_drop4_metadata.json \
  --summary-csv-name sed_points_v4_aperture_conditioned_drop4_summary.csv \
  --summary-json-name sed_points_v4_aperture_conditioned_drop4_summary.json \
  --summary-md-name sed_points_v4_aperture_conditioned_drop4_summary.md \
  --report-html apply/report/stage_g_v4_aperture_conditioned_drop4_report.html \
  --overwrite-run-dir

"${PYTHON_BIN}" apply/report/build_v4_response_audit.py
"${PYTHON_BIN}" apply/report/build_v4_annnorm_report.py
