#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_SUFFIX="${RUN_SUFFIX:-v4_r68_aperture_drop4}"

STAGE_B_BASE="${STAGE_B_BASE:-apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal}"
STAGE_C_DIR="${STAGE_C_DIR:-apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024}"
SOURCE_STAGE_D_NPZ="${SOURCE_STAGE_D_NPZ:-apply/output/stage_d_v3_candidate_psfborrow/runs/v3_stage_d_psfborrow_slurm_42029/background_v3_candidate_psfborrow.npz}"
SOURCE_STAGE_D_META="${SOURCE_STAGE_D_META:-apply/output/stage_d_v3_candidate_psfborrow/runs/v3_stage_d_psfborrow_slurm_42029/background_v3_candidate_psfborrow_metadata.json}"
CELL_LEDGER="${CELL_LEDGER:-apply/config/cell_ledger_v3_candidate.csv}"
DROP4_SELECTOR="${DROP4_SELECTOR:-apply/config/cell_selector_v4_drop4_psfborrow.csv}"

PSF_OUT_DIR="apply/output/stage_b_v4_aperture_variants"
PSF_RUN_ID="v4_r68_from_psfborrow"
PSF_NPZ="${PSF_OUT_DIR}/runs/${PSF_RUN_ID}/psf_v4_r68_aperture.npz"
PSF_META="${PSF_OUT_DIR}/runs/${PSF_RUN_ID}/psf_v4_r68_aperture_metadata.json"

STAGE_D_OUT_DIR="apply/output/stage_d_v4_r68_aperture"
STAGE_D_RUN_ID="${RUN_SUFFIX}_stage_d"
STAGE_D_NPZ="${STAGE_D_OUT_DIR}/runs/${STAGE_D_RUN_ID}/background_v4_r68_aperture.npz"
STAGE_D_META="${STAGE_D_OUT_DIR}/runs/${STAGE_D_RUN_ID}/background_v4_r68_aperture_metadata.json"

STAGE_E_OUT_DIR="apply/output/stage_e_v4_r68_aperture"
STAGE_E_RUN_ID="${RUN_SUFFIX}_stage_e"
STAGE_E_NPZ="${STAGE_E_OUT_DIR}/runs/${STAGE_E_RUN_ID}/signal_v4_r68_aperture.npz"
STAGE_E_META="${STAGE_E_OUT_DIR}/runs/${STAGE_E_RUN_ID}/signal_v4_r68_aperture_metadata.json"

STAGE_F_OUT_DIR="apply/output/stage_f_v4_r68_aperture"
STAGE_F_RUN_ID="${RUN_SUFFIX}_stage_f"
STAGE_F_NPZ="${STAGE_F_OUT_DIR}/runs/${STAGE_F_RUN_ID}/fit_v4_r68_aperture.npz"
STAGE_F_META="${STAGE_F_OUT_DIR}/runs/${STAGE_F_RUN_ID}/fit_v4_r68_aperture_metadata.json"

STAGE_G_OUT_DIR="apply/output/stage_g_v4_r68_aperture"
STAGE_G_RUN_ID="${RUN_SUFFIX}_stage_g"

"${PYTHON_BIN}" apply/stages/02d_build_psf_aperture_variant.py \
  --input-dir "${STAGE_B_BASE}" \
  --output-dir "${PSF_OUT_DIR}" \
  --run-id "${PSF_RUN_ID}" \
  --overwrite-run-dir \
  --aperture-source r68 \
  --containment-fraction 0.68

"${PYTHON_BIN}" apply/report/build_v3_annnorm_from_stage_d.py \
  --source-stage-d-npz "${SOURCE_STAGE_D_NPZ}" \
  --source-stage-d-metadata "${SOURCE_STAGE_D_META}" \
  --source-stage-e-npz apply/output/stage_e_v3_candidate_annnorm/runs/v3_stage_e_annnorm_from_psfborrow/signal_v3_candidate_annnorm.npz \
  --source-stage-e-metadata apply/output/stage_e_v3_candidate_annnorm/runs/v3_stage_e_annnorm_from_psfborrow/signal_v3_candidate_annnorm_metadata.json \
  --psf-npz "${PSF_NPZ}" \
  --psf-metadata "${PSF_META}" \
  --cell-selection-csv "${CELL_LEDGER}" \
  --baseline-selector-csv "${DROP4_SELECTOR}" \
  --stage-d-output-dir "${STAGE_D_OUT_DIR}" \
  --stage-d-run-id "${STAGE_D_RUN_ID}" \
  --stage-e-output-dir apply/output/stage_e_v4_r68_aperture_tmp_unused \
  --stage-e-run-id tmp_unused \
  --stage-d-npz-name background_v4_r68_aperture.npz \
  --stage-d-metadata-name background_v4_r68_aperture_metadata.json \
  --stage-d-summary-csv-name background_v4_r68_aperture_summary.csv \
  --stage-d-summary-md-name background_v4_r68_aperture_summary.md \
  --overwrite-run-dir \
  --no-promote-current \
  --no-plots

"${PYTHON_BIN}" apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C_DIR}" \
  --background-npz "${STAGE_D_NPZ}" \
  --background-metadata "${STAGE_D_META}" \
  --cell-selection-csv "${CELL_LEDGER}" \
  --output-dir "${STAGE_E_OUT_DIR}" \
  --run-id "${STAGE_E_RUN_ID}" \
  --npz-name signal_v4_r68_aperture.npz \
  --metadata-name signal_v4_r68_aperture_metadata.json \
  --summary-csv-name signal_v4_r68_aperture_summary.csv \
  --summary-md-name signal_v4_r68_aperture_summary.md \
  --report-html apply/report/stage_e_v4_r68_aperture_report.html \
  --quality-min-total-sigma 0 \
  --quality-max-total-sigma 300 \
  --overwrite-run-dir

"${PYTHON_BIN}" apply/stages/06_fit.py \
  --response-npz apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz \
  --response-metadata apply/output/stage_a_v3_candidate/response_2d_v3_candidate_metadata.json \
  --signal-npz "${STAGE_E_NPZ}" \
  --signal-metadata "${STAGE_E_META}" \
  --stage-c-dir "${STAGE_C_DIR}" \
  --cell-subset-csv "${DROP4_SELECTOR}" \
  --output-dir "${STAGE_F_OUT_DIR}" \
  --run-id "${STAGE_F_RUN_ID}" \
  --npz-name fit_v4_r68_aperture.npz \
  --metadata-name fit_v4_r68_aperture_metadata.json \
  --summary-csv-name fit_v4_r68_aperture_summary.csv \
  --summary-md-name fit_v4_r68_aperture_summary.md \
  --report-html apply/report/stage_f_v4_r68_aperture_report.html \
  --overwrite-run-dir

"${PYTHON_BIN}" apply/stages/07_sed_points.py \
  --response-npz apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz \
  --response-metadata apply/output/stage_a_v3_candidate/response_2d_v3_candidate_metadata.json \
  --signal-npz "${STAGE_E_NPZ}" \
  --signal-metadata "${STAGE_E_META}" \
  --stage-f-npz "${STAGE_F_NPZ}" \
  --stage-f-metadata "${STAGE_F_META}" \
  --output-dir "${STAGE_G_OUT_DIR}" \
  --run-id "${STAGE_G_RUN_ID}" \
  --baseline-name v4_r68_aperture_drop4 \
  --required-cell-ids "" \
  --excluded-cell-ids "4,17,39,43" \
  --skip-expected-stage-f-validation \
  --npz-name sed_points_v4_r68_aperture.npz \
  --metadata-name sed_points_v4_r68_aperture_metadata.json \
  --summary-csv-name sed_points_v4_r68_aperture_summary.csv \
  --summary-json-name sed_points_v4_r68_aperture_summary.json \
  --summary-md-name sed_points_v4_r68_aperture_summary.md \
  --report-html apply/report/stage_g_v4_r68_aperture_report.html \
  --overwrite-run-dir
