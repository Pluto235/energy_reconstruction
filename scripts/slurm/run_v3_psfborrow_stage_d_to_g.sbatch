#!/bin/bash
#SBATCH -J v3_psfborrow_dg
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH -t 36:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

CELL_LEDGER=${CELL_LEDGER:-apply/config/cell_ledger_v3_candidate.csv}
BASELINE_SELECTOR=${BASELINE_SELECTOR:-apply/config/cell_selector_v3_baseline_psfborrow.csv}
SYSTEMATICS_SELECTOR=${SYSTEMATICS_SELECTOR:-apply/config/cell_selector_v3_systematics.csv}
HIGH_ENERGY_SELECTOR=${HIGH_ENERGY_SELECTOR:-apply/config/cell_selector_v3_high_energy_probes.csv}
STAGE_A_DIR=${STAGE_A_DIR:-apply/output/stage_a_v3_candidate}
STAGE_B_DIR=${STAGE_B_DIR:-apply/output/stage_b_v3_candidate_psfborrow}
STAGE_C_RUN_DIR=${STAGE_C_RUN_DIR:-apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024}
STAGE_D_DIR=${STAGE_D_DIR:-apply/output/stage_d_v3_candidate_psfborrow}
STAGE_E_DIR=${STAGE_E_DIR:-apply/output/stage_e_v3_candidate_psfborrow}
STAGE_F_DIR=${STAGE_F_DIR:-apply/output/stage_f_v3_baseline_psfborrow}
STAGE_G_DIR=${STAGE_G_DIR:-apply/output/stage_g_v3_baseline_psfborrow}
RUN_SUFFIX=${RUN_SUFFIX:-slurm_${SLURM_JOB_ID}}
RUN_D=${RUN_D:-v3_stage_d_psfborrow_${RUN_SUFFIX}}
RUN_E=${RUN_E:-v3_stage_e_psfborrow_${RUN_SUFFIX}}
RUN_F=${RUN_F:-v3_stage_f_psfborrow_${RUN_SUFFIX}}
RUN_G=${RUN_G:-v3_stage_g_psfborrow_${RUN_SUFFIX}}
STAGE_D_RUN_DIR="${STAGE_D_DIR}/runs/${RUN_D}"
STAGE_E_RUN_DIR="${STAGE_E_DIR}/runs/${RUN_E}"
STAGE_F_RUN_DIR="${STAGE_F_DIR}/runs/${RUN_F}"
STAGE_G_RUN_DIR="${STAGE_G_DIR}/runs/${RUN_G}"

MAX_BATCHES=${MAX_BATCHES:-}

if [[ ! -d "${STAGE_C_RUN_DIR}/obs_events" ]]; then
  echo "Stage C input is missing: ${STAGE_C_RUN_DIR}/obs_events" >&2
  exit 1
fi
if [[ ! -f "${STAGE_B_DIR}/current/psf_v3_candidate.npz" ]]; then
  echo "PSF borrowing Stage B current is missing: ${STAGE_B_DIR}/current/psf_v3_candidate.npz" >&2
  exit 1
fi
if [[ ! -f "${BASELINE_SELECTOR}" ]]; then
  echo "PSF borrowing selector is missing: ${BASELINE_SELECTOR}" >&2
  exit 1
fi

echo "===== Stage D v3 PSF-borrow annulus quadratic surface background ====="
CMD_D=(
  python apply/stages/04_background.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --psf-npz "${STAGE_B_DIR}/current/psf_v3_candidate.npz"
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
  --roi-surface-order 2
  --batch-size 500000
  --workers "${SLURM_CPUS_PER_TASK:-18}"
  --print-every 10
  --npz-name background_v3_candidate_psfborrow.npz
  --metadata-name background_v3_candidate_psfborrow_metadata.json
  --summary-csv-name background_v3_candidate_psfborrow_summary.csv
  --summary-md-name background_v3_candidate_psfborrow_summary.md
  --no-promote-current
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_D+=(--max-batches "${MAX_BATCHES}")
fi
printf '%q ' "${CMD_D[@]}"
echo
"${CMD_D[@]}"

echo "===== Stage E v3 PSF-borrow signal ====="
CMD_E=(
  python apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --background-npz "${STAGE_D_RUN_DIR}/background_v3_candidate_psfborrow.npz"
  --background-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_psfborrow_metadata.json"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_E_DIR}"
  --report-html apply/report/stage_e_v3_candidate_psfborrow_report.html
  --run-id "${RUN_E}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma 50
  --quality-max-total-sigma 250
  --print-every 10
  --npz-name signal_v3_candidate_psfborrow.npz
  --metadata-name signal_v3_candidate_psfborrow_metadata.json
  --summary-csv-name signal_v3_candidate_psfborrow_summary.csv
  --summary-md-name signal_v3_candidate_psfborrow_summary.md
  --no-promote-current
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
if [[ ! -f "${STAGE_E_RUN_DIR}/signal_v3_candidate_psfborrow.npz" ]]; then
  exit "${stage_e_status}"
fi

echo "===== Stage F v3 PSF-borrow baseline fit ====="
CMD_F=(
  python apply/stages/06_fit.py
  --response-npz "${STAGE_A_DIR}/response_2d_v3_candidate.npz"
  --response-metadata "${STAGE_A_DIR}/response_2d_v3_candidate_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v3_candidate_psfborrow.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v3_candidate_psfborrow_metadata.json"
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --cell-subset-csv "${BASELINE_SELECTOR}"
  --output-dir "${STAGE_F_DIR}"
  --report-html apply/report/stage_f_v3_baseline_psfborrow_report.html
  --run-id "${RUN_F}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name fit_v3_baseline_psfborrow.npz
  --metadata-name fit_v3_baseline_psfborrow_metadata.json
  --summary-csv-name fit_v3_baseline_psfborrow_summary.csv
  --summary-md-name fit_v3_baseline_psfborrow_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_F[@]}"
echo
set +e
"${CMD_F[@]}"
stage_f_status=$?
set -e
if [[ ! -f "${STAGE_F_RUN_DIR}/fit_v3_baseline_psfborrow.npz" ]]; then
  exit "${stage_f_status}"
fi

REQUIRED_CELL_IDS=$(
  python - <<PY
import csv
rows=list(csv.DictReader(open("${BASELINE_SELECTOR}")))
print(",".join(row["cell_id"] for row in rows if str(row["include"]).strip() in {"1","true","True","yes","include"}))
PY
)
EXCLUDED_CELL_IDS=$(
  python - <<PY
import csv
rows=list(csv.DictReader(open("${BASELINE_SELECTOR}")))
print(",".join(row["cell_id"] for row in rows if str(row["include"]).strip() not in {"1","true","True","yes","include"}))
PY
)

echo "===== Stage G v3 PSF-borrow SED points ====="
CMD_G=(
  python apply/stages/07_sed_points.py
  --response-npz "${STAGE_A_DIR}/response_2d_v3_candidate.npz"
  --response-metadata "${STAGE_A_DIR}/response_2d_v3_candidate_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v3_candidate_psfborrow.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v3_candidate_psfborrow_metadata.json"
  --stage-f-npz "${STAGE_F_RUN_DIR}/fit_v3_baseline_psfborrow.npz"
  --stage-f-metadata "${STAGE_F_RUN_DIR}/fit_v3_baseline_psfborrow_metadata.json"
  --output-dir "${STAGE_G_DIR}"
  --report-html apply/report/stage_g_v3_baseline_psfborrow_report.html
  --run-id "${RUN_G}"
  --baseline-name v3_baseline_psfborrow
  --required-cell-ids "${REQUIRED_CELL_IDS}"
  --excluded-cell-ids "${EXCLUDED_CELL_IDS}"
  --skip-expected-stage-f-validation
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name sed_points_v3_baseline_psfborrow.npz
  --metadata-name sed_points_v3_baseline_psfborrow_metadata.json
  --summary-csv-name sed_points_v3_baseline_psfborrow_summary.csv
  --summary-json-name sed_points_v3_baseline_psfborrow_summary.json
  --summary-md-name sed_points_v3_baseline_psfborrow_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_G[@]}"
echo
"${CMD_G[@]}"

echo "===== v3 PSF-borrow report diagnostic figures ====="
python apply/report/build_fit_cell_counts_skymap.py \
  --stage-d-npz "${STAGE_D_RUN_DIR}/background_v3_candidate_psfborrow.npz" \
  --stage-d-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_psfborrow_metadata.json" \
  --selector-csv "${BASELINE_SELECTOR}" \
  --map-key counts_map \
  --output-png apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_counts_grid.png \
  --output-metadata apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_counts_grid_meta.json \
  --title "v3_baseline_psfborrow fit-cell Stage D counts maps" \
  --roi-circle-deg 6 \
  --crop-radius-deg 6 \
  --circular-roi-mask

python apply/report/build_fit_cell_counts_skymap.py \
  --stage-d-npz "${STAGE_D_RUN_DIR}/background_v3_candidate_psfborrow.npz" \
  --stage-d-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_psfborrow_metadata.json" \
  --selector-csv "${BASELINE_SELECTOR}" \
  --map-key excess_map \
  --output-png apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_excess_grid.png \
  --output-metadata apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_excess_grid_meta.json \
  --title "v3_baseline_psfborrow fit-cell Stage D excess maps" \
  --roi-circle-deg 6 \
  --crop-radius-deg 6 \
  --circular-roi-mask

python apply/report/build_v3_full_report.py \
  --stage-a-dir "${STAGE_A_DIR}" \
  --stage-b-dir apply/output/stage_b_v3_candidate/current \
  --stage-c-dir "${STAGE_C_RUN_DIR}" \
  --stage-d-dir apply/output/stage_d_v3_candidate/runs/v3_stage_d_slurm_42024 \
  --stage-e-dir apply/output/stage_e_v3_candidate/runs/v3_stage_e_slurm_42024 \
  --stage-f-dir apply/output/stage_f_v3_baseline/runs/v3_stage_f_slurm_42024 \
  --stage-g-dir apply/output/stage_g_v3_baseline/runs/v3_stage_g_slurm_42024 \
  --raw-ledger-csv "${CELL_LEDGER}" \
  --baseline-selector-csv apply/config/cell_selector_v3_baseline.csv \
  --systematics-selector-csv "${SYSTEMATICS_SELECTOR}" \
  --high-energy-selector-csv "${HIGH_ENERGY_SELECTOR}" \
  --psfborrow-stage-b-dir "${STAGE_B_DIR}/current" \
  --psfborrow-stage-d-dir "${STAGE_D_RUN_DIR}" \
  --psfborrow-stage-e-dir "${STAGE_E_RUN_DIR}" \
  --psfborrow-stage-f-dir "${STAGE_F_RUN_DIR}" \
  --psfborrow-stage-g-dir "${STAGE_G_RUN_DIR}" \
  --psfborrow-selector-csv "${BASELINE_SELECTOR}" \
  --psfborrow-stage-f-metadata-name fit_v3_baseline_psfborrow_metadata.json \
  --psfborrow-stage-g-metadata-name sed_points_v3_baseline_psfborrow_metadata.json \
  --psfborrow-stage-f-report-html apply/report/stage_f_v3_baseline_psfborrow_report.html \
  --psfborrow-stage-g-report-html apply/report/stage_g_v3_baseline_psfborrow_report.html \
  --psfborrow-fit-cell-counts-skymap apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_counts_grid.png \
  --psfborrow-fit-cell-excess-skymap apply/report/assets/crab-v3-psfborrow-fit-cell-skymaps/crab_v3_psfborrow_fit_excess_grid.png \
  --output-html apply/report/crab_sed_v3_stage_a_to_g_report.html

echo "v3 PSF-borrow Stage D-G chain finished"
