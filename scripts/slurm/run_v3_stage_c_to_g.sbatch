#!/bin/bash
#SBATCH -J v3_c_to_g
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
BASELINE_SELECTOR=${BASELINE_SELECTOR:-apply/config/cell_selector_v3_baseline.csv}
SYSTEMATICS_SELECTOR=${SYSTEMATICS_SELECTOR:-apply/config/cell_selector_v3_systematics.csv}
HIGH_ENERGY_SELECTOR=${HIGH_ENERGY_SELECTOR:-apply/config/cell_selector_v3_high_energy_probes.csv}
STAGE_A_DIR=${STAGE_A_DIR:-apply/output/stage_a_v3_candidate}
STAGE_B_DIR=${STAGE_B_DIR:-apply/output/stage_b_v3_candidate}
STAGE_C_DIR=${STAGE_C_DIR:-apply/output/stage_c_v3_candidate}
STAGE_D_DIR=${STAGE_D_DIR:-apply/output/stage_d_v3_candidate}
STAGE_E_DIR=${STAGE_E_DIR:-apply/output/stage_e_v3_candidate}
STAGE_F_DIR=${STAGE_F_DIR:-apply/output/stage_f_v3_baseline}
STAGE_G_DIR=${STAGE_G_DIR:-apply/output/stage_g_v3_baseline}
RUN_SUFFIX=${RUN_SUFFIX:-slurm_${SLURM_JOB_ID}}
RUN_C=${RUN_C:-v3_stage_c_${RUN_SUFFIX}}
RUN_D=${RUN_D:-v3_stage_d_${RUN_SUFFIX}}
RUN_E=${RUN_E:-v3_stage_e_${RUN_SUFFIX}}
RUN_F=${RUN_F:-v3_stage_f_${RUN_SUFFIX}}
RUN_G=${RUN_G:-v3_stage_g_${RUN_SUFFIX}}
STAGE_C_RUN_DIR="${STAGE_C_DIR}/runs/${RUN_C}"
STAGE_D_RUN_DIR="${STAGE_D_DIR}/runs/${RUN_D}"
STAGE_E_RUN_DIR="${STAGE_E_DIR}/runs/${RUN_E}"
STAGE_F_RUN_DIR="${STAGE_F_DIR}/runs/${RUN_F}"
STAGE_G_RUN_DIR="${STAGE_G_DIR}/runs/${RUN_G}"

OBS_ROOT=${OBS_ROOT:-/mnt/mydisk/WCDA_observation_eval}
TIME_ROOT=${TIME_ROOT:-/mnt/mydisk/WCDA_observation_eval/recovered_time}
MAX_FILES=${MAX_FILES:-}
MAX_BATCHES=${MAX_BATCHES:-}

echo "===== Stage C v3 candidate observation reduction ====="
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
  --print-every 25
)
if [[ -n "${MAX_FILES}" ]]; then
  CMD_C+=(--max-files "${MAX_FILES}")
fi
printf '%q ' "${CMD_C[@]}"
echo
"${CMD_C[@]}"

echo "===== Stage D v3 annulus quadratic surface background ====="
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
  --npz-name background_v3_candidate.npz
  --metadata-name background_v3_candidate_metadata.json
  --summary-csv-name background_v3_candidate_summary.csv
  --summary-md-name background_v3_candidate_summary.md
  --no-promote-current
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_D+=(--max-batches "${MAX_BATCHES}")
fi
printf '%q ' "${CMD_D[@]}"
echo
"${CMD_D[@]}"

echo "===== Stage E v3 candidate signal ====="
CMD_E=(
  python apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --background-npz "${STAGE_D_RUN_DIR}/background_v3_candidate.npz"
  --background-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_metadata.json"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_E_DIR}"
  --report-html apply/report/stage_e_v3_candidate_report.html
  --run-id "${RUN_E}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma 50
  --quality-max-total-sigma 250
  --print-every 10
  --npz-name signal_v3_candidate.npz
  --metadata-name signal_v3_candidate_metadata.json
  --summary-csv-name signal_v3_candidate_summary.csv
  --summary-md-name signal_v3_candidate_summary.md
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
if [[ ! -f "${STAGE_E_RUN_DIR}/signal_v3_candidate.npz" ]]; then
  exit "${stage_e_status}"
fi

echo "===== Stage F v3 baseline fit ====="
CMD_F=(
  python apply/stages/06_fit.py
  --response-npz "${STAGE_A_DIR}/response_2d_v3_candidate.npz"
  --response-metadata "${STAGE_A_DIR}/response_2d_v3_candidate_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v3_candidate.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v3_candidate_metadata.json"
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --cell-subset-csv "${BASELINE_SELECTOR}"
  --output-dir "${STAGE_F_DIR}"
  --report-html apply/report/stage_f_v3_baseline_report.html
  --run-id "${RUN_F}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name fit_v3_baseline.npz
  --metadata-name fit_v3_baseline_metadata.json
  --summary-csv-name fit_v3_baseline_summary.csv
  --summary-md-name fit_v3_baseline_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_F[@]}"
echo
set +e
"${CMD_F[@]}"
stage_f_status=$?
set -e
if [[ ! -f "${STAGE_F_RUN_DIR}/fit_v3_baseline.npz" ]]; then
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

echo "===== Stage G v3 baseline SED points ====="
CMD_G=(
  python apply/stages/07_sed_points.py
  --response-npz "${STAGE_A_DIR}/response_2d_v3_candidate.npz"
  --response-metadata "${STAGE_A_DIR}/response_2d_v3_candidate_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v3_candidate.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v3_candidate_metadata.json"
  --stage-f-npz "${STAGE_F_RUN_DIR}/fit_v3_baseline.npz"
  --stage-f-metadata "${STAGE_F_RUN_DIR}/fit_v3_baseline_metadata.json"
  --output-dir "${STAGE_G_DIR}"
  --report-html apply/report/stage_g_v3_baseline_report.html
  --run-id "${RUN_G}"
  --baseline-name v3_baseline
  --required-cell-ids "${REQUIRED_CELL_IDS}"
  --excluded-cell-ids "${EXCLUDED_CELL_IDS}"
  --skip-expected-stage-f-validation
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name sed_points_v3_baseline.npz
  --metadata-name sed_points_v3_baseline_metadata.json
  --summary-csv-name sed_points_v3_baseline_summary.csv
  --summary-json-name sed_points_v3_baseline_summary.json
  --summary-md-name sed_points_v3_baseline_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_G[@]}"
echo
"${CMD_G[@]}"

echo "===== v3 report diagnostic figures ====="
python apply/report/build_v3_cell_selection_diagnostics.py \
  --response-npz "${STAGE_A_DIR}/response_2d_v3_candidate.npz" \
  --ledger-csv "${CELL_LEDGER}" \
  --baseline-selector-csv "${BASELINE_SELECTOR}" \
  --systematics-selector-csv "${SYSTEMATICS_SELECTOR}" \
  --high-energy-selector-csv "${HIGH_ENERGY_SELECTOR}" \
  --output-dir apply/report/assets/v3-cell-selection \
  --html apply/report/v3_cell_selection_diagnostics.html

python apply/report/build_fit_cell_counts_skymap.py \
  --stage-d-npz "${STAGE_D_RUN_DIR}/background_v3_candidate.npz" \
  --stage-d-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_metadata.json" \
  --selector-csv "${BASELINE_SELECTOR}" \
  --map-key counts_map \
  --output-png apply/report/assets/crab-v3-baseline-fit-cell-skymaps/crab_v3_baseline_fit_counts_grid.png \
  --output-metadata apply/report/assets/crab-v3-baseline-fit-cell-skymaps/crab_v3_baseline_fit_counts_grid_meta.json \
  --title "v3_baseline fit-cell Stage D counts maps" \
  --roi-circle-deg 6 \
  --crop-radius-deg 6 \
  --circular-roi-mask

python apply/report/build_fit_cell_counts_skymap.py \
  --stage-d-npz "${STAGE_D_RUN_DIR}/background_v3_candidate.npz" \
  --stage-d-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_metadata.json" \
  --selector-csv "${BASELINE_SELECTOR}" \
  --map-key excess_map \
  --output-png apply/report/assets/crab-v3-baseline-fit-cell-skymaps/crab_v3_baseline_fit_excess_grid.png \
  --output-metadata apply/report/assets/crab-v3-baseline-fit-cell-skymaps/crab_v3_baseline_fit_excess_grid_meta.json \
  --title "v3_baseline fit-cell Stage D excess maps" \
  --roi-circle-deg 6 \
  --crop-radius-deg 6 \
  --circular-roi-mask

python apply/report/build_fit_cell_profile_diagnostics.py \
  --stage-d-npz "${STAGE_D_RUN_DIR}/background_v3_candidate.npz" \
  --stage-d-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_metadata.json" \
  --selector-csv "${BASELINE_SELECTOR}" \
  --map-key counts_map \
  --output-dir apply/report/assets/crab-v3-baseline-fit-cell-profiles \
  --output-prefix crab_v3_baseline_fit \
  --title-prefix "v3_baseline fit-cell"

python apply/report/build_fit_cell_profile_diagnostics.py \
  --stage-d-npz "${STAGE_D_RUN_DIR}/background_v3_candidate.npz" \
  --stage-d-metadata "${STAGE_D_RUN_DIR}/background_v3_candidate_metadata.json" \
  --selector-csv "${BASELINE_SELECTOR}" \
  --map-key excess_map \
  --output-dir apply/report/assets/crab-v3-baseline-fit-cell-profiles \
  --output-prefix crab_v3_baseline_fit \
  --title-prefix "v3_baseline fit-cell"

python apply/report/build_v3_full_report.py \
  --stage-a-dir "${STAGE_A_DIR}" \
  --stage-b-dir "${STAGE_B_DIR}/current" \
  --stage-c-dir "${STAGE_C_RUN_DIR}" \
  --stage-d-dir "${STAGE_D_RUN_DIR}" \
  --stage-e-dir "${STAGE_E_RUN_DIR}" \
  --stage-f-dir "${STAGE_F_RUN_DIR}" \
  --stage-g-dir "${STAGE_G_RUN_DIR}" \
  --raw-ledger-csv "${CELL_LEDGER}" \
  --baseline-selector-csv "${BASELINE_SELECTOR}" \
  --systematics-selector-csv "${SYSTEMATICS_SELECTOR}" \
  --output-html apply/report/crab_sed_v3_stage_a_to_g_report.html

echo "v3 Stage C-G chain finished"
