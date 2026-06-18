#!/bin/bash
#SBATCH -J v1_nhit_only
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

SELECTOR=${SELECTOR:-apply/config/cell_selector_v1_nhit_only.csv}
SOURCE_CACHE=${SOURCE_CACHE:-/mnt/mydisk/WCDA_simulation_binned_response_v1}
NHIT_CACHE=${NHIT_CACHE:-/mnt/mydisk/WCDA_simulation_binned_response_v1_nhit_only}
STAGE_A_DIR=${STAGE_A_DIR:-apply/output/stage_a_v1_nhit_only}
STAGE_B_DIR=${STAGE_B_DIR:-apply/output/stage_b_v1_nhit_only}
STAGE_C_DIR=${STAGE_C_DIR:-apply/output/stage_c_v1_nhit_only}
STAGE_D_DIR=${STAGE_D_DIR:-apply/output/stage_d_v1_nhit_only}
STAGE_E_DIR=${STAGE_E_DIR:-apply/output/stage_e_v1_nhit_only}
STAGE_F_DIR=${STAGE_F_DIR:-apply/output/stage_f_v1_nhit_only}
STAGE_G_DIR=${STAGE_G_DIR:-apply/output/stage_g_v1_nhit_only}
DIAG_DIR=${DIAG_DIR:-apply/report/assets/v1-nhit-only-control}

RUN_SUFFIX=${RUN_SUFFIX:-slurm_${SLURM_JOB_ID}}
RUN_B=${RUN_B:-v1_stage_b_nhit_only_${RUN_SUFFIX}}
RUN_C=${RUN_C:-v1_stage_c_nhit_only_${RUN_SUFFIX}}
RUN_D=${RUN_D:-v1_stage_d_nhit_only_${RUN_SUFFIX}}
RUN_E=${RUN_E:-v1_stage_e_nhit_only_${RUN_SUFFIX}}
RUN_F=${RUN_F:-v1_stage_f_nhit_only_${RUN_SUFFIX}}
RUN_G=${RUN_G:-v1_stage_g_nhit_only_${RUN_SUFFIX}}

STAGE_B_RUN_DIR="${STAGE_B_DIR}/runs/${RUN_B}"
STAGE_C_RUN_DIR="${STAGE_C_DIR}/runs/${RUN_C}"
STAGE_D_RUN_DIR="${STAGE_D_DIR}/runs/${RUN_D}"
STAGE_E_RUN_DIR="${STAGE_E_DIR}/runs/${RUN_E}"
STAGE_F_RUN_DIR="${STAGE_F_DIR}/runs/${RUN_F}"
STAGE_G_RUN_DIR="${STAGE_G_DIR}/runs/${RUN_G}"

OBS_ROOT=${OBS_ROOT:-/mnt/mydisk/WCDA_observation_eval}
TIME_ROOT=${TIME_ROOT:-/mnt/mydisk/WCDA_observation_eval/recovered_time}
MAX_FILES=${MAX_FILES:-}
MAX_BATCHES=${MAX_BATCHES:-}

if [[ ! -f "${SELECTOR}" ]]; then
  echo "v1-bin Nhit-only selector is missing: ${SELECTOR}" >&2
  exit 1
fi

REQUIRED_CELL_IDS=$(
  python - <<PY
import csv
rows=list(csv.DictReader(open("${SELECTOR}")))
print(",".join(row["cell_id"] for row in rows if str(row.get("include", "1")).strip() in {"1","true","True","yes","include"}))
PY
)

echo "===== Build v1-bin PredE-blind Nhit-only symlink cache ====="
python apply/stages/build_v3_nhit_only_cache.py \
  --selector-csv "${SELECTOR}" \
  --source-root "${SOURCE_CACHE}" \
  --target-root "${NHIT_CACHE}" \
  --metadata-name v1_nhit_only_cache_metadata.json \
  --overwrite

echo "===== Stage A v1-bin Nhit-only response ====="
python apply/stages/01_build_response.py \
  --binned-root "${NHIT_CACHE}" \
  --cell-selection-csv "${SELECTOR}" \
  --output-dir "${STAGE_A_DIR}" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --denominator-workers 1 \
  --numerator-workers "${SLURM_CPUS_PER_TASK:-18}" \
  --numerator-files-per-task 250 \
  --numerator-progress-every 20 \
  --npz-name response_2d_v1_nhit_only.npz \
  --metadata-name response_2d_v1_nhit_only_metadata.json

echo "===== Stage B v1-bin Nhit-only PSF ====="
python apply/stages/02_build_psf.py \
  --binned-root "${NHIT_CACHE}" \
  --cell-selection-csv "${SELECTOR}" \
  --stage-a-metadata "${STAGE_A_DIR}/response_2d_v1_nhit_only_metadata.json" \
  --output-dir "${STAGE_B_DIR}" \
  --run-id "${RUN_B}" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --lhaaso-lat-deg 29.45 \
  --source-dec-deg 22.01 \
  --theta-min-deg 0.0 \
  --theta-max-deg 50.0 \
  --theta-step-deg 1.0 \
  --hour-angle-samples 200000 \
  --allow-incomplete-theta-support \
  --allow-low-stat-psf-fallback \
  --min-events-per-cell 1000 \
  --min-effective-events 200 \
  --core-fit-max-deg 3.0 \
  --theta-missing-mass-fail-threshold 0.10 \
  --containment-warning-tolerance 0.12 \
  --angle-check-max-events 20000 \
  --file-progress-every 5000 \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --npz-name psf_v1_nhit_only.npz \
  --metadata-name psf_v1_nhit_only_metadata.json \
  --summary-csv-name psf_v1_nhit_only_summary.csv \
  --summary-md-name psf_v1_nhit_only_summary.md

echo "===== Stage C v1-bin Nhit-only observation reduction ====="
CMD_C=(
  python apply/stages/03_reduce_obs.py
  --obs-root "${OBS_ROOT}"
  --time-root "${TIME_ROOT}"
  --cell-selection-csv "${SELECTOR}"
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

echo "===== Stage D v1-bin Nhit-only annulus quadratic surface background ====="
CMD_D=(
  python apply/stages/04_background.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --psf-npz "${STAGE_B_RUN_DIR}/psf_v1_nhit_only.npz"
  --cell-selection-csv "${SELECTOR}"
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
  --npz-name background_v1_nhit_only.npz
  --metadata-name background_v1_nhit_only_metadata.json
  --summary-csv-name background_v1_nhit_only_summary.csv
  --summary-md-name background_v1_nhit_only_summary.md
  --no-promote-current
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_D+=(--max-batches "${MAX_BATCHES}")
fi
printf '%q ' "${CMD_D[@]}"
echo
"${CMD_D[@]}"

echo "===== Stage E v1-bin Nhit-only signal ====="
CMD_E=(
  python apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --background-npz "${STAGE_D_RUN_DIR}/background_v1_nhit_only.npz"
  --background-metadata "${STAGE_D_RUN_DIR}/background_v1_nhit_only_metadata.json"
  --cell-selection-csv "${SELECTOR}"
  --output-dir "${STAGE_E_DIR}"
  --report-html apply/report/stage_e_v1_nhit_only_report.html
  --run-id "${RUN_E}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma 50
  --quality-max-total-sigma 250
  --print-every 10
  --npz-name signal_v1_nhit_only.npz
  --metadata-name signal_v1_nhit_only_metadata.json
  --summary-csv-name signal_v1_nhit_only_summary.csv
  --summary-md-name signal_v1_nhit_only_summary.md
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
if [[ ! -f "${STAGE_E_RUN_DIR}/signal_v1_nhit_only.npz" ]]; then
  exit "${stage_e_status}"
fi

echo "===== Stage F v1-bin Nhit-only fit ====="
CMD_F=(
  python apply/stages/06_fit.py
  --response-npz "${STAGE_A_DIR}/response_2d_v1_nhit_only.npz"
  --response-metadata "${STAGE_A_DIR}/response_2d_v1_nhit_only_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v1_nhit_only.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v1_nhit_only_metadata.json"
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --cell-subset-csv "${SELECTOR}"
  --output-dir "${STAGE_F_DIR}"
  --report-html apply/report/stage_f_v1_nhit_only_report.html
  --run-id "${RUN_F}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name fit_v1_nhit_only.npz
  --metadata-name fit_v1_nhit_only_metadata.json
  --summary-csv-name fit_v1_nhit_only_summary.csv
  --summary-md-name fit_v1_nhit_only_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_F[@]}"
echo
set +e
"${CMD_F[@]}"
stage_f_status=$?
set -e
if [[ ! -f "${STAGE_F_RUN_DIR}/fit_v1_nhit_only.npz" ]]; then
  exit "${stage_f_status}"
fi

echo "===== Stage G v1-bin Nhit-only SED points ====="
CMD_G=(
  python apply/stages/07_sed_points.py
  --response-npz "${STAGE_A_DIR}/response_2d_v1_nhit_only.npz"
  --response-metadata "${STAGE_A_DIR}/response_2d_v1_nhit_only_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v1_nhit_only.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v1_nhit_only_metadata.json"
  --stage-f-npz "${STAGE_F_RUN_DIR}/fit_v1_nhit_only.npz"
  --stage-f-metadata "${STAGE_F_RUN_DIR}/fit_v1_nhit_only_metadata.json"
  --output-dir "${STAGE_G_DIR}"
  --report-html apply/report/stage_g_v1_nhit_only_report.html
  --run-id "${RUN_G}"
  --baseline-name v1_nhit_only
  --required-cell-ids "${REQUIRED_CELL_IDS}"
  --excluded-cell-ids ""
  --skip-expected-stage-f-validation
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name sed_points_v1_nhit_only.npz
  --metadata-name sed_points_v1_nhit_only_metadata.json
  --summary-csv-name sed_points_v1_nhit_only_summary.csv
  --summary-json-name sed_points_v1_nhit_only_summary.json
  --summary-md-name sed_points_v1_nhit_only_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_G[@]}"
echo
"${CMD_G[@]}"

echo "===== v1-bin Nhit-only official/tutorial forward-fold diagnostics ====="
python apply/report/build_v3_normalization_diagnostics.py \
  --output-dir "${DIAG_DIR}" \
  --response-npz "${STAGE_A_DIR}/response_2d_v1_nhit_only.npz" \
  --stage-f-npz "v1_nhit_only=${STAGE_F_RUN_DIR}/fit_v1_nhit_only.npz" \
  --skip-offsource

echo "===== Rebuild integrated v3 report with v3 and v1-bin Nhit-only overlays ====="
python apply/report/build_v3_full_report.py

echo "===== v1-bin Nhit-only control finished ====="
echo "Stage G: ${STAGE_G_RUN_DIR}/sed_points_v1_nhit_only_summary.csv"
echo "Forward-fold: ${DIAG_DIR}/v3_official_forward_fold_summary.csv"
echo "Integrated report: apply/report/crab_sed_v3_stage_a_to_g_report.html"
