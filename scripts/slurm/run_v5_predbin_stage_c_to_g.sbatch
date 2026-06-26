#!/bin/bash
#SBATCH -J v5_predbin_cg
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

STRATEGY=${STRATEGY:?Set STRATEGY=gap025 or STRATEGY=gap1}
CELL_LEDGER=${CELL_LEDGER:-apply/config/cell_ledger_v5_predbin_${STRATEGY}.csv}
FIT_SELECTOR=${FIT_SELECTOR:-apply/config/cell_selector_v5_predbin_${STRATEGY}.csv}
STAGE_A_DIR=${STAGE_A_DIR:-apply/output/stage_a_v5_predbin_${STRATEGY}}
STAGE_B_DIR=${STAGE_B_DIR:-apply/output/stage_b_v5_predbin_${STRATEGY}}
STAGE_A_AP_DIR=${STAGE_A_AP_DIR:-apply/output/stage_a_v5_predbin_${STRATEGY}_aperture_conditioned}
STAGE_C_DIR=${STAGE_C_DIR:-apply/output/stage_c_v5_predbin_${STRATEGY}}
STAGE_D_DIR=${STAGE_D_DIR:-apply/output/stage_d_v5_predbin_${STRATEGY}}
STAGE_E_DIR=${STAGE_E_DIR:-apply/output/stage_e_v5_predbin_${STRATEGY}_containment1_annnorm}
STAGE_F_DIR=${STAGE_F_DIR:-apply/output/stage_f_v5_predbin_${STRATEGY}_aperture_conditioned}
STAGE_G_DIR=${STAGE_G_DIR:-apply/output/stage_g_v5_predbin_${STRATEGY}_aperture_conditioned}
RUN_SUFFIX=${RUN_SUFFIX:-slurm_${SLURM_JOB_ID}}
RUN_C=${RUN_C:-v5_predbin_${STRATEGY}_stage_c_${RUN_SUFFIX}}
RUN_D=${RUN_D:-v5_predbin_${STRATEGY}_stage_d_${RUN_SUFFIX}}
RUN_E=${RUN_E:-v5_predbin_${STRATEGY}_stage_e_containment1_${RUN_SUFFIX}}
RUN_F=${RUN_F:-v5_predbin_${STRATEGY}_stage_f_${RUN_SUFFIX}}
RUN_G=${RUN_G:-v5_predbin_${STRATEGY}_stage_g_${RUN_SUFFIX}}
STAGE_C_RUN_DIR="${STAGE_C_DIR}/runs/${RUN_C}"
STAGE_D_RUN_DIR="${STAGE_D_DIR}/runs/${RUN_D}"
STAGE_E_RUN_DIR="${STAGE_E_DIR}/runs/${RUN_E}"
STAGE_F_RUN_DIR="${STAGE_F_DIR}/runs/${RUN_F}"
STAGE_G_RUN_DIR="${STAGE_G_DIR}/runs/${RUN_G}"

OBS_ROOT=${OBS_ROOT:-/mnt/mydisk/WCDA_observation_eval}
TIME_ROOT=${TIME_ROOT:-/mnt/mydisk/WCDA_observation_eval/recovered_time}
MAX_FILES=${MAX_FILES:-}
MAX_BATCHES=${MAX_BATCHES:-}

echo "===== Stage C v5 ${STRATEGY} observation reduction ====="
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

echo "===== Stage D v5 ${STRATEGY} annulus-normalized quadratic background ====="
CMD_D=(
  python apply/stages/04_background.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --psf-npz "${STAGE_B_DIR}/current/psf_v5_predbin_${STRATEGY}.npz"
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
  --npz-name background_v5_predbin_${STRATEGY}.npz
  --metadata-name background_v5_predbin_${STRATEGY}_metadata.json
  --summary-csv-name background_v5_predbin_${STRATEGY}_summary.csv
  --summary-md-name background_v5_predbin_${STRATEGY}_summary.md
  --no-promote-current
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_D+=(--max-batches "${MAX_BATCHES}")
fi
printf '%q ' "${CMD_D[@]}"
echo
"${CMD_D[@]}"

echo "===== Stage E v5 ${STRATEGY} containment1 signal ====="
CMD_E=(
  python apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --background-npz "${STAGE_D_RUN_DIR}/background_v5_predbin_${STRATEGY}.npz"
  --background-metadata "${STAGE_D_RUN_DIR}/background_v5_predbin_${STRATEGY}_metadata.json"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_E_DIR}"
  --report-html apply/report/stage_e_v5_predbin_${STRATEGY}_containment1_report.html
  --run-id "${RUN_E}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma 50
  --quality-max-total-sigma 300
  --containment-override fixed1
  --print-every 10
  --npz-name signal_v5_predbin_${STRATEGY}_containment1.npz
  --metadata-name signal_v5_predbin_${STRATEGY}_containment1_metadata.json
  --summary-csv-name signal_v5_predbin_${STRATEGY}_containment1_summary.csv
  --summary-md-name signal_v5_predbin_${STRATEGY}_containment1_summary.md
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
if [[ ! -f "${STAGE_E_RUN_DIR}/signal_v5_predbin_${STRATEGY}_containment1.npz" ]]; then
  exit "${stage_e_status}"
fi

echo "===== Stage F v5 ${STRATEGY} aperture-conditioned fit ====="
CMD_F=(
  python apply/stages/06_fit.py
  --response-npz "${STAGE_A_AP_DIR}/response_2d_v5_predbin_${STRATEGY}_aperture_conditioned.npz"
  --response-metadata "${STAGE_A_AP_DIR}/response_2d_v5_predbin_${STRATEGY}_aperture_conditioned_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v5_predbin_${STRATEGY}_containment1.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v5_predbin_${STRATEGY}_containment1_metadata.json"
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --cell-subset-csv "${FIT_SELECTOR}"
  --output-dir "${STAGE_F_DIR}"
  --report-html apply/report/stage_f_v5_predbin_${STRATEGY}_aperture_conditioned_report.html
  --run-id "${RUN_F}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name fit_v5_predbin_${STRATEGY}_aperture_conditioned.npz
  --metadata-name fit_v5_predbin_${STRATEGY}_aperture_conditioned_metadata.json
  --summary-csv-name fit_v5_predbin_${STRATEGY}_aperture_conditioned_summary.csv
  --summary-md-name fit_v5_predbin_${STRATEGY}_aperture_conditioned_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_F[@]}"
echo
set +e
"${CMD_F[@]}"
stage_f_status=$?
set -e
if [[ ! -f "${STAGE_F_RUN_DIR}/fit_v5_predbin_${STRATEGY}_aperture_conditioned.npz" ]]; then
  exit "${stage_f_status}"
fi

REQUIRED_CELL_IDS=$(
  python - <<PY
import csv
rows=list(csv.DictReader(open("${FIT_SELECTOR}")))
print(",".join(row["cell_id"] for row in rows if str(row["include"]).strip().lower() in {"1","true","yes","y","include"}))
PY
)
EXCLUDED_CELL_IDS=$(
  python - <<PY
import csv
rows=list(csv.DictReader(open("${FIT_SELECTOR}")))
print(",".join(row["cell_id"] for row in rows if str(row["include"]).strip().lower() not in {"1","true","yes","y","include"}))
PY
)

echo "===== Stage G v5 ${STRATEGY} SED points ====="
CMD_G=(
  python apply/stages/07_sed_points.py
  --response-npz "${STAGE_A_AP_DIR}/response_2d_v5_predbin_${STRATEGY}_aperture_conditioned.npz"
  --response-metadata "${STAGE_A_AP_DIR}/response_2d_v5_predbin_${STRATEGY}_aperture_conditioned_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v5_predbin_${STRATEGY}_containment1.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v5_predbin_${STRATEGY}_containment1_metadata.json"
  --stage-f-npz "${STAGE_F_RUN_DIR}/fit_v5_predbin_${STRATEGY}_aperture_conditioned.npz"
  --stage-f-metadata "${STAGE_F_RUN_DIR}/fit_v5_predbin_${STRATEGY}_aperture_conditioned_metadata.json"
  --output-dir "${STAGE_G_DIR}"
  --report-html apply/report/stage_g_v5_predbin_${STRATEGY}_aperture_conditioned_report.html
  --run-id "${RUN_G}"
  --baseline-name v5_predbin_${STRATEGY}_aperture_conditioned
  --required-cell-ids "${REQUIRED_CELL_IDS}"
  --excluded-cell-ids "${EXCLUDED_CELL_IDS}"
  --skip-expected-stage-f-validation
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name sed_points_v5_predbin_${STRATEGY}_aperture_conditioned.npz
  --metadata-name sed_points_v5_predbin_${STRATEGY}_aperture_conditioned_metadata.json
  --summary-csv-name sed_points_v5_predbin_${STRATEGY}_aperture_conditioned_summary.csv
  --summary-json-name sed_points_v5_predbin_${STRATEGY}_aperture_conditioned_summary.json
  --summary-md-name sed_points_v5_predbin_${STRATEGY}_aperture_conditioned_summary.md
  --no-promote-current
)
printf '%q ' "${CMD_G[@]}"
echo
"${CMD_G[@]}"

python apply/report/build_v5_predbin_ablation_report.py

echo "v5 ${STRATEGY} Stage C-G chain finished"
