#!/bin/bash
#SBATCH -J v6_64748_cg
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

RUN_TAG=v6_64748_split56_baselinev4
CELL_LEDGER=apply/config/cell_ledger_v6_64748_split56_candidate.csv
FIT_SELECTOR=apply/config/cell_selector_v6_64748_split56_drop4_psfborrow.csv
STAGE_A_AP_DIR=apply/output/stage_a_v6_64748_split56_aperture_conditioned
STAGE_B_DIR=apply/output/stage_b_v6_64748_split56
STAGE_C_DIR=apply/output/stage_c_v6_64748_split56
STAGE_D_DIR=apply/output/stage_d_v6_64748_split56_annnorm
STAGE_E_DIR=apply/output/stage_e_v6_64748_split56_containment1_annnorm
STAGE_F_DIR=apply/output/stage_f_v6_64748_split56_baselinev4
STAGE_G_DIR=apply/output/stage_g_v6_64748_split56_baselinev4
RUN_C=v6_64748_split56_stage_c_halfyear
RUN_D=v6_64748_split56_stage_d_annnorm
RUN_E=v6_64748_split56_stage_e_containment1_annnorm
RUN_F=v6_64748_split56_stage_f_baselinev4
RUN_G=v6_64748_split56_stage_g_baselinev4
STAGE_C_RUN_DIR="${STAGE_C_DIR}/runs/${RUN_C}"
STAGE_D_RUN_DIR="${STAGE_D_DIR}/runs/${RUN_D}"
STAGE_E_RUN_DIR="${STAGE_E_DIR}/runs/${RUN_E}"
STAGE_F_RUN_DIR="${STAGE_F_DIR}/runs/${RUN_F}"
MAX_FILES=${MAX_FILES:-}
MAX_BATCHES=${MAX_BATCHES:-}
OBS_ROOT=${OBS_ROOT:-/mnt/mydisk/WCDA_observation_eval_64748}
TIME_ROOT=${TIME_ROOT:-${OBS_ROOT}/recovered_time}

echo "===== Stage C v6 64748 split56 half-year observation reduction ====="
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
  --overwrite-run-dir
)
if [[ -n "${MAX_FILES}" ]]; then
  CMD_C+=(--max-files "${MAX_FILES}")
fi
printf '%q ' "${CMD_C[@]}"
echo
"${CMD_C[@]}"

echo "===== Stage D v6 64748 split56 annulus-normalized quadratic background ====="
CMD_D=(
  python apply/stages/04_background.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --psf-npz "${STAGE_B_DIR}/runs/v6_64748_split56_stage_b_psf/psf_v6_64748_split56.npz"
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
  --npz-name background_v6_64748_split56_annnorm.npz
  --metadata-name background_v6_64748_split56_annnorm_metadata.json
  --summary-csv-name background_v6_64748_split56_annnorm_summary.csv
  --summary-md-name background_v6_64748_split56_annnorm_summary.md
  --overwrite-run-dir
)
if [[ -n "${MAX_BATCHES}" ]]; then
  CMD_D+=(--max-batches "${MAX_BATCHES}")
fi
printf '%q ' "${CMD_D[@]}"
echo
"${CMD_D[@]}"

echo "===== Stage E v6 64748 split56 containment-1 signal ====="
CMD_E=(
  python apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --background-npz "${STAGE_D_RUN_DIR}/background_v6_64748_split56_annnorm.npz"
  --background-metadata "${STAGE_D_RUN_DIR}/background_v6_64748_split56_annnorm_metadata.json"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${STAGE_E_DIR}"
  --report-html apply/report/stage_e_v6_64748_split56_containment1_annnorm_report.html
  --run-id "${RUN_E}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma 50
  --quality-max-total-sigma 300
  --containment-override fixed1
  --print-every 10
  --npz-name signal_v6_64748_split56_containment1_annnorm.npz
  --metadata-name signal_v6_64748_split56_containment1_annnorm_metadata.json
  --summary-csv-name signal_v6_64748_split56_containment1_annnorm_summary.csv
  --summary-md-name signal_v6_64748_split56_containment1_annnorm_summary.md
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
if [[ ! -f "${STAGE_E_RUN_DIR}/signal_v6_64748_split56_containment1_annnorm.npz" ]]; then
  exit "${stage_e_status}"
fi

echo "===== Stage F v6 64748 split56 baselinev4 aperture-conditioned fit ====="
CMD_F=(
  python apply/stages/06_fit.py
  --response-npz "${STAGE_A_AP_DIR}/response_2d_v6_64748_split56_aperture_conditioned.npz"
  --response-metadata "${STAGE_A_AP_DIR}/response_2d_v6_64748_split56_aperture_conditioned_metadata.json"
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v6_64748_split56_containment1_annnorm.npz"
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v6_64748_split56_containment1_annnorm_metadata.json"
  --stage-c-dir "${STAGE_C_RUN_DIR}"
  --cell-subset-csv "${FIT_SELECTOR}"
  --output-dir "${STAGE_F_DIR}"
  --report-html apply/report/stage_f_v6_64748_split56_baselinev4_report.html
  --run-id "${RUN_F}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
  --npz-name fit_v6_64748_split56_baselinev4.npz
  --metadata-name fit_v6_64748_split56_baselinev4_metadata.json
  --summary-csv-name fit_v6_64748_split56_baselinev4_summary.csv
  --summary-md-name fit_v6_64748_split56_baselinev4_summary.md
  --overwrite-run-dir
)
printf '%q ' "${CMD_F[@]}"
echo
set +e
"${CMD_F[@]}"
stage_f_status=$?
set -e
if [[ ! -f "${STAGE_F_RUN_DIR}/fit_v6_64748_split56_baselinev4.npz" ]]; then
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

echo "===== Stage G v6 64748 split56 baselinev4 SED points ====="
python apply/stages/07_sed_points.py \
  --response-npz "${STAGE_A_AP_DIR}/response_2d_v6_64748_split56_aperture_conditioned.npz" \
  --response-metadata "${STAGE_A_AP_DIR}/response_2d_v6_64748_split56_aperture_conditioned_metadata.json" \
  --signal-npz "${STAGE_E_RUN_DIR}/signal_v6_64748_split56_containment1_annnorm.npz" \
  --signal-metadata "${STAGE_E_RUN_DIR}/signal_v6_64748_split56_containment1_annnorm_metadata.json" \
  --stage-f-npz "${STAGE_F_RUN_DIR}/fit_v6_64748_split56_baselinev4.npz" \
  --stage-f-metadata "${STAGE_F_RUN_DIR}/fit_v6_64748_split56_baselinev4_metadata.json" \
  --output-dir "${STAGE_G_DIR}" \
  --report-html apply/report/stage_g_v6_64748_split56_baselinev4_report.html \
  --run-id "${RUN_G}" \
  --baseline-name "${RUN_TAG}" \
  --required-cell-ids "${REQUIRED_CELL_IDS}" \
  --excluded-cell-ids "${EXCLUDED_CELL_IDS}" \
  --skip-expected-stage-f-validation \
  --pivot-tev 3 \
  --reference-phi0 2.114e-12 \
  --reference-gamma 2.69 \
  --npz-name sed_points_v6_64748_split56_baselinev4.npz \
  --metadata-name sed_points_v6_64748_split56_baselinev4_metadata.json \
  --summary-csv-name sed_points_v6_64748_split56_baselinev4_summary.csv \
  --summary-json-name sed_points_v6_64748_split56_baselinev4_summary.json \
  --summary-md-name sed_points_v6_64748_split56_baselinev4_summary.md \
  --overwrite-run-dir

echo "===== Build final split56 report ====="
python apply/report/build_v6_64748_split56_report.py

echo "v6 64748 split56 Stage C-G chain finished"
