#!/bin/bash
#SBATCH -J build_signal_e
#SBATCH -p main
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 04:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

echo "==== Job Info ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "=================="

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

python -V
which python

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

STAGE_C_DIR=${STAGE_C_DIR:-apply/output/stage_c/current}
BACKGROUND_NPZ=${BACKGROUND_NPZ:-apply/output/stage_d/current/background_v1.npz}
BACKGROUND_METADATA=${BACKGROUND_METADATA:-apply/output/stage_d/current/background_v1_metadata.json}
CELL_SELECTION=${CELL_SELECTION:-apply/config/cell_selection_v1.csv}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_e}
REPORT_HTML=${REPORT_HTML:-apply/report/stage_e_report.html}
RUN_ID=${RUN_ID:-slurm_${SLURM_JOB_ID}}
MAX_BATCHES=${MAX_BATCHES:-}
NO_PLOTS=${NO_PLOTS:-}

echo "===== Start Stage E signal extraction ====="
echo "Stage C dir: ${STAGE_C_DIR}"
echo "Background NPZ: ${BACKGROUND_NPZ}"
echo "Background metadata: ${BACKGROUND_METADATA}"
echo "Cell selection: ${CELL_SELECTION}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Report HTML: ${REPORT_HTML}"
echo "Run id: ${RUN_ID}"
echo "Max batches: ${MAX_BATCHES:-<all>}"

python - <<PY
import json
from pathlib import Path

path = Path("${BACKGROUND_METADATA}")
if not path.exists():
    print(f"Stage D metadata preflight: missing {path}")
else:
    metadata = json.loads(path.read_text())
    model = metadata.get("background_model") or {}
    mode = model.get("background_mode") or model.get("background_scope") or metadata.get("background_mode") or metadata.get("background_scope")
    form = model.get("background_form") or metadata.get("background_form")
    method = model.get("method") or metadata.get("method")
    print(f"Stage D metadata preflight: mode={mode!r} form={form!r} method={method!r}")
PY

CMD=(
  python apply/stages/05_signal.py
  --stage-c-dir "${STAGE_C_DIR}"
  --background-npz "${BACKGROUND_NPZ}"
  --background-metadata "${BACKGROUND_METADATA}"
  --cell-selection-csv "${CELL_SELECTION}"
  --output-dir "${OUTPUT_DIR}"
  --report-html "${REPORT_HTML}"
  --run-id "${RUN_ID}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --batch-size 500000
  --quality-min-total-sigma 50
  --quality-max-total-sigma 200
  --print-every 10
)

if [[ -n "${MAX_BATCHES}" ]]; then
  CMD+=(--max-batches "${MAX_BATCHES}")
fi

if [[ -n "${NO_PLOTS}" ]]; then
  CMD+=(--no-plots)
fi

"${CMD[@]}"

echo "Stage E signal extraction finished"
