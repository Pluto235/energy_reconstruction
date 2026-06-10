#!/bin/bash
#SBATCH -J fit_stage_f
#SBATCH -p main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 01:00:00

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

RESPONSE_NPZ=${RESPONSE_NPZ:-apply/output/stage_a/response_2d.npz}
RESPONSE_METADATA=${RESPONSE_METADATA:-apply/output/stage_a/response_2d_metadata.json}
SIGNAL_NPZ=${SIGNAL_NPZ:-apply/output/stage_e/current/signal_v1.npz}
SIGNAL_METADATA=${SIGNAL_METADATA:-apply/output/stage_e/current/signal_v1_metadata.json}
STAGE_C_DIR=${STAGE_C_DIR:-apply/output/stage_c/current}
CELL_SUBSET_CSV=${CELL_SUBSET_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_f}
REPORT_HTML=${REPORT_HTML:-apply/report/stage_f_report.html}
RUN_ID=${RUN_ID:-slurm_${SLURM_JOB_ID}}
NO_PLOTS=${NO_PLOTS:-}

echo "===== Start Stage F forward-folding fit ====="
echo "Response NPZ: ${RESPONSE_NPZ}"
echo "Response metadata: ${RESPONSE_METADATA}"
echo "Signal NPZ: ${SIGNAL_NPZ}"
echo "Signal metadata: ${SIGNAL_METADATA}"
echo "Stage C dir: ${STAGE_C_DIR}"
echo "Cell subset CSV: ${CELL_SUBSET_CSV:-<none>}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Report HTML: ${REPORT_HTML}"
echo "Run id: ${RUN_ID}"

CMD=(
  python apply/stages/06_fit.py
  --response-npz "${RESPONSE_NPZ}"
  --response-metadata "${RESPONSE_METADATA}"
  --signal-npz "${SIGNAL_NPZ}"
  --signal-metadata "${SIGNAL_METADATA}"
  --stage-c-dir "${STAGE_C_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --report-html "${REPORT_HTML}"
  --run-id "${RUN_ID}"
  --source-ra-deg 83.63
  --source-dec-deg 22.01
  --lhaaso-lat-deg 29.45
  --lhaaso-lon-deg 100.14
  --exposure-sample-step-sec 60
  --pivot-tev 3
  --reference-phi0 2.114e-12
  --reference-gamma 2.69
)

if [[ -n "${NO_PLOTS}" ]]; then
  CMD+=(--no-plots)
fi

if [[ -n "${CELL_SUBSET_CSV}" ]]; then
  CMD+=(--cell-subset-csv "${CELL_SUBSET_CSV}")
fi

"${CMD[@]}"

echo "Stage F forward-folding fit finished"
