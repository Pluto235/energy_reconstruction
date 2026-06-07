#!/bin/bash
#SBATCH -J reduce_obs_c
#SBATCH -p main
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 24:00:00

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

OBS_ROOT=${OBS_ROOT:-/mnt/mydisk/WCDA_observation_eval}
TIME_ROOT=${TIME_ROOT:-/mnt/mydisk/WCDA_observation_eval/recovered_time}
CELL_SELECTION=${CELL_SELECTION:-apply/config/cell_selection_v1.csv}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_c}
DAY_PREFIX=${DAY_PREFIX:-}
MAX_FILES=${MAX_FILES:-}
RUN_ID=${RUN_ID:-slurm_${SLURM_JOB_ID}}

echo "===== Start Stage C observation reduction ====="
echo "Obs root: ${OBS_ROOT}"
echo "Time root: ${TIME_ROOT}"
echo "Cell selection: ${CELL_SELECTION}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Run id: ${RUN_ID}"
echo "Day prefix: ${DAY_PREFIX:-<all>}"
echo "Max files: ${MAX_FILES:-<all>}"

CMD=(
  python apply/stages/03_reduce_obs.py
  --obs-root "${OBS_ROOT}"
  --time-root "${TIME_ROOT}"
  --cell-selection-csv "${CELL_SELECTION}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --tree-name t_eventout
  --time-tree-name t_recovered_time
  --entries-per-chunk 200000
  --workers "${SLURM_CPUS_PER_TASK:-12}"
  --match-status-equals 0
  --cut-pinc-max 1.1
  --cut-fitstat-equals 0
  --cut-theta-max-deg 50
  --cut-dcedge-min 20
  --gap-threshold-sec 60
  --print-every 25
)

if [[ -n "${DAY_PREFIX}" ]]; then
  CMD+=(--day-prefix "${DAY_PREFIX}")
fi

if [[ -n "${MAX_FILES}" ]]; then
  CMD+=(--max-files "${MAX_FILES}")
fi

"${CMD[@]}"

echo "Stage C observation reduction finished"
