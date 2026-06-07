#!/bin/bash
#SBATCH -J apply_observation_energy
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 72:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

echo "==== Job Info ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "=================="

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

python -V
which python
nvidia-smi || true

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:$PYTHONPATH

INPUT_ROOT=${INPUT_ROOT:-/mnt/mydisk/WCDA_observation}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/mydisk/WCDA_observation_eval}
RUN_DIR=${RUN_DIR:-/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666}
DAY_PREFIX=${DAY_PREFIX:-}
SUMMARY_PATH=${SUMMARY_PATH:-${OUTPUT_ROOT}/apply_summary.json}

echo "===== Start observation inference ====="
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run dir: ${RUN_DIR}"
echo "Day prefix: ${DAY_PREFIX:-<all>}"
echo "Summary path: ${SUMMARY_PATH}"
echo "Expected layout: ${INPUT_ROOT}/MMDD/*.root -> ${OUTPUT_ROOT}/MMDD/*.root"
echo "Note: launch this only after observation data migration is complete."

CMD=(
  python apply/apply_observation_energy.py
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --run-dir "${RUN_DIR}"
  --summary-path "${SUMMARY_PATH}"
  --device cuda
  --gpu-ids 0,1,2,3
  --batch-size 256
  --reader-workers 0
  --cut-pinc-max 1.1
  --cut-fitstat-equals 0
  --cut-theta-max-deg 50
  --cut-dcedge-min 20
  --print-every 10
)

if [[ -n "${DAY_PREFIX}" ]]; then
  CMD+=(--day-prefix "${DAY_PREFIX}")
fi

"${CMD[@]}"

echo "Observation inference finished"
