#!/bin/bash
#SBATCH --job-name=apply_obs_64748_january
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --chdir=/mnt/newdisk/shijie/energy
#SBATCH --output=/mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH --error=/mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=/mnt/newdisk/shijie/energy
ENV_PREFIX=/mnt/newdisk/shijie/conda/envs/py310
INPUT_ROOT=/mnt/newdisk/shijie/WCDA_obs
OUTPUT_ROOT=/mnt/newdisk/shijie/WCDA_obs_eval_64748
RUN_DIR="${REPO_ROOT}/runs/theta_recoxy_position_embed_noemax_no_core_cut_64748"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[[ -x "${ENV_PREFIX}/bin/python" ]] || { echo "py310 environment is not ready: ${ENV_PREFIX}" >&2; exit 64; }
[[ -d "${INPUT_ROOT}/0101" ]] || { echo "January input directory is missing" >&2; exit 64; }
[[ -r "${RUN_DIR}/config.json" && -r "${RUN_DIR}/checkpoints/best_model.pt" ]] || {
  echo "64748 config or checkpoint is missing" >&2; exit 64;
}
[[ ! -L "${OUTPUT_ROOT}" ]] || { echo "Output root must not be a symlink" >&2; exit 64; }

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.january_inference.lock"
flock -n 9 || { echo "Another January inference is using this output root" >&2; exit 64; }
# Existing outputs need a separate provenance review before resuming.
if [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -path '*/01??/*.root' -print -quit)" ]]; then
  echo "January outputs already exist; refusing to overwrite or silently skip them" >&2
  exit 64
fi

echo "JobID=${SLURM_JOB_ID:?Submit this script with sbatch} Node=${SLURM_NODELIST:-unknown} GPUs=${CUDA_VISIBLE_DEVICES:-unset}"
date -Is
echo "Python=${ENV_PREFIX}/bin/python"
"${ENV_PREFIX}/bin/python" -V
echo "Git commit=$(git rev-parse HEAD)"
echo "Input=${INPUT_ROOT} Output=${OUTPUT_ROOT} DayPrefix=01 Run=${RUN_DIR}"
echo "SHA256 (checkpoint, config, inference entrypoint):"
sha256sum "${RUN_DIR}/checkpoints/best_model.pt" "${RUN_DIR}/config.json" apply/apply_observation_energy.py
echo "64748 configuration:"
cat "${RUN_DIR}/config.json"
echo

"${ENV_PREFIX}/bin/python" -u apply/apply_observation_energy.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --day-prefix 01 \
  --summary-path "${OUTPUT_ROOT}/apply_summary_01_${SLURM_JOB_ID}.json" \
  --device cuda \
  --gpu-ids 0 \
  --batch-size 128 \
  --reader-workers 0 \
  --step-size "100 MB" \
  --cut-pinc-max 1.1 \
  --cut-fitstat-equals 0 \
  --cut-theta-max-deg 50 \
  --cut-dcedge-min 20 \
  --print-every 10

echo "January 64748 observation inference finished"
