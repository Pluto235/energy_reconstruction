#!/bin/bash
#SBATCH -J allcuts_compare
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 72:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

PROJECT_ROOT=/home/server/projects/energy_reconstruction
RUNS_DIR=${PROJECT_ROOT}/runs
ROOT_PATH=/home/server/mydisk/WCDA_simulation

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

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

N_FILES=10000
EPOCHS=500
BATCH_SIZE=512
MAX_POINTS=500
THETA_EMBED_DIM=16
THETA_EMBED_DROPOUT=0.0

COMMON_ARGS="
  --root_path ${ROOT_PATH}
  --n_files ${N_FILES}
  --epochs ${EPOCHS}
  --batch_size ${BATCH_SIZE}
  --max_points ${MAX_POINTS}
  --num_workers 4
  --io_workers 32
  --norm_mode per_event
  --sample_mode weighted_q
  --Emin 100
  --theta_embed_dim ${THETA_EMBED_DIM}
  --theta_embed_dropout ${THETA_EMBED_DROPOUT}
  --eval_space log
  --save_arrays
"

echo "===== Run 1: strict cuts ====="
TAG="allcuts_strict"
RUN_DIR=${RUNS_DIR}/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.theta.main_theta \
  ${COMMON_ARGS} \
  --require_fitstat0 true \
  --dcedge_min 20 \
  --dangle_max_deg 3 \
  --theta_max_deg 30 \
  --pinc_max 1.1 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run 1 finished: ${RUN_DIR} ====="

echo "===== Run 2: relaxed cuts ====="
TAG="allcuts_relaxed"
RUN_DIR=${RUNS_DIR}/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.theta.main_theta \
  ${COMMON_ARGS} \
  --require_fitstat0 false \
  --dcedge_min 0 \
  --dangle_max_deg 999 \
  --theta_max_deg 50 \
  --pinc_max 9999 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run 2 finished: ${RUN_DIR} ====="
echo "✅ All allcuts comparison experiments finished successfully"
