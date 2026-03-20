#!/bin/bash
#SBATCH -J core_selection_compare
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

# ============================================================
# Common theta arguments
# Keep aligned with the maintained theta training defaults/script.
# ============================================================
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
  --theta_max_deg 30
  --dangle_max_deg 3
  --pinc_max 1.1
  --theta_embed_dim ${THETA_EMBED_DIM}
  --theta_embed_dropout ${THETA_EMBED_DROPOUT}
  --eval_space log
  --save_arrays
"

# ============================================================
# Experiment 1: true core-box selection
# ============================================================
echo "===== Run 1: true core-box selection ====="

TAG="corebox_true"
RUN_DIR=${RUNS_DIR}/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.theta.main_theta \
  ${COMMON_ARGS} \
  --dcedge_min 0 \
  --use_core_box \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run 1 finished: ${RUN_DIR} ====="

# ============================================================
# Experiment 2: reconstructed core edge cut (20 m)
# ============================================================
echo "===== Run 2: reconstructed core edge cut 20 m ====="

TAG="dcedge20"
RUN_DIR=${RUNS_DIR}/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.theta.main_theta \
  ${COMMON_ARGS} \
  --dcedge_min 20 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run 2 finished: ${RUN_DIR} ====="

# ============================================================
# Experiment 3: no core selection
# ============================================================
echo "===== Run 3: no core selection ====="

TAG="no_core_cut"
RUN_DIR=${RUNS_DIR}/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.theta.main_theta \
  ${COMMON_ARGS} \
  --dcedge_min 0 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run 3 finished: ${RUN_DIR} ====="

echo "✅ All core-selection comparison experiments finished successfully"
