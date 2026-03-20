#!/bin/bash
#SBATCH -J wcda_loss_mode
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 72:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/slurm_logs/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/slurm_logs/%x_%j.err

set -eo pipefail

echo "==== Job Info ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "=================="

# ---------- conda ----------
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

python -V
which python
nvidia-smi || true

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:$PYTHONPATH

# ============================================================
# Common arguments（只在这里改一次）
# ============================================================
ROOT_PATH=/home/server/mydisk/WCDA_simulation
N_FILES=10000
EPOCHS=500
BATCH_SIZE=512
MAX_POINTS=500

COMMON_ARGS="
  --root_path ${ROOT_PATH}
  --n_files ${N_FILES}
  --epochs ${EPOCHS}
  --batch_size ${BATCH_SIZE}
  --max_points ${MAX_POINTS}
  --num_workers 4
  --io_workers 32
  --norm_mode per_event
  --sample_mode random
  --Emin 100
  --dcedge_min 20
  --theta_max_deg 30
  --dangle_max_deg 3
  --pinc_max 1.1
  --eval_space log
  --save_arrays
"

# ============================================================
# Experiment 1: loss_mode = mse
# ============================================================
echo "===== Run 1: loss_mode = mse ====="

LOSS_MODE="mse"
TAG="loss_mse"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  ${COMMON_ARGS} \
  --loss_mode ${LOSS_MODE} \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== mse finished ====="


# ============================================================
# Experiment 2: loss_mode = huber
# ============================================================
echo "===== Run 2: loss_mode = huber ====="

LOSS_MODE="huber"
TAG="loss_huber"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  ${COMMON_ARGS} \
  --loss_mode ${LOSS_MODE} \
  --huber_delta 0.2 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== huber finished ====="


# ============================================================
# Experiment 3: loss_mode = rel
# ============================================================
echo "===== Run 3: loss_mode = rel ====="

LOSS_MODE="rel"
TAG="loss_rel"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  ${COMMON_ARGS} \
  --loss_mode ${LOSS_MODE} \
  --rel_delta 0.3 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== rel finished ====="


echo "✅ All loss_mode experiments finished successfully"
echo "✅ 所有的训练是在平权上进行的，评估是在特定能谱上进行的"
