#!/bin/bash
#SBATCH -J mcw_hist_loss
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
# Common arguments（保持和 MSE 实验一致，保证可对比）
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
  --Emin 100
  --dcedge_min 20
  --theta_max_deg 30
  --dangle_max_deg 3
  --pinc_max 1.1
  --eval_space log
  --save_arrays
"

echo "===== Run : mc_weighted histogram loss ====="

TAG="mcw_hist_loss"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  ${COMMON_ARGS} \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "✅ ===== mc_weighted histogram loss finished ====="
echo "mc_weighted histogram 在评估阶段也考虑看weight，但是在画图阶段并没有修改，utils的图的代码依然是旧的"
