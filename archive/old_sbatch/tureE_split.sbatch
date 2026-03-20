#!/bin/bash
#SBATCH -J trueE_split
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

# ---------------- conda ----------------
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:$PYTHONPATH

# ============================================================
# Common arguments（只在这里统一改）
# ============================================================
COMMON_ARGS="
  --root_path /home/server/mydisk/WCDA_simulation
  --n_files 10000
  --epochs 500
  --batch_size 512
  --max_points 500
  --num_workers 4
  --io_workers 8
  --norm_mode per_event
  --sample_mode random
  --eval_space log
  --save_arrays
  --theta_max_deg 30
  --dangle_max_deg 3
  --pinc_max 1.1
"

# ============================================================
# Experiment 1: 100 ≤ E_true < 10000 GeV
# ============================================================
echo "===== Run 1: 100 ≤ E_true < 1e4 GeV ====="

TAG="trueE_1e2_1e4"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  ${COMMON_ARGS} \
  --Emin 100 \
  --Emax 10000 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run 1 finished ====="

# ============================================================
# Experiment 2: E_true ≥ 10000 GeV
# ============================================================
echo "===== Run 2: E_true ≥ 1e4 GeV ====="

TAG="trueE_ge_1e4"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  ${COMMON_ARGS} \
  --Emin 10000 \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run 2 finished ====="

echo "✅ All true-energy-split experiments finished"
