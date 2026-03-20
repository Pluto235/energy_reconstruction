#!/bin/bash
#SBATCH -J eval_theta_mid_2709
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 72:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

# ===== 环境 =====
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

# ===== 进入项目目录 =====
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:$PYTHONPATH

echo "Begin evaluation based on job 2709"
# ===== 运行评估 =====
python -m src.theta.evaluate_only \
    --run_dir /home/server/projects/energy_reconstruction/runs/theta_embed_2709

echo "✅ Evaluation only finished successfully"
