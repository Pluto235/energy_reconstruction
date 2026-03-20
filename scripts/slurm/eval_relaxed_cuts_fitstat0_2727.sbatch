#!/bin/bash
#SBATCH -J eval_relaxed_fitstat0_2727
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 24:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

PROJECT_ROOT=/home/server/projects/energy_reconstruction
RUN_DIR=${PROJECT_ROOT}/runs/fitstat0_2727

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

echo "===== dcedge baseline ====="
python -m src.theta.eval_compare_relaxed \
  --run_dir "${RUN_DIR}" \
  --out_dir_name fig_eval_dcedge20_baseline \
  --eval_dcedge_min 20

echo "===== dcedge relaxed ====="
python -m src.theta.eval_compare_relaxed \
  --run_dir "${RUN_DIR}" \
  --out_dir_name fig_eval_dcedge0_relaxed \
  --eval_dcedge_min 0

echo "===== dangle baseline ====="
python -m src.theta.eval_compare_relaxed \
  --run_dir "${RUN_DIR}" \
  --out_dir_name fig_eval_dangle3_baseline \
  --eval_dangle_max_deg 3

echo "===== dangle relaxed ====="
python -m src.theta.eval_compare_relaxed \
  --run_dir "${RUN_DIR}" \
  --out_dir_name fig_eval_dangle_relaxed \
  --eval_dangle_max_deg 999

echo "===== pincness baseline ====="
python -m src.theta.eval_compare_relaxed \
  --run_dir "${RUN_DIR}" \
  --out_dir_name fig_eval_pinc1p1_baseline \
  --eval_pinc_max 1.1

echo "===== pincness relaxed ====="
python -m src.theta.eval_compare_relaxed \
  --run_dir "${RUN_DIR}" \
  --out_dir_name fig_eval_pinc_relaxed \
  --eval_pinc_max 9999

echo "✅ All relaxed-cut evaluations finished successfully"
