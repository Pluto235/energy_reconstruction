#!/bin/bash
#SBATCH -J theta_truecore_smoke
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00

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

ROOT_PATH=/home/server/mydisk/WCDA_simulation
TAG="theta_truecore_points_smoketest"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.theta.main_theta \
  --root_path "${ROOT_PATH}" \
  --n_files 20 \
  --epochs 2 \
  --batch_size 16 \
  --max_points 128 \
  --num_workers 0 \
  --io_workers 2 \
  --norm_mode per_event \
  --sample_mode weighted_q \
  --Emin 100 \
  --Emax 10000 \
  --dcedge_min 0 \
  --theta_max_deg 30 \
  --dangle_max_deg 3 \
  --pinc_max 1.1 \
  --theta_embed_dim 8 \
  --theta_embed_dropout 0.0 \
  --eval_space log \
  --save_arrays \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "✅ True-core smoke test finished"
