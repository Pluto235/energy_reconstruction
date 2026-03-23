#!/bin/bash
#SBATCH -J theta_nv_smoke
#SBATCH -p main
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 02:00:00

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
export CUDA_VISIBLE_DEVICES=""

ROOT_PATH=/home/server/mydisk/WCDA_simulation
TAG="theta_nv_embed_smoketest"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

COMMON_ARGS="
  --root_path ${ROOT_PATH}
  --n_files 6
  --epochs 1
  --batch_size 8
  --max_points 64
  --num_workers 0
  --io_workers 1
  --norm_mode per_event
  --sample_mode weighted_q
  --Emin 100
  --dcedge_min 20
  --theta_max_deg 30
  --dangle_max_deg 3
  --pinc_max 1.1
  --theta_embed_dim 8
  --theta_embed_dropout 0.0
  --nv_embed_dim 8
  --nv_embed_dropout 0.0
  --nv_scale 3000
  --eval_space log
  --save_arrays
  --run_dir ${RUN_DIR}
  --tag ${TAG}
"

echo "===== Start WCDA energy reconstruction smoke test with theta + nv embedding ====="
echo "Run dir: ${RUN_DIR}"

python -m src.theta.main_theta ${COMMON_ARGS}

echo "✅ NV embedding smoke test finished"
