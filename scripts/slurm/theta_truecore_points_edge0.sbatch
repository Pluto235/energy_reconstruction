#!/bin/bash
#SBATCH -J theta_truecore_edge0
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 72:00:00

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
N_FILES=10000
EPOCHS=500
BATCH_SIZE=512
MAX_POINTS=500

THETA_EMBED_DIM=16
THETA_EMBED_DROPOUT=0.0

TAG="theta_truecore_points_edge0"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

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
  --dcedge_min 0
  --theta_max_deg 30
  --dangle_max_deg 3
  --pinc_max 1.1
  --theta_embed_dim ${THETA_EMBED_DIM}
  --theta_embed_dropout ${THETA_EMBED_DROPOUT}
  --eval_space log
  --save_arrays
  --run_dir ${RUN_DIR}
  --tag ${TAG}
"

echo "===== Start WCDA energy reconstruction with true-core-centered points (edge0) ====="
echo "Run dir: ${RUN_DIR}"

python -m src.theta.main_theta ${COMMON_ARGS}

echo "✅ Training & evaluation finished successfully"
