#!/bin/bash
#SBATCH -J wcda_dc_core_seq
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
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
# Experiment A
# ============================================================
echo "===== Run A: dcedge_min=20, core_box=OFF ====="

TAG="dc20_core0"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  --root_path /home/server/mydisk/WCDA_simulation/ \
  --n_files 10000 \
  --epochs 500 \
  --batch_size 256 \
  --max_points 500 \
  --num_workers 4 \
  --io_workers 16 \
  --norm_mode per_event \
  --sample_mode random \
  --Emin 100 \
  --dcedge_min 20 \
  --theta_max_deg 30 \
  --dangle_max_deg 3 \
  --pinc_max 1.1 \
  --eval_space log \
  --save_arrays \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run A finished ====="


# ============================================================
# Experiment B
# ============================================================
echo "===== Run B: dcedge_min=0, core_box=ON ====="

TAG="dc0_core1"
RUN_DIR=/home/server/projects/energy_reconstruction/runs/${TAG}_${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

python -m src.main \
  --root_path /home/server/mydisk/WCDA_simulation/ \
  --n_files 10000 \
  --epochs 500 \
  --batch_size 256 \
  --max_points 500 \
  --num_workers 4 \
  --io_workers 16 \
  --norm_mode per_event \
  --sample_mode random \
  --Emin 100 \
  --dcedge_min 0 \
  --theta_max_deg 30 \
  --dangle_max_deg 3 \
  --pinc_max 1.1 \
  --use_core_box \
  --eval_space log \
  --save_arrays \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}"

echo "===== Run B finished ====="

echo "✅ All experiments finished successfully"
