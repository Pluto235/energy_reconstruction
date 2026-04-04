#!/bin/bash
#SBATCH -J apply_sim_all_bin
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

INPUT_ROOT=/mnt/mydisk/WCDA_simulation
OUTPUT_ROOT=/mnt/mydisk/WCDA_simulation_binned
RUN_DIR=/home/server/projects/energy_reconstruction/runs/no_core_cut_2724
TAG="apply_simulation_all_bin"

echo "===== Start apply-stage full-simulation inference ====="
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run dir: ${RUN_DIR}"

python apply/simulation_all_bin.py \
  --input-root "${INPUT_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --gpu-ids 0,1,2,3 \
  --batch-size 128 \
  --reader-workers 0 \
  --print-every 20

echo "✅ Apply-stage full-simulation inference finished"
