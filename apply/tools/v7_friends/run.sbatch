#!/bin/bash
#SBATCH --partition=main
#SBATCH --qos=default-safe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=v7friends
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export V7_GIT_COMMIT=$(git rev-parse HEAD)
exec /mnt/newdisk/shijie/conda/envs/py310/bin/python tools/v7_friends/friends.py \
  --inventory output/v7_friends_20261015/input_inventory.json \
  --run "$1" --stage /mnt/newdisk/shijie/WCDA_simulation/v7_friends/.staging/summary \
  --workers "$SLURM_CPUS_PER_TASK"
