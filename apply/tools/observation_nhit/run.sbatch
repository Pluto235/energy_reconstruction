#!/bin/bash
#SBATCH --job-name=january-nhit-friends
#SBATCH --partition=main
#SBATCH --qos=default-safe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /mnt/newdisk/shijie/conda/envs/py310/bin/python -u -B tools/observation_nhit/run.py --run-id "${1:?run ID required}"
