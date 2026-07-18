#!/bin/bash
#SBATCH -J psf_2king_v6
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=24G
#SBATCH -t 01:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

python apply/report/fit_v6_stage_b_fermi_double_king.py \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --random-starts 16 \
  --gamma-min 1.05
