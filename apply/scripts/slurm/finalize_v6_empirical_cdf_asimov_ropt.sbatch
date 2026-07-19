#!/bin/bash
#SBATCH -J v6_empirical_report
#SBATCH -p main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
PROJECT_ROOT=${PROJECT_ROOT:-/home/server/projects/energy_reconstruction}
cd "${PROJECT_ROOT}"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

bash apply/scripts/finalize_v6_empirical_cdf_asimov_ropt.sh
