#!/bin/bash
#SBATCH -J v5_psf_cmp
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH -t 72:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

export PYTHON_BIN=${PYTHON_BIN:-/home/server/anaconda3/envs/py310/bin/python}
export WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-18}}

echo "==== Job Info ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Python: ${PYTHON_BIN}"
echo "=================="

apply/run_v5_psf_comparison_pipeline.sh
