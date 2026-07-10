#!/usr/bin/env bash
#SBATCH --job-name=v6_64748_theta_profiles
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=apply/output/slurm_logs/v6_64748_theta_profiles_%j.out
#SBATCH --error=apply/output/slurm_logs/v6_64748_theta_profiles_%j.err

set -eo pipefail

cd /home/server/projects/energy_reconstruction
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

python apply/report/build_v6_stage_b_theta_profile_cache.py
