#!/bin/bash
#SBATCH -J v6_202201_diagnose
#SBATCH -p main
#SBATCH --qos=default-safe
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH -e /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export PYTHONPATH=/mnt/newdisk/shijie/energy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/newdisk/shijie/conda/envs/py310/bin/python
TAG=v6_64748_202201_scheme_R_double_rayleigh_analytic_bon
"$PY" tests/test_january_v6_diagnostic.py
"$PY" tools/january_v6_diagnostic.py --output-dir "output/$TAG/background_diagnostic"
