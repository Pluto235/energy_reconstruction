#!/bin/bash
#SBATCH -J v6_202201_gate
#SBATCH -p main
#SBATCH --qos=default-safe
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH -e /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export PYTHONPATH=/mnt/newdisk/shijie/energy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/newdisk/shijie/conda/envs/py310/bin/python
TAG=v6_64748_202201_scheme_R_double_rayleigh_analytic_bon
"$PY" tests/test_january_v6_inputs.py
"$PY" tools/january_v6_inputs.py --output "output/$TAG/analysis_manifest.json"
"$PY" stages/03_reduce_obs.py \
  --obs-root /mnt/newdisk/shijie/WCDA_obs_eval_64748 \
  --time-root /mnt/newdisk/shijie/WCDA_obs_eval_64748_time_aligned \
  --file-glob 'Esg202201*.root' --day-prefix 01 --max-files 1 \
  --cell-selection-csv config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv \
  --output-dir "output/$TAG/smoke_c" --run-id "${TAG}_smoke_c_${SLURM_JOB_ID}" --no-promote-current \
  --tree-name t_eventout --time-tree-name t_recovered_time \
  --entries-per-chunk 200000 --workers 4 --match-status-equals 0 \
  --cut-pinc-max 1.1 --cut-fitstat-equals 0 --cut-theta-max-deg 50 --cut-dcedge-min 20 \
  --gap-threshold-sec 60 --print-every 1
