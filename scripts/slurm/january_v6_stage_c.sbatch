#!/bin/bash
#SBATCH -J v6_202201_c
#SBATCH -p main
#SBATCH --qos=default-safe
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH -e /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
export PYTHONPATH=/mnt/newdisk/shijie/energy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/newdisk/shijie/conda/envs/py310/bin/python
TAG=v6_64748_202201_scheme_R_double_rayleigh_analytic_bon
"$PY" -c 'import json;from pathlib import Path;m=json.loads(Path("output/"+"v6_64748_202201_scheme_R_double_rayleigh_analytic_bon"+"/analysis_manifest.json").read_text());assert m["result"]=="PASS" and m["file_count"]==681 and m["prediction_entries"]==48726074'
"$PY" stages/03_reduce_obs.py \
  --obs-root /mnt/newdisk/shijie/WCDA_obs_eval_64748 \
  --time-root /mnt/newdisk/shijie/WCDA_obs_eval_64748_time_aligned \
  --file-glob 'Esg202201*.root' --day-prefix 01 \
  --cell-selection-csv config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv \
  --output-dir "output/$TAG/c" --run-id "${TAG}_stage_c" --no-promote-current \
  --tree-name t_eventout --time-tree-name t_recovered_time \
  --entries-per-chunk 200000 --workers "${SLURM_CPUS_PER_TASK}" --match-status-equals 0 \
  --cut-pinc-max 1.1 --cut-fitstat-equals 0 --cut-theta-max-deg 50 --cut-dcedge-min 20 \
  --gap-threshold-sec 60 --print-every 25
