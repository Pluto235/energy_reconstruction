#!/bin/bash
#SBATCH -J v6_202201_report
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
for test_file in tests/test_january_v6_{inputs,gti,validate,diagnostic}.py; do
  "$PY" "$test_file"
done
"$PY" report/build_january_v6_report.py \
  --renderer /mnt/newdisk/shijie/energy/.recovery/2026-10-15/january-v6-renderer/scripts/render_notion_report.py \
  --jobs 66081,66082,66083,66084,66085,66086,66087,66088
