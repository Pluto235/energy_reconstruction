#!/bin/bash
#SBATCH -J v6_nonneg6
#SBATCH -p main
#SBATCH --qos=default-safe
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.out
#SBATCH -e /mnt/newdisk/shijie/energy/logs/slurm/%x_%j.err
set -euo pipefail
cd /mnt/newdisk/shijie/energy
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=/mnt/newdisk/shijie/conda/envs/py310/bin/python
SNAPSHOT=".recovery/2026-10-15/nonnegative6-source/${SLURM_JOB_ID}"
mkdir -p "$SNAPSHOT"
cp -p apply/tools/january_v6_nonnegative*.py apply/stages/nonnegative_quadratic.py \
  apply/report/build_january_v6_nonnegative_report.py \
  apply/config/january_v6_nonnegative_pseudoexperiments.json \
  scripts/slurm/january_v6_nonnegative.sbatch "$SNAPSHOT/"
if [ "$1" = smoke ]; then
  "$PY" -m unittest apply.tests.test_nonnegative_quadratic apply.tests.test_background_analytic_bon apply.tests.test_january_v6_validate
fi
case "$1" in
  spectral-toys) "$PY" apply/tools/january_v6_nonnegative_spectral_toys.py ;;
  report) "$PY" apply/report/build_january_v6_nonnegative_report.py --jobs "$2" \
    --renderer /mnt/newdisk/shijie/energy/.recovery/2026-10-15/january-v6-renderer/scripts/render_notion_report.py ;;
  verify)
    "$PY" -m unittest apply.tests.test_nonnegative_quadratic apply.tests.test_background_analytic_bon apply.tests.test_january_v6_validate
    "$PY" apply/tools/january_v6_nonnegative_delivery.py --jobs "$2" ;;
  *) "$PY" apply/tools/january_v6_nonnegative.py "$1" --workers "${SLURM_CPUS_PER_TASK}" ;;
esac
