#!/bin/bash
#SBATCH -J v6_202201_de
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
BASE=v6_64748_nhit100_reselect44_split56_miss030
ROOT="/mnt/newdisk/shijie/energy/apply/output/$TAG"
SELECTOR="/mnt/newdisk/shijie/energy/apply/config/cell_selector_${BASE}_fit.csv"
included=$($PY -c 'import csv,sys;print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"]=="1"))' "$SELECTOR")
C="$ROOT/c/runs/${TAG}_stage_c"
PSF="$ROOT/psf/runs/${TAG}_stage_b_psf/psf_${TAG}.npz"
D="$ROOT/d/runs/${TAG}_stage_d"
"$PY" tests/test_january_v6_validate.py
"$PY" tools/january_v6_validate.py --stage c
"$PY" tools/january_v6_validate.py --stage psf
"$PY" stages/04_background.py \
  --stage-c-dir "$C" --psf-npz "$PSF" --cell-selection-csv "$SELECTOR" \
  --output-dir "$ROOT/d" --run-id "${TAG}_stage_d" --no-promote-current \
  --background-mode crab_roi_local --roi-background-method annulus-quadratic --roi-fit-statistic weighted-ls \
  --roi-fiducial-deg 6 --roi-edge-diagnostic-deg 8 --roi-grid-step-deg 0.1 \
  --on-aperture-integration analytic-quadratic --analytic-required-cell-ids "$included" \
  --annulus-default-inner-deg 1.5 --annulus-width-deg 2 --annulus-max-inner-deg 4.5 \
  --annulus-normalize-surface --roi-surface-order 2 --batch-size 500000 \
  --workers "${SLURM_CPUS_PER_TASK}" --print-every 10 \
  --npz-name "background_${TAG}.npz" --metadata-name "background_${TAG}_metadata.json" \
  --summary-csv-name "background_${TAG}_summary.csv" --summary-md-name "background_${TAG}_summary.md"
"$PY" tools/january_v6_validate.py --stage d
"$PY" stages/05_signal.py \
  --stage-c-dir "$C" --background-npz "$D/background_${TAG}.npz" \
  --background-metadata "$D/background_${TAG}_metadata.json" --cell-selection-csv "$SELECTOR" \
  --output-dir "$ROOT/e" --run-id "${TAG}_stage_e" --no-promote-current \
  --report-html "$ROOT/stage_e_202201.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --batch-size 500000 --quality-min-total-sigma=-inf --quality-max-total-sigma=inf \
  --containment-override fixed1 --print-every 10 \
  --npz-name "signal_${TAG}.npz" --metadata-name "signal_${TAG}_metadata.json" \
  --summary-csv-name "signal_${TAG}_summary.csv" --summary-md-name "signal_${TAG}_summary.md"
"$PY" tools/january_v6_validate.py --stage e
