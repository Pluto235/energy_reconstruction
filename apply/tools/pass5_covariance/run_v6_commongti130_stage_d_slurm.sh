#!/usr/bin/env bash
#SBATCH --job-name=v6_cgti_d
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v6_commongti130_stage_d_%j.out
#SBATCH --error=logs/v6_commongti130_stage_d_%j.err

set -euo pipefail

APPLY_DIR=${APPLY_DIR:-/home/server/projects/energy_reconstruction/apply}
PYTHON=${PYTHON:-/home/server/anaconda3/envs/py310/bin/python}
TAG=v6_64748_nhit100_reselect44_split56_miss030_commongti130
STAGE_C_DIR="$APPLY_DIR/output/stage_c_v6_64748_nhit100_reselect44_split56_miss030_commongti130/runs/v6_64748_nhit100_reselect44_split56_miss030_commongti130_stage_c"
OUTPUT_DIR="$APPLY_DIR/output/stage_d_${TAG}"
RUN_ID="${TAG}_stage_d"

mkdir -p "$APPLY_DIR/logs"
cd "$APPLY_DIR"

if [ ! -d "$STAGE_C_DIR/obs_events" ]; then
    echo "Missing filtered common-GTI Stage C dir: $STAGE_C_DIR" >&2
    echo "Run build_common_gti_obs_events.py first." >&2
    exit 1
fi

"$PYTHON" stages/04_background.py \
    --stage-c-dir "$STAGE_C_DIR" \
    --psf-npz "output/stage_b_v6_64748_nhit100_reselect44_split56_miss030/runs/v6_64748_nhit100_reselect44_split56_miss030_stage_b_psf/psf_v6_64748_nhit100_reselect44_split56_miss030.npz" \
    --cell-selection-csv "config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$RUN_ID" \
    --no-promote-current \
    --background-mode crab_roi_local \
    --roi-background-method annulus-quadratic \
    --roi-fiducial-deg 6.0 \
    --roi-edge-diagnostic-deg 8.0 \
    --roi-grid-step-deg 0.1 \
    --roi-edge-margin-deg 0.25 \
    --roi-source-mask-deg 2.0 \
    --roi-source-mask-r-opt-factor 2.0 \
    --annulus-default-inner-deg 1.5 \
    --annulus-width-deg 2.0 \
    --annulus-max-inner-deg 4.5 \
    --annulus-source-mask-min-deg 1.5 \
    --annulus-source-mask-r-opt-factor 2.0 \
    --annulus-source-mask-margin-deg 0.2 \
    --roi-surface-order 2 \
    --surface-condition-max 1.0e8 \
    --surface-min-training-pixels 80 \
    --batch-size 500000 \
    --workers 32 \
    --npz-name "background_${TAG}.npz" \
    --metadata-name "background_${TAG}_metadata.json" \
    --summary-csv-name "background_${TAG}_summary.csv" \
    --summary-md-name "background_${TAG}_summary.md"

echo "V6_COMMONGTI130_STAGE_D_COMPLETE $OUTPUT_DIR/runs/$RUN_ID/background_${TAG}_metadata.json"
