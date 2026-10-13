#!/usr/bin/env bash
#SBATCH --job-name=v6_cgti_e
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v6_commongti130_stage_e_%j.out
#SBATCH --error=logs/v6_commongti130_stage_e_%j.err

set -euo pipefail

APPLY_DIR=${APPLY_DIR:-/home/server/projects/energy_reconstruction/apply}
PYTHON=${PYTHON:-/home/server/anaconda3/envs/py310/bin/python}
TAG=v6_64748_nhit100_reselect44_split56_miss030_commongti130
STAGE_C_DIR="$APPLY_DIR/output/stage_c_v6_64748_nhit100_reselect44_split56_miss030_commongti130/runs/v6_64748_nhit100_reselect44_split56_miss030_commongti130_stage_c"
STAGE_D_DIR="$APPLY_DIR/output/stage_d_${TAG}/runs/${TAG}_stage_d"
OUTPUT_DIR="$APPLY_DIR/output/stage_e_${TAG}_containment1"
RUN_ID="${TAG}_stage_e_containment1"

mkdir -p "$APPLY_DIR/logs"
cd "$APPLY_DIR"

if [ ! -f "$STAGE_D_DIR/background_${TAG}.npz" ]; then
    echo "Missing Stage D output for commongti130: $STAGE_D_DIR/background_${TAG}.npz" >&2
    echo "Run run_v6_commongti130_stage_d_slurm.sh first." >&2
    exit 1
fi

"$PYTHON" stages/05_signal.py \
    --stage-c-dir "$STAGE_C_DIR" \
    --background-npz "$STAGE_D_DIR/background_${TAG}.npz" \
    --background-metadata "$STAGE_D_DIR/background_${TAG}_metadata.json" \
    --cell-selection-csv "config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$RUN_ID" \
    --no-promote-current \
    --containment-override fixed1 \
    --npz-name "signal_${TAG}_containment1.npz" \
    --metadata-name "signal_${TAG}_containment1_metadata.json" \
    --summary-csv-name "signal_${TAG}_containment1_summary.csv" \
    --summary-md-name "signal_${TAG}_containment1_summary.md"

echo "V6_COMMONGTI130_STAGE_E_COMPLETE $OUTPUT_DIR/runs/$RUN_ID/signal_${TAG}_containment1_metadata.json"
