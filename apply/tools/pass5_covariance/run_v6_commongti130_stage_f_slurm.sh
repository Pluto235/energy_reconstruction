#!/usr/bin/env bash
#SBATCH --job-name=v6_cgti_f
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v6_commongti130_stage_f_%j.out
#SBATCH --error=logs/v6_commongti130_stage_f_%j.err

set -euo pipefail

APPLY_DIR=${APPLY_DIR:-/home/server/projects/energy_reconstruction/apply}
PYTHON=${PYTHON:-/home/server/anaconda3/envs/py310/bin/python}
TAG=v6_64748_nhit100_reselect44_split56_miss030_commongti130
STAGE_E_DIR="$APPLY_DIR/output/stage_e_${TAG}_containment1/runs/${TAG}_stage_e_containment1"
OUTPUT_DIR="$APPLY_DIR/output/stage_f_$TAG"
RUN_ID="${TAG}_stage_f"

mkdir -p "$APPLY_DIR/tools/pass5_covariance/logs"
cd "$APPLY_DIR"

if [ ! -f "$STAGE_E_DIR/signal_${TAG}_containment1.npz" ]; then
    echo "Missing Stage E output for commongti130: $STAGE_E_DIR/signal_${TAG}_containment1.npz" >&2
    echo "Run run_v6_commongti130_stage_e_slurm.sh first." >&2
    exit 1
fi

"$PYTHON" stages/06_fit.py \
    --response-npz output/stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned/response_2d_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned.npz \
    --response-metadata output/stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned/response_2d_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned_metadata.json \
    --signal-npz "$STAGE_E_DIR/signal_${TAG}_containment1.npz" \
    --signal-metadata "$STAGE_E_DIR/signal_${TAG}_containment1_metadata.json" \
    --source-files-csv tools/pass5_covariance/common_gti/common_gti_source_files.csv \
    --cell-subset-csv config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$RUN_ID" \
    --no-promote-current \
    --report-html "report/stage_f_${TAG}_report.html" \
    --npz-name "fit_${TAG}.npz" \
    --metadata-name "fit_${TAG}_metadata.json" \
    --summary-csv-name "fit_${TAG}_summary.csv" \
    --summary-md-name "fit_${TAG}_summary.md"

echo "V6_COMMONGTI130_FIT_COMPLETE $OUTPUT_DIR/runs/$RUN_ID/fit_${TAG}_metadata.json"
