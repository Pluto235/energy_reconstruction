#!/usr/bin/env bash
#SBATCH --job-name=v6_fit_gti
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v6_sorted_gti_fit_%j.out
#SBATCH --error=logs/v6_sorted_gti_fit_%j.err

set -euo pipefail

APPLY_DIR=${APPLY_DIR:-/home/server/projects/energy_reconstruction/apply}
PYTHON=${PYTHON:-/home/server/anaconda3/envs/py310/bin/python}
TAG=v6_64748_nhit100_reselect44_split56_miss030_sortedgti149
RUN_ID=${TAG}_stage_f
OUTPUT_DIR="$APPLY_DIR/output/stage_f_$TAG"

mkdir -p "$APPLY_DIR/tools/pass5_covariance/logs"
cd "$APPLY_DIR"

"$PYTHON" stages/06_fit.py \
    --response-npz output/stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned/response_2d_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned.npz \
    --response-metadata output/stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned/response_2d_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned_metadata.json \
    --signal-npz output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm/runs/v6_64748_nhit100_reselect44_split56_miss030_stage_e_containment1_annnorm/signal_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm.npz \
    --signal-metadata output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm/runs/v6_64748_nhit100_reselect44_split56_miss030_stage_e_containment1_annnorm/signal_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm_metadata.json \
    --source-files-csv tools/pass5_covariance/v6_gti_output/v6_sorted_gti_source_files.csv \
    --cell-subset-csv config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$RUN_ID" \
    --no-promote-current \
    --report-html "report/stage_f_${TAG}_report.html" \
    --npz-name "fit_${TAG}.npz" \
    --metadata-name "fit_${TAG}_metadata.json" \
    --summary-csv-name "fit_${TAG}_summary.csv" \
    --summary-md-name "fit_${TAG}_summary.md"

echo "V6_SORTED_GTI_FIT_COMPLETE $OUTPUT_DIR/runs/$RUN_ID/fit_${TAG}_metadata.json"
