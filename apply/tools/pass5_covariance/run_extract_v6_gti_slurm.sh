#!/usr/bin/env bash
#SBATCH --job-name=v6_gti
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_v6_gti_%j.out
#SBATCH --error=logs/extract_v6_gti_%j.err

set -euo pipefail

APPLY_DIR=${APPLY_DIR:-/home/server/projects/energy_reconstruction/apply}
RUN_DIR="$APPLY_DIR/tools/pass5_covariance"
SOURCE_FILES="$APPLY_DIR/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear/source_files.csv"
OUTPUT_DIR="$RUN_DIR/v6_gti_output"
PYTHON=${PYTHON:-/home/server/anaconda3/envs/py310/bin/python}

mkdir -p "$RUN_DIR/logs" "$OUTPUT_DIR"
cd "$APPLY_DIR"

"$PYTHON" "$RUN_DIR/extract_v6_gti.py" \
    --source-files "$SOURCE_FILES" \
    --output-tsv "$OUTPUT_DIR/v6_sorted_gti.tsv" \
    --manifest-json "$OUTPUT_DIR/v6_sorted_gti_manifest.json" \
    --match-status 0 \
    --gap-threshold-sec 60 \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

"$PYTHON" "$RUN_DIR/build_gti_source_files.py" \
    --source-files "$SOURCE_FILES" \
    --gti-tsv "$OUTPUT_DIR/v6_sorted_gti.tsv" \
    --output-csv "$OUTPUT_DIR/v6_sorted_gti_source_files.csv" \
    --manifest-json "$OUTPUT_DIR/v6_sorted_gti_source_files_manifest.json"

echo "V6_GTI_EXTRACTION_COMPLETE $OUTPUT_DIR/v6_sorted_gti.tsv"
