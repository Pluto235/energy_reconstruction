#!/usr/bin/env bash
#SBATCH --job-name=bkgscan_lowE
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bkgscan_lowE_%A_%a.out
#SBATCH --error=logs/bkgscan_lowE_%A_%a.err
#
# Check 3: low-Nhit background-systematics scan (UN-FROZEN WLS annulus-quadratic).
#
# One array task per variant runs the un-frozen Stage D -> E -> F -> G chain into a
# _diag/bkgscan/<variant>/ namespace, then the final array index runs the aggregator.
# Everything written by this job lives under _diag/bkgscan/ so nothing frozen
# (poisson_pooled / poisson_unbinned outputs, manifests, reports) is touched.
#
# Stage D here uses  --roi-fit-statistic weighted-ls  with NO --pooling-manifest,
# i.e. the un-frozen WLS path (no frozen manifest / baseline-SHA guard).
#
# Submit with the array sized to the variant count, e.g.:
#   N=$(/home/server/anaconda3/envs/py310/bin/python \
#        tools/background_lowE_systematics_scan.py --emit-variant-names | wc -l)
#   sbatch --array=0-$((N-1)) scripts/slurm/lowE_bkg_systematics_scan.sbatch
# Then aggregate (after all variants finish):
#   sbatch --dependency=afterok:<jobid> --array=999 scripts/slurm/lowE_bkg_systematics_scan.sbatch

set -euo pipefail

# --- env / paths (mirrors tools/pass5_covariance/run_v6_commongti130_stage_*_slurm.sh) ---
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

APPLY_DIR=${APPLY_DIR:-/home/server/projects/energy_reconstruction/apply}
REPO_DIR=${REPO_DIR:-/home/server/projects/energy_reconstruction}
PYTHON=${PYTHON:-/home/server/anaconda3/envs/py310/bin/python}
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cd "$APPLY_DIR"
mkdir -p "$APPLY_DIR/logs"

# --- baseline inputs (frozen upstream products; READ-only here) ---
BASE_TAG=v6_64748_nhit100_reselect44_split56_miss030
STAGE_C_DIR="output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear"
PSF_NPZ="output/stage_b_${BASE_TAG}/runs/${BASE_TAG}_stage_b_psf/psf_${BASE_TAG}.npz"
CELL_SELECTION_CSV="config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv"
RESPONSE_NPZ="output/stage_a_${BASE_TAG}_aperture_conditioned/response_2d_${BASE_TAG}_aperture_conditioned.npz"
RESPONSE_META="output/stage_a_${BASE_TAG}_aperture_conditioned/response_2d_${BASE_TAG}_aperture_conditioned_metadata.json"
FIT_SELECTOR_CSV="config/cell_selector_${BASE_TAG}_fit.csv"
SOURCE_FILES_CSV="tools/pass5_covariance/common_gti/common_gti_source_files.csv"

# --- diagnostic namespace: nothing frozen is written under here ---
# Stages run with cwd=$APPLY_DIR, so all --output-dir paths are relative to it and
# start with "output/" (no "apply/" prefix). The emitted Stage-D command inherits
# this prefix so its --output-dir matches the Stage E/F/G input paths below.
OUTPUT_PREFIX="output/${BASE_TAG}"
SCAN_DIR="${APPLY_DIR}/output/${BASE_TAG}/_diag/bkgscan"
SCAN_TOOL="tools/background_lowE_systematics_scan.py"

# --- variant list (baseline + one-knob-at-a-time) ---
mapfile -t VARIANTS < <("$PYTHON" "$SCAN_TOOL" --emit-variant-names)
N_VARIANTS=${#VARIANTS[@]}

# ---------------------------------------------------------------------------
# Aggregation task: run with --array=999 (any index >= N_VARIANTS) after the
# per-variant tasks finish.
# ---------------------------------------------------------------------------
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$N_VARIANTS" ]; then
    echo "BKGSCAN_AGGREGATE start scan_dir=$SCAN_DIR"
    "$PYTHON" "$SCAN_TOOL" --aggregate \
        --scan-dir "$SCAN_DIR" \
        --out-json "$SCAN_DIR/bkgscan_lowE_systematics.json" \
        --out-png "$SCAN_DIR/bkgscan_lowE_systematics.png"
    echo "BKGSCAN_AGGREGATE_COMPLETE $SCAN_DIR/bkgscan_lowE_systematics.json"
    exit 0
fi

# ---------------------------------------------------------------------------
# Per-variant task.
# ---------------------------------------------------------------------------
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "BKGSCAN_VARIANT $VARIANT (task ${SLURM_ARRAY_TASK_ID:-0}/${N_VARIANTS})"

VARIANT_DIR="output/${BASE_TAG}/_diag/bkgscan/${VARIANT}"
STAGE_D_ROOT="${VARIANT_DIR}/stage_d"
STAGE_E_ROOT="${VARIANT_DIR}/stage_e"
STAGE_F_ROOT="${VARIANT_DIR}/stage_f"
STAGE_G_ROOT="${VARIANT_DIR}/stage_g"

STAGE_D_RUN="bkgscan_${VARIANT}_stage_d"
STAGE_E_RUN="bkgscan_${VARIANT}_stage_e"
STAGE_F_RUN="bkgscan_${VARIANT}_stage_f"
STAGE_G_RUN="bkgscan_${VARIANT}_stage_g"

BKG_NPZ="background_bkgscan_${VARIANT}.npz"
BKG_META="background_bkgscan_${VARIANT}_metadata.json"
SIG_NPZ="signal_bkgscan_${VARIANT}.npz"
SIG_META="signal_bkgscan_${VARIANT}_metadata.json"
FIT_NPZ="fit_bkgscan_${VARIANT}.npz"
FIT_META="fit_bkgscan_${VARIANT}_metadata.json"
SED_NPZ="sed_points_bkgscan_${VARIANT}.npz"
SED_META="sed_points_bkgscan_${VARIANT}_metadata.json"

STAGE_D_RUN_DIR="${STAGE_D_ROOT}/runs/${STAGE_D_RUN}"
STAGE_E_RUN_DIR="${STAGE_E_ROOT}/runs/${STAGE_E_RUN}"
STAGE_F_RUN_DIR="${STAGE_F_ROOT}/runs/${STAGE_F_RUN}"

# --- Stage D: un-frozen WLS annulus-quadratic (NO --pooling-manifest) ---
# The exact per-variant Stage-D argv comes from the tool's --emit-commands, so the
# knob overrides stay in one place. Pick the line for this variant's output-dir.
STAGE_D_CMD=$("$PYTHON" "$SCAN_TOOL" --emit-commands \
    --python "$PYTHON" \
    --output-prefix "$OUTPUT_PREFIX" \
    --stage-c-dir "$STAGE_C_DIR" \
    --psf-npz "$PSF_NPZ" \
    --cell-selection-csv "$CELL_SELECTION_CSV" \
    --roi-fiducial-deg 6.0 \
    --roi-grid-step-deg 0.1 \
    --annulus-max-inner-deg 4.5 \
    | grep -F "/bkgscan/${VARIANT}/stage_d ")
if [ -z "$STAGE_D_CMD" ]; then
    echo "No emitted Stage-D command for variant $VARIANT" >&2
    exit 1
fi
# Guard: never let a frozen pooled-Poisson path slip in.
if echo "$STAGE_D_CMD" | grep -q -- "--pooling-manifest"; then
    echo "Refusing to run: emitted Stage-D command contains --pooling-manifest" >&2
    exit 1
fi
echo "STAGE_D_CMD: $STAGE_D_CMD"
eval "$STAGE_D_CMD"

# --- Stage E: signal from this variant's background ---
"$PYTHON" stages/05_signal.py \
    --stage-c-dir "$STAGE_C_DIR" \
    --background-npz "${STAGE_D_RUN_DIR}/${BKG_NPZ}" \
    --background-metadata "${STAGE_D_RUN_DIR}/${BKG_META}" \
    --cell-selection-csv "$CELL_SELECTION_CSV" \
    --output-dir "$STAGE_E_ROOT" \
    --run-id "$STAGE_E_RUN" \
    --no-promote-current \
    --containment-override fixed1 \
    --npz-name "$SIG_NPZ" \
    --metadata-name "$SIG_META" \
    --summary-csv-name "signal_bkgscan_${VARIANT}_summary.csv" \
    --summary-md-name "signal_bkgscan_${VARIANT}_summary.md"

# --- Stage F: spectral fit ---
"$PYTHON" stages/06_fit.py \
    --response-npz "$RESPONSE_NPZ" \
    --response-metadata "$RESPONSE_META" \
    --signal-npz "${STAGE_E_RUN_DIR}/${SIG_NPZ}" \
    --signal-metadata "${STAGE_E_RUN_DIR}/${SIG_META}" \
    --source-files-csv "$SOURCE_FILES_CSV" \
    --cell-subset-csv "$FIT_SELECTOR_CSV" \
    --output-dir "$STAGE_F_ROOT" \
    --run-id "$STAGE_F_RUN" \
    --no-promote-current \
    --report-html "report/_diag/bkgscan/stage_f_bkgscan_${VARIANT}_report.html" \
    --npz-name "$FIT_NPZ" \
    --metadata-name "$FIT_META" \
    --summary-csv-name "fit_bkgscan_${VARIANT}_summary.csv" \
    --summary-md-name "fit_bkgscan_${VARIANT}_summary.md"

# --- Stage G: diagnostic SED points ---
"$PYTHON" stages/07_sed_points.py \
    --response-npz "$RESPONSE_NPZ" \
    --response-metadata "$RESPONSE_META" \
    --signal-npz "${STAGE_E_RUN_DIR}/${SIG_NPZ}" \
    --signal-metadata "${STAGE_E_RUN_DIR}/${SIG_META}" \
    --stage-f-npz "${STAGE_F_RUN_DIR}/${FIT_NPZ}" \
    --stage-f-metadata "${STAGE_F_RUN_DIR}/${FIT_META}" \
    --output-dir "$STAGE_G_ROOT" \
    --run-id "$STAGE_G_RUN" \
    --no-promote-current \
    --report-html "report/_diag/bkgscan/stage_g_bkgscan_${VARIANT}_report.html" \
    --npz-name "$SED_NPZ" \
    --metadata-name "$SED_META" \
    --summary-csv-name "sed_points_bkgscan_${VARIANT}_summary.csv" \
    --summary-json-name "sed_points_bkgscan_${VARIANT}_summary.json" \
    --summary-md-name "sed_points_bkgscan_${VARIANT}_summary.md"

echo "BKGSCAN_VARIANT_COMPLETE $VARIANT ${STAGE_G_ROOT}/runs/${STAGE_G_RUN}/${SED_NPZ}"
