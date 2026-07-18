#!/bin/bash
#SBATCH -J v6_64748_poisson_unbinned_bootstrap
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:?ANALYTIC_BON_BASELINE_SHA is required}

EXPERIMENT=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_unbinned
BRANCH=h010_x0_y0
ROOT=apply/output/${EXPERIMENT}/branches/${BRANCH}
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
MANIFEST=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
STAGE_C=apply/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear
PSF=apply/output/stage_b_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh/runs/v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_stage_b_psf/psf_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh.npz
RESPONSE_ROOT=apply/output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh_annnorm
RESPONSE=${RESPONSE_ROOT}/response_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh.npz
RESPONSE_META=${RESPONSE_ROOT}/response_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh_metadata.json
D_DIR=${ROOT}/stage_d/runs/stage_d
E_DIR=${ROOT}/stage_e/runs/stage_e
F_ROOT=${ROOT}/stage_f
F_DIR=${F_ROOT}/runs/stage_f
G_ROOT=${ROOT}/stage_g
D_STEM=background_${BRANCH}
E_STEM=signal_${BRANCH}
F_STEM=fit_${BRANCH}
G_STEM=sed_points_${BRANCH}
OFF_EVENTS=${D_DIR}/off_events_${BRANCH}.npz
COV_NPZ=scheme_R_double_rayleigh_poisson_unbinned_background_covariance.npz
COV_META=scheme_R_double_rayleigh_poisson_unbinned_background_covariance.json

# 1) Unbinned bootstrap covariance (event resample + unbinned refit for order-2; analytic Poisson otherwise).
python apply/tools/bootstrap_v6_poisson_background_unbinned.py \
  --stage-d-npz "${D_DIR}/${D_STEM}.npz" --stage-e-npz "${E_DIR}/${E_STEM}.npz" \
  --off-events-npz "${OFF_EVENTS}" --pooling-manifest "${MANIFEST}" \
  --output-npz "${COV_NPZ}" --metadata-json "${COV_META}" \
  --replicates 1000 --seed 64748 --workers "${SLURM_CPUS_PER_TASK:-32}"

# 2) Registered time-split acceptance: two MJD halves (median split), unbinned, then the gate.
# NOTE: MJD_SPLIT is the calendar midpoint of the matched livetime; refine to the median event
# MJD for exactly-equal statistics if desired.
MJD_SPLIT=59670.16
TS_ROOT=apply/output/${EXPERIMENT}/time_split
run_half () { # $1=tag  $2..=extra mjd args
  local tag=$1; shift
  python apply/stages/04_background.py \
    --stage-c-dir "${STAGE_C}" --psf-npz "${PSF}" --cell-selection-csv "${SELECTOR}" \
    --output-dir "${TS_ROOT}" --run-id "${tag}" --no-promote-current --overwrite-run-dir \
    --background-mode crab_roi_local --roi-background-method annulus-quadratic \
    --roi-fit-statistic poisson --pooling-manifest "${MANIFEST}" \
    --poisson-quadratic-fit unbinned --allow-annulus-count-mismatch \
    --roi-fiducial-deg 6 --roi-edge-diagnostic-deg 8 --roi-grid-step-deg 0.10 \
    --roi-grid-offset-x-fraction 0 --roi-grid-offset-y-fraction 0 \
    --on-aperture-integration analytic-quadratic --annulus-normalize-surface \
    --annulus-default-inner-deg 1.5 --annulus-width-deg 2.0 --annulus-max-inner-deg 4.5 \
    --batch-size 500000 --workers "${SLURM_CPUS_PER_TASK:-32}" --print-every 100000 --no-plots \
    --npz-name background_${tag}.npz --metadata-name background_${tag}_metadata.json \
    --summary-csv-name background_${tag}_summary.csv --summary-md-name background_${tag}_summary.md "$@"
}
run_half half1 --mjd-max "${MJD_SPLIT}"
run_half half2 --mjd-min "${MJD_SPLIT}"

python apply/tools/time_split_shape_consistency.py \
  --nominal-npz "${D_DIR}/${D_STEM}.npz" \
  --half1-npz "${TS_ROOT}/runs/half1/background_half1.npz" \
  --half2-npz "${TS_ROOT}/runs/half2/background_half2.npz" \
  --bootstrap-cov-npz "${COV_NPZ}" --k 1.0 \
  --output-json scheme_R_double_rayleigh_poisson_unbinned_time_split.json

# 3) Covariance-aware Stage F + Stage G on the nominal unbinned branch.
python apply/stages/06_fit.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_DIR}/${E_STEM}.npz" --signal-metadata "${E_DIR}/${E_STEM}_metadata.json" \
  --excess-covariance-npz "${COV_NPZ}" --stage-c-dir "${STAGE_C}" --cell-subset-csv "${SELECTOR}" \
  --output-dir "${F_ROOT}" --run-id stage_f --no-promote-current --overwrite-run-dir \
  --report-html "${F_DIR}/stage_f_report.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --lhaaso-lat-deg 29.45 --lhaaso-lon-deg 100.14 --exposure-sample-step-sec 60 \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${F_STEM}.npz" --metadata-name "${F_STEM}_metadata.json" \
  --summary-csv-name "${F_STEM}_summary.csv" --summary-md-name "${F_STEM}_summary.md"

echo "Unbinned bootstrap + time-split + Stage F complete."
