#!/bin/bash
#SBATCH -J v6_64748_poisson_unbinned_grid
#SBATCH -p main
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%A_%a.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%A_%a.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:?ANALYTIC_BON_BASELINE_SHA is required}

steps=(0.05 0.05 0.05 0.05 0.10 0.10 0.10 0.10 0.20 0.20 0.20 0.20)
xfrac=(0 0.5 0 0.5 0 0.5 0 0.5 0 0.5 0 0.5)
yfrac=(0 0 0.5 0.5 0 0 0.5 0.5 0 0 0.5 0.5)
branch_ids=(h005_x0_y0 h005_xh_y0 h005_x0_yh h005_xh_yh h010_x0_y0 h010_xh_y0 h010_x0_yh h010_xh_yh h020_x0_y0 h020_xh_y0 h020_x0_yh h020_xh_yh)

INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
STEP=${steps[$INDEX]}
XFRAC=${xfrac[$INDEX]}
YFRAC=${yfrac[$INDEX]}
BRANCH=${branch_ids[$INDEX]}
EXPERIMENT=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_unbinned
ROOT=apply/output/${EXPERIMENT}/branches/${BRANCH}
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
MANIFEST=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
STAGE_C=apply/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear
PSF=apply/output/stage_b_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh/runs/v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_stage_b_psf/psf_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh.npz
RESPONSE_ROOT=apply/output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh_annnorm
RESPONSE=${RESPONSE_ROOT}/response_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh.npz
RESPONSE_META=${RESPONSE_ROOT}/response_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh_metadata.json
D_ROOT=${ROOT}/stage_d
E_ROOT=${ROOT}/stage_e
F_ROOT=${ROOT}/stage_f
D_DIR=${D_ROOT}/runs/stage_d
E_DIR=${E_ROOT}/runs/stage_e
F_DIR=${F_ROOT}/runs/stage_f
D_STEM=background_${BRANCH}
E_STEM=signal_${BRANCH}
F_STEM=fit_${BRANCH}

# Only the nominal branch (h010_x0_y0, INDEX 4) dumps OFF events for the unbinned bootstrap.
DUMP_ARG=""
if [[ "${INDEX}" == "4" ]]; then
  DUMP_ARG="--dump-off-events-npz ${D_DIR}/off_events_${BRANCH}.npz"
fi

test -s "${MANIFEST}"
python apply/stages/04_background.py \
  --stage-c-dir "${STAGE_C}" --psf-npz "${PSF}" --cell-selection-csv "${SELECTOR}" \
  --output-dir "${D_ROOT}" --run-id stage_d --no-promote-current --overwrite-run-dir \
  --background-mode crab_roi_local --roi-background-method annulus-quadratic \
  --roi-fit-statistic poisson --pooling-manifest "${MANIFEST}" \
  --poisson-quadratic-fit unbinned ${DUMP_ARG} \
  --roi-fiducial-deg 6 --roi-edge-diagnostic-deg 8 --roi-grid-step-deg "${STEP}" \
  --roi-grid-offset-x-fraction "${XFRAC}" --roi-grid-offset-y-fraction "${YFRAC}" \
  --on-aperture-integration analytic-quadratic --annulus-normalize-surface \
  --annulus-default-inner-deg 1.5 --annulus-width-deg 2.0 --annulus-max-inner-deg 4.5 \
  --batch-size 500000 --workers "${SLURM_CPUS_PER_TASK:-32}" --print-every 10 --no-plots \
  --npz-name "${D_STEM}.npz" --metadata-name "${D_STEM}_metadata.json" \
  --summary-csv-name "${D_STEM}_summary.csv" --summary-md-name "${D_STEM}_summary.md"

set +e
python apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C}" --background-npz "${D_DIR}/${D_STEM}.npz" \
  --background-metadata "${D_DIR}/${D_STEM}_metadata.json" --cell-selection-csv "${SELECTOR}" --included-only \
  --output-dir "${E_ROOT}" --run-id stage_e --no-promote-current --overwrite-run-dir \
  --report-html "${E_DIR}/stage_e_report.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --batch-size 500000 --quality-min-total-sigma 50 --quality-max-total-sigma 300 \
  --containment-override fixed1 --print-every 10 --no-plots \
  --npz-name "${E_STEM}.npz" --metadata-name "${E_STEM}_metadata.json" \
  --summary-csv-name "${E_STEM}_summary.csv" --summary-md-name "${E_STEM}_summary.md"
stage_e_status=$?
set -e
[[ -s "${E_DIR}/${E_STEM}.npz" ]] || exit "${stage_e_status}"

python apply/stages/06_fit.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_DIR}/${E_STEM}.npz" --signal-metadata "${E_DIR}/${E_STEM}_metadata.json" \
  --stage-c-dir "${STAGE_C}" --cell-subset-csv "${SELECTOR}" \
  --output-dir "${F_ROOT}" --run-id stage_f --no-promote-current --overwrite-run-dir \
  --report-html "${F_DIR}/stage_f_report.html" --source-ra-deg 83.63 --source-dec-deg 22.01 \
  --lhaaso-lat-deg 29.45 --lhaaso-lon-deg 100.14 --exposure-sample-step-sec 60 \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 --no-plots \
  --npz-name "${F_STEM}.npz" --metadata-name "${F_STEM}_metadata.json" \
  --summary-csv-name "${F_STEM}_summary.csv" --summary-md-name "${F_STEM}_summary.md"

echo "Completed unbinned ${BRANCH}: h=${STEP}, xfrac=${XFRAC}, yfrac=${YFRAC}"
