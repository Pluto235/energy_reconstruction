#!/bin/bash
#SBATCH -J v6_64748_R_analytic
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

PSF_KIND=${PSF_KIND:?PSF_KIND must be 1R or 2R}
BASE_RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
TARGET_CONTAINMENT=0.7129790300890827
SELECTOR=apply/config/cell_selector_${BASE_RUN_ID}_fit.csv
EXPECTED_SELECTOR_SHA=c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6

case "${PSF_KIND}" in
  1R)
    PSF_RUN_ID=${BASE_RUN_ID}
    ANALYSIS_ID=${BASE_RUN_ID}_analytic_bon
    EXPERIMENT_ID=${BASE_RUN_ID}_scheme_R_fixed712979_analytic_bon
    ;;
  2R)
    PSF_RUN_ID=${BASE_RUN_ID}_double_rayleigh
    ANALYSIS_ID=${PSF_RUN_ID}_analytic_bon
    EXPERIMENT_ID=${PSF_RUN_ID}_scheme_R_fixed712979_analytic_bon
    ;;
  *)
    echo "Unsupported PSF_KIND=${PSF_KIND}; expected 1R or 2R" >&2
    exit 2
    ;;
esac

actual_selector_sha=$(sha256sum "${SELECTOR}" | awk '{print $1}')
if [[ "${actual_selector_sha}" != "${EXPECTED_SELECTOR_SHA}" ]]; then
  echo "Selector changed: ${actual_selector_sha}" >&2
  exit 2
fi

PSF_DIR=apply/output/stage_b_${PSF_RUN_ID}/runs/${PSF_RUN_ID}_stage_b_psf
PSF=${PSF_DIR}/psf_${PSF_RUN_ID}.npz
STAGE_C=apply/output/stage_c_${SOURCE_RUN_ID}/runs/${SOURCE_RUN_ID}_stage_c_halfyear
RESPONSE_ROOT=apply/output/stage_e_${BASE_RUN_ID}_fixed712979_rayleigh_annnorm
RESPONSE=${RESPONSE_ROOT}/response_${BASE_RUN_ID}_fixed712979_rayleigh.npz
RESPONSE_META=${RESPONSE_ROOT}/response_${BASE_RUN_ID}_fixed712979_rayleigh_metadata.json

D_ROOT=apply/output/stage_d_${ANALYSIS_ID}_annnorm
D_RUN=${ANALYSIS_ID}_stage_d_annnorm
D_DIR=${D_ROOT}/runs/${D_RUN}
D_STEM=background_${ANALYSIS_ID}_annnorm
E_ROOT=apply/output/stage_e_${ANALYSIS_ID}_containment1_annnorm
E_RUN=${ANALYSIS_ID}_stage_e_containment1_annnorm
E_DIR=${E_ROOT}/runs/${E_RUN}
E_STEM=signal_${ANALYSIS_ID}_containment1_annnorm
F_ROOT=apply/output/stage_f_${EXPERIMENT_ID}
F_RUN=${EXPERIMENT_ID}_stage_f
F_DIR=${F_ROOT}/runs/${F_RUN}
F_STEM=fit_${EXPERIMENT_ID}
G_ROOT=apply/output/stage_g_${EXPERIMENT_ID}
G_RUN=${EXPERIMENT_ID}_stage_g
G_STEM=sed_points_${EXPERIMENT_ID}

python apply/report/validate_v6_scheme_r_double_rayleigh_analytic_bon.py --preflight-only

python apply/stages/04_background.py \
  --stage-c-dir "${STAGE_C}" --psf-npz "${PSF}" --cell-selection-csv "${SELECTOR}" --included-only \
  --output-dir "${D_ROOT}" --run-id "${D_RUN}" --no-promote-current --overwrite-run-dir \
  --background-mode crab_roi_local --roi-background-method annulus-quadratic \
  --roi-fiducial-deg 6 --roi-edge-diagnostic-deg 8 --roi-grid-step-deg 0.1 \
  --on-aperture-integration analytic-quadratic \
  --annulus-default-inner-deg 1.5 --annulus-width-deg 2.0 --annulus-max-inner-deg 4.5 \
  --annulus-normalize-surface --roi-surface-order 2 \
  --batch-size 500000 --workers "${SLURM_CPUS_PER_TASK:-32}" --print-every 10 \
  --npz-name "${D_STEM}.npz" --metadata-name "${D_STEM}_metadata.json" \
  --summary-csv-name "${D_STEM}_summary.csv" --summary-md-name "${D_STEM}_summary.md"

set +e
python apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C}" \
  --background-npz "${D_DIR}/${D_STEM}.npz" --background-metadata "${D_DIR}/${D_STEM}_metadata.json" \
  --cell-selection-csv "${SELECTOR}" --included-only \
  --output-dir "${E_ROOT}" --run-id "${E_RUN}" --no-promote-current --overwrite-run-dir \
  --report-html "apply/report/stage_e_${ANALYSIS_ID}_report.html" \
  --source-ra-deg 83.63 --source-dec-deg 22.01 --batch-size 500000 \
  --quality-min-total-sigma 50 --quality-max-total-sigma 300 --containment-override fixed1 \
  --print-every 10 --npz-name "${E_STEM}.npz" --metadata-name "${E_STEM}_metadata.json" \
  --summary-csv-name "${E_STEM}_summary.csv" --summary-md-name "${E_STEM}_summary.md"
stage_e_status=$?
set -e
[[ -f "${E_DIR}/${E_STEM}.npz" ]] || exit "${stage_e_status}"

python apply/stages/06_fit.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_DIR}/${E_STEM}.npz" --signal-metadata "${E_DIR}/${E_STEM}_metadata.json" \
  --stage-c-dir "${STAGE_C}" --cell-subset-csv "${SELECTOR}" \
  --output-dir "${F_ROOT}" --run-id "${F_RUN}" --no-promote-current --overwrite-run-dir \
  --report-html "apply/report/stage_f_${EXPERIMENT_ID}_report.html" \
  --source-ra-deg 83.63 --source-dec-deg 22.01 --lhaaso-lat-deg 29.45 --lhaaso-lon-deg 100.14 \
  --exposure-sample-step-sec 60 --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${F_STEM}.npz" --metadata-name "${F_STEM}_metadata.json" \
  --summary-csv-name "${F_STEM}_summary.csv" --summary-md-name "${F_STEM}_summary.md"

included=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] == "1"))' "${SELECTOR}")
excluded=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] != "1"))' "${SELECTOR}")

python apply/stages/07_sed_points.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_DIR}/${E_STEM}.npz" --signal-metadata "${E_DIR}/${E_STEM}_metadata.json" \
  --stage-f-npz "${F_DIR}/${F_STEM}.npz" --stage-f-metadata "${F_DIR}/${F_STEM}_metadata.json" \
  --output-dir "${G_ROOT}" --run-id "${G_RUN}" --no-promote-current --overwrite-run-dir \
  --report-html "apply/report/stage_g_${EXPERIMENT_ID}_report.html" \
  --baseline-name "${EXPERIMENT_ID}" --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 --skip-expected-stage-f-validation \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${G_STEM}.npz" --metadata-name "${G_STEM}_metadata.json" \
  --summary-csv-name "${G_STEM}_summary.csv" --summary-json-name "${G_STEM}_summary.json" \
  --summary-md-name "${G_STEM}_summary.md"

python apply/report/validate_v6_scheme_r_double_rayleigh_analytic_bon.py --branch "${PSF_KIND}"
echo "Scheme R ${PSF_KIND} analytic B_on Stage D-G complete: ${SLURM_JOB_ID}; target=${TARGET_CONTAINMENT}"
