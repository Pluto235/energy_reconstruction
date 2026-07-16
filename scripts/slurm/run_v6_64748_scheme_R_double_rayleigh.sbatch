#!/bin/bash
#SBATCH -J v6_64748_R_2R
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

BASE_RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
PSF_RUN_ID=${BASE_RUN_ID}_double_rayleigh
EXPERIMENT_ID=${PSF_RUN_ID}_scheme_R_fixed712979
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
SELECTOR=apply/config/cell_selector_${BASE_RUN_ID}_fit.csv
EXPECTED_SELECTOR_SHA=c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6

RESPONSE_ROOT=apply/output/stage_e_${BASE_RUN_ID}_fixed712979_rayleigh_annnorm
RESPONSE=${RESPONSE_ROOT}/response_${BASE_RUN_ID}_fixed712979_rayleigh.npz
RESPONSE_META=${RESPONSE_ROOT}/response_${BASE_RUN_ID}_fixed712979_rayleigh_metadata.json
STAGE_C=apply/output/stage_c_${SOURCE_RUN_ID}/runs/${SOURCE_RUN_ID}_stage_c_halfyear
STAGE_E=apply/output/stage_e_${PSF_RUN_ID}_containment1_annnorm/runs/${PSF_RUN_ID}_stage_e_containment1_annnorm
SIGNAL=${STAGE_E}/signal_${PSF_RUN_ID}_containment1_annnorm.npz
SIGNAL_META=${STAGE_E}/signal_${PSF_RUN_ID}_containment1_annnorm_metadata.json

F_ROOT=apply/output/stage_f_${EXPERIMENT_ID}
F_RUN=${EXPERIMENT_ID}_stage_f
F_DIR=${F_ROOT}/runs/${F_RUN}
F_STEM=fit_${EXPERIMENT_ID}
G_ROOT=apply/output/stage_g_${EXPERIMENT_ID}
G_RUN=${EXPERIMENT_ID}_stage_g
G_STEM=sed_points_${EXPERIMENT_ID}

actual_selector_sha=$(sha256sum "${SELECTOR}" | awk '{print $1}')
if [[ "${actual_selector_sha}" != "${EXPECTED_SELECTOR_SHA}" ]]; then
  echo "Selector changed: ${actual_selector_sha}" >&2
  exit 2
fi

python apply/report/validate_v6_scheme_r_double_rayleigh.py --preflight-only

python apply/stages/06_fit.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${SIGNAL}" --signal-metadata "${SIGNAL_META}" \
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
  --signal-npz "${SIGNAL}" --signal-metadata "${SIGNAL_META}" \
  --stage-f-npz "${F_DIR}/${F_STEM}.npz" --stage-f-metadata "${F_DIR}/${F_STEM}_metadata.json" \
  --output-dir "${G_ROOT}" --run-id "${G_RUN}" --no-promote-current --overwrite-run-dir \
  --report-html "apply/report/stage_g_${EXPERIMENT_ID}_report.html" \
  --baseline-name "${EXPERIMENT_ID}" --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 --skip-expected-stage-f-validation \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${G_STEM}.npz" --metadata-name "${G_STEM}_metadata.json" \
  --summary-csv-name "${G_STEM}_summary.csv" --summary-json-name "${G_STEM}_summary.json" \
  --summary-md-name "${G_STEM}_summary.md"

python apply/report/validate_v6_scheme_r_double_rayleigh.py
echo "Scheme R double-Rayleigh Stage F/G complete: ${SLURM_JOB_ID}"
