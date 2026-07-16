#!/bin/bash
#SBATCH -J v6_64748_poisson_bootstrap
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

EXPERIMENT=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled
BRANCH=h010_x0_y0
ROOT=apply/output/${EXPERIMENT}/branches/${BRANCH}
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
MANIFEST=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
STAGE_C=apply/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear
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
COV_NPZ=scheme_R_double_rayleigh_poisson_background_covariance.npz
COV_META=scheme_R_double_rayleigh_poisson_background_covariance.json

python apply/tools/bootstrap_v6_poisson_background.py \
  --stage-d-npz "${D_DIR}/${D_STEM}.npz" --stage-e-npz "${E_DIR}/${E_STEM}.npz" \
  --pooling-manifest "${MANIFEST}" \
  --output-npz "${COV_NPZ}" --metadata-json "${COV_META}" \
  --replicates 1000 --seed 64748 --workers "${SLURM_CPUS_PER_TASK:-32}"

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

included=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] == "1"))' "${SELECTOR}")
excluded=$(python -c 'import csv,sys; print(",".join(r["cell_id"] for r in csv.DictReader(open(sys.argv[1])) if r["include"] != "1"))' "${SELECTOR}")
python apply/stages/07_sed_points.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_DIR}/${E_STEM}.npz" --signal-metadata "${E_DIR}/${E_STEM}_metadata.json" \
  --stage-f-npz "${F_DIR}/${F_STEM}.npz" --stage-f-metadata "${F_DIR}/${F_STEM}_metadata.json" \
  --output-dir "${G_ROOT}" --run-id stage_g --no-promote-current --overwrite-run-dir \
  --report-html "${G_ROOT}/runs/stage_g/stage_g_report.html" --baseline-name "${EXPERIMENT}" \
  --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 --skip-expected-stage-f-validation \
  --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${G_STEM}.npz" --metadata-name "${G_STEM}_metadata.json" \
  --summary-csv-name "${G_STEM}_summary.csv" --summary-json-name "${G_STEM}_summary.json" \
  --summary-md-name "${G_STEM}_summary.md"

echo "Completed nominal bootstrap, covariance-aware Stage F, and Stage G"
