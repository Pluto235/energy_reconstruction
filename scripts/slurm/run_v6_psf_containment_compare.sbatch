#!/bin/bash
#SBATCH -J v6_psf_contain_cmp
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

CONFIG=apply/config/v6_psf_containment_compare.json
COMPARE_ROOT=apply/output/v6_psf_containment_compare
SIGNAL_ROOT=apply/output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm/runs/v6_64748_nhit100_reselect44_split56_miss030_stage_e_containment1_annnorm
SIGNAL_NPZ=${SIGNAL_ROOT}/signal_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm.npz
SIGNAL_META=${SIGNAL_ROOT}/signal_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm_metadata.json
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
STAGE_C=apply/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear
RESPONSE_R=${COMPARE_ROOT}/response_r_fixed0715.npz
RESPONSE_R_META=${COMPARE_ROOT}/response_r_fixed0715_metadata.json
RESPONSE_B=apply/output/stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned/response_2d_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned.npz
RESPONSE_B_META=apply/output/stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned/response_2d_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned_metadata.json

if [[ "${FINALIZE_ONLY:-0}" == "1" ]]; then
  python apply/stages/09_compare_psf_containment.py finalize --config "${CONFIG}"
  exit 0
fi

python apply/stages/09_compare_psf_containment.py preflight --config "${CONFIG}"
python apply/stages/09_compare_psf_containment.py mc-stats --config "${CONFIG}" --workers "${SLURM_CPUS_PER_TASK:-32}" --files-per-task 250

included=$(python - <<'PY'
import csv
rows=csv.DictReader(open('apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv'))
print(','.join(row['cell_id'] for row in rows if row['include']=='1'))
PY
)
excluded=$(python - <<'PY'
import csv
rows=csv.DictReader(open('apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv'))
print(','.join(row['cell_id'] for row in rows if row['include']!='1'))
PY
)

run_fit() {
  local scheme=$1 response=$2 metadata=$3
  python apply/stages/06_fit.py \
    --response-npz "${response}" \
    --response-metadata "${metadata}" \
    --signal-npz "${SIGNAL_NPZ}" \
    --signal-metadata "${SIGNAL_META}" \
    --stage-c-dir "${STAGE_C}" \
    --cell-subset-csv "${SELECTOR}" \
    --output-dir "${COMPARE_ROOT}/stage_f_${scheme}" \
    --run-id "v6_psf_containment_compare_${scheme}" \
    --no-promote-current \
    --overwrite-run-dir \
    --report-html "${COMPARE_ROOT}/stage_f_${scheme}_report.html" \
    --npz-name "fit_${scheme}.npz" \
    --metadata-name "fit_${scheme}_metadata.json" \
    --summary-csv-name "fit_${scheme}_summary.csv" \
    --summary-md-name "fit_${scheme}_summary.md"
}

run_fit r "${RESPONSE_R}" "${RESPONSE_R_META}"
run_fit b "${RESPONSE_B}" "${RESPONSE_B_META}"

run_sed() {
  local scheme=$1 response=$2 metadata=$3
  local fit_root=${COMPARE_ROOT}/stage_f_${scheme}/runs/v6_psf_containment_compare_${scheme}
  python apply/stages/07_sed_points.py \
    --response-npz "${response}" \
    --response-metadata "${metadata}" \
    --signal-npz "${SIGNAL_NPZ}" \
    --signal-metadata "${SIGNAL_META}" \
    --stage-f-npz "${fit_root}/fit_${scheme}.npz" \
    --stage-f-metadata "${fit_root}/fit_${scheme}_metadata.json" \
    --output-dir "${COMPARE_ROOT}/stage_g_${scheme}" \
    --run-id "v6_psf_containment_compare_${scheme}" \
    --baseline-name "v6_psf_containment_compare_${scheme}" \
    --required-cell-ids "${included}" \
    --excluded-cell-ids "${excluded}" \
    --skip-expected-stage-f-validation \
    --no-promote-current \
    --overwrite-run-dir \
    --report-html "${COMPARE_ROOT}/stage_g_${scheme}_report.html" \
    --npz-name "sed_points_${scheme}.npz" \
    --metadata-name "sed_points_${scheme}_metadata.json" \
    --summary-csv-name "sed_points_${scheme}_summary.csv" \
    --summary-json-name "sed_points_${scheme}_summary.json" \
    --summary-md-name "sed_points_${scheme}_summary.md"
}

run_sed r "${RESPONSE_R}" "${RESPONSE_R_META}"
run_sed b "${RESPONSE_B}" "${RESPONSE_B_META}"
python apply/stages/09_compare_psf_containment.py finalize --config "${CONFIG}"

echo "v6 PSF containment comparison complete: ${SLURM_JOB_ID}"
