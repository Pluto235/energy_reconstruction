#!/bin/bash
#SBATCH -J v6_64748_scheme_R
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

RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
SELECTOR=apply/config/cell_selector_${RUN_ID}_fit.csv
LEDGER=apply/config/cell_ledger_${SOURCE_RUN_ID}_candidate.csv
STAGE_C=apply/output/stage_c_${SOURCE_RUN_ID}/runs/${SOURCE_RUN_ID}_stage_c_halfyear
STAGE_D=apply/output/stage_d_${RUN_ID}_annnorm/runs/${RUN_ID}_stage_d_annnorm

E_ROOT=apply/output/stage_e_${RUN_ID}_fixed715_rayleigh_annnorm
E_RUN=${RUN_ID}_stage_e_fixed715_rayleigh_annnorm
E_DIR=${E_ROOT}/runs/${E_RUN}
E_STEM=signal_${RUN_ID}_fixed715_rayleigh_annnorm
RESPONSE=${E_ROOT}/response_${RUN_ID}_fixed715_rayleigh.npz
RESPONSE_META=${E_ROOT}/response_${RUN_ID}_fixed715_rayleigh_metadata.json

F_ROOT=apply/output/stage_f_${RUN_ID}_scheme_R
F_RUN=${RUN_ID}_stage_f_scheme_R
F_DIR=${F_ROOT}/runs/${F_RUN}
F_STEM=fit_${RUN_ID}_scheme_R

G_ROOT=apply/output/stage_g_${RUN_ID}_scheme_R
G_RUN=${RUN_ID}_stage_g_scheme_R
G_DIR=${G_ROOT}/runs/${G_RUN}
G_STEM=sed_points_${RUN_ID}_scheme_R

python apply/stages/09_compare_psf_containment.py preflight --config apply/config/v6_psf_containment_compare.json
mkdir -p "${E_ROOT}"
cp apply/output/v6_psf_containment_compare/response_r_fixed0715.npz "${RESPONSE}"
python - "${RESPONSE}" "${RESPONSE_META}" <<'PY'
import json, shutil, sys
from pathlib import Path
src = Path("apply/output/v6_psf_containment_compare/response_r_fixed0715_metadata.json")
response, target = map(Path, sys.argv[1:])
data = json.loads(src.read_text())
data["npz_path"] = str(response.resolve())
data["formal_scheme"] = "R"
data["single_application_contract"] = {
    "formula": "Aeff_R = 0.715 * Aeff_nominal",
    "application_count": 1,
    "application_location": "derived fixed715 Stage E response preparation",
    "downstream_containment": 1.0,
    "aperture_conditioned_response_used": False,
}
target.write_text(json.dumps(data, indent=2) + "\n")
PY

set +e
python apply/stages/05_signal.py \
  --stage-c-dir "${STAGE_C}" \
  --background-npz "${STAGE_D}/background_${RUN_ID}_annnorm.npz" \
  --background-metadata "${STAGE_D}/background_${RUN_ID}_annnorm_metadata.json" \
  --cell-selection-csv "${LEDGER}" \
  --output-dir "${E_ROOT}" --run-id "${E_RUN}" --no-promote-current --overwrite-run-dir \
  --report-html "apply/report/stage_e_${RUN_ID}_scheme_R_report.html" \
  --source-ra-deg 83.63 --source-dec-deg 22.01 --batch-size 500000 \
  --quality-min-total-sigma 50 --quality-max-total-sigma 300 --containment-override fixed1 \
  --npz-name "${E_STEM}.npz" --metadata-name "${E_STEM}_metadata.json" \
  --summary-csv-name "${E_STEM}_summary.csv" --summary-md-name "${E_STEM}_summary.md"
e_status=$?
set -e
[[ -f "${E_DIR}/${E_STEM}.npz" ]] || exit "${e_status}"

python apply/stages/06_fit.py \
  --response-npz "${RESPONSE}" --response-metadata "${RESPONSE_META}" \
  --signal-npz "${E_DIR}/${E_STEM}.npz" --signal-metadata "${E_DIR}/${E_STEM}_metadata.json" \
  --stage-c-dir "${STAGE_C}" --cell-subset-csv "${SELECTOR}" \
  --output-dir "${F_ROOT}" --run-id "${F_RUN}" --no-promote-current --overwrite-run-dir \
  --report-html "apply/report/stage_f_${RUN_ID}_scheme_R_report.html" \
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
  --report-html "apply/report/stage_g_${RUN_ID}_scheme_R_report.html" \
  --baseline-name "${RUN_ID}_scheme_R" --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 \
  --skip-expected-stage-f-validation --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${G_STEM}.npz" --metadata-name "${G_STEM}_metadata.json" \
  --summary-csv-name "${G_STEM}_summary.csv" --summary-json-name "${G_STEM}_summary.json" \
  --summary-md-name "${G_STEM}_summary.md"

echo "Scheme R Stage E/F/G complete: ${SLURM_JOB_ID}"
