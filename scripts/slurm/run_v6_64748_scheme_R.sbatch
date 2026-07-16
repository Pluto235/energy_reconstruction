#!/bin/bash
#SBATCH -J v6_64748_R_1p58
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

E_ROOT=apply/output/stage_e_${RUN_ID}_fixed712979_rayleigh_annnorm
E_RUN=${RUN_ID}_stage_e_fixed712979_rayleigh_annnorm
E_DIR=${E_ROOT}/runs/${E_RUN}
E_STEM=signal_${RUN_ID}_fixed712979_rayleigh_annnorm
RESPONSE=${E_ROOT}/response_${RUN_ID}_fixed712979_rayleigh.npz
RESPONSE_META=${E_ROOT}/response_${RUN_ID}_fixed712979_rayleigh_metadata.json

F_ROOT=apply/output/stage_f_${RUN_ID}_scheme_R_fixed712979
F_RUN=${RUN_ID}_stage_f_scheme_R_fixed712979
F_DIR=${F_ROOT}/runs/${F_RUN}
F_STEM=fit_${RUN_ID}_scheme_R_fixed712979

G_ROOT=apply/output/stage_g_${RUN_ID}_scheme_R_fixed712979
G_RUN=${RUN_ID}_stage_g_scheme_R_fixed712979
G_DIR=${G_ROOT}/runs/${G_RUN}
G_STEM=sed_points_${RUN_ID}_scheme_R_fixed712979

python apply/stages/09_compare_psf_containment.py preflight --config apply/config/v6_psf_containment_compare.json
mkdir -p "${E_ROOT}"
python - "${RESPONSE}" "${RESPONSE_META}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np

response, target = map(Path, sys.argv[1:])
nominal = Path(
    "apply/output/stage_a_v6_64748_nhit100_highEplus1_split56/"
    "response_2d_v6_64748_nhit100_highEplus1_split56.npz"
)
nominal_meta = nominal.with_name(
    "response_2d_v6_64748_nhit100_highEplus1_split56_metadata.json"
)
radius_factor = 1.58
containment = 1.0 - math.exp(-0.5 * radius_factor**2)

with np.load(nominal, allow_pickle=True) as source:
    arrays = {key: source[key] for key in source.files}
arrays["a_eff"] = (arrays["a_eff"] * containment).astype(arrays["a_eff"].dtype, copy=False)
np.savez_compressed(response, **arrays)

data = json.loads(nominal_meta.read_text())
data.update(
    {
        "npz_path": str(response.resolve()),
        "formal_scheme": "R",
        "derived_response": {
            "mode": "fixed_rayleigh_containment",
            "source_nominal_response": str(nominal.resolve()),
            "radius_factor_sigma": radius_factor,
            "containment": containment,
            "formula": "Aeff_R = containment * Aeff_nominal",
            "rayleigh_note": (
                "For a 2D Gaussian PSF, r_opt = 1.58 sigma gives "
                "Rayleigh containment = 1 - exp(-1.58^2 / 2)."
            ),
        },
    }
)
data["single_application_contract"] = {
    "formula": "Aeff_R = 0.7129790300890827 * Aeff_nominal",
    "application_count": 1,
    "application_location": "derived fixed712979 response preparation",
    "downstream_containment": 1.0,
    "aperture_conditioned_response_used": False,
}
target.write_text(json.dumps(data, indent=2) + "\n")

with np.load(response, allow_pickle=True) as derived, np.load(nominal, allow_pickle=True) as source:
    mask = np.isfinite(source["a_eff"]) & (source["a_eff"] != 0)
    ratio = derived["a_eff"][mask] / source["a_eff"][mask]
    if not np.allclose(ratio, containment, rtol=2e-7, atol=0.0):
        raise SystemExit(
            f"Derived response ratio mismatch: [{ratio.min()}, {ratio.max()}] != {containment}"
        )
    for key in source.files:
        if key == "a_eff":
            continue
        derived_array = derived[key]
        source_array = source[key]
        equal = (
            np.array_equal(derived_array, source_array, equal_nan=True)
            if source_array.dtype.kind in "fc"
            else np.array_equal(derived_array, source_array)
        )
        if not equal:
            raise SystemExit(f"Unexpected non-a_eff response change: {key}")

print(f"Strict Scheme R containment: {containment:.16f} at {radius_factor:.2f} sigma")
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
  --baseline-name "${RUN_ID}_scheme_R_fixed712979" --required-cell-ids "${included}" --excluded-cell-ids "${excluded}" \
  --plot-hide-last-prede-points 2 \
  --skip-expected-stage-f-validation --pivot-tev 3 --reference-phi0 2.114e-12 --reference-gamma 2.69 \
  --npz-name "${G_STEM}.npz" --metadata-name "${G_STEM}_metadata.json" \
  --summary-csv-name "${G_STEM}_summary.csv" --summary-json-name "${G_STEM}_summary.json" \
  --summary-md-name "${G_STEM}_summary.md"

echo "Scheme R Stage E/F/G complete: ${SLURM_JOB_ID}"
