#!/bin/bash
#SBATCH -J v6_64748_poisson_manifest
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:?ANALYTIC_BON_BASELINE_SHA is required}
EXPERIMENT=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled
OUTPUT_ROOT=apply/output/${EXPERIMENT}
SELECTOR=apply/config/cell_selector_v6_64748_nhit100_reselect44_split56_miss030_fit.csv
SOURCE_TABLE=apply/config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv
PSF_DIR=apply/output/stage_b_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh/runs/v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_stage_b_psf
PSF=${PSF_DIR}/psf_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh.npz
PSF_META=${PSF_DIR}/psf_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_metadata.json
STAGE_C=apply/output/stage_c_v6_64748_nhit100_highEplus1_split56/runs/v6_64748_nhit100_highEplus1_split56_stage_c_halfyear
CANONICAL=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
STAGED=${OUTPUT_ROOT}/manifest/cell_background_pooling.json

git merge-base --is-ancestor "${ANALYTIC_BON_BASELINE_SHA}" HEAD
[[ "$(sha256sum "${SELECTOR}" | awk '{print $1}')" == c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6 ]]
test -s "${SOURCE_TABLE}"
test -s "${PSF}"
test -s "${PSF_META}"
test -s "${STAGE_C}/obs_events_metadata.json"
test -s "${STAGE_C}/source_files.csv"

python - "${SELECTOR}" "${SOURCE_TABLE}" "${PSF}" <<'PY'
import csv
import sys
import numpy as np

selector, source, psf = sys.argv[1:]
selector_rows = list(csv.DictReader(open(selector, encoding="utf-8")))
source_rows = list(csv.DictReader(open(source, encoding="utf-8")))
with np.load(psf, allow_pickle=False) as handle:
    psf_ids = set(np.asarray(handle["cell_id"], dtype=int).tolist())
donors = {int(row["cell_id"]) for row in source_rows if row["tail_bin_flag"] == "0"}
targets = {int(row["cell_id"]) for row in selector_rows if row["include"] == "1"}
tails = {13, 26, 39, 52, 65, 78, 91}
assert len(source_rows) == 91 and len(donors) == 84 and len(targets) == 44
assert ({int(row["cell_id"]) for row in source_rows} - donors) == tails
assert donors <= psf_ids
PY

mkdir -p "$(dirname "${STAGED}")" "$(dirname "${CANONICAL}")"
python apply/tools/build_v6_poisson_pooling_manifest.py \
  --stage-c-dir "${STAGE_C}" --psf-npz "${PSF}" --psf-metadata "${PSF_META}" \
  --selector "${SELECTOR}" --source-table "${SOURCE_TABLE}" --output "${STAGED}" \
  --analytic-baseline-sha "${ANALYTIC_BON_BASELINE_SHA}" \
  --batch-size 500000 --print-every 10

python - "${STAGED}" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="ascii"))
declared = payload.pop("manifest_sha256")
actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
assert declared == actual
assert len(payload["donor_universe_cell_ids"]) == 84
assert len(payload["target_cell_ids"]) == 44
assert payload["analytic_bon_baseline_sha"]
PY

cp "${STAGED}" "${CANONICAL}.tmp"
mv "${CANONICAL}.tmp" "${CANONICAL}"
echo "Frozen pooling manifest: ${CANONICAL}"
