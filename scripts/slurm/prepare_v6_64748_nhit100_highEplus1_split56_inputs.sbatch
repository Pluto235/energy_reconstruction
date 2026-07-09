#!/bin/bash
#SBATCH -J prep_v6_64748_n100_hi1
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH -t 24:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

RUN_ID=v6_64748_nhit100_highEplus1_split56
SOURCE_BINNED_ROOT=${SOURCE_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v6_64748}
TARGET_BINNED_ROOT=${TARGET_BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate}
RUN_DIR=${RUN_DIR:-/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_noemax_no_core_cut_64748}
CANDIDATE_LEDGER=${CANDIDATE_LEDGER:-apply/config/cell_ledger_${RUN_ID}_candidate.csv}
PREFIT_SELECTOR=${PREFIT_SELECTOR:-apply/config/cell_selector_${RUN_ID}_prefit.csv}
SYSTEMATICS_SELECTOR=${SYSTEMATICS_SELECTOR:-apply/config/cell_selector_${RUN_ID}_systematics.csv}
HIGH_ENERGY_SELECTOR=${HIGH_ENERGY_SELECTOR:-apply/config/cell_selector_${RUN_ID}_high_energy_probes.csv}
MANIFEST_JSON=${MANIFEST_JSON:-apply/config/${RUN_ID}_strategy_manifest.json}
DIAGNOSTICS_HTML=${DIAGNOSTICS_HTML:-apply/report/${RUN_ID}_cell_selection_diagnostics.html}

echo "===== Prepare ${RUN_ID} candidate cache/config ====="
echo "Source: ${SOURCE_BINNED_ROOT}"
echo "Target: ${TARGET_BINNED_ROOT}"
date -Is
df -h /mnt/mydisk /home

python apply/stages/00c_prepare_v6_split56_inputs.py \
  --source-binned-root "${SOURCE_BINNED_ROOT}" \
  --target-binned-root "${TARGET_BINNED_ROOT}" \
  --source-bin-counts-csv "${SOURCE_BINNED_ROOT}/summary/bin_counts.csv" \
  --candidate-ledger-csv "${CANDIDATE_LEDGER}" \
  --prefit-selector-csv "${PREFIT_SELECTOR}" \
  --systematics-selector-csv "${SYSTEMATICS_SELECTOR}" \
  --high-energy-selector-csv "${HIGH_ENERGY_SELECTOR}" \
  --manifest-json "${MANIFEST_JSON}" \
  --diagnostics-html "${DIAGNOSTICS_HTML}" \
  --run-dir "${RUN_DIR}" \
  --version "${RUN_ID}_candidate" \
  --strategy-name "nhit100_highEplus1_split56" \
  --source-low-nhit-bin "[100,200)" \
  --target-low-nhit-bin "[100,200)" \
  --source-high-nhit-bin ">=2000" \
  --target-high-nhit-bin "[2000,3000)" \
  --write-configs \
  --write-diagnostics \
  --prepare-cache \
  --overwrite-filtered-cache \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --progress-every 100

python - <<'PY'
import csv
import json
from pathlib import Path

run_id = "v6_64748_nhit100_highEplus1_split56"
ledger = Path(f"apply/config/cell_ledger_{run_id}_candidate.csv")
prefit = Path(f"apply/config/cell_selector_{run_id}_prefit.csv")
rows = list(csv.DictReader(ledger.open(newline="")))
prefit_rows = list(csv.DictReader(prefit.open(newline="")))
truthy = {"1", "true", "yes", "y"}
payload = {
    "ledger_rows": len(rows),
    "prefit_rows": len(prefit_rows),
    "first_nhit_bin": rows[0]["nhit_bin"] if rows else None,
    "tail_included": [
        row["cell_id"] for row in prefit_rows
        if row.get("predE_bin") == ">=6" and str(row.get("include", "")).lower() in truthy
    ],
    "prefit_include_count": sum(str(row.get("include", "")).lower() in truthy for row in prefit_rows),
}
print(json.dumps(payload, indent=2))
if payload["ledger_rows"] != 91 or payload["prefit_rows"] != 91:
    raise SystemExit("Expected 91 ledger/prefit rows")
if payload["first_nhit_bin"] != "[100,200)":
    raise SystemExit("First Nhit bin is not [100,200)")
if payload["tail_included"]:
    raise SystemExit("Prefit selector unexpectedly includes >=6 diagnostic tail")
PY

du -sh "${TARGET_BINNED_ROOT}" "${SOURCE_BINNED_ROOT}" 2>/dev/null || true
df -h /mnt/mydisk /home
date -Is
echo "${RUN_ID} candidate cache/config ready"
