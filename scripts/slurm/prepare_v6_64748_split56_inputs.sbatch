#!/bin/bash
#SBATCH -J prep_v6_64748_split56
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

SOURCE_BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64748
TARGET_BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_split56_candidate
CANDIDATE_LEDGER=apply/config/cell_ledger_v6_64748_split56_candidate.csv
FIT_SELECTOR=apply/config/cell_selector_v6_64748_split56_drop4_psfborrow.csv

echo "===== Prepare v6 64748 split56 candidate cache ====="
echo "Source: ${SOURCE_BINNED_ROOT}"
echo "Target: ${TARGET_BINNED_ROOT}"
date -Is
df -h /mnt/mydisk /home

if [[ -L "${TARGET_BINNED_ROOT}" ]]; then
  echo "Removing old symlink target ${TARGET_BINNED_ROOT}"
  rm -- "${TARGET_BINNED_ROOT}"
fi

python apply/stages/00c_prepare_v6_split56_inputs.py \
  --source-binned-root "${SOURCE_BINNED_ROOT}" \
  --target-binned-root "${TARGET_BINNED_ROOT}" \
  --source-bin-counts-csv "${SOURCE_BINNED_ROOT}/summary/bin_counts.csv" \
  --candidate-ledger-csv "${CANDIDATE_LEDGER}" \
  --prefit-selector-csv /tmp/v6_64748_unused_prefit_selector.csv \
  --systematics-selector-csv /tmp/v6_64748_unused_systematics_selector.csv \
  --high-energy-selector-csv /tmp/v6_64748_unused_high_energy_selector.csv \
  --manifest-json /tmp/v6_64748_unused_strategy_manifest.json \
  --diagnostics-html /tmp/v6_64748_unused_cell_selection_diagnostics.html \
  --prepare-cache \
  --overwrite-filtered-cache \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --progress-every 100

for stale_manifest in "${TARGET_BINNED_ROOT}"/summary/v6_*_split56_prepare_manifest.json; do
  [[ -f "${stale_manifest}" ]] || continue
  [[ "${stale_manifest}" == *"v6_64748_split56_prepare_manifest.json" ]] && continue
  mv "${stale_manifest}" "${TARGET_BINNED_ROOT}/summary/v6_64748_split56_prepare_manifest.json"
done

python - <<'PY'
import json
from pathlib import Path

manifest = Path("/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_split56_candidate/summary/v6_64748_split56_prepare_manifest.json")

def scrub(value):
    if isinstance(value, dict):
        return {key.replace("64670", "64748"): scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        return value.replace("64670", "64748")
    return value

if manifest.exists():
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")
PY

python - <<'PY'
from pathlib import Path
from apply.simulation_all_bin import sanitize_label
import csv

target = Path("/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_split56_candidate")
ledger = Path("apply/config/cell_ledger_v6_64748_split56_candidate.csv")
fit_selector = Path("apply/config/cell_selector_v6_64748_split56_drop4_psfborrow.csv")
rows = list(csv.DictReader(ledger.open(newline="")))
truthy = {"1", "true", "yes", "y"}
fit_keys = {
    (row["nhit_bin"], row["predE_bin"])
    for row in csv.DictReader(fit_selector.open(newline=""))
    if str(row.get("include", "")).strip().lower() in truthy
}
missing = []
empty = []
optional_missing = []
for row in rows:
    cell_dir = target / f"nhit_{sanitize_label(row['nhit_bin'])}" / f"predE_{sanitize_label(row['predE_bin'])}"
    key = (row["nhit_bin"], row["predE_bin"])
    if not cell_dir.exists():
        if key in fit_keys:
            missing.append((row["cell_id"], str(cell_dir)))
        else:
            optional_missing.append((row["cell_id"], str(cell_dir)))
    elif not any(cell_dir.iterdir()):
        empty.append((row["cell_id"], str(cell_dir)))
print(
    f"validated_cells={len(rows)} required_missing={len(missing)} "
    f"optional_missing={len(optional_missing)} empty={len(empty)}"
)
if missing or empty:
    print("missing_examples=", missing[:10])
    print("empty_examples=", empty[:10])
    raise SystemExit(1)
if optional_missing:
    print("optional_missing_examples=", optional_missing[:10])
PY

du -sh "${TARGET_BINNED_ROOT}" "${SOURCE_BINNED_ROOT}" 2>/dev/null || true
df -h /mnt/mydisk /home
date -Is
echo "v6 64748 split56 candidate cache ready"
