#!/bin/bash
#SBATCH -J psf_b_v6_64748_r44_m030
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 06:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate
CELL_SELECTION=apply/config/cell_ledger_${SOURCE_RUN_ID}_candidate.csv
STAGE_A_METADATA=apply/output/stage_a_${SOURCE_RUN_ID}/response_2d_${SOURCE_RUN_ID}_metadata.json
OUTPUT_DIR=apply/output/stage_b_${RUN_ID}
RUN_STAGE_B=${RUN_ID}_stage_b_psf

python apply/stages/02_build_psf.py \
  --binned-root "${BINNED_ROOT}" \
  --cell-selection-csv "${CELL_SELECTION}" \
  --stage-a-metadata "${STAGE_A_METADATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_STAGE_B}" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --lhaaso-lat-deg 29.45 \
  --source-dec-deg 22.01 \
  --theta-min-deg 0.0 \
  --theta-max-deg 50.0 \
  --theta-step-deg 1.0 \
  --hour-angle-samples 200000 \
  --allow-missing-cell-dirs \
  --allow-incomplete-theta-support \
  --allow-low-stat-psf-fallback \
  --min-events-per-cell 1000 \
  --min-effective-events 200 \
  --core-fit-max-deg 3.0 \
  --theta-missing-mass-fail-threshold 0.30 \
  --containment-warning-tolerance 0.12 \
  --angle-check-max-events 20000 \
  --file-progress-every 5000 \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --npz-name "psf_${RUN_ID}.npz" \
  --metadata-name "psf_${RUN_ID}_metadata.json" \
  --summary-csv-name "psf_${RUN_ID}_summary.csv" \
  --summary-md-name "psf_${RUN_ID}_summary.md" \
  --overwrite-run-dir

python apply/stages/02f_build_unfiltered_psf_diagnostics.py \
  --binned-root "${BINNED_ROOT}" \
  --cell-selection-csv "${CELL_SELECTION}" \
  --output-npz "${OUTPUT_DIR}/runs/${RUN_STAGE_B}/psf_${RUN_ID}_unfiltered_diagnostic.npz" \
  --output-summary-csv "${OUTPUT_DIR}/runs/${RUN_STAGE_B}/psf_${RUN_ID}_unfiltered_diagnostic_summary.csv" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --profile-max-deg 5.0 \
  --profile-bin-width-deg 0.05 \
  --core-fit-max-deg 3.0 \
  --workers "${SLURM_CPUS_PER_TASK:-18}"

python - <<'PY'
import numpy as np
from pathlib import Path

run_id = "v6_64748_nhit100_reselect44_split56_miss030"
path = Path(f"apply/output/stage_b_{run_id}/runs/{run_id}_stage_b_psf/psf_{run_id}.npz")
with np.load(path, allow_pickle=False) as psf:
    cell_ids = np.asarray(psf["cell_id"], dtype=int)
    profiles = np.asarray(psf["profile_density"], dtype=float)
    effective = np.asarray(psf["effective_events"], dtype=float)
    core_effective = np.asarray(psf["core_fit_effective_events"], dtype=float)
    missing = np.asarray(psf["theta_missing_crab_probability_mass"], dtype=float)

required = {42, 56, 57, 70, 71, 86}
failures = []
for cell_id in sorted(required):
    matches = np.flatnonzero(cell_ids == cell_id)
    if matches.size != 1:
        failures.append(f"cell {cell_id}: found {matches.size} rows")
        continue
    idx = int(matches[0])
    if not np.any(profiles[idx] > 0):
        failures.append(f"cell {cell_id}: zero/fallback radial profile")
    if effective[idx] < 200 or core_effective[idx] < 200:
        failures.append(
            f"cell {cell_id}: Neff={effective[idx]:.3f}, core Neff={core_effective[idx]:.3f}"
        )
    if missing[idx] > 0.30:
        failures.append(f"cell {cell_id}: missing theta mass={missing[idx]:.6f}")

if failures:
    raise SystemExit("Stage B miss030 validation failed:\n" + "\n".join(failures))
print("Validated non-fallback Rayleigh profiles for cells", sorted(required))
PY
