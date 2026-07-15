#!/bin/bash
#SBATCH -J psf_b_v6_64748_r44_dr
#SBATCH -p main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}

BASE_RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
RUN_ID=${BASE_RUN_ID}_double_rayleigh
SOURCE_RUN_ID=v6_64748_nhit100_highEplus1_split56
SOURCE_DIR=apply/output/stage_b_${BASE_RUN_ID}/runs/${BASE_RUN_ID}_stage_b_psf
OUTPUT_DIR=apply/output/stage_b_${RUN_ID}

python apply/stages/02e_build_psf_v5_compare.py \
  --psf-method double_rayleigh_mixture \
  --binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate \
  --cell-selection-csv apply/config/cell_ledger_${SOURCE_RUN_ID}_candidate.csv \
  --stage-a-metadata apply/output/stage_a_${SOURCE_RUN_ID}/response_2d_${SOURCE_RUN_ID}_metadata.json \
  --double-rayleigh-stage-b-source-npz "${SOURCE_DIR}/psf_${BASE_RUN_ID}.npz" \
  --double-rayleigh-stage-b-source-metadata "${SOURCE_DIR}/psf_${BASE_RUN_ID}_metadata.json" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}_stage_b_psf" \
  --containment-warning-tolerance 0.12 \
  --no-borrow-v3-fallback-psf \
  --no-promote-current \
  --npz-name "psf_${RUN_ID}.npz" \
  --metadata-name "psf_${RUN_ID}_metadata.json" \
  --summary-csv-name "psf_${RUN_ID}_summary.csv" \
  --summary-md-name "psf_${RUN_ID}_summary.md" \
  --overwrite-run-dir

python - <<'PY'
import numpy as np
from pathlib import Path

base = "v6_64748_nhit100_reselect44_split56_miss030"
run = f"{base}_double_rayleigh"
new_path = Path(f"apply/output/stage_b_{run}/runs/{run}_stage_b_psf/psf_{run}.npz")
old_path = Path(f"apply/output/stage_b_{base}/runs/{base}_stage_b_psf/psf_{base}.npz")
with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
    assert np.array_equal(new["profile_density"], old["profile_density"])
    assert np.all(new["psf_method"].astype(str) == "double_rayleigh_mixture")
    quality = new["double_rayleigh_fit_quality"].astype(str)
    ok = quality == "ok"
    assert np.all((new["double_rayleigh_A"][ok] > 0) & (new["double_rayleigh_A"][ok] < 1))
    assert np.all((new["double_rayleigh_sigma1_deg"][ok] > 0) & (new["double_rayleigh_sigma1_deg"][ok] < new["double_rayleigh_sigma2_deg"][ok]))
    assert np.all(new["double_rayleigh_r_opt_deg"][ok] <= new["profile_edges_deg"][-1])
print(f"Validated isolated double-Rayleigh Stage B: {new_path}")
PY
