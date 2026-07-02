#!/bin/bash
#SBATCH -J prep_v6_split56
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH -t 12:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

SOURCE_BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64670
TARGET_BINNED_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64670_split56_candidate
CANDIDATE_LEDGER=apply/config/cell_ledger_v6_split56_candidate.csv
PREFIT_SELECTOR=apply/config/cell_selector_v6_split56_drop4_psfborrow.prefit.csv
FINAL_SELECTOR=apply/config/cell_selector_v6_split56_drop4_psfborrow.csv
BASELINE_SELECTOR=apply/config/cell_selector_v6_drop4_psfborrow.csv

python apply/stages/00a_validate_v6_mc_cache.py \
  --binned-root "${SOURCE_BINNED_ROOT}" \
  --expected-files 10000

python apply/stages/00c_prepare_v6_split56_inputs.py \
  --source-binned-root "${SOURCE_BINNED_ROOT}" \
  --target-binned-root "${TARGET_BINNED_ROOT}" \
  --source-bin-counts-csv "${SOURCE_BINNED_ROOT}/summary/bin_counts.csv" \
  --candidate-ledger-csv "${CANDIDATE_LEDGER}" \
  --prefit-selector-csv "${PREFIT_SELECTOR}" \
  --systematics-selector-csv apply/config/cell_selector_v6_split56_systematics.csv \
  --high-energy-selector-csv apply/config/cell_selector_v6_split56_high_energy_probes.csv \
  --manifest-json apply/config/v6_64670_split56_strategy_manifest.json \
  --diagnostics-html apply/report/v6_64670_split56_cell_selection_diagnostics.html \
  --write-configs \
  --prepare-cache \
  --write-diagnostics \
  --workers "${SLURM_CPUS_PER_TASK:-18}" \
  --progress-every 500

python apply/stages/00d_enforce_v6_split56_drop4_selector.py \
  --input-selector-csv "${PREFIT_SELECTOR}" \
  --output-selector-csv "${FINAL_SELECTOR}" \
  --candidate-ledger-csv "${CANDIDATE_LEDGER}" \
  --baseline-selector-csv "${BASELINE_SELECTOR}" \
  --subset-version v6_64670_split56_drop4_psfborrow \
  --expected-candidate-cells 91 \
  --expected-fit-cells 27
