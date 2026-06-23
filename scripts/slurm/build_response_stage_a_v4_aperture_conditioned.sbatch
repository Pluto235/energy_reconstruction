#!/bin/bash
#SBATCH -J build_resp_a_v4_ap
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

BINNED_ROOT=${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v3_candidate}
CELL_SELECTION=${CELL_SELECTION:-apply/config/cell_ledger_v3_candidate.csv}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_a_v4_aperture_conditioned}
APERTURE_PSF_NPZ=${APERTURE_PSF_NPZ:-apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal/psf_v3_candidate.npz}
NUMERATOR_WORKERS=${NUMERATOR_WORKERS:-${SLURM_CPUS_PER_TASK:-18}}
DENOMINATOR_WORKERS=${DENOMINATOR_WORKERS:-1}

rm -rf "${OUTPUT_DIR}"

python apply/stages/01_build_response.py \
  --binned-root "${BINNED_ROOT}" \
  --cell-selection-csv "${CELL_SELECTION}" \
  --output-dir "${OUTPUT_DIR}" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --allow-missing-cell-dirs \
  --denominator-workers "${DENOMINATOR_WORKERS}" \
  --numerator-workers "${NUMERATOR_WORKERS}" \
  --numerator-files-per-task 250 \
  --numerator-progress-every 20 \
  --aperture-psf-npz "${APERTURE_PSF_NPZ}" \
  --npz-name response_2d_v4_aperture_conditioned.npz \
  --metadata-name response_2d_v4_aperture_conditioned_metadata.json
