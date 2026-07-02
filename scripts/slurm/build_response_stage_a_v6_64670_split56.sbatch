#!/bin/bash
#SBATCH -J resp_a_v6_split56
#SBATCH -p main
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH -t 02:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

BINNED_ROOT=${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v6_64670_split56_candidate}
CELL_SELECTION=${CELL_SELECTION:-apply/config/cell_ledger_v6_split56_candidate.csv}
RUN_DIR=${RUN_DIR:-/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_a_v6_64670_split56}
NPZ_NAME=${NPZ_NAME:-response_2d_v6_64670_split56.npz}
METADATA_NAME=${METADATA_NAME:-response_2d_v6_64670_split56_metadata.json}
APERTURE_PSF_NPZ=${APERTURE_PSF_NPZ:-}
NUMERATOR_WORKERS=${NUMERATOR_WORKERS:-${SLURM_CPUS_PER_TASK:-18}}
DENOMINATOR_WORKERS=${DENOMINATOR_WORKERS:-1}

CMD=(
  python apply/stages/01_build_response.py
  --binned-root "${BINNED_ROOT}"
  --cell-selection-csv "${CELL_SELECTION}"
  --run-dir "${RUN_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --tree-name t_eventout
  --weight-branch mc_weight
  --allow-missing-cell-dirs
  --denominator-workers "${DENOMINATOR_WORKERS}"
  --numerator-workers "${NUMERATOR_WORKERS}"
  --numerator-files-per-task 250
  --numerator-progress-every 20
  --npz-name "${NPZ_NAME}"
  --metadata-name "${METADATA_NAME}"
)
if [[ -n "${APERTURE_PSF_NPZ}" ]]; then
  CMD+=(--aperture-psf-npz "${APERTURE_PSF_NPZ}")
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
