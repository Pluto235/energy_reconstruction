#!/bin/bash
#SBATCH -J resp_a_v5_ap
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

STRATEGY=${STRATEGY:?Set STRATEGY=gap025 or STRATEGY=gap1}
BINNED_ROOT=${BINNED_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v5_predbin_${STRATEGY}}
CELL_LEDGER=${CELL_LEDGER:-apply/config/cell_ledger_v5_predbin_${STRATEGY}.csv}
STAGE_B_DIR=${STAGE_B_DIR:-apply/output/stage_b_v5_predbin_${STRATEGY}}
APERTURE_PSF_NPZ=${APERTURE_PSF_NPZ:-${STAGE_B_DIR}/current/psf_v5_predbin_${STRATEGY}.npz}
OUTPUT_DIR=${OUTPUT_DIR:-apply/output/stage_a_v5_predbin_${STRATEGY}_aperture_conditioned}
NUMERATOR_WORKERS=${NUMERATOR_WORKERS:-${SLURM_CPUS_PER_TASK:-18}}
DENOMINATOR_WORKERS=${DENOMINATOR_WORKERS:-1}
MAX_FILES_PER_CELL=${MAX_FILES_PER_CELL:-}

rm -rf "${OUTPUT_DIR}"

CMD=(
  python apply/stages/01_build_response.py
  --binned-root "${BINNED_ROOT}"
  --cell-selection-csv "${CELL_LEDGER}"
  --output-dir "${OUTPUT_DIR}"
  --tree-name t_eventout
  --weight-branch mc_weight
  --allow-missing-cell-dirs
  --denominator-workers "${DENOMINATOR_WORKERS}"
  --numerator-workers "${NUMERATOR_WORKERS}"
  --numerator-files-per-task 250
  --numerator-progress-every 20
  --aperture-psf-npz "${APERTURE_PSF_NPZ}"
  --npz-name response_2d_v5_predbin_${STRATEGY}_aperture_conditioned.npz
  --metadata-name response_2d_v5_predbin_${STRATEGY}_aperture_conditioned_metadata.json
)
if [[ -n "${MAX_FILES_PER_CELL}" ]]; then
  CMD+=(--max-binned-files-per-cell "${MAX_FILES_PER_CELL}")
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
