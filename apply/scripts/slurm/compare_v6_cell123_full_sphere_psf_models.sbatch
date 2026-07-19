#!/bin/bash
#SBATCH -J v6_c123_psfcmp
#SBATCH -p main
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

CODE_ROOT=${CODE_ROOT:-/home/server/projects/energy_reconstruction_double_king_finalize}
DATA_ROOT=${DATA_ROOT:-/home/server/projects/energy_reconstruction}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

INPUT_DIR=${DATA_ROOT}/apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-spherical-rayleigh-king-fit
OUTPUT_DIR=${DATA_ROOT}/apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-full-sphere-model-comparison
STEM=v6_64748_nhit100_reselect44_split56_miss030_cell123_spherical_rayleigh_king_fit

python apply/report/compare_v6_cell123_full_sphere_psf_models.py \
  --input-npz "${INPUT_DIR}/${STEM}.npz" \
  --input-metadata "${INPUT_DIR}/${STEM}_metadata.json" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${SLURM_CPUS_PER_TASK:-3}" \
  --random-starts 32 \
  --quadrature-order 20
