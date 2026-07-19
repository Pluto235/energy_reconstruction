#!/bin/bash
#SBATCH -J v6_c123_2kcomp
#SBATCH -p main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

CODE_ROOT=${CODE_ROOT:-/home/server/projects/energy_reconstruction_double_king_finalize}
DATA_ROOT=${DATA_ROOT:-/home/server/projects/energy_reconstruction}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

INPUT_DIR=${DATA_ROOT}/apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-full-sphere-model-comparison
OUTPUT_DIR=${DATA_ROOT}/apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-spherical-king-components
STEM=v6_64748_nhit100_reselect44_split56_miss030_cell123_full_sphere_psf_model_comparison

python apply/report/plot_v6_cell123_double_spherical_king_components.py \
  --input-npz "${INPUT_DIR}/${STEM}.npz" \
  --input-metadata "${INPUT_DIR}/${STEM}_metadata.json" \
  --output-dir "${OUTPUT_DIR}" \
  --quadrature-order 20
