#!/bin/bash
#SBATCH -J v6_nhit_1r2r
#SBATCH -p main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

CODE_ROOT=${CODE_ROOT:-/home/server/projects/energy_reconstruction_double_king_finalize}
DATA_ROOT=${DATA_ROOT:-/home/server/projects/energy_reconstruction}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

BASE_RUN_ID=v6_64748_nhit100_reselect44_split56_miss030
INPUT_DIR=${DATA_ROOT}/apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-nhit-folded-rayleigh-exact
OUTPUT_DIR=${DATA_ROOT}/apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-nhit-folded-1r-2r-comparison

python apply/report/compare_v6_nhit_folded_rayleigh_models.py \
  --input-npz "${INPUT_DIR}/${BASE_RUN_ID}_nhit_folded_rayleigh.npz" \
  --output-dir "${OUTPUT_DIR}" \
  --random-starts 32 \
  --seed 20260721
