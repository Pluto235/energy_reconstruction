#!/bin/bash
#SBATCH -J v6_c123_rayking
#SBATCH -p main
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

CODE_ROOT=${CODE_ROOT:-/home/server/projects/energy_reconstruction_double_king_finalize}
DATA_ROOT=${DATA_ROOT:-/home/server/projects/energy_reconstruction}
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

BASE=v6_64748_nhit100_reselect44_split56_miss030
FERMI_RUN=${BASE}_fermi_double_king

python apply/report/fit_v6_cell123_spherical_rayleigh_king.py \
  --binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate \
  --cell-ledger "${DATA_ROOT}/apply/config/cell_ledger_v6_64748_nhit100_highEplus1_split56_candidate.csv" \
  --source-psf "${DATA_ROOT}/apply/output/stage_b_${BASE}/runs/${BASE}_stage_b_psf/psf_${BASE}.npz" \
  --source-metadata "${DATA_ROOT}/apply/output/stage_b_${BASE}/runs/${BASE}_stage_b_psf/psf_${BASE}_metadata.json" \
  --double-king-npz "${DATA_ROOT}/apply/output/stage_b_${FERMI_RUN}/runs/${FERMI_RUN}/psf_${FERMI_RUN}.npz" \
  --output-dir "${DATA_ROOT}/apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-spherical-rayleigh-king-fit" \
  --cell-ids 1,2,3 \
  --workers "${SLURM_CPUS_PER_TASK:-3}" \
  --random-starts 24 \
  --quadrature-order 20 \
  --file-progress-every 1000
