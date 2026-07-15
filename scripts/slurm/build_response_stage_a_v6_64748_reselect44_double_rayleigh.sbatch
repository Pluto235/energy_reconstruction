#!/bin/bash
#SBATCH -J resp_a_v6_64748_r44_dr
#SBATCH -p main
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 08:00:00
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
PSF_RUN=${RUN_ID}_stage_b_psf

python apply/stages/01_build_response.py \
  --binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate \
  --cell-selection-csv apply/config/cell_ledger_${SOURCE_RUN_ID}_candidate.csv \
  --run-dir /home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_noemax_no_core_cut_64748 \
  --output-dir "apply/output/stage_a_${RUN_ID}_aperture_conditioned" \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --allow-missing-cell-dirs \
  --denominator-workers 1 \
  --numerator-workers "${SLURM_CPUS_PER_TASK:-18}" \
  --numerator-files-per-task 250 \
  --numerator-progress-every 20 \
  --aperture-psf-npz "apply/output/stage_b_${RUN_ID}/runs/${PSF_RUN}/psf_${RUN_ID}.npz" \
  --npz-name "response_2d_${RUN_ID}_aperture_conditioned.npz" \
  --metadata-name "response_2d_${RUN_ID}_aperture_conditioned_metadata.json"
