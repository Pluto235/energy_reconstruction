#!/bin/bash
#SBATCH -J v6_spec_precision
#SBATCH -p main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction

ONE_D=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030/nhit-grouped-fit/v6_64748_nhit_grouped_fit.json
TWO_D=apply/output/stage_f_v6_64748_nhit100_reselect44_split56_miss030/runs/v6_64748_nhit100_reselect44_split56_miss030_stage_f/fit_v6_64748_nhit100_reselect44_split56_miss030_metadata.json
OUTPUT=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030/precision-comparison/v6_1d_vs_2d_spectral_relative_uncertainty

python apply/report/plot_v6_spectral_precision_comparison.py \
  --one-d-json "${ONE_D}" \
  --two-d-json "${TWO_D}" \
  --output-prefix "${OUTPUT}"

echo "Wrote ${OUTPUT}.png, ${OUTPUT}.pdf, and ${OUTPUT}.csv"
