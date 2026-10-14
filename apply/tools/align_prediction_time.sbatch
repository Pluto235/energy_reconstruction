#!/bin/bash
#SBATCH --job-name=align_january_time
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/align_january_time_%j.out
#SBATCH --error=logs/slurm/align_january_time_%j.err
set -euo pipefail
cd /mnt/newdisk/shijie/energy/apply
PY=/mnt/newdisk/shijie/conda/envs/py310/bin/python
STATE=/mnt/newdisk/shijie/energy/.recovery/2026-10-14/prediction-time-alignment
mkdir -p "$STATE"
args=(--original-root /mnt/newdisk/shijie/WCDA_obs
      --prediction-root /mnt/newdisk/shijie/WCDA_obs_eval_64748
      --time-root /mnt/newdisk/shijie/WCDA_obs/recovered_time
      --observation-receipts /mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january/receipts
      --time-receipts /mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-january-time/receipts
                      /mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-january-time-v2/receipts
                      /mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-january-time-v3/receipts
      --workers "${SLURM_CPUS_PER_TASK:-4}")
if [[ ${1:-month} == smoke ]]; then
    args+=(--file 0101/Esg20220101_00.root
           --output-root "/mnt/newdisk/shijie/WCDA_obs_eval_64748_time_smoke_${SLURM_JOB_ID}")
else
    args+=(--output-root /mnt/newdisk/shijie/WCDA_obs_eval_64748_time_aligned)
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" tools/align_prediction_time.py "${args[@]}" --report "$STATE/align_${SLURM_JOB_ID}.json"
"$PY" tools/align_prediction_time.py "${args[@]}" --verify-only --report "$STATE/verify_${SLURM_JOB_ID}.json"
