#!/bin/bash
#SBATCH -J stage_obs_64748_nvme
#SBATCH -p main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 04:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/mnt/mydisk/WCDA_observation_eval_64748}
STAGED_ROOT=${STAGED_ROOT:-/tmp/WCDA_observation_eval_64748_stagec}
LOW_IO_PRIORITY=${LOW_IO_PRIORITY:-0}

echo "===== Stage 64748 observation eval to NVMe for Stage C ====="
echo "source=${SOURCE_ROOT}"
echo "staged=${STAGED_ROOT}"
df -h /mnt/mydisk /tmp

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Missing source observation eval root: ${SOURCE_ROOT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${STAGED_ROOT}")"
RSYNC_CMD=(rsync -a --delete)
if [[ "${LOW_IO_PRIORITY}" == "1" ]] && command -v ionice >/dev/null 2>&1; then
  RSYNC_CMD=(ionice -c3 "${RSYNC_CMD[@]}")
fi
printf '%q ' "${RSYNC_CMD[@]}" "${SOURCE_ROOT}/" "${STAGED_ROOT}/"
echo
"${RSYNC_CMD[@]}" "${SOURCE_ROOT}/" "${STAGED_ROOT}/"

source_roots=$(find "${SOURCE_ROOT}" -type f -name 'Esg*.root' ! -name '*.time.root' | wc -l)
staged_roots=$(find "${STAGED_ROOT}" -type f -name 'Esg*.root' ! -name '*.time.root' | wc -l)
source_times=$(find "${SOURCE_ROOT}/recovered_time" -type f -name '*.time.root' | wc -l)
staged_times=$(find "${STAGED_ROOT}/recovered_time" -type f -name '*.time.root' | wc -l)
echo "source_eval_roots=${source_roots}"
echo "staged_eval_roots=${staged_roots}"
echo "source_time_roots=${source_times}"
echo "staged_time_roots=${staged_times}"
du -sh "${SOURCE_ROOT}" "${STAGED_ROOT}"
df -h /tmp

if [[ "${source_roots}" != "${staged_roots}" ]]; then
  echo "Eval ROOT count mismatch: source=${source_roots}, staged=${staged_roots}" >&2
  exit 3
fi
if [[ "${source_times}" != "${staged_times}" ]]; then
  echo "Recovered-time ROOT count mismatch: source=${source_times}, staged=${staged_times}" >&2
  exit 4
fi

echo "64748 observation eval NVMe staging finished"
