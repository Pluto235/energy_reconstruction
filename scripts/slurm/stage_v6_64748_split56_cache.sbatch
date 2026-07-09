#!/bin/bash
#SBATCH -J stage_v6_64748_cache
#SBATCH -p main
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_split56_candidate}
STAGED_ROOT=${STAGED_ROOT:-/dev/shm/WCDA_simulation_binned_response_v6_64748_split56_candidate}
LOW_IO_PRIORITY=${LOW_IO_PRIORITY:-0}

echo "===== Stage v6 64748 split56 cache to fast local storage ====="
echo "source=${SOURCE_ROOT}"
echo "staged=${STAGED_ROOT}"
df -h /mnt/mydisk /dev/shm

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Missing source cache: ${SOURCE_ROOT}" >&2
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

source_files=$(find "${SOURCE_ROOT}" -type f | wc -l)
staged_files=$(find "${STAGED_ROOT}" -type f | wc -l)
echo "source_files=${source_files}"
echo "staged_files=${staged_files}"
du -sh "${SOURCE_ROOT}" "${STAGED_ROOT}"
df -h /dev/shm

if [[ "${source_files}" != "${staged_files}" ]]; then
  echo "Staged file count mismatch: source=${source_files}, staged=${staged_files}" >&2
  exit 3
fi

echo "v6 64748 split56 cache staging finished"
