#!/usr/bin/env bash
#SBATCH --job-name=wcda_time_receive
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
set -euo pipefail
if [[ $# -lt 2 ]]; then echo 'Usage: sbatch sync_observation_time.sbatch SNAPSHOT_ROOT CONFIG [--names NAME...]' >&2; exit 2; fi
snapshot=$1; config=$2; shift 2
# Existing receiver packages and SSH socket live on this same ETO host.
export PYTHONPATH="/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january/pydeps:$snapshot/scripts/data${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$snapshot/scripts/data/sync_observation_time.py" --config "$config" receive "$@"
