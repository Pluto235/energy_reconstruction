#!/usr/bin/env bash
# Four source-side HTCondor workers bound staging to at most four hourly files.
set -euo pipefail
case "${1:-}" in 0|1|2|3) ;; *) echo 'Expected shard index 0..3' >&2; exit 2 ;; esac
work=/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u
days=()
if [[ $1 == 0 ]]; then days+=(31); fi
for ((n=$1+1; n<=30; n+=4)); do days+=("$n"); done
for n in "${days[@]}"; do
  day=$(printf '202201%02d' "$n")
  python3 "$work/code/prepare_observation_day.py" "$day" "$work/manifests/$day.txt" \
    /scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013 "$work/audit" \
    /scratchfs2/lhaaso/liushijie/WCDA_observation_20220101_20220630
done
