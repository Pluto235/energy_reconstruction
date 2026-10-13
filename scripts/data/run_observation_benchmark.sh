#!/usr/bin/env bash
set -euo pipefail
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u
work=/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013
if [[ ${1:-} == warm ]]; then
  shift
  exec python3 "$work/code/benchmark_observation_io.py" "$@" --baseline-repeat
fi
exec python3 "$work/code/benchmark_observation_io.py" "$@"
