#!/usr/bin/env bash
set -euo pipefail
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u
work=/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013/.control
if [[ ${1:-} == warm ]]; then
  shift
  exec python3 "$work/code/benchmark_observation_io.py" "$@" --baseline-repeat
fi
exec python3 "$work/code/benchmark_observation_io.py" "$@"
