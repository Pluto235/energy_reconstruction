#!/usr/bin/env bash
# Immutable source snapshot and task-owned, checksum-pinned legacy dependencies.
set -euo pipefail
ulimit -c 0
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
control=$(dirname "$code")
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u
cog="$control/deps/Cog"
sla="$control/deps/slalib"
(cd "$control" && sha256sum --check deps/dependency_sha256.txt)
scratch=${_CONDOR_SCRATCH_DIR:?}/wcda_time
mkdir -p "$scratch"
export OBS_TIME_BINARY="$scratch/recover_observation_time"
export OBS_TIME_REFERENCE="$scratch/legacy_reference"
g++ -std=c++17 -O2 -I"$cog" -I"$sla/include" "$code/recover_observation_time.cc" \
  "$cog/hpatimer.cc" "$cog/papi.cc" -L"$sla/lib" -Wl,-rpath,"$sla/lib" \
  $(root-config --cflags --libs) -lMinuit -lsla -o "$OBS_TIME_BINARY"
case "${1:-}" in
  test)
    export OBS_TIME_ROOT_FIXTURES="$scratch/fixtures"
    python3 "$code/test_observation_time.py" ;;
  worker)
    shift
    g++ -std=c++17 -O2 -I"$cog" -I"$sla/include" "$control/deps/legacy_reference.cc" \
      "$cog/hpatimer.cc" "$cog/papi.cc" -L"$sla/lib" -Wl,-rpath,"$sla/lib" \
      $(root-config --cflags --libs) -lMinuit -lsla -o "$OBS_TIME_REFERENCE"
    python3 "$code/sync_observation_time.py" --config "$control/config.json" worker "$@" ;;
  *) echo 'Expected test or worker [--index N --workers N --once NAME]' >&2; exit 2 ;;
esac
