#!/usr/bin/env bash
# files DAY NAME... runs a measured cohort; shard INDEX COUNT resumes the month.
set -euo pipefail
work=$(cd "$(dirname "$0")/.." && pwd -P)
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u
run_day() {
  local day=$1
  shift
  python3 "$work/code/prepare_observation_day.py" "$day" "$work/manifests/$day.txt" \
    "$(dirname "$work")" "$work/audit" \
    /scratchfs2/lhaaso/liushijie/WCDA_observation_20220101_20220630 "$@"
}
case "${1:-}" in
  cohort)
    label=$2; day=$3; shift 3
    [[ $label =~ ^[a-z0-9_]+$ ]] || exit 2
    gate="$work/optimization_20261013/$label"
    touch "$gate/ready_$day"
    deadline=$((SECONDS+1800))
    until [[ -f "$gate/start" ]]; do
      (( SECONDS < deadline )) || exit 3
      sleep 1
    done
    run_day "$day" --names "$@" ;;
  files) day=$2; shift 2; run_day "$day" --names "$@" ;;
  shard)
    index=$2; count=$3
    [[ $index =~ ^[0-9]+$ && $count =~ ^[1-8]$ && $index -lt $count ]] || exit 2
    mapfile -t days < <(python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); from observation_state import DAYS; print("\n".join(DAYS))' "$work/code")
    (( ${#days[@]} > 0 )) || exit 2
    for ((n=index; n<${#days[@]}; n+=count)); do run_day "${days[n]}"; done ;;
  *) echo 'Expected files DAY NAME..., cohort LABEL DAY NAME..., or shard INDEX COUNT' >&2; exit 2 ;;
esac
