#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
WORKERS=${WORKERS:-32}
MEMORY_MB=${MEMORY_MB:-4000}
PRIORITY=${PRIORITY:-9}
WALLTIME=${WALLTIME:-mid}
RECOVERY_JOBS="$RUN_DIR/strict_recovery/recovery_jobs.tsv"

if [[ ! -s "$RECOVERY_JOBS" ]]; then
    echo "Missing recovery queue: $RECOVERY_JOBS" >&2
    exit 2
fi

recovery_count=$(awk 'END {print NR - 1}' "$RECOVERY_JOBS")
if (( recovery_count <= 0 )); then
    echo "No J2000 recovery jobs are required"
    exit 0
fi

export GROUP=lhaaso
export GROUPNAME=lhaaso
set +u
source /afs/ihep.ac.cn/users/x/xishaoqiang/.bashrc_everyone
set -u

cd "$RUN_DIR"
hep_sub -g lhaaso -mem "$MEMORY_MB" -prio "$PRIORITY" -schedd schedd07 -wt "$WALLTIME" \
    "$RUN_DIR/run_strict_j2000_recovery_worker.sh" \
    -argu "%{ProcId}" "$WORKERS" -n "$WORKERS"

echo "J2000_RECOVERY_ARRAY_SUBMITTED jobs=$recovery_count workers=$WORKERS memory_mb=$MEMORY_MB walltime=$WALLTIME"
