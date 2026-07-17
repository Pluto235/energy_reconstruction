#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
MEMORY_MB=${MEMORY_MB:-4000}
PRIORITY=${PRIORITY:-9}
WALLTIME=${WALLTIME:-short}
JOBS="$RUN_DIR/gti_hour_selection/jobs.tsv"

export GROUP=lhaaso
export GROUPNAME=lhaaso
set +u
source /afs/ihep.ac.cn/users/x/xishaoqiang/.bashrc_everyone
set -u

job_count=$(awk 'END {print NR - 1}' "$JOBS")
if (( job_count <= 0 )); then
    echo "No jobs found in $JOBS" >&2
    exit 2
fi

cd "$RUN_DIR"
hep_sub -g lhaaso -mem "$MEMORY_MB" -prio "$PRIORITY" -wt "$WALLTIME" \
    "$RUN_DIR/run_gti_hour.sh" -argu "%{ProcId}" -n "$job_count"

echo "GTI_HOUR_ARRAY_SUBMITTED jobs=$job_count memory_mb=$MEMORY_MB priority=$PRIORITY walltime=$WALLTIME"
