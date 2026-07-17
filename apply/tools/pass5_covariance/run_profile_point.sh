#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_2022H1_covariance}
INDEX=${1:?profile point index is required}

export GROUP=lhaaso
export GROUPNAME=lhaaso
source /afs/ihep.ac.cn/users/x/xishaoqiang/.bashrc_everyone
cd "$RUN_DIR"

LABEL=$(python3 - "$INDEX" <<'PY'
import json
import sys

with open("profile_manifest.json", encoding="utf-8") as handle:
    manifest = json.load(handle)
print(manifest["points"][int(sys.argv[1])]["label"])
PY
)

gtlike "profiles/input_${LABEL}.yaml" "profiles/output_${LABEL}.yaml"

