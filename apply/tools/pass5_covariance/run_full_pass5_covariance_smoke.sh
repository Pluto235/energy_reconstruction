#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
OFFICIAL_DIR=/home/lhaaso/liushijie/energy/wcda_crab_sed_pass5_20260616_104941

export GROUP=lhaaso
export GROUPNAME=lhaaso
set +u
source /afs/ihep.ac.cn/users/x/xishaoqiang/.bashrc_everyone
set -u

cd "$OFFICIAL_DIR"
cp src_v2.yaml "$RUN_DIR/full_pass5_covariance_input.yaml"
tune_yaml "$RUN_DIR/full_pass5_covariance_input.yaml" set_source free_all_norm
tune_yaml "$RUN_DIR/full_pass5_covariance_input.yaml" set_source free_one_sed J0534+2200
"$RUN_DIR/bin/gtlike_cov" \
    "$RUN_DIR/full_pass5_covariance_input.yaml" \
    "$RUN_DIR/full_pass5_covariance_smoke.yaml"
