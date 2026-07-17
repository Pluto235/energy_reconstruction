#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
INDEX=${1:?job index is required}
DI_DIR=/home/lhaaso/hushicong/8_All_sky_survey/script/DI_mask_v1.1.1_nbins_v2
CONFIG="$DI_DIR/src/config/DI_nhit_7bins_cygni_Cob_PincOpt_4hours_le2000.txt"
BINARY="$RUN_DIR/bin/DI_Main_pinc_temp_Cod_2_gti"
JOBS="$RUN_DIR/gti_hour_selection/jobs.tsv"

set +u
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
set -u

row=$(awk -F '\t' -v job_index="$INDEX" 'NR > 1 && $1 == job_index {sub(/\r$/, ""); print; exit}' "$JOBS")
if [[ -z "$row" ]]; then
    echo "No GTI hour job row for index $INDEX" >&2
    exit 2
fi
IFS=$'\t' read -r _ hour event_list gti_file output_acc output_bkg _ _ <<< "$row"

mkdir -p "$(dirname "$output_acc")"
if [[ ! -s "$output_acc" || ! -s "$output_bkg" ]]; then
    rm -f "$output_acc" "$output_bkg"
    "$BINARY" "$event_list" "$CONFIG" "$output_acc" "$output_bkg" "$event_list" "$gti_file"
fi
[[ -s "$output_acc" ]] || { echo "Missing GTI acceptance output: $output_acc" >&2; exit 3; }
[[ -s "$output_bkg" ]] || { echo "Missing GTI background output: $output_bkg" >&2; exit 4; }
echo "GTI_HOUR_COMPLETE $INDEX $hour $output_acc $output_bkg"
