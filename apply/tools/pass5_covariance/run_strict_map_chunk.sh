#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
INDEX=${1:?job index is required}
DI_DIR=/home/lhaaso/hushicong/8_All_sky_survey/script/DI_mask_v1.1.1_nbins_v2
CONFIG="$DI_DIR/src/config/DI_nhit_7bins_cygni_Cob_PincOpt_4hours_le2000.txt"

set +u
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
set -u

row=$(awk -F '\t' -v job_index="$INDEX" 'NR > 1 && $1 == job_index {sub(/\r$/, ""); print; exit}' "$RUN_DIR/strict_hour_selection/jobs.tsv")
if [[ -z "$row" ]]; then
    echo "No job row for index $INDEX" >&2
    exit 2
fi
IFS=$'\t' read -r _ label list_path output_bkg output_j2000 <<< "$row"

mkdir -p "$(dirname "$output_bkg")"
if [[ ! -s "$output_bkg" && ! -s "$output_j2000" ]]; then
    "$DI_DIR/DI_Main_pinc_temp_Step2" "$list_path" "$CONFIG" "$output_bkg"
    [[ -s "$output_bkg" ]] || { echo "Missing Step2 output: $output_bkg" >&2; exit 3; }
fi
if [[ ! -s "$output_j2000" ]]; then
    "$DI_DIR/DI_Bkg_Jnow2J2000_v5_G2E_daily" \
        "$output_bkg" hon hbkg hoff "$output_j2000"
    [[ -s "$output_j2000" ]] || { echo "Missing J2000 output: $output_j2000" >&2; exit 4; }
fi
rm -f "$output_bkg"
echo "STRICT_MAP_CHUNK_COMPLETE $INDEX $label $output_j2000"
