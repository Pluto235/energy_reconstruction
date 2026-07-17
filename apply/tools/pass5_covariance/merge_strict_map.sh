#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
DI_DIR=/home/lhaaso/hushicong/8_All_sky_survey/script/DI_mask_v1.1.1_nbins_v2
JOBS="$RUN_DIR/strict_hour_selection/jobs.tsv"
MAP_LIST=${MAP_LIST:-$RUN_DIR/strict_hour_selection/strict_j2000_maps.list}
OUTPUT=${OUTPUT:-$RUN_DIR/pass5_v6_sorted_gti_map.root}

set +u
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
set -u

if [[ ! -s "$MAP_LIST" ]]; then
    awk -F '\t' 'NR > 1 {print $5}' "$JOBS" > "$MAP_LIST"
fi
missing=0
map_count=0
while IFS= read -r path; do
    map_count=$((map_count + 1))
    if [[ ! -s "$path" ]]; then
        echo "Missing map: $path" >&2
        missing=$((missing + 1))
    fi
done < "$MAP_LIST"
if (( map_count == 0 )); then
    echo "No strict map chunks listed in $MAP_LIST" >&2
    exit 2
fi
if (( missing > 0 )); then
    echo "$missing strict map chunks are missing" >&2
    exit 3
fi

mkdir -p "$(dirname "$OUTPUT")"
"$DI_DIR/DI_Merge" "$MAP_LIST" hon hbkg hoff step1 "$OUTPUT"
[[ -s "$OUTPUT" ]] || { echo "Missing merged map: $OUTPUT" >&2; exit 4; }
echo "STRICT_MAP_MERGE_COMPLETE maps=$map_count output=$OUTPUT"
