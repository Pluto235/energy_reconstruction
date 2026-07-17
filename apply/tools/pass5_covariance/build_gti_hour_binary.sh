#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
DI_DIR=/home/lhaaso/hushicong/8_All_sky_survey/script/DI_mask_v1.1.1_nbins_v2
SOURCE_FILE="$RUN_DIR/upstream/DI_Main_pinc_temp_Cod_2_gti.cc"
OBJECT_FILE="$RUN_DIR/bin/DI_Main_pinc_temp_Cod_2_gti.o"
OUTPUT_FILE="$RUN_DIR/bin/DI_Main_pinc_temp_Cod_2_gti"

set +u
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
set -u

mkdir -p "$RUN_DIR/bin"
g++ -O2 -Wall -fPIC -D_MAIN_ \
    $(root-config --cflags) \
    -I"$DI_DIR" -I"$SLALIB_INCDIR" \
    -c "$SOURCE_FILE" -o "$OBJECT_FILE"
g++ -O2 "$OBJECT_FILE" "$DI_DIR/src/hpatimer.o" "$DI_DIR/src/papi.o" \
    $(root-config --ldflags --libs) \
    -lMinuit -lz -L"$SLALIB_LIBDIR" -lsla \
    -o "$OUTPUT_FILE"

status=0
"$OUTPUT_FILE" >/dev/null 2>&1 || status=$?
if [[ $status -ne 255 ]]; then
    echo "Unexpected no-argument exit status from $OUTPUT_FILE: $status" >&2
    exit 2
fi
echo "GTI_HOUR_BINARY_BUILD_COMPLETE $OUTPUT_FILE"
