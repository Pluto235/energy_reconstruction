#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
RECOVERY_JOBS=${RECOVERY_JOBS:-$RUN_DIR/strict_recovery/recovery_jobs.tsv}
WORKER_ID=${1:?worker id is required}
WORKER_COUNT=${2:?worker count is required}
DI_DIR=/home/lhaaso/hushicong/8_All_sky_survey/script/DI_mask_v1.1.1_nbins_v2
EOS_MGM=${EOS_MGM:-root://eos01.ihep.ac.cn}
XRDFS=(env -u LD_LIBRARY_PATH /usr/bin/xrdfs)
XRDCP=(env -u LD_LIBRARY_PATH /usr/bin/xrdcp)

set +u
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
set -u

completed=0
failed=0
WORK_ROOT=${TMPDIR:-/tmp}/pass5_j2000_recovery_${WORKER_ID}_$$
mkdir -p "$WORK_ROOT"
trap 'rm -rf "$WORK_ROOT"' EXIT

eos_path() {
    local uri=$1
    printf '%s\n' "${uri#${EOS_MGM}/}"
}

remote_size() {
    local path
    path=$(eos_path "$1")
    "${XRDFS[@]}" "$EOS_MGM" stat "$path" 2>/dev/null | awk '/^Size:/ {print $2}'
}

remote_remove() {
    local path
    path=$(eos_path "$1")
    "${XRDFS[@]}" "$EOS_MGM" rm "$path" >/dev/null 2>&1 || true
}

while IFS=$'\t' read -r recovery_index original_index label output_bkg output_j2000; do
    [[ "$recovery_index" == "recovery_index" ]] && continue
    (( recovery_index % WORKER_COUNT == WORKER_ID )) || continue

    j2000_size=$(remote_size "$output_j2000")
    if [[ -n "$j2000_size" && "$j2000_size" -gt 0 ]]; then
        echo "J2000_RECOVERY_SKIP original_index=$original_index label=$label output=$output_j2000"
        completed=$((completed + 1))
        continue
    fi
    bkg_size=$(remote_size "$output_bkg")
    if [[ -z "$bkg_size" || "$bkg_size" -le 0 ]]; then
        echo "J2000_RECOVERY_MISSING_BKG original_index=$original_index label=$label input=$output_bkg" >&2
        failed=$((failed + 1))
        continue
    fi

    local_bkg="$WORK_ROOT/${label}_BKG.root"
    local_j2000="$WORK_ROOT/${label}_BKG_J2000.root"
    rm -f "$local_bkg" "$local_j2000"
    remote_remove "$output_j2000"
    if "${XRDCP[@]}" --force --retry 3 --cksum adler32 "$output_bkg" "$local_bkg" \
        && [[ -s "$local_bkg" ]] \
        && "$DI_DIR/DI_Bkg_Jnow2J2000_v5_G2E_daily" \
            "$local_bkg" hon hbkg hoff "$local_j2000" \
        && [[ -s "$local_j2000" ]] \
        && "${XRDCP[@]}" --force --posc --retry 3 --cksum adler32 --rm-bad-cksum \
            "$local_j2000" "$output_j2000"; then
        local_size=$(stat -c %s "$local_j2000")
        uploaded_size=$(remote_size "$output_j2000")
        if [[ -n "$uploaded_size" && "$uploaded_size" -eq "$local_size" ]]; then
            remote_remove "$output_bkg"
            echo "J2000_RECOVERY_COMPLETE original_index=$original_index label=$label output=$output_j2000"
            completed=$((completed + 1))
        else
            remote_remove "$output_j2000"
            echo "J2000_RECOVERY_SIZE_MISMATCH original_index=$original_index label=$label" >&2
            failed=$((failed + 1))
        fi
    else
        remote_remove "$output_j2000"
        echo "J2000_RECOVERY_FAILED original_index=$original_index label=$label" >&2
        failed=$((failed + 1))
    fi
    rm -f "$local_bkg" "$local_j2000"
done < "$RECOVERY_JOBS"

echo "J2000_RECOVERY_WORKER_DONE worker=$WORKER_ID completed=$completed failed=$failed"
(( failed == 0 ))
