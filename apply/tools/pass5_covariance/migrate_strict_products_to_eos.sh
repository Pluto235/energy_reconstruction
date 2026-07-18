#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
EOS_MGM=${EOS_MGM:-root://eos01.ihep.ac.cn}
EOS_ROOT=${EOS_ROOT:-/eos/user/l/liushijie/pass5_crab_v6_sorted_gti}
DELETE_SOURCE=${DELETE_SOURCE:-${1:-0}}
XRDFS=(env -u LD_LIBRARY_PATH /usr/bin/xrdfs)
XRDCP=(env -u LD_LIBRARY_PATH /usr/bin/xrdcp)

declare -a SOURCES=(
    /scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_hours
    /scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_map_chunks
)

manifest_for() {
    local root=$1
    local output=$2
    find "$root" -type f -printf '%P\t%s\n' | LC_ALL=C sort > "$output"
}

remote_manifest_for() {
    local root=$1
    local output=$2
    "${XRDFS[@]}" "$EOS_MGM" ls -l -R "$root" \
        | awk -v root="$root/" '$1 !~ /^d/ {path=$7; sub("^" root, "", path); print path "\t" $4}' \
        | LC_ALL=C sort > "$output"
}

mkdir -p "$RUN_DIR/migration_audit"
"${XRDFS[@]}" "$EOS_MGM" mkdir -p "$EOS_ROOT"

for i in "${!SOURCES[@]}"; do
    source_dir=${SOURCES[$i]}
    name=$(basename "$source_dir")
    target_dir="$EOS_ROOT/$name"
    source_manifest="$RUN_DIR/migration_audit/${name}_source.tsv"
    target_manifest="$RUN_DIR/migration_audit/${name}_target.tsv"

    if [[ ! -d "$source_dir" ]]; then
        echo "Missing source directory: $source_dir" >&2
        exit 2
    fi

    manifest_for "$source_dir" "$source_manifest"
    "${XRDCP[@]}" --recursive --parallel 8 --force --posc --retry 3 \
        --retry-policy force --cksum adler32 --rm-bad-cksum \
        "$source_dir" "$EOS_MGM//$EOS_ROOT/"
    remote_manifest_for "$target_dir" "$target_manifest"

    if ! cmp -s "$source_manifest" "$target_manifest"; then
        echo "Migration verification failed for $name" >&2
        diff -u "$source_manifest" "$target_manifest" | head -n 100 >&2 || true
        exit 3
    fi

    file_count=$(wc -l < "$source_manifest")
    byte_count=$(awk -F '\t' '{sum += $2} END {printf "%.0f", sum}' "$source_manifest")
    echo "MIGRATION_VERIFIED name=$name files=$file_count bytes=$byte_count target=$target_dir"

    if [[ "$DELETE_SOURCE" == 1 ]]; then
        while IFS=$'\t' read -r relative_path _; do
            rm -f -- "$source_dir/$relative_path"
        done < "$source_manifest"
        find "$source_dir" -depth -type d -empty -delete
        echo "MIGRATION_SOURCE_REMOVED name=$name source=$source_dir"
    fi
done

echo "STRICT_PRODUCTS_MIGRATION_COMPLETE eos_root=$EOS_ROOT delete_source=$DELETE_SOURCE"
