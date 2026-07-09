#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-liushijie@lxlogin.ihep.ac.cn}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratchfs/lhaaso/liushijie/WCDA_observation_20220101_20220630}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/lhaaso/liushijie/WCDA_reconstruction/manifests/obs_filtered_20220301_20220630}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/mydisk/WCDA_observation}"
DAY_PREFIX="${DAY_PREFIX:-}"
MAX_DAYS="${MAX_DAYS:-0}"
DRY_RUN="${DRY_RUN:-0}"

remote_script=$(
  cat <<'EOS'
set -euo pipefail
for manifest in "${MANIFEST_ROOT}"/2022*.txt; do
  [[ -f "$manifest" ]] || continue
  ymd="$(basename "$manifest" .txt)"
  mmdd="${ymd:4:4}"
  if [[ -n "${DAY_PREFIX}" && "${mmdd}" != ${DAY_PREFIX}* ]]; then
    continue
  fi
  day_dir="${REMOTE_ROOT}/${mmdd}"
  manifest_count="$(awk 'NF {count++} END {print count+0}' "$manifest")"
  root_count="$(find "$day_dir" -maxdepth 1 -type f -name 'Esg*.root' 2>/dev/null | wc -l)"
  partial_count="$(find "$day_dir" -maxdepth 1 -type f -name '*.partial' 2>/dev/null | wc -l)"
  printf '%s\t%s\t%s\t%s\n' "$mmdd" "$manifest_count" "$root_count" "$partial_count"
done
EOS
)

synced=0
mkdir -p "$LOCAL_ROOT"

while IFS=$'\t' read -r mmdd manifest_count root_count partial_count; do
  [[ -n "$mmdd" ]] || continue
  if [[ "$manifest_count" -eq 0 || "$root_count" -ne "$manifest_count" || "$partial_count" -ne 0 ]]; then
    printf 'WAIT %s manifest=%s roots=%s partials=%s\n' "$mmdd" "$manifest_count" "$root_count" "$partial_count"
    continue
  fi

  local_count="$(find "${LOCAL_ROOT}/${mmdd}" -maxdepth 1 -type f -name 'Esg*.root' 2>/dev/null | wc -l || true)"
  if [[ "$local_count" -eq "$root_count" ]]; then
    printf 'SKIP_LOCAL_COMPLETE %s files=%s\n' "$mmdd" "$local_count"
    continue
  fi

  printf 'SYNC %s remote_files=%s local_files=%s\n' "$mmdd" "$root_count" "$local_count"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "${LOCAL_ROOT}/${mmdd}"
    rsync -av --partial --append-verify \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "${REMOTE}:${REMOTE_ROOT}/${mmdd}/" \
      "${LOCAL_ROOT}/${mmdd}/"
  fi
  synced=$((synced + 1))
  if [[ "$MAX_DAYS" -gt 0 && "$synced" -ge "$MAX_DAYS" ]]; then
    break
  fi
done < <(
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "REMOTE_ROOT='${REMOTE_ROOT}' MANIFEST_ROOT='${MANIFEST_ROOT}' DAY_PREFIX='${DAY_PREFIX}' bash -s" <<< "$remote_script"
)

printf 'SYNCED_DAYS %s\n' "$synced"
