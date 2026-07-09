#!/usr/bin/env bash
#SBATCH -J apply_obs_64748_pack
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

if [[ -z "${DAYS:-}" ]]; then
  echo "DAYS is required, for example: DAYS='0607 0608'" >&2
  exit 64
fi

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
set -u

REPO_ROOT=${REPO_ROOT:-/home/server/projects/energy_reconstruction}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/mydisk/WCDA_observation_eval_64748}
TIME_ROOT=${TIME_ROOT:-${OUTPUT_ROOT}/recovered_time}
PROVENANCE_ROOT=${PROVENANCE_ROOT:-${OUTPUT_ROOT}/provenance/v6_64748_halfyear}
BATCH_ROOT_BASE=${BATCH_ROOT_BASE:-${PROVENANCE_ROOT}/batch_roots}
RUN_DIR=${RUN_DIR:-/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_noemax_no_core_cut_64748}
BATCH_SIZE=${BATCH_SIZE:-128}
STEP_SIZE=${STEP_SIZE:-50 MB}
READER_WORKERS=${READER_WORKERS:-0}
MAX_PARALLEL=${MAX_PARALLEL:-2}

mkdir -p "${PROVENANCE_ROOT}" "${BATCH_ROOT_BASE}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

normalize_days() {
  tr ',' ' ' <<<"${DAYS}"
}

month_label() {
  case "$1" in
    01) echo jan ;;
    02) echo feb ;;
    03) echo mar ;;
    04) echo apr ;;
    05) echo may ;;
    06) echo jun ;;
    *) echo "m$1" ;;
  esac
}

run_day() {
  local day="$1"
  if [[ ! "${day}" =~ ^[0-9]{4}$ ]]; then
    echo "Invalid day ${day}: expected MMDD" >&2
    return 65
  fi

  local month="${day:0:2}"
  local label
  label="$(month_label "${month}")"
  local batch_root="${BATCH_ROOT_BASE}/${label}_${day}"
  local day_log="${PROVENANCE_ROOT}/daypack_${day}_${SLURM_JOB_ID}.log"
  local summary_path="${PROVENANCE_ROOT}/apply_summary_${day}_daypack_${SLURM_JOB_ID}.json"
  local validation_path="${PROVENANCE_ROOT}/day_${day}_validation_daypack_${SLURM_JOB_ID}.json"

  {
    echo "===== Daypack ${day} start ====="
    date -Is
    echo "JobID: ${SLURM_JOB_ID}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
    echo "Batch root: ${batch_root}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "Step size: ${STEP_SIZE}"
    df -h /mnt/mydisk /home

    mkdir -p "${batch_root}"
    DAY_PREFIX="${day}" LOCAL_ROOT="${batch_root}" \
      "${REPO_ROOT}/scripts/data/sync_ihep_filtered_days.sh"

    echo "===== Input count ${day} ====="
    find "${batch_root}/${day}" -maxdepth 1 -type f -name 'Esg*.root' | wc -l
    du -sh "${batch_root}" 2>/dev/null || true

    python apply/apply_observation_energy.py \
      --input-root "${batch_root}" \
      --output-root "${OUTPUT_ROOT}" \
      --run-dir "${RUN_DIR}" \
      --summary-path "${summary_path}" \
      --device cuda \
      --gpu-ids 0 \
      --batch-size "${BATCH_SIZE}" \
      --reader-workers "${READER_WORKERS}" \
      --step-size "${STEP_SIZE}" \
      --cut-pinc-max 1.1 \
      --cut-fitstat-equals 0 \
      --cut-theta-max-deg 50 \
      --cut-dcedge-min 20 \
      --skip-existing \
      --day-prefix "${month}" \
      --print-every 10

    python apply/validate_observation_eval_month.py \
      --obs-root "${OUTPUT_ROOT}" \
      --time-root "${TIME_ROOT}" \
      --day-prefix "${day}" \
      --summary-json "${validation_path}" \
      --print-missing 1000

    rm -rf -- "${batch_root:?}/${day}"
    rmdir --ignore-fail-on-non-empty "${batch_root}" || true

    echo "===== Daypack ${day} done ====="
    date -Is
    df -h /mnt/mydisk /home
  } > >(tee -a "${day_log}") 2>&1
}

echo "===== 64748 daypack job ====="
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "DAYS: ${DAYS}"
echo "MAX_PARALLEL: ${MAX_PARALLEL}"
date -Is
df -h /mnt/mydisk /home

pids=()
days=()
for day in $(normalize_days); do
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    if ! wait "${pids[0]}"; then
      echo "Day ${days[0]} failed" >&2
      exit 1
    fi
    pids=("${pids[@]:1}")
    days=("${days[@]:1}")
  done
  run_day "${day}" &
  pids+=("$!")
  days+=("${day}")
done

for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "Day ${days[$idx]} failed" >&2
    exit 1
  fi
done

echo "===== 64748 daypack complete ====="
date -Is
df -h /mnt/mydisk /home
