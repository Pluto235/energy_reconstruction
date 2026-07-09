#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage: submit_observation_64748_day.sh MMDD [MMDD ...]

Submit one Slurm chain per day:
  1. sync filtered IHEP input into a day batch root
  2. run 64748 observation inference on one GPU with READER_WORKERS=0
  3. validate that exact day and clean the day batch root after validation passes

Environment overrides:
  PROVENANCE_ROOT, OUTPUT_ROOT, TIME_ROOT, BATCH_ROOT_BASE, REPO_ROOT, LOG_DIR
  PARTITION, READER_WORKERS, BATCH_SIZE, STEP_SIZE, GPU_IDS
  PREP_TIME, APPLY_TIME, VALIDATE_TIME, APPLY_CPUS, APPLY_MEM
  ASSUME_PREPARED=1 to skip the sync step and submit apply/validate for existing batch roots
USAGE
  exit 64
fi

PROVENANCE_ROOT=${PROVENANCE_ROOT:-/mnt/mydisk/WCDA_observation_eval_64748/provenance/v6_64748_halfyear}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/mydisk/WCDA_observation_eval_64748}
TIME_ROOT=${TIME_ROOT:-${OUTPUT_ROOT}/recovered_time}
BATCH_ROOT_BASE=${BATCH_ROOT_BASE:-${PROVENANCE_ROOT}/batch_roots}
REPO_ROOT=${REPO_ROOT:-/home/server/projects/energy_reconstruction}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/slurm}
PARTITION=${PARTITION:-main}
SUBMIT_SCRIPT_DIR=${SUBMIT_SCRIPT_DIR:-${PROVENANCE_ROOT}/slurm_submit_scripts}

READER_WORKERS=${READER_WORKERS:-0}
BATCH_SIZE=${BATCH_SIZE:-192}
STEP_SIZE=${STEP_SIZE:-50 MB}
GPU_IDS=${GPU_IDS:-0}

PREP_TIME=${PREP_TIME:-02:00:00}
APPLY_TIME=${APPLY_TIME:-12:00:00}
VALIDATE_TIME=${VALIDATE_TIME:-02:00:00}
APPLY_CPUS=${APPLY_CPUS:-8}
APPLY_MEM=${APPLY_MEM:-16G}
ASSUME_PREPARED=${ASSUME_PREPARED:-0}

mkdir -p "${LOG_DIR}" "${BATCH_ROOT_BASE}" "${PROVENANCE_ROOT}" "${SUBMIT_SCRIPT_DIR}"

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

validate_day_arg() {
  local day="$1"
  if [[ ! "${day}" =~ ^[0-9]{4}$ ]]; then
    echo "Invalid day ${day}: expected MMDD" >&2
    exit 65
  fi
}

submit_prep() {
  local day="$1"
  local batch_root="$2"
  local submit_script="${SUBMIT_SCRIPT_DIR}/prep_64748_${day}_$$_$(date +%Y%m%dT%H%M%S).sbatch"
  cat > "${submit_script}" <<EOF
#!/usr/bin/env bash
#SBATCH -J prep_64748_${day}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t ${PREP_TIME}
#SBATCH -o ${LOG_DIR}/%x_%j.out
#SBATCH -e ${LOG_DIR}/%x_%j.err

set -euo pipefail
mkdir -p '${batch_root}'
date -Is
df -h /mnt/mydisk /home
DAY_PREFIX='${day}' LOCAL_ROOT='${batch_root}' '${REPO_ROOT}/scripts/data/sync_ihep_filtered_days.sh'
echo 'Synced day ${day} into ${batch_root}'
find '${batch_root}/${day}' -maxdepth 1 -type f -name 'Esg*.root' | wc -l
du -sh '${batch_root}' 2>/dev/null || true
df -h /mnt/mydisk /home
EOF
  sbatch --parsable "${submit_script}"
}

submit_apply() {
  local day="$1"
  local batch_root="$2"
  local prep_job="$3"
  local month_prefix="${day:0:2}"
  local dependency_args=()
  if [[ -n "${prep_job}" ]]; then
    dependency_args=(--dependency="afterok:${prep_job}")
  fi
  sbatch --parsable \
    "${dependency_args[@]}" \
    -J "apply_obs_64748_${day}" \
    -p "${PARTITION}" \
    --gres=gpu:1 \
    --cpus-per-task="${APPLY_CPUS}" \
    --mem="${APPLY_MEM}" \
    -t "${APPLY_TIME}" \
    --export=ALL,DAY_PREFIX="${month_prefix}",INPUT_ROOT="${batch_root}",OUTPUT_ROOT="${OUTPUT_ROOT}",TIME_ROOT="${TIME_ROOT}",SYNC_FILTERED_INPUTS=0,CLEAN_INPUT_AFTER=0,VALIDATE_AFTER=0,READER_WORKERS="${READER_WORKERS}",GPU_IDS="${GPU_IDS}",BATCH_SIZE="${BATCH_SIZE}",STEP_SIZE="${STEP_SIZE}",SUMMARY_PATH="${PROVENANCE_ROOT}/apply_summary_${day}.json",PROVENANCE_ROOT="${PROVENANCE_ROOT}" \
    "${REPO_ROOT}/scripts/slurm/apply_observation_energy_64748_month.sbatch"
}

submit_validate() {
  local day="$1"
  local batch_root="$2"
  local apply_job="$3"
  local submit_script="${SUBMIT_SCRIPT_DIR}/validate_clean_${day}_$$_$(date +%Y%m%dT%H%M%S).sbatch"
  cat > "${submit_script}" <<EOF
#!/usr/bin/env bash
#SBATCH -J validate_clean_${day}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t ${VALIDATE_TIME}
#SBATCH -o ${LOG_DIR}/%x_%j.out
#SBATCH -e ${LOG_DIR}/%x_%j.err
#SBATCH --dependency=afterok:${apply_job}

set -eo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd '${REPO_ROOT}'
date -Is
python apply/validate_observation_eval_month.py \
  --obs-root "${OUTPUT_ROOT}" \
  --time-root "${TIME_ROOT}" \
  --day-prefix "${day}" \
  --summary-json "${PROVENANCE_ROOT}/day_${day}_validation_\${SLURM_JOB_ID}.json" \
  --print-missing 1000
rm -rf -- '${batch_root}/${day}'
rmdir --ignore-fail-on-non-empty '${batch_root}' || true
echo 'Validated and cleaned day ${day} batch root ${batch_root}'
df -h /mnt/mydisk /home
EOF
  sbatch --parsable "${submit_script}"
}

printf 'day\tprep_job\tapply_job\tvalidate_job\tbatch_root\n'
for day in "$@"; do
  validate_day_arg "${day}"
  month="${day:0:2}"
  label="$(month_label "${month}")"
  batch_root="${BATCH_ROOT_BASE}/${label}_${day}"
  if [[ "${ASSUME_PREPARED}" == "1" ]]; then
    prep_job=""
    prep_label="prepared"
  else
    prep_job="$(submit_prep "${day}" "${batch_root}")"
    prep_label="${prep_job}"
  fi
  apply_job="$(submit_apply "${day}" "${batch_root}" "${prep_job}")"
  validate_job="$(submit_validate "${day}" "${batch_root}" "${apply_job}")"
  printf '%s\t%s\t%s\t%s\t%s\n' "${day}" "${prep_label}" "${apply_job}" "${validate_job}" "${batch_root}"
done
