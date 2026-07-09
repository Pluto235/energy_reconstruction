#!/bin/bash
#SBATCH -J apply_obs_64748
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 96:00:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

if [[ -z "${DAY_PREFIX:-}" ]]; then
  echo "DAY_PREFIX is required, for example 03, 04, 05, or 06" >&2
  exit 64
fi

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

INPUT_ROOT=${INPUT_ROOT:-/mnt/mydisk/WCDA_observation}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/mydisk/WCDA_observation_eval_64748}
TIME_ROOT=${TIME_ROOT:-${OUTPUT_ROOT}/recovered_time}
RUN_DIR=${RUN_DIR:-/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_noemax_no_core_cut_64748}
SUMMARY_PATH=${SUMMARY_PATH:-${OUTPUT_ROOT}/apply_summary_${DAY_PREFIX}.json}
PROVENANCE_ROOT=${PROVENANCE_ROOT:-${OUTPUT_ROOT}/provenance/v6_64748_halfyear}
SYNC_FILTERED_INPUTS=${SYNC_FILTERED_INPUTS:-1}
CLEAN_INPUT_AFTER=${CLEAN_INPUT_AFTER:-1}
GPU_IDS=${GPU_IDS:-0,1,2,3}
BATCH_SIZE=${BATCH_SIZE:-256}
STEP_SIZE=${STEP_SIZE:-100 MB}
VALIDATE_AFTER=${VALIDATE_AFTER:-1}
READER_WORKERS=${READER_WORKERS:-0}

mkdir -p "${PROVENANCE_ROOT}"
PROVENANCE_LOG="${PROVENANCE_ROOT}/month_${DAY_PREFIX}_slurm_${SLURM_JOB_ID}.log"
exec > >(tee -a "${PROVENANCE_LOG}") 2>&1

echo "===== Job Info ====="
echo "JobID: ${SLURM_JOB_ID}"
echo "NodeList: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Day prefix: ${DAY_PREFIX}"
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Time root: ${TIME_ROOT}"
echo "Run dir: ${RUN_DIR}"
echo "GPU ids: ${GPU_IDS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Step size: ${STEP_SIZE}"
echo "Validate after: ${VALIDATE_AFTER}"
echo "Reader workers: ${READER_WORKERS}"
echo "Summary path: ${SUMMARY_PATH}"
echo "Provenance log: ${PROVENANCE_LOG}"
date -Is
df -h /mnt/mydisk /home

if [[ "${SYNC_FILTERED_INPUTS}" == "1" ]]; then
  echo "===== Sync IHEP filtered inputs ====="
  DAY_PREFIX="${DAY_PREFIX}" \
  LOCAL_ROOT="${INPUT_ROOT}" \
  /home/server/projects/energy_reconstruction/scripts/data/sync_ihep_filtered_days.sh
fi

echo "===== Input counts before inference ====="
find "${INPUT_ROOT}" -mindepth 2 -maxdepth 2 -type f -path "*/${DAY_PREFIX}??/Esg*.root" | wc -l
du -sh "${INPUT_ROOT}" 2>/dev/null || true

echo "===== Start 64748 observation inference ====="
python apply/apply_observation_energy.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --summary-path "${SUMMARY_PATH}" \
  --device cuda \
  --gpu-ids "${GPU_IDS}" \
  --batch-size "${BATCH_SIZE}" \
  --reader-workers "${READER_WORKERS}" \
  --step-size "${STEP_SIZE}" \
  --cut-pinc-max 1.1 \
  --cut-fitstat-equals 0 \
  --cut-theta-max-deg 50 \
  --cut-dcedge-min 20 \
  --skip-existing \
  --day-prefix "${DAY_PREFIX}" \
  --print-every 10

if [[ "${VALIDATE_AFTER}" == "1" ]]; then
  echo "===== Validate 64748 month ====="
  python apply/validate_observation_eval_month.py \
    --obs-root "${OUTPUT_ROOT}" \
    --time-root "${TIME_ROOT}" \
    --day-prefix "${DAY_PREFIX}" \
    --summary-json "${PROVENANCE_ROOT}/month_${DAY_PREFIX}_validation_${SLURM_JOB_ID}.json"
else
  echo "===== Skip validation because VALIDATE_AFTER=${VALIDATE_AFTER} ====="
fi

echo "===== Output counts after inference ====="
find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type f -path "*/${DAY_PREFIX}??/Esg*.root" | wc -l
find "${TIME_ROOT}" -mindepth 2 -maxdepth 2 -type f -path "*/${DAY_PREFIX}??/*.time.root" | wc -l
du -sh "${OUTPUT_ROOT}" "${TIME_ROOT}" 2>/dev/null || true
df -h /mnt/mydisk /home

if [[ "${CLEAN_INPUT_AFTER}" == "1" ]]; then
  echo "===== Clean synced filtered inputs for ${DAY_PREFIX} ====="
  while IFS= read -r day_dir; do
    base="${day_dir##*/}"
    case "${base}" in
      ${DAY_PREFIX}??) rm -rf -- "${day_dir}" ;;
      *) echo "Refusing unexpected input directory ${day_dir}" >&2; exit 65 ;;
    esac
  done < <(find "${INPUT_ROOT}" -maxdepth 1 -type d -name "${DAY_PREFIX}??" | sort)
fi

echo "===== Final storage ====="
du -sh "${INPUT_ROOT}" "${OUTPUT_ROOT}" "${TIME_ROOT}" 2>/dev/null || true
df -h /mnt/mydisk /home
date -Is
echo "Month ${DAY_PREFIX} 64748 inference finished"
