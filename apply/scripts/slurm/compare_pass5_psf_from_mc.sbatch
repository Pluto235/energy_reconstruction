#!/usr/bin/env bash
#SBATCH --job-name=pass5_psf_mc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/server/projects/energy_reconstruction/logs/pass5_psf_mc_%j.out
#SBATCH --error=/home/server/projects/energy_reconstruction/logs/pass5_psf_mc_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/server/projects/energy_reconstruction}"
PYTHON_BIN="${PYTHON_BIN:-/home/server/anaconda3/envs/py310/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-/mnt/mydisk/WCDA_simulation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-apply/output/pass5_psf_mc_comparison}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

cd "${REPO_ROOT}"
mkdir -p logs "${OUTPUT_ROOT}"

cmd=(
  "${PYTHON_BIN}"
  apply/tools/compare_pass5_psf_from_mc.py
  --input-root "${INPUT_ROOT}"
  --output-dir "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
  --workers "${WORKERS}"
  --nhit-definition "${NHIT_DEFINITION:-pass5_nq05t30}"
  --hit-charge-branch "${HIT_CHARGE_BRANCH:-vq}"
  --pincness-calibration "${PINCNESS_CALIBRATION:-1.07}"
  --rmds-max "${RMDS_MAX:-20}"
  --theta-max-deg 50
  --source-dec-deg 22.01
  --latitude-deg 29.45
  --primary-variant crab_theta
)

if [[ -n "${MAX_FILES:-}" ]]; then
  cmd+=(--max-files "${MAX_FILES}")
fi
if [[ -n "${FILE_INDEX_MIN:-}" ]]; then
  cmd+=(--file-index-min "${FILE_INDEX_MIN}")
fi
if [[ -n "${FILE_INDEX_MAX:-}" ]]; then
  cmd+=(--file-index-max "${FILE_INDEX_MAX}")
fi
if [[ -n "${DCEDGE_MIN:-}" ]]; then
  cmd+=(--dcedge-min "${DCEDGE_MIN}")
fi
if [[ "${NO_PLOTS:-0}" == "1" ]]; then
  cmd+=(--no-plots)
fi
if [[ "${STRICT_FILES:-0}" == "1" ]]; then
  cmd+=(--strict-files)
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
