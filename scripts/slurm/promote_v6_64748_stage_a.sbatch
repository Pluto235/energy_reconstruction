#!/bin/bash
#SBATCH -J promote_a_v6_64748
#SBATCH -p main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:15:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail

cd /home/server/projects/energy_reconstruction

SOURCE_DIR=${SOURCE_DIR:-apply/output/stage_a_v6_64748_split56_finetmp}
TARGET_DIR=${TARGET_DIR:-apply/output/stage_a_v6_64748_split56}
NPZ_NAME=${NPZ_NAME:-response_2d_v6_64748_split56.npz}
METADATA_NAME=${METADATA_NAME:-response_2d_v6_64748_split56_metadata.json}

if [[ ! -f "${SOURCE_DIR}/${NPZ_NAME}" ]]; then
  echo "Missing source NPZ: ${SOURCE_DIR}/${NPZ_NAME}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_DIR}/${METADATA_NAME}" ]]; then
  echo "Missing source metadata: ${SOURCE_DIR}/${METADATA_NAME}" >&2
  exit 3
fi

rsync -a --delete "${SOURCE_DIR}/" "${TARGET_DIR}/"

python - <<'PY'
import json
import os
from pathlib import Path

target_dir = Path(os.environ.get("TARGET_DIR", "apply/output/stage_a_v6_64748_split56"))
source_dir = os.environ.get("SOURCE_DIR", "apply/output/stage_a_v6_64748_split56_finetmp")
npz_name = os.environ.get("NPZ_NAME", "response_2d_v6_64748_split56.npz")
metadata_name = os.environ.get("METADATA_NAME", "response_2d_v6_64748_split56_metadata.json")
metadata_path = target_dir / metadata_name
meta = json.loads(metadata_path.read_text(encoding="utf-8"))
meta["npz_path"] = str((Path.cwd() / target_dir / npz_name).resolve())
meta["promoted_from_output_dir"] = source_dir
meta["staged_binned_root_used"] = meta.get("binned_root")
metadata_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

ls -lh "${TARGET_DIR}/${NPZ_NAME}" "${TARGET_DIR}/${METADATA_NAME}"
echo "Stage A v6 64748 split56 promotion finished"
