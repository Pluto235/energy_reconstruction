#!/bin/bash
#SBATCH -J validate_v6_64748_n100_hi1
#SBATCH -p main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 00:30:00

#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -eo pipefail

source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}

python apply/report/build_v6_64748_nhit100_highEplus1_report.py

python - <<'PY'
import json
from pathlib import Path

validation = Path("apply/report/assets/v6-64748-nhit100-highEplus1/report_validation.json")
payload = json.loads(validation.read_text(encoding="utf-8"))
missing = payload["html_image_validation"]["missing_image_refs"]
offenders = payload["metadata_contamination"]["offenders"]
print(json.dumps({
    "report_path": payload["report_path"],
    "image_ref_count": payload["html_image_validation"]["image_ref_count"],
    "missing_image_refs": missing,
    "metadata_offenders": len(offenders),
}, indent=2))
if missing:
    raise SystemExit("Missing report image references")
if offenders:
    raise SystemExit("Metadata contamination offenders found")
PY

echo "v6 64748 nhit100 highEplus1 report validation finished"
