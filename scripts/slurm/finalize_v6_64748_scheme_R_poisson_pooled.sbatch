#!/bin/bash
#SBATCH -J v6_64748_poisson_finalize
#SBATCH -p main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.out
#SBATCH -e /home/server/projects/energy_reconstruction/logs/slurm/%x_%j.err

set -euo pipefail
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/server/projects/energy_reconstruction
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH:-}
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:?ANALYTIC_BON_BASELINE_SHA is required}
MANIFEST_JOB_ID=${MANIFEST_JOB_ID:?MANIFEST_JOB_ID is required}
GRID_JOB_ID=${GRID_JOB_ID:?GRID_JOB_ID is required}
BOOTSTRAP_JOB_ID=${BOOTSTRAP_JOB_ID:?BOOTSTRAP_JOB_ID is required}

EXPERIMENT=v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled
OUTPUT_ROOT=apply/output/${EXPERIMENT}
REGISTRY=${OUTPUT_ROOT}/branches.json
MANIFEST=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
CONVERGENCE=scheme_R_double_rayleigh_poisson_grid_convergence.json
VALIDATION=scheme_R_double_rayleigh_poisson_pooled_validation.json
COV_NPZ=scheme_R_double_rayleigh_poisson_background_covariance.npz
COV_META=scheme_R_double_rayleigh_poisson_background_covariance.json
REPORT=apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_poisson_pooled_report.html
ASSETS=apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-pooled

python - "${REGISTRY}" "${MANIFEST}" "${MANIFEST_JOB_ID}" "${GRID_JOB_ID}" "${BOOTSTRAP_JOB_ID}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

registry_path, manifest_path, manifest_job, grid_job, bootstrap_job = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="ascii"))
prov = manifest["provenance"]
response = Path("apply/output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh_annnorm/response_v6_64748_nhit100_reselect44_split56_miss030_fixed712979_rayleigh.npz")
response_sha = hashlib.sha256(response.read_bytes()).hexdigest()
branch_ids = (
    "h005_x0_y0", "h005_xh_y0", "h005_x0_yh", "h005_xh_yh",
    "h010_x0_y0", "h010_xh_y0", "h010_x0_yh", "h010_xh_yh",
    "h020_x0_y0", "h020_xh_y0", "h020_x0_yh", "h020_xh_yh",
)
branches = []
for index, branch in enumerate(branch_ids):
    root = Path("apply/output") / manifest["experiment_id"] / "branches" / branch
    d = root / "stage_d/runs/stage_d"
    e = root / "stage_e/runs/stage_e"
    f = root / "stage_f/runs/stage_f"
    branches.append({
        "branch_id": branch,
        "stage_d_npz": str(d / f"background_{branch}.npz"),
        "stage_d_metadata": str(d / f"background_{branch}_metadata.json"),
        "stage_e_npz": str(e / f"signal_{branch}.npz"),
        "stage_f_npz": str(f / f"fit_{branch}.npz"),
        "stage_f_metadata": str(f / f"fit_{branch}_metadata.json"),
        "provenance": {
            "pooling_manifest_sha256": manifest["manifest_sha256"],
            "selector_sha256": prov["selector"]["sha256"],
            "psf_sha256": prov["psf_npz"]["sha256"],
            "stage_c_sha256": prov["stage_c_metadata"]["sha256"],
            "response_sha256": response_sha,
            "analytic_bon_baseline_sha": manifest["analytic_bon_baseline_sha"],
            "code_sha": manifest["implementation_commit_sha"],
        },
    })
payload = {
    "schema_version": 1,
    "experiment": manifest["experiment_id"],
    "branches": branches,
    "diagnostics": {
        "pooling_manifest": manifest_path,
        "bootstrap_npz": "scheme_R_double_rayleigh_poisson_background_covariance.npz",
        "legacy_sed_npz": "apply/output/stage_g_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979/runs/v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_stage_g/sed_points_v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979.npz",
        "nominal_sed_npz": str(Path("apply/output") / manifest["experiment_id"] / "branches/h010_x0_y0/stage_g/runs/stage_g/sed_points_h010_x0_y0.npz"),
    },
    "slurm_jobs": {
        "manifest": {"job_id": manifest_job, "state": "COMPLETED"},
        "bootstrap": {"job_id": bootstrap_job, "state": "COMPLETED"},
        "grid": {"job_id": grid_job, "state": "COMPLETED", "tasks": {
            branch: {"job_id": f"{grid_job}_{index}", "state": "COMPLETED"}
            for index, branch in enumerate(branch_ids)
        }},
    },
}
path = Path(registry_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

mkdir -p "${ASSETS}"
set +e
python apply/report/build_v6_poisson_grid_convergence.py \
  --branches-json "${REGISTRY}" --output-json "${CONVERGENCE}" --plot-dir "${ASSETS}"
convergence_status=$?
set -e

export V6_REPORT_POISSON_MANIFEST="${MANIFEST}"
export V6_REPORT_GRID_CONVERGENCE="${CONVERGENCE}"
export V6_REPORT_BACKGROUND_COVARIANCE="${COV_NPZ}"
export V6_REPORT_PATH="${REPORT}"
export V6_REPORT_ASSET_DIR="${ASSETS}"
export V6_REPORT_EXPERIMENT_ID="${EXPERIMENT}"
export V6_REPORT_INPUT_COMMIT="$(git rev-parse HEAD)"
export V6_REPORT_STAGE_D_RUN_DIR="${OUTPUT_ROOT}/branches/h010_x0_y0/stage_d/runs/stage_d"
export V6_REPORT_STAGE_E_RUN_DIR="${OUTPUT_ROOT}/branches/h010_x0_y0/stage_e/runs/stage_e"
export V6_REPORT_STAGE_F_RUN_DIR="${OUTPUT_ROOT}/branches/h010_x0_y0/stage_f/runs/stage_f"
export V6_REPORT_STAGE_G_RUN_DIR="${OUTPUT_ROOT}/branches/h010_x0_y0/stage_g/runs/stage_g"
export PIPELINE_JOB_IDS="manifest:${MANIFEST_JOB_ID};grid:${GRID_JOB_ID};bootstrap:${BOOTSTRAP_JOB_ID};finalizer:${SLURM_JOB_ID}"

python apply/report/build_v6_64748_nhit100_highEplus1_report.py
python apply/report/prepare_v6_scheme_r_double_rayleigh_poisson_pooled_report.py

set +e
python apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py \
  --branches-json "${REGISTRY}" --output-json "${VALIDATION}" --convergence-json "${CONVERGENCE}" \
  --bootstrap-npz "${COV_NPZ}" --bootstrap-metadata "${COV_META}" --report "${REPORT}" \
  --analytic-baseline-sha "${ANALYTIC_BON_BASELINE_SHA}" --require-report
validation_status=$?
set -e

if [[ ${convergence_status} -ne 0 || ${validation_status} -ne 0 ]]; then
  echo "Poisson grid validation failed; report and evidence were preserved" >&2
  exit 1
fi
echo "Final Poisson pooled report: ${REPORT}"
