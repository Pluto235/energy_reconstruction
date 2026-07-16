#!/bin/bash
set -euo pipefail

cd /home/server/projects/energy_reconstruction
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:-527af03}
git merge-base --is-ancestor "${ANALYTIC_BON_BASELINE_SHA}" HEAD

manifest_job=$(sbatch --parsable \
  --export=ALL,ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}" \
  scripts/slurm/build_v6_64748_poisson_pooling_manifest.sbatch)
grid_job=$(sbatch --parsable --dependency=afterok:"${manifest_job}" \
  --export=ALL,ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}" \
  scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch)
bootstrap_job=$(sbatch --parsable --dependency=afterok:"${grid_job}_4" \
  --export=ALL,ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}" \
  scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch)
finalizer_job=$(sbatch --parsable --dependency=afterok:"${grid_job}:${bootstrap_job}" \
  --export=ALL,ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}",MANIFEST_JOB_ID="${manifest_job}",GRID_JOB_ID="${grid_job}",BOOTSTRAP_JOB_ID="${bootstrap_job}" \
  scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch)

printf 'manifest=%s\ngrid=%s\nbootstrap=%s\nfinalizer=%s\n' \
  "${manifest_job}" "${grid_job}" "${bootstrap_job}" "${finalizer_job}"
