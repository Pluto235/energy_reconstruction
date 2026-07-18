#!/bin/bash
# Submit the unbinned pooled-Poisson experiment chain.
# Reuses the FROZEN pooling manifest (order/pooling unchanged) — no manifest job.
# grid (12 branches, D->E->F, unbinned + OFF-event dump on nominal)
#   -> bootstrap+time-split+covariance-aware Stage F on the nominal branch.
# The finalize/report step is intentionally NOT chained here yet (adapt the pooled
# finalize sbatch: swap experiment id + report/asset paths, reuse the report builders).
set -euo pipefail

cd /home/server/projects/energy_reconstruction
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:-527af03}
git merge-base --is-ancestor "${ANALYTIC_BON_BASELINE_SHA}" HEAD

MANIFEST=apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
test -s "${MANIFEST}"  # reuse frozen manifest; no rebuild

grid_job=$(sbatch --parsable \
  --export=ALL,ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}" \
  scripts/slurm/run_v6_64748_scheme_R_poisson_unbinned_grid_branch.sbatch)
bootstrap_job=$(sbatch --parsable --dependency=afterok:"${grid_job}_4" \
  --export=ALL,ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}" \
  scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_unbinned_background.sbatch)

printf 'grid=%s\nbootstrap=%s\n' "${grid_job}" "${bootstrap_job}"
