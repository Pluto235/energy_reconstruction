#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/server/projects/energy_reconstruction}
cd "${PROJECT_ROOT}"
export ANALYTIC_BON_BASELINE_SHA=${ANALYTIC_BON_BASELINE_SHA:-527af0380b1cf44de4a0c20d642174e210ca9485}
git merge-base --is-ancestor "${ANALYTIC_BON_BASELINE_SHA}" HEAD

BASE=v6_64748_nhit100_reselect44_split56_miss030_empirical_cdf_asimov_ropt
test ! -e "apply/output/stage_b_${BASE}_iter0"
test ! -e "apply/output/stage_b_${BASE}"
test ! -e "apply/output/stage_a_${BASE}_iter0_aperture_conditioned"
test ! -e "apply/output/stage_a_${BASE}_aperture_conditioned"
test ! -e "apply/output/${BASE}_poisson_unbinned"
test ! -e "apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_empirical_cdf_asimov_ropt_poisson_unbinned_report.html"
test ! -e "apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-empirical-cdf-asimov-ropt"

opt0_job=$(sbatch --parsable --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" \
  apply/scripts/slurm/optimize_v6_empirical_cdf_asimov_iter0.sbatch)
iter0_job=$(sbatch --parsable --dependency=afterok:"${opt0_job}" \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}" \
  apply/scripts/slurm/run_v6_empirical_cdf_asimov_iter0.sbatch)
final_job=$(sbatch --parsable --dependency=afterok:"${iter0_job}" \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",ANALYTIC_BON_BASELINE_SHA="${ANALYTIC_BON_BASELINE_SHA}",OPT0_JOB_ID="${opt0_job}",ITER0_JOB_ID="${iter0_job}" \
  apply/scripts/slurm/run_v6_empirical_cdf_asimov_final.sbatch)

printf 'opt0=%s\niter0=%s\nfinal=%s\n' "${opt0_job}" "${iter0_job}" "${final_job}"
