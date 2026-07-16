#!/bin/bash
set -eo pipefail

cd /home/server/projects/energy_reconstruction

r1_job=$(sbatch --parsable --export=ALL,PSF_KIND=1R scripts/slurm/run_v6_64748_scheme_R_analytic_bon.sbatch)
r2_job=$(sbatch --parsable --export=ALL,PSF_KIND=2R scripts/slurm/run_v6_64748_scheme_R_analytic_bon.sbatch)
report_job=$(sbatch \
  --parsable \
  --dependency="afterok:${r1_job}:${r2_job}" \
  --export="ALL,ANALYTIC_R1_JOB_ID=${r1_job},ANALYTIC_R2_JOB_ID=${r2_job}" \
  scripts/slurm/finalize_v6_64748_scheme_R_analytic_bon.sbatch)

echo "Scheme R analytic B_on 1R job: ${r1_job}"
echo "Scheme R analytic B_on 2R job: ${r2_job}"
echo "Scheme R analytic B_on report job: ${report_job} (afterok:${r1_job}:${r2_job})"
