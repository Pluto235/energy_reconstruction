#!/bin/bash
set -eo pipefail

cd /home/server/projects/energy_reconstruction

fit_job=$(sbatch --parsable scripts/slurm/run_v6_64748_scheme_R_double_rayleigh.sbatch)
report_job=$(sbatch \
  --parsable \
  --dependency="afterok:${fit_job}" \
  --export="ALL,SCHEME_R_2R_JOB_ID=${fit_job}" \
  scripts/slurm/finalize_v6_64748_scheme_R_double_rayleigh.sbatch)

echo "Scheme R double-Rayleigh Stage F/G job: ${fit_job}"
echo "Scheme R double-Rayleigh report job: ${report_job} (afterok:${fit_job})"
