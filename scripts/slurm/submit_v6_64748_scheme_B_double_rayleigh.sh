#!/bin/bash
set -euo pipefail
cd /home/server/projects/energy_reconstruction

stage_b_job=$(sbatch --parsable scripts/slurm/build_psf_stage_b_v6_64748_reselect44_double_rayleigh.sbatch)
stage_a_job=$(sbatch --parsable --dependency=afterok:"${stage_b_job}" scripts/slurm/build_response_stage_a_v6_64748_reselect44_double_rayleigh.sbatch)
stage_dg_job=$(sbatch --parsable --dependency=afterok:"${stage_a_job}" scripts/slurm/run_v6_64748_reselect44_double_rayleigh_stage_d_to_g.sbatch)
pipeline_jobs="stage_b_double_rayleigh:${stage_b_job};stage_a_aperture:${stage_a_job};stage_d_to_g:${stage_dg_job}"
report_job=$(
  sbatch --parsable \
    --dependency=afterok:"${stage_dg_job}" \
    --export=ALL,PIPELINE_JOB_IDS="${pipeline_jobs}" \
    scripts/slurm/finalize_v6_64748_scheme_B_double_rayleigh.sbatch
)

echo "Stage B double-Rayleigh: ${stage_b_job}"
echo "Stage A aperture response: ${stage_a_job}"
echo "Stage D-G: ${stage_dg_job}"
echo "Validation/report: ${report_job}"
