#!/bin/bash
set -euo pipefail

cd /home/server/projects/energy_reconstruction

stage_b_job=$(sbatch --parsable scripts/slurm/build_psf_stage_b_v6_64748_nhit100_reselect44_split56_miss030.sbatch)
aperture_job=$(sbatch --parsable --dependency=afterok:"${stage_b_job}" scripts/slurm/build_response_stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture.sbatch)
pipeline_jobs="stage_b:${stage_b_job};stage_a_aperture:${aperture_job}"
stage_dg_job=$(
  sbatch --parsable \
    --dependency=afterok:"${aperture_job}" \
    --export=ALL,PIPELINE_JOB_IDS="${pipeline_jobs}" \
    scripts/slurm/run_v6_64748_nhit100_reselect44_split56_miss030_stage_d_to_g.sbatch
)

echo "Stage B: ${stage_b_job}"
echo "Stage A aperture: ${aperture_job}"
echo "Stage D-G/report: ${stage_dg_job}"
