#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

submit_one() {
  local strategy="$1"
  echo "Submitting v5 PredE ablation strategy: ${strategy}"

  local prep_job
  prep_job=$(sbatch --parsable --export=ALL,STRATEGY="${strategy}" scripts/slurm/prepare_v5_predbin_ablation.sbatch)
  echo "  prepare: ${prep_job}"

  local stage_a_job
  stage_a_job=$(sbatch --parsable --dependency=afterok:"${prep_job}" --export=ALL,STRATEGY="${strategy}" scripts/slurm/build_response_stage_a_v5_predbin.sbatch)
  echo "  stage_a: ${stage_a_job}"

  local stage_b_job
  stage_b_job=$(sbatch --parsable --dependency=afterok:"${stage_a_job}" --export=ALL,STRATEGY="${strategy}" scripts/slurm/build_psf_stage_b_v5_predbin.sbatch)
  echo "  stage_b: ${stage_b_job}"

  local stage_a_ap_job
  stage_a_ap_job=$(sbatch --parsable --dependency=afterok:"${stage_b_job}" --export=ALL,STRATEGY="${strategy}" scripts/slurm/build_response_stage_a_v5_predbin_aperture_conditioned.sbatch)
  echo "  stage_a_aperture: ${stage_a_ap_job}"

  local stage_cg_job
  stage_cg_job=$(sbatch --parsable --dependency=afterok:"${stage_a_ap_job}" --export=ALL,STRATEGY="${strategy}" scripts/slurm/run_v5_predbin_stage_c_to_g.sbatch)
  echo "  stage_c_to_g: ${stage_cg_job}"
}

submit_one gap025
submit_one gap1
