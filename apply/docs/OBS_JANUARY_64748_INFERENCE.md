# January observation inference with model 64748

ETO launcher: `scripts/slurm/apply_observation_energy_64748_january.sbatch`.

This launcher fixes the Python environment to `/mnt/newdisk/shijie/conda/envs/py310`,
input to `/mnt/newdisk/shijie/WCDA_obs`, day prefix to `01`, model to
`runs/theta_recoxy_position_embed_noemax_no_core_cut_64748`, and output to
`/mnt/newdisk/shijie/WCDA_obs_eval_64748`. It reuses `apply/apply_observation_energy.py`.

Before production, complete the environment agent's validation and a separate
single-file / 128-event inference smoke test. This launcher processes the entire
January sample; no inference job was submitted while preparing it.

Submit on ETO after those checks pass:

```bash
sbatch /mnt/newdisk/shijie/energy/scripts/slurm/apply_observation_energy_64748_january.sbatch
```

Resources: two GPUs, 16 CPU cores, 64 GB RAM, 72 hours; batch size 352 per GPU,
two reader threads per worker. Slurm 66051 tested 8,192 real events: 17.44 seconds,
7.50 GB peak allocated / 13.35 GB reserved per GPU on a 16 GiB V100.
The gated production job first validates two files on both GPUs before the full month.

Logs: `/mnt/newdisk/shijie/energy/logs/slurm/apply_obs_64748_january_<jobid>.out`
and `.err`. Standard output records the job, Python executable/version, Git commit,
checkpoint/config/inference-entrypoint SHA-256, full model config, and selected paths.
The summary is `WCDA_obs_eval_64748/apply_summary_01_<jobid>.json`.

The launcher keeps the input data and does not perform acquisition, cleanup or
time/J2000 recovery. A file lock prevents concurrent January runs in this output
directory. Existing January ROOT outputs cause a refusal; review their provenance
before arranging a resume. Automatic skipping is deliberately absent because the
generic entrypoint's skip check does not verify model provenance.

Preparation checks: Bash syntax and isolated fixture execution covering fixed
arguments, provenance logging, missing model, existing output and lock contention.
The fixture used a stub Python executable; it did not load the real model or read
observation events.

Dual-GPU production was queued as 66052 after successful day acceptance 66045;
independent final validation is 66053. Superseded pending jobs 66046/66047 were
cancelled intentionally. See OBS_JANUARY_64748_ACCEPTANCE.md for current evidence.
