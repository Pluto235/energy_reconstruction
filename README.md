# energy_reconstruction

This repository is now maintained around the theta-based WCDA energy reconstruction pipeline only.

## Maintained structure

- `src/theta`: theta training, evaluation, and dataset/model code
- `src/common`: shared EdgeConv implementation and plotting/utilities
- `scripts/slurm`: maintained Slurm entrypoints
- `scripts/data`: helper scripts for dataset preprocessing and splitting
- `runs`: experiment outputs, preserved in place
- `notebook`: exploratory and analysis notebooks, kept and adapted to the current package layout
- `archive`: historical code, old sbatch files, and root-level artifacts that are no longer part of the active pipeline

## Active entrypoints

- Train: `python -m src.theta.main_theta`
- Evaluate an existing run: `python -m src.theta.evaluate_only --run_dir <RUN_DIR>`

## Recommended commands

Inspect training arguments:

```bash
cd /home/server/projects/energy_reconstruction
python -m src.theta.main_theta --help
```

Run a theta training job directly:

```bash
cd /home/server/projects/energy_reconstruction
python -m src.theta.main_theta \
  --root_path /home/server/mydisk/WCDA_simulation \
  --n_files 10000 \
  --epochs 500 \
  --batch_size 512 \
  --max_points 500 \
  --num_workers 4 \
  --io_workers 32 \
  --norm_mode per_event \
  --sample_mode weighted_q \
  --Emin 100 \
  --dcedge_min 20 \
  --theta_max_deg 30 \
  --dangle_max_deg 3 \
  --pinc_max 1.1 \
  --theta_embed_dim 16 \
  --theta_embed_dropout 0.0 \
  --eval_space log \
  --save_arrays \
  --run_dir /home/server/projects/energy_reconstruction/runs/my_theta_run \
  --tag my_theta_run
```

Submit the maintained smoke test:

```bash
cd /home/server/projects/energy_reconstruction
sbatch scripts/slurm/theta_smoketest.sbatch
```

## Slurm scripts

- Maintained sbatch files live under `scripts/slurm/`
- Data helper scripts live under `scripts/data/`
- The smoke test script is `scripts/slurm/theta_smoketest.sbatch`

## Notes

- Historical non-theta code has been moved to `archive/src_non_theta/`.
- Historical theta worktree code has been moved to `archive/theta-old/`.
- Historical non-theta sbatch scripts have been moved to `archive/old_sbatch/`.
- Existing experiment outputs under `runs/` are preserved and were not modified.
