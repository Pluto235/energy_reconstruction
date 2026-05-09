# Repository Guidelines

## Project Structure & Module Organization
Active code lives under `src/`. Use `src/theta/` for the maintained WCDA theta-based pipeline and `src/common/` for shared utilities such as `EdgeConv`, detector configuration, and transforms. Cluster entrypoints live in `scripts/slurm/`; data helpers live in `scripts/data/`. Analysis notebooks are under `notebook/`, while `apply/` contains post-training plotting scripts plus dataset-application entrypoints such as `apply/simulation_all_bin.py` and `apply/apply_observation_energy.py`. Treat `archive/` as historical reference only. Large outputs belong in `runs/`, `logs/`, and `notebook/generated/` and are ignored by Git.

## Build, Test, and Development Commands
Install the Python stack with `pip install -r requirements.txt`.
Inspect training options with `python -m src.theta.main_theta --help`.
Run local training with `python -m src.theta.main_theta --root_path /home/server/mydisk/WCDA_simulation --run_dir runs/dev_run --tag dev_run`.
Evaluate an existing run with `python -m src.theta.evaluate_only --run_dir runs/<run_name>`.
Flatten the observation directory layout with `python scripts/data/flatten_observation_layout.py --input-root /mnt/mydisk/WCDA_observation --apply`.
Smoke-test the observation inference flow with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/server/anaconda3/envs/py310/bin/python apply/apply_observation_energy.py --input-root /mnt/mydisk/WCDA_observation/0101/Esg20220101_14.root --output-root /tmp/wcda_observation_eval_smoke --run-dir /home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666 --max-events-per-file 128 --step-size "5 MB" --device cpu --batch-size 64 --print-every 1`.
Launch full observation inference only after the input migration is complete with `sbatch scripts/slurm/apply_observation_energy.sbatch`.
Use Slurm for cluster validation: `sbatch scripts/slurm/theta_smoketest.sbatch`. Existing sbatch scripts assume the repo root is `/home/server/projects/energy_reconstruction`, `PYTHONPATH` includes that path, and the `py310` conda environment is available.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, `snake_case` for functions, variables, and file names, and `CamelCase` only for classes such as `ParticleDataset` and `ParticleNetRegressor`. Follow the existing argparse style: explicit long flags like `--dcedge_min` and `--theta_embed_dim`, with config persisted to `run_dir/config.json`. Keep modules focused; new training or evaluation entrypoints should stay under `src/theta/`, while reusable helpers belong in `src/common/`.

## Git Layout & Working Tree
This repository root is `/mnt/mydisk/server/projects/energy_reconstruction`. The primary branch is `main`, which currently tracks `origin/main`; the only configured remote is `origin` at `git@github.com:Pluto235/energy_reconstruction.git`. There are several working branches for experiments and refactors, including `centered-theta`, `detector-ord`, `nv-embed`, `position-embed`, `project_cleanup_theta`, `proton-only`, and `theta-compare`.

The working tree is currently dirty. Modified tracked files include code under `src/theta/`, `src/common/`, `apply/`, and `work/work_log.md`. Untracked files include `AGENTS.md`, new Slurm scripts under `scripts/slurm/`, and `src/common/hit_coordinate_transform.py`. Start every task with `git status --short --branch`, avoid overwriting unrelated local changes, and do not commit generated outputs from ignored directories.

## Testing Guidelines
There is no dedicated automated test suite in this repository. Validate changes with a targeted smoke test or evaluation run instead of full training whenever possible. For training-path changes, prefer `sbatch scripts/slurm/theta_smoketest.sbatch`. For evaluation or notebook changes, run the relevant `python -m src.theta...` command and confirm expected artifacts such as `metrics.json`, `preds.npz`, and plot outputs. For observation-data inference, validate with a single-file smoke test first and confirm that the output ROOT keeps `t_eventout_h`, writes `ml_logE_pred` and `ml_energy_pred`, removes `vx`, `vy`, `vq`, and `vt`, and only contains events passing `pincness < 1.1`, `fitstat == 0`, and `theta < 50 deg`. Do not commit generated `runs/`, logs, notebook output files, or temporary observation-eval outputs.

## Observation Apply Workflow
The finalized observation energy estimator is `/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666`, which uses both theta embedding and position/core embedding. Observation inputs are expected under `/mnt/mydisk/WCDA_observation` with direct `MMDD` subdirectories such as `/mnt/mydisk/WCDA_observation/0101`, and evaluated outputs are written to `/mnt/mydisk/WCDA_observation_eval` with the same `MMDD/...` layout.

`apply/apply_observation_energy.py` is the dedicated entrypoint for this workflow. It loads `t_eventout`, applies the observation-side cuts `pincness < 1.1`, `fitstat == 0`, and `theta < 50 deg`, runs inference for the surviving events only, writes `ml_logE_pred` and `ml_energy_pred` into the output `t_eventout`, preserves `t_eventout_h`, and drops the hit-level branches `vx`, `vy`, `vq`, and `vt` to reduce file size.

Do not start the full `/mnt/mydisk/WCDA_observation` batch job until the observation data migration into the flattened `MMDD/...` layout is complete. Before any large run, execute the single-file smoke command above with `--max-events-per-file` and check the generated `apply_summary.json` under the output root.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects, for example `Add apply plotting helpers` or `Remove recoxy smoke test script and ignore codex file`. Keep commits scoped to one change. PRs should include a concise summary, the affected paths, the exact validation command(s) you ran, and any output locations or screenshots needed to review notebook or plotting changes.
