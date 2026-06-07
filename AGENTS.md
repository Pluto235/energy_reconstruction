# Repository Guidelines

## Project Structure & Module Organization
Active code lives under `src/`. Use `src/theta/` for the maintained WCDA theta-based pipeline and `src/common/` for shared utilities such as `EdgeConv`, detector configuration, and transforms. Cluster entrypoints live in `scripts/slurm/`; data helpers live in `scripts/data/`. Analysis notebooks are under `notebook/`, while `apply/` is the post-training application and SED workspace: `apply/simulation_all_bin.py` runs MC inference and Nhit × predicted-energy binning, `apply/apply_observation_energy.py` runs observation inference, `apply/config/` stores SED selection configs such as `cell_selection_v1.csv`, and `apply/report/` stores SED markdown/HTML reports. Treat `archive/` as historical reference only. Large outputs belong in `runs/`, `logs/`, `notebook/generated/`, `apply/plot/`, `apply/summary_selectedcuts/`, and `apply/output/` and are ignored by Git.

## Build, Test, and Development Commands
Install the Python stack with `pip install -r requirements.txt`.
Inspect training options with `python -m src.theta.main_theta --help`.
Run local training with `python -m src.theta.main_theta --root_path /home/server/mydisk/WCDA_simulation --run_dir runs/dev_run --tag dev_run`.
Evaluate an existing run with `python -m src.theta.evaluate_only --run_dir runs/<run_name>`.
Flatten the observation directory layout with `python scripts/data/flatten_observation_layout.py --input-root /mnt/mydisk/WCDA_observation --apply`.
Smoke-test the observation inference flow with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/server/anaconda3/envs/py310/bin/python apply/apply_observation_energy.py --input-root /mnt/mydisk/WCDA_observation/0101/Esg20220101_14.root --output-root /tmp/wcda_observation_eval_smoke --run-dir /home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666 --max-events-per-file 128 --step-size "5 MB" --device cpu --batch-size 64 --print-every 1`.
Launch full observation inference only after the input migration is complete with `sbatch scripts/slurm/apply_observation_energy.sbatch`.
Use Slurm for cluster validation: `sbatch scripts/slurm/theta_smoketest.sbatch`. Existing sbatch scripts assume the repo root is `/home/server/projects/energy_reconstruction`, `PYTHONPATH` includes that path, and the `py310` conda environment is available.
Publish the Crab SED roadmap HTML by copying `apply/report/roadmap.html` to `/mnt/mydisk/server/projects/any-reports/crab-sed-roadmap/index.html`, and keep the legacy `/mnt/mydisk/server/projects/any-reports/crab-sed-roadmap.html` synchronized if that old URL should remain valid. Whenever publishing a new or renamed report page, update `/mnt/mydisk/server/projects/any-reports/index.html` so the report appears on the any-reports homepage and does not keep stale titles, dates, links, or descriptions. Then commit and push the `any-reports` repository. The public page is `https://pluto235.github.io/any-reports/crab-sed-roadmap/`.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, `snake_case` for functions, variables, and file names, and `CamelCase` only for classes such as `ParticleDataset` and `ParticleNetRegressor`. Follow the existing argparse style: explicit long flags like `--dcedge_min` and `--theta_embed_dim`, with config persisted to `run_dir/config.json`. Keep modules focused; new training or evaluation entrypoints should stay under `src/theta/`, while reusable helpers belong in `src/common/`.

## Git Layout & Working Tree
This repository root is `/mnt/mydisk/server/projects/energy_reconstruction`. The primary branch is `main`, which currently tracks `origin/main`; the only configured remote is `origin` at `git@github.com:Pluto235/energy_reconstruction.git`. There are several working branches for experiments and refactors, including `centered-theta`, `detector-ord`, `nv-embed`, `position-embed`, `project_cleanup_theta`, `proton-only`, and `theta-compare`.

The working tree is currently dirty. Modified tracked files include code under `src/theta/`, `src/common/`, `apply/`, and `work/work_log.md`. Untracked files include `AGENTS.md`, new Slurm scripts under `scripts/slurm/`, and `src/common/hit_coordinate_transform.py`. Start every task with `git status --short --branch`, avoid overwriting unrelated local changes, and do not commit generated outputs from ignored directories.

## Testing Guidelines
There is no dedicated automated test suite in this repository. Validate changes with a targeted smoke test or evaluation run instead of full training whenever possible. For training-path changes, prefer `sbatch scripts/slurm/theta_smoketest.sbatch`. For evaluation or notebook changes, run the relevant `python -m src.theta...` command and confirm expected artifacts such as `metrics.json`, `preds.npz`, and plot outputs. For observation-data inference, validate with a single-file smoke test first and confirm that the output ROOT keeps `t_eventout_h`, writes `ml_logE_pred` and `ml_energy_pred`, removes `vx`, `vy`, `vq`, and `vt`, and only contains events passing `pincness < 1.1`, `fitstat == 0`, `theta < 50 deg`, and `dcedge > 20 m`. Do not commit generated `runs/`, logs, notebook output files, or temporary observation-eval outputs.

## Observation Apply Workflow
The finalized observation energy estimator is `/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666`, which uses both theta embedding and position/core embedding. Observation inputs are expected under `/mnt/mydisk/WCDA_observation` with direct `MMDD` subdirectories such as `/mnt/mydisk/WCDA_observation/0101`, and evaluated outputs are written to `/mnt/mydisk/WCDA_observation_eval` with the same `MMDD/...` layout.

`apply/apply_observation_energy.py` is the dedicated entrypoint for this workflow. It loads `t_eventout`, applies the observation-side cuts `pincness < 1.1`, `fitstat == 0`, `theta < 50 deg`, and `dcedge > 20 m`, runs inference for the surviving events only, writes `ml_logE_pred` and `ml_energy_pred` into the output `t_eventout`, preserves `t_eventout_h`, and drops the hit-level branches `vx`, `vy`, `vq`, and `vt` to reduce file size.

Do not start the full `/mnt/mydisk/WCDA_observation` batch job until the observation data migration into the flattened `MMDD/...` layout is complete. Before any large run, execute the single-file smoke command above with `--max-events-per-file` and check the generated `apply_summary.json` under the output root.

## SED Apply Workflow
The Crab SED roadmap source lives in `apply/report/roadmap.md`; its checked-in HTML companion is `apply/report/roadmap.html`. Keep the two synchronized when roadmap content changes. The current v1 fitting-cell selection is `apply/config/cell_selection_v1.csv`, with 18 physical-band cells selected from the 60 formal acceptable candidates in `apply/summary_selectedcuts/bin_counts.csv`.

Stage B PSF production should use the parallel Slurm entrypoint `scripts/slurm/build_psf_stage_b.sbatch`. It runs `apply/stages/02_build_psf.py` against the Stage A response metadata at `apply/output/stage_a/response_2d_metadata.json`, writes outputs to `apply/output/stage_b/`, and passes `--workers "${SLURM_CPUS_PER_TASK:-18}"` so selected cells are processed concurrently. The current production resource request is 18 CPU cores and 64 GB RAM on the `main` partition; if the node is idle, prefer this parallel job over the old serial run. Before relaunching after code or Stage A metadata changes, cancel any stale Stage B job with `scancel <jobid>` to avoid simultaneous writes to `apply/output/stage_b/`.

Launch Stage B with `sbatch scripts/slurm/build_psf_stage_b.sbatch`. Monitor it with `squeue -j <jobid>`, `tail -f logs/slurm/build_psf_b_<jobid>.out`, and `tail -f logs/slurm/build_psf_b_<jobid>.err`. A healthy parallel run prints `Processing 18 cells with 18 workers.` and then interleaved per-cell file-read progress. Validate completion by checking `apply/output/stage_b/psf_v1_metadata.json`, `psf_v1_summary.csv`, `psf_v1_summary.md`, and `psf_v1.npz`; the metadata `stage_a_snapshot` should show the intended Stage A response type, currently `primary_thrown_response` with `absolute_effective_area_status` set to `available`.

For GitHub Pages publishing, treat `/mnt/mydisk/server/projects/any-reports/index.html` as the manual registry for reports shown on the homepage. Adding a file under `any-reports/` makes it reachable by direct URL, but it will not appear on the homepage unless `index.html` is updated.

`apply/plot/` and `apply/summary_selectedcuts/` contain local generated diagnostics and summaries from the MC apply prototype. They are useful for development but ignored by Git. The reference thesis PDF is kept locally under `apply/report/reference/` and is also ignored by Git.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects, for example `Add apply plotting helpers` or `Remove recoxy smoke test script and ignore codex file`. Keep commits scoped to one change. PRs should include a concise summary, the affected paths, the exact validation command(s) you ran, and any output locations or screenshots needed to review notebook or plotting changes.
