# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LHAASO WCDA energy reconstruction: a point-cloud regression pipeline that predicts `log10(mc_energy / GeV)` from variable-length detector hit patterns using a ParticleNet/EdgeConv-style model.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Inspect all training options
python -m src.theta.main_theta --help

# Minimal local training run
python -m src.theta.main_theta \
  --root_path /home/server/mydisk/WCDA_simulation \
  --run_dir runs/dev_run --tag dev_run

# Re-evaluate an existing run
python -m src.theta.evaluate_only --run_dir runs/<run_name>

# Evaluation with cut overrides (generalization studies)
python -m src.theta.eval_compare_relaxed --run_dir runs/<run_name> \
  --eval_pinc_max 2.0 --eval_dcedge_min 10.0

# Smoke test on cluster (20 files, 2 epochs)
sbatch scripts/slurm/theta_smoketest.sbatch

# Flatten observation directory layout before inference
python scripts/data/flatten_observation_layout.py \
  --input-root /mnt/mydisk/WCDA_observation --apply

# Single-file observation inference smoke test
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/server/anaconda3/envs/py310/bin/python apply/apply_observation_energy.py \
  --input-root /mnt/mydisk/WCDA_observation/0101/Esg20220101_14.root \
  --output-root /tmp/wcda_observation_eval_smoke \
  --run-dir runs/theta_recoxy_position_embed_midenergy_8666 \
  --max-events-per-file 128 --device cpu --batch-size 64

# Full observation batch (only after directory migration is complete)
sbatch scripts/slurm/apply_observation_energy.sbatch
```

There is no automated test suite. Validate changes with a smoke test or targeted evaluation run.

## Architecture

### Data flow

```
ROOT files (.root)
  → ParticleDataset (src/theta/ParticleDataset_theta.py)
      - reads branches: vx, vy, vt, vq, vidmc, mc_energy, theta, xc, yc, pincness, dcedge, dangle, fitstat, mc_weight
      - applies event-level cuts (Emin, Emax, pinc_min/max, dcedge_min, dangle_max_deg, theta_max_deg, fitstat, core_box)
      - maps hit detector IDs → global (x, y) coordinates via build_hit_points()
      - features per hit: (vq, vt), truncated/sampled to max_points, padded, packed as (points, features, mask)
      - event-level conditioning: costheta = cos(theta), reco_core_xy = (xc/core_scale, yc/core_scale)
      - target: log10(mc_energy)
  → DataLoader (train/val/test split by file, reproducible seed)
  → ParticleNetRegressor (src/theta/ParticleRegressor_theta.py)
      - BatchNorm on input features (vq, vt)
      - 3 × EdgeConvBlock with KNN (k=16 per block): 64→128→256 channels
      - feature fusion: concat all block outputs → Conv1d(→256) + BN + ReLU
      - masked global average pooling
      - optional theta MLP: costheta scalar → embed (dim=theta_embed_dim, default 16)
      - optional core MLP: (xc_norm, yc_norm) → embed (dim=core_embed_dim, default 16)
      - FC head: 256+theta_embed+core_embed → 256 → 128 → 1
  → logE_pred (log10 GeV)
```

### Key modules

| Path | Role |
|------|------|
| `src/theta/main_theta.py` | Training entrypoint; argparse → config.json → train + evaluate |
| `src/theta/ParticleDataset_theta.py` | ROOT loading, event filtering, hit sampling, tensor packaging |
| `src/theta/ParticleRegressor_theta.py` | EdgeConv model with optional theta/core conditioning |
| `src/theta/train_theta.py` | Training loop: Adam + cosine LR, histogram reweighting, early stopping |
| `src/theta/evaluate_theta.py` | Metrics (bias, log-RMS, resolution), plots, preds.npz export |
| `src/theta/evaluate_only.py` | Re-evaluate from saved run_dir |
| `src/theta/eval_compare_relaxed.py` | Override eval cuts on a finished run for generalization studies |
| `src/common/EdgeConv.py` | KNN + EdgeConv implementation shared across all models |
| `src/common/hit_coordinate_transform.py` | Map detector IDs → global (x, y) via WCDAConfig |
| `src/common/WCDA_configuration.py` | WCDA detector geometry (survey positions) |
| `apply/simulation_all_bin.py` | Apply a trained model to simulation, binned output |
| `apply/apply_observation_energy.py` | Apply model to real WCDA observation ROOT files |

### Run output layout

Each training run writes to `runs/<tag>/`:
- `config.json` — full CLI arguments used
- `best_model.pth` — best checkpoint by val loss
- `metrics.json` — final train/val/test metrics
- `preds.npz` — per-event predictions and truth
- `train_stats.json` — per-epoch loss/metric history
- `*.png` — loss curves, resolution curves, bias plots

### Observation apply workflow

- Finalized model: `runs/theta_recoxy_position_embed_midenergy_8666` (theta + position/core embedding)
- Input: `/mnt/mydisk/WCDA_observation/<MMDD>/` (must be in flattened layout)
- Output: `/mnt/mydisk/WCDA_observation_eval/<MMDD>/` (same structure)
- Output ROOT keeps `t_eventout_h`, adds `ml_logE_pred` and `ml_energy_pred`, drops hit-level branches `vx`, `vy`, `vq`, `vt`
- Only events passing `pincness < 1.1`, `fitstat == 0`, `theta < 50 deg` are written

## Conventions

- Config is persisted to `run_dir/config.json` at training start; `evaluate_only.py` reads it back
- Eval cuts default to training cuts unless explicitly overridden via `--eval_*` flags; `None` means "inherit from train"
- `point_frame` in dataset metadata is `"global_detector_plane"` (current) vs older `"detector"` (local hit coords)
- Slurm scripts assume repo root at `/home/server/projects/energy_reconstruction`, `PYTHONPATH` includes that root, and the `py310` conda environment is active
- `runs/`, `logs/`, `notebook/generated/` are gitignored; never commit generated outputs
- Commit messages use short imperative subjects, e.g. `Add global hit coordinates, pinc_min cut`
