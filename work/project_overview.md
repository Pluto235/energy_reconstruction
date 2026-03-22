# Project Overview

## Project Goal

- This project targets energy reconstruction for LHAASO WCDA simulated events.
- The maintained pipeline is focused on event-level regression of `log10(mc_energy / GeV)` from variable-length hit patterns.
- The main research question is how well a point-cloud model can reconstruct energy under different event-quality and geometry selections.
- A second core question is how sensitive the model is to selection variables such as `theta`, `dcedge`, `dangle`, `pincness`, and `fitstat`.

## Current Implemented Modules

### Active Pipeline

- `src/theta/main_theta.py`
  - Main training and evaluation entrypoint.
- `src/theta/ParticleDataset_theta.py`
  - ROOT loading, event filtering, hit truncation, normalization, padding, and tensor packaging.
- `src/theta/ParticleRegressor_theta.py`
  - ParticleNet/EdgeConv-style regressor with optional theta embedding.
- `src/theta/train_theta.py`
  - Training loop, histogram-based reweighting over `logE`, early stopping, and checkpoint saving.
- `src/theta/evaluate_theta.py`
  - Evaluation, weighted/unweighted metrics, plots, and prediction export.
- `src/theta/evaluate_only.py`
  - Re-evaluation of an existing run.
- `src/theta/eval_compare_relaxed.py`
  - Evaluation-only cut override for controlled generalization studies.
- `src/common/EdgeConv.py`
  - Shared KNN and EdgeConv implementation.
- `src/common/utils.py`
  - Plotting and metric utilities.

### Experiment Infrastructure

- `scripts/slurm/`
  - Maintained Slurm entrypoints for training and comparison jobs.
- `notebook/`
  - Analysis notebooks for cut studies and result comparison.
- `runs/`
  - Preserved experiment outputs, configs, checkpoints, metrics, and figures.
- `archive/`
  - Historical non-theta and older theta code paths.

## Model And Method Summary

### Input Representation

- Event hits use `vx`, `vy`, `vt`, and `vq`.
- Points are constructed from hit coordinates centered by true MC core `(mc_xc, mc_yc)`.
- Features are currently based on charge and time.
- The regression target is `log10(mc_energy)`.

### Event-Level Selection

- The pipeline supports cuts on:
  - `Emin`, `Emax`
  - `dcedge_min`
  - `dangle_max_deg`
  - `theta_max_deg`
  - `pinc_max`
  - `fitstat`
  - optional true core box
  - optional `vqsamp` ratio

### Network

- Backbone: three EdgeConv blocks with feature fusion and global average pooling.
- Global conditioning: event-level `costheta = cos(theta)` can be embedded with a small MLP and concatenated before the FC head.
- This is the main maintained architectural extension beyond the older non-theta baseline.

### Training

- Optimizer: Adam with cosine annealing scheduler.
- Early stopping is used in most runs.
- Loss options: `mse`, `huber`, `rel`.
- Current mainline experiments mostly use `huber`.
- Training uses histogram-based inverse-frequency weighting in `logE` to partially compensate for spectral imbalance.

### Evaluation

- Both unweighted and `mc_weight`-weighted metrics are produced.
- Main reported curve-level quantities:
  - bias
  - log RMS error
  - resolution
- Important note:
  - current `resolution` is defined from the spread of `logE_pred` in true-energy bins, not from a standard residual-width convention.

## Current Progress Status

### Stable Enough For Internal Study

- The strict-cut theta pipeline is operational and has produced multiple completed runs.
- Existing studies include:
  - theta embedding vs no theta embedding
  - `fitstat` comparison
  - `dcedge` / core selection comparison
  - `dangle` comparison
  - `pincness` comparison
  - relaxed-evaluation-only comparisons

### Main Observed Findings

- Theta embedding improves over the older non-theta baseline in the existing comparison runs.
- Relaxing `dangle` causes the strongest degradation among the tested evaluation-only cut changes.
- Relaxing `pincness` also degrades performance noticeably.
- Relaxing `dcedge` has a smaller effect than `dangle` and `pincness`.
- A fully relaxed all-cuts training regime is not yet reliable at large scale.

### Maturity Assessment

- The project is usable for controlled internal experiments on selected event domains.
- It is still research-stage rather than production-stable.

## Known Issues

### Scientific / Modeling Risks

- Strong selection dependence:
  - the default strict selection keeps only a small fraction of events.
- Generalization outside the strict training domain is limited.
- Some selection variables use reconstructed quantities, while others use truth-level quantities, which can bias offline conclusions.
- Direct comparison of metrics across different cut-defined samples must be interpreted carefully because the sample distribution changes.

### Engineering / Pipeline Risks

- Paths are heavily coupled to the current filesystem layout.
- File-level train/test splitting is reproducible only as long as file ordering and file set remain unchanged.
- The dataset currently loads filtered events into memory, which is inefficient for very loose cuts.
- The repository currently contains some uncommitted experimental changes in the working tree.

### Correctness Risks Identified

- Evaluation has a bug path when `costheta` is missing but still moved to device.
- `preds.npz` currently saves only the last minibatch `costheta` rather than the full-event array.
- CLI-exposed `min_count` and `max_weight` are not passed through to `train_model(...)` in the current training call.

## Next-Step Plan

### Immediate

- Fix the known evaluation and configuration correctness issues.
- Preserve evaluation artifacts with consistent effective configs and full per-event saved arrays.
- Record exact file lists or stronger run metadata for reproducibility.

### Near-Term Research

- Expand evaluation-only relaxed-cut studies to additional variables, especially `theta` and `fitstat`.
- Produce 2D performance summaries versus energy and zenith-related variables.
- Clarify whether `theta` should be used only as a cut, only as an input, or both.

### Medium-Term Improvement

- Improve data handling for large relaxed datasets, likely through a more scalable loading strategy.
- Revisit the balance between hard event cuts and conditioning variables as model inputs.
- Standardize metric definitions for HEP-facing reporting and comparison.
