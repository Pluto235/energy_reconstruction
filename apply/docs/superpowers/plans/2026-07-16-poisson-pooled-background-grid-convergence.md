# Scheme R Double-Rayleigh Poisson-Pooled Background Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Scheme R double-Rayleigh Stage D observed-count weighted least-squares background surface with a positive Poisson-likelihood surface, stabilize sparse cells through frozen local pooling, and validate the result with a 12-branch event-level grid convergence experiment.

**Architecture:** Keep `0.1 deg` as the nominal background-map sampling and reuse the prerequisite analytic circular-aperture integration commit. Fit ordinary constant/plane/quadratic intensity polynomials with exact rectangular pixel integrals and explicit positivity over the full `rho < 6 deg` disk. Build one grid-independent pooling manifest from 84 non-tail cells, apply it unchanged to all 12 grid branches, propagate every branch through Stage F, and add a nominal-only bootstrap covariance diagnostic without changing the legacy conservative primary comparison.

**Tech Stack:** Python 3.10, NumPy, SciPy constrained optimization, PyArrow Stage C datasets, iminuit Stage F fits, `unittest`, Slurm arrays, JSON/NPZ/HTML report artifacts.

---

Implement in a dedicated git worktree created after the analytic-B_on prerequisite is pushed. Do not develop inside the shared production checkout or sync over another agent's uncommitted files. Merge the reviewed implementation to `main` before submitting production Slurm jobs from `/home/server/projects/energy_reconstruction/apply`.

## Fixed Scientific Contract

- Production target: current double-Rayleigh PSF with `F(r_opt)=0.7129790300890827` and Scheme R response `Aeff_R=0.7129790300890827*Aeff_nominal`, applied once.
- New experiment id: `v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled`.
- Existing pixel-WLS, analytic-B_on/WLS, Scheme B, and Scheme R products are read-only baselines. Never overwrite their run directories, reports, assets, `current`, or `latest` links.
- Stage B/C inputs, the 44-cell Stage F selector, containment override, response, LogPar model, and Stage G grouping remain unchanged.
- Nominal donor universe: the 84 candidate cells with `predE < 6`; the seven `predE >= 6` diagnostic tails are excluded.
- Pooling never crosses an `Nhit` bin. Sparse targets expand to nearest `predE` neighbors within the same `Nhit`; equal-distance ties choose lower `predE` first.
- Independent quadratic eligibility: continuous `N_annulus >= 20_000` and validation passes.
- Pooled quadratic target: expand neighbors until continuous combined `N_annulus >= 10_000`.
- If the full same-`Nhit` donor set cannot reach `10_000`, or cross-validation rejects quadratic, choose a shared positive plane. If the target has `<100` annulus events, it contributes only its normalization and no shape information.
- Candidate order selection uses eight azimuthal leave-one-sector-out folds; pooled models additionally use leave-one-donor-cell-out. Apply the one-standard-error rule across constant, plane, and quadratic candidates.
- The pooling topology, chosen order, donor list, continuous counts, validation scores, input SHAs, and analytic-integration baseline SHA are frozen in one manifest used by all branches.
- Nominal grid: `h=0.1 deg`, offset `(0,0)`. Convergence branches use `h in {0.05,0.1,0.2}` and offsets `(0,0)`, `(h/2,0)`, `(0,h/2)`, `(h/2,h/2)`.
- All 12 branches run Stage D/E/F. Only nominal runs production Stage G and the 1,000-replicate background bootstrap.
- Legacy conservative Stage F remains the primary new-versus-old comparison. A covariance-aware fit is a named diagnostic and must not silently become preferred.

## File Map

**Create**

- `apply/stages/poisson_roi_background.py`: exact polynomial pixel integrals, positive profiled-Poisson fitting, deterministic CV, and frozen-pool execution.
- `apply/tools/build_v6_poisson_pooling_manifest.py`: scan continuous event radii, construct the 84-cell donor graph, run nominal CV, and write the immutable manifest.
- `apply/tools/bootstrap_v6_poisson_background.py`: nominal parametric bootstrap and `44x44 Cov(B_on)` output.
- `apply/report/build_v6_poisson_grid_convergence.py`: aggregate 12 Stage D/E/F branches and calculate registered convergence metrics.
- `apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py`: preflight, computation, convergence, bootstrap, report, SHA, and Slurm validation.
- `apply/report/prepare_v6_scheme_r_double_rayleigh_poisson_pooled_report.py`: add the old pixel-WLS, analytic-B_on/WLS, Poisson nominal, covariance-aware, and grid-envelope sections.
- `apply/tests/test_poisson_roi_background.py`
- `apply/tests/test_poisson_pooling_manifest.py`
- `apply/tests/test_fit_full_covariance.py`
- `apply/tests/test_poisson_grid_convergence.py`
- `scripts/slurm/build_v6_64748_poisson_pooling_manifest.sbatch`
- `scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch`
- `scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch`
- `scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch`
- `scripts/slurm/submit_v6_64748_scheme_R_poisson_pooled.sh`

**Modify**

- `apply/stages/04_background.py`: add Poisson/manifest mode and independent x/y grid offsets while retaining WLS defaults.
- `apply/stages/06_fit.py`: accept an optional full excess covariance and emit separately named covariance-aware fits.
- `apply/report/build_v6_64748_nhit100_highEplus1_report.py`: render optional Poisson/pooling provenance without changing existing reports when variables are absent.
- `apply/devlog.md`: prepend one final experiment entry after implementation and validation.

**Generated And Committed After Slurm**

- `apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json`
- `apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_poisson_pooled_report.html`
- `apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-pooled/`
- `scheme_R_double_rayleigh_poisson_pooled_validation.json`
- `scheme_R_double_rayleigh_poisson_pooled_comparison.json`
- `scheme_R_double_rayleigh_poisson_grid_convergence.json`
- `scheme_R_double_rayleigh_poisson_background_covariance.npz`

### Task 1: Lock The Analytic-Integration Baseline

**Files:**
- Verify: `apply/stages/04_background.py`
- Verify: `apply/tests/test_background_analytic_bon.py`
- Verify: `apply/report/validate_v6_scheme_r_double_rayleigh_analytic_bon.py`

- [ ] **Step 1: Require the completed analytic-B_on commit**

Run after the other agent has pushed its work:

```bash
git fetch origin main
git worktree add -b feat/poisson-pooled-background ../energy-reconstruction-poisson origin/main
cd ../energy-reconstruction-poisson
test -z "$(git status --short)"
export ANALYTIC_BON_BASELINE_SHA=$(git rev-parse origin/main)
git show --stat --oneline "$ANALYTIC_BON_BASELINE_SHA"
```

Expected: the commit contains the analytic disk integrator, positivity check, tests, isolated runner/finalizer, and analytic-B_on report validator.

- [ ] **Step 2: Verify the prerequisite API and tests**

```bash
python -m unittest apply.tests.test_background_analytic_bon -v
python -m py_compile apply/stages/04_background.py \
  apply/report/validate_v6_scheme_r_double_rayleigh_analytic_bon.py
```

Expected: all tests pass and compilation exits `0`.

- [ ] **Step 3: Record the exact prerequisite SHA in the new preflight contract**

The new validator must require `ANALYTIC_BON_BASELINE_SHA`, verify it is an ancestor of `HEAD`, and write it to every manifest, Stage D metadata file, comparison JSON, and final report. Missing or non-ancestor SHAs are fatal.

### Task 2: Implement Exact Positive Poisson Surface Fitting

**Files:**
- Create: `apply/stages/poisson_roi_background.py`
- Create: `apply/tests/test_poisson_roi_background.py`

- [ ] **Step 1: Write failing exact-pixel-integral tests**

Use these basis integrals for a rectangle `[x0,x1] x [y0,y1]`:

```python
expected = np.array([
    (x1 - x0) * (y1 - y0),
    0.5 * (x1**2 - x0**2) * (y1 - y0),
    (x1 - x0) * 0.5 * (y1**2 - y0**2),
    (x1**3 - x0**3) * (y1 - y0) / 3.0,
    0.25 * (x1**2 - x0**2) * (y1**2 - y0**2),
    (x1 - x0) * (y1**3 - y0**3) / 3.0,
])
actual = quadratic_rectangle_basis_integrals(x0, x1, y0, y1)
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
```

Also test that summing four `0.1 deg` child pixels exactly equals one `0.2 deg` parent pixel for all six basis terms.

- [ ] **Step 2: Run the tests and verify they fail**

```bash
python -m unittest apply.tests.test_poisson_roi_background.ExactPixelIntegralTests -v
```

Expected: import failure because `poisson_roi_background.py` does not exist.

- [ ] **Step 3: Implement the immutable fit interfaces**

Define these public types and functions exactly:

```python
@dataclass(frozen=True)
class SurfaceFit:
    order: int
    shape_coefficients: np.ndarray  # [1, x, y, x2, xy, y2], c0 fixed to 1
    donor_cell_ids: tuple[int, ...]
    annulus_normalizations: dict[int, float]
    poisson_deviance: float
    ndof: int
    positive_minimum: float
    positive_minimum_xy: tuple[float, float]
    optimizer_status: dict[str, object]

def quadratic_rectangle_basis_integrals(
    x0: float, x1: float, y0: float, y1: float
) -> np.ndarray: ...

def fit_profiled_poisson_surface(
    counts_by_cell: dict[int, np.ndarray],
    pixel_basis_by_cell: dict[int, np.ndarray],
    annulus_mask_by_cell: dict[int, np.ndarray],
    donor_cell_ids: Sequence[int],
    order: int,
    positivity_radius_deg: float,
    shape_contributor_by_cell: dict[int, bool],
) -> SurfaceFit: ...
```

For cell `b`, profile its annulus normalization as `A_b=sum(z_b)` and use

```python
pixel_shape = pixel_basis @ shape_coefficients
pixel_probability = pixel_shape / pixel_shape.sum()
nll_b = -np.sum(z_b * np.log(pixel_probability))
```

Fix `c0=1` to remove the scale degeneracy. Use exact rectangle basis integrals, not center samples.

- [ ] **Step 4: Add positivity and recovery tests**

Generate deterministic constant, plane, and quadratic Poisson maps with seed `64748`. Assert:

```python
self.assertGreater(result.positive_minimum, 0.0)
np.testing.assert_allclose(result.shape_coefficients, truth, rtol=0.08, atol=0.01)
```

Add a negative-curvature case whose unconstrained optimum crosses zero inside `rho<6`; assert the constrained result stays positive. Add an impossible initialization case and assert a typed `PoissonSurfaceFitError`, never silent clipping.

- [ ] **Step 5: Implement deterministic constrained optimization**

Use SciPy constrained minimization with a cutting-plane loop:

1. constrain the polynomial at the training-pixel corners and a fixed `0.25 deg` support grid inside `rho<6`;
2. find the exact minimum of the fitted quadratic on the centered disk;
3. if the minimum is below `1e-12` times the fitted central intensity, add the violating point as a linear constraint and refit;
4. fail after eight iterations rather than clipping the model.

The final postcondition is `positive_minimum > 0` over the entire disk, not only at grid centers.

- [ ] **Step 6: Run focused and full local tests**

```bash
python -m unittest apply.tests.test_poisson_roi_background -v
python -m unittest discover -s apply/tests -v
python -m py_compile apply/stages/poisson_roi_background.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the pure fitting unit**

```bash
git add apply/stages/poisson_roi_background.py \
  apply/tests/test_poisson_roi_background.py
git commit -m "feat(stage-d): add positive Poisson ROI surface fitting"
```

### Task 3: Build And Freeze The 84-Cell Pooling Manifest

**Files:**
- Create: `apply/tools/build_v6_poisson_pooling_manifest.py`
- Create: `apply/tests/test_poisson_pooling_manifest.py`
- Create after Slurm: `apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json`

- [ ] **Step 1: Write failing deterministic-neighbor tests**

Construct one synthetic `Nhit` group and assert:

```python
counts = {1: 25_000, 2: 8_000, 3: 1_500, 4: 50}
pred_e_center = {1: 2.25, 2: 2.75, 3: 3.125, 4: 3.375}
manifest = build_pooling_manifest(counts, pred_e_center, target_ids=(1, 3, 4))
self.assertEqual(manifest[1]["mode"], "independent")
self.assertEqual(manifest[3]["donor_cell_ids"], [2, 3, 4])
self.assertFalse(manifest[4]["shape_contributor"])
```

Also assert that no donor crosses `Nhit`, no `predE>=6` cell appears, and lower-`predE` wins exact distance ties.

- [ ] **Step 2: Implement continuous annulus counting**

Read Stage C Parquet columns `ra_mean_deg`, `dec_mean_deg`, and `cell_id`; calculate continuous Crab-centered `rho`; count events using each cell's exact existing annulus boundaries. Do not use pixel-center annulus masks for manifest thresholds.

- [ ] **Step 3: Implement fixed tiering and local expansion**

Apply the accepted `20k/10k/100` rules. The manifest must include:

```json
{
  "schema_version": 1,
  "target_cell_ids": [1],
  "donor_universe_cell_ids": [1],
  "excluded_tail_cell_ids": [13, 26, 39, 52, 65, 78, 91],
  "cells": {
    "1": {
      "continuous_annulus_count": 2625211,
      "mode": "independent",
      "surface_order": 2,
      "donor_cell_ids": [1],
      "shape_contributor": true
    }
  }
}
```

The real arrays contain all 44 targets and all 84 donor-universe ids.

- [ ] **Step 4: Add the registered cross-validation**

Use eight fixed azimuth sectors starting at angle zero. For independent candidates, compute leave-one-sector-out held-out Poisson deviance. For pooled candidates, also leave out each donor cell, profile only its normalization, and score its spatial counts. Apply the one-standard-error rule and store every candidate's fold scores and selected order.

- [ ] **Step 5: Add provenance and self-hash validation**

Store SHA256 for the 44-cell selector, 91-cell source table, Stage B PSF NPZ/metadata, Stage C metadata/source-files CSV, analytic-B_on baseline commit, code file, and manifest payload. Reload the written JSON and assert its canonical JSON SHA matches `manifest_sha256`.

- [ ] **Step 6: Run local fixture tests**

```bash
python -m unittest apply.tests.test_poisson_pooling_manifest -v
python -m py_compile apply/tools/build_v6_poisson_pooling_manifest.py
```

Expected: pass without reading production events.

- [ ] **Step 7: Commit the manifest builder**

```bash
git add apply/tools/build_v6_poisson_pooling_manifest.py \
  apply/tests/test_poisson_pooling_manifest.py
git commit -m "feat(stage-d): freeze sparse-cell background pooling"
```

### Task 4: Integrate Poisson/Pooling And Grid Phase Into Stage D

**Files:**
- Modify: `apply/stages/04_background.py`
- Extend: `apply/tests/test_poisson_roi_background.py`

- [ ] **Step 1: Write failing CLI compatibility tests**

Assert defaults remain byte-for-byte compatible:

```python
with mock.patch("sys.argv", ["04_background.py"]):
    args = stage04.parse_args()
self.assertEqual(args.roi_fit_statistic, "weighted-ls")
self.assertEqual(args.roi_grid_offset_x_fraction, 0.0)
self.assertEqual(args.roi_grid_offset_y_fraction, 0.0)
self.assertIsNone(args.pooling_manifest)
```

Assert offset fractions outside `{0,0.5}` and Poisson mode without a manifest fail preflight.

- [ ] **Step 2: Add explicit CLI flags**

```text
--roi-fit-statistic {weighted-ls,poisson}
--pooling-manifest PATH
--roi-grid-offset-x-fraction {0,0.5}
--roi-grid-offset-y-fraction {0,0.5}
```

Keep all existing defaults unchanged. In Poisson mode require `annulus-quadratic`, `annulus-normalize-surface`, and `analytic-quadratic` aperture integration.

- [ ] **Step 3: Make grid phase explicit**

Construct edges as `[-8 + offset, 8 + offset]` with enough exterior padding to preserve complete `rho<8` coverage, then mask back to the same physical fiducial disk. Metadata must record step, offset fraction, offset degrees, shape, and physical coverage. The four phases must contain the same event-level fiducial and annulus populations up to exact continuous boundary rules.

- [ ] **Step 4: Load and enforce the frozen manifest**

Scan only the manifest's 84 donor-universe cells, fit exactly the manifest-specified donor groups and orders, and write Stage D arrays for all 84 cells so Stage E can select the original 44 by id. Recompute and compare continuous annulus counts; any mismatch with the manifest is fatal.

- [ ] **Step 5: Use profiled Poisson normalization and analytic B_on**

For cell `b` with shared shape `q(x,y)`:

```python
B_on_b = N_annulus_b * integral_disk(q, r_opt_b) / integral_annulus(q, inner_b, outer_b)
```

Use the prerequisite analytic disk API through a tested density-to-legacy coefficient adapter. Use closed-form polynomial annulus integrals. Do not apply a second `annulus_surface_scale` and never clip a negative surface.

- [ ] **Step 6: Extend Stage D output contracts**

Add NPZ/metadata fields for fit statistic, density coefficients, polynomial order, pool id, donor ids, manifest SHA, continuous annulus counts, Poisson deviance/ndof, CV scores, exact positive minimum, optimizer status, grid phase, analytic B_on, and legacy pixel-center B_on as a diagnostic only.

- [ ] **Step 7: Verify old WLS behavior and new synthetic behavior**

```bash
python -m unittest apply.tests.test_background_analytic_bon -v
python -m unittest apply.tests.test_poisson_roi_background -v
python -m py_compile apply/stages/04_background.py
```

Expected: old tests pass; synthetic Poisson results are invariant under exact `0.1 -> 0.2` rebinning within test tolerance.

- [ ] **Step 8: Commit Stage D integration**

```bash
git add apply/stages/04_background.py \
  apply/tests/test_background_analytic_bon.py \
  apply/tests/test_poisson_roi_background.py
git commit -m "feat(stage-d): add pooled Poisson background mode"
```

### Task 5: Add Nominal Bootstrap Covariance And Correlated Stage F

**Files:**
- Create: `apply/tools/bootstrap_v6_poisson_background.py`
- Modify: `apply/stages/06_fit.py`
- Create: `apply/tests/test_fit_full_covariance.py`

- [ ] **Step 1: Write failing generalized-chi2 tests**

For residual vector `r` and covariance `C`, require:

```python
actual = generalized_chi2(residual, covariance)
expected = float(residual @ np.linalg.solve(covariance, residual))
self.assertAlmostEqual(actual, expected, places=12)
```

Assert diagonal `C` exactly reproduces the existing scalar-error objective and a non-positive-definite matrix fails with a clear message.

- [ ] **Step 2: Add optional Stage F covariance input**

Add `--excess-covariance-npz`. Validate exact 44-cell id/order and load `excess_covariance`. Use Cholesky solves; never invert explicitly. Emit `pl_background_covariance` and `logpar_background_covariance` fit records, marginal pulls, and whitened residuals. Keep `preferred_fit` on `logpar_conservative`/`pl_conservative` exactly as before.

- [ ] **Step 3: Implement deterministic bootstrap**

The bootstrap script reads nominal fitted pixel expectations and the frozen manifest. For each replicate, draw independent Poisson pixel counts, refit every frozen surface without changing donors/order, calculate 44 analytic `B_on` values, and store them. Use seed `64748`, `100` smoke replicates, `1000` production replicates, and deterministic replicate-index seed splitting across 32 workers.

- [ ] **Step 4: Write the covariance artifact**

Output arrays:

```text
cell_id
B_on_nominal
B_on_bootstrap_mean
B_on_bootstrap_samples
B_on_covariance
excess_covariance = diag(N_on) + B_on_covariance
```

Metadata includes bootstrap count, seed, manifest SHA, Stage D SHA, eigenvalues, condition number, per-entry Monte Carlo standard error, and refit failure count. Any failed production replicate is fatal; smoke failures may only be used to debug.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest apply.tests.test_fit_full_covariance -v
python -m py_compile apply/stages/06_fit.py \
  apply/tools/bootstrap_v6_poisson_background.py
git add apply/stages/06_fit.py \
  apply/tools/bootstrap_v6_poisson_background.py \
  apply/tests/test_fit_full_covariance.py
git commit -m "feat(stage-f): add Poisson background covariance diagnostics"
```

### Task 6: Implement The Registered Grid-Convergence Validator

**Files:**
- Create: `apply/report/build_v6_poisson_grid_convergence.py`
- Create: `apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py`
- Create: `apply/tests/test_poisson_grid_convergence.py`

- [ ] **Step 1: Write failing threshold tests**

Synthetic fixtures must exercise every registered gate:

```python
self.assertTrue(evaluate_cell_gate(all_sigma_units=[0.2] * 44).passed)
self.assertFalse(evaluate_cell_gate(all_sigma_units=[0.2] * 43 + [0.51]).passed)
self.assertFalse(evaluate_beta_gate(delta_beta=0.0031, sigma_beta=0.0121).passed)
self.assertFalse(evaluate_pull_gate(delta_pull=0.51).passed)
```

- [ ] **Step 2: Aggregate all 12 branches by explicit branch id**

Use ids `h005_x0_y0`, `h005_xh_y0`, `h005_x0_yh`, `h005_xh_yh`, and corresponding `h010`/`h020` forms. Reject missing, duplicate, or unexpected branches. Verify identical manifest, selector, PSF, Stage C, response, analytic baseline, and code SHAs.

- [ ] **Step 3: Implement the registered gates**

- all 44 cells: `abs(delta_B_on)/sigma_excess <= 0.5`;
- at least 90%: `<=0.25`;
- same thresholds for fixed-resolution four-phase RMS;
- every fit-parameter Mahalanobis distance from nominal `<=0.5`;
- every `abs(delta_beta) <= 0.25*sigma_beta_nominal`;
- every `abs(delta_pull) <=0.5`;
- no grid-only migration across `abs(pull)=5`;
- all Stage D and Stage F fits valid and positive;
- all Slurm jobs `COMPLETED`.

Every check is serialized as `{name, passed, observed, limit, evidence}`. A failed check makes the overall validation fail; it never selects a better-looking branch.

- [ ] **Step 4: Write comparison artifacts and plots**

Produce per-cell `B_on` phase/resolution envelopes, pull envelopes, `phi0/alpha/beta` branch plots, joint Mahalanobis distances, model-tier/donor maps, CV deviance tables, covariance/correlation heatmaps, and legacy-versus-new SED ratio panels.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest apply.tests.test_poisson_grid_convergence -v
python -m py_compile \
  apply/report/build_v6_poisson_grid_convergence.py \
  apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py
git add apply/report/build_v6_poisson_grid_convergence.py \
  apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py \
  apply/tests/test_poisson_grid_convergence.py
git commit -m "feat(report): validate Poisson background grid convergence"
```

### Task 7: Add The Isolated Slurm Workflow

**Files:**
- Create: `scripts/slurm/build_v6_64748_poisson_pooling_manifest.sbatch`
- Create: `scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch`
- Create: `scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch`
- Create: `scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch`
- Create: `scripts/slurm/submit_v6_64748_scheme_R_poisson_pooled.sh`

- [ ] **Step 1: Implement manifest preflight job**

Request `32 CPU / 96 GB / 4 h`. Verify the analytic baseline SHA, selector SHA, PSF target, Stage C provenance, 84 donor ids, 44 target ids, and fixed thresholds before scanning events. Write the manifest to a new output namespace and copy the validated canonical JSON to `apply/config/` only after its self-hash passes.

- [ ] **Step 2: Implement the 12-index grid mapping**

The array runner maps indices exactly:

```bash
steps=(0.05 0.05 0.05 0.05 0.10 0.10 0.10 0.10 0.20 0.20 0.20 0.20)
xfrac=(0 0.5 0 0.5 0 0.5 0 0.5 0 0.5 0 0.5)
yfrac=(0 0 0.5 0.5 0 0 0.5 0.5 0 0 0.5 0.5)
```

Each task requests `32 CPU / 128 GB / 8 h`, runs Stage D/E/F, uses `--no-promote-current`, and writes a branch-specific namespace. Index `4` is nominal `0.1 deg + (0,0)`.

- [ ] **Step 3: Implement nominal bootstrap and Stage G jobs**

After array task `4` succeeds, request `32 CPU / 128 GB / 12 h` for 1,000 bootstrap replicates and covariance-aware Stage F. Generate Stage G only for nominal. No other branch creates Stage G.

- [ ] **Step 4: Implement finalizer and dependency graph**

Request `4 CPU / 32 GB / 2 h`. Submit:

```text
manifest -> grid array 0-11%4
grid task 4 -> bootstrap
whole grid array + bootstrap -> finalizer/report
```

Pass every Slurm job id to the validator and report metadata. `afterok` is mandatory; no manual polling loop.

- [ ] **Step 5: Static-check and commit runners**

```bash
bash -n scripts/slurm/build_v6_64748_poisson_pooling_manifest.sbatch
bash -n scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch
bash -n scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch
bash -n scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch
bash -n scripts/slurm/submit_v6_64748_scheme_R_poisson_pooled.sh
git add \
  scripts/slurm/build_v6_64748_poisson_pooling_manifest.sbatch \
  scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch \
  scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch \
  scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch \
  scripts/slurm/submit_v6_64748_scheme_R_poisson_pooled.sh
git commit -m "feat(slurm): run Poisson background grid convergence"
```

### Task 8: Build The Independent Report And Final Validation

**Files:**
- Create: `apply/report/prepare_v6_scheme_r_double_rayleigh_poisson_pooled_report.py`
- Modify: `apply/report/build_v6_64748_nhit100_highEplus1_report.py`
- Generate: final HTML/assets/JSON listed in the File Map
- Modify: `apply/devlog.md`

- [ ] **Step 1: Add optional report sections without changing old output**

Only render Poisson sections when `V6_REPORT_POISSON_MANIFEST`, `V6_REPORT_GRID_CONVERGENCE`, and `V6_REPORT_BACKGROUND_COVARIANCE` are set. Existing report environment combinations must remain unchanged.

- [ ] **Step 2: Build the scientific comparison**

The final report must show four distinct contracts:

1. old R-2R pixel-WLS plus pixel-center B_on;
2. prerequisite R-2R WLS plus analytic B_on;
3. new nominal Poisson-pooled plus legacy conservative errors;
4. new nominal Poisson-pooled covariance-aware diagnostic.

Include the 12-branch envelope, model tier and donor table, continuous annulus counts, CV evidence, positive minima, response/selector/manifest SHAs, bootstrap covariance, Slurm provenance, and every failed/passed convergence gate. R-1R is secondary historical context only.

- [ ] **Step 3: Run local static verification before ETO sync**

```bash
python -m unittest discover -s apply/tests -v
python -m py_compile \
  apply/stages/04_background.py \
  apply/stages/06_fit.py \
  apply/stages/poisson_roi_background.py \
  apply/tools/build_v6_poisson_pooling_manifest.py \
  apply/tools/bootstrap_v6_poisson_background.py \
  apply/report/build_v6_poisson_grid_convergence.py \
  apply/report/prepare_v6_scheme_r_double_rayleigh_poisson_pooled_report.py \
  apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py
```

Expected: all tests pass and compilation exits `0`.

- [ ] **Step 4: Sync and submit only after the analytic baseline is committed**

Follow the workspace ETO rule. Confirm no unrelated user edits would be deleted before using the prescribed `rsync --delete`; exclude `devlog.md` on the return sync. Submit the new workflow from `/home/server/projects/energy_reconstruction` and record all job ids.

- [ ] **Step 5: Run final validator**

```bash
python apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py --require-report
```

Expected: report references exist, metadata contamination is zero, all 12 branch jobs and bootstrap/finalizer are `COMPLETED`, all SHAs match, covariance is valid, and every registered convergence result is explicitly present. The command exits nonzero if convergence fails, while preserving the report and evidence.

- [ ] **Step 6: Sync targeted results back and update devlog**

Sync the final HTML, dedicated asset directory, canonical manifest, source/test files, and Slurm scripts. Prepend exactly one newest-first devlog entry describing the completed Poisson-pooled grid-convergence experiment.

- [ ] **Step 7: Commit only this experiment and push**

```bash
git status --short
git add \
  apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json \
  apply/stages/04_background.py apply/stages/06_fit.py apply/stages/poisson_roi_background.py \
  apply/tools/build_v6_poisson_pooling_manifest.py apply/tools/bootstrap_v6_poisson_background.py \
  apply/tests/test_poisson_roi_background.py apply/tests/test_poisson_pooling_manifest.py \
  apply/tests/test_fit_full_covariance.py apply/tests/test_poisson_grid_convergence.py \
  apply/report/build_v6_poisson_grid_convergence.py \
  apply/report/prepare_v6_scheme_r_double_rayleigh_poisson_pooled_report.py \
  apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py \
  apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_poisson_pooled_report.html \
  apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-pooled \
  scripts/slurm/build_v6_64748_poisson_pooling_manifest.sbatch \
  scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch \
  scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch \
  scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch \
  scripts/slurm/submit_v6_64748_scheme_R_poisson_pooled.sh \
  apply/devlog.md
git diff --cached --check
git commit -m "feat: validate pooled Poisson background convergence"
git push origin main
```

Inspect `git diff --cached --name-only` before committing. Remove any path not listed above and do not include unrelated user modifications.

## Final Acceptance Checklist

- [ ] Analytic-B_on prerequisite SHA is recorded and is an ancestor of the implementation commit.
- [ ] Old WLS defaults and existing reports remain unchanged.
- [ ] Nominal background grid remains `0.1 deg`; no production claim is based on `0.05 deg` alone.
- [ ] Ordinary polynomial intensity uses exact rectangle and circle/annulus integrals.
- [ ] Every fitted surface is strictly positive on `rho<6 deg`; no clipping is used.
- [ ] Donor universe is exactly 84 non-tail cells; targets remain exactly the original 44.
- [ ] Pooling never crosses `Nhit`; manifest and SHA are identical across 12 branches.
- [ ] `20k/10k/100` rules and one-standard-error CV are applied without looking at Stage F quality.
- [ ] All 12 event-level grid branches reach Stage F; only nominal reaches Stage G/bootstrap.
- [ ] Per-cell, phase, beta, Mahalanobis, pull, and large-pull migration gates are all reported.
- [ ] Legacy conservative and covariance-aware fits are clearly separated.
- [ ] Production bootstrap uses 1,000 successful replicates with seed `64748`.
- [ ] Slurm array uses `%4` concurrency and the accepted resource limits.
- [ ] Final report/assets/validation/comparison paths are independent and complete.
- [ ] `devlog.md` is newest-first; commit contains only this experiment; `origin/main` is pushed.
