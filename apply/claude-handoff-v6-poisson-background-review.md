# Claude Handoff: v6 Double-Rayleigh Poisson Background Review

This document is the starting context for an independent Claude review of the
LHAASO-WCDA Crab SED background-estimation work completed on 2026-07-16.

## Review mandate

Start in review mode. Do not edit code, rerun production Slurm jobs, change
registered thresholds, regenerate reports, commit, push, reset, clean, or sync
directories until the review findings and next-experiment recommendation have
been discussed with the user.

The review should answer two separate questions:

1. Is the positive pooled-Poisson implementation statistically and
   computationally correct?
2. Is there enough source-blind evidence to prefer it over the fair legacy
   comparator, analytic-`B_on` weighted least squares?

Do not use the Crab Stage F chi-square by itself to select a background model.
That would allow a background estimate to be preferred because it absorbs
source-region fluctuations or compensates for response/model errors.

## Required deliverable

Return findings first, ordered by severity, with exact file and line references.
Separate the response into:

- code-correctness findings;
- statistical/modeling risks;
- validation or test gaps;
- conclusions that are supported by current evidence;
- a scoped next-experiment recommendation.

If no correctness bug is found, say so explicitly. Do not turn modeling
preferences into code-bug findings. Identify assumptions and residual risks.

## Project and execution environment

- Repository: `git@github.com:Pluto235/energy_reconstruction.git`
- Authoritative remote branch: `origin/main`
- Final artifact commit: `9037544d1dc9187b6264b920313230c6494d483b`
- Implementation tip used by production: `98f2cc6ebe15bd5ff1240340f0bd0374f8d282ea`
- Analytic `B_on` prerequisite: `527af0380b1cf44de4a0c20d642174e210ca9485`
- Original implementation plan: `f031f35`
- ETO production checkout: `/home/server/projects/energy_reconstruction`
- Clean review worktree: `/mnt/mydisk/server/projects/energy-reconstruction-poisson`
- Local Mac report copy: `/Users/luoji/Documents/projects/energy/apply`

Important checkout state:

- The clean review worktree is at `9037544` and matches `origin/main`.
- The ETO production checkout branch `main` is at `98f2cc6`, one commit behind
  `origin/main`, because its unrelated dirty files were deliberately preserved.
- Final outputs are physically present in the production tree.
- Review source from the clean worktree. Do not reset, clean, or fast-forward
  the dirty production checkout as part of review.

Heavy computation belongs on ETO through Slurm. The local Mac is for reading,
small checks, and report inspection only.

## Scientific contract held fixed

The experiment is:

```text
v6_64748_nhit100_reselect44_split56_miss030_double_rayleigh_scheme_R_fixed712979_poisson_pooled
```

Fixed inputs and analysis choices:

- double-Rayleigh PSF aperture with
  `F(r_opt)=0.7129790300890827`;
- Scheme R response
  `Aeff_R=0.7129790300890827*Aeff_nominal`, applied exactly once;
- the same Stage C observation events and exposure;
- the same 44-cell Stage F selector;
- containment override equal to one in Stage E;
- the same response, LogPar definition, pivot, Stage F forward folding, and
  Stage G grouping;
- `N_on` counted event-by-event with the continuous spherical source distance;
- `B_on` evaluated by analytic integration of the fitted polynomial over the
  true circular aperture.

The response/PSF/spectrum contracts were intentionally not tuned during this
background experiment.

## Why this experiment exists

The original Stage D implementation fitted a quadratic background surface on a
`0.1 deg` grid and calculated `B_on` by summing pixels whose centers fell inside
the PSF aperture. Stage E calculated `N_on` from continuous event distances.

For cell 1, changing the PSF aperture from `0.853184 deg` to `0.833739 deg`
removed 16 whole background pixels:

```text
N_on:   168751 -> 161349    delta = -7402
B_on:   166526 -> 155053    delta = -11473
excess:   2225 ->   6296    delta = +4071
```

The PSF-radius difference was only `0.019445 deg`; the large excess shift came
from inconsistent continuous versus pixel-center aperture integration. The
analytic-`B_on` prerequisite fixed this separate issue before Poisson fitting
was introduced.

## New background estimator

### Surface family

The intensity shape is an ordinary positive polynomial:

```text
q(x,y) = c0 + cx*x + cy*y + cxx*x^2 + cxy*x*y + cyy*y^2
```

The implementation supports constant, plane, and quadratic orders. It does not
use `exp(quadratic)`, because the ordinary polynomial retains closed-form
rectangle, annulus, and centered-disk integrals.

### Profiled Poisson objective

For cell `b`, the fitted shape is integrated exactly over each square training
pixel. With annulus total `N_ann,b` profiled out,

```text
p_bi(q) = integral_pixel_i(q) / integral_annulus_b(q)
-log L_b = -sum_i n_bi * log(p_bi(q))
```

For a pooled group, shape parameters are shared and each cell keeps its own
normalization. The final background expectation is

```text
B_on,b = N_ann,b * integral_disk(r_opt,b, q) / integral_annulus_b(q)
```

The surface is constrained positive throughout the `rho < 6 deg` fiducial
disk. Negative predictions are not silently clipped.

### Pooling and order selection

The frozen manifest contains:

- 84 non-tail donor cells (`predE < 6`);
- the original 44 spectrum target cells;
- no pooling across an `Nhit` group;
- nearest-`predE` expansion inside the same `Nhit` group;
- independent quadratic eligibility at continuous
  `N_annulus >= 20,000`;
- pooled-shape target count `>=10,000`;
- targets below 100 annulus events do not determine shape;
- eight azimuth-sector leave-one-out CV;
- leave-one-donor-cell-out CV for pooled candidates;
- one-standard-error selection among constant, plane, and quadratic models.

Final order allocation:

```text
37 plane cells
 7 quadratic cells: 1, 2, 4, 5, 6, 19, 20
```

Final mode allocation:

```text
 7 independent quadratic
16 independent plane selected by CV fallback
21 shared-plane fallback
```

Manifest self-hash:

```text
6f03ac649a41e8538b05a8886d1911b3e0990dc279f953459a418630145d1b47
```

### Background uncertainty

The nominal branch ran 1,000 fixed-seed Poisson parametric-bootstrap refits:

```text
requested/completed: 1000/1000
refit failures:      0
seed:                64748
workers:             32
```

The artifact contains `44x44 Cov(B_on)` and

```text
Cov(excess) = diag(N_on) + Cov(B_on)
```

The reproduced covariance is finite, symmetric, positive definite, and has
minimum excess-covariance eigenvalue `12.937772917245463`. Stage F retains the
legacy conservative fit as primary and labels the full-covariance fit as a
diagnostic.

## Registered grid experiment

All branches read the same frozen pooling manifest. The experiment varies only
grid resolution and phase:

```text
h = 0.05, 0.10, 0.20 deg
phase = (0,0), (h/2,0), (0,h/2), (h/2,h/2)
```

This gives 12 Stage D/E/F branches. Nominal is `h010_x0_y0`.

Slurm provenance:

```text
manifest job:  65246       COMPLETED
grid array:    65247_0-11  12/12 COMPLETED
bootstrap:     65248       COMPLETED
finalizer:     65249       FAILED by registered scientific gates
```

The finalizer failure is expected behavior, not an engineering crash. It wrote
the report and evidence, then exited nonzero because convergence was not
accepted.

## Results that passed

- 66/66 unit tests passed.
- Python compilation, shell syntax, and staged-diff checks passed.
- 84/84 fitted donor surfaces were positive.
- 12/12 Stage D and Stage F branches were valid.
- Event-level `N_on` was identical across branches; aggregate
  `N_on=518,803`.
- Phase-RMS half-sigma and 90%-coverage gates passed.
- Maximum spectral-parameter Mahalanobis distance was `0.141762 < 0.5`.
- Maximum `abs(delta_beta)` was `0.00123589 < 0.00358104`.
- No cell migrated across `abs(pull)=5` solely because of grid choice.
- Bootstrap covariance integrity and Stage F covariance linkage passed.
- The final HTML references 28/28 existing images.
- Metadata-contamination count is zero.

## Registered failures

Do not weaken these limits:

| Gate | Observed | Required | Status |
|---|---:|---:|---|
| maximum `abs(delta B_on)/sigma_excess` | 0.786048 | <=0.5 | failed |
| cells with `abs(delta B_on)/sigma_excess<=0.25` | 39/44 = 88.64% | >=90% | failed |
| maximum pull envelope | 0.782051 | <=0.5 | failed |

The largest background/pull drift is cell 20. Instability is concentrated in
the quadratic cells:

| Cell | `B_on` envelope / sigma | phase RMS / sigma | pull envelope |
|---:|---:|---:|---:|
| 20 | 0.7860 | 0.2658 | 0.7821 |
| 5  | 0.6969 | 0.2825 | 0.6929 |
| 6  | 0.4904 | 0.1469 | 0.4845 |
| 4  | 0.4025 | 0.1591 | 0.4014 |
| 19 | 0.3785 | 0.2496 | 0.3738 |
| 1  | 0.2297 | 0.1101 | 0.1568 |
| 2  | 0.2151 | 0.1271 | 0.1892 |

All 37 plane cells are effectively grid-stable at the registered scale. This
points to the curvature layer, not to a general Poisson or pooling failure.

## Spectral comparison

These numbers are diagnostics, not the background-model selection criterion:

| Contract | phi0 | alpha | beta | chi2/ndof |
|---|---:|---:|---:|---:|
| old R-2R pixel-WLS/pixel `B_on` | 2.31213e-12 | 2.77611 | 0.05944 | 521.41/41 |
| R-2R WLS/analytic `B_on` | 2.42654e-12 | 2.76128 | 0.12018 | 452.57/41 |
| Poisson pooled/conservative | 2.43592e-12 | 2.73905 | 0.09976 | 578.40/41 |
| Poisson pooled/full covariance | 2.48046e-12 | 2.73990 | 0.10741 | 802.43/41 |

The analytic-WLS branch has lower Stage F chi-square, but current evidence does
not show whether that is superior source-blind background prediction or
compensation for another mismatch. Even its `chi2/ndof` is about 11, so the
global spectral tension is not solved by any background branch.

## Priority review targets

### P0: Statistical correctness

Review these first:

- `apply/stages/poisson_roi_background.py`
  - `quadratic_rectangle_basis_integrals`
  - `fit_profiled_poisson_surface`
  - normalization profiling and order parameterization
  - full-disk positivity and boundary constraints
- `apply/stages/04_background.py`
  - `estimate_roi_poisson_pooled_background`
  - exact grid offsets and event assignment
  - manifest enforcement
  - analytic disk/annulus `B_on` calculation
- `apply/tools/build_v6_poisson_pooling_manifest.py`
  - continuous annulus counts
  - 84-donor/44-target construction
  - neighbor expansion, CV folds, and one-SE rule

Questions to answer:

1. Is the profiled Poisson likelihood correct for cells with different annulus
   masks and shared shape?
2. Are exact rectangle integrals used consistently at shifted boundary pixels?
3. Does the positivity constraint cover the entire physical domain, including
   boundary-pixel support, without biasing curvature unnecessarily?
4. Is the one-SE comparison valid when pooled donor cells have very different
   event counts?
5. Are zero-normalization donors excluded from shape information correctly?

### P1: Uncertainty and Stage F

- `apply/tools/bootstrap_v6_poisson_background.py`
- `apply/stages/06_fit.py`
  - `generalized_chi2`
  - `--excess-covariance-npz`
  - cell-order alignment and Cholesky objective
- `apply/tests/test_fit_full_covariance.py`

Questions to answer:

1. Does the parametric bootstrap reproduce all relevant uncertainty, or does it
   condition on a selected model/order in a way that understates uncertainty?
2. Is `diag(N_on)+Cov(B_on)` the correct covariance under the disjoint on and
   training regions actually used?
3. Are shared-shape cross-cell correlations preserved through Stage F?
4. Are covariance-aware pulls and goodness-of-fit labels interpreted correctly?

### P2: Convergence and reporting

- `apply/report/build_v6_poisson_grid_convergence.py`
- `apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py`
- `apply/tests/test_poisson_grid_convergence.py`
- `scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch`
- `scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch`
- `scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch`

Questions to answer:

1. Are the 12 branches identical except for registered resolution/phase?
2. Are `B_on` and pull envelopes calculated against nominal with the intended
   error denominator?
3. Are phase RMS and resolution envelopes separated correctly?
4. Is the finalizer's nonzero exit unambiguously scientific-gate failure?
5. Could any report/validator path accidentally read a legacy artifact?

## Evidence and artifact paths

In the clean ETO worktree or repository root:

- final report:
  `apply/report/crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_poisson_pooled_report.html`
- frozen manifest:
  `apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json`
- convergence evidence:
  `scheme_R_double_rayleigh_poisson_grid_convergence.json`
- complete validation:
  `scheme_R_double_rayleigh_poisson_pooled_validation.json`
- bootstrap covariance:
  `scheme_R_double_rayleigh_poisson_background_covariance.npz`
  and `.json`
- four-contract comparison:
  `apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-pooled/scheme_R_double_rayleigh_poisson_pooled_comparison.json`
- CV summary:
  `apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-pooled/cv_deviance_table.csv`
- original implementation plan:
  `apply/docs/superpowers/plans/2026-07-16-poisson-pooled-background-grid-convergence.md`
- broad project operations handoff:
  `apply/codex-claude.md`

## Lightweight reproduction commands

Use the clean worktree:

```bash
cd /mnt/mydisk/server/projects/energy-reconstruction-poisson
git status --short
git rev-parse HEAD origin/main
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
export PYTHONPATH=/mnt/mydisk/server/projects/energy-reconstruction-poisson:${PYTHONPATH:-}
```

Expected: clean status and both SHAs equal `9037544...`.

Run local-size tests only:

```bash
python -m unittest discover -s apply/tests -v
python -m py_compile \
  apply/stages/04_background.py \
  apply/stages/06_fit.py \
  apply/stages/poisson_roi_background.py \
  apply/tools/build_v6_poisson_pooling_manifest.py \
  apply/tools/bootstrap_v6_poisson_background.py \
  apply/report/build_v6_poisson_grid_convergence.py \
  apply/report/validate_v6_scheme_r_double_rayleigh_poisson_pooled.py
bash -n scripts/slurm/build_v6_64748_poisson_pooling_manifest.sbatch
bash -n scripts/slurm/run_v6_64748_scheme_R_poisson_grid_branch.sbatch
bash -n scripts/slurm/bootstrap_v6_64748_scheme_R_poisson_background.sbatch
bash -n scripts/slurm/finalize_v6_64748_scheme_R_poisson_pooled.sbatch
bash -n scripts/slurm/submit_v6_64748_scheme_R_poisson_pooled.sh
```

Expected: 66 tests pass; compilation and shell checks exit zero.

Inspect registered evidence without recomputation:

```bash
jq '.passed, [.checks[] | select(.passed == false)]' \
  scheme_R_double_rayleigh_poisson_grid_convergence.json
jq '{donors:(.donor_universe_cell_ids|length), targets:(.target_cell_ids|length), manifest_sha256}' \
  apply/config/cell_background_pooling_v6_64748_nhit100_reselect44_double_rayleigh_poisson.json
jq '.contracts' \
  apply/report/assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R-poisson-pooled/scheme_R_double_rayleigh_poisson_pooled_comparison.json
```

Running the all-phase validator with `--require-report` is expected to return
nonzero because the registered convergence artifact is `passed=false`. Do not
report that expected exit as a new engineering failure.

## Recommended next scientific work, pending review

Do not implement these before completing the review.

1. Run analytic-WLS through the same 12 event-level branches and score both
   estimators on identical held-out annulus folds using Poisson deviance. This
   is the missing fair empirical comparison.
2. Test an unbinned Poisson point-process fit for the seven quadratic cells.
   Keep the `0.1 deg` map only for diagnostics; continuous event coordinates
   remove grid phase from the shape fit.
3. If binning is retained, make curvature acceptance conditional on an
   independent stability validation. Do not simply relax the existing gate.
4. Reparameterize the quadratic into radial trace
   `(cxx+cyy)` and anisotropic components. A centered circular `B_on` depends
   only on the intercept and radial trace, so pooling or shrinkage can target
   the quantity that changes the aperture integral.
5. Revisit pooled CV aggregation with both per-event and equal-donor weights so
   high-count donors cannot dominate order selection silently.
6. Validate on blank-sky pseudo-sources and injected sources. Require calibrated
   significance tails, unbiased `phi0/alpha/beta`, and correct 68%/95%
   coverage before promoting a background model.
7. Use a new grid phase set or time split for the next acceptance run; do not
   tune on the existing 12 branches and then claim independent validation on
   those same branches.

## Prohibited shortcuts

- Do not lower the `0.5 sigma`, `90% at 0.25 sigma`, or `0.5 pull` gates.
- Do not choose the branch with the lowest Crab Stage F chi-square.
- Do not overwrite the current report or any existing Stage D-G namespace.
- Do not change the response, PSF, selector, containment, and background model
  in the same experiment.
- Do not convert the intentional finalizer failure to success without preserving
  a separate scientific convergence status.
- Do not include unrelated dirty production files in a commit.
- Do not run large event scans interactively or on the local Mac.

## Reader self-check

Before returning the review, confirm the handoff lets you answer all of these:

1. Which commit is authoritative, and why is the production checkout behind?
2. Why did finalizer job 65249 fail even though all computations completed?
3. Which cells are quadratic, and which dominate grid instability?
4. What changed between pixel-WLS, analytic-WLS, and pooled Poisson?
5. Why is Stage F chi-square not the primary model-selection metric?
6. Which files implement positivity, pooling, covariance, and convergence?
7. Which registered limits must remain unchanged?
8. What fair comparison is still missing?

If any answer is ambiguous, identify that ambiguity as a handoff/documentation
gap before reviewing implementation details.
