# January v6 continuous-disk nonnegative weighted-LS branch

This is an independent constrained analysis, not a reproduction of the legacy unconstrained algorithm. The old failed January output/report remains untouched. No current/latest link is promoted.

## Frozen scientific definition

Six coefficients in the original tangent-plane degree coordinates, original weighted-LS objective `sum((z-q)^2/max(z,1))`, original pixel-annulus normalization `sum(z)/sum(max(q_pixel,0))`, original analytic circular ON integral. Require nonnegativity throughout the continuous radius-6-degree disk; zero contact is allowed. Radius8 remains coverage diagnostic. Constrained fitting never clips the solved polynomial, changes the objective or adds a positive epsilon. The denominator retains its original max operation; its effect is quantified.

The new solver normalizes coordinates to a unit disk and density to the annulus mean. SLSQP uses analytic objective/constraint derivatives and an exact quadratic disk-minimum oracle (interior/boundary stationary candidates). `ftol=1e-12` is dimensionless in normalized objective space. Numerical nonnegativity tolerance is `1e-10 * mean_annulus_counts_per_pixel`, in counts per original 0.1-degree pixel. Normalization and finite positive analytic B_on remain mandatory. Unconstrained-feasible solutions return the original numpy least-squares coefficients unchanged.

Use the exact frozen selector (44 ordered IDs, SHA256 c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6), double Rayleigh apertures, Scheme R response and fixed1 signal containment. No Poisson estimator, pooling, inference, time alignment, cell reselection, aperture adjustment, v7 or Pass5 common-GTI analysis.

## Entry points and scope

- `apply/stages/nonnegative_quadratic.py`: constrained weighted-LS fit and old-normalization/analytic integration.
- `apply/tools/january_v6_nonnegative.py`: source/hash gates, independent C recount, all91 background diagnostics, fixed44 D/E, background bootstrap and independent truth coverage, two error-definition F/G comparison via existing functions.
- `apply/config/january_v6_nonnegative_pseudoexperiments.json`: design frozen after real-data/Poisson smoke and before any spectral results.
- `apply/tools/january_v6_nonnegative_spectral_toys.py`: conditional PL/LogPar/SED propagation ensembles.
- `apply/report/build_january_v6_nonnegative_report.py`: reviewed-source all91 recomputation, receipts, scientific plots and comprehensive HTML via the deployed notion-html-report renderer.
- `apply/tools/january_v6_nonnegative_delivery.py`: actual completed Slurm states, exact committed Git source blobs, dataset/artifact hashes and HTML links/anchors.
- `scripts/slurm/january_v6_nonnegative.sbatch`: CPU-only smoke/de/uncertainty/spectra/spectral-toys/report/verify jobs with explicit afterok submission gates.

Legacy stages and entry points are unchanged. Candidate failures remain explicitly separated from fixed44 acceptance. All input/generated arrays and reports stay in Git-ignored output directories.

## Uncertainty audit and frozen experiment

Old direct-expectation Stage E/F/G variance is N_on+B_on. The alternative replaces its B_on term with Var_boot(Bhat): N_on+Var_boot(Bhat), under disjoint ON/annulus and independent-cell Poisson assumptions. Adding the bootstrap variance to the old expression would double-count background variance. The old inverse-Hessian covariance scaled by chi2/ndof is not used as the constrained uncertainty.

Each observed cell gets512 plug-in Poisson background draws, refitting weights and normalization. Report normal intervals using the bootstrap SD; lower interval endpoints are truncated to zero without modifying the fitted surface. Independent constant, positive tilted quadratic, interior-zero and boundary-zero truths use N_ann=10/1000, 256outer and128inner draws, cell62 geometry. Failures remain in the coverage denominator. Report bias/RMSE, boundary/failure rates, 68%/95% coverage and Monte Carlo SE. Predeclared gates include relative bias <=10%; complete criteria are in the design JSON.

PL and LogPar reference truths are fixed before spectral results, each256 realizations with independent Poisson ON and disjoint saved background draws. Both error definitions refit both models and rebuild SED. This is conditional propagation using the observed variance estimates, not full nested spectral coverage certification. Coverage testing one aperture geometry does not certify all44 geometries. A failed or limited uncertainty calibration leaves all spectra/SED diagnostic-only, irrespective of numerical positivity or optimizer convergence.

## Actual execution and provenance

ETO base commit81b8a96 includes f979c5b, dd8b060 and7d46912. Initial working tree is clean. The Mac apply directory has no Git metadata; local code is edited/tested there and transferred by an explicit source-only rsync file list, preserving remote generated artifacts. Local devlog contains extra earlier-task entries: they are preserved locally; the remote log is updated from its own original text to avoid committing unrelated changes.

Cluster main/default-safe:88 configured CPU,170000MiB node memory. Request16CPU/24GiB, noGPU. Existing unrelated five v7 dependency jobs are preserved.

Old root: `/mnt/newdisk/shijie/energy/apply/output/v6_64748_202201_scheme_R_double_rayleigh_analytic_bon/`.
New root: `/mnt/newdisk/shijie/energy/apply/output/v6_64748_202201_scheme_R_double_rayleigh_analytic_bon_nonnegative6/`.

-66091: failed smoke1:0, historical receipt path was apply-relative while launch cwd was repository root; preserve the empty failed-run directory and logs. A failing regression then explicit path-resolution fix addresses the root cause.
-66092: smoke completed0:0 in11s,21fixtures; hashes all681 parquet parts and key inputs, rebuilds the full21,559,258-event ROI map and independently recounts spherical ON, equal to accepted cache. No source inference/alignment rerun.
-66093: D/E completed0:0 in3s. All44 required cells numerically pass. Constraints change cells48/62/70; other41 retain the original feasible weighted-LS optimum. Cell62 B_on changes from-0.8789260838758293 to0.22988897486089355; minimum6=-1.3877296384733443e-15 in tolerance. Negative-excess cells4/5/6/21/75 remain.
-66094: production uncertainty job submitted afterok66093; all44×512 conditional bootstrap draws have zero fit failures. Independent nested coverage continues. Its exact deployed source is retained at `.recovery/2026-10-15/nonnegative6-source/66094/`.

The design SHA256 is bf3badf99eb5f283a9a07bbc4918e634469045c08877ff71ebe03b9ce463b634, fixed before F/G. Source snapshots are kept per later Slurm job. The report re-executes all91 backgrounds from the final committed source and verifies identical statuses/numbers, preserving the distinction between original generation and final source verification.

Targeted tests currently23pass, including feasible equivalence, constrained-negative fit, exact active oracle internal/boundary/zero/grid counterexample cases, independent quadrature integral, optimizer/input/normalization failures, historical relative paths, invalid-replicate denominators, selector/response contracts and original analytic integration. Independent code review checks variance replacement, F/G APIs and acceptance semantics; identified all91 exception-scope defect is corrected before execution.

Final actual jobs, science conclusions, hashes and delivery commit are appended after production and independent delivery verification. No scientific pass is inferred from a zero job exit or existing output.
