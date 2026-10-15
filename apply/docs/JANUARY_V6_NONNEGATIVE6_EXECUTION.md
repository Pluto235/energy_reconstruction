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
-66094: production uncertainty job submitted afterok66093; all44×512 conditional bootstrap draws have zero fit failures. Independent nested coverage completes as recorded below. Its exact deployed source is retained at `.recovery/2026-10-15/nonnegative6-source/66094/`.

The design SHA256 is bf3badf99eb5f283a9a07bbc4918e634469045c08877ff71ebe03b9ce463b634, fixed before F/G. Source snapshots are kept per later Slurm job. The report re-executes all91 backgrounds from the final committed source and verifies identical statuses/numbers, preserving the distinction between original generation and final source verification.

Targeted tests currently23pass, including feasible equivalence, constrained-negative fit, exact active oracle internal/boundary/zero/grid counterexample cases, independent quadrature integral, optimizer/input/normalization failures, historical relative paths, invalid-replicate denominators, selector/response contracts and original analytic integration. Independent code review checks variance replacement, F/G APIs and acceptance semantics; identified all91 exception-scope defect is corrected before execution.

Final actual jobs, science conclusions and hashes follow. No scientific pass is inferred from a zero job exit or existing output.


## Final execution and acceptance

All production, spectral propagation and report/verification stages complete on ETO Slurm:

| Job | Stage | State / exit | Elapsed | Allocation |
| --- | --- | --- | --- | --- |
| 66094 | background bootstrap + independent nested coverage | COMPLETED / 0:0 | 00:32:40 | 16CPU /24G |
| 66095 | PL/LogPar + SED, both error definitions | COMPLETED / 0:0 | 00:00:02 | 16CPU /24G |
| 66096 | 512 conditional spectral/SED realizations | COMPLETED / 0:0 | 00:00:56 | 16CPU /24G |
| 66097 | initial full report and final-code all91 recomputation | COMPLETED / 0:0 | 00:00:09 | 4CPU /8G |
| 66098 | initial exact-Git/artifact/HTML verification | COMPLETED / 0:0 | 00:00:05 | 4CPU /8G |
| 66099 | report refresh after browser-found table overflow fix | COMPLETED / 0:0 | 00:00:09 | 4CPU /8G |
| 66100 | final verification,23fixtures | COMPLETED / 0:0 | 00:00:05 | 4CPU /8G |

The initial66091failed smoke remains explicitly failed;66092/66093passed as above. Dependencies are afterok at every production transition. Report refresh uses afterok66098 then final verification afterok66099. The first verified report/receipts are preserved under `report_generation_66097/`, not overwritten without history. A live sstat sample for66094.batch at21:16 records AveCPU1:39:58 and MaxRSS1,280,528KiB; final sacct reports MaxRSS0, which is not interpreted as a measured zero peak. Existing unrelated five v7 dependency jobs remain untouched.

Runtime: `/mnt/newdisk/shijie/conda/envs/py310/bin/python`, Python3.10.22, numpy2.2.6, scipy1.15.3, pyarrow25.0.1, iminuit2.33.0, all BLAS/OpenMP threads1; noGPU. The Mac performs only source edits, small fixtures, checks, file transfer and report viewing.

### Scientific result

Data integrity, constrained numerical solve and positive finite analytic background integrals pass for all44 required cells. The final reviewed source reproduces every91cell status and valid-fit numerical value. Constraints change48/62/70. The other41cells agree with legacy B_on within max absolute3.1651e-10counts; denominator-clipping delta is0for all44 and maximum absolute annulus normalization residual is7.27596e-12counts.

Uncertainty calibration fails:6of8independent truth/count cases fail predeclared bias/coverage/failure gates. AtN_ann=10, 95%coverage is72.27%(constant),73.83%(positive tilt),84.38%(interior zero),53.52%(boundary zero), instead of nominal95%. AtN_ann=1000, interior-zero relative bias is+76.18%; boundary-zero95%coverage is77.73%. These conclusions are conditional on the stated truth families and cell62geometry.

The `failure_rate` in coverage.json is the fraction of outer trials lacking a complete valid point+interval result. All recorded failures in this run are explicitly `Nested bootstrap failures n/128`; the outer point fits themselves succeed in all2048trials. Low-count interval-construction failures are7.42%,8.59%,12.11%,10.55% respectively, and remain in the256-trial coverage denominators. This is distinct from claiming the outer optimizer fails at those rates. The observed44x512background bootstrap has0failures. Bias/RMSE are computed only where an estimator exists, with the failure fraction reported separately.

Cell62 constrained B_on=0.22988897486089355, bootstrapSD=0.5874212901668984, conditional plug-in bias=+0.33464241878155254, constraint-trigger fraction=95.703125%. Its normal95%diagnostic interval is[0,1.3812135473400668]. Positivity does not imply accurate low-count background or calibrated errors.

Both44cell F variants converge, retaining all cells and all negative excess. Both G variants contain18SED points:7Nhit+11predE groups, with every group retaining every assigned cell (`n_valid_cells==n_cells`). Source pivot, flux/logarithm/unit conventions, grouping and median-energy definitions remain frozen. Propagated errors replace B_on with Var_boot(Bhat), never add both terms.

| Errors | Model | phi0 (TeV^-1 cm^-2 s^-1 at3TeV) | Shape | chi2/dof | asymptotic p |
| --- | --- | --- | --- | --- | --- |
| legacy sqrt(N+B) | PL | 1.97679874e-12 | gamma2.62314963 | 92.7341/42 | 1.08456e-5 |
| legacy sqrt(N+B) | LogPar | 2.18296724e-12 | alpha2.73357438,beta0.12230474 | 74.6833/41 | 0.00101560 |
| propagated sqrt(N+VarB) | PL | 1.98959051e-12 | gamma2.61923560 | 108.0365/42 | 9.97326e-8 |
| propagated sqrt(N+VarB) | LogPar | 2.20989153e-12 | alpha2.74299486,beta0.13416977 | 80.6097/41 | 0.000217814 |

The512conditional spectral trials have0Minuit failures, but propagated nominal95%interval coverage is84.77%for PLgamma,82.42%for LogParalpha and82.03%for LogParbeta. Variance-only propagation does not remove estimator bias or guarantee spectral coverage. SED interval/bias/coverage summaries are saved for every group. They remain conditional on observed background variances and frozen per-fit spectral shapes, not a full all44nested coverage certification.

Final science status: `DIAGNOSTIC_ONLY_UNCERTAINTY_NOT_CERTIFIED`; asymptotic spectral goodness also fails. Diagnostic spectra/SED are delivered; no formal scientific pass or exact legacy reproduction is claimed. No result-driven cell deletion, radius change, alternative fit/model branch or variance inflation is introduced.

### Delivery and verification

New run contains `analysis_manifest.json`, `analysis_manifest_completed.json`, `background_all91.json`, two signal NPZs, background bootstrap NPZ/JSON, coverageJSON, PL/LogPar JSON+NPZ, two18point SED CSV/JSON sets,512spectral/SED trial records, four scientific PNGs, Slurm submission/accounting/logs, environment metadata, `report_202201_nonnegative6.html`, `final_acceptance.json`, `delivery_receipt.json`, and post-exit `delivery_completion.json`.

Report+verification scientific source commit is **8b425c9** (`january-v6-nonnegative6`), following92df798initialcode and62b43c8independent closed-form check/conditioning disclosure. The final documentation commit is a later descendant and does not relabel generation history. Exact deployed source files match committed Git blobs; the worktree is clean during66100. Fifty-three receipt artifacts rehash correctly on ETO and locally, all681parquet parts are reverified, and every HTML artifact link and anchor passes.

- HTML SHA256: `df21a2f6f0c25c36f4421fec3c831ff89ad4a3fd06c5d8e857e53746f9fc5474`.
- delivery_receipt.json SHA256: `c906f236c86ae6cdaa0f903b746b6f228ce9f51c8ff16d4658fc0a3bff7c1959`.
- delivery_completion.json SHA256: `c571c7ccc5ca3c4e05c398f1ffb04b54bcdac1ad815192cbcf305e7cef30e388`.

The post-exit completion file independently records66100COMPLETED0:0, all ten actual jobs including the failed smoke, live resource sample, final log hashes and desktop browser QA. It binds the receipt/report hashes and is itself bound by this committed document.

Desktop browser inspection confirms first-screen diagnostic labeling, anchor navigation, all four images loaded and tables constrained to820px with horizontal scrolling for1266pxcontent. Four scientific PNGs are visually checked. Mobile viewport behavior is not separately tested. The read-only independent final audit finds no must-fix scientific/data consistency issue. No heavy local computation, observation/MC inference, time alignment, full C rebuild, unrelated cancellation or current/latest promotion occurs.


## Original-v6-style reproduction report (2026-10-15)

User-requested presentation companion: `apply/report/crab_sed_v6_64748_202201_nonnegative6_reproduction_report.html`, generated by `apply/report/build_january_v6_reproduction_report.py`. Assets and validation live in `apply/report/assets/v6-64748-202201-nonnegative6-reproduction/`. The original report and every certified analysis artifact are preserved.

The companion reuses the specified original v6 HTML CSS and stage structure: executive checks, jobs, inputs, selector, analytic-background contract, baseline comparison, cell62, pulls, time audit, A/B MC diagnostics, D/E ROI maps and profiles, all44 signal rows, F fit tables, G SED/ratios, uncertainty calibration and final parameters. Its25 images include5 explicitly labeled historical MC/PSF figures, the existing January diagnostics, and newly rendered January maps/profiles/pulls/SED. The archived Stage G external references are visual comparisons only; its Pass5 reference helper refits archived reference points for the ratio graph, without reoptimizing January backgrounds or spectra.

The first rendering job66101 completed0:0 in2m14s (main/default-safe,4CPU,8GiB). Embedded self-checks verify all53 original delivery artifact hashes before/after, fixed44 identity/order, each SED grouping's complete44 coverage and7/11 point counts, sum(pull^2)=saved chi2 for both models/error definitions, and all local image/anchor links. Numerical results remain diagnostic-only with failed scientific calibration.

Direct browser preview of a file URL was rejected by the browser URL policy. No alternate browser route was used. Static HTML/link checks and visual inspection of key generated PNGs replace that preview; desktop/mobile rendered layout is not certified. The layout uses original CSS plus responsive figures and horizontally scrollable tables. Final generation provenance, exact source/input/image hashes and completed Slurm status accompany the report.

Committed-source figure render66102 completed0:0 in2m15s. A final HTML-only refresh66103 completed0:0 in2s from `8bcb9b5e61e78f814fba6813a01748fb0841e77f`, stripping trailing CSS whitespace inherited from the original template. The refresh verified all25 PNG hashes before bypassing plotting in process; all25 hashes remain identical to the visually inspected first rendering. Exact source Git blobs, local source/input/image hashes, PNG CRCs,17 unique anchors and all links pass. HTML SHA256 `2c3b000db72fd430505affa8a77789d6d6dd30c2f81e258edabf96bce8b6abeb`; validation SHA256 `ccff2d178c54fbcb8e309afbd8219b6cdf0aed6e4d9ebe62d629da020d41d08c`. The final completion receipt records all3 render jobs, figure generation and HTML refresh separately. Generated HTML/PNG/validation/completion files follow the original report directory convention and are committed with delivery. Large analysis arrays remain in ignored output directories.


### Stage G reference overlay follow-up

The user requested Pass5 and HSC points/curves and removal of the 1LHAASO WCDA full-array PL. The seven archived Pass5 points and the existing Stage G unweighted log-space LogPar point fit are added to both January SED overlay figures. Pass5 has no usable uncertainty column, so no error bars are invented. The full-array PL curve is omitted; all existing January points, Pool-1/MAGIC/H.E.S.S./HAWC references and the independent spectral results are retained. The dedicated report plot uses existing reference readers and flux functions; shared Stage G analysis code is untouched.

At the first Pass5 update, HSC identity awaited user clarification: the proposed source was the original v6 report's seven Official tutorial v0.99 WCDA points, associated with the archived hushicong guide. No HSC series was substituted before confirmation. Preview job66104 passed0:0 in2s, checking seven Pass5 points, its reference curve and absence of the full-array PL legend. Plotting is performed on ETO via Slurm.

Reference update66105 completed0:0 in4s from `bb101ea1ec8dae8528704f32ac68f32cd5fa791b`. Both SED overlays changed; the other23 image hashes and all53 original analysis artifact hashes are unchanged. Exact source Git blobs and all local input/image/link checks pass. New HTML SHA256 `7e6434cd222093c241bb7cbbc1c9723536efd1769ce0dc132b7d70fa18dab021`. HSC was pending source confirmation at that stage; the confirmation and follow-up are recorded below.


The user confirmed HSC as the original v6 Official tutorial v0.99 WCDA dataset. Both Stage G overlay figures now include its seven archived points with their supplied error bars and an orange dashed LogPar reference curve, explicitly labeled as an unweighted log-space point fit. Existing v6 readers and fit helpers are reused; the CSV is bound in the report input hashes. Pass5 points/curve remain and the full-array PL stays removed. Only the two overlays and report metadata/text are regenerated; January analysis is not rerun.

HSC update66106 completed0:0 in3s (2CPU/4GiB), generated from `a97548f8ee7aa6616068540d54fceb448f0f2471`. Both overlays were visually inspected; source Git blobs, all input/image hashes, PNG CRCs,17 anchors and25 image links pass. Only the two SED overlays changed; the other23 images and all53 original analysis artifacts remain identical on ETO and Mac. HTML SHA256 `fa28de94fb56670f3622a21bd04bafb56251fcd4382c36549f66d3f75f9d7ae2`; validation SHA256 `8a1abb3d08f82cd01b139d638578ea62883a86e345313bbd8f347f27366a3e28`; completion SHA256 `2ffa5a2ed7e1247f32753011196eb98c772c592e226b34bb6085132dacb63169`. HSC identification is resolved and the requested reference overlay update is complete; original scientific status remains diagnostic-only.


### Stage G blue covariance band follow-up

The original v6 blue shading is the Stage F LogPar pointwise nominal1sigma band. Reuse `sed_uncertainty_band` from the original report helper with the saved January propagated-error LogPar covariance in `(log10_phi0, alpha, beta)`. For L=ln(E/3TeV), g=(ln10,-L,-L^2), compute sigma_logSED=sqrt(g^T C g) including all off-diagonal terms, then SED*exp(+-sigma_logSED). No chi2/dof scaling, new fit or additional systematic term is introduced. Both January overlay figures gain the same blue alpha0.18 shading, labeled conditional1sigma. It is a pointwise conditional diagnostic band under the fitted Gaussian approximation, not a simultaneous band or a coverage-certified interval; the existing failed uncertainty calibration remains explicit.

The update touches only the report builder and its one local plot caller, two overlay PNGs, HTML, provenance receipts and documentation/devlog. Existing reference points/curves and the other23 images are preserved. Input checks require the known parameter basis, finite symmetric positive-semidefinite covariance, finite ordered positive bounds and identical band/line central values.

Band update66107 completed0:0 in3s (2CPU/4GiB) from `2ad8666ef11e92f3ff710dc0c0d664d6f2f37f73`. An independent central finite-difference Jacobian of the Stage G SED function reproduces both band bounds at seven energies0.12–260TeV; the3TeV pivot reproduces ln(10)*sqrt(C00). Both updated PNGs were visually checked. Exact source Git blobs, input/image hashes, PNG CRCs,17 anchors and25 image links pass; all53 original analysis artifacts and the other23 PNGs are unchanged on ETO/Mac. HTML SHA256 `fe14bd9e7142d64c04b8ff03408f4d43e886a1e296953cc592a2016e9076a93a`; validation SHA256 `7d3fbf540713ceec94923caeb360b74d888c822fa895df761671f9d649fe2568`; completion SHA256 `b5af7ac468a1f8db09a39741a57fdabba1711d1300f8ff32c90321f880340ad6`. No new scientific fit or coverage claim is made.
