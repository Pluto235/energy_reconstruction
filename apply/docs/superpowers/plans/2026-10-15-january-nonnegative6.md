# January v6 nonnegative6 execution plan

**Goal:** execute the frozen January analysis with continuously nonnegative quadratic weighted-LS backgrounds on ETO; independently assess uncertainty calibration.
**Architecture:** a dedicated constrained fitter and January entry reuse Stage D geometry/integral, accepted Stage C/PSF/exposure/response and Stage F/G. No legacy default changes or promotions.
**Stack:** existing numpy/scipy/iminuit/pyarrow, CPU Slurm main/default-safe.

- [x] Add failing fixtures for feasible LS equivalence, negative LS refit, interior/boundary/touching minima, between-grid counterexample, integrals and explicit failure.
- [x] Implement six-variable constrained weighted-LS with exact continuous-disk minimum, scale-relative tolerances in counts per 0.1-degree pixel, unchanged clipped-pixel annulus normalization and analytic disk integral.
- [x] Bind existing receipt, all parquet parts, GTI, PSF, nominal/Scheme R response and cached diagnostic maps. Recount spherical N_on from accepted C during Slurm smoke. Preserve all 91 cell IDs; scientific gates concern ordered frozen44 only.
- [x] Run fixtures and real cell62 / other-cell and Poisson smoke on Slurm. Measure cost before freezing production design.
- [x] Before spectral results, save design: plug-in parametric bootstrap background intervals/variance; independent constant, positive tilted quadratic, interior-zero and boundary-zero truth scenarios, both low and higher annulus counts. Save bias, RMSE, boundary/failure rates, 68/95 coverage and binomial MC error; failures count against coverage. Use nested bootstrap for per-replicate reported intervals.
- [x] Submit D/E, uncertainty batch, then F/G with afterok gates. Preserve sqrt(N+B) comparison; corrected errors replace B by Var(Bhat), never add both. Keep statistical sqrt(N) comparison diagnostic.
- [x] Report per-cell old/new coefficients, B, minima, objectives, normalization and uncertainty; spectra/SED/pulls and error sensitivity; separate data, numerical, background, uncertainty and fit-quality conclusions. If calibration fails retain diagnostic spectrum and fail scientific acceptance.
- [x] Verify actual sacct exit status/resources, hash artifacts and exact deployed source/Git blobs, write receipt/report, devlog, commit and push through existing local Git bridge.

**Predeclared limits:** independent cell Poisson pixels model the old tangent-plane map; no detector-systematic uncertainty, independent hardware deadtime, exposure/MC uncertainty or universal unbiasedness claim. Bootstrap normal intervals may fail at low counts/boundaries; that is a scientific result, not a reason to tune truth scenarios or delete cells.

Execution complete: science gates evaluated and failed explicitly; diagnostic products and verified receipts delivered. See JANUARY_V6_NONNEGATIVE6_EXECUTION.md for actual jobs, source commits and hashes.
