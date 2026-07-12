# apply devlog

Lightweight running log of Codex-mediated modifications.
Newest at top. Format: `- **<version>** · YYYY-MM-DD HH:MM · [<sha>] <summary>`.
Detailed design rationale belongs in `report/` or `docs/`. Authoritative history belongs in `git log`.

---

- **energy** · 2026-07-12 14:58 · [uncommitted] Add an aggregate v6 Dec profile comparing observed counts, fitted background, and post-subtraction excess.

- **energy** · 2026-07-12 14:50 · [uncommitted] Match the v6 Nhit-grouped SED points to the blue Stage F LogPar curve.

- **energy** · 2026-07-11 12:25 · [uncommitted] Add the established Pass5 point-fit LogPar curve to the v6 reselect44 Stage G SED overlay.

- **energy** · 2026-07-11 12:04 · [uncommitted] Remove only v6 predE-grouped points from the reselect44 Stage G external-reference SED overlay.

- **energy** · 2026-07-11 11:08 · [uncommitted] Archive the active v6 64748 model resolution, RMS, bias, loss, metrics, and training configuration.

- **energy** · 2026-07-11 10:59 · [uncommitted] Add normalized v6 64748 true-energy overlays for the final 39-cell 2D selector.

- **energy** · 2026-07-10 15:30 · [uncommitted] Add complete per-cell raw MC theta profiles with missing-mass labels and fit-cell shading to the v6 64748 report.

- **energy** · 2026-07-10 13:50 · [uncommitted] Add per-panel cell-number labels to the v6 64748 Stage B radial PSF grid.

- **energy** · 2026-07-10 13:43 · [uncommitted] Rebuild the v6 64748 report with fit-cell shaded PSF profiles and an external-reference SED overlay.

- **energy** · 2026-07-09 23:50 · [uncommitted] Complete the v6 `_64748` nhit100 highEplus1 Stage A-G Slurm chain, selectors, validation, and final report.

- **energy** · 2026-07-09 20:43 · [uncommitted] Complete the v6 `_64748` split56 half-year observation apply chain, storage audit, final reports, and 64670 Jan-Feb retention boundary.

- **energy** · 2026-07-09 00:22 · [uncommitted] Add Codex/Claude handoff documenting servers, data paths, workflow, and core code roles.

- **energy** · 2026-07-08 16:44 · [4162c63] Add the v5 PSF double-Rayleigh mixture branch, Slurm pipeline integration, diagnostics, and regenerated comparison report.

- **energy** · 2026-07-02 15:27 · [a72baa0] Add the v6 `_64670` split56 baselinev4 selector, Slurm apply chain, validation, and final report.

- **energy** · 2026-07-02 11:26 · [d07f855] Add fit-cell shaded Stage B radial PSF grid to the v6 `_64670` baselinev4 report.

- **energy** · 2026-07-02 09:04 · [uncommitted] Complete the v6 `_64670` half-year Crab Stage A-G rerun and add the final baselinev4 diagnostic report.

- **energy** · 2026-07-02 07:50 · [uncommitted] Shorten the v6 `_64670` Stage C-G chain time limit after a successful Stage C smoke test.

- **energy** · 2026-07-02 03:43 · [uncommitted] Shorten the v6 `_64670` Stage B PSF job time limit after validating the nominal Stage A response.

- **energy** · 2026-07-02 01:40 · [uncommitted] Shorten the v6 `_64670` Stage A response job time limit to improve Slurm backfill before resubmission.

- **energy** · 2026-07-01 19:53 · [uncommitted] Shorten the v6 `_64670` Phase 2 prepare job time limit to allow Slurm backfill without changing inputs.

- **energy** · 2026-07-01 11:19 · [14c2e8c] Move the v5 PSF Stage F pull-grid colorbar into a dedicated column so it no longer overlaps the observed-data panel.

- **energy** · 2026-07-01 10:34 · [373a6e1] Add Rayleigh reference curves to the observed-data v5 PSF profile grid and clarify the caption.

- **energy** · 2026-06-30 22:23 · [123f38e] Rebuild v4 split56 ridge-only baseline and compress the v4 report around the current Stage A-G results.

- **energy** · 2026-06-30 21:54 · [cccb7c6] Add observed-data v5 PSF aperture branch with MC-ratio guard and four-branch comparison report.

- **energy** · 2026-06-30 20:51 · [uncommitted] Add v5.4 de-circularized upgrade: K rebuilt from MEASURED data/MC predE moments (0 free params, shift+broaden, self-consistent with v5.3 fitted Delta) and per-cell significance-gated de-Poissoned background systematic (replaces row-pool); off-source null tests pass (leave-one-out held-out pull RMS 1.07; systematic absorbs the -26σ off-source background failure to -1.8σ); cell chi2/ndof 5.10→1.06 with 7-bin SED invariant (rows conserved to 2e-16).

- **energy** · 2026-06-30 20:37 · [uncommitted] Add v5.3 two-component instrument-side cell-level fix (Nhit-row-dependent row-conserving dispersion kernel K for high-Nhit migration + de-Poissoned off-source-calibrated background systematic for low-Nhit); validated forward-fold reproduces Stage F 117.3/23, full fix drives cell chi2/ndof 5.10→0.96 with the 7-bin Nhit-marginalized SED provably invariant (rows conserved to 2e-16) and MC untouched.

- **energy** · 2026-06-30 20:35 · [uncommitted] Add v5 cellscheck phase-0 triage + v5.2 K demo: localize Stage F cell chi2≈5 to two orthogonal causes (high-Nhit predE energy migration; low-Nhit annulus-background failure, cell 15 worst), ruling out PSF/containment/spectral/aperture/range/MC-stats via instrument-side proxies only.

- **energy** · 2026-06-30 15:38 · [04659e7] Add final Nhit flux point diagnostic tables to the v5 PredE binning report.

- **energy** · 2026-06-29 19:21 · [uncommitted] Add v6 `_64670` MC provenance, drop4 selector enforcement, and staged Slurm wrappers.

- **energy** · 2026-06-29 17:31 · [856e62e] Add v6 roadmap for rerunning apply on the `_64670` half-year dataset.

- **energy** · 2026-06-29 15:40 · [b8d0f98] Promote v5 migration 12-bin flux points into a final SED report.

- **energy** · 2026-06-29 14:20 · [12b6235] Keep v5 cell-selection pull-grid colorbar outside plot panels.

- **energy** · 2026-06-29 13:52 · [uncommitted] Restore Rayleigh fit overlays in v5 PredE PSF radial profile grids.

- **energy** · 2026-06-29 12:19 · [uncommitted] Increase SED overlay color contrast between baseline_v4 and official pass5 curves.

- **energy** · 2026-06-29 11:15 · [uncommitted] Add fit-cell shaded Stage B PSF radial profile grids to the v5 PredE binning report.

- **energy** · 2026-06-29 11:02 · [uncommitted] Add official pass5 point-fit LogPar curve to the v5 PredE binning SED overlay.

- **energy** · 2026-06-27 17:06 · [uncommitted] Add v5 PSF comparison pipeline outputs, Stage F/G three-branch diagnostics, and final HTML report.

- **energy** · 2026-06-27 14:00 · [uncommitted] Isolate `_64670` half-year eval output, add resumable observation inference, and sync complete IHEP filtered days.

- **energy** · 2026-06-27 11:24 · [1c8ad60] Switch half-year Crab observation workflow to the no-core-cut 64670 inference model.

- **energy** · 2026-06-27 11:13 · [3143ab8] Add v5 cell-selection selectors, Stage F/G comparison workflow, and final SED report.

- **energy** · 2026-06-27 10:47 · [a7ac3ab] Add Stage F covariance SED bands and LogPar correlation matrix to the v4 baseline report.

- **energy** · 2026-06-26 16:45 · [d3075c7] Clarify batch-wise eval retention strategy for the half-year Crab workflow.

- **energy** · 2026-06-26 15:46 · [a4396b6] Add storage-risk mitigation plan for the half-year Crab observation workflow.

- **energy** · 2026-06-26 15:23 · [61cb6fb] Add half-year Crab observation workflow spanning IHEP filtering, time recovery, ETO inference, and Stage C/G products.

- **energy** · 2026-06-26 14:24 · [4c21ea9] Replace v4 Stage G primary SED figure with response-contract overlay.

- **energy** · 2026-06-26 11:40 · [9c6f07b] Reorganize v4 Crab SED report into baseline-first structure with diagnostics appendices.

- **energy** · 2026-06-25 17:45 · [c17ca12] Add v5 MC-migration binning comparison report with 12-bin and 7-bin SED overlays.

- **energy** · 2026-06-24 14:03 · [b65df22] Add response-informed v4 rebinning diagnostics for candidate predE grouping.

- **energy** · 2026-06-24 13:23 · [10aaea2] Add v4_supercell repair report with super-cell fit, response-morph profile, and binning diagnostics.

- **energy** · 2026-06-23 21:06 · [f96d8b6] Add v4 residual-source ablation diagnostics for response migration, background scale, and super-cell checks.

- **energy** · 2026-06-23 13:39 · [e2dce69] Add v4 cell-level root-cause localization crossmatch to the report.

- **energy** · 2026-06-23 13:00 · [22d6b61] Add v4 response-contract SED overlay with official, v0.99, H.E.S.S., and MAGIC references.

- **energy** · 2026-06-23 00:00 · [11a80e1] Promote aperture-conditioned response branch to the primary v4 report result.

- **energy** · 2026-06-22 23:46 · [9d2a433] Add aperture-conditioned Stage A response contract check for v4 double-containment audit.

- **energy** · 2026-06-22 22:08 · [c6e870a] Add v4 empirical r68 aperture control and SED comparison.

- **energy** · 2026-06-22 21:51 · [4a7f379] Add v4 response containment audit for low-energy SED bias.

- **energy** · 2026-06-22 21:35 · [5ac07c2] Switch v4 Stage B PSF fit-cell highlighting from orange to green.

- **energy** · 2026-06-22 21:28 · [b882e18] Highlight current v4 fit cells on the Stage B candidate radial PSF grid.

- **energy** · 2026-06-22 20:57 · [d6180ba] Add v4 low-energy SED root-cause diagnostics.

- **energy** · 2026-06-22 20:36 · [f64499c] Add v4 drop4 cell-selection bias control report and diagnostics.

- **energy** · 2026-06-22 19:59 · [1e2459d] Remove borrowed-cell red styling from the direct PSF profile diagnostic.

- **energy** · 2026-06-22 18:54 · [787ebfe] Switch the v3 annnorm report to direct own-cell PSFs for cells 39/52/65.

- **energy** · 2026-06-22 16:08 · [85eab87] Remove the unnormalized active PSF radial profile figure from the v3 report.

- **energy** · 2026-06-22 15:51 · [3bb1c76] Add v4 annnorm official pass5 forward-fold report and diagnostics.

- **energy** · 2026-06-22 15:35 · [e445b21] Restore Stage B PSF diagnostics in the latest v3 background report.

- **energy** · 2026-06-22 15:08 · [25905f2] Limit the v3 annnorm before/after Dec profile comparison to ±6 degrees.

- **energy** · 2026-06-22 14:51 · [5f8881e] Restore v3 RA/Dec profile diagnostics using the latest annulus-normalized background maps.

- **energy** · 2026-06-22 14:07 · [6b6e8a1] Rewrite v3 integrated report around the latest annulus-normalized background result.

- **energy** · 2026-06-19 14:35 · [1d1f04a] Add v3 annulus-normalized derived background run and report comparison.

- **energy** · 2026-06-19 14:02 · [60a69f3] Add v3 annulus-normalized background option and report diagnostics.

- **energy** · 2026-06-19 11:50 · [843db7e] Document the v3 Stage D fitted 2D background surface formula in the report.

- **energy** · 2026-06-19 11:46 · [c46b281] Mark Stage D annulus bands on v3 Dec count profile diagnostics.

- **energy** · 2026-06-19 11:42 · [c093a93] Restore raw summed Dec profiles in the v3 background comparison figure.

- **energy** · 2026-06-19 11:35 · [2e1584a] Mark Stage D annulus bands on v3 RA profile diagnostics.

- **energy** · 2026-06-18 14:02 · [020306a] Add v1-bin Nhit-only control workflow, results, and SED overlay comparison.

- **energy** · 2026-06-17 23:19 · [27e1047] Add Nhit-only SED comparison overlay to the v3 integrated report.

- **energy** · 2026-06-17 22:40 · [71079f1] Add Nhit-only control reports and final v3 diagnostic conclusion.

- **energy** · 2026-06-17 21:44 · [d8b6c39] Add PredE-blind Nhit-only control section to the v3 integrated report.

- **energy** · 2026-06-17 20:55 · [948bef9] Add PredE-blind v3 Nhit-only control selector, cache builder, and Slurm runner.

- **energy** · 2026-06-17 20:14 · [9c8a82a] Explain PSF effective-events calculation in the v3 report.
- **energy** · 2026-06-17 19:48 · [83c8a67] Explain the r_opt and sigma relation in the v3 report.
- **energy** · 2026-06-17 19:26 · [6c611bf] Clarify PSF sigma marker interpretation in the v3 report.
- **energy** · 2026-06-17 17:54 · [699e533] Mark fit PSF sigma positions on active own-cell radial diagnostics.
- **energy** · 2026-06-17 17:03 · [2065cab] Overlay fit PSF curves on active own-cell radial diagnostics.
- **energy** · 2026-06-17 16:50 · [57defa5] Clarify active PSF borrowing sources in the v3 report.
- **energy** · 2026-06-17 15:26 · [a990891] Remove the unnormalized active PSF radial plot from the v3 report.
- **energy** · 2026-06-17 15:20 · [20d9940] Add active-cell MC theta profile diagnostics to the v3 report.
- **energy** · 2026-06-17 12:09 · [de14f47] Show own-cell normalized radial PSF diagnostics for borrowed v3 cells.
- **energy** · 2026-06-17 11:30 · [a38facc] Add normalized active PSF radial profile diagnostic to the v3 report.
- **energy** · 2026-06-16 15:51 · [0d89f8b] Add v3 official forward-fold and off-source normalization diagnostics to the integrated report.
- **energy** · 2026-06-16 15:26 · [4db3867] Update v3 selector sensitivity figures and tables around the active 30-cell PSF-borrow branch.
- **energy** · 2026-06-16 15:15 · [e444f80] Add active fit-cell PSF profile diagnostics and PSF summary table to the v3 report.
- **energy** · 2026-06-16 15:05 · [3e79cc2] Reorder v3 Stage D skymap diagnostics and remove duplicate baseline fit-cell maps.
- **energy** · 2026-06-16 14:56 · [1bd1be6] Restore the standalone official/tutorial WCDA SED comparison figure in the v3 report.
- **energy** · 2026-06-16 14:33 · [b6eeb4f] Overlay official pass5 and tutorial v0.99 points on the v3 Stage G SED plot.
- **energy** · 2026-06-16 13:45 · [a3da2b9] Clarify v3 30-cell selection in the integrated report.
- **energy** · 2026-06-16 13:26 · [3e1e1dc] Add tutorial v0.99 WCDA-only Crab SED points and overlay to the v3 integrated report.
- **energy** · 2026-06-16 12:20 · [876ef99] Add official WCDA pass5 Crab SED points and overlay to the v3 integrated report.
- **energy** · 2026-06-13 00:17 · [uncommitted] Add v3 PSF borrowing systematic Stage B variant, D-G Slurm chain, selector, and integrated report comparison.
- **energy** · 2026-06-12 23:38 · [7e04701] Document v3 cell 39 high-theta PSF support audit and repair options.
- **energy** · 2026-06-12 22:32 · [5c649b1] Document v3 PSF theta-support lessons in the integrated report.
- **energy** · 2026-06-12 22:02 · [bd043d4] Freeze v3 baseline selector at 30 MC-ridge cells with PSF follow-up annotations.
- **energy** · 2026-06-12 20:41 · [06455ab] Replace v3 hard-coded physical ridge with MC occupancy plus PSF-quality selector.
- **energy** · 2026-06-12 17:13 · [162fff4] Add explanatory captions to the v3 integrated report figures.
- **energy** · 2026-06-12 14:35 · [ff5a535] Add v3 full-cell Crab SED pipeline with 84 retained cells and fit-cell highlighting.
- **energy** · 2026-06-12 13:19 · [3198068] Add roadmap v3 MC normalized energy-distribution overlay diagnostic.
- **energy** · 2026-06-12 13:15 · [5182772] Set roadmap v3 mixed predE binning with low-end 0.5 dex and high-energy wide bins.
- **energy** · 2026-06-12 13:12 · [997fb6f] Clarify roadmap v3 high-energy predE binning strategy.
- **energy** · 2026-06-12 12:56 · [ff46d37] Refine roadmap v3 predE binning and accepted implementation decisions.
- **energy** · 2026-06-12 11:53 · [804c425] Add roadmap v3 for HAWC-style cell selection and annulus 2D surface background.
- **energy** · 2026-06-11 14:37 · [94e38db] Archive the LHAASO gamma source analysis guide Markdown under report references.
- **energy** · 2026-06-11 14:32 · [7446e96] Archive the LHAASO Data Analysis Tutorial Markdown under report references.
- **energy** · 2026-06-11 14:07 · [f150a1d] Overlay the 6-degree fiducial circle on the v2 counts skymap and remove the separate ROI crop figure.
- **energy** · 2026-06-11 13:16 · [5c0ac43] Switch v2 profile diagnostics to counts maps and add a 6-degree ROI counts skymap.
- **energy** · 2026-06-11 12:24 · [ade3eb0] Expand normalized profile diagnostics to all raw65 cells with v2_baseline24 fit cells highlighted.
- **energy** · 2026-06-11 11:22 · [3a6a1c6] Add normalized fit-cell excess profile diagnostics to the Stage A-G SED report.
- **energy** · 2026-06-11 11:12 · [28dadf5] Add v2_baseline24 fit-cell background-subtracted skymap to the Stage A-G SED report.
- **energy** · 2026-06-11 10:49 · [95c05ab] Add v2_baseline24 fit-cell counts skymap to the Stage A-G SED report.
- **energy** · 2026-06-11 10:35 · [9aad49e] Add v2_baseline24 selector and refreshed Stage F/G SED reports without cells 64/65.
- **energy** · 2026-06-10 18:00 · [uncommitted] Pull revised Stage G external-reference report and outputs from ETO and restore local current/latest links.
- **energy_reconstruction** · 2026-06-10 14:58 · [9af371b] Add MAGIC, H.E.S.S., and HAWC Crab SED reference overlays to Stage G diagnostics.
- **energy** · 2026-06-10 14:21 · [uncommitted] Pull revised Stage G pool1 reference report and outputs from ETO and restore local current/latest links.
- **energy_reconstruction** · 2026-06-10 13:47 · [uncommitted] Add WCDA-1 Pool-1 Table 1 reference points to Stage G diagnostic SED outputs and report.
- **energy** · 2026-06-10 09:59 · [uncommitted] Pull Stage G SED point outputs, report, and script updates from ETO and restore local current/latest links.
- **energy** · 2026-06-09 21:45 · [uncommitted] Pull Stage F baseline reports and outputs from ETO for local review.
- **energy** · 2026-06-09 18:12 · [429533d] Replace Stage D/E skymap quicklook subtraction with formal Stage D ROI excess maps.
- **energy** · 2026-06-09 17:15 · [f3bd5ef] Add v1 skymap diagnostics to Stage C/D/E reports and align Stage D with ROI-local background.
- **energy** · 2026-06-09 13:41 · [35379c6] Translate Stage E report to Chinese and expand methodology explanation.
- **energy** · 2026-06-09 11:42 · [uncommitted] Pull latest Stage E signal extraction outputs from ETO and repair local current/latest pointers.
- **energy** · 2026-06-09 11:13 · [uncommitted] Sync full report directory from ETO into local apply reports.
- **energy** · 2026-06-09 11:07 · [uncommitted] Pull latest Crab v2 cell skymap Markdown and HTML reports from ETO.
- **energy** · 2026-06-05 14:08 · [uncommitted] Expand Stage B report explanation of MC reco-true PSF and core Rayleigh fit.
- **energy** · 2026-06-04 11:27 · [uncommitted] Fix Stage B report diagnostic image paths for local Mac viewing.
- **energy** · 2026-06-04 11:21 · [uncommitted] Pull updated apply folder from ETO and restore local devlog tracking.
- **energy** · 2026-06-03 20:42 · [uncommitted] Add apply devlog and require devlog skill tracking for future changes.
