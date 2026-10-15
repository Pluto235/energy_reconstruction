# apply devlog

Lightweight running log of Codex-mediated modifications.
Newest at top. Format: `- **<version>** · YYYY-MM-DD HH:MM · [<sha>] <summary>`.
Detailed design rationale belongs in `report/` or `docs/`. Authoritative history belongs in `git log`.

---

- **energy_reconstruction** · 2026-10-15 13:34 · [uncommitted] 根据确认来源补充HSC v0.99的7个SED点、误差棒和点拟合LogPar，更新两张Stage G叠加图及来源记录。
- **energy_reconstruction** · 2026-10-15 13:28 · [uncommitted] 更新一月Stage G叠加图：加入7个Pass5点及点拟合LogPar，移除full-array PL；HSC数据来源等待确认。

- **energy_reconstruction** · 2026-10-15 11:50 · [531ab91] 按原 v6 版式生成一月非负约束复现报告：17章节、25图、44-cell账本与18点SED；保留诊断状态并核对原交付哈希。

- **energy_reconstruction** · 2026-10-15 11:30 · [uncommitted] Complete nonnegative6 delivery66100:23tests,53artifact hashes,44cells/18SED points retained; document failed6/8uncertainty cases and poor spectral goodness, preserve diagnostic-only status and final report/receipt hashes.
- **energy_reconstruction** · 2026-10-15 11:27 · [uncommitted] Complete Slurm66094–66098 and independent diagnostic-result audit; fix browser-confirmed wide-table overflow before regenerating report and delivery hashes.
- **energy_reconstruction** · 2026-10-15 11:02 · [uncommitted] Verify constrained radial fixture against its independent closed-form boundary optimum and clarify successful-estimator conditioning of bias/RMSE in the uncertainty report.
- **energy_reconstruction** · 2026-10-15 10:55 · [uncommitted] Add independent January nonnegative6 weighted-LS solver,23fixtures,source gates,Slurm D/E and frozen bootstrap/coverage/spectral diagnostics; preserve legacy failure and all44 cells.
- **energy_reconstruction** · 2026-10-15 00:52 · [7d46912] Record completed delivery66090: thirteen tests, 681 hashed parquet parts, exact Git source bindings, report/receipt hashes and two independent reviews; retain scientific failure and blocked spectrum.
- **energy_reconstruction** · 2026-10-15 00:48 · [uncommitted] Verify January report66089, bind 44-cell diagnostic JSON to NPZ and Stage C, preserve scientific failure, and add final 681-parquet/source-commit delivery receipt with thirteen fixtures.
- **energy_reconstruction** · 2026-10-15 00:42 · [uncommitted] Execute frozen January v6 on ETO: 681-file C and GTI pass, rebuild exact 2R PSF, preserve cell62 negative analytic background and block formal E–G; add diagnostics, twelve fixtures, Slurm receipts and report source.
- **energy_reconstruction** · 2026-10-15 00:04 · [uncommitted] Align January prediction time friends by unique event identity; fixtures and Slurm66078/66080 pass 681 files/48726074 rows twice, account for 464 source anomalies (0 predicted), preserve full time sources and failed-scope evidence.

- **energy_reconstruction** · 2026-10-14 23:23 · [uncommitted] Record completed January four-GPU inference 66070 and full acceptance 66071: 681 files/31 days, 48,726,074 predictions, 17.46 GB, unchanged provenance; save measured runtimes/resources and stop the monitor.

- **energy_reconstruction** · 2026-10-14 11:33 · [uncommitted] Abandon two-GPU jobs and clean 48 old outputs (1.10 GB), pass four-V100 batch/sample validation 66068, and queue fresh January inference 66070 plus validator 66071 at batch 384 with preserved provenance and updated ETA/monitor.

- **energy_reconstruction** · 2026-10-14 10:41 · [uncommitted] Pause January inference 66063 in requeue-hold and hold validator 66064 for user GPU reset, pause automation, verify CUDA contexts released, and preserve outputs plus partial-file evidence for reviewed resumption.

- **energy_reconstruction** · 2026-10-14 09:45 · [uncommitted] Exclude time-recovery sidecars from direct MMDD inference discovery, pass five regressions, and relaunch gated two-GPU January inference/validation as 66063/66064 with batch 352 and explicit provenance transition.

- **energy_reconstruction** · 2026-10-14 02:02 · [uncommitted] Tune FP32 batch 352 to 13.35 GB reserved per V100, queue dual-GPU January job 66052 plus validation 66053 after day acceptance, replace pending single-GPU jobs and record measured ETA with updated follow-up.

- **energy_reconstruction** · 2026-10-14 01:33 · [uncommitted] Fix uproot 5.7 object-array output with explicit typed TTrees, pass 128-event 64748 smoke 66044 and two regression cases, start day acceptance 66045 and gate month/validation 66046-66047 with quiet background follow-up.

- **energy_reconstruction** · 2026-10-14 01:08 · [uncommitted] Add fixed January 64748 Slurm launcher for new py310/input/output paths, SHA-256/config logging and overwrite/concurrency guards; verify syntax, isolated execution and scheduler test-only without inference.

- **energy_reconstruction** · 2026-10-14 01:04 · [uncommitted] Reuse ETO Conda for new-disk Python 3.10 py310, install CUDA 12.4 Torch and project dependencies, and verify both V100s plus unchanged 64748 config/weights in Slurm 66040 without inference.

- **energy_reconstruction** · 2026-10-14 00:41 · [uncommitted] Archive nine AstroCluster energy recoveries (24,669 files / 16.95 GB), verify 50 checkpoints, restore runs and reconstruct 64748 config from matching snapshots; retain ETO source and document runtime gaps.

- **energy_reconstruction** · 2026-10-14 00:27 · [uncommitted] Extend the validated observation acquisition to isolated February manifests (28 days / 600 files), preserve January defaults and receipt guards, deploy the 5.5 GiB queue and submit eight IHEP producers with an independent ETO receiver.

- **energy_reconstruction** · 2026-10-13 23:52 · [uncommitted] Verify January acquisition completion at 22:53: 681 files / 31 dates / 151.096 GB on ETO, exact manifest and receipt reconciliation, empty staging and safe retirement of the post-output ROOT shutdown task.

- **energy_reconstruction** · 2026-10-13 20:46 · [uncommitted] Isolate repeat ROOT/CVMFS failures on lhws261, reschedule only two affected January shards and exclude both failed nodes for future matches; preserve their outputs and reclaim stale reservations safely.

- **energy_reconstruction** · 2026-10-13 20:10 · [uncommitted] Plan independent January time/J2000 recovery with legacy event-key reuse, missing-event completion, a bounded EOS queue and one-to-two-worker validation before parallel production.

- **energy_reconstruction** · 2026-10-13 20:05 · [uncommitted] Diagnose ROOT/CVMFS failures on lhws247, preserve task evidence and reschedule only four affected January shards with that node excluded; verify restored production, transfer receipts and cleanup on the other four workers.

- **energy_reconstruction** · 2026-10-13 19:42 · [uncommitted] Submit scratch-based recovery through IHEP hep_sub site policy as 3726350.0 and 3726352.0–3726358.0; retire the unmatched idle native-submit trial and verify rotating cleanup of received files.

- **energy_reconstruction** · 2026-10-13 19:38 · [uncommitted] Recover stalled January acquisition from IHEP home EDQUOT by moving verified controls to scratch, sharing the new path across callers, recalculating a 5.5 GiB rotating queue and submitting replacement shards 3726308.0–7.

- **energy_reconstruction** · 2026-10-13 17:29 · [uncommitted] Optimize January recovery with one cut scan, measured 32 MiB caches, durable hash-free receipts and a 5.75 GiB reserved queue; validate 4/8-worker cohorts and submit eight production shards.

- **energy_reconstruction** · 2026-10-13 15:35 · [uncommitted] Guard observation receipt cleanup against non-staging paths and symlinks; verify the first rebuilt full-hour files end to end on ETO.

- **energy_reconstruction** · 2026-10-13 15:19 · [uncommitted] Bound January observation staging to four files after IHEP quota exhaustion; fail on ROOT write errors and delete only temporary copies after ETO SHA-256 receipts.

- **energy_reconstruction** · 2026-10-13 14:35 · [uncommitted] Restore January observation acquisition with the verified 30-branch pincness contract, reuse Jan31 archive, submit 31 IHEP day workers, and gate ETO transfers on ROOT audits.

- **energy_reconstruction** · 2026-10-13 14:06 · [uncommitted] Verify completed MC sync to ETO: 10,000 expected eventout ROOT files, 174,198,529,036 bytes, rsync exit 0 at 13:54:27.

- **energy_reconstruction** · 2026-10-13 12:56 · [uncommitted] Start resumable direct IHEP-to-ETO synchronization of 10,000 MC eventout ROOT files (174.2 GB) into /mnt/newdisk/shijie/WCDA_simulation.

- **energy_reconstruction** · 2026-10-13 12:07 · [uncommitted] Restore GitHub main and Mac-only source changes to ETO /mnt/newdisk/shijie/energy; retain differing originals, merge devlogs, and relocate recovered output links.

- **energy_reconstruction** · 2026-07-31 16:20 · [uncommitted] Audit and finalize the Crab v6 versus Pass5 common-GTI covariance comparison with terminal manifests, live-time provenance, and qualified uncertainty reporting.

- **energy_reconstruction** · 2026-07-21 17:12 · [uncommitted] Refit Pass5 seven-bin double-Rayleigh PSFs from 10,000 official WCDA MC files and expose full-sample overflow containment.

- **energy_reconstruction** · 2026-07-21 14:20 · [d55cc9e] Fit seven predE-folded Nhit profiles with conditional double Rayleigh, improving KL by 10.5-80.3x over single Rayleigh.

- **energy_reconstruction** · 2026-07-21 (IHEP+ETO) · [uncommitted] Root-caused the reproducible ExitCode-34 failure (identical on both hep_sub attempts) by running `bash -x prepare_strict_pass5_fit.sh` interactively: the live-time ROOT one-liner correctly computes and prints `LIVE_DAYS=130.592212689228`, then the whole script dies silently right there under `set -e -o pipefail` -- `root -q`'s implicit end-of-macro TFile/TTree global-destructor cleanup (the macro never calls `f.Close()`) apparently returns nonzero even after the printf succeeds. Fixed by adding an explicit `gSystem->Exit(0);` after the printf. Verified by rerunning interactively on the login node (skips the hep_sub queue entirely): FINISHED_EXIT=0, MIGRAD converged, `covariance_status=3`, `common_gti_fit/covariance_fit.yaml` produced, pass5 live time 130.592212689228 d.
  **Pass5 vs v6 common-GTI covariance comparison is now fully complete on both sides.** Generated the final report (`report/crab_v6_vs_pass5_common_gti_covariance_report.html`, assets under `report/assets/v6-vs-pass5-common-gti-covariance/`): v6 chi2/ndof=20.94 (worse than the 149-day full-sample run's 11.7, smaller-sample same-bin-count effect), PDG scale factor sqrt(chi2/ndof)=4.58. Raw-HESSE joint error-volume ratio v6/Pass5=0.528 (v6 looks smaller); after chi2/ndof inflation the ratio flips to 50.6 (v6 looks much larger) and per-parameter reduction goes from {phi0:+25%, alpha:-8%, beta:+57%} raw to {phi0:-243%, alpha:-395%, beta:-99%} inflated -- the naive HESSE comparison and the goodness-of-fit-corrected comparison give opposite headline conclusions, exactly the literature-review caveat this task set out to surface. Synced final report assets and `common_gti_fit/` to ETO.

- **energy_reconstruction** · 2026-07-21 (IHEP) · [uncommitted] First `prepare_strict_pass5_fit.sh` submission (cluster 87712967) completed with ExitCode 34 and totally empty stdout/stderr -- `mkdir -p "$FIT_DIR"` ran (directory exists) but nothing after it produced a byte of output, so the failure is upstream of or inside the first `root -l -b -q` live-time extraction, with no diagnostic trail. Resubmitted as-is per user call (suspected transient environment issue rather than a code bug), cluster 87954887.

- **energy_reconstruction** · 2026-07-20 20:15 · [24f465f] Collapse 91 Stage-B cells over predE into seven sumw-weighted Nhit profiles with exact-bin Rayleigh diagnostics.

- **energy_reconstruction** · 2026-07-20 (IHEP) · [uncommitted] Merge confirmed clean (`STRICT_MAP_MERGE_COMPLETE maps=928`, one benign minAccCorr warning). Parameterized `prepare_strict_pass5_fit.sh`'s `MAP`/`FIT_DIR` (were hardcoded to the old full-sample `pass5_v6_sorted_gti_map.root` / `sorted_gti_fit`) per the handoff's guidance not to reuse that name for the 928-map common-GTI intersection. Submitted the official Pass5 gtselect/gtsrcmap/gtlike/gtlike_cov chain against `pass5_v6_common_gti_map.root`, output dir `common_gti_fit`, cluster 87712967, `-wt long`.

- **energy_reconstruction** · 2026-07-20 (IHEP) · [uncommitted] `-wt long` fixed the timeout: 28/28 completed clean, 928/928 nonempty J2000 maps, and `scan_j2000_integrity.C` confirms all 928 pass ROOT-level integrity (TOTAL=928 GOOD=928 BAD=0) -- the corrupted-recovery-output problem is fully resolved. Resubmitted `merge_strict_map.sh` (also `-wt long` this time), cluster 87689726.

- **energy_reconstruction** · 2026-07-20 (IHEP) · [uncommitted] The 94-chunk retry (cluster 87548556) itself hit every worker ExitCode 124 again (66/94 complete, 0/16 WORKER_DONE), leaving 28 chunks still missing (928 -> 900 nonempty). Two independently-sized arrays (128 workers/822 jobs; 16 workers/94 jobs) both died at ~6 processed chunks per worker regardless of worker count, pointing at a fixed `hep_sub` walltime class rather than per-job slowness. Found `hep_sub -wt {test,short,mid,long,special}`; `submit_strict_step2_recovery.sh` never set it (default class). Added `WALLTIME` (default `long`) and `-wt "$WALLTIME"` to the hep_sub call. Resubmitted the remaining 28 labels 1 worker per job, cluster 87657898, `-wt long`.

- **energy_reconstruction** · 2026-07-19 22:59 · [e207416] Fit cells 1-3 with one spherical King and reject it: KL is 10.7-11.4x double-King and Delta AIC is 1.65e4-5.49e4 worse.

- **energy_reconstruction** · 2026-07-19 13:08 · [88879d8] Add a two-iteration empirical-CDF Asimov aperture workflow with aperture-conditioned response rebuilding, Stage G overlay, and validated v6 report.

- **energy_reconstruction** · 2026-07-19 11:15 · [f6f5d86] Plot cells 1-3 double spherical-King fits with weighted core and tail density/survival components, exposing their crossover radii.

- **energy_reconstruction** · 2026-07-19 10:47 · [b4b05c2] Compare full-sphere Rayleigh+King, double-Rayleigh, and double spherical-King fits for cells 1-3, finding 2K best overall and 2R best in the extreme survival tail.

- **energy_reconstruction** · 2026-07-19 00:07 · [fde0419] Fit full-sphere raw-MC cells 1-3 with Rayleigh cores plus spherical-King tails, improving KL 1.4-3.5x while exposing boundary-limited far tails.

- **energy_reconstruction** · 2026-07-19 (IHEP) · [uncommitted] After the fixed merge script got past its precheck, `DI_Merge` itself segfaulted partway through 928 files with repeated `hon_N/hbkg_N/hoff_N is bad` errors. Added `scan_j2000_integrity.C` (opens every accepted J2000 URI via `TFile::Open`, flags ROOT's own "probably not closed" recovery warnings / zombie files / missing hMJD_all_fine / fewer than the expected 22 keys). Found 94/928 (10.1%) recovered maps were nonzero-but-truncated (55 unopenable zombies, 39 partial-key salvage) -- the same failure class the whole recovery effort exists to fix, this time from the recovery workers themselves, and invisible to every prior byte-size-only check, including `run_strict_step2_recovery_worker.sh`'s own idempotency skip (`remote_size -gt 0`). Removed the 94 corrupt EOS files first (otherwise the skip check would no-op on them), then added `build_targeted_step2_requeue.py` to hand-build a `step2_recovery_jobs.tsv` for just those 94 labels (reusing the hour-list relocation logic from `build_strict_recovery_manifest.py`, since that script's `classify()` can't see this failure mode). Resubmitted, cluster 87548556 (94 jobs, 16 workers). Next completion check must rerun `scan_j2000_integrity.C`, not just recount nonempty files, before merging again.

- **energy_reconstruction** · 2026-07-19 (IHEP) · [uncommitted] Fix `merge_strict_map.sh`'s pre-merge existence check, which used bash `[[ -s "$path" ]]` on `root://...` XRootD URIs and therefore always reported all 928 maps missing (cluster 87542452 died on this false negative before ever calling `DI_Merge`; EOS content itself was intact). New `map_exists()` dispatches to `env -u LD_LIBRARY_PATH /usr/bin/xrdfs <host> stat <path>` for `root://` entries, plain `-s` otherwise. Resubmitted the merge, cluster 87544183.

- **energy_reconstruction** · 2026-07-19 (IHEP) · [uncommitted] Pass5 EOS map recovery finished clean: reduced 16-worker Step2 array (cluster 87513951) completed 60/60 with ExitCode 0; rebuilt `strict_recovery_manifest.json` one more time to reach the expected final state (928 accepted / 150 rejected / 0 pending). Submitted `merge_strict_map.sh` with `MAP_LIST=strict_recovery/accepted_maps.list` (928 maps) -> `pass5_v6_common_gti_map.root`, cluster 87542452 (superseded, see fix above).

- **energy_reconstruction** · 2026-07-18 21:08 · [715437d] Add untruncated raw-MC survival diagnostics for cells 1-3, exposing 17.9-29.5% weighted probability beyond 5 degrees.

- **energy_reconstruction** · 2026-07-18 19:12 · [a50b205] Add Fermi-style double-King Stage B profile fitting and regenerate the 91-cell PSF grid with unchanged diagnostic and final-SED selectors.

- **energy_reconstruction** · 2026-07-18 12:27 · [9e3f2eb] Expand the Scheme R double-Rayleigh Poisson-unbinned prototype into a complete Stage A-G report with PSF, background, pull, bootstrap, time-split, and final SED figures.

- **energy_reconstruction** · 2026-07-18 01:20 · [uncommitted] Add a grid-free (unbinned) continuous-Poisson curvature fit for the 7 order-2 background cells and stand up the parallel `poisson_unbinned` experiment. New `fit_profiled_poisson_surface_unbinned` (point-evaluated likelihood + analytic annulus normalization, reusing the cutting-plane positivity); Stage-D plumbing `--poisson-quadratic-fit {binned,unbinned}` (default binned), `--allow-annulus-count-mismatch`, `--dump-off-events-npz`; new `bootstrap_v6_poisson_background_unbinned.py` (event resample + unbinned refit for order-2, analytic Poisson for order<=1) and `time_split_shape_consistency.py` (median-MJD g(τ)=B_on/N_ann gate); `compare_unbinned_quadratic_bon.py` and `test_unbinned_poisson_surface.py`; 3 `poisson_unbinned` sbatch. Production run (grid 65261, bootstrap 65262): 12-branch B_on envelope = 0 (was the pooled 0.786σ failure), time-split gate PASS (7/7, max 0.834σ), bootstrap 1000/1000 (min excess eigenvalue 12.86, PD). SED unchanged: conservative LogPar chi2/ndof 14.20 ≈ pooled 14.11. New report `crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_poisson_unbinned_report.html`.

- **energy_reconstruction** · 2026-07-18 17:15 · [uncommitted] Add the Scheme B counterpart of the Scheme R double-Rayleigh Poisson-unbinned background report. New ETO Slurm scripts (`scripts/slurm/{run,bootstrap}_v6_64748_scheme_B_poisson_{grid_branch,background}.sbatch`, `..._poisson_unbinned_{grid_branch,background}.sbatch`, `finalize_v6_64748_scheme_B_poisson_unbinned_report.sbatch`, `submit_v6_64748_scheme_B_poisson_unbinned{,_only}.sh`) mirror the Scheme R chain 1:1 — same double-Rayleigh Stage B/C PSF and frozen pooling manifest, only the response swapped to Scheme B's existing MC aperture-conditioned response (`stage_a_v6_64748_..._aperture_conditioned`, containment exactly 1, no Rayleigh factor). New `report/prepare_v6_scheme_b_double_rayleigh_poisson_unbinned_report.py` (copy of the Scheme R enrichment script, retargeted). Ran the full pooled-Poisson (jobs 65293/65294) and unbinned Poisson (jobs 65306/65307/65308) chains on ETO — 26 Slurm jobs total, all completed within ~45 minutes wall clock. Results: 12-branch B_on envelope max shift 0.719% (near-zero regression check, not exactly 0 like Scheme R since the response differs), time-split gate PASS (7/7 order-2 cells, max 0.834σ, equal-statistics median-MJD split), bootstrap 1000/1000 replicates completed with 0 failures. LogPar chi2/ndof: pooled conservative 14.06, unbinned conservative 14.16 (Scheme R: 14.11/14.20) — structurally consistent with Scheme R. New report `crab_sed_v6_64748_nhit100_reselect44_scheme_B_double_rayleigh_poisson_unbinned_report.html`; section structure (15 `<h2>` sections, 26 image refs) verified identical to the Scheme R report.

- **energy_reconstruction** · 2026-07-18 16:30 · [uncommitted] Add forward-fold closure tooling. New `tools/closure_forward_fold_v6.py`: imports `stages/06_fit.py` and forward-folds an injected PL/LogPar spectrum (`--model {logpar,powerlaw}`, `--phi0/--alpha/--gamma/--beta`, `--pivot-tev`) through the same `model_counts()` math into a MOCK Stage-E signal NPZ (schema matches `05_signal.py`; `excess`=fold, `N_on=round(excess+B_on)`, errors + B_on/containment/labels copied from the real Stage-E NPZ), plus metadata JSON — for later `06_fit.py`+`07_sed_points.py` self-consistency re-fits. Includes `--true-energy-scale-delta` (relabels response true-E axis `logE+=log10(1+δ)`) and a `--self-test` mode over a fabricated 3×5×2 response. Guarded companion flag in `stages/06_fit.py`: `--energy-scale-delta` (default 0.0, verified no-op — edges pass through unchanged; non-zero relabels `loge_edges` once at load before every fold) via new `apply_energy_scale_delta()`; delta echoed into `forward_folding` metadata.

- **energy_reconstruction** · 2026-07-18 (IHEP) · [uncommitted] Diagnose cluster 87341711's stall via `condor_history` (93/128 workers exited cleanly, 35/128 hit ExitCode 124 = timeout, none tripped the FAILED/SIZE_MISMATCH grep patterns used for monitoring); rebuilt `strict_recovery_manifest.json` with the array fully inactive (868 accepted, only 60 real Step2 gaps, not the ~316 implied by the raw completion counter) and submitted a reduced 16-worker Step2 recovery array, cluster 87513951.

- **energy_reconstruction** · 2026-07-18 13:45 · [uncommitted] Add a per-cell fitted background-density surface grid to the poisson-unbinned report. New `report/plot_fitted_background_surfaces.py` reconstructs each cell's quadratic density from Stage-D `surface_density_coefficients` and renders the relative deviation `(q/q0-1)%` over the source-centered ROI for all 84 (Nhit,predE) cells in the physical grid layout (order-2 SED targets red-framed with τ, order-2 pooled donors orange, order-1 planes gray; ON aperture + annulus circle overlays; per-panel colour autoscale). Coefficients pulled from the ETO Stage-D NPZ (`extract_surf.py`); asset `fitted_background_surfaces_grid.png` inserted into the "Grid-Free Unbinned Poisson Method" section. Local-only hand-edit — not yet wired into the report builder, so a finalize re-run would drop it until the builder is extended.

- **energy_reconstruction** · 2026-07-17 23:47 · [uncommitted] Add a Claude handoff for EOS recovery and the common-GTI Pass5-versus-v6 covariance comparison.

- **energy_reconstruction** · 2026-07-17 13:27 · [uncommitted] Add a Claude review handoff for the pooled-Poisson background implementation, evidence, and next scientific checks.

- **energy_reconstruction** · 2026-07-16 20:16 · [uncommitted] Complete the pooled-Poisson 12-branch grid study and 1,000-replicate covariance validation, recording quadratic-cell convergence failures.

- **energy_reconstruction** · 2026-07-16 17:06 · [uncommitted] Add analytic quadratic on-aperture background integration and the isolated Scheme R 1R/2R validation report.

- **energy_reconstruction** · 2026-07-16 16:48 · [uncommitted] Add the implementation plan for positive pooled-Poisson background fitting and 12-branch grid convergence.

- **energy_reconstruction** · 2026-07-16 14:01 · [uncommitted] Add an isolated Scheme R double-Rayleigh fixed-1.58-sigma experiment, validation, and three-branch report.

- **energy_reconstruction** · 2026-07-16 11:35 · [a2ffd6f] Recompute and regenerate v6 Scheme R with the exact 71.297903% containment for a 1.58-sigma Rayleigh aperture.

- **energy_reconstruction** · 2026-07-15 14:02 · [uncommitted] Replace v6 reselect44 Scheme B with an isolated double-Rayleigh aperture chain, validated Slurm rerun, and old-versus-new report.

- **energy_reconstruction** · 2026-07-15 11:29 · [d60fc41] Align all five v5 PSF profile grids to the Rayleigh-baseline MC profile and common per-cell axes.

- **energy_reconstruction** · 2026-07-14 00:41 · [2d06233] Add a PPT-ready spectral precision figure comparing 1D/2D relative uncertainty, per-energy improvement, and joint covariance volume.

- **energy_reconstruction** · 2026-07-14 00:22 · [63645bd] Add a PPT-ready PNG/PDF table comparing controlled 1D and 2D v6 LogPar parameter uncertainties.

- **energy_reconstruction** · 2026-07-14 00:21 · [662e527] Add an English PPT workflow diagram using the numbered-card design and the v6 observation/MC-to-SED analysis logic.

- **energy_reconstruction** · 2026-07-13 23:59 · [eb64386] Add a PPT-ready v6 workflow diagram tracing observation and simulation inputs through Stage A-G to the final SED.

- **energy_reconstruction** · 2026-07-13 19:45 · [cf3aea5] Hide the two highest-energy predE outliers from the Scheme R Stage G ratio figure while retaining all Stage G data points.

- **energy_reconstruction** · 2026-07-13 17:37 · [c05d3a8] Match predE-grouped points in the new Stage G overlays to the green square styling used by the Stage G ratio plot.

- **energy_reconstruction** · 2026-07-13 16:58 · [34c5c21] Add separate Scheme B/R Stage G external overlays containing both Nhit-grouped and predE-grouped SED points while preserving the Nhit-only figures.

- **energy_reconstruction** · 2026-07-13 15:31 · [567a3ab] Replace decorative profile stripes with per-cell Stage D annulus spans in all v6 RA/Dec background diagnostics.

- **energy_reconstruction** · 2026-07-13 14:22 · [3f36252] Replace circular-ROI RA/Dec projections with matched one-degree central slices in the v6 background before/after diagnostics.

- **energy_reconstruction** · 2026-07-13 13:55 · [88a755a] Add matched-support RA/Dec observed-count profiles before background subtraction to both v6 Scheme B and Scheme R reports.

- **energy_reconstruction** · 2026-07-13 11:31 · [8222a28] Add parallel full v6 Scheme B and fixed-71.5%-Rayleigh Scheme R Stage A-G reports with isolated Slurm outputs.

- **energy_reconstruction** · 2026-07-13 00:21 · [uncommitted] Add a seven-bin Nhit-grouped v6 forward-folding diagnostic and compare it with the official 2D fit.

- **energy_reconstruction** · 2026-07-12 23:48 · [uncommitted] Refit v6 after diagnostically removing five largest-pull display cells and document the remaining poor goodness of fit.

- **energy_reconstruction** · 2026-07-12 21:34 · [uncommitted] Add the v6 fixed-Rayleigh versus MC aperture-conditioned PSF response comparison, Slurm workflow, and independent HTML report.

- **energy_reconstruction** · 2026-07-12 18:50 · [uncommitted] Replace the Stage G external ratio panels with the official Pass5 WCDA point-fit LogPar comparison.

- **energy_reconstruction** · 2026-07-12 17:58 · [uncommitted] Append the final v6 LogPar parameters, uncertainties, fit diagnostics, and goodness-of-fit caveat to the report.

- **energy_reconstruction** · 2026-07-12 17:37 · [uncommitted] Add normalized RA- and Dec-offset Stage D excess profile grids with the final 44 fit cells highlighted.

- **energy_reconstruction** · 2026-07-12 17:29 · [uncommitted] Add the 84-ID Stage D observed counts skymap before the ROI excess grid in the latest v6 report.

- **energy_reconstruction** · 2026-07-12 17:08 · [uncommitted] Keep flat, presentation-ready copies of all 14 latest v6 report figures in the dedicated asset directory.

- **energy_reconstruction** · 2026-07-12 17:05 · [uncommitted] Renumber latest v6 figure labels to display cells 1-84 while leaving the predE >= 6 tail unnumbered and all analysis data unchanged.

- **energy_reconstruction** · 2026-07-12 16:54 · [uncommitted] Add the 91-cell true-energy distribution grid and selection explanation to the latest v6 report.

- **energy** · 2026-07-12 16:45 · [uncommitted] Add unfiltered diagnostic Rayleigh profiles, swap selected cell 90 for 75, and refit the overwritten v6 Stage A-G report.

- **energy_reconstruction** · 2026-07-12 16:26 · [09ca526] Remove approved v1-v3 obsolete intermediate outputs and legacy response caches from local and ETO storage.

- **energy** · 2026-07-12 15:51 · [uncommitted] Extend the v6 true-energy grid to 91 cells and display the C75-in/C90-out selection override.

- **energy** · 2026-07-12 15:43 · [uncommitted] Add the 84-cell v6 true-energy distribution grid with fit selection and adjacent-overlap diagnostics.

- **energy** · 2026-07-12 15:12 · [uncommitted] Archive all 13 images referenced by the latest v6 report in its dedicated assets directory.

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
