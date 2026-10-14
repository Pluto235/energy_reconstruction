# 202201 frozen v6 Crab execution record

Goal: execute the independent January 2022 Crab analysis on ETO with model 64748, the frozen 44-cell selector, double Rayleigh PSF, Scheme R and analytic B_on.

The source request fixes the scientific design. Routine path adaptation and missing cache reconstruction are authorized. The active remote checkout is clean at f979c5b before this work; local apply is a non-Git working copy. Targeted source synchronization preserves other local files and remote generated outputs.

- [x] Gate the accepted 681 prediction files / 48,726,074 events against alignment receipts and original input bindings; preserve the 464 original time anomalies.
- [x] Audit original full time sequences with the old 60-second gap definition, globally merge overlapping intervals, and feed interval rows through the existing Stage F source trajectory integrator.
- [x] Verify response/model/cuts/binning/primary denominator and PSF; reuse compatible arrays and rebuild missing 2R from saved original profiles with the original algorithm.
- [x] Run fixture tests and real single-file Stage C smoke under Slurm, then monthly C and production D. D fails the required cell62 positivity contract; E–G stay blocked. This is not a completed valid spectrum.
- [x] Validate aperture scaling once, 44-cell order, analytic integration, January provenance, exposure, convergence and fit quality separately; report66089 completes machine-readable receipts and HTML with scientific failure explicit.
- [x] Prepend devlog, commit and push code/config/docs; analysis code dd8b060 and final verification code7d46912 are pushed, with outputs outside Git.

Run tag: `v6_64748_202201_scheme_R_double_rayleigh_analytic_bon`. All new outputs and stage reports are under `apply/output/<run-tag>/`. No current/latest links are promoted; existing directories are never overwritten.

Implementation reaches only new January source gate, GTI audit, Slurm entry and final validation/report code, plus their tests and this record. Existing scientific stages are reused with explicit arguments. Stage F already accepts `--source-files-csv`, allowing audited GTIs without replacing its source-theta calculation.

Slurm partition/QOS inspected: main/default-safe, 88 configured CPUs, 170000 MiB node memory, max 32 CPUs/96 GiB per job, max 64 CPUs/160 GiB per user. CPU-only analysis and PSF reconstruction require no GPUs. Initial existing v7 dependency jobs are preserved.

Initial findings: frozen selector SHA256 matches c85c3b03839ee6b4d3321bbaa87a0dc171a5f8427c4ed7ff298fbc19f295b4f6. Old C–G entry permits mismatched time rows and masks Stage E exit failures; the January entry omits those behaviors. Direct ETO GitHub SSH access currently fails host-key validation; use the established local Git transfer/push route without changing cluster SSH trust settings.

## Actual execution

Source gate and real one-file C smoke66081 pass (74,357 selected events). Full C66082 passes681files, 21,559,258 selected parquet rows. All input qualitycuts were already applied by accepted inference; Nhit<100 explains27,166,816 out-of-ledger rows, including the4predE<2 events.

Saved-profile PSF66084 passes after path-only66083failure. All91radii bitwise match historical double-Rayleigh aperture arrays. [MC/PSF compatibility evidence](JANUARY_V6_MC_COMPATIBILITY.json) contains model/preprocessing/cuts/bins/weights/denominator/area/unit checks and checksums; no MC inference/bin rebuild is needed.

GTI66086 performs a one-file rawtime smoke then full681file audit:69,279,369 rawtimes, status0=69,278,905, nonzero464 retained, IESfallback146. There are756merged intervals, 2,294,210.295814seconds /26.5533599053days effective gap-defined live time, 3.819604seconds overlap removed. UTCrange2021-12-31T16:00:20.830030Z through2022-01-31T15:59:31.734623Z. Source-theta exposure from the old60-second integrator is811,515.329108seconds /9.39253853134days. StageC's prediction-sparse roughestimate23.5472966260days is not used for exposure. No independent hardware deadtime record is available.

ProductionD66085 fails with cell62 minimum−0.001473146297436counts/pixel; diagnostic66087 keeps all91cells plus an ordered44cell table, original coefficients/rank/condition, rawintegral−0.878926083876counts and actual annulus10events/5648pixels. Cell62rank6 and condition65.4631553 show a solved linearfit, but its rawpolynomial cannot define a nonnegative background. No fallback, clipping, tuning, selection change or43cell fit is used. Existing StageE spherical N_on code is used only for diagnostics; formal E/F/G and spectrum/SED remain undefined.

Fixed44PSF fallbackcells56,61,70,75,86,87 remain in the scheme. Negative diagnosticexcess occurs in4,5,6,21,75; N_on<10 in61,62,70,86,87,88,89. These are diagnostics, not removal rules. No spectralpull or goodness-of-fit is available.

Report66088 passes data/response/exposure checks but fails due to missing renderer deployment; its Markdown and logs are preserved under failed_report_66088. The renderer is then deployed with its assets; retry stays within the same pinned scientific inputs. New output directory remains `/mnt/newdisk/shijie/energy/apply/output/v6_64748_202201_scheme_R_double_rayleigh_analytic_bon/`.

Report retry66089 completes0:0 in3seconds. HTML anchors/local artifact links/images and copied artifact hashes pass local inspection. The cell62 figure displays the actual un-clipped negative quadratic; source exposure figure uses9.39254days, not the26.55336day detector live interval sum. Twelve initial fixtures pass; an additional JSON-versus-NPZ/count binding regression brings the final targeted suite to13tests. Added final delivery verification also checks all681parquet part hashes and their canonical dataset digest, pins source file hashes to actual Git blobs, retains the report generation base commit and records the later committed source version separately.

The report is `report_202201.html`; data/science receipt is `final_acceptance.json`; commit/dataset delivery receipt is `delivery_receipt.json`. These live under the isolated run directory and are Git-ignored. Local report copy uses the same relative layout, with parquet data retained only on ETO. Review and provenance strengthening change only checks/reporting; no scientific stages, model, selector or already accepted input is altered.

## Final delivery verification

Slurm66090 completes `0:0` in4seconds with13fixtures passed, the Git worktree clean and all committed source blobs matching the deployed files. All681parquet parts, totaling881,189,730bytes, are SHA256-bound by the canonical part-list digest in `delivery_receipt.json`. Its source commit is `7d469123f83476bf39c7eac7a5871c5bfd8cea20`; report generation predates that commit, so the receipt preserves basecommitf979c5b and the exact generation-source hashes separately rather than relabeling history.

Final report SHA256: `30dac1fef241cb140c1024c5645004494971f53b1e1694f2e249f48056b04adc`. Data/science receipt SHA256: `1599545471068ca1d8f783d89bffa7f18f373c8bc5410e4cf677ec4c733a3b98`. The local copied receipts and HTML anchors/artifact links/images pass checks; all three scientific PNGs are visually inspected. Full-page browser rendering is not verified because the browser tool disallows file URLs; no workaround is attempted.

Both independent spec and code-quality reviews find no must-fix issue in the executed path. GitHub main is verified at7d46912 after normal push via the local bare Git bridge, without changing ETO SSH trust settings. The final scientific status remains `SCIENTIFIC_BACKGROUND_CONTRACT_FAILED`; E/F/G are `NOT_RUN_D_FAILED`, and no valid44-cell spectrum/SED is claimed. Existing fivev7pending jobs are unchanged.
