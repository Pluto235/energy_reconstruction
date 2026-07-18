# Pass5 versus v6 Crab covariance comparison

This workflow compares the v6 64748 2D `Nhit x predE` Crab fit with the
official Pass5 Nhit-only fit on identical recovered-time GTIs. Both final
covariances are expressed for a natural-log LogPar model at a 3 TeV pivot, and
the plotted uncertainty range is 1-100 TeV.

## Audited sample contract

- v6 files with recovered time: `3969`
- nominal 2022-H1 hourly slots absent before Stage C: `375`
- Pass5 event-level matches for the selected hours: `3969`
- sorted recovered-time GTI intervals: `8641`
- true gaps greater than 60 seconds: `4672` (`522529.189930 s`)
- exact sorted-GTI live time: `149.089914423326 d`
- negative MJD steps in the original event order: `4170`
- obsolete unsorted Stage C rough estimate: `125.670245718786 d`

The old 125.67-day estimate must not be used for this comparison. Stage C
applied `np.diff` to event-order MJD without sorting, counted positive jumps as
gaps, and ignored the corresponding negative time wraps. The GTI extractor
sorts valid `match_status == 0` MJD values inside each file before finding real
gaps.

The IHEP run directory retains its historical `125d` name:

```text
/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance
```

All products inside the current workflow are identified as `sorted_gti`.

## 1. Reconstruct v6 GTIs on ETO

Run the GTI audit through Slurm:

```bash
cd /home/server/projects/energy_reconstruction/apply/tools/pass5_covariance
sbatch run_extract_v6_gti_slurm.sh
```

The authoritative outputs are:

```text
v6_gti_output/v6_sorted_gti.tsv
v6_gti_output/v6_sorted_gti_manifest.json
v6_gti_output/v6_sorted_gti_source_files.csv
```

`build_gti_source_files.py` expands the Stage C file ledger into one exposure
row per GTI. Stage F can then integrate the true source zenith exposure rather
than spreading a per-file rough duration uniformly over the file span.

## 2. Refit v6 with exact GTIs

```bash
cd /home/server/projects/energy_reconstruction/apply/tools/pass5_covariance
sbatch run_v6_sorted_gti_fit_slurm.sh
```

This creates an independent, non-promoted Stage F run under:

```text
apply/output/stage_f_v6_64748_nhit100_reselect44_split56_miss030_sortedgti149
```

It reuses the same 44-cell Stage A response and Stage E excess counts while
recomputing the source-theta exposure from the exact GTIs.

## 3. Mask Pass5 event-level hours on IHEP

Build the isolated GTI-aware copy of the official hourly program:

```bash
cd /home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance
./build_gti_hour_binary.sh
```

Build the 3969 hourly jobs:

```bash
python3 build_gti_hour_jobs.py \
  --source-files v6_source_files.csv \
  --gti-tsv v6_gti_output/v6_sorted_gti.tsv \
  --output-dir gti_hour_selection \
  --scratch-dir /scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_hours
```

Each event is tested against the corresponding v6 GTI before filling the
official Pass5 `hacc`, `hon`, `hmjd`, and `hMJD_all_fine` products. Existing
official maps are never overwritten.

Submit the hourly array to the short queue:

```bash
./submit_gti_hours.sh
```

## 4. Generate and merge Direct Integration maps

After all hourly `acc/bkg` pairs exist, build 4-hour DI jobs from the masked
scratch products:

```bash
python3 build_strict_hour_lists.py \
  --source-files v6_source_files.csv \
  --gti-manifest v6_gti_output/v6_sorted_gti_manifest.json \
  --data-root /scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_hours \
  --xrootd-prefix "" \
  --output-dir strict_hour_selection \
  --scratch-dir /scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_map_chunks

./submit_strict_map_chunks.sh
```

The DI jobs intentionally have no short wall-time limit. Merge only after all
1078 J2000 chunks are present:

```bash
hep_sub -g lhaaso -mem 8000 merge_strict_map.sh
```

The merged map is:

```text
pass5_v6_sorted_gti_map.root
```

## 5. Fit Pass5 and export HESSE covariance

```bash
hep_sub -g lhaaso -mem 15000 prepare_strict_pass5_fit.sh
```

The official workflow is preserved through ROI preparation and nuisance-source
fitting. Before the final covariance fit, the Crab LogPar parameters are
exactly reparameterized from the official 10 TeV pivot to 3 TeV. The isolated
`gtlike_cov` executable adds full covariance, correlation, EDM, covariance
status, minimum value, and function-call metadata without changing the
likelihood or HESSE rule.

Final Pass5 output:

```text
sorted_gti_fit/covariance_fit.yaml
```

## 6. Plot the comparison

```bash
python report/plot_v6_vs_pass5_covariance.py \
  --pass5-yaml sorted_gti_fit/covariance_fit.yaml \
  --pass5-live-days "$(cat sorted_gti_fit/strict_pass5_live_days.txt)" \
  --v6-json output/stage_f_v6_64748_nhit100_reselect44_split56_miss030_sortedgti149/runs/v6_64748_nhit100_reselect44_split56_miss030_sortedgti149_stage_f/fit_v6_64748_nhit100_reselect44_split56_miss030_sortedgti149_metadata.json \
  --v6-live-days 149.08991442332612 \
  --gti-manifest tools/pass5_covariance/v6_gti_output/v6_sorted_gti_manifest.json \
  --output-dir report/assets/v6-vs-pass5-sorted-gti-covariance \
  --report-html report/crab_v6_vs_pass5_sorted_gti_covariance_report.html
```

Both PNG and vector PDF figures are generated.

## Interpretation limits

- Covariances are formal HESSE statistical uncertainties; systematics are not
  included.
- v6 uses a conservative chi-square objective, while Pass5 uses a Poisson
  likelihood.
- Pass5 uses `30 <= Nhit < 2000`; v6 uses `100 <= Nhit < 3000` with different
  bin edges. This is a full-pipeline comparison, not an isolated predE
  ablation.
- Pass5 is Nhit-only and has no event-by-event reconstructed-energy cut. The
  reported spectral-uncertainty comparison is evaluated over 1-100 TeV.
- The v6 goodness of fit remains poor. A smaller formal covariance alone is not
  proof of a smaller total uncertainty.
- Cross-method covariance is unavailable, so the spectrum-ratio band is not a
  rigorous difference significance.
