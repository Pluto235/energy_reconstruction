# Claude Handoff: Pass5 versus v6 Common-GTI Covariance Comparison

Last updated: 2026-07-17 23:47 CST

## Objective

Complete the controlled Crab spectral-precision comparison requested by the
user:

- official Pass5, Nhit-only;
- v6 model 64748, 2D `Nhit x predE`;
- the same observation times in both pipelines;
- natural-log LogPar with a 3 TeV pivot;
- full Minuit/HESSE covariance;
- relative spectral uncertainty over 1-100 TeV;
- final PNG/PDF/CSV/JSON/HTML comparison products.

The task is not complete. A long IHEP recovery array is running. Do not submit
a duplicate array, merge incomplete maps, or compare pipelines with different
time selections.

## First action: monitor, do not resubmit

Connect to IHEP with:

```bash
ssh -o RemoteCommand=none -o RequestTTY=no ihep
```

The only active production array is:

```text
schedd:  schedd07
cluster: 87341711
purpose: recover 822 Step2 + J2000 chunks using 128 workers
```

Status at 2026-07-17 23:47 CST:

```text
128 Running
255 / 822 STEP2_RECOVERY_COMPLETE
0 STEP2_RECOVERY_WORKER_DONE
0 failure markers
361 nonempty J2000 maps on EOS
```

`WORKER_DONE=0` is expected at this stage: each worker processes 6-7 chunks
sequentially and only emits `WORKER_DONE` after its complete modulo partition.

Use these exact checks:

```bash
RUN=/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance

env -u LD_LIBRARY_PATH /usr/bin/condor_q \
  -name scheduler@schedd07.ihep.ac.cn \
  -constraint 'ClusterId==87341711' -af JobStatus | sort | uniq -c

grep -h 'STEP2_RECOVERY_COMPLETE' \
  "$RUN"/run_strict_step2_recovery_worker.sh.out.87341711.* | wc -l

grep -h 'STEP2_RECOVERY_WORKER_DONE' \
  "$RUN"/run_strict_step2_recovery_worker.sh.out.87341711.* | wc -l

grep -h 'STEP2_RECOVERY_.*FAILED\|SIZE_MISMATCH\|failed=[1-9]' \
  "$RUN"/run_strict_step2_recovery_worker.sh.{out,err}.87341711.* | wc -l

find /eos/user/l/liushijie/pass5_crab_v6_sorted_gti/\
pass5_crab_v6_sorted_gti_map_chunks \
  -maxdepth 1 -type f -name '*_BKG_J2000.root' -size +0c | wc -l
```

Expected final numbers if all workers succeed:

```text
822 STEP2_RECOVERY_COMPLETE
128 STEP2_RECOVERY_WORKER_DONE
0 failure markers
928 nonempty J2000 maps on EOS
```

Do not rerun `build_strict_recovery_manifest.py` while cluster `87341711` is
active. The workers stream `strict_recovery/step2_recovery_jobs.tsv`; replacing
that file during execution can change their assignments.

## Paths and environments

Repository:

```text
git@github.com:Pluto235/energy_reconstruction.git
```

Workspaces:

```text
Mac working copy:
/Users/luoji/Documents/projects/energy/apply

ETO Git checkout and heavy v6 compute:
/home/server/projects/energy_reconstruction

IHEP Pass5 run directory:
/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance

EOS recovery root:
/eos/user/l/liushijie/pass5_crab_v6_sorted_gti

EOS hourly products:
/eos/user/l/liushijie/pass5_crab_v6_sorted_gti/pass5_crab_v6_sorted_gti_hours

EOS map products:
/eos/user/l/liushijie/pass5_crab_v6_sorted_gti/pass5_crab_v6_sorted_gti_map_chunks
```

The IHEP run directory retains the historical string `125d`, but 125.67 days
is not the correct full-sample live time. Do not infer the analysis exposure
from the directory name.

Important environment rule:

- sourcing the legacy LHAASO environment can make modern `condor_q`, `xrdfs`,
  and `xrdcp` load incompatible C++ libraries;
- use `env -u LD_LIBRARY_PATH /usr/bin/condor_q|xrdfs|xrdcp` for modern system
  commands;
- the DI ROOT executables still require the legacy CVMFS environment;
- the recovery workers already separate these environments correctly.

Compute nodes do not mount `/eos`. Worker-side EOS access must use XRootD URIs
such as:

```text
root://eos01.ihep.ac.cn//eos/user/l/liushijie/...
```

## Current sample audit

The original v6 observation selection contains:

```text
3969 processed hourly files
8641 sorted GTI intervals
4672 real gaps greater than 60 s
522529.189930 s total real-gap duration
149.089914423326 d sorted-GTI live time
4170 negative MJD steps in original event order
```

The obsolete value `125.670245718786 d` came from applying `np.diff` to
unsorted event-order MJD and is wrong. The corrected full-sample live time is
`149.089914423326 d`.

Authoritative GTI files on ETO/local:

```text
apply/tools/pass5_covariance/v6_gti_output/v6_sorted_gti.tsv
apply/tools/pass5_covariance/v6_gti_output/v6_sorted_gti_manifest.json
apply/tools/pass5_covariance/v6_gti_output/v6_sorted_gti_source_files.csv
```

Pass5 has all 3969 corresponding hourly source files. The official DI stage,
however, rejects some four-hour chunks after exact GTI masking. The audited
classification is:

```text
 96 accepted before recovery
 10 valid BKG files recovered directly to J2000
822 BKG files corrupted by the old scratch quota failure and being recomputed
104 rejected by official minAccCorr=0.1
 46 rejected because the central Pass5 interval has no usable event product
---
1078 candidate four-hour chunks
```

After recovery, the expected scientifically accepted set is:

```text
928 accepted maps
104 rejected_acceptance
 46 rejected_no_central_events
150 rejected total
```

The 150 rejected chunks must not be represented as zero-count maps.

## Recovery provenance and lessons

Original DI array:

```text
cluster 87259359, schedd07
1078 jobs, now finished
```

It failed operationally because the user scratch quota reached its hard limit.
Many BKG files were nonzero but corrupt; checking only file size greater than
zero was insufficient. Original error logs showed 822 BKG write failures.

Migration:

```text
successful cluster: 87339606
hours: 7938 files, 20,813,837,638 bytes
maps:   938 files,  5,991,729,631 bytes
```

Source and EOS manifests matched by relative path and byte count before the
scratch copies were removed. The migration released about 26 GB of scratch
quota.

Earlier migration attempts `87338875` and `87339233` failed before copying or
deleting data because compute nodes lacked the EOS mount / inherited an
incompatible `xrdfs`; they are obsolete.

J2000 recovery:

```text
failed corrupt-BKG smoke: 87340303
removed idle smoke:       87340488
successful smoke:         87340605
successful 10-worker run: 87346464
```

The successful smoke wrote a 261,823,127-byte EOS ROOT file. The direct
J2000 array completed 10/10 workers with zero failures. Do not resubmit it.

Active Step2 recovery:

```text
cluster 87341711, 128 workers
```

Each worker:

1. reads the exact GTI-masked hourly ROOT products through XRootD;
2. writes Step2 BKG and J2000 files to node-local temporary storage;
3. uploads only the final J2000 ROOT file;
4. verifies remote byte size after XRootD checksum verification;
5. removes temporary local files;
6. continues with the next modulo-assigned chunk.

The source hourly EOS directory must not be deleted until all common-GTI work
is complete.

## Recovery scripts

All recovery code is under:

```text
apply/tools/pass5_covariance/
```

Key files:

```text
migrate_strict_products_to_eos.sh
build_strict_recovery_manifest.py
run_strict_j2000_recovery_worker.sh
submit_strict_j2000_recovery.sh
run_strict_step2_recovery_worker.sh
submit_strict_step2_recovery.sh
merge_strict_map.sh
prepare_strict_pass5_fit.sh
run_v6_sorted_gti_fit_slurm.sh
```

`README.md` section 4 still describes the pre-failure expectation that all
1078 chunks should be merged. That instruction is obsolete. Follow this
handoff and the audited 928-map accepted list instead.

## What to do when cluster 87341711 ends

1. Confirm the queue is empty and inspect history/exit codes.
2. Require 822 `STEP2_RECOVERY_COMPLETE` markers.
3. Require 128 `STEP2_RECOVERY_WORKER_DONE` markers with `failed=0`.
4. Require zero `FAILED`, `SIZE_MISMATCH`, or nonzero-failed markers.
5. Require 928 nonempty J2000 files on EOS.
6. Only then rerun the recovery manifest builder.

Commands:

```bash
cd /home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance
./build_strict_recovery_manifest.py

wc -l strict_recovery/accepted_maps.list
wc -l strict_recovery/rejected_jobs.tsv
```

Expected manifest after a clean recovery:

```text
accepted:                    928
recover_j2000:                 0
recover_step2:                 0
rejected_acceptance:         104
rejected_no_central_events:   46
```

`accepted_maps.list` must have exactly 928 lines.

If the array ends with failures, do not resubmit all 822 chunks. Rerun the
manifest builder only after the old array is fully inactive; it will classify
existing nonempty J2000 files as accepted and generate a reduced Step2 queue
for the missing infrastructure-failure chunks. Official quality rejections
must remain rejected.

## Critical scientific issue: construct a common GTI

The 149.089914-day v6 fit and the recovered Pass5 accepted maps no longer use
identical times, because official Pass5 rejects 150 four-hour candidate chunks.
Therefore the existing full-sample v6 covariance cannot be fairly compared to
the 928-map Pass5 result.

The correct next experiment is the intersection:

```text
common GTI = v6 sorted GTI intersect Pass5 accepted central windows
```

Use:

```text
strict_hour_selection/strict_hour_manifest.json
strict_recovery/rejected_jobs.tsv
v6_gti_output/v6_sorted_gti.tsv
v6_gti_output/v6_sorted_gti_source_files.csv
```

The strict-hour manifest records each job label and its
`central_available_hours`. Accepted labels are the 1078 job labels minus the
150 rows in `rejected_jobs.tsv`. Filter the v6 GTIs/source-file rows to those
accepted central hours and write an audited common-GTI manifest with:

- accepted/rejected labels;
- included source-file IDs and hours;
- GTI count;
- exact live time in seconds/days;
- self-consistency checks against the Pass5 accepted-map list.

Do not merely rerun Stage F with a shorter exposure. The current
`run_v6_sorted_gti_fit_slurm.sh` reuses the full-sample Stage E signal NPZ:

```text
output/stage_e_v6_64748_nhit100_reselect44_split56_miss030_containment1_annnorm/
```

That means the observed excess counts would still correspond to all 149 days.
For a fair common-time comparison, rerun the v6 observation side from Stage C
through Stage E, then Stage F, with the common accepted-hour/GTI selection.
Reuse the same Stage A response and fixed 44-cell selector; do not retrain or
retune the model.

Heavy v6 work must run on ETO through Slurm. Do not run it on the Mac.

## Pass5 merge and fit after common-time v6 is defined

The updated `merge_strict_map.sh` accepts an explicit `MAP_LIST` and refuses
missing/empty paths. Use the audited 928-map list:

```bash
cd /home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance

MAP_LIST="$PWD/strict_recovery/accepted_maps.list" \
OUTPUT="$PWD/pass5_v6_common_gti_map.root" \
hep_sub -g lhaaso -mem 8000 merge_strict_map.sh
```

Do not use `pass5_v6_sorted_gti_map.root` as a name implying the original full
149-day sample unless the report clearly records the accepted-map intersection.

Before the fit, parameterize `prepare_strict_pass5_fit.sh` or make an isolated
common-GTI copy so it reads the new map and writes a new fit directory. Preserve
the existing workflow:

- official Pass5 Nhit bins `30 <= Nhit < 2000`;
- Poisson likelihood;
- LogPar exact pivot transform to 3 TeV;
- full HESSE covariance from `bin/gtlike_cov`;
- no change to official source/background fitting steps.

Required validation:

```text
covariance_status = 3
sqrt(diag(covariance)) equals YAML parameter errors
pivot = 3 TeV
merged-map live time is consistent with the accepted Pass5 sample
```

## Existing v6 full-sample reference (not the final comparator)

The already completed full sorted-GTI v6 Stage F result is:

```text
log10_phi0 = -11.710764813580107
alpha      = 2.760706576796731
beta       = 0.10713218339024227

sigma(log10_phi0) = 0.00654135533422906
sigma(alpha)      = 0.020474284815346522
sigma(beta)       = 0.013574753574570171

chi2 / ndof = 479.856 / 41 = 11.70379
p = 3.2810e-76
```

Covariance:

```text
[[ 4.278933245814835e-05, -7.511910944890118e-06,  3.680182292150205e-05],
 [-7.511910944890118e-06,  4.1920433764066424e-04, 2.0676280353837653e-04],
 [ 3.680182292150205e-05,  2.0676280353837653e-04, 1.8427677256329487e-04]]
```

Path:

```text
apply/output/stage_f_v6_64748_nhit100_reselect44_split56_miss030_sortedgti149/
runs/v6_64748_nhit100_reselect44_split56_miss030_sortedgti149_stage_f/
fit_v6_64748_nhit100_reselect44_split56_miss030_sortedgti149_metadata.json
```

The very poor v6 goodness of fit must remain prominent in the final report.

## Final comparison products

Plotting entry point:

```text
apply/report/plot_v6_vs_pass5_covariance.py
```

It has already passed a smoke test with an older Pass5 YAML and is configured
for 1-100 TeV. Final suggested outputs:

```text
apply/report/assets/v6-vs-pass5-common-gti-covariance/
apply/report/crab_v6_vs_pass5_common_gti_covariance_report.html
```

The report must state:

- both covariances are formal HESSE statistical uncertainties only;
- v6 uses a conservative chi-square objective, Pass5 a Poisson likelihood;
- Pass5 uses `30 <= Nhit < 2000`, v6 uses `100 <= Nhit < 3000` and different
  binning;
- this is a full-pipeline comparison, not an isolated predE ablation;
- the v6 goodness of fit is poor;
- a smaller formal covariance does not prove smaller total uncertainty;
- cross-method covariance is unavailable, so a ratio/difference band is not a
  rigorous significance test.

## Git and handoff hygiene

The ETO production checkout is intentionally dirty with unrelated work. Never
reset, clean, or bulk-stage it. Stage only this task's paths.

Current task paths include:

```text
apply/tools/pass5_covariance/
apply/report/plot_v6_vs_pass5_covariance.py
apply/claude-handoff-pass5-v6-common-gti-covariance.md
```

`apply/devlog.md` already has unrelated uncommitted entries. Add one concise
newest-first entry for this task, but do not accidentally commit unrelated
working-tree changes. Use partial staging for only the new devlog line if
necessary.

After the entire scientific task is complete:

1. sync only explicit task files between Mac/ETO/IHEP;
2. update `apply/devlog.md` with the final result;
3. stage only this task's files;
4. commit in `/home/server/projects/energy_reconstruction`;
5. run `git push origin main`;
6. preserve all unrelated ETO modifications.

## Non-negotiable guardrails

- Do not duplicate cluster `87341711` while it is active.
- Do not rewrite `step2_recovery_jobs.tsv` while its workers are active.
- Do not treat rejected Pass5 chunks as zero-count observations.
- Do not lower `minAccCorr=0.1` to force acceptance.
- Do not delete EOS hourly sources before common-GTI processing is complete.
- Do not use 125.67 days as the exposure.
- Do not compare 928-map Pass5 against the existing 149-day v6 covariance and
  label it a fair comparison.
- Do not run heavy processing on the Mac.
- Do not reset or clean the dirty ETO checkout.

