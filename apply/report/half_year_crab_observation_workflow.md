# Half-Year Crab Observation Workflow

**Date:** 2026-06-26

This workflow extends the current Crab observation sample from the existing
2022-01-01 through 2022-02-28 dataset to a half-year dataset. The recommended
default half-year window is:

```text
2022-01-01 <= observation date <= 2022-06-30
```

If the target is a different continuous six-month interval, replace
`START_DATE` and `END_DATE` consistently in the IHEP filtering, time recovery,
transfer, and ETO inference steps before submitting jobs.

## Current State

ETO currently holds the production two-month observation sample:

| Item | Path | Current state |
|---|---|---|
| Raw filtered observation ROOT | `/mnt/mydisk/WCDA_observation/<MMDD>/EsgYYYYMMDD_HH.root` | 58 day directories, `0101` to `0228`, 1258 ROOT files |
| ML energy eval ROOT | `/mnt/mydisk/WCDA_observation_eval/<MMDD>/EsgYYYYMMDD_HH.root` | same 1258 ROOT files |
| Recovered time friend ROOT | `/mnt/mydisk/WCDA_observation_eval/recovered_time/<MMDD>/EsgYYYYMMDD_HH.time.root` | same 1258 ROOT files |

IHEP has the source data needed to extend the sample:

| Item | Path | Notes |
|---|---|---|
| Full observation ROOT | `/eos/lhaaso/ai/wcda/raw/2022/<MMDD>/EsgYYYYMMDD_HH.root` | 2022 full-year source for branch-reduced observation inputs |
| Reduced time source | `/eos/lhaaso/raw/wcda/2022/<MMDD>/*.reduced.root` | source of truth for `sec/subsec`, matched by `(irun, ies, iseq, ievent)` |
| Existing IHEP eval sample | `/home/lhaaso/liushijie/WCDA_observation_eval` | current 2022-01-01 to 2022-02-28 eval ROOT sample |
| Existing recovered time | `/home/lhaaso/liushijie/WCDA_observation_eval/recovered_time` | current friend tree sample and handoff notes |

IHEP month-level source coverage checked on 2026-06-26:

| Month | Esg ROOT count in `/eos/lhaaso/ai/wcda/raw/2022` |
|---|---:|
| 2022-01 | 681 |
| 2022-02 | 600 |
| 2022-03 | 690 |
| 2022-04 | 649 |
| 2022-05 | 687 |
| 2022-06 | 662 |

The January-June source has about 3969 hourly ROOT files before quality/branch
reduction. The existing 1258-file ETO sample corresponds to the first two
months after filtering.

## Important Contract

Do not pair a recovered-time friend tree made from a different entry sequence
with an eval ROOT file.

The current ETO Stage C code,
`apply/stages/03_reduce_obs.py`, reads:

```text
eval ROOT:      t_eventout
time friend:    t_recovered_time
alignment:      same file stem and same entry index
required time:  mjd, ra_mean_deg, dec_mean_deg, match_status
```

`apply/apply_observation_energy.py` writes only events passing:

```text
pincness < 1.1
fitstat == 0
theta < 50 deg
dcedge > 20 m
```

and drops hit-level branches:

```text
vx, vy, vq, vt
```

Therefore the safest existing flow is:

```text
IHEP full Esg ROOT
  -> IHEP branch/pincness reduction
  -> transfer reduced observation ROOT to ETO /mnt/mydisk/WCDA_observation
  -> ETO ML energy inference, producing /mnt/mydisk/WCDA_observation_eval
  -> transfer eval ROOT back to IHEP, or run time recovery where reduced EOS is accessible
  -> recover time and RA/Dec against the eval ROOT entry sequence
  -> transfer recovered_time friend ROOT to ETO
  -> ETO Stage C through Stage G half-year dataset
```

Recovering time before ETO inference is only safe if the inference code is
changed to preserve an entry mapping and Stage C is changed to join by
`(irun, ies, iseq, ievent)` instead of entry number.

## IHEP Step 1: Build Branch-Reduced Observation Inputs

Use the existing IHEP filtering workspace:

```text
/home/lhaaso/liushijie/WCDA_reconstruction
```

Relevant files:

```text
scripts/filter_observed_pincness.C
scripts/condor_filter_observed_day.sh
scripts/submit_obs_filter_hepjob.sh
```

The ROOT macro keeps only analysis-needed event branches and applies:

```text
pincness < 1.1
```

Kept branches include:

```text
n, nfit0, nfit, nfitb, vnfit, fitstat, nrange, nv, vidmc,
vx, vy, vt, vq, theta, phi, xc, yc, dcedge, dcedgepool,
istationcore, ccindex, chi2, rmds, pincness, compactness,
f5w, irun, iseq, ies, ievent
```

Submit the missing extension window first, because ETO already has 2022-01-01
through 2022-02-28:

```bash
cd /home/lhaaso/liushijie/WCDA_reconstruction

START_DATE=20220301 \
END_DATE=20220630 \
REMOTE_DATA_BASE=/eos/lhaaso/ai/wcda/raw/2022 \
REMOTE_OUTPUT_ROOT=/home/lhaaso/liushijie/WCDA_observation_20220101_20220630 \
REMOTE_MANIFEST_ROOT=/home/lhaaso/liushijie/WCDA_reconstruction/manifests/obs_filtered_20220301_20220630 \
REMOTE_LOG_ROOT=/home/lhaaso/liushijie/WCDA_reconstruction/logs/hepjob/obs_filtered_20220301_20220630 \
PINCNESS_MAX=1.1 \
MAX_WAVE_SIZE=4 \
scripts/submit_obs_filter_hepjob.sh
```

Expected output layout:

```text
/home/lhaaso/liushijie/WCDA_observation_20220101_20220630/<MMDD>/EsgYYYYMMDD_HH.root
```

The script submits one day per HepJob/Condor task and verifies that each day
has the same number of output ROOT files as manifest input files.

## IHEP Step 2: Transfer Observation Inputs To ETO

Transfer only the new March-June day directories to avoid overwriting the
existing ETO January-February sample:

```bash
rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_20220101_20220630/03*/ \
  ETO:/mnt/mydisk/WCDA_observation/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_20220101_20220630/04*/ \
  ETO:/mnt/mydisk/WCDA_observation/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_20220101_20220630/05*/ \
  ETO:/mnt/mydisk/WCDA_observation/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_20220101_20220630/06*/ \
  ETO:/mnt/mydisk/WCDA_observation/
```

After transfer, verify on ETO:

```bash
find /mnt/mydisk/WCDA_observation -mindepth 1 -maxdepth 1 -type d \
  -regex '.*/[0-9][0-9][0-9][0-9]' | sed 's#.*/##' | sort | head

find /mnt/mydisk/WCDA_observation -mindepth 1 -maxdepth 1 -type d \
  -regex '.*/[0-9][0-9][0-9][0-9]' | sed 's#.*/##' | sort | tail

find /mnt/mydisk/WCDA_observation -mindepth 2 -maxdepth 2 \
  -type f -name 'Esg*.root' | wc -l
```

For the recommended window, the day directories should span `0101` to `0630`.

## ETO Step 3: Run ML Energy Inference

Use the existing trained model:

```text
/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666
```

Smoke test one new file before the full batch:

```bash
cd /home/server/projects/energy_reconstruction
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/server/anaconda3/envs/py310/bin/python apply/apply_observation_energy.py \
  --input-root /mnt/mydisk/WCDA_observation/0301/Esg20220301_00.root \
  --output-root /tmp/wcda_observation_eval_0301_smoke \
  --run-dir /home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666 \
  --max-events-per-file 128 \
  --step-size "5 MB" \
  --device cpu \
  --batch-size 64 \
  --print-every 1
```

Confirm the smoke output:

```text
t_eventout_h is present
t_eventout has ml_logE_pred and ml_energy_pred
vx, vy, vq, vt are absent
events satisfy pincness < 1.1, fitstat == 0, theta < 50 deg, dcedge > 20 m
```

Run the full new-month inference through Slurm. Prefer month/day prefixes so a
failed month can be rerun without rewriting all January-February outputs:

```bash
cd /home/server/projects/energy_reconstruction

for prefix in 03 04 05 06; do
  DAY_PREFIX="${prefix}" \
  INPUT_ROOT=/mnt/mydisk/WCDA_observation \
  OUTPUT_ROOT=/mnt/mydisk/WCDA_observation_eval \
  SUMMARY_PATH=/mnt/mydisk/WCDA_observation_eval/apply_summary_${prefix}.json \
  sbatch scripts/slurm/apply_observation_energy.sbatch
done
```

Monitor:

```bash
squeue -u server
tail -f logs/slurm/apply_observation_energy_<jobid>.out
tail -f logs/slurm/apply_observation_energy_<jobid>.err
```

## IHEP Step 4: Recover Time And RA/Dec For Eval ROOT

The existing time recovery workspace is:

```text
/home/lhaaso/liushijie/time-recover
```

Relevant scripts:

```text
scripts/build_recover_wcda_time.sh
scripts/build_reduced_manifest.sh
scripts/recover_one.sh
scripts/submit_recovery_condor.sh
scripts/check_recovery_condor.sh
scripts/report_unrecovered_reduced.py
```

The recovery program matches each eval event to reduced data by:

```text
(irun, ies, iseq, ievent)
```

It looks up the reduced file under:

```text
/eos/lhaaso/raw/wcda/2022/<MMDD>/*.reduced.root
```

Then it verifies `wcda.ievent == eval.ievent`, copies `sec/subsec`, computes
Unix time and MJD, and with `WITH_RA_DEC=1` computes:

```text
ra_mean_deg
dec_mean_deg
```

Output friend tree:

```text
t_recovered_time
```

Output branches:

```text
tai_sec, tai_subsec, unix_sec, unix_subsec, mjd, mjdns4,
ra_mean, dec_mean, ra_mean_deg, dec_mean_deg, match_status
```

`match_status == 0` means successful recovery.

If time recovery must run on IHEP, first transfer the newly inferred eval ROOT
from ETO back to IHEP, preserving the `MMDD/EsgYYYYMMDD_HH.root` layout:

```bash
mkdir -p /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  ETO:/mnt/mydisk/WCDA_observation_eval/03*/ \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  ETO:/mnt/mydisk/WCDA_observation_eval/04*/ \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  ETO:/mnt/mydisk/WCDA_observation_eval/05*/ \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  ETO:/mnt/mydisk/WCDA_observation_eval/06*/ \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/
```

Build the recovery binary and reduced manifest:

```bash
cd /home/lhaaso/liushijie/time-recover
scripts/build_recover_wcda_time.sh

REDUCED_ROOT=/eos/lhaaso/raw/wcda/2022 \
REDUCED_MANIFEST=/home/lhaaso/liushijie/time-recover/condor/reduced_2022_manifest.tsv \
scripts/build_reduced_manifest.sh
```

Smoke recover one eval ROOT with RA/Dec:

```bash
EVAL_ROOT=/home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630 \
WITH_RA_DEC=1 \
scripts/recover_one.sh 0301/Esg20220301_00.root
```

For the new March-June batch, prepare a custom file list and submit Condor. The
existing submit script expects an `EXPECTED_COUNT`; set it to the number of
files in the custom eval root.

```bash
cd /home/lhaaso/liushijie/time-recover

EVAL_ROOT=/home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630 \
EXPECTED_COUNT=$(find /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630 \
  -mindepth 2 -maxdepth 2 -type f -path '*/[0-9][0-9][0-9][0-9]/Esg????????_??.root' | wc -l) \
MAX_MATERIALIZE=50 \
scripts/submit_recovery_condor.sh
```

Check recovery:

```bash
EVAL_ROOT=/home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630 \
scripts/check_recovery_condor.sh

EVAL_ROOT=/home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630 \
scripts/report_unrecovered_reduced.py
```

The previous two-month production run had:

```text
inputs=1258
time_roots=1258
summaries=1258
total_entries=127,692,389
matched=127,691,852
unrecovered_events=537
```

For the half-year extension, accept only small, explained unrecovered counts.
Investigate any day with missing `.time.root`, unreadable summary JSON, entry
mismatch, or large nonzero `match_status` counts before transferring back.

## IHEP Step 5: Transfer Recovered Time To ETO

Transfer only the new recovered-time directories:

```bash
rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/recovered_time/03*/ \
  ETO:/mnt/mydisk/WCDA_observation_eval/recovered_time/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/recovered_time/04*/ \
  ETO:/mnt/mydisk/WCDA_observation_eval/recovered_time/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/recovered_time/05*/ \
  ETO:/mnt/mydisk/WCDA_observation_eval/recovered_time/

rsync -av \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /home/lhaaso/liushijie/WCDA_observation_eval_20220101_20220630/recovered_time/06*/ \
  ETO:/mnt/mydisk/WCDA_observation_eval/recovered_time/
```

Verify on ETO:

```bash
obs_count=$(find /mnt/mydisk/WCDA_observation_eval -mindepth 2 -maxdepth 2 \
  -type f -name 'Esg*.root' | wc -l)
time_count=$(find /mnt/mydisk/WCDA_observation_eval/recovered_time -mindepth 2 -maxdepth 2 \
  -type f -name '*.time.root' | wc -l)
printf 'obs=%s time=%s\n' "$obs_count" "$time_count"
```

The counts should match unless there is a documented, intentional exclusion.

## ETO Step 6: Build The Half-Year Stage C Dataset

Run a Stage C smoke test first:

```bash
cd /home/server/projects/energy_reconstruction
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

python apply/stages/03_reduce_obs.py \
  --obs-root /mnt/mydisk/WCDA_observation_eval \
  --time-root /mnt/mydisk/WCDA_observation_eval/recovered_time \
  --cell-selection-csv apply/config/cell_ledger_v3_candidate.csv \
  --output-dir apply/output/stage_c_half_year_smoke \
  --run-id smoke_0301 \
  --day-prefix 0301 \
  --max-files 1 \
  --entries-per-chunk 200000 \
  --workers 1 \
  --match-status-equals 0 \
  --cut-pinc-max 1.1 \
  --cut-fitstat-equals 0 \
  --cut-theta-max-deg 50 \
  --cut-dcedge-min 20 \
  --gap-threshold-sec 60 \
  --overwrite-run-dir
```

Then run Stage C for the full half-year dataset:

```bash
cd /home/server/projects/energy_reconstruction

OBS_ROOT=/mnt/mydisk/WCDA_observation_eval \
TIME_ROOT=/mnt/mydisk/WCDA_observation_eval/recovered_time \
CELL_SELECTION=apply/config/cell_ledger_v3_candidate.csv \
OUTPUT_DIR=apply/output/stage_c_half_year_v3_candidate \
RUN_ID=v3_stage_c_20220101_20220630 \
sbatch scripts/slurm/reduce_obs_stage_c.sbatch
```

Expected Stage C outputs:

```text
apply/output/stage_c_half_year_v3_candidate/runs/v3_stage_c_20220101_20220630/
  obs_events/
  source_files.csv
  cutflow.csv
  cell_counts.csv
  obs_events_summary.md
  obs_events_metadata.json
```

Key checks:

```text
source file count equals eval/time file count
missing time files = 0
entry mismatch files = 0
match_status failures are understood
mjd coverage spans the intended half-year window
Crab ROI event count increased relative to the two-month Stage C run
```

## ETO Step 7: Produce The Half-Year Crab Products

The minimal half-year product is the Stage C parquet dataset. To regenerate
Crab signal, fit, and SED products using existing response/PSF assets, run the
Stage D-G chain with `STAGE_C_DIR` pointed at the new Stage C run.

For the current primary v4 aperture-conditioned branch, the reusable script is:

```text
apply/run_v4_aperture_conditioned_response_pipeline.sh
```

It currently defaults to the old Stage C run:

```text
apply/output/stage_c_v3_candidate/runs/v3_stage_c_slurm_42024
```

Override it for half-year products:

```bash
cd /home/server/projects/energy_reconstruction
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310

STAGE_C_DIR=apply/output/stage_c_half_year_v3_candidate/runs/v3_stage_c_20220101_20220630 \
STAGE_F_RUN_ID=v4_stage_f_aperture_conditioned_drop4_half_year_20220101_20220630 \
STAGE_G_RUN_ID=v4_stage_g_aperture_conditioned_drop4_half_year_20220101_20220630 \
PYTHON_BIN=/home/server/anaconda3/envs/py310/bin/python \
bash apply/run_v4_aperture_conditioned_response_pipeline.sh
```

If Stage D/E should also be rebuilt instead of reusing existing background or
signal assets, use `scripts/slurm/run_v3_stage_c_to_g.sbatch` as the template
and set new output directories and run IDs for every stage. Do not overwrite
the existing two-month baseline directories.

## Validation Checklist

Before large jobs:

- Confirm the exact half-year window with the user.
- Confirm whether new data should extend the existing ETO trees in place or
  write to versioned roots such as `/mnt/mydisk/WCDA_observation_20220101_20220630`.
- Confirm available ETO disk space for raw filtered ROOT, eval ROOT, friend
  ROOT, and Stage C/G outputs.
- Confirm IHEP Condor policy values if the defaults in `submit_recovery_condor.sh`
  fail: `SCHEDD_NAME`, `MAX_MATERIALIZE`, `ACCOUNTING_GROUP`,
  `HEPJOB_REALGROUP`, `HEPJOB_WALLTIME`.

After branch-reduced input production:

- Per-day manifest count equals per-day output ROOT count.
- Output ROOT files open and contain `t_eventout` and `t_eventout_h`.
- Required branches for ETO inference are present:
  `vx`, `vy`, `vt`, `vq`, `vidmc`, `theta`, `xc`, `yc`, `pincness`,
  `fitstat`, `dcedge`.

After ETO inference:

- `ml_logE_pred` and `ml_energy_pred` exist.
- `vx`, `vy`, `vq`, `vt` are absent.
- File names and `MMDD` layout match the input tree.
- `apply_summary_03.json` through `apply_summary_06.json` have no failed files.

After time recovery:

- Every eval ROOT has a matching `.time.root`.
- `t_eventout` and `t_recovered_time` entry counts match for every file.
- `match_status == 0` dominates; nonzero statuses are summarized and explained.
- `ra_mean_deg`, `dec_mean_deg`, and `mjd` are finite for selected events.

After Stage C:

- `source_files.csv` has no missing time files and no entry mismatches.
- `obs_events_metadata.json` reports the intended MJD/date coverage.
- Cell counts and Crab ROI counts are plausible compared with the two-month
  reference, roughly scaling with exposure where detector uptime is comparable.

## Open Decisions

1. Confirm the half-year interval. Recommended answer: use
   2022-01-01 through 2022-06-30 to extend the existing 2022-01-01 through
   2022-02-28 sample.
2. Confirm storage layout. Recommended answer: write new intermediate products
   to versioned roots during production, then promote to `/mnt/mydisk/WCDA_*`
   only after file counts and entry matching pass.
3. Confirm whether final products need only Stage C parquet plus skymap
   diagnostics, or the full Stage D-G SED chain. Recommended answer: produce
   Stage C first, then run the existing v4 aperture-conditioned Stage F/G branch
   with the half-year Stage C run.
