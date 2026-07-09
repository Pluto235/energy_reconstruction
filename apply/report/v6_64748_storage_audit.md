# v6 64748 storage and provenance audit

Audit time: 2026-07-09 20:43 Asia/Shanghai.

## Scope

This audit records the ETO/IHEP storage and provenance state for moving the v6 mainline from model/data identifier `64670` to `64748`.

- Main flow: `64748`, 2022-01-01 through 2022-06-30.
- Comparison-only flow: `64670`, retained only for 2022-01 and 2022-02.
- ETO compute used Slurm. IHEP time-recovery batch work used HTCondor/HRCondor logs retained under provenance.

## Storage

ETO pre-cleanup snapshot from `/mnt/mydisk/WCDA_observation_eval_64670/provenance/v6_64748_migration_20260707/`:

```text
/mnt/mydisk: 3.6T total, 3.3T used, 131G free, 97% used
/home:      184G total, 167G used, 8.0G free, 96% used

/mnt/mydisk/WCDA_observation_eval_64670: 107G
/mnt/mydisk/WCDA_observation_eval_64748: 47G
/mnt/mydisk/WCDA_observation:            14G
```

ETO post-initial-cleanup snapshot from the same provenance directory:

```text
/mnt/mydisk: 3.6T total, 3.3T used, 190G free, 95% used

/mnt/mydisk/WCDA_observation_eval_64670: 47G
/mnt/mydisk/WCDA_observation_eval_64748: 47G
/mnt/mydisk/WCDA_observation:            14G
```

Final ETO snapshot after the `64748` half-year expansion and scratch-cache cleanup:

```text
/mnt/mydisk: 3.6T total, 3.4T used, 80G free, 98% used
/home:      184G total, 167G used, 7.9G free, 96% used
/:          273G total, 61G used, 198G free, 24% used
/dev/shm:   95G total, 9.8M used, 95G free, 1% used

/mnt/mydisk/WCDA_observation_eval_64670: 39G
/mnt/mydisk/WCDA_observation_eval_64748: 107G
/mnt/mydisk/WCDA_observation_eval_64670_recycle_20260709/recovered_time_202203_202206: 7.8G
```

IHEP storage snapshot is preserved in `/mnt/mydisk/WCDA_observation_eval_64670/provenance/v6_64748_migration_20260707/ihep_backup_df_and_counts_before_cleanup.txt`. Current live SSH recheck from ETO failed on 2026-07-09: `ihep` did not resolve and `lxslc.ihep.ac.cn` timed out.

## 64670 retention

Retained local comparison data:

```text
/mnt/mydisk/WCDA_observation_eval_64670/01??/Esg202201*.root
/mnt/mydisk/WCDA_observation_eval_64670/02??/Esg202202*.root
/mnt/mydisk/WCDA_observation_eval_64670/recovered_time/01??/Esg202201*.time.root
/mnt/mydisk/WCDA_observation_eval_64670/recovered_time/02??/Esg202202*.time.root
```

Final retained counts:

| Month | Eval ROOT | Recovered-time ROOT |
|---|---:|---:|
| 2022-01 | 681 | 681 |
| 2022-02 | 600 | 600 |
| 2022-03 | 0 | 0 |
| 2022-04 | 0 | 0 |
| 2022-05 | 0 | 0 |
| 2022-06 | 0 | 0 |

Cleanup and migration records:

- Initial 2022-03 through 2022-06 local eval ROOT cleanup recorded 2,688 ROOT files and 122 day directories in `64670_cleanup_counts.txt`.
- 2022-03 through 2022-06 recovered-time ROOT files were moved to `/mnt/mydisk/WCDA_observation_eval_64670_recycle_20260709/recovered_time_202203_202206` instead of being deleted.
- Metadata, provenance, reports, scripts, and the 2022-01/2022-02 comparison set remain under `/mnt/mydisk/WCDA_observation_eval_64670`.

## 64748 half-year observation data

Main data path:

```text
/mnt/mydisk/WCDA_observation_eval_64748
/mnt/mydisk/WCDA_observation_eval_64748/recovered_time
/mnt/mydisk/WCDA_observation_eval_64748/provenance/v6_64748_halfyear
```

Validated month coverage:

| Month | Eval ROOT | Recovered-time ROOT | Entries | Entry mismatch | Bad time sidecars |
|---|---:|---:|---:|---:|---:|
| 2022-01 | 681 | 681 | 48,726,074 | 0 | 1 / 393 |
| 2022-02 | 600 | 600 | 43,696,689 | 0 | 0 |
| 2022-03 | 690 | 690 | 44,102,479 | 0 | 2 / 238 |
| 2022-04 | 649 | 649 | 37,959,710 | 0 | 0 |
| 2022-05 | 687 | 687 | 42,292,565 | 0 | 1 / 93 |
| 2022-06 | 662 | 662 | 37,659,477 | 0 | 0 |
| Total | 3,969 | 3,969 | 254,436,994 | 0 | 4 |

The apparent count `3976` from an unrestricted `find ... -name '*.root'` includes provenance backup/timefix ROOT files. The main-flow `MMDD/EsgYYYYMMDD_HH.root` count is `3969`.

## Scheduler records

ETO Slurm jobs:

| Job ID | Role | Final state |
|---:|---|---|
| 64788 | March 64748 observation inference | COMPLETED, 0:0 |
| 64789 | Initial April 64748 observation inference | CANCELLED, later rerun in smaller batches |
| 65012-65023 | June daypack observation inference | COMPLETED, 0:0 |
| 65024 | Half-year observation validation | COMPLETED, 0:0 |
| 65078 | Stage A nominal response | COMPLETED, 0:0 |
| 65084 | Stage A promotion | COMPLETED, 0:0 |
| 65085 | Stage B PSF | COMPLETED, 0:0 |
| 65087 | Stage A aperture-conditioned response | COMPLETED, 0:0 |
| 65091 | Stage C through Stage G and final report | COMPLETED, 0:0 |

IHEP HTCondor/HRCondor provenance:

| Cluster | Role | Final state |
|---:|---|---|
| 86502580 | 2022-04-25 timefix, 2 procs | Both procs terminated with exit code 0 |
| 86504612 | 2022-04-26 timefix, 1 proc | Terminated with exit code 0 |

The Condor logs are retained under:

```text
/mnt/mydisk/WCDA_observation_eval_64748/provenance/v6_64748_halfyear/timefix_0425_20260708T192544/condor_logs
/mnt/mydisk/WCDA_observation_eval_64748/provenance/v6_64748_halfyear/timefix_0426_20260708T194538/condor_logs
```

## Stage A-G key results

Stage C metadata: `/mnt/mydisk/server/projects/energy_reconstruction/apply/output/stage_c_v6_64748_split56/runs/v6_64748_split56_stage_c_halfyear/obs_events_metadata.json`.

- Processed files: 3,969 / 3,969.
- Missing time files: 0.
- Entry mismatch files: 0.
- Input entries: 254,436,994.
- Selected rows: 80,017,935.
- Rough live time: 125.6702457 days.
- Month selected rows: 202201 15,138,450; 202202 13,713,717; 202203 13,540,746; 202204 11,687,800; 202205 13,938,825; 202206 11,998,397.

Stage D:

- Run: `v6_64748_split56_stage_d_annnorm`.
- Status: `failed` / not promotable for the full 91-cell candidate grid because several excluded or diagnostic cells have invalid or fragile ROI-local background estimates.
- Downstream Stage E/F/G used the intended 27-cell baselinev4 split56 subset and passed their gates.

Stage E:

- `N_on = 372,020`.
- `B_on = 332,201.4604`.
- `Excess = 39,818.5396`.
- Formal sigma: 67.7697.
- Quality gate: passed.

Stage F:

- Preferred fit: conservative LogPar.
- `phi0 = 2.529908897e-12`, `alpha = 2.708580987`, `beta = 0.149600527`.
- `chi2 = 321.818`, `ndof = 24`.
- Quality: passed; physical flux status OK.
- Exposure: total live 125.6702457 days, source visible 42.2055001 days.

Stage G:

- 7 Nhit diagnostic points and 12 predE diagnostic points.
- Quality status: passed.
- Diagnostic-only: true; publication-ready: false.

Final reports:

```text
/mnt/mydisk/server/projects/energy_reconstruction/apply/report/crab_sed_v6_64748_split56_baselinev4_report.html
/mnt/mydisk/server/projects/energy_reconstruction/apply/report/crab_sed_v6_64748_baselinev4_report.html
```

## Runtime notes

The Stage C read path is limited by `/mnt/mydisk` HDD I/O, not GPU. A 32-worker attempt saturated disk I/O and stalled; throttling to 8 active workers improved observed throughput, and all worker processes were restored before job completion. No user Slurm jobs remain in `squeue` at audit time.
