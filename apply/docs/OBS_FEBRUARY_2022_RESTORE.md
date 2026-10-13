# February 2022 observation acquisition

February acquisition was submitted on 2026-10-14 at 00:24–00:25 CST. The frozen source manifests contain **28 dates / 600 unique hourly files**, with 18–23 files per date. No February files were found in the historical reduced archive. This is input recovery only; no inference, training or time recovery runs in these jobs.

## Paths and contract

- Raw source: `/eos/lhaaso/ai/wcda/raw/2022/02??/Esg202202??_HH.root`, read through `root://eos01.ihep.ac.cn/`.
- Task staging: `/scratchfs2/lhaaso/liushijie/WCDA_obs_202202_trimmed_20261014`.
- Source control: staging `.control/`, with `manifests/`, `audit_files/`, `acks/`, `audit/`, `logs/` and `operations/`.
- ETO destination: `/mnt/newdisk/shijie/WCDA_obs/<MMDD>/`, dates `0201`–`0228`.
- ETO receiver state: `/mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-february`.
- Existing SSH ControlMaster: `/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january/ihep.sock`, account `liushijie@lxlogin.ihep.ac.cn`. Keep it available while February reception runs.

The January reduction macro and data contract are unchanged: only `pincness < 1.1`; exactly 30 retained event branches, including `vx`, `vy`, `vq`, `vt`, `vidmc`; drop `vflag`, `vnpe`, `vqsamp`; preserve selected-event order, original filename and the complete `t_eventout_h`. Full retained-branch/header read-back remains enabled on IHEP. ETO checks ROOT metadata, branch/entry/header counts and transfer bytes, fsyncs its receipt, then persists the matching IHEP acknowledgment before deleting only this run's temporary file. New files use UUID receipts without full-file hashes; legacy receipts remain supported. Original data, old archives and confirmed ETO files are retained.

## Configuration and quota

The production scripts now read an optional sibling `run_config.json`; its absence preserves the January defaults. Deploy `scripts/data/observation_february_2022.json` as `code/run_config.json` on both source and receiver. It fixes month `202202`, run date `20261014` and expected count `600`. Source manifests are immutable snapshots; the receiver requires all configured dates and exactly the configured unique names, rather than assuming 24 files per day. Month-specific filename guards and queue scans reject January names in the February run. The shard wrapper derives its control directory from the deployed snapshot and partitions the configured calendar dates.

Fresh scratch quota at 00:21 CST: **517,239,084 KiB used**, soft **524,288,000 KiB**, hard **532,466,892 KiB**. The queue remains **5.5 GiB = 5,905,580,032 bytes**, reserving **640 MiB** per initial writer, including partials and complete pending files under one shared lock. This leaves approximately **1.22 GiB below soft quota** and **9.02 GiB below hard quota** at launch, including existing control allocations. The queue does not reserve storage against unrelated projects: recheck personal quota if other tasks grow. ETO has approximately 8.8 TiB free. No raw hourly inputs are cached locally.

## Submitted jobs and receiver

Scheduler: `scheduler@schedd11.ihep.ac.cn`. Eight disjoint producers, each 1 CPU / 4,000 MB / AlmaLinux9, submitted through the official `hep_sub` environment:

| Shard | Job |
|---|---|
| 0 | 3727304.0 |
| 1 | 3727305.0 |
| 2 | 3727307.0 |
| 3 | 3727308.0 |
| 4 | 3727309.0 |
| 5 | 3727310.0 |
| 6 | 3727311.0 |
| 7 | 3727312.0 |

All job owners, commands and shard arguments were checked; requirements retain the site defaults and exclude previously failing CVMFS machines `lhws247.ihep.ac.cn` and `lhws261.ihep.ac.cn`. Submission details and initial job ads are preserved under source `operations/`. HTCondor's JSON serializes requirements as `/Expr(...)/`; the administrative edit unwraps that representation before passing the expression to `condor_qedit`. No other project's jobs were changed.

ETO receiver supervisor PID **219775** runs independently with a nonblocking process lock, using the existing January isolated `pydeps`. Inspect `pid`, `exit_code`, `sync_rolling.log`, `progress.json`, `receipts/` and eventual `completed.json` in the February state directory. Successful receipt triggers source cleanup while producers continue, with byte-budget backpressure only. Resume the same source shard to skip confirmed/pending outputs; restart the receiver only after checking its PID and lock. Do not overwrite January snapshots or receipts.

## Validation and current status

Five lightweight local checks passed, followed by **seven small source checks using the February snapshot**, Python compilation and shell syntax. These cover month/date/path isolation, duplicate/incomplete manifests, partial/pending byte accounting, concurrent reservations, v2/legacy identities, exact cuts/branches/order/header, real write-limit failure and retry, and receiver metadata/size checks. ROOT fixtures contain only five synthetic events; complete-hour processing runs exclusively in HTCondor. The ROOT macro and cache algorithm were not changed or re-benchmarked; January measurements are documented in `OBS_JANUARY_2022_RESTORE.md`.

At 00:26 CST all eight producers were submitted and idle, awaiting site matching; the ETO receiver was alive with no errors and zero confirmed files. February is **running in the background, not complete**. A new full-hour receipt and source cleanup must be verified after the first producer publishes.
