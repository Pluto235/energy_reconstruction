# January 2022 observation recovery — optimized production

Status: the optimized workflow has passed complete-hour tests, 4/8-worker transfer cohorts, and has been submitted for full-month continuation. At the end of the cohorts, 98 of the 681 files were confirmed on ETO; monthly completion remains pending. Source/target paths and the original 31 manifests are unchanged.

## Data contract

Only `pincness < 1.1`. Keep all 30 branches defined in `scripts/data/filter_observed_pincness_loop_scoped.C`; remove `vflag`, `vnpe`, `vqsamp`; retain `vx/vy/vq/vt/vidmc`. Preserve selected event order, original filenames and the complete `t_eventout_h`. No other physical cuts, inference, training or time recovery.

- Inputs: `/eos/lhaaso/ai/wcda/raw/2022/01??/Esg202201??_HH.root`.
- Source work: `/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013`.
- Task-owned staging: `/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013/<MMDD>/`.
- Destination: `/mnt/newdisk/shijie/WCDA_obs/<MMDD>/`.
- Preserved archive: `/scratchfs2/lhaaso/liushijie/WCDA_observation_20220101_20220630/0131/`, 23 files / 4,941,889,023 bytes.

The 681 names come from the existing manifests. No assumption of 24 files/day is made. Daily completion requires exactly the original day's names; monthly completion requires all 31 days, all 681 names, and an empty staging queue.

## Measured ROOT changes

New files no longer get an extra source `GetEntries(selection)` pass. The necessary pincness-only scan returns selected indices, counts, header schema and timings; the subsequent loop reads only selected events' retained branches. All hit branches are not read for rejected events.

The old cache learned only pincness and still contained one branch in the hit phase. The selected hit loop now discards that cache, creates a 32 MiB cache, explicitly registers all 30 retained branches and stops cache learning. The scan has its own 8 MiB pincness cache. ROOT counters measure client read calls/bytes, not server-side RPC counts.

Complete-hour HTCondor tests (seconds; no data hashes computed even in the reproduced baseline):

| Input | Old I/O path, first run | Old I/O path, warm repeat | One scan, hit cache off | One scan, 32 MiB | One scan, 128 MiB | Hit read calls: old → 32 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 20220105 hour 00 | 257.8 | 234.1 | 174.0 | 139.7 | 253.9 | 136,438 → 1,744 |
| 20220106 hour 00 | 264.2 | 194.8 | 215.1 | 152.4 | 151.9 | 138,692 → 1,613 |

Against the warm-order control, mean preparation fell from 214.5 to 146.1 seconds (about 32%). The first-run baseline includes additional startup/cold-file effects; its approximately 44% reduction is not treated as a pure cache effect. Explicit caching reduced read calls by roughly 80× while reading approximately 3–4% more hit bytes. The 128 MiB results were less consistent and used more memory, so production uses 32 MiB.

All selected events' 30 retained numeric leaves, including variable-length hit arrays, were compared directly with the baseline in original order. All header string values and header schemas/counts were compared. No full-file hash was used for equivalence. Cache off / 32 / 128 MiB all passed on both complete files.

Warm full read-back cost 3.4–4.0 seconds versus 3.1–3.6 seconds for sampling. The small saving does not justify weaker checks: production retains **full read-back of every output's retained event branches and full header**, checking pincness in that same pass. Source files that ROOT marks recovered/zombie, missing branches, read failures, Fill/write/close failures and count mismatches are rejected.

ROOT benchmark jobs: `3725506.0`, `3725507.0`; warm old-path controls: `3725599.0`, `3725600.0`. Detailed results and frozen baseline code live under source `optimization_20261013/`.

## Durable receipt migration without new hashes

`audit_files/<name>.json` version 2 carries a random receipt UUID, source/output counts, sizes and timings. New records contain no SHA-256 or other full-file hash. Existing SHA records and plain hash acknowledgments remain valid legacy identities and are retained; the receiver does not recompute or invent their hashes.

ETO reads each received ROOT's trees, exact event branch set, per-branch event counts, complete header branch set when recorded, header/event counts, ROOT header/end boundary and transferred byte count. Source-side full content checks remain mandatory. ETO uses uproot for metadata only; its isolated runtime packages are under receiver state `pydeps/`.

The order is: copy resumably into ETO `incoming/` → verify ROOT/bytes → fsync file and publish final name → fsync a local receipt in `receipts/` → atomically fsync the source acknowledgment → unlink only this task's temporary source file. Confirmed ETO files are never overwritten. An inconsistent confirmed target is preserved and reported as an error. Unconfirmed conflicting targets are retained under `preserved_unconfirmed/`.

Legacy migration reused and checked 82 existing targets, including pending old outputs, without rehashing. A legacy plain-hash acknowledgment cannot confirm a new UUID. Connection/transfer errors are retried; durable receipts let receiver restarts resume without recopying confirmed files. If the receiver dies, producers eventually wait for capacity rather than filling scratch.

## Byte-bound producer/receiver queue

Producers continue immediately after publishing a complete, validated output. They wait only when a shared byte reservation cannot fit. Per-file locks prevent duplicate processing and allow a restarted worker to reclaim only its own stale partial. A shared NFS `flock` protects reservations, promotion and cleanup; the inspected mount uses remote locking (`local_lock=none`).

The budget includes **the maximum reserved size of every writing `.partial` plus all complete pending files**, and also counts any untracked task outputs. A process-level `RLIMIT_FSIZE` enforces the reserved maximum during ROOT writes. An oversized file fails without publishing, its partial is removed under the queue lock, and it retries with a larger reservation. Normal failures have three attempts; persistent errors remain visible in job logs. A missing audited file with no matching receipt fails visibly rather than being declared complete.

Production quota was rechecked: 516,983,868 KiB used, soft limit 524,288,000 KiB, hard limit 532,466,892 KiB. Capacity is the minimum of 8 GiB, available soft-limit space minus 1 GiB, and hard-limit space minus 6 GiB, rounded down to 256 MiB. Current capacity: **5.75 GiB (6,174,015,488 bytes)**; initial file reservation: **640 MiB**. Eight reservations total 5 GiB. Full original hourly inputs are never cached on personal scratch.

`queue_config.json`, `queue_ledger.json` and `queue_status.json` are under the source work directory. Reservations survive worker failure; the per-file lock permits safe stale-partial recovery after the previous owner exits. Source/ETO acknowledgment records remain durable across restarts. Only the explicitly task-owned staging tree can be cleaned; archive hard-link aliases can be unlinked without removing the archive.

## Four versus eight producers: actual end-to-end cohorts

Both cohorts started at a batch barrier after all workers were ready and ran through real ETO transfers and cleanup. Four workers processed two actual manifest inputs each (January 10–13); eight workers processed one each (January 14–21). Inputs differ, but total original event counts match within 0.5% and output bytes within 2.1%; the comparison is a small practical throughput test, not a claim of linear scaling.

| Producers | Files | Original events | Output GB | Source window s | To final ACK s | MiB/s through ETO ACK | Peak RSS MiB / worker |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 40,421,436 | 2.926 | 493.9 | 531.6 | 5.25 | 706.2 |
| 8 | 8 | 40,220,949 | 2.988 | 246.2 | 347.9 | 8.19 | 693.0 |

Eight producers delivered about 56% more output bytes per second through final confirmation. Peak reserved/queued usage was 3.37 GiB at four producers and 5.00 GiB at eight, both below 5.75 GiB. All sixteen files reached ETO and their temporary copies were safely removed. Four-worker transfers took 8.5–16.4 seconds per file; ETO metadata checks took 0.9–2.0 seconds.

The eight-worker source window approximately halved, while final confirmation improved less because transfers/acknowledgments drain a larger burst. Main remaining costs are retained hit reading/decompression/writing and shared EOS/Lustre waits; the serial receiver and source confirmation/cleanup add a tail. Eight-worker CPU time was about 129–146 seconds versus 187–208 seconds of source preparation per file, so waiting remains material. Cache changes did not eliminate all I/O waits. Actual eight-worker backpressure/reservation overhead in this small cohort was below 0.3 seconds/file; continuous production is still bounded when transfer backlog grows.

Cohort jobs: four `3725607`–`3725610`; eight `3725681`–`3725688`, all `.0`. Source `optimization_20261013/cohort4/summary.json` and `cohort8/summary.json` retain individual timing records. Each source record separately reports scan, selected hit read/write, validation, fsync, CPU, peak RSS, reservation/backpressure, failed attempt time and retry count. ETO receipts/logs report transfer, metadata verification and confirmation/cleanup.

## Active production and checks

Selected concurrency: **8 producers**, each 1 CPU / 4 GB, AlmaLinux9, default 15-hour walltime on `scheduler@schedd11.ihep.ac.cn`.

Production job IDs: `3725732.0`, `3725733.0`, `3725734.0`, `3725735.0`, `3725736.0`, `3725737.0`, `3725738.0`, `3725739.0`.

The entrypoint is `code/run_observation_shard.sh shard INDEX 8`, with disjoint dates and unchanged complete manifests. Resubmitting the same shard skips confirmed/pending outputs and rebuilds only unfinished work. Queue limits permit fewer simultaneous writers when backlog or a larger reservation consumes capacity.

The previous four jobs `3725172.0`–`3725175.0` were first checked for owner/command/workdir, then held after all current files were complete. Their snapshots, manifests, receipts and logs are preserved under source `optimization_20261013/`. They must not be released alongside the new jobs. Retirement status is recorded in `retired_jobs.json`.

Inspect source `logs/optimized_shard<N>.out/.err`, `audit_files/`, `acks/`, `audit/` and queue state. ETO receiver state is `/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january`; inspect `progress.json`, `receipts/`, `sync_rolling.log` (now links to `sync_optimized.log`), `pid`, `exit_code`, and eventual `completed.json`. The previous log is preserved as `sync_rolling_pre_optimization.log`.

Small regressions cover strict cuts/NaN/order/header/30 branches, all cache modes, ROOT write failure, kernel-enforced partial limits and retry growth, nonblocking publication, legacy/v2 receipt compatibility, wrong-UUID acknowledgment rejection, concurrent byte reservation limits, metadata/size/path validation, and cleanup path guards. Six source unit tests, Python compilation and shell syntax passed. Full-hour ROOT checks ran only in HTCondor. The Mac is outside the data path; no SSH key/password was read or copied.

ETA from these small cohorts: roughly **4–7 hours remaining** for the approximately 583 files after the cohort checkpoint. The lower end assumes steady production overlaps transfers; the conservative end extrapolates the measured eight-file launch-to-ACK window. EOS/Lustre load and varied hourly file sizes can change this. Monthly completion is not claimed. New transfers have ROOT/byte checks and full source read-back, but no bitwise hash comparison. Energy inference, training and time recovery are outside this run.
