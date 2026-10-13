# January 2022 event time and J2000 recovery

This workflow restores the 31-day / 681-file January observation manifest as independent friends of the pincness-only observation files. Keep every selected event, including duplicates and failures, in its original order. Final products belong under `/mnt/newdisk/shijie/WCDA_obs/recovered_time/<MMDD>/`.

## Isolated implementation and dependencies

- ETO implementation checkout: `/mnt/newdisk/shijie/energy-time-recovery`, branch `codex/january-time-recovery`, in the energy repository. Do not overwrite the unrelated inference edits in `/mnt/newdisk/shijie/energy`.
- ETO current-run state: `/mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-january-time-v2`.
- IHEP current-run control: `/scratchfs2/lhaaso/liushijie/WCDA_time_recovery_202201_20261014_v2/.control`.
- IHEP current-run temporary EOS: `/eos/user/l/liushijie/WCDA_time_recovery_202201_20261014_v2`.
- Existing ETO IHEP connection: `.recovery/2026-10-13/obs-january/ihep.sock`. Do not copy credentials.

Production uses task-owned, unmodified Cog and SLALIB copies in `.control/deps/Cog` and `.control/deps/slalib`. The worker verifies `deps/dependency_sha256.txt` before compiling. Cog was copied from the historical algorithm's dependency directory; the provenance is retained in `deps/ORIGIN.txt`. No runtime path points into another user's home. The reference source is byte-identical to the user's `/home/lhaaso/liushijie/time-recover/src/recover_wcda_time.cc` and is stored as `.control/deps/legacy_reference.cc`.

The public `hushicong/public/v0.99` and `v1.0` headers lack the required `Ttai_t` conversions, `Tmjd_t::getns4`, `papi::hcs2eqa` and `papi::eqa2eqm` API. A v0.99 batch compilation failed before any production request or output publication. Do not replace the pinned dependency with those headers or change conversion formulas to make them compile.

## Validation and publication

`recover_observation_time.cc` builds only the current day's old event index. Old main/friend counts must agree; successful times must reproduce the Cog time conversion exactly. Old files with unauditable fallback are excluded from reuse. Conflicting valid times stop publication of the affected target. Coordinates require identical angles and evidence, or an exact recomputation check. Missing time is recovered by reduced file, with fallback allowed only when the exact name is absent and the alternative is unique.

Raw scalar selection is collected once, retaining order and duplicates. Missing event indices are grouped by reduced key; only those rows are read into the bounded scalar cache. `reduced_rows_read` counts these actual reads, so repeated missing events retain separate output rows without repeated source reads. Loading every row of each reduced source is prohibited: the first real-hour trial exceeded the 512 MiB cache limit with that approach.

Each file contains the original eleven `t_recovered_time` branches and a `t_alignment` tree with entry index, four integer event keys, theta and phi. Status 0 means valid time and coordinates; statuses 1–5 preserve the legacy failure meanings; status 6 preserves valid time when coordinates fail. JSONL records identify every failed row and successful fallback, including available time and source.

The producer performs full ROOT readback and EOS readback before marking a request ready. Ambiguous reduced keys are rejected rather than choosing a path by filename. The ETO receiver compares all six alignment scalars, types, count and ordering in bounded chunks. It then publishes without replacing existing targets, persists the local receipt, obtains a durable IHEP acknowledgment, and only then removes that request's temporary EOS files. Retries retain byte reservations and stop after three computation attempts. Transfer failures preserve finished products.

Run the lightweight protocol suite locally:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/data -p test_observation_time.py -v
bash -n scripts/data/run_observation_time_shard.sh scripts/slurm/sync_observation_time.sbatch
```

Run ROOT fixtures through IHEP HTCondor using the worker entrypoint's `test` mode. The suite checks reordered and repeated events, cross-file reuse and conflicts, count mismatches, coordinate provenance and failure, strict fallback, truncated reads, failed writes, and publication/acknowledgment restarts.

Use `hep_sub -g lhaaso -os AlmaLinux9 -m 4000 -cpu 1 -schedd schedd11`; production requests 4 GiB scratch and excludes `lhws247.ihep.ac.cn` and `lhws261.ihep.ac.cn`. Preserve site-generated matching requirements when adding exclusions. Direct Condor queries need both `-pool cm01.ihep.ac.cn` and `-name scheduler@schedd11.ihep.ac.cn`.

The ETO receiver runs through `scripts/slurm/sync_observation_time.sbatch SNAPSHOT_ROOT CONFIG`, with 1 CPU / 2 GiB. Deploy a committed source snapshot, and bind requests to its SHA and configuration identity. The first receiver uses `--names Esg20220101_00.root`; no full-month requests are issued until its reference comparison, reception, acknowledgment and cleanup succeed.

## Execution evidence (2026-10-14, Asia/Shanghai)

- ETO main has advanced to `ac4f7d3`; the existing isolated recovery branch starts at `0eb6270`. Other tasks have uncommitted inference edits on main.
- January observation progress reports 681/681 files, 151,096,492,606 bytes, no errors, and no queued data.
- Prior Slurm probes `66035` and `66038` completed with exit 0. The latter received and verified a 5,925-byte ROOT probe, persisted acknowledgment on IHEP and cleaned its EOS temporary copy. Evidence remains in the ETO state directory.
- Lightweight regression: 9 protocol tests passed; 6 ROOT-dependent tests skipped locally. Pinned-environment IHEP job `3727568.0` passed the earlier 14-test suite. Job `3727578.0` reproduced silent selection of duplicate reduced keys; the fixed implementation and an additional interrupted-upload regression are checked by job `3727582.0` before production.
- Rechecked personal EOS quota: 802,284,381,769 / 1,000,000,000,000 bytes; 9,731 / 250,000 files. Recheck scratch, applicable group quota and the February observation queue before reserving recovery capacity. Control budget is 64 MiB; EOS queue is at most 16 GiB with 20 GiB free safety margin.
- Implementation commit `a190b7b` is pushed. Fresh ETO Slurm job `66048` verified the IHEP connection from the batch node. The first pilot, IHEP `3727653.0`, failed before upload because its whole-source reduced cache exceeded 512 MiB; its three failed attempts and logs remain in the original `20261013` control directory. Receiver `66049` was canceled. No friend was published. Default-class predecessor `3727621.0` was retired while idle.
- The selective-cache regression first failed in job `3727671.0` with `Reduced cache reads unused source entries`; the revised implementation passed all 15 tests in job `3727677.0` (227.174 seconds). Revised requests use a separate run directory and configuration bound to the new commit, preserving the original immutable requests.
- IHEP audit job `3727662.0` verified and saved all 393 historical `0101_22` failed event keys in the original control directory's `probes/legacy_failures_0101_22.json`.

File delivery, successful event recovery and temporary cleanup are separate acceptance totals. Do not report the month complete while any manifest file is missing or conflicted. Report the old `0101_22` 393 failures separately against current keys and list any remaining failures.

## Accepted files and transport revision

The original January receipts contain **69,279,369 selected events in 681 files**. All files fit the validated 4 GiB scratch envelope; the largest target has 363,560 events (`Esg20220126_22.root`).

Commit `c13e789` produced four accepted files, totaling **912,934 events**, all with valid time and coordinates: `0101_00`, `0101_01`, `0101_22` and `0102_00`. Their receipts, immutable configuration and acknowledgments remain in the prior `obs-january-time` / `WCDA_time_recovery_202201_20261014` run. Its `retired_run_acceptance.json` records the exact names and zero queued bytes. These files must not be recomputed or overwritten by the current run.

Both pilot hours matched the unmodified reference algorithm exactly, row by row. `0101_00` reused 228,604 times and recovered 91,654 new times; its recovery, reference and upload took 87.406, 135.595 and 2.086 seconds respectively. Peak RSS was 715,052 KiB. `0101_22` contains 116,551 successful events. Slurm audit `66056` confirmed that all **393 historical failures were restored**, with zero remaining and zero absent target keys. Pilot receipts and per-event historical audit remain in the prior run directory.

The first production workers `3727913.0` / `3727914.0` each generated one valid product, then stopped safely because queue accounting interpreted the timestamp column of the actual seven-column `xrdfs -l` listing as its size. Receiver `66057` was canceled. Slurm job `66058` verified, published, acknowledged and cleaned both retained products using their original request identities. The prior producer concurrency is fenced at zero; its unprocessed requests remain as evidence and must not be restarted.

The current transport revision reads the fourth column's byte count, skips directories, and rejects unknown formats. Regression tests include a real listing layout, partial files, nonzero directory aggregate sizes and malformed output. The C++ recovery source and scientific conversion are unchanged. A live IHEP probe checks upload/readback, actual listing sizes, reservation capacity and cleanup before restarting production.

The current run's expected name set is the original manifest minus the four explicitly accepted files: **677 files / 68,366,435 events**. Its source commit and configuration are new, and every new request retains its own immutable identity. Final January acceptance must combine the four prior receipts with the 677 current receipts, require an exact 681-name union without duplicates, revalidate every friend against its original observation using the appropriate frozen configuration, and confirm both temporary EOS queues are empty. A current-run completion marker for 677 files alone does not prove full-month acceptance.

The IHEP site walltime table gives `lhaaso_default=54000` (15 hours) and `lhaaso_short=1800` (30 minutes). Use the default category for production. Exclude `lhws247`, `lhws261`, and the observed startup-stalled `bws0995` from this task's future matches. The latter's original job `3727721.0` never claimed a file; its diagnostics are retained, and the validated retry was `3727837.0` on `aws003`. Do not alter other jobs or cluster services.

Downstream energy inference must apply its final `survivor_event_indices` to this friend and alignment tree, or reassociate by event key. These friends correspond to the complete pincness-only input, before additional inference cuts.

Parser revision verification: IHEP job `3728070.0` passed all 17 tests in 13.125 seconds. Live probe `3728064.0` verified a 19-byte partial, a 99/100-byte reservation boundary, rejection above capacity, exact EOS readback and zero bytes after cleanup.

## Current production and follow-up

Frozen execution commit: `cbb1ca20fac9091db006c3951d01e49a1fd9bdbf`; current configuration ID: `e0b8da048b870b0e8706a6109c3db6f2`. IHEP producers are `3728077.0` (worker 0, observed on bws1015) and `3728078.0` (worker 1, observed on bws1008). Both have produced successive files with the repaired byte accounting; the first checked snapshot had 28 ready products and no producer errors. ETO receiver is Slurm `66059`.

The receiver initially dispatches the complete remaining request set serially. Until that pass finishes, `progress.json` can still be absent or unchanged while producers make durable EOS products. Inspect source `requests`, `ready`, job logs and states as well as destination receipts. Ready products are not counted as delivered until full ETO verification and acknowledgment finish.

A 15-minute Codex heartbeat named **Energy 一月时间恢复验收** (automation ID `energy`) continues this chat's task through final validation. It records monitoring history and updates ETA from sustained production, with scheduler and backpressure waits separate. The runtime `check_observation_priority.py` watches the ongoing February acquisition without altering its jobs; it reduces only this recovery's concurrency after two slow 15-minute windows, and restores two lanes when acquisition finishes. A user pause must also pause the heartbeat and create `monitor_pause.json`.

After current-run completion, submit `final_acceptance.py` from the v2 state directory to ETO Slurm with 1 CPU / 2 GiB and the existing observation `pydeps` on PYTHONPATH. It rechecks all 681 friends against their observations using their respective frozen configurations, verifies the exact manifest union and total 69,279,369 rows, checks exception accounting, and confirms both source queues are empty. Only its successful `final_acceptance.json` establishes full-month acceptance. Update this document and the devlog, commit and push the final report, then stop the heartbeat. Do not archive the chat automatically.


## 2026-10-14 06:28 incident and priority fence

The v2 receiver 66059 remains active; 235 v2 friends plus the original four have durable receipts. The 677 requests were dispatched. February remains at 537/600 confirmed files and both preceding 15-minute windows have zero confirmation throughput. The priority checker therefore changed active_workers from one to zero. Worker 3728077.0 was held only after verifying its owner and exact current Cmd: the frozen worker checks concurrency only outside the request scan and would otherwise continue claiming within that scan. Worker 3728078.0 has already exited.

Its stderr proves node scratch ENOSPC while creating the request directory after an output write failed; 22 January 6 requests also retain attempt-one retry states because old optional friend Esg20220106_13.time.root is a recovered/zombie ROOT. No affected retry is accepted. Source ready/cleaned records and receipt identities remain unchanged; reservations remain accounted. Source control uses 14,920 KiB, below 64 MiB. Incident metadata and log tails are preserved in the v2 state and IHEP probes/incident_20261014_0625.json.

Do not release or blindly resubmit these workers. Repair the per-request concurrency gate and scratch failure handling, test the unreadable optional legacy exclusion without weakening exact reduced-ies unreadability or conflict rules, then create a new frozen deployment for only unconfirmed requests while carrying both earlier receipt scopes. Do not edit v2 code/config/request SHA in place. Reconcile stale running states and task-owned partials only after proving the source job is stopped and publication absent. Final acceptance must include all preserved scopes, exactly 681 files and 69,279,369 events. No completion ETA is asserted while priority is zero and recovery migration remains outstanding.


## Repair regression candidate (2026-10-14 06:43)

Three regressions reproduce the incident on the frozen source: a second request is claimed after concurrency becomes zero, request-directory failure escapes without a retry state, and an explicitly audited corrupt legacy friend remains reusable. The minimal Python repair checks concurrency before every request, includes mkdir in the bounded error handler, and records excluded legacy friends in the new immutable config while retaining their manifest rows with reuse=0. The C++ algorithm, exact-ies behavior, conflict rules and published products are unchanged. All 14 lightweight protocol checks pass; full 20-test IHEP run 3728396.0 is pending. Production remains fenced until this run passes and a new identity scope is deployed.
