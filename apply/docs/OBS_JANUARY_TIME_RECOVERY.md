# January 2022 event time and J2000 recovery

This workflow restores the 31-day / 681-file January observation manifest as independent friends of the pincness-only observation files. Keep every selected event, including duplicates and failures, in its original order. Final products belong under `/mnt/newdisk/shijie/WCDA_obs/recovered_time/<MMDD>/`.

## Isolated implementation and dependencies

- ETO implementation checkout: `/mnt/newdisk/shijie/energy-time-recovery`, branch `codex/january-time-recovery`, in the energy repository. Do not overwrite the unrelated inference edits in `/mnt/newdisk/shijie/energy`.
- ETO current-run state: `/mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-january-time-v3`.
- IHEP current-run control: `/scratchfs2/lhaaso/liushijie/WCDA_time_recovery_202201_20261014_v3/.control`.
- IHEP current-run temporary EOS: `/eos/user/l/liushijie/WCDA_time_recovery_202201_20261014_v3`.
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


## Prepared v3 scope (2026-10-14 06:48)

Execution candidate de2c69083b0f9fefed4f7e1c05c64a13dd69993e is committed and pushed. Full IHEP regression 3728396.0 is idle/queued, with RequestDisk=4194304 KiB and the three required node exclusions; it must pass 20 tests before production. v3 state is `/mnt/newdisk/shijie/energy/.recovery/2026-10-14/obs-january-time-v3`, config ID `613438f5e1e5424292425d3b731a2dd5`, IHEP control `/scratchfs2/lhaaso/liushijie/WCDA_time_recovery_202201_20261014_v3/.control`, EOS root with the same v3 suffix. It has no production jobs yet and concurrency=0; source control is 813 KiB. Config records the exact excluded optional legacy friend and audit reason.

The carried receipt scopes preserve c13 four files/912,934 events and cbb v2 235 files/22,962,884 events. Remaining 442 files have 45,403,551 events. Disjoint-name and total-event self-checks pass (681/69,279,369). v3 final_acceptance.py now fully verifies all three scopes using each original config and requires exact cleaned-name sets and zero queues for each. Syntax checks pass; complete ROOT acceptance is still pending production. Later inference must still apply survivor_event_indices to friend/alignment.

Both v2 source jobs are now absent: the held owned 3728077.0 was removed after identity verification; 3728078.0 history proves exit code 1 on bws1008, with exact ENOSPC stderr preserved. The old source EOS listing was empty and all 235 ready records had durable acks and cleaned states. Exactly 25 unconfirmed reservations were backed up and released under queue locking; no final output or EOS file was deleted. Evidence is in v2 control probes/retired_reservations_20261014.json and probes/retired_source_status.json. ETO receiver 66059 was then canceled only after checking its owner/command and 235 receipts; retired_receiver.json preserves its ad. Retained retry/running states are historical evidence, not active processes. Do not restart old receivers/workers. EOS user quota has 197,715,618,231 bytes free (over 20 GiB safety). Node scratch free-space history was unavailable; do not claim host-wide vs job quota attribution without new evidence.


## Regression packaging repair (2026-10-14 06:56)

Test job 3728396.0 ran 20 checks but all failed at module import because the test-only control copy omitted observation_state.py. This does not establish a behavioral regression. Archived test_3728396.err/out retain the evidence. Copied the exact helper from the frozen de2c690 snapshot; SHA-256 equality of all five runtime/test files against the v3 production package is verified. The production v3 package already contained this helper and its code/config identity is unchanged. Resubmitted one bounded full regression as 3728412.0 with logs/test_retry.out and test_retry.err, the same 4 GiB RequestDisk and three node exclusions. The regression gate remains closed pending actual success. February remains 537/600 with two zero-throughput windows; v3 has zero production jobs/requests/ready/receipts, active_workers=0, control 832 KiB.


## Scratch diagnosis before another regression (2026-10-14 07:16)

Regression 3728412.0 exited code 1 on slot123@bws1012 during g++ assembly output with ENOSPC; no tests ran. RequestDisk in history is 4,194,304 KiB. Earlier production ENOSPC was on bws1008, so this is not confined to the first node. Collector Disk advertisements (about 15 GB for sampled BWS slots) do not establish actual writable blocks/inodes or quota in the job sandbox. Do not attribute or blindly resubmit.

Two bounded 1 CPU/512 MiB diagnostic jobs are queued: 3728436.0 pinned to bws1012, and 3728440.0 pinned to the previously proven pilot node aws003. Both use the owned repairtest/.control/probes/check_scratch.sh, RequestDisk=4 GiB, and logs/scratch_probe[ _aws003 ].out/err. They inspect statvfs plus df blocks/inodes and own quota, and create/fsync/remove only a unique 1 MiB test file in their own scratch and /tmp. No existing data is removed. Owner and exact probe Cmd were verified before resource/node edits. Read their results before selecting any next full-test run. v3 remains untouched, zero claims/production jobs, and the final 681-file gate remains closed.


## Probe queue analysis (2026-10-14 07:29)

Both probes remain idle without output. condor_q -better-analyze shows aws003 has all 192 matched slots serving others and zero currently able to run the probe; bws1012 has 253 serving others and three technically eligible slots ignoring user priority. Read-only direct SSH to both compute nodes times out, so it supplies no filesystem evidence. Collector metadata shows 389 unclaimed AWS slots overall. After verifying the existing idle comparative job owner/Cmd, broadened only 3728440.0 from exact aws003 to the AWS pool, retaining 4 GiB RequestDisk and all three exclusions. No new or duplicate probe was submitted. Its historical output filename remains scratch_probe_aws003.out; read actual host from its JSON. 3728436.0 stays pinned to the failed bws1012 for diagnosis.

Prepared (not submitted) the owned repairtest/.control/probes/run_checked_test.sh: it checks actual job-sandbox free blocks >=4 GiB and free inodes >=1024, then creates/fsyncs/removes its unique 1 MiB file before exec of the unchanged frozen test entry point. bash -n passes. Use it only after probe evidence identifies a viable node/path; verify its own exact Cmd if controlling that future test job. Full 20-test success remains mandatory. No production code/config/request identity changed.


## Actual scratch capacity evidence (2026-10-14 07:44)

Both probes finished with exit 0. 3728436 on bws1012 measured 139,202,560 available bytes (133 MiB) on /dev/nvme0n1p3, a 7.0 TiB root filesystem at 100%; scratch and /tmp share it. Inodes still have 278,468 free and the unique 1 MiB write succeeds. This proves insufficient block capacity, not inode exhaustion, at measurement time. 3728440 history identifies advertised node aws001; its OS hostname is cws015. That root filesystem is also 100%, with only 1,102,540,800 available bytes and 2,159,433 free inodes. Its 1 MiB probe succeeds but it does not meet the 4 GiB gate. Empty quota output supplies no further quota claim. No existing files were removed.

A fresh bws1012 Collector sample still advertises Disk=14,364,883 KiB; it cannot substitute for actual statvfs/writability. Full probe JSON and archived output are in v3/probes/scratch_diagnosis.json and scratch_probe*.out. Next bounded probe 3728491.0 uses the same diagnostic script on different-family node lhws112 (selected from 1,215 unclaimed LHWS slots), site -wn and -wt test (5 minutes), 1 CPU/512 MiB/4 GiB, with three required exclusions. Owner and exact probe Cmd were checked before resource edits; logs/scratch_probe_lhws112.out/err. It is idle; do not launch another full regression until actual viable scratch is demonstrated. Production remains zero while February is at 537/600.


## Eligible node and checked full regression (2026-10-14 08:04)

Reverse matching showed that every lhws112 slot rejects the probe account COMMON.test.lhaaso; they allow unrelated u07/moedal groups. No account or node policy was changed. Using the documented ClassAd.matches direction (verified by true-account/false-account asserts), evaluated real node Requirements against the unchanged probe account, then replaced only the existing owned idle job placement with five currently authorized LHWS nodes (lhws132/140/220/229/304). The retained account-placement JSON records old/new requirements. No duplicate probe was submitted.

3728491 then executed on lhws132: scratch and /tmp have 712,878,235,648 available bytes (664 GiB), 467,692,952 free inodes, and successful unique 1 MiB fsync/write/remove. The 4 GiB gate is demonstrated. Its output remains named scratch_probe_lhws112.out for continuity; actual host is lhws132. Archived output and authorized_placement_3728491.json are in v3/probes.

Submitted the checked full regression as 3728592.0 using native hep_sub -wn lhws132 -wt short (30 minutes), 1 CPU/4000 MiB, RequestDisk=4 GiB and the three exclusions. Owner and exact Cmd repairtest/.control/probes/run_checked_test.sh are verified; actual account COMMON.short.lhaaso is independently allowed by the node START expression. Logs are repairtest/.control/logs/test_checked_lhws132.out/err. It remains idle; all 20 checks and successful exit are required before the production gate opens. The checked launcher remeasures its own job scratch before compilation, so a changed node condition fails closed. v3 frozen code/config identity and all 239 confirmed friends remain untouched; February remains 537/600 and production concurrency stays zero.


## Complete scheduler matching for the regression (2026-10-14 08:17)

Although lhws132 START allowed the test account, its slots advertise Memory=3957 MiB and the test requested 4000, so all slots rejected the complete job Requirements. After verifying the same owned idle checked-test Cmd, changed only 3728592 RequestMemory to 3900 MiB. This retains ample headroom for the small fixtures/build (production recovery measured under 716 MiB RSS); source production budget and frozen code/config remain unchanged. Verified both directions of matching against actual slot ads: job.matches(slot) checks site policy; slot.matches(job) checks complete job constraints. probes/memory_match_3728592.json records the evidence. It is still idle with no test output, so the test gate remains closed and no duplicate test was submitted. Future production placement must check complete resource matching as well as account eligibility; the 4000 MiB production profile does not fit this 3957 MiB slot.

ETO receiver sbatch --test-only accepts the frozen command (1 CPU/2 GiB, energy/apply), but currently predicts 2026-10-15 23:16 even with a two-hour limit. ETO clock is 2026-10-14 08:16, node advertises idle with zero allocated CPU/memory; there is no actual receiver job yet. Treat this as tentative scheduler evidence, not completion ETA. Do not change Slurm services or unrelated pending jobs; inspect the real job reason only after the regression gate permits submission.


## Full regression passed and v3 receiver active (2026-10-14 08:40)

3728592 eventually ran on lhws132, but all 20 Python checks failed at import because observation_state.py also reads the frozen filter_observed_pincness_loop_scoped.C asset omitted from the test-only copy. Its C++ ROOT self-regressions succeeded; this was still not a passing test gate. Corrected the deployment by syncing the entire frozen scripts/data directory, not a hand-picked file list. All 18 frozen files have verified SHA-256 values, import succeeds with KEEP=30, and the checked launcher now asserts the complete package hash manifest before compiling. Production v3 already had the complete directory, so no scientific code/config/request SHA changed.

Bounded corrected run 3728812.0 passed all 20 tests in 10.271 seconds and history confirms ExitCode=0, slot37@lhws132, RequestMemory=3900 MiB, RequestDisk=4194304 KiB. The displayed unreadable-ROOT errors and synthetic FAILED event are intentional negative fixtures. Archived complete logs, full-package hashes, and regression_acceptance_3728812.json are in v3/probes. The test gate is now passed.

Submitted ETO receiver Slurm66062 from energy/apply with the exact de2c690 snapshot/config, 1 CPU/2 GiB. It started immediately at 08:39:48 and is RUNNING; the prior test-only 39-hour prediction was not its actual scheduling result. First sample has six durably dispatched requests (seven local request records) while the initial serial 442-request pass runs. No production source workers have been submitted, concurrency stays zero while February is 537/600, and v3 output receipts remain zero. The original 239 accepted friend identities remain carried unchanged. Inspect actual request/ready counts during initial dispatch rather than missing/stale progress.json.

While the verified test node was busy, one full-4000-MiB alternate probe 3728753.0 ran on lhws235 and measured only 12,570,624 free bytes on a 100% root filesystem; it is rejected for recovery. It finished before the attempted ad edit, so history RequestDisk is the site default 2 KiB, not a 4 GiB reservation. It performed only the bounded 1 MiB write/remove diagnostics; no heavy test or recovery ran there. Do not misreport this as a viable full-profile node or repeat the finished probe. Future production remains subject to fresh actual scratch capacity and complete two-way scheduling/account checks.


## User changes priority to January (2026-10-14 09:35)

The user explicitly requests pausing February acquisition and completing January time/J2000 recovery first. This supersedes the previous February-throughput throttle. February has 537/600 receipts; no source jobs with its exact control Cmd remain on schedd11. Verified ETO wrapper PID219775 and receiver PID219777 against owner and full command line, then SIGSTOP both. Their state is T; February monitor_pause.json records the pause, original commands, and no automatic resume. Existing receipts, finished files, queue contents and partials are retained. Do not resume February without user instruction.

All442 v3 requests are durably dispatched and Slurm66062 remains the receiver. Runtime user_priority_override.json now requests two January workers. The runtime priority checker verifies February receiver processes stay stopped and follows this explicit override instead of reducing January concurrency because February throughput is zero. A January monitor_pause.json still stops this recovery. Frozen scientific code/config/request identities are unchanged; old4 and v2235 receipts retain their original identities.

Submitted January workers3729557.0(index0) and3729558.0(index1) under source concurrency0, then verified Owner=liushijie, exact v3/code/run_observation_time_shard.sh Cmd, actual account LHAASO.lhaaso.default, 4GiB RequestDisk and all three excluded nodes. Both matching directions succeed against40 lhws132 slots. Production memory is now3900MiB rather than4000: the healthy node has3957MiB per slot, measured recovery RSS was below716MiB, and the complete20 tests passed at3900. This is a resource placement adjustment; no algorithm, alignment, conflict, fallback, capacity or publication rule is relaxed. Earlier actual scratch evidence on this node showed664GiB available and successful fsync. Source probes/january_first_placement.json records job/account/resource matching. After setup, the runtime priority checker successfully sets concurrency2. Jobs were initially idle; genuine production and delivery must be checked from logs/ready/receipts, not inferred from submission.

Final acceptance remains mandatory across all three scopes:4+235+442=681 files and69,279,369 events. Full ROOT validation is run through ETO Slurm before claiming month completion. Future energy inference must apply final survivor_event_indices identically to friend/alignment.


## January source scheduling follow-up (2026-10-14 09:48)

Both verified January workers remain idle because all40 healthy lhws132 slots are busy. No production has been inferred from submission. A single bounded default-account scratch probe3729640.0 initially targeted currently free lhws283, which also became entirely busy before matching. After owner/exact-probe-Cmd/idle checks, broadened that same diagnostic to the eligible LHWS pool while retaining4GiB disk and excluding the three prohibited nodes and known-full lhws235. Bidirectional checks found27 authorized free slots at sampling time; actual probe execution and filesystem evidence remain required before broadening production placement. Its exact Cmd is repairtest/.control/probes/check_scratch.sh, outputs logs/scratch_probe_january_first_283.out/err retain historical filenames; actual host must come from its JSON/history. Do not repeat the diagnostic or submit duplicate source workers while it is queued.

The user's February pause also pauses its separate wcda hourly monitoring automation; January energy monitoring stays ACTIVE and must continue until all681 files pass final acceptance. No automatic February resume is authorized. Light pause/placement evidence is saved in v3/probes and mirrored to the Mac. Queue waiting is separate from compute ETA. Source old3728077 history identifies prior successful node bws1015, but all256 slots there are currently busy; this is not current writable-capacity evidence. Own schedd11 queue has only these3 idle recovery/probe jobs, so another own queued workload is not being silently prioritized ahead of them.


## Alternate healthy source node verified (2026-10-14 10:09)

Diagnostic3729640.0 completed with ExitCode0, history slot41@lhws161.ihep.ac.cn, actual hostname lhws161, RequestDisk4194304KiB and default account LHAASO.lhaaso.default. Its scratch and /tmp share /dev/sda3, with356,981,579,776 available bytes (333GiB),467,389,047 free inodes and successful unique1MiB write/fsync/remove. Empty quota output is not expanded into a quota claim. Historical output filename scratch_probe_january_first_283.out remains unchanged.

After verifying both existing source jobs3729557.0 and3729558.0 are owned by liushijie, have the precise v3 frozen worker Cmd and are idle, expanded only their placement from measured lhws132 to measured lhws132 OR lhws161. Preserved3900MiB memory,4GiB scratch, all three node exclusions, account and complete site-generated requirements. Both matching directions succeed for56 lhws161 slots, with2 unclaimed at sampling time. No new production job was submitted and no frozen scientific/config/request identity changed. The full probe and before/after requirements are archived in v3/probes/healthy_lhws161_placement.json. Actual process/log/ready/receipt evidence is still required before declaring production resumed; final acceptance remains the full three-scope681-file gate.


## Production resumed and receiver recovered after reboot (2026-10-14 11:20)

Both original January source jobs3729557.0 and3729558.0 started at10:25:30 on lhws161 slots42/31 with the exact frozen v3 Cmd. Direct job inspection through condor_ssh_to_job confirms both Python workers and both persistent ROOT recovery processes, about696–741MiB process RSS and359,775,031,296 available scratch bytes. No source job was repeated or replaced. At11:17 the source has95 ready files /10,309,341 events, all status0, zero fallback and no failed requests. Worker0 has46 ready/112 remaining, mean elapsed66.26s; worker1 has49 ready/235 remaining, mean elapsed62.29s. Because requests are assigned by day parity, use the slower remaining shard for ETA: about4.1h of compute from this sample, with changing day composition and final validation excluded. Do not use a simple equal split or the earlier53s baseline.

ETO rebooted at11:04:51 (boot ID3b0f1619-3ddc-4910-a360-0caede9585fc); its previous receiver process and IHEP SSH master were absent although Slurm66062 still reported RUNNING. Its last pre-reboot durable count is30 files. Verified66062 owner=server, precise snapshot command and energy/apply cwd, archived reboot evidence, then canceled only this stale task job. No Slurm service, other job or original product was changed. February stopped processes also disappeared on reboot; retain its explicit pause, preserve old PID evidence as historical_processes and leave active processes empty so reused PIDs cannot be mistaken for a resumed receiver. Its wcda monitor remains PAUSED.

Direct Mac-to-ETO IPv4/IPv6 timed out. Existing PDC-shell forwarding reaches ETO normally, and direct ihep-shell also works. Use existing SSH identities only: normal Mac SSH authentication with AddKeysToAgent=yes and temporary agent forwarding rebuilt the same task master socket on ETO; no private-key/password content was read as data or transferred. The previous stale socket was removed only after failed master check, same-owner/socket-type verification. New master11064 was verified with a real IHEP hostname command.

Submitted replacement Slurm66067 from energy/apply with1CPU/2GiB and the unchanged de2c690 snapshot/config. Its actual receiver PID11359 is running; durable receipts/cleaned increased30→37 after restart. At that sample the month is4+235+37=276/681 confirmed, while95 v3 files are produced; these are different counts. Source queue7,121,975,296bytes is below16GiB. Initial renewed progress.json can lag while it consumes the ready backlog, so use durable receipts and source cleaned evidence. Keep confirmed outputs and all three scope identities; final681-file full Slurm acceptance is still pending.

Fallback Mac transport (no SSH config or service changes):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o RemoteCommand=none -o RequestTTY=no \
  -o 'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o RemoteCommand=none -o RequestTTY=no -W "%h:%p" PDC-shell' ETO
```

ETO-to-IHEP continues using the original task socket. For condor_ssh_to_job with Python, pass a single fully shell-quoted command (for example shlex.join) because the tool joins command arguments before remote eval; an unquoted -c body produces a shell syntax error and proves nothing about the job.
