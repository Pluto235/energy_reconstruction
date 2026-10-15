# January 2022 observation Nhit friend: final delivery

Final independent audit: **2026-10-16 04:09:34 Asia/Shanghai**, Slurm **66179**.
Closure verification: **04:09:41**, Slurm **66180**. Both exited 0.
All **31 days / 681 files / 69,279,369 events** passed. The failure list is empty.
The two explicitly authorized poolused exceptions remain separately identified.
`completed.json` was written only after the full audit succeeded.

## Delivered data and scope

| Product | Files | Events | Actual bytes |
|---|---:|---:|---:|
| Preserved `legacy_with_charge` | 55 | 5,757,151 | 1,523,864,929 |
| New `scalar_only` | 626 | 63,522,218 | 705,546,891 |
| Total | 681 | 69,279,369 | **2,229,411,820** |

The mixed product is **2.229 GB / 2.076 GiB**, not an extrapolated scalar-only
month. These are actual ROOT file sizes; audit/log storage is separate. All 55
inherited files retained their original bytes and receipts, including the three
pilot reference friends. No accepted output was overwritten or re-extracted.

Formal data: `/mnt/newdisk/shijie/WCDA_obs_friends/nhit/full/<MMDD>/Esg202201DD_HH.nhit.root`.
Original inputs: `/mnt/newdisk/shijie/WCDA_obs/<MMDD>/`. The frozen restoration
manifest, existing input UUID/size/mtime/header snapshots and old friend manifest
agree on 681 inputs and 69,279,369 events. No inference-output manifest was used;
previously deleted events were not restored. Original inputs total 151,096,492,606
bytes. Snapshot verification does not constitute a newly computed independent
SHA256 over all those input bytes.

The sole IHEP event cut is `pincness < 1.1`. Nhit uses the complete raw hit arrays,
before model truncation/sampling/padding/normalization:

```python
sum((vqsamp >= float32(0.3)) & (abs(vt) < 30))
```

Array lengths must equal nv and participating charge/time must be finite.
No vidmc is read, no new friend stores vqsamp, and no surrogate native
nq03t30s0/1/2 branch is created. No poolused cut or other physics cut is applied.
No model inference, MC, PSF, aperture or background calculation was performed.

## Acceptance and coverage

Both candidate and target four-key identities are checked for duplicates before
alignment. Final friends follow target order, with contiguous input_entry and
verified nv, ihep_entry bounds/uniqueness, segment identity and pool correspondence.
Transfer and final digests, complete scalar readback and source tables are checked.
Only successful candidates are atomically published without replacement; durable
ETO receipts precede source acknowledgments and cleanup.

Final coverage is **100% of files and events**. Accepted duplicates **0**, missing
targets **0**, excluded surplus candidates **0**. Missing/surplus identities would
be retained in per-file evidence; none remain in the completed catalog.
`t_nhit`, `t_segments` and provenance expose the common scalar interface:
`irun, ies, iseq, ievent, input_entry, ihep_entry, nv, nhit, poolused, segment_index`.

**Acceptance change:** ETO does **not** independently recompute Nhit from original
charge and does **not** compare all 30 common input branches. Source calculation,
digest/full readback and key/nv/source alignment apply to the new production.
Exact independent Nhit/bin comparison applies to the three complete pilot files,
582,417 events; it is not a claim of independent charge recomputation for the whole
month. The inherited files also retain their original acceptance evidence.

Across per-file source tables there are **107,953 segment rows**: **107,951** use
verified real headers and **2** use the explicit user authorization. This is a sum
of file-local segment rows, not a count of globally distinct segments.
Header-backed events: **69,277,566**; user-confirmed events: **1,803**.
Every delivered event has `poolused=7`; non-7 retention is proven by fixtures,
not by an observed non-7 event in this month.

| Authorized file | Exact (irun, ies, iseq) | Affected events | Source |
|---|---|---:|---|
| Esg20220102_12.root | (139579, 1, 3336) | 146 | user_confirmed_override |
| Esg20220103_05.root | (139579, 2, 4700) | 1,657 | user_confirmed_override |

The two reduced headers remain unavailable; the user's private verification is
not independently inspected evidence. Reduced-header UUID/digest stay empty and
`reduced_header_missing=1`. The sampled-source header identity digest is distinct
from the missing reduced-header digest. Every assignment binds the exact input
snapshot and authorization SHA256
`2da7534d5a7d65cd36c4c9d889b35e43aa340b70aa62c41c819eb460ef09a5f8`.
No exception was extended to Jan23, Jan28 or any other segment.

## Nhit distribution and boundaries

| Nhit range | Events |
|---|---:|
| <30 | 26,546,223 |
| [30,60) | 28,680,362 |
| [60,100) | 8,859,990 |
| [100,200) | 3,914,214 |
| [200,300) | 749,603 |
| [300,500) | 367,702 |
| [500,800) | 108,888 |
| [800,2000) | 49,440 |
| [2000,+∞) | 2,947 |

Exact lower-edge event counts are 30: 1,508,080; 60: 437,215; 100: 101,508; 200: 13,387; 300: 4,002; 500: 793; 800: 159; 2000: 20.
Source counters for all PINC candidates in the 626 accepted new files observed
**5 charge-boundary hits** and **123 time-boundary hits**, and **0 events above
4096 hits**. Charge 0.3 is included and times ±30 are excluded. The over-model-limit
case is covered by a real C++/ROOT fixture with **100,001 hits**, not by these data.

Targeted Slurm checks cover duplicate/missing/reordered/empty keys and empty-event
ROOT roundtrip; float32 charge/time and all eight bin edges; full arrays beyond the
model limit; missing/conflicting/nonunique/wrong-identity headers; non-7 pool
retention; precise authorization scope/digest/input binding; and replica endpoint,
metadata-digest, UUID and source-table corruption rejection. Initial fixtures ran
in 66153 and the retry migration checks; repair suites ran in **66166 (9 groups),
66169 (10), 66176 (12)** and were repeated before deployment. Workers also ran the
unchanged C++ fixture and applicable actual native ROOT non-7 header fixture.

## Same-basis performance

The baseline is the preceding **scalar** implementation, not the older slow
charge-transfer workflow. Three complete files were compared event by event:

| File | Events | Previous scalar source / s | Optimized source / s |
|---|---:|---:|---:|
| Esg20220101_00.root | 320,258 | 213.4 | 154.792 |
| Esg20220116_00.root | 165,969 | 216.3 | 125.401 |
| Esg20220131_00.root | 96,190 | 102.2 | 73.052 |

All keys, nv, nhit, poolused and ihep_entry match exactly; duplicates, missing,
surplus and bin differences are zero. Source total **531.900 → 353.244 seconds**:
**1.506× measured speedup**, **33.6% less time**. Hosts and EOS/cache conditions
were not controlled, so individual optimization contributions are not isolated.

| Day | PINC scan / s | Hit read+compute / s | Header / s | Write / s | Readback / s | Transfer / s | ETO / s | Processing total / s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20220101 | 22.393 | 127.664 | 2.856 | 0.433 | 0.062 | 1.896 | 2.077 | 158.764 |
| 20220116 | 12.626 | 107.549 | 3.444 | 0.241 | 0.046 | 2.599 | 1.455 | 129.454 |
| 20220131 | 6.254 | 63.344 | 2.791 | 0.157 | 0.031 | 2.629 | 1.164 | 76.846 |

The shared pilot Python/ROOT startup was **2.064 s**, once for the session. The
scheduler JobStartDate to source-session start was **8.222 s**, separately measured
with one-second scheduler timestamps. The source session elapsed **355.480 s**;
summing source+transfer+ETO processing gives **365.612 s**, excluding queue/poll
waiting and shared startup. Per-file source totals include bookkeeping beyond the
named phases. These are processing measurements, not month wall time.

HTCondor **3741346** processed all three pilot files in one persistent process.
Restart **3742229** processed zero and skipped all three confirmed checkpoints;
both sessions rejected a duplicate worker lock attempt. Four production workers
later processed **149 / 148 / 148 / 146** files in their long-lived sessions, in
addition to 31 files accepted before the controlled migration. ROOT/C++ compilation
was reused, header reads were capped at four per worker with deterministic table
order, and cached branch reads excluded vidmc. At most four source workers ran.

Month continuation wall time was **25,265.153 s = 7 h 1 min 5 s**, from the first
production submission (Oct15 20:50:10 CST) to the final repair receiver's terminal
confirmation (Oct16 03:51:15 CST). This
includes queueing, controlled restarts, failure diagnosis, repair preparation,
30-minute check gaps and final receiver cleanup. It excludes the earlier 55-file
production, pilot and independent final audit. New-file throughput was
**89.198 files/h**, or **9.051 million new events/h**. Initial production alone
finished in **14,777.030 s**, accepting 622 new files (**151.532 files/h**) and
leaving four data failures; that rate is not the end-to-end completion rate.
R3 waited about 63 minutes for a slot before processing the last file.

Across the 626 accepted new files: source **49,883.094 s**, PINC scan **5,671.319 s**,
hit I/O+compute **39,170.319 s**, headers **3,485.913 s**, write **129.055 s**,
readback **27.852 s**, transfer **1,321.470 s**, ETO **654.023 s**. Source+transfer+ETO
sum **51,858.587 s** across parallel workers. Shared session startup columns must
not be summed per file; process-startup instrumentation is absent for the first
31 accepted new files and is not inferred retroactively. `file_timings.csv`
contains all per-file phases, session IDs, hosts and execution commits.

## Failures, diagnosis and recovery

| Stage | Outcome and preserved evidence |
|---|---|
| Old v5 takeover | Verified Owner/Cmd/Arguments and retired 3739735/3739748/3739750; receivers/recovery paths disabled. Retained 55 receipts. Cleaned only five unaccepted source temporaries and four ETO unfinished requests after recording inventories. See `audit/legacy_*`. |
| Controlled retry-ceiling migration | At 86 accepted files, held/drained original 3742240–3742243 and resumed isolated verified code with 3742283–3742286. Four interrupted, unconfirmed checkpoints were inventoried/removed; accepted files were skipped. See `migrations/retry_ceiling_v2/` and four checkpoint-restart records. |
| Initial production | 677 accepted / 4 isolated failures; Slurm 66162/66157/66158/66163 terminal. Missing exact headers on Jan02_12 and Jan03_05; uproot zero/compression failures on Jan23_00 and Jan28_06. Original failed.json files remain. |
| Repair 1 | Slurm 66167 / Condor 3743102–3743105: Jan02/03 accepted using only the two authorizations; Jan23 accepted using its recovered real header. Jan28 repeated b00 error. Independent audit 66168: 680 files. |
| Repair 2 | Slurm 66170 / Condor 3743199: Jan28 native ROOT StreamerInfo/zombie preflight failed **before Nhit extraction**. Audit 66171 remains 680; no blind rerun of accepted files. |
| Metadata investigations | Slurm 66172–66175: bounded metadata reads only; preserved both all-zero and valid bytes and corrected an initially unsupported endpoint query. Evidence in `audit/header_io_v1` through `header_io_v4`. No full reduced ROOT download or EOS mutation. |
| Repair 3 | Slurm 66177 / Condor 3743478: independent actual eos92 endpoint and metadata check, then native direct-read verification; Jan28_06's 70,091 events accepted. Three immutable repair bindings retained in repairs.json. |
| Final audit repair | 66178 failed because rsync's ROOT exclusion also excluded task directories ending `.root`, omitting native probe JSON. Audit code now includes directories before excluding ROOT payloads and archives every repair. Targeted rsync regression passed; no friend changed. Full rerun 66179 and closure 66180 pass. |

`operation_history.json` indexes **6 failed file attempts** (four initial, Jan28
repair1 and repair2), **4 checkpoint cleanups**, **3 receiver transport failure
records**, **4 resolved initial failures**, and **0 additional header retries on
successful accepted segment rows**. The latter is not a count of all failed probe
I/O. Connection/transfer attempts are capped at three. Earlier v5 incidents,
migration logs and independent diagnostic/setup failures remain in their separate
audit directories; they were not erased or recast as successful attempts.
Initial, repair1 and repair2 terminal reports are preserved under
`audit/initial_terminal`, `audit/repair1_terminal`, `audit/repair2_terminal`.

Jan28's actual uproot DataServer was eos40:1095 on zero-filled metadata reads and
eos92:1095 on good reads. The fsid request alone did not ensure routing, especially
when ROOT was already loaded. Repair3 used a separate no-ROOT process, verified
actual eos92 and metadata, and privately passed the resolved EOS URL to native
TNetXNGFile. Native readback again verified eos92, one integer wcdah row/key, UUID
`7f3e55f8-7fc2-11ec-b339-4fe6020abeef` and header `(140739,2,7581,7)` before Nhit.
The probe and t_segments/provenance bind v3 evidence SHA256
`601234660787675712483945250327c069a727c967183b1b249722cd2acd519e`.
Reduced-header UUID and hashes here are real; Jan28 is not a manual pool7 exception.
The underlying EOS zero-content/routing cause remains unresolved; the source was
not repaired in place. Temporary access parameters were never written to reports.

## Space, receipts and closure

Observed combined source staging peak: **73,826,871 bytes (70.407 MiB)** against a
**2,147,483,648-byte (2 GiB)** budget, including retained preceding stages and
metadata diagnostics for later repairs. Logs, controls, partials and in-flight
reservations were budgeted. This is the largest sampled apparent-byte observation,
not an exact filesystem high-water measurement or allocated-quota usage.

Slurm66180 independently re-queried the current **personal** scratch quota:
used **530,063,450,112**, soft **536,870,912,000**, hard **545,246,097,408 bytes**;
177,189 files against 200,000. It did not use shared free space or a stale quota.
The production and three repair stages contain **0 active source jobs, 0 ROOT
payloads and 0 partial/reservation files**. All **626 new ETO receipts** match
source acknowledgments (622 original + 3 repair1 + 0 repair2 + 1 repair3).
The four retained source stages total **71,639,233 bytes** of metadata/logs/code;
diagnostic archives and the old audit histories are retained separately.

The same-chat 30-minute timer is retired only after this report and code are
published. Its final identity, cancellation result and current tick acknowledgment
are recorded in `audit/heartbeat/stop.json`, `stopped.json` and `handled.json`.
The immutable data-completion proof and the later delivery/publication/stop proof
are separate records; no ongoing polling is required after delivery.

## Evidence and code provenance

All paths below are relative to the ETO state root:
`/mnt/newdisk/shijie/WCDA_obs_friends/nhit/manifests/20261015_january_v6_scalar/`.

| Artifact | Contents |
|---|---|
| completed.json | 681/69,279,369 completion and final audit/catalog digests |
| final_audit.json | Full scalar/input/digest/source-table audit and mixed totals |
| file_catalog.csv | All 681 paths via receipts, event counts, bytes, file/receipt SHA256 |
| receipts/*.json | Durable per-file acceptance, source/target UUIDs, snapshots, versions |
| file_timings.csv | 626 new-file phase timings and persistent session startup attribution |
| failed_files.csv, failures.json | Empty unresolved failures; original failures retained elsewhere |
| operation_history.json, resolutions/, repairs.json | Failures/restarts and immutable repair bindings |
| audit/header_fullmonth/source/result.json | Independent exact-header search and UUID/identity evidence |
| ../20261016_january_v6_repair3/source_archive/production/Esg20220128_06.root/native_header_probe.json | Actual endpoint, metadata, UUID/header and evidence-binding readback |
| authorizations/poolused_override_status.json | Accepted status for the exact two authorized segments |
| audit/delivery_check.json, delivery_source_inventory.json | Fresh quota, matched source ACKs, empty active queues |
| audit/github_publications.json | Local execution commits → GitHub commits with identical trees |
| audit/heartbeat/ | Original timer, tick receipts and final stop audit |

Final-audit SHA256: `4332d620b15db0c05c4f0c43a34e1391ecfe12f0818d656eacf11b8fef9949bb`.
Catalog SHA256: `e0a5c210b49138c7a4b6becec5729712b6e496abcbd75bfa9b04551b013bc694`.
Completion-marker SHA256: `cbcecd7e51b7bc377583d35538ab82d4b2c2531060d63d17cab901034b71719f`.

| Successful generation | Files | Local execution commit | Published mapping |
|---|---:|---|---|
| Initial scalar snapshot | 31 | 182a93591d85f4beaf057598331f79bfea3d6d59 | 9041c78847bf5eee9deef36e978d186367d42117 |
| Retry-safe persistent snapshot | 591 | e7a168995b75d452d33b8d68a4e1ac53fffb23b7 | Source package unchanged in published 27c913b68115636846bf1f0e54ddce0b41691ed0; mapping explains differing local tree |
| Repair1 | 3 | fcfdba468b2194d15830bb831652c14742f0040f | 416f0f0225df9dbf983f3ae1737eef3218961694 |
| Repair3 | 1 | 73e4bdf1bab30994813adca044e2420903753fcb | 585f7cbb946d4b9a852769c77da0e4ee0902e52a |
| Final audit helper | — | 8b442c51c4b04ceff281bc50d4c456eac2fe0c94 | d16b9ab2f86ab70b43958519800a3e2b44d3ed3b |

The scientific C++ implementation remained byte-identical through the repairs
(SHA256 `961beb57da56fb601e235b76881b6a12b5385315b1fdef8dcffdb24db163446b`).
Source execution commits stay in provenance; publishing does not rewrite them.
GitHub SSH lacked permission, so exact trees were published using the connected
GitHub API with an expected-head check to branch `january-v6-nonnegative6`.
The final documentation/closure-tool commit and matching remote commit are added
to the publication audit after creation. No data or user AGENTS.md edit is included.

Remaining verification limits are the two private attestations, unresolved EOS
root cause, sampled rather than continuous staging high-water observation, absent
early process-startup instrumentation, and the explicit ETO Nhit/30-branch scope
above. They do not conceal any remaining unaccepted target file.
