# January 2022 scalar Nhit continuation

2026-10-15. Run `20261015_january_v6_scalar` is independent of the paused v5 records
and the previous benchmark. Its ETO state is
`/mnt/newdisk/shijie/WCDA_obs_friends/nhit/manifests/20261015_january_v6_scalar/`;
its IHEP stage is `/scratchfs2/lhaaso/liushijie/WCDA_nhit_20261015_january_v6_scalar/`.

The authorized universe remains the 31-day / 681-file original WCDA_obs input
snapshot, 69,279,369 events. Slurm 66152 checked the restoration receipts, current
ROOT identity/size/mtime/header snapshot and the old manifest, then rechecked
55 durable accepted friends (5,757,151 events, 1,523,864,929 bytes). Their bytes,
scalar event alignment, algorithm and provenance agree; their charge arrays remain.
Those files are labelled `legacy_with_charge`. Nothing is rewritten to save space.
The recovery receipts are metadata-and-byte receipts; this does not establish a
new independent SHA256 of all 151 GB of original observation inputs.

The new source worker reads PINC plus irun/ies/iseq/ievent/nv/vqsamp/vt, never vidmc.
It selects only `pincness < 1.1`, uses complete stored hits, validates lengths and
finite charge/time, and computes `sum((vqsamp >= float32(0.3)) & (abs(vt) < 30))`.
No charge arrays or surrogate native Nhit branches are written. Unique, identity
checked segment wcdah headers supply poolused with no default and no pool cut.
Four concurrent header readers preserve the original header-table order.

ETO uses Slurm to verify transferred digests, candidate and target key uniqueness,
source-entry bounds/uniqueness, nv and segment/pool correspondence. It aligns the
four-key identity in target order, logs surplus keys, quarantines missing targets,
adds contiguous input_entry, writes and completely reads back the scalar friend,
and publishes using a no-replace hard link. A durable ETO receipt precedes source
acknowledgment and source ROOT cleanup. Unlike legacy acceptance, **ETO does not
recompute Nhit from original charge or compare all 30 common branches**. Full-file
pilot equality is evidence for its three files, not independent recomputation of
all later files. Scalar schema: irun, ies, iseq, ievent, input_entry, ihep_entry,
nv, nhit, poolused, segment_index. Keep all events below Nhit=30 and all eight bins.

The three existing `_00` friends for Jan01/Jan16/Jan31 are read-only references.
Tests write only under the new manifest's pilot_results. Automatic expansion
requires exact keys/nv/nhit/poolused/ihep_entry, zero duplicate/missing/surplus/bin
mismatch, and aggregate source speedup over **213.4 / 216.3 / 102.2 seconds**, the
previous scalar implementation. It also requires one worker to finish three files,
a real restarted worker to skip all checkpoints, and a failed duplicate lock probe.

Up to four persistent HTCondor workers own deterministic production shards. Each
reuses ROOT/compiled C++ and advances after isolated file failures. Jobs checkpoint
per file and exit after eight hours; the receiver restarts only absent/finished
shards, never blindly retries a possibly accepted scheduler submission. Site
Requirements are preserved when excluding lhws183 and bws0995; the shell also
refuses to compute on those hosts during a scheduling race. Connection/transfer
operations have at most three attempts. All use `ihep-work`, with its interactive
RemoteCommand disabled, ControlMaster/ControlPath disabled and IdentityAgent=none.
Slurm 66151 verified that connection independently from the actual compute context.

Personal `lfs quota -u liushijie /scratchfs2` is queried before deployment, every
receiver pass, and under a source budget lock for each reservation. The stage's
2 GiB budget counts actual data, partials, logs and controls plus conservative
in-flight reservations. Each file reserves two 128 MiB output limits plus 16 MiB
control slack; 512 MiB remains below the current personal soft/hard limit. Stage
and quota histories support measured peak usage; no shared-free-space assumption.

Start/resume from ETO apply with:

```
sbatch tools/observation_nhit_scalar/run.sbatch 20261015_january_v6_scalar
```

Code/config/input changes reject resumption. `progress.json` is provisional;
`finished.json` may report terminal failures. Only `completed.json`, after all
681 durable receipts and final digest checks, certifies a completed month.

Known data dependency: Esg20220102_12 requires run139579/es1/seq3336 reduced.root.
The old investigation found only es2 at that sequence, which is not substitutable.
The new workflow must isolate it unless an identity-exact reliable header is found.
No poolused=7 imputation is permitted.

Preflight: four Python synthetic groups passed in Slurm 66153 (duplicate/missing/
reordered/empty keys, charge/time/bin boundaries, unique/conflicting/missing header,
non-7 pool and empty ROOT roundtrip). Every source worker additionally exercises
its deployed C++ reader with a zero-hit event, exact boundaries, a rejected PINC
boundary event and 100001 stored hits. Production results and measured performance
will be added after execution; this implementation note is not a completion claim.

## Full-file evidence and operational checks

All three isolated full-file outputs passed exact comparison on aws003:

| File | Events | Previous scalar source / s | Optimized source / s |
|---|---:|---:|---:|
| Esg20220101_00.root | 320258 | 213.4 | 154.792 |
| Esg20220116_00.root | 165969 | 216.3 | 125.401 |
| Esg20220131_00.root | 96190 | 102.2 | 73.052 |

Total 582417 events: zero duplicate/missing/surplus keys and zero differences in
keys, nv, Nhit, poolused, ihep_entry or eight-bin assignment. Aggregate source
speedup is approximately 1.51x (531.9 to 353.2 seconds, 33.6% reduction). This is
against the previous scalar implementation, not the older charge-transfer scheme.
These runs used different hosts and do not control server/cache state; the gain
is an observed workflow comparison, not a causal separation of each optimization.
Detailed phases and actual transfer/ETO timings are in each pilot receipt.

| File day | PINC scan / s | Hit read+compute / s | Header / s | Write / s | Readback / s | Transfer / s | ETO / s | Processing sum / s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan01 | 22.393 | 127.664 | 2.856 | 0.433 | 0.062 | 1.896 | 2.077 | 158.764 |
| Jan16 | 12.626 | 107.549 | 3.444 | 0.241 | 0.046 | 2.599 | 1.455 | 129.454 |
| Jan31 | 6.254 | 63.344 | 2.791 | 0.157 | 0.031 | 2.629 | 1.164 | 76.846 |

Processing sum is source elapsed + transfer + ETO elapsed; source elapsed also
includes setup/bookkeeping within the file. The Python/ROOT startup is 2.064 s
once for the shared pilot session, not once per row. Scheduler/environment startup
is recorded separately in the pilot startup evidence. These sums exclude queueing
and receiver polling delays and must not be presented as monthly wall throughput.

HTCondor 3741346 processed all three in one process/session. Restart 3742229
processed zero files and skipped all three durable checkpoints. Both sessions
rejected a second process's attempt to acquire their worker lock and passed actual
C++ ROOT boundary/100001-hit fixtures. The first session's scheduler JobStartDate
and its source session record are saved under audit/pilot_started.json, allowing
startup before Python plus ROOT setup to be distinguished from source processing.

Slurm 66155 verified a fresh connection specifically to lxlogin004 after the local
alias was found to point at round-robin lxlogin. `ihep-work` now fixes HostName to
lxlogin004 and verifies its key against the already trusted lxlogin cluster key;
no host-key bypass, socket or agent is used. Early exec-channel retries are in the
controller log. Old tasks 3739735/3739748/3739750 are retired; five own unaccepted
source temporaries and four local unfinished request arrays were removed only
after preserving source logs, task/failure records and cleanup inventory.

Fresh `hep_clus -g lhaaso --walltime` reports default=15 h, long=720 h,
short=30 min. The operational Slurm supervisor uses exact site-generated class
attributes (short for pilot, default for production); each worker voluntarily
checkpoints/exits after 8 h. It preserves Requirements and both node exclusions.
The supervisor also holds this run's source jobs if the receiver detects changed
input identity, stopping further file expansion. These controls are separate from
frozen scientific source code. Run it alongside a resumed controller:

```
sbatch --partition=main --qos=default-safe --mem=1G --time=7-00:00:00 \
  --wrap='cd /mnt/newdisk/shijie/energy && OPENBLAS_NUM_THREADS=1 /mnt/newdisk/shijie/conda/envs/py310/bin/python -u -B apply/tools/supervise_observation_nhit_scalar.py /mnt/newdisk/shijie/WCDA_obs_friends/nhit/manifests/20261015_january_v6_scalar'
```

The independent header investigation completed under HTCondor 3742218. Direct
XRootD directory listings of Jan01/02/03 found no run139579/es1/seq3336 source.
The same sequence's only Jan02 reduced file has a unique header **(139579,2,3336)**,
UUID 591bc956-6b8a-11ec-b2ef-55e6020abeef: it is a different event stream and cannot
supply the missing es1 header, even though it reports poolused=7. The investigation
retains both initial diagnostic implementation failures and corrected results in
audit/header_probe. This does not prove absence from every external archive.

ETO execution commit: 182a93591d85f4beaf057598331f79bfea3d6d59. GitHub publication:
9041c78847bf5eee9deef36e978d186367d42117, with exactly the same tree
(e9d079f109b2ca2a535a2377fd5ae672e8322bcd), published through the connected API
because ETO's SSH key is not authorized by GitHub. The publication audit records
both identities; data products retain their actual ETO execution-code provenance.

## Audited control update during production

At 86 durable accepted files, the four source jobs were held and their execution
hosts released. The receiver first drained every ready candidate, then stopped
on the explicit hold. Jobs 3742240–3742243 were retired only after checking their
Owner/Cmd/Arguments and CurrentHosts=0. `migrations/retry_ceiling_v2/` preserves
old code/config, complete source metadata/logs, scheduler records, receipt count
and the four unconfirmed ROOT intermediates' size/mtime inventory. No accepted
file is re-extracted or overwritten. New workers own and audit cleanup of only
those incomplete intermediates before restarting their deterministic shards.

The update prevents a header's three exhausted connection attempts from triggering
three additional whole-file attempts, adds a source stop flag for changed input
identity, records interrupted-file cleanup, and includes shell/environment startup
in a separate timing field. Site-native short/default classes are now selected at
submission, preserving original Requirements; the supervisor remains as a guard.
The scientific contract and C++ Nhit reader are byte-identical to the passing
three-file pilot. Successful header reads/mapping and scalar acceptance are
unchanged. Five synthetic groups pass in Slurm 66161, including a mocked header
connection that is attempted exactly three times and cannot cascade to file retry.
Original pilot/production receipts keep their actual code provenance; the code
transition is explicit rather than relabelling old outputs as the new version.


The final Slurm audit exports `final_audit.json`, `file_catalog.csv`,
`file_timings.csv`, `operation_history.json` and `failed_files.csv`, rechecking all published file digests, scalar keys/order/nv,
segment mappings and current input snapshots. Its throughput clock includes
production queueing, controlled restarts and receiver final checks. Source logs,
per-file attempts and checkpoint cleanup records are copied into `source_archive/`.
The timing CSV identifies each persistent session and separates per-file stages
from its shared startup; the operation history indexes file failures, recovered
attempts, header retries, receiver transport failures and checkpoint cleanup.
A dependent metadata-only HTCondor investigation searches all January reconstruction
directories for failed headers, checks candidate identity/header uniqueness/ROOT
UUID and records the raw AI file's available metadata. It cannot publish a friend
or substitute a mismatched stream. Its evidence appears under
`audit/header_fullmonth/`; execution is deferred until production is terminal.

## Thirty-minute follow-up

At the user's request, active chat polling stops after submission. ETO Slurm
66165 runs `tools/monitor_observation_nhit_30m.py`, queuing one check into the same
Codex chat every 1800 seconds using the ETO daemon. Slurm 66164 verified daemon
access from the actual scheduler context; the queue API also accepted a scoped
installation self-test. The first regular check is 2026-10-15 23:04:59 Asia/Shanghai.
This is an ETO timer, not a native app Scheduled entry: the automation-update tool
was unavailable in this connection. No Mac filesystem, agent or socket is used.

`audit/heartbeat/` contains the durable prompt, configuration, timer job identity,
pending/handled tick receipts and logs. A file lock prevents duplicate timers;
an unhandled or ambiguous queue request prevents more requests from accumulating.
Normal checks stay quiet; actionable failures and the final result are reported.
After final audit, failure investigation, documentation and publication, the
checking turn writes `stop.json` and retires this timer. Receiver/source polling
continues at its existing cadence to accept outputs and release staging promptly.
