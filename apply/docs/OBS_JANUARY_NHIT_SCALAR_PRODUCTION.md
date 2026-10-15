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
