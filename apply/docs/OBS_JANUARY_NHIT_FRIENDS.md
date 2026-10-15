# January observation Nhit friends

Contract: preserve the existing 31-day / 681-file ETO observation event universe.
No additional cuts, inference, MC processing, or restoration of discarded events.
Nhit is `sum((vqsamp >= float32(0.3)) & (abs(vt) < 30))` over complete stored hit arrays.
The eight bin lower edges are 30,60,100,200,300,500,800,2000; events below 30 remain.
Poolused is the configuration from the identity-checked reduced segment `wcdah`,
not an event-level branch in the AI observation input. This does not certify
equivalence to historical HSC native Nhit summaries.

## Storage and interface

ETO root: `/mnt/newdisk/shijie/WCDA_obs_friends/nhit`.
`sample512/` contains checksum-verified copies, with original locations in its catalog.
`full/MMDD/Esg202201DD_HH.nhit.root` is a complete, validated `t_nhit` friend.
`manifests/<run_id>/` holds the frozen inputs, code hashes, source records, independent
ETO receipts, failures, scheduler submissions and progress. These are not Git inputs.

`t_nhit`: `irun,ies,iseq,ievent,input_entry,ihep_entry,nv,vqsamp,nhit,poolused,segment_index`.
`t_segments`: index, segment triple, poolused, reduced source path/UUID, original
header filename/digest. `provenance` records the contract, code hashes and source.
Use event keys to authenticate association; row order alone is not authentication.
Original `vt`, `vidmc`, `rmds` and other retained branches remain in WCDA_obs.

## Execution

From ETO `apply/`, submit `sbatch tools/observation_nhit/run.sbatch 20261015_january_v4`.
The Slurm receiver uses the existing IHEP ControlMaster socket and submits one
IHEP HTCondor task per file (1 CPU / 4000 MB, site hep_sub). IHEP extraction is the
user-authorized exception to ETO-only compute. ROOT/NumPy/Awkward/uproot are from
LCG_106 on IHEP and the existing py310 environment on ETO; no new dependency.
Source reads use `root://eos01.ihep.ac.cn//eos/...`, including XRootD stat with a
60-second timeout. The v1 pilot (Slurm 66130 / Condor 3739146) stalled on the
compute-node EOS FUSE mount and was stopped before any output. Condor diagnostic
3739170 successfully read the identical file over XRootD; v1 records are preserved.
The v2 pilot matched all source keys and all 160 segment headers, then failed on
uproot 5.3.7's dictionary-of-strings writer. The source table now uses an Awkward
record array; all six synthetic checks pass on both IHEP LCG and ETO py310.
V2 failed output remains isolated; v3 starts the full-file gate afresh.
V3 was assigned back to lhws183, which again never reached Python's start marker.
After hold/release the scheduler retained its old execution claim. V4 uses a new
source directory and excludes that node in each job's requirements, preserving
all original site requirements. A submission race onto that node holds the job
and stops for inspection instead of continuing blindly.

The complete 0101/0116/0131 `_00` files run sequentially. All must pass before
automatic expansion, capped at four producers. Fresh Lustre personal quota is
checked before submission, with 1 GiB below soft / 6 GiB below hard reserved,
and at most 4 GiB total reservations. Each task reserves input file size + key
request size + 64 MiB; RLIMIT_FSIZE bounds its output. Requests and failed outputs
consume that reservation. Failed evidence is preserved and can pause a full queue.

Every source event key is scanned to detect duplicate matches. Selected data are
read in bounded spans and output in ETO input order. Per-block SHA256 digests cover
all original retained branches, including hit-array lengths, dtype and bytes; ETO
recomputes these from its input. IHEP performs complete friend readback; ETO checks
the transfer SHA256, identities, segment mappings and independently recomputes Nhit.
Exact bytes intentionally reject changed NaN payloads or signed-zero representations.

## Recovery and evidence

One receiver per run is protected by flock. Code/input changes reject resumption.
Targets publish using no-replace hard links, with a durable ETO receipt before
source cleanup. A restart rechecks any published-but-unreceipted target and repeats
unfinished acknowledgments. Do not manually remove source results before receipt.
Transport failures retry three times; submission is not blindly retried because
an interrupted response may conceal a live cluster. Inspect its submission record.
Data failures are isolated, with no guessed fields or deleted target events.
Pilot failure disables expansion. Jobs with no result after 20h stop the receiver
for scheduler inspection. Restart with the same run ID and unchanged source.

Run `python -B tools/observation_nhit/test_nhit.py` for synthetic contract checks.
`progress.json` is provisional; only `completed.json` certifies all 681 files.
`summary.json` aggregates only verified files and explicitly reports incomplete coverage.
Boundary counts of zero mean not covered in actual data, even when synthetic
boundary tests pass. A missing original event cannot be recovered by this pipeline.

## First complete-file acceptance, 2026-10-15

The frozen manifest contains 69,279,369 retained ETO events in 681 files;
their corresponding raw files contain 2,216,905,881 entries. The accepted universe
is the retained ETO set, not every raw event.

Run `20261015_january_v4`, receiver Slurm **66136**, has published the complete
`full/0101/Esg20220101_00.nhit.root`: 320,258 unique matches, zero missing or duplicate
matches, all retained branch contents equal, 100% independently recomputed Nhit
agreement and zero bin disagreements. Its 160 segment headers all report poolused=7.
All eight bins are represented; 159,394 below-30 events are also retained.
Exact Nhit=2000, charge=0.3 and absolute time=30 were not present in this file;
their synthetic checks must not be presented as actual-data boundary coverage.

Source extraction/readback took 611 seconds, peak RSS 847,216 KiB, output
71,905,759 bytes. ETO independent validation took 39 seconds. The second and third
full-file pilots and the remaining month are **not certified by this first result**.
The receiver automatically expands only after all three complete-file pilots pass.
Read `manifests/20261015_january_v4/progress.json`, `summary.json` and `failures.json`
for current status; only that run's `completed.json` certifies the complete month.
