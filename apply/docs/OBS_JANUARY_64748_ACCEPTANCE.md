# 64748 January inference acceptance and production

ETO project: `/mnt/newdisk/shijie/energy`. Control scripts and receipts:
`.recovery/2026-10-14/inference-64748/`.

## Current four-GPU restart (October 14)

The user restored all four V100s and explicitly abandoned the old two-GPU run.
Jobs 66063/66064 are cancelled. Cleanup removed only the old dedicated January
prediction directory: 48 ROOT files, 1,095,453,833 bytes. Originals, weights,
configs, detector geometry, pilot data, logs and receipts are retained;
`abandon-2gpu-cleanup.json` records every removed output. New production starts
with an empty `/mnt/newdisk/shijie/WCDA_obs_eval_64748`.

Slurm 66068 passed all five discovery/TTree regressions, capacity/finite-output
checks on each of four GPUs at batches 352/384/400, and a simultaneous four-file
32,768-event native-sharding sample. Independent checks preserved headers, event
identities/cuts and finite consistent energies. The sample took 28.892 seconds,
1,134.14 events/s overall. Batch 384 is the largest tested batch within the chosen
88% Torch-reserved-memory budget on every card, with no material throughput loss:
8.180 GB live / 14.561 GB reserved per GPU. Batch 400 reserved 15.171 GB without a
speed benefit. No model, preprocessing, precision or event-cut changes are made.

Production 66070 requests four GPUs, 32 CPUs and 96 GB RAM on main, afterok:66068.
It verifies `four-gpu-tuning.json`, unchanged `production-provenance.json` hashes,
and the exact launcher GPU/batch settings, then repeats the four-file acceptance
before running all 681 January files / 31 dates. Validator 66071 is afterok:66070
and binds both the four-GPU sample receipt and final summary to that producer.
The overwrite/concurrency guards remain active. Unbuffered child output improves
progress logging. Current IDs are in `inference-64748/jobs.json`; controls are
`month_4gpu.sbatch`, `gate_four_gpu.py`, `verify_four_gpu_sample.py` and
`verify_month_4gpu.py` under the same ignored evidence directory.

The four native shards use seeds 42, 100042, 200042 and 300042. Their weighted hit
samples differ from the abandoned two-GPU partition; no old prediction file is
reused. Reproduction must preserve four GPUs, file ordering and this seed policy.

The pilot retention predicts about 49 million selected events in January.
The short four-GPU sample implies approximately 12 hours of inference at its
measured rate. Allow roughly 10–14 hours of inference plus 1–2 hours for independent
validation, and update from full-month progress; startup, file sizes and shared
I/O limit the accuracy of this early estimate. The 30-minute monitor is resumed
for the new job IDs and stays quiet when healthy progress has no meaningful change.

## Accepted 128-event smoke test

- Slurm `66044`: COMPLETED, exit `0:0`, 8 seconds including startup/validation.
- Same `apply/apply_observation_energy.py` entry; 1 file, at most 128 events, batch 32, one V100.
- 128 predictions written; 128 input events / 14,609 hits checked for array/vidmc alignment, finite values and detector IDs.
- Inference elapsed 4.66 seconds; GPU peak allocated 694,487,552 bytes, reserved 1,249,902,592 bytes; process peak RSS 1,118,980 KiB.
- Strict weight loading passed; all output energies finite and positive; energy agrees with `10**ml_logE_pred`.
- Header tree fields and contents preserved; both output objects are TTrees.
- Independent source cuts/event identities match output; raw hit feature branches are absent.
- Weight, config and detector geometry hashes unchanged.

Output: `apply/output/inference_64748_acceptance/smoke/`. Failed attempt `66042` is preserved as
`apply/output/inference_64748_acceptance/smoke_failed_66042/`; it failed before inference during header writing.

## ROOT compatibility fix

Under uproot 5.7.7, dictionary assignment chooses RNTuple writing, which rejects
the original NumPy object arrays for header strings and jagged hits. Slurm probe
`66043` reproduced this for headers and events and verified explicit TTree writing.
The shared observation entrypoint now uses `mktree` for headers and derives the
event TTree schema from the source branch interpretations before extending data.
This also preserves a typed zero-event output.

`apply/tests/test_observation_root_writing.py` reproduces the original failure and
passes after the fix for both selected events and an all-rejected input. The
expected extra `nvidmc` is the writer's counter for the retained jagged ID branch.
No model parameters, preprocessing, event cuts or input data were changed.

## Current gated jobs

| Stage | Slurm | State when configured | Parameters / gate |
|---|---|---|---|
| January 1 pilot | 66045 | PASS, completed 02:15 +08:00 | 23/23 files; 1,522,147 selected events; no failures |
| Full January inference | 66070 | Fresh four-GPU run after tuning 66068 and its own four-file sample | 4 GPU / 32 CPU / 96 GB, batch 384 per GPU; unchanged model/geometry and approved code hash |
| Full January validation | 66071 | afterok:66070 | 681 filenames / 31 days, independent cuts and event identities, finite energy consistency, exact header content and schema |

The pilot has its own `apply/output/inference_64748_acceptance/day/` directory.
Production writes `/mnt/newdisk/shijie/WCDA_obs_eval_64748/` and does not reuse pilot
files. The production launcher refuses existing January output and holds its
output lock. The pilot is reprocessed as part of the full month so the production
sample uses one batch configuration and the unchanged entrypoint throughout.

Successful completion of a producer is insufficient: the final acceptance receipt
is `.recovery/2026-10-14/inference-64748/month-acceptance.json`, written only after
all independent checks pass. `day-acceptance.json` is similarly required before
production starts. A failed prerequisite prevents downstream execution.

Evidence: `smoke-acceptance.json`, later `day-acceptance.json` / `month-acceptance.json`,
`acceptance.py`, `month_gate.py`, `verify_month.py`, job scripts and
`logs/slurm/64748_*_<jobid>.out/.err`. Monitor live Slurm state rather than assuming
the job states in this document remain current.

These checks establish computation/data-contract integrity. They do not validate
the physical calibration or scientific suitability of the resulting spectrum.

Background follow-up: thread heartbeat automation 64748 checks every 30 minutes, stays quiet during unchanged healthy progress, and reports stage completion/failure or required action. It stops after final acceptance.

## Historical two-GPU tuning and estimated completion

User requested both available V100s. Pending single-GPU jobs 66046/66047 were
verified as task-owned and cancelled; 66052/66053 replace them. The running day
acceptance 66045 was not interrupted. Production requests both GPUs and uses the
entrypoint's native file sharding, batch 352 and two reader threads per worker.

Slurm 66050/66051 benchmarked FP32 batches without changing the model or event
preprocessing. GPU-only throughput was approximately 1,055 events/s over batches
256–384. Batch 384 reserved 14.56 GB and exceeded the chosen headroom budget.
Batch 352 reserved 13.35 GB (about 79% of Torch-reported available memory), with
7.50 GB peak live allocation. A real 8,192-event run with reader-workers=2 passed
in 17.44 seconds, about 470 events/s end to end. Higher memory occupancy alone
did not materially improve GPU-only throughput; native two-worker execution is
the main parallelism improvement.

Historical estimate before the failed launch (superseded; recompute from 66063):
At 2026-10-14 01:56 +08:00, the day pilot had completed 7/23 files, writing
935,604 events in 35 minutes. Its provisional completion was about 02:24. The
January inputs contain 69,279,369 events; native shards contain 34,879,970 and
34,399,399 input events. Using the pilot's 70.93% retention and 70–90% of the
measured per-GPU throughput yields roughly 16.3–20.9 hours of month inference,
plus an estimated 1–2 hours of independent validation. The provisional final
window is October 14 evening through October 15 early morning, approximately
20:00–02:00 Asia/Shanghai. This assumes a successful day pilot and no subsequent
GPU contention; refresh it from the two-GPU sample and actual month progress.

Evidence: two-gpu-tuning.json, batch-tuning-66050.json, two-gpu-eta.json and
gpu_usage_<jobid>.csv in the control directory. The production job first runs
two source files with 4,096 events each on both GPUs, verifies finite energies,
header contents and source event identities, and writes two-gpu-acceptance.json.
Only then does it invoke the prepared full January launcher.

Native multi-GPU workers start with seeds 42 and 100042. Their per-file weighted
hit sampling can differ from a one-GPU run; preserve the GPU count, input ordering,
seed policy and code hash when reproducing this production configuration.

## October 14 relaunch after discovery failure

The day pilot 66045 passed at 02:15:34 +08:00. Job 66052 passed its two-file,
8,192-event dual-GPU sample (603.79 events/s including startup) but failed when
scanning the full input root: recursive discovery included
`recovered_time/0101/Esg20220101_00.time.root`. This is an input-selection failure,
not a prerequisite to finish recovering all event timestamps.

Discovery now accepts only direct `MMDD/*.root` inputs, or direct `*.root` when
an individual MMDD directory is supplied. Single-file input is unchanged. A
fixture containing original files, time sidecars and unrelated nested files
failed twice before the fix; all three discovery tests and both existing TTree
writing tests now pass. The live January scan finds 681 files / 31 dates while
excluding the time-recovery directory. No source data is moved or deleted.

Pilot receipt and hashes are retained unchanged. `production-provenance.json`
records the approved transition from entrypoint SHA256
`91e804e5043b6ab1ef41f985300c528840e26ed367fb884ee548f38010e287cd` to
`862706fc8a6d4cfab41e65ff495143a0db4b762ab994c036823e405a77f67375`.
Reversing only the discovery patch reproduces the pilot hash; checkpoint, config,
detector geometry, preprocessing and cuts are unchanged. The gate checks this
transition, then the new job reruns the regression tests and real dual-GPU sample.
Final validation binds the producer through `INFERENCE_JOB_ID=66063`, verifies the
matching two-GPU receipt and uses the approved production provenance.

Jobs 66063 / 66064 replace 66052 / 66053. The blocked old validator 66053 is
cancelled; prior control scripts and sample receipt are preserved in
`inference-64748/failed-66052/`. New production starts from an empty output root
with the existing lock and overwrite refusal intact. Current job IDs remain in
`inference-64748/jobs.json`.

Batch 352 is retained. Batches 128–384 already measured roughly 1,055–1,064
GPU-only events/s; batch 384 reserves 14.56 GB without a throughput gain, while
352 reserves 13.35 GB. Larger batches therefore add memory pressure without a
measured speed benefit. No mixed precision or model changes are introduced.
The earlier calendar completion window is obsolete; refresh the estimate after
66063 starts and real month progress is available.

## User pause for shutdown and GPU reset (10:40 +08:00)

At the user's request, 66063 was terminated and requeued in a held state using
`scontrol requeuehold`; 66064 was held separately. This releases CUDA contexts,
unlike process suspension. The 30-minute automation 64748 is PAUSED. No automatic
release or resubmission is authorized until the user requests resumption.
Verification found no GPU compute processes, 5 MiB on each V100 and 0% utilization.
The unrelated CPU-only time receiver 66062 remains running; host shutdown will
interrupt that separate workflow.

Inputs, checkpoints and all existing outputs remain in place. At pause request,
48 output paths existed; these two were actively being written and must be
reviewed before reuse: `0103/Esg20220103_00.root` and
`0103/Esg20220103_01.root`. No partial file was deleted or accepted as complete.
Other outputs also require provenance and content verification before a resume;
the full-month acceptance is still pending. Current evidence is
`pause-66063.json`, with prior job registry, provenance, sample receipt and logs
preserved in `paused-66063/` under the control directory.

Do not simply release 66063: the production launcher deliberately refuses an
existing output set. Resumption must review the preserved files, keep source
ordering/shards/seeds and approved provenance, and arrange an explicit resume or
fresh output directory before rebuilding the validator dependency. Previous
calendar completion estimates are withdrawn for this user pause.
