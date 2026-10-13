# 64748 January inference acceptance and production

ETO project: `/mnt/newdisk/shijie/energy`. Control scripts and receipts:
`.recovery/2026-10-14/inference-64748/`.

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

## Gated jobs

| Stage | Slurm | State when configured | Parameters / gate |
|---|---|---|---|
| January 1 pilot | 66045 | Running | All 23 files; batch 32, 100 MB chunks; verifies every selected hit and output file |
| Full January inference | 66052 | Waiting on 66045 success | 2 GPU / 16 CPU / 64 GB, batch 352 per GPU; provenance/memory gate and a two-GPU sample before production |
| Full January validation | 66053 | Waiting on 66052 success | 681 filenames / 31 days, independent cuts and event identities, finite energy consistency, exact header content and schema |

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

## Two-GPU tuning and estimated completion

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
