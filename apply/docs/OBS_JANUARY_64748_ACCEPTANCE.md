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
| Full January inference | 66046 | Waiting on 66045 success | Prepared January launcher; batch 128; repeat provenance and memory gate before starting |
| Full January validation | 66047 | Waiting on 66046 success | 681 filenames / 31 days, independent cuts and event identities, finite energy consistency, exact header content and schema |

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
