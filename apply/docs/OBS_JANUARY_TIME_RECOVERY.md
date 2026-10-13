# January 2022 event time and J2000 recovery

This workflow restores the 31-day / 681-file January observation manifest as independent friends of the pincness-only observation files. Keep every selected event, including duplicates and failures, in its original order. Final products belong under `/mnt/newdisk/shijie/WCDA_obs/recovered_time/<MMDD>/`.

## Isolated implementation and dependencies

- ETO implementation checkout: `/mnt/newdisk/shijie/energy-time-recovery`, branch `codex/january-time-recovery`, in the energy repository. Do not overwrite the unrelated inference edits in `/mnt/newdisk/shijie/energy`.
- ETO state: `/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january-time`.
- IHEP control: `/scratchfs2/lhaaso/liushijie/WCDA_time_recovery_202201_20261013/.control`.
- IHEP temporary EOS: `/eos/user/l/liushijie/WCDA_time_recovery_202201_20261013`.
- Existing ETO IHEP connection: `.recovery/2026-10-13/obs-january/ihep.sock`. Do not copy credentials.

Production uses task-owned, unmodified Cog and SLALIB copies in `.control/deps/Cog` and `.control/deps/slalib`. The worker verifies `deps/dependency_sha256.txt` before compiling. Cog was copied from the historical algorithm's dependency directory; the provenance is retained in `deps/ORIGIN.txt`. No runtime path points into another user's home. The reference source is byte-identical to the user's `/home/lhaaso/liushijie/time-recover/src/recover_wcda_time.cc` and is stored as `.control/deps/legacy_reference.cc`.

The public `hushicong/public/v0.99` and `v1.0` headers lack the required `Ttai_t` conversions, `Tmjd_t::getns4`, `papi::hcs2eqa` and `papi::eqa2eqm` API. A v0.99 batch compilation failed before any production request or output publication. Do not replace the pinned dependency with those headers or change conversion formulas to make them compile.

## Validation and publication

`recover_observation_time.cc` builds only the current day's old event index. Old main/friend counts must agree; successful times must reproduce the Cog time conversion exactly. Old files with unauditable fallback are excluded from reuse. Conflicting valid times stop publication of the affected target. Coordinates require identical angles and evidence, or an exact recomputation check. Missing time is recovered by reduced file, with fallback allowed only when the exact name is absent and the alternative is unique.

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

File delivery, successful event recovery and temporary cleanup are separate acceptance totals. Do not report the month complete while any manifest file is missing or conflicted. Report the old `0101_22` 393 failures separately against current keys and list any remaining failures.

Downstream energy inference must apply its final `survivor_event_indices` to this friend and alignment tree, or reassociate by event key. These friends correspond to the complete pincness-only input, before additional inference cuts.
