# January 2022 observation-input restoration

Status at 2026-10-13 15:33 (Asia/Shanghai): all four shards are running. January 2 and 3 hour 00 have been newly filtered, fully read back and SHA-256 verified on ETO (581,595,261 and 575,874,303 bytes; 318,251 and 309,894 selected events). January 4 hour 00 passed source checks and is transferring. January 31 was copied successfully (23 files, 4,941,889,023 bytes). The remaining January dates are being rebuilt. The bounded rolling pipeline below replaces the failed whole-day staging attempt; monthly completion requires its `completed.json` and receiver exit code 0.

## Verified historical contract

The IHEP macro `filter_observed_pincness_loop_scoped.C` keeps the established 30 event branches and applies only `pincness < 1.1`. It removes `vflag`, `vnpe`, and `vqsamp`; preserves the complete `t_eventout_h`; and preserves the selected event order. `vx`, `vy`, `vq`, `vt`, and `vidmc` remain essential model inputs. Their removal belongs after model inference, not in this acquisition step. No new fitstat, theta, dcedge, Nhit or predicted-energy cut is added here.

- Originals: `/eos/lhaaso/ai/wcda/raw/2022/01??/Esg202201??_HH.root`.
- Snapshot: 681 hourly files across all 31 January dates; manifests use the actual filenames, not an assumed 744 files.
- Reusable archive: `/scratchfs2/lhaaso/liushijie/WCDA_observation_20220101_20220630/0131/`, 23 files totaling 4,941,889,023 bytes. No complete January 1–30 filtered archive was found in the inspected locations.
- Temporary IHEP staging: `/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013/<MMDD>/`.
- ETO destination: `/mnt/newdisk/shijie/WCDA_obs/<MMDD>/`.

Archive candidates are hard-linked into the separate staging tree, then checked against the original's selected count, header count, exact branch set and pincness cut. The original archive is preserved. New files are written to `.partial`, fully read back (including retained event and header branches), checked against the same contract, and promoted only after validation. Original names and daily layout are preserved for later inference/time matching.

## Storage failure and corrected flow

The first January 1–30 cluster `3724925` failed because the user's scratchfs2 quota reached its hard limit, approximately 508 GiB, despite the filesystem having free capacity. ROOT reported basket write errors and some jobs crashed. That cluster was stopped; only files in this task's new staging tree were removed. EOS originals and the previous archive were unchanged. After Lustre quota accounting caught up, usage was 516,983,856 KiB, below the 524,288,000 KiB soft limit.

The historical macro now fails on event read, basket fill, header clone or final write errors. Incomplete output never gets a transfer receipt.

The replacement flow bounds staging to at most four hourly files:

1. Four HTCondor shards process disjoint dates sequentially. Shard 0 first revalidates January 31, then handles days 1,5,...29; the other shards handle days 2,6,...30; 3,7,...27; and 4,8,...28.
2. For each output, the producer completes the ROOT checks, computes SHA-256, and atomically publishes `audit_files/<name>.json`.
3. ETO receives that single file with resumable rsync, checks filename, byte count, ROOT signature and SHA-256, then atomically writes a matching source-side acknowledgment.
4. The producer removes only its staging file after the acknowledgment. The cleanup code rejects output roots outside this exact run tree and rejects staging symlinks. Unlinking an archive hard link preserves the original archive.
5. Atomic `audit/YYYYMMDD.json` records whole-day completion. ETO marks a day complete only after all its file receipts match; `completed.json` requires all 31 days / 681 files.

A restart reuses acknowledged records and verified ETO files. A producer waits at most six hours for an acknowledgment; the receiver has a 36-hour deadline. These limits make a lost transfer process fail visibly. A changed upstream input requires a fresh run directory and source manifest; this recovery assumes the original January inputs remain immutable.

## Execution and inspection

IHEP work directory: `/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013`.

- Tracked entrypoint: `scripts/data/run_observation_shard.sh`.
- HTCondor: 1 CPU / 4 GB per worker, AlmaLinux9, default 15-hour walltime, `scheduler@schedd11.ihep.ac.cn`.
- Submitted jobs: `3725172.0` (shard 0), `3725173.0` (1), `3725174.0` (2), `3725175.0` (3).
- Runtime: `/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh`, ROOT 6.32.02.
- `manifests/*.txt` and `source_inventory.json`: exact source snapshot.
- `logs/rolling_shard<N>.out` / `.err`: distinct worker logs.
- `audit_files/`, `acks/`, `audit/`: durable per-file receipts, transfer acknowledgments and day reports.
- Earlier idle clusters `3724878`, `3724894`, `3724913` were removed. Short cluster `3724922` successfully validated the January 31 archive before its first copy.

ETO receiver state: `/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january/`.

Inspect `progress.json`, `sync_rolling.log`, `pid`, `exit_code`, and eventual `completed.json`. The receiver runs detached on ETO. It uses the direct ETO→IHEP multiplexed connection `ihep.sock`; no private key was copied to ETO, and the Mac is outside the data path. If the connection is lost, it must be authenticated again before restarting the receiver. The receiver and source entrypoint snapshots are under `code/` in their respective run directories.

## Verification

- Tiny ROOT fixture: strict pincness boundary, NaN exclusion, exactly 30 retained branches, required hit features, complete header, count mismatch and missing-file rejection.
- Cleanup guard: an output root outside this task-created staging tree is rejected before accessing inputs.
- Deliberately restricted file size: disk write failure returns a nonzero filter result.
- Real-schema smoke: first 2,048 events of `0101/Esg20220101_00.root`, 54 selected, complete 160-entry header, output 331,992 bytes.
- Receiver: valid file accepted; corrupt signature, same-size ROOT-like content with a wrong SHA-256, and path escape rejected.
- Python compilation, shell syntax, and remote ROOT/receiver unit checks passed.

No energy inference, reduced time-data transfer, or time recovery is included. The monthly production outputs and their checksums become verified only as the running pipeline completes; estimates must use measured full-file throughput.

The first rebuilt files took 223–246 seconds from batch start to source verification, including runtime startup; ETO receipts followed in 37–69 seconds. A provisional January estimate is 10–14 hours from 15:33, subject to later input sizes and I/O contention. This is an estimate, not a completion guarantee. If a shard reaches its walltime, resubmit the same shard index: verified receipts survive, so already received files are skipped.
