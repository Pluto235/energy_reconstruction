# January 2022 observation-input restoration

Status: preparation/synchronization in progress, not yet complete.

## Verified historical contract

The IHEP macro `filter_observed_pincness_loop_scoped.C` keeps the established 30 event branches and applies only `pincness < 1.1`. It removes `vflag`, `vnpe`, and `vqsamp`; preserves the complete `t_eventout_h`; and preserves the selected event order. `vx`, `vy`, `vq`, `vt`, and `vidmc` remain essential model inputs. Their removal belongs after model inference, not in this acquisition step. No new fitstat, theta, dcedge, Nhit or predicted-energy cut is added here.

- Original inputs: `/eos/lhaaso/ai/wcda/raw/2022/01??/Esg202201??_HH.root`.
- Source snapshot: 681 hourly files across all 31 January dates; actual source filenames determine the inventory, not an assumed 24 files per day.
- Reusable archive: `/scratchfs2/lhaaso/liushijie/WCDA_observation_20220101_20220630/0131/`, 23 files totaling 4,941,889,023 bytes. No complete January 1–30 filtered archive was found in the inspected locations.
- IHEP prepared output: `/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013/<MMDD>/`.
- ETO destination confirmed by the user: `/mnt/newdisk/shijie/WCDA_obs/<MMDD>/`.

Archive reuse hard-links each candidate into the separate prepared-output tree, then checks it against the original input's selected count, header count, branch set and pincness cut. Invalid candidates are quarantined only in the new output tree; the archive and raw data are not modified. New files are written to `.partial` and promoted only after the same validation succeeds. This preserves original names and daily layout for later inference/time matching.

## Execution

IHEP work directory: `/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013`.

- One-date HTCondor workers, 1 CPU and 4 GB each, AlmaLinux9, long walltime, schedd11.
- ROOT runtime: `/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh` (ROOT 6.32.02 verified).
- Production cluster `3724913`: 30 tasks for January 1–30.
- Archive/validation cluster `3724894`: January 31.
- Initial single-hour cluster `3724878` was removed while idle after the bounded real-schema smoke test passed, before production submission.
- `manifests/*.txt`: exact source URIs; `source_inventory.json`: expected counts; `logs/`: batch outputs; `audit/YYYYMMDD.json`: atomic completion marker containing per-file checks.

ETO receiver state: `/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january/`.

A detached receiver polls the completed IHEP daily audits, copies only completed daily ROOT files with resumable rsync (no deletion), then checks each expected filename, size and ROOT signature. It saves `progress.json` and writes `completed.json` only after all 31 days / 681 files have passed. `sync.log`, `pid` and `exit_code` track the receiver. SSH uses a direct ETO→IHEP multiplexed session authenticated temporarily from the Mac; no private key is copied to ETO and the Mac is not in the data path.

## Checks performed before production

- ROOT fixture: strict pincness boundary, NaN exclusion, exactly 30 retained branches, required hit features, complete two-entry header, mismatch rejection and missing-file rejection.
- Real input prefix: first 2,048 events of `0101/Esg20220101_00.root`; 54 selected, complete 160-entry header, output 331,992 bytes; validation passed.
- Receiver unit check: valid file accepted, corrupted ROOT signature and path escape rejected.
- Python compilation and shell syntax checks.

No energy inference or time recovery is part of this request. Full files, monthly completion and transfer checksums still need the running pipeline to finish; source-side audits check counts/schema/cuts/header, not every retained hit value. A full dataset SHA-256 audit is not part of the current receiver.
