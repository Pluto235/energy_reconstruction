# January prediction/time identity alignment

The January prediction trees contain only inference survivors. The recovered
full observation time friends retain the original observation row order.
Equal row numbers therefore identify different events after the first rejected
event; truncating to the shorter length cannot correct this.

`apply/tools/align_prediction_time.py` generates a separate prediction-order
friend. Stage C's existing `t_recovered_time` interface stays unchanged.
The join uses the exact file-local integer key `(irun, ies, iseq, ievent)`.
No training, inference, or Stage C–G work is part of this correction.

## Source and output contract

- Original observations: `/mnt/newdisk/shijie/WCDA_obs/<MMDD>/Esg*.root`.
- Existing predictions: `/mnt/newdisk/shijie/WCDA_obs_eval_64748/<MMDD>/Esg*.root`.
- Original time friends: `/mnt/newdisk/shijie/WCDA_obs/recovered_time/<MMDD>/Esg*.time.root`.
- New prediction-order friends: `/mnt/newdisk/shijie/WCDA_obs_eval_64748_time_aligned/<MMDD>/Esg*.time.root`.

Every original time friend is bound to an observation receipt, recovery receipt,
ROOT recovery request ID, recovery code commit, summary, exception sidecar, and
receipt-recorded file stat. Its `t_alignment` identity and zero-based row index
are checked against every original event before the join. The three recovery
receipt directories cover the original, v2 and v3 deliveries; exactly one matching
request is required. January input and prediction inventories must agree on all
681 names.

Both original and prediction keys must be unique within each file. Missing keys,
duplicate keys, inconsistent source counts, changed source records, or unsupported
branch representations fail explicitly with a filename, error type and evidence
in the acceptance report. No nearest-time matching or row truncation is used.

All time branches are selected without casting or recalculation. Each output
branch retains its scalar dtype and ROOT typename. Reopened output values are
compared byte for byte with the original friend at the mapped indices, including
NaNs, coordinates, time precision and nonzero `match_status`. The output
`t_alignment` stores prediction event keys and their **original** row index;
it is an audit tree, not an extra Stage C input.

A `.time.receipt.json` beside each output records source paths, ROOT UUIDs, file
stats, observation record ID, recovery receipt hashes, sidecar hashes, mapping
and prediction-key digests, tool digest, branch types, counts and status
histograms. It also stores the complete output SHA-256. ROOT embeds the same
provenance. Input ROOT files are bound by their existing receipts, stats, UUIDs
and fully checked event keys; this tool does not recompute full hashes of the
large raw observation files.

The output root must be new and disjoint from all input roots. Each ROOT file is
published via an atomic hard link that cannot replace an existing path. The tool
also rejects an existing receipt/report. On failure, partial successful outputs
and evidence remain for inspection; rerun into a reviewed new output directory,
never delete or overwrite a source to resume. `--verify-only` accepts an existing
output only after checking its provenance and SHA-256, remapping all events and
rechecking every output field against source values.

## Validation and Slurm

The fixture reproduces original A/B/C and prediction A/C, preserves a nonzero
status and NaN, checks reordered prediction identities, checks missing keys in
each of the four fields and duplicate original/prediction keys, and rejects
changed source records and output overwrite.

```bash
cd /mnt/newdisk/shijie/energy
/mnt/newdisk/shijie/conda/envs/py310/bin/python -m unittest discover \
  -s apply/tests -p test_align_prediction_time.py
sbatch apply/tools/align_prediction_time.sbatch smoke
# Only after smoke acceptance:
sbatch apply/tools/align_prediction_time.sbatch month
```

The supplied CPU-only Slurm job runs from `apply/`. It first aligns and verifies
each written file, then performs a separate all-file `--verify-only` pass.
Reports are saved under
`/mnt/newdisk/shijie/energy/.recovery/2026-10-14/prediction-time-alignment/`.
Month acceptance requires exact output coverage, no mapping ambiguity/missing
keys, 69,279,369 original rows, 48,726,074 prediction/output rows and all 464
original nonzero statuses accounted for. The status count among predictions is
reported separately, including the rejected-by-inference remainder.

## Completed acceptance (2026-10-15 Asia/Shanghai)

Fixture tests pass (2 tests with the cases listed above). Real single-file Slurm
job **66078** passes both alignment and independent verification: 320,258 original
rows, 228,604 prediction rows, first mapped original indices `[0, 2, 3, 5, 7]`.

Full-month job **66080** completed with **ExitCode 0:0**, elapsed **4m15s**, peak
RSS **1,023,920 KiB**, using four CPUs and no GPU. Both passes accept **681/681
files**, **69,279,369 original events**, and **48,726,074 prediction/aligned rows**.
Every output field is bitwise equal to the mapped source, with original ROOT
branch types preserved. Missing/ambiguous event mappings: **0/0**.

Original status histogram: `[69278905, 464, 0, 0, 0, 0, 0]`.
Prediction/aligned histogram: `[48726074, 0, 0, 0, 0, 0, 0]`.
**None of the 464 original status-1 anomalies enters the prediction sample**;
the tool does not drop status rows. Original friends, full time sequences and
exception records are preserved.

Machine-readable summary: `PREDICTION_TIME_ALIGNMENT_ACCEPTANCE.json`.
Detailed 681-file provenance and acceptance reports:
`.recovery/2026-10-14/prediction-time-alignment/align_66080.json` and
`verify_66080.json`, plus one `.time.receipt.json` beside each output ROOT.
The checked-in summary records report SHA-256 digests and anomaly-file counts.

The initial full job **66079** correctly rejected 235 files because its launch
parameters omitted the v2 recovery receipt directory. Its failure report and
446 successful partial outputs are preserved, with the latter moved intact to
`/mnt/newdisk/shijie/WCDA_obs_eval_64748_time_aligned_partial_66079/`.
Job 66080 uses all three receipt scopes and writes a fresh final directory;
no validation was bypassed and no source or existing target was overwritten.

## Future Stage C connection (not executed)

From the ETO repository root, the existing Stage C interface accepts:

```bash
/mnt/newdisk/shijie/conda/envs/py310/bin/python apply/stages/03_reduce_obs.py \
  --obs-root /mnt/newdisk/shijie/WCDA_obs_eval_64748 \
  --time-root /mnt/newdisk/shijie/WCDA_obs_eval_64748_time_aligned \
  --cell-selection-csv apply/config/cell_selection_v1.csv \
  --output-dir apply/output/stage_c \
  --run-id january_64748_identity_aligned \
  --no-promote-current
```

Use the science branch's intended cell selection when launching a future
analysis. Do not pass `--allow-entry-mismatch`: it is an unsafe prefix shortcut
for these filtered predictions. Nonzero statuses remain in the new friend;
Stage C applies its existing `--match-status-equals 0` selection later.

Keep the full original time series and all recovery audit products indefinitely
for GTI/exposure auditing. The filtered friend is an event alignment artifact;
it must not replace the original full time sequence in exposure calculations.
This acceptance verifies identity/time alignment, not physical calibration or
any subsequent spectrum result.
