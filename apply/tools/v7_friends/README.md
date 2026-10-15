# Unfiltered v7 MC friends

Run `friends.py` only through ETO Slurm; `relay.py` runs on Mac using existing `ihep` and `ETO` SSH aliases. No inference or physics cuts occur. No original ROOT file is written. `chi2` is diagnostic-only per explicit user instruction on 2026-10-15; all other shared fields retain strict validation.

The frozen inventory is `apply/output/v7_friends_20261015/input_inventory.json`. It contains the 10,000 ETO eventout file names and stat identities; the separate IHEP pairing inventory establishes name existence only. A six-field exact float32 byte key is derived from `mc_energy, mc_dangle, mc_weight, mc_theta, theta, pincness`. Nonfinite values, duplicates in either file, missing/ambiguous matches and reused summary rows quarantine the entire file. This is content correspondence, not independent physical event-ID authentication. Current eventout files lack the native four event identity fields. Schema changes fail closed.

Every other common scalar field is compared after casting the higher precision value to the lower precision storage type, without tolerance. Integer fields compare exactly. Four user-defined rmds/theta boundaries are separately evaluated in float64, with the open interval endpoints preserved. No PINC boundary or correction is used. Quantile distributions and examples are recorded for all common fields, including chi2. Non-chi2 precision violations and any boundary flip quarantine the file.

The authority is native, **unprefixed** `nq03t30s0/1/2`. Counts must be nonnegative integers; int64 addition checks overflow before each add. Native integer dtypes and values are retained. Raw, unmodified `vqsamp` and `vt` are read for every event and compared against the sum. Any discrepancy is quarantined for review, never repaired. Native `poolused` values are retained, including non-7 values; unexpected values have diagnostic examples. Nhit below 30 is retained and separately counted outside the eight requested diagnostic bins. Unused summary rows are recorded and quarantined pending an explanation, never inserted into a friend.

Each immutable version directory contains `contract.json`, `files/<stem>/receipt.json`, successful `files/<stem>/<stem>.v7friend`, `summary.json` and `final_audit.json`. The ROOT TTree is `v7_friend`; all branches have prefix `v7_`: native fields, `nhit_authoritative` (int64), `eventout_entry` (int64), `summary_entry` (int64), and `match_status` (uint8, 1=unique verified content match). Entry order is exactly the original eventout order. Use an explicit ROOT friend alias such as `v7`; no registration is added to the original files. The `.v7friend` extension prevents discovery via `*.root`, including recursive discovery. Current `simulation_all_bin.py` also scans only immediate `.root` files.

A temporary package is written, closed and reopened; every output value and dtype is compared with the matched payload. Only then is its directory atomically renamed to the final path. Quarantined packages contain a receipt and no friend. Existing packages are never overwritten. The run path incorporates the core code hash. Never reuse a run path for a changed inventory, contract or code; keep old versions. No missing receipt or job exit code implies scientific success. Read summary diagnostics and run `audit.py` after processing.

Example ETO commands (from `apply/`; substitute the frozen version path):

```sh
srun --ntasks=1 --partition=main --qos=default-safe --cpus-per-task=1 --mem=1G --time=00:02:00 /mnt/newdisk/shijie/conda/envs/py310/bin/python tools/v7_friends/test_friends.py
sbatch --output=output/v7_friends_20261015/full_%j.log tools/v7_friends/run.sbatch /mnt/newdisk/shijie/WCDA_simulation/v7_friends/run_<codehash>
```

Mac relay command from `apply/`:

```sh
python3 tools/v7_friends/relay.py --inventory output/v7_friends_20261015/input_inventory.json --run /mnt/newdisk/shijie/WCDA_simulation/v7_friends/run_<codehash>
```

The relay transfers at most 100 native summaries at a time via a tar stream; Mac never parses ROOT. Failed transfers have no ready marker. Existing staging content must hash-identically match before reuse. It waits for receipts, confirms their summary hashes, and deletes only that batch's staging files. A failed/quarantined receipt is retained for diagnosis and is not automatically retried. Retry an explicitly reviewed failed subset in a new version/run directory. If the controller disconnects, rerun with the identical inventory/run after checking the Slurm job is alive; do not launch two relays or writers concurrently.

Final independent audit (Slurm, 1 CPU / 2 GB):

```sh
python tools/v7_friends/audit.py --inventory output/v7_friends_20261015/input_inventory.json --run /mnt/newdisk/shijie/WCDA_simulation/v7_friends/run_<codehash>
```

It reopens published friends, checks hashes, identities, mappings, schema, integer sums and metadata, and verifies quarantined packages have no friend. Original source stat identities are checked again; original SHA256 values were computed in Slurm during generation. Data and generated logs/manifests stay outside Git.

For faster direct ETO→IHEP transport, use the existing Mac agent through an explicit `ssh -A ETO` session, then run the same relay on ETO with `--on-eto`. This uses existing agent identities without copying keys or changing SSH configuration. Keep that SSH session alive until transfer completes. A failed Mac relay can be stopped at its owned process IDs; its unready incoming batch is safely retransferred. The independent Slurm writer remains running.

Within each bounded batch, four disjoint tar streams run concurrently. The total batch remains at most 100 summaries; the next batch starts only after receipts and hash-checked cleanup. `test_publication.py` exercises reordered synthetic ROOT files, non-7 poolused, ignored mc-prefixed counts, the chi2 exception, independent-field quarantine and scoped reporting. `test_relay.py` checks the local/SSH command paths.

Submit `finalize.sbatch RUN` with `--dependency=afterok:GENERATION_JOB` to run the independent published-file audit, threshold-adjacent raw-hit diagnostics, and `delivery_summary.json` creation. The final diagnostic only examines recorded mismatch examples; it does not resolve upstream definitions or release quarantined files. Exact `|vt|=30` hits can be a useful clue but are not a publication waiver. `delivery_summary.json` distinguishes published statistics from all fully matched diagnostic statistics and lists failures with retry guidance.
