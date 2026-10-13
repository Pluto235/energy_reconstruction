# January observation pipeline optimization execution checklist

Goal: submit and verify a faster, byte-bounded IHEP→ETO January recovery while retaining all existing receipts and immutable inputs.

Architecture: retain the pincness-only scan plus selected hit reads. Return counts/timings from that scan, explicitly benchmark cache changes, and publish validated outputs into a shared byte-reserved queue. ETO persists metadata/size receipts before source cleanup. Legacy SHA receipts remain valid identities without new hashing.

Stack: ROOT 6.32, Python standard library / uproot on ETO, NFS file locks, HTCondor.

- [x] Snapshot source/ETO state; confirm job ownership and versions. Pause the old receiver, let producers finish their current outputs, then hold only the four identified jobs and preserve receipts.
- [x] Add regression cases for legacy/v2 acknowledgments, exclusive byte reservations, stale partial recovery, strict cuts/order/header retention and write failures.
- [x] Instrument the ROOT scan and selected hit phases. Use returned counts; compare historical cache behavior, cache off, 32 MiB and 128 MiB on two complete hourly inputs in HTCondor. Compare every retained leaf value/header and measure full vs sampled validation. Keep full validation unless measured overhead justifies sampling.
- [x] Implement durable receipts without hashing, per-file locks, a shared ledger counting reservations plus queued outputs, and kernel-enforced partial size limits. Producer retries cannot overwrite confirmed data; receiver retries transfers and cleans only acknowledged task staging.
- [x] Deploy snapshots. Requery personal quota and set capacity with safety headroom. Run an optimized four-worker cohort through real ETO transfers; then an eight-worker cohort. Compare normalized throughput, read calls/bytes, CPU/wall, memory and queue backlog before selecting production concurrency.
- [x] Verify a new whole-hour file reached ETO and its temporary source copy was safely removed. Submit production jobs, verify progress, update operation docs/devlog, commit/push and deploy the exact version. Report measurements, live IDs, capacity, remaining work and ETA.

No model inference, raw-file disk caching or changes to originals, archives, other projects or confirmed ETO files.
