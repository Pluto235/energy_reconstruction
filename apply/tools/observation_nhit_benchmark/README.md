# PINC-only scalar friend benchmark

This isolated benchmark evaluates the user's proposed extraction path. It never
writes production `full/`, production manifests, source tasks, or receiver config.
The one-worker test uses three complete files (Jan01/Jan16/Jan31 hour00), fresh
personal quota checks and bounded additional staging. Existing monthly work keeps
running. Test outputs are not accepted production friends.

- Baseline: a frozen copy of the current extractor, including all common-branch
  digests, charge arrays, complete source readback and independent ETO validation.
- Candidate: ROOT scans only `pincness < 1.1`, reads selected events' full
  `vqsamp/vt/vidmc/nv` and identities with the previously measured 32 MiB cache,
  computes the unchanged Nhit definition and retrieves identity-checked segment
  headers using the same helper as baseline. Charge arrays are not saved or sent.
- ETO: validate transfer digests, reject duplicate keys, align candidates to the
  original observation input, record missing/surplus events, and compare all
  matched keys/nv/Nhit/poolused/source entries against the freshly rerun baseline.
  Missing targets fail this benchmark. Equality of Nhit implies equality in all
  eight specified bins. This is candidate-versus-baseline evidence, not independent
  ETO charge-based recomputation of the candidate.
- Execution order is baseline/candidate, candidate/baseline, baseline/candidate.
  Both implementations run in one Condor worker on one host; cache state cannot
  be reset, so this is a small paired test, not a cluster-throughput guarantee.
- ROOT setup/compile cost is reported separately. Per-file source time includes
  metadata/header checks and complete output readback. Transfers and ETO checks
  are timed separately, excluding controller polling and scheduler queue delay.
- The baseline and candidate intentionally perform different validation work.
  Any speedup represents the proposed workflow, not just changing the match site.

Files: `benchmark.py` controller/source worker, `pinc_only.C` candidate implementation.
Run alignment self-check with `benchmark.py --selftest` in the existing ETO py310
runtime. C++ threshold/time boundary assertions run in the actual IHEP worker.
The isolated run ID and paths are fixed to prevent accidental production reuse;
restarting the controller rejects an existing run instead of duplicating jobs.

Initial launch: Slurm 66148, Condor 3739428. Evidence directory:
`/mnt/newdisk/shijie/WCDA_obs_friends/nhit/benchmarks/20261015_pinc_only_v1/`.
Source directory: `/scratchfs2/lhaaso/liushijie/WCDA_nhit_bench_20261015_pinc_only_v1/`.
The final comparison and provenance are in `<name>.comparison.json`, manifest,
source ready records and `complete.json`. These generated files stay outside Git.

`finalize.py` runs in a dependent Slurm job after the paired comparisons. It
creates `*.aligned.root` test friends with input_entry, performs full scalar and
segment readback, and writes `*.final.json` plus `summary.json`. Its ETO time
includes fresh identity alignment, baseline comparison, writing, fsync and digest.
Slurm 66150 completed this step. Results: source speedup 1.68×, measured processing
speedup 1.78×, 582417 events with zero mismatches, 148.08 MB → 6.26 MB after alignment.
See `apply/docs/OBS_JANUARY_NHIT_PINC_BENCHMARK.md` for evidence and limits.
