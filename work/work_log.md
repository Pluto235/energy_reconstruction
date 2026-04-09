# Work Log

## Timestamp

- 2026-03-19 16:58:08 CST (+0800)

## Current Work Summary

- Reviewed the maintained theta-based WCDA energy reconstruction pipeline.
- Reconstructed the active code path from dataset loading to training and evaluation.
- Compared the current branch structure with the older non-theta baseline and the earlier theta branch.
- Checked existing runs, metrics, Slurm scripts, and notebooks to infer the current research direction.

## Current Focus Areas

- Theta embedding as an event-level conditioning mechanism.
- Generalization under relaxed event-quality and geometry cuts.
- Comparison studies for:
  - `fitstat`
  - `dcedge` / core selection
  - `dangle`
  - `pincness`
  - `theta`
- Evaluation-only cut override studies on an already trained strict-cut model.

## Recently Performed Experiments

- `fitstat0` vs `fitstat_all`
  - Impact appears small under the current strict baseline setup.
- `dangle < 3 deg` vs relaxed `dangle`
  - Strong degradation when the cut is relaxed.
- `pincness < 1.1` vs relaxed `pincness`
  - Noticeable degradation when the cut is relaxed.
- `dcedge` / core-selection comparisons
  - Relaxing `dcedge` appears less harmful than relaxing `dangle` or `pincness`.
- `theta` comparison on a 10% sample
  - Relaxed `theta` looked acceptable or even favorable in the limited test, but this remains uncertain.
- `allcuts` strict vs relaxed
  - Strict version completed.
  - Large-scale fully relaxed training did not complete cleanly.
  - A 10% relaxed run completed but performed very poorly.

## Current Conclusions

- The strict-cut theta pipeline is operational for internal experiments.
- The model is not yet robust over broadly relaxed event selections.
- `dangle` is currently the clearest failure mode for out-of-domain evaluation.
- `pincness` also carries substantial generalization sensitivity.
- Full all-cuts relaxation likely changes the sample domain too strongly for the current training recipe.
- The practical value of `fitstat` filtering is still unclear but looks limited in the current runs.

## Open Questions

- Whether the apparent benefit of relaxed `theta` is real or sample-dependent is unclear.
- Whether `fitstat` should remain a hard cut, a study variable, or be removed is still unclear.
- It is unclear how much of the current performance comes from strict sample cleaning rather than a more intrinsically robust model.
- It is unclear whether future improvements should prioritize better data coverage or stronger conditional modeling.

## Current Technical Risks

- Evaluation correctness issues are present and should be fixed before further comparison studies are trusted.
- Some current working-tree changes are uncommitted experimental modifications.
- The current pipeline is strongly tied to local absolute paths.
- Dataset loading may not scale well for relaxed high-statistics regimes.

## Next Planned Steps

- Fix the known evaluation and training-argument pass-through bugs.
- Consolidate the current relaxed-evaluation findings into a stable reference note.
- Add stronger reproducibility metadata for runs.
- Extend evaluation-only studies to `theta` and possibly `fitstat`.
- Consider a more scalable data-loading strategy before attempting another full relaxed all-cuts training run.

## 2026-03-23 01:00 CST (+0800)

### Branch

- `nv-embed`

### Position-Embed Preservation

- Confirmed the previous experiment branch was `position-embed`.
- Committed the completed position/core embedding work and pushed it to remote.
- Commit:
  - `e144a9b183ed34515aaab8d6daa6a3c1610a6b5a`
- Remote branch:
  - `origin/position-embed`

### New NV Experiment Setup

- Synced `main`, created `nv-embed`, and pushed it to remote.
- `nv` here means the per-event effective hit count from ROOT branch `nv`.
- Goal:
  - keep the existing theta embedding
  - add `nv` as another event-level conditioning input

### NV Conditioning Design

- Dataset now returns both:
  - `costheta = cos(theta)`
  - normalized `nv`
- Chosen normalization:
  - `nv_norm = log1p(nv) / log1p(3000.0)`
- Reason:
  - `nv` is a positive count variable with a wide dynamic range
  - log scaling keeps it in a stable `O(1)` range without discarding monotonic count information
- Model change:
  - added a small `nv` MLP branch
  - concatenated `point-cloud pooled feature + theta embedding + nv embedding` before the regression head

### Modified Files

- `src/theta/ParticleDataset_theta.py`
  - return normalized `nv`
  - record `nv` transform in dataset stats
- `src/theta/ParticleRegressor_theta.py`
  - add `nv` embedding branch and forward input
- `src/theta/main_theta.py`
  - add CLI/config parameters for `nv`
  - wire dataset/model/sanity forward
- `src/theta/train_theta.py`
  - pass `nv` through training and validation
- `src/theta/evaluate_theta.py`
  - pass `nv` through evaluation
  - save full-event `nv` array in `preds.npz`
- `src/theta/evaluate_only.py`
  - add backward-compatible support for `nv` config/model input
- `src/theta/eval_compare_relaxed.py`
  - add backward-compatible support for `nv` config/model input
- `scripts/slurm/theta_nv_embed_smoketest.sbatch`
  - new smoke test script
- `scripts/slurm/theta_nv_embed.sbatch`
  - new formal training script

### Smoke Test

- Final smoke script:
  - `scripts/slurm/theta_nv_embed_smoketest.sbatch`
- Successful smoke job:
  - `2739`
- Output directory:
  - `runs/theta_nv_embed_smoketest_2739`
- Logs:
  - `logs/slurm/theta_nv_smoke_2739.out`
  - `logs/slurm/theta_nv_smoke_2739.err`
- Result:
  - completed successfully
  - generated config, dataset stats, loss log, checkpoint, metrics, preds, and evaluation figures
- Note:
  - an earlier GPU-targeting smoke attempt was blocked by the running 4-GPU training job, so the final smoke script was switched to CPU mode for fast validation

### Formal Training Submission

- Formal script:
  - `scripts/slurm/theta_nv_embed.sbatch`
- Formal job:
  - `2740`
- Planned output directory:
  - `runs/theta_nv_embed_2740`
- Submission state at check time:
  - successfully submitted and pending on GPU resources

## 2026-03-23 12:30 CST (+0800)

### Local Workspace Untracking

- Goal:
  - keep `work/`, `tasks/`, `runs/`, and `logs/` available locally across branches
  - stop tracking `work/` and `tasks/` in the active branch heads
- Branches cleaned:
  - `main`
  - `nv-embed`
  - `exp/truecore-points`
- Branch already local-only:
  - `position-embed`
- Result:
  - remote branch HEADs no longer track `work/` and `tasks/`
  - local copies remain on disk for comparison and experimental notes
- Note:
  - `runs/` and `logs/` were already local/shared and remained untouched

## 2026-03-23 13:15 CST (+0800)

### 2724 vs 2735 Comparison

- Compared:
  - `runs/dcedge20_2724`
  - `runs/theta_position_embed_2735`
- Judgement:
  - `runs/dcedge20_2724` is the `theta embed only` baseline
  - `runs/theta_position_embed_2735` is `theta + position embed`
- Evidence used:
  - `runs/dcedge20_2724/config.json` has `theta_embed_dim=16` and no `core_embed_dim`
  - `runs/theta_position_embed_2735/config.json` has `theta_embed_dim=16`, `core_embed_dim=16`, `core_scale_x=130.0`, `core_scale_y=110.0`
  - `logs/slurm/theta_pos_embed_2735.out` explicitly starts with `theta + true-core embedding`
  - `metrics.json`, `preds.npz`, `dataset_*_stats.json` were read from both runs before plotting
- Plotting method:
  - reused notebook-style binning on the combined true-energy range
  - used weighted resolution, bias, and log RMS definitions consistent with the notebook
- Notebook:
  - `notebook/theta_only_vs_position_embed_2724_2735.ipynb`
- Output directory:
  - `notebook/generated/theta_only_vs_position_embed_2724_2735/`
- Generated figures:
  - `resolution_theta_only_vs_position_embed_2724_2735.png`
  - `bias_theta_only_vs_position_embed_2724_2735.png`
  - `logRMS_theta_only_vs_position_embed_2724_2735.png`
- Notes:
  - no retraining was performed
  - the comparison summary JSON was saved under `notebook/generated/theta_only_vs_position_embed_2724_2735/`
  - figures were regenerated with display-friendly titles: `Theta-only baseline` vs `Theta + position embedding`

## 2026-03-23 14:10 CST (+0800)

### 2724 vs 2740 Comparison

- Compared:
  - `runs/dcedge20_2724`
  - `runs/theta_nv_embed_2740`
- Judgement:
  - `runs/dcedge20_2724` is the `theta-only baseline`
  - `runs/theta_nv_embed_2740` is `theta + nv embedding`
- Evidence used:
  - `runs/dcedge20_2724/config.json` has `theta_embed_dim=16` and no `nv_*` fields
  - `runs/theta_nv_embed_2740/config.json` adds `nv_embed_dim=8`, `nv_embed_dropout=0.0`, `nv_scale=3000.0`
  - `logs/slurm/theta_nv_embed_2740.out` explicitly starts with `theta + nv embedding`
  - `metrics.json`, `preds.npz`, `dataset_*_stats.json` were read from both runs before plotting
- Plotting method:
  - reused notebook-style shared binning on the combined true-energy range
  - used weighted resolution, bias, and log RMS definitions consistent with the notebook comparisons
- Notebook:
  - `notebook/theta_only_vs_nv_embed_2724_2740.ipynb`
- Output directory:
  - `notebook/generated/theta_only_vs_nv_embed_2724_2740/`
- Generated figures:
  - `resolution_theta_only_vs_nv_embed_2724_2740.png`
  - `bias_theta_only_vs_nv_embed_2724_2740.png`
  - `logRMS_theta_only_vs_nv_embed_2724_2740.png`
- Notes:
  - no retraining was performed
  - this is a reasonable control comparison, with the only meaningful config additions being the `nv` embedding fields plus bookkeeping fields in the new run config

## 2026-03-25 16:13 CST (+0800)

### theta_embed_mid_2709 All-Energy Eval

- Evaluated model:
  - `runs/theta_embed_mid_2709`
  - checkpoint: `runs/theta_embed_mid_2709/checkpoints/best_model.pt`
- Original training energy range:
  - `Emin=1000 GeV`
  - `Emax=100000 GeV`
  - i.e. `1 TeV - 100 TeV`
- Evaluation-only energy override:
  - reused `src/theta/eval_compare_relaxed.py`
  - added minimal CLI support to explicitly set eval float cuts to `None`
  - ran with `--eval_Emin none --eval_Emax none`
  - other eval cuts kept at the training-domain values:
    - `pinc_max=1.1`
    - `dcedge_min=20`
    - `dangle_max_deg=3`
    - `theta_max_deg=30`
    - `require_fitstat0=true`
- Output directory:
  - `runs/theta_embed_mid_2709/fig_eval_all_energy`
- Artifacts confirmed:
  - `effective_eval_config.json`
  - `metrics.json`
  - `preds.npz`
  - unweighted and weighted evaluation figures
- Effective result:
  - completed successfully without retraining
  - `metrics.json` reports `n=438017`
  - `preds.npz` true-energy coverage is approximately `2.8 GeV` to `998 TeV` on the kept eval sample

## 2026-04-03 20:44 CST (+0800)

### Apply-Stage Full-Simulation Inference And 2D Binning

- Working directory:
  - `apply/`
- Goal:
  - build an apply-stage script to run inference over all events in `/mnt/mydisk/WCDA_simulation`
  - reuse the maintained theta pipeline preprocessing and model-loading logic
  - split output by `nhit` and predicted energy bins

### Reuse And Consistency Check

- Checked the maintained training/evaluation path:
  - `src/theta/main_theta.py`
  - `src/theta/ParticleDataset_theta.py`
  - `src/theta/ParticleRegressor_theta.py`
  - `src/theta/evaluate_theta.py`
- Confirmed model inputs in the maintained `no_core_cut_2724` run:
  - points:
    - detector hit coordinates `[vx, vy]`
  - features:
    - `[vq, vt]`
  - event-level conditioning:
    - `costheta = cos(theta)`
- Confirmed preprocessing behavior reused for apply:
  - `process_features(...)` from `src/common/EdgeConv.py`
  - per-event feature normalization from `ParticleDataset_theta.py`
  - hit truncation with `sample_mode=weighted_q`
  - `max_points=500`
- Confirmed model output semantics:
  - network output is already `log10(E / GeV)`
  - no inverse-transform step is used in `evaluate_theta.py`
- Checked `nhit` correspondence:
  - current project uses event-level branch `nv`
  - `n` is not equivalent to `nv`
  - a sample ROOT inspection showed large `n - nv` differences, so apply binning should use `nv`
- Confirmed existing ROOT I/O reuse path:
  - input reading already uses `uproot`
  - output writing can safely reuse `uproot.recreate(...)`

### New Script

- Added:
  - `apply/simulation_all_bin.py`
- Script behavior:
  - traverses all ROOT files under `/mnt/mydisk/WCDA_simulation`
  - loads model weights from `runs/no_core_cut_2724/checkpoints/best_model.pt`
  - reproduces the maintained preprocessing path for inference
  - predicts `ml_logE_pred`
  - bins events by:
    - `nhit := nv`
    - predicted `log10(E / GeV)`
  - writes per-bin ROOT outputs under `/mnt/mydisk/WCDA_simulation_binned`
  - preserves original branches from each source file and appends:
    - `ml_logE_pred`
    - `nhit_bin`
    - `predE_bin`
  - writes:
    - `summary/bin_counts.csv`
    - `summary/bin_counts.md`
    - `summary/run_summary.json`
- Output organization choice:
  - one directory per 2D bin
  - inside each bin directory, one ROOT file per source ROOT file
- Reason:
  - avoids collecting all events for a bin in memory
  - scales better to 10000 input files
  - preserves source-file traceability

### Environment And Device Check

- Training/inference environment confirmed from Slurm scripts:
  - `conda activate py310`
- GPU status checked before dry-run:
  - GPU `0/2/3` largely idle
  - GPU `1` busy
- Dry-run used:
  - `cuda:0`

### Dry-Run Validation

- Dry-run command:
  - `PYTHONPATH=/mnt/mydisk/server/projects/energy_reconstruction python apply/simulation_all_bin.py --max-files 1 --batch-size 128 --device cuda --gpu-id 0 --output-root /mnt/mydisk/WCDA_simulation_binned_dryrun --print-every 1`
- Dry-run result:
  - completed successfully on `1` ROOT file
  - processed source file:
    - `Egr000000000_01_n0010_eventout.root`
  - counts:
    - total events: `3808`
    - inferred events: `3808`
    - `nhit` out-of-range events: `199`
- Dry-run artifacts confirmed:
  - `/mnt/mydisk/WCDA_simulation_binned_dryrun/summary/bin_counts.csv`
  - `/mnt/mydisk/WCDA_simulation_binned_dryrun/summary/bin_counts.md`
  - `/mnt/mydisk/WCDA_simulation_binned_dryrun/summary/run_summary.json`
  - multiple per-bin ROOT outputs under `/mnt/mydisk/WCDA_simulation_binned_dryrun/nhit_*/predE_*/`

### Notes

- The apply script intentionally does not modify:
  - training code
  - evaluation code
  - any artifact under `runs/no_core_cut_2724`
- Full 10000-file production inference has not yet been launched in this turn.
- Based on the 1-file dry-run, the likely low-statistics region is:
  - very high predicted energy bins
  - very high `nhit` bins
  - very low `nhit` bins after excluding `<30` from formal outputs
- This is physically expected because:
  - extreme-energy showers are rarer
  - very large `nv` events are rare
  - very low-`nv` events tend to stay concentrated at lower predicted energies

## 2026-04-03 21:05 CST (+0800)

### Apply CPU-Side Read/Preprocess Optimization Test

- Scope:
  - optimize only CPU-side file reading / preprocessing
  - keep GPU inference path unchanged
- Script updated:
  - `apply/simulation_all_bin.py`
- Added controls:
  - `--reader-workers`
  - `--prefetch-files`
  - `--reader-backend {thread,process}`
- Implementation:
  - file-level prefetch / preprocess pipeline
  - sequential baseline still available with `--reader-workers 0`
  - no change to model weights, forward path, or GPU selection logic

### Benchmark Setup

- Test sample:
  - first `3` ROOT files from `/mnt/mydisk/WCDA_simulation`
- Shared settings:
  - `batch-size=128`
  - `device=cuda`
  - physical GPU `2`
  - same output logic and same summary generation
- Commands used:
  - sequential baseline:
    - `python apply/simulation_all_bin.py --max-files 3 --batch-size 128 --device cuda --gpu-id 2 --reader-workers 0 ...`
  - process-prefetch test:
    - `python apply/simulation_all_bin.py --max-files 3 --batch-size 128 --device cuda --gpu-id 2 --reader-workers 4 --reader-backend process --prefetch-files 4 ...`
  - thread-prefetch test:
    - `python apply/simulation_all_bin.py --max-files 3 --batch-size 128 --device cuda --gpu-id 2 --reader-workers 4 --reader-backend thread --prefetch-files 4 ...`

### Benchmark Result

- Sequential baseline:
  - script-reported elapsed:
    - `122.80 s`
  - wall time:
    - `125.48 s`
- Process prefetch (`4` workers):
  - script-reported elapsed:
    - `124.30 s`
  - wall time:
    - `126.94 s`
- Thread prefetch (`4` workers):
  - script-reported elapsed:
    - `128.63 s`
  - wall time:
    - `131.36 s`

### Conclusion

- Under the current implementation and data shape, CPU-side parallel prefetch did **not** improve end-to-end runtime.
- The likely reasons are:
  - per-file ROOT write-out remains a significant fixed cost
  - process backend pays extra IPC / serialization cost for prepared arrays
  - thread backend avoids IPC but still does not hide enough preprocessing / I/O latency to beat the simple sequential path
- Practical recommendation for now:
  - keep production runs on the sequential reader path:
    - `--reader-workers 0`
  - do not assume many CPU cores will automatically speed this apply stage up
- Future optimization directions, if needed later:
  - more detailed stage timing per file
  - reduce ROOT write fragmentation
  - batch writes or merge per-bin outputs less frequently
  - revisit whether all branches must be loaded before writeback

## 2026-04-03 21:25 CST (+0800)

### Apply Multi-GPU Sharding Support

- User action requested:
  - cancel queued apply job `4456`
  - update apply-stage code so it can use `4` GPUs for higher throughput
- Queue action:
  - cancelled job:
    - `4456`

### Code Change

- Updated:
  - `apply/simulation_all_bin.py`
- Added multi-GPU support via file sharding:
  - new CLI:
    - `--gpu-ids`
  - behavior:
    - if multiple GPU ids are provided, the script splits input ROOT files round-robin across GPUs
    - one spawned worker process owns one GPU
    - each GPU loads the same trained model weights independently
    - each worker writes ROOT outputs for only its assigned files
    - the parent process merges final counters and writes the global summary
- Reason for this design:
  - better fit for apply-stage file-based batch processing than `torch.nn.DataParallel`
  - avoids cross-GPU synchronization on each minibatch
  - preserves the current per-file output structure

### Multi-GPU Test

- Validation command:
  - `python apply/simulation_all_bin.py --max-files 2 --batch-size 128 --device cuda --gpu-ids 0,3 --reader-workers 0 --output-root /mnt/mydisk/WCDA_simulation_binned_multigpu_test --print-every 1`
- Result:
  - completed successfully
  - shard on GPU `0` processed `1` file in about `40.79 s`
  - shard on GPU `3` processed `1` file in about `41.24 s`
  - overall wall-clock elapsed:
    - `44.28 s`
  - merged summary completed correctly
  - total events in the 2-file test:
    - `7615`
  - inferred events:
    - `7615`
  - out-of-range `nhit` events:
    - `402`

### Slurm Script Update

- Updated:
  - `scripts/slurm/apply_simulation_all_bin.sbatch`
- New resource request:
  - `--gres=gpu:4`
  - `--cpus-per-task=32`
  - `--mem=64G`
- New apply command:
  - passes `--gpu-ids 0,1,2,3`

### Notes

- The tested multi-GPU path was validated on `2` currently freer GPUs.
- The same sharding logic generalizes directly to `4` GPUs in the updated sbatch script.

## 2026-04-04 18:02 CST (+0800)

### Filtered Apply Inference Variant

- Requested change:
  - keep current running job `4461` untouched
  - create a new apply variant that performs inference only after event-level filtering
  - preserve the same ROOT read/write structure
  - write to a new output root directory so it does not conflict with the unfiltered run

### Git Baseline

- Committed the pre-change apply code as a baseline:
  - commit:
    - `9577630`
  - message:
    - `Add apply-stage simulation binning pipeline`

### Event-Cut Logic Added

- Updated:
  - `apply/simulation_all_bin.py`
- New optional CLI controls:
  - `--apply-event-cuts`
  - `--cut-pinc-max`
  - `--cut-dangle-max-deg`
  - `--cut-theta-max-deg`
  - `--cut-fitstat-equals`
- Implemented event-level cuts before preprocessing / inference:
  - `pincness < 1.1`
  - `mc_dangle < 3 deg`
  - `fitstat == 0`
  - `theta < 30 deg`
- Important:
  - output organization is unchanged
  - only the selected events are inferred and written out
  - summary still reports total raw events and successful inferred events separately

### Dry-Run Check

- Dry-run command:
  - `python apply/simulation_all_bin.py --max-files 1 --batch-size 128 --device cuda --gpu-id 0 --reader-workers 0 --apply-event-cuts --cut-pinc-max 1.1 --cut-dangle-max-deg 3.0 --cut-theta-max-deg 30.0 --cut-fitstat-equals 0 --output-root /mnt/mydisk/WCDA_simulation_binned_selectedcuts_test --print-every 1`
- Result:
  - completed successfully
  - source file total events:
    - `3808`
  - inferred events after cuts:
    - `336`
  - out-of-range `nhit` events after cuts:
    - `1`
- Interpretation:
  - confirms the new event cuts are active before inference

### New Slurm Job

- Added sbatch:
  - `scripts/slurm/apply_simulation_all_bin_selectedcuts.sbatch`
- New output root:
  - `/mnt/mydisk/WCDA_simulation_binned_selectedcuts`
- Submitted new filtered 4-GPU job:
  - job id:
    - `5122`
  - name:
    - `apply_sim_cutsel`
- Existing unfiltered 4-GPU job kept running:
  - job id:
    - `4461`
  - state at check time:
    - `RUNNING`

### Queue Priority Adjustment

- Used Slurm admin commands to move the new filtered job as far forward as possible:
  - `scontrol top 5122`
  - `scontrol update JobId=5122 Nice=-10000`
- State after adjustment:
  - `PENDING`
  - reason:
    - `None`
- At check time the main partition queue showed:
  - `5122` at the top of the pending list
