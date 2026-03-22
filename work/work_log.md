# Work Log

## 2026-03-22 Update

- All recent changes were made on the `dev` branch.
- Switched the active theta dataset point centering from reconstructed core `(xc, yc)` to true MC core `(mc_xc, mc_yc)` in `src/theta/ParticleDataset_theta.py`.
- Kept `use_core_box` on `mc_xc` / `mc_yc` and clarified the related dataset logging / stats output with `true_core_box` and `point_center` metadata.
- Added a minimal no-core-limit smoke test Slurm script: `scripts/slurm/theta_truecore_points_smoketest.sbatch`.
- Added a formal no-core-limit training Slurm script: `scripts/slurm/theta_truecore_points_edge0.sbatch`.
- Submitted smoke test job `2732`; it completed successfully and produced config, dataset stats, checkpoint, loss curve, metrics, prediction arrays, and evaluation figures under `runs/theta_truecore_points_smoketest_2732`.
- Submitted formal training job `2733`; it entered `RUNNING` and created `runs/theta_truecore_points_edge0_2733`.
- Main caveat: using `mc_xc` / `mc_yc` injects truth-level core information, so these new runs are not directly comparable to older reco-core-centered runs.

## 2026-03-22 Compare Update

- Compared `runs/theta_truecore_points_edge0_2733` against `runs/no_core_cut_2724` as a true-core-points vs reco-core-points study.
- Checked `config.json`, `metrics.json`, `preds.npz`, and dataset stats for both runs.
- Training settings are effectively matched on the main control parameters: same root path, `n_files=10000`, `epochs=500`, `batch_size=512`, `max_points=500`, `sample_mode=weighted_q`, `dcedge_min=0`, `use_core_box=false`, `theta_embed_dim=16`, `loss_mode=huber`, and the same seed.
- This is a reasonable mainline control experiment by training configuration, but not a perfectly strict one-variable rerun.
- Caveats found during the check:
  - the older `no_core_cut_2724` run predates explicit `fitstat` fields in `config.json`, so those settings are not fully recorded in the same schema
  - `no_core_cut_2724` has `dataset_test_stats.json -> files.n_fail = 1`
  - dataset event totals are not exactly identical between the two runs, so the underlying file set or readout was not perfectly frozen
- Created notebook `notebook/truecore_vs_recocore_points_compare_2733_2724.ipynb`.
- The notebook reads:
  - `config.json`
  - `fig/metrics.json`
  - `fig/preds.npz`
  - `dataset_train_stats.json`
  - `dataset_test_stats.json`
- Generated weighted comparison figures under `notebook/generated/truecore_vs_recocore_points_compare_2733_2724/`:
  - `resolution_weighted_true_vs_reco_core_points.png`
  - `bias_weighted_true_vs_reco_core_points.png`
  - `logRMS_weighted_true_vs_reco_core_points.png`

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
