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
