# v6 roadmap: rerun apply on `WCDA_observation_eval_64670`

## Goal

Use the newly generated half-year observation dataset

```text
/mnt/mydisk/WCDA_observation_eval_64670
/mnt/mydisk/WCDA_observation_eval_64670/recovered_time
```

to rerun the full Crab `apply` analysis from scratch as v6.

The v6 physics and analysis contract should match the current `baselinev4` branch:

- same two-dimensional `(Nhit, log10 E_pred)` candidate grid used by v3/v4;
- same v4 drop4 cell-selection policy and excluded cells;
- same annulus-normalized ROI-local background strategy;
- same aperture-conditioned response contract;
- same Stage F/G forward-folding and diagnostic SED flow.

The critical difference is that v6 observation ROOT files were made with a
different trained model from v1-v5. Therefore v6 must rebuild every MC/model
dependent artifact with the same `_64670` model. Do not reuse v1-v5 Stage A/B
response, PSF, signal, fit, or SED products as v6 inputs.

## Hard Constraints

1. Run heavy work on ETO through Slurm, not on the Mac workspace.
2. Keep v6 outputs isolated under new v6 names. Do not overwrite v1-v5 output
   directories or `current` symlinks unless explicitly intended and documented.
3. Do not mix model generations:
   - v6 observation input must be `/mnt/mydisk/WCDA_observation_eval_64670`.
   - v6 MC binned response input must be generated with
     `/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670`.
   - v6 Stage A/B must read the v6 MC binned cache, not the old
     `/mnt/mydisk/WCDA_simulation_binned_response_v1` cache.
4. Keep `devlog.md` newest-first. After meaningful edits, commit and push only
   the intended files; the ETO worktree currently contains unrelated dirty files.

If any command below needs to be changed because a file is missing, stop and ask
one concrete question before substituting a different dataset or model.

## Baseline v4 Contract To Preserve

Current v4 primary result is the aperture-conditioned branch documented by:

```text
apply/run_v4_aperture_conditioned_response_pipeline.sh
apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz
apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow/signal_v4_containment1_annnorm.npz
apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz
apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4/sed_points_v4_aperture_conditioned_drop4.npz
```

The v4 fit selector is:

```text
apply/config/cell_selector_v4_drop4_psfborrow.csv
```

The v4 fit cells are:

```text
1,2,3,
14,15,16,
26,27,28,29,30,
40,41,42,
52,53,54,55,
65,66,67,68,69,
81,82,83
```

The v4 drop4 control exclusions are:

```text
4,17,39,43
```

For v6, preserve these included/excluded cell IDs unless the `_64670` MC
preflight shows that a preserved cell is structurally invalid. If that happens,
do not silently change the selector; write the failed prefit evidence and ask
whether to keep v4 cell IDs for comparability or regenerate a new v6 selector.

## Recommended v6 Names

Use these names so downstream inspection is unambiguous:

```text
MC binned cache:
  /mnt/mydisk/WCDA_simulation_binned_response_v6_64670

Configs:
  apply/config/cell_ledger_v6_candidate.csv
  apply/config/cell_selector_v6_drop4_psfborrow.csv

Stage A:
  apply/output/stage_a_v6_64670
  response_2d_v6_64670.npz
  response_2d_v6_64670_metadata.json

Stage B:
  apply/output/stage_b_v6_64670
  run id: v6_psf_from_64670
  psf_v6_64670.npz
  psf_v6_64670_metadata.json

Stage A aperture-conditioned:
  apply/output/stage_a_v6_64670_aperture_conditioned
  response_2d_v6_64670_aperture_conditioned.npz
  response_2d_v6_64670_aperture_conditioned_metadata.json

Stage C:
  apply/output/stage_c_v6_64670
  run id: v6_stage_c_64670_halfyear

Stage D:
  apply/output/stage_d_v6_64670_annnorm
  run id: v6_stage_d_64670_annnorm

Stage E:
  apply/output/stage_e_v6_64670_containment1_annnorm
  run id: v6_stage_e_64670_containment1_annnorm

Stage F:
  apply/output/stage_f_v6_64670_baselinev4
  run id: v6_stage_f_64670_baselinev4

Stage G:
  apply/output/stage_g_v6_64670_baselinev4
  run id: v6_stage_g_64670_baselinev4
```

## Phase 0: Preflight The Dataset

Run from:

```bash
cd /home/server/projects/energy_reconstruction
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}
```

Verify observation and time friend trees:

```bash
find /mnt/mydisk/WCDA_observation_eval_64670 -mindepth 2 -maxdepth 2 \
  -type f -name 'Esg*.root' | wc -l

find /mnt/mydisk/WCDA_observation_eval_64670/recovered_time -mindepth 2 -maxdepth 2 \
  -type f -name 'Esg*.time.root' | wc -l

du -sh /mnt/mydisk/WCDA_observation_eval_64670 \
       /mnt/mydisk/WCDA_observation_eval_64670/recovered_time \
       /home/server/projects/energy_reconstruction/apply/output
df -h /mnt/mydisk /home/server
```

Acceptance gate:

- eval ROOT count and recovered-time ROOT count must match by `MMDD/Esg...` stem;
- target date coverage should match the intended half-year window;
- free space on `/mnt/mydisk` should stay above 250G before starting Stage C/D;
- no recovered-time file should have large `match_status != 0` counts.

If the half-year window is not `2022-01-01` to `2022-06-30`, update run labels
and reports consistently before starting Stage C.

## Phase 1: Build The `_64670` MC Binned Cache

The old MC binned cache is tied to older model predictions. Rebuild MC event
predictions and cell-bin caches using the same `_64670` model as the observation
eval ROOT files.

Start from `scripts/slurm/apply_simulation_all_bin.sbatch`, but create or run a
v6 override with:

```bash
INPUT_ROOT=/mnt/mydisk/WCDA_simulation
OUTPUT_ROOT=/mnt/mydisk/WCDA_simulation_binned_response_v6_64670
RUN_DIR=/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670
```

Keep the same event/cut semantics used by the current response pipeline. If the
existing script still defaults to an older run dir such as `no_core_cut_2724`,
patch a dedicated v6 Slurm wrapper rather than editing a shared production
script in place.

Suggested smoke test before full MC inference:

```bash
python apply/simulation_all_bin.py \
  --input-root /mnt/mydisk/WCDA_simulation \
  --run-dir /home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670 \
  --output-root /tmp/WCDA_simulation_binned_response_v6_64670_smoke \
  --max-files 2 \
  --device cpu \
  --batch-size 64 \
  --print-every 1
```

Then submit the full Slurm job. Save job IDs and logs in the final report.

Acceptance gate:

- `/mnt/mydisk/WCDA_simulation_binned_response_v6_64670/summary/bin_counts.csv`
  exists;
- bin labels cover the same source labels needed to derive the v3/v4 candidate
  grid;
- model path in summary metadata is the `_64670` run dir;
- no old `_8666` or v1-v5 binned root appears in v6 metadata.

## Phase 2: Create v6 Candidate Ledger And Drop4 Selector

Use the existing v3/v4 binning logic as the template, but write new v6 config
files. The starting point is:

```text
apply/stages/00_prepare_v3_inputs.py
apply/config/cell_ledger_v3_candidate.csv
apply/config/cell_selector_v4_drop4_psfborrow.csv
```

Recommended approach:

1. Run or patch `00_prepare_v3_inputs.py` with:

```bash
python apply/stages/00_prepare_v3_inputs.py \
  --source-binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64670 \
  --target-binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64670_candidate \
  --source-bin-counts-csv /mnt/mydisk/WCDA_simulation_binned_response_v6_64670/summary/bin_counts.csv \
  --candidate-ledger-csv apply/config/cell_ledger_v6_candidate.csv \
  --baseline-selector-csv apply/config/cell_selector_v6_drop4_psfborrow.csv \
  --write-configs \
  --prepare-cache \
  --write-diagnostics \
  --workers 18
```

2. Enforce the v4 drop4 cell contract on `cell_selector_v6_drop4_psfborrow.csv`:
   included cells must be the v4 fit cell IDs listed above; cells `4,17,39,43`
   must remain excluded with explicit `v4_drop4_bias_control` reasons.
3. Refresh `mc_count`, `central99_flag`, `ridge_peak_fraction`, and PSF quality
   fields from the `_64670` MC, but do not use Crab excess, Stage F pulls, or
   Stage G residuals to define the selector.

Acceptance gate:

- selector has the same included and excluded cell IDs as v4;
- all rows reference v6 or `_64670` in `subset_version` or metadata;
- no selector decision is derived from on-source Crab data.

## Phase 3: Build v6 Stage B PSF And Stage A Responses

Build nominal Stage A first. Stage B uses this metadata to verify the cell
contract, and the final v6 fit later uses the aperture-conditioned Stage A
response:

```bash
python apply/stages/01_build_response.py \
  --binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64670_candidate \
  --cell-selection-csv apply/config/cell_ledger_v6_candidate.csv \
  --output-dir apply/output/stage_a_v6_64670 \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --allow-missing-cell-dirs \
  --denominator-workers 1 \
  --numerator-workers 18 \
  --numerator-files-per-task 250 \
  --npz-name response_2d_v6_64670.npz \
  --metadata-name response_2d_v6_64670_metadata.json
```

Build nominal Stage B PSF:

```bash
python apply/stages/02_build_psf.py \
  --binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64670_candidate \
  --cell-selection-csv apply/config/cell_ledger_v6_candidate.csv \
  --stage-a-metadata apply/output/stage_a_v6_64670/response_2d_v6_64670_metadata.json \
  --output-dir apply/output/stage_b_v6_64670 \
  --run-id v6_psf_from_64670 \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --allow-low-stat-psf-fallback \
  --npz-name psf_v6_64670.npz \
  --metadata-name psf_v6_64670_metadata.json \
  --summary-csv-name psf_v6_64670_summary.csv \
  --summary-md-name psf_v6_64670_summary.md \
  --overwrite-run-dir
```

Build aperture-conditioned Stage A using the v6 PSF:

```bash
python apply/stages/01_build_response.py \
  --binned-root /mnt/mydisk/WCDA_simulation_binned_response_v6_64670_candidate \
  --cell-selection-csv apply/config/cell_ledger_v6_candidate.csv \
  --output-dir apply/output/stage_a_v6_64670_aperture_conditioned \
  --tree-name t_eventout \
  --weight-branch mc_weight \
  --allow-missing-cell-dirs \
  --denominator-workers 1 \
  --numerator-workers 18 \
  --numerator-files-per-task 250 \
  --aperture-psf-npz apply/output/stage_b_v6_64670/runs/v6_psf_from_64670/psf_v6_64670.npz \
  --npz-name response_2d_v6_64670_aperture_conditioned.npz \
  --metadata-name response_2d_v6_64670_aperture_conditioned_metadata.json
```

Acceptance gate:

- Stage A metadata records the `_64670` MC binned root;
- aperture-conditioned metadata records the v6 PSF path;
- no path under `stage_a_v3_candidate`, `stage_b_v3_candidate_psfborrow`,
  `stage_a_v4_aperture_conditioned`, or v5 appears as an input.

## Phase 4: Build Stage C From Half-Year `_64670` Observation

Run Stage C on the new observation and recovered-time trees:

```bash
python apply/stages/03_reduce_obs.py \
  --obs-root /mnt/mydisk/WCDA_observation_eval_64670 \
  --time-root /mnt/mydisk/WCDA_observation_eval_64670/recovered_time \
  --cell-selection-csv apply/config/cell_ledger_v6_candidate.csv \
  --output-dir apply/output/stage_c_v6_64670 \
  --run-id v6_stage_c_64670_halfyear \
  --workers 18 \
  --entries-per-chunk 200000 \
  --print-every 25 \
  --overwrite-run-dir
```

For a smoke test, use `--max-files 8` and a separate run id first. Do not promote
the smoke run to `current`.

Acceptance gate:

- `source_files.csv` spans the intended half-year window;
- eval/time entry mismatches are zero unless already explained by recovery logs;
- total live time is roughly six months of the filtered data, not the old
  two-month v3 value;
- ROI coverage diagnostics are present and sane.

## Phase 5: Build Stage D/E With v4 Background And Containment Contract

Use the v4/v3 annulus-normalized ROI-local background family, but point every
input to v6 paths. The nominal Stage D command should be equivalent to the
v3/v4 `annulus_quadratic_annulus_normalized` background:

```bash
python apply/stages/04_background.py \
  --stage-c-dir apply/output/stage_c_v6_64670/runs/v6_stage_c_64670_halfyear \
  --psf-npz apply/output/stage_b_v6_64670/runs/v6_psf_from_64670/psf_v6_64670.npz \
  --cell-selection-csv apply/config/cell_ledger_v6_candidate.csv \
  --output-dir apply/output/stage_d_v6_64670_annnorm \
  --run-id v6_stage_d_64670_annnorm \
  --background-mode crab_roi_local \
  --roi-background-method annulus-quadratic \
  --roi-fiducial-deg 6 \
  --roi-edge-diagnostic-deg 8 \
  --roi-grid-step-deg 0.1 \
  --annulus-default-inner-deg 1.5 \
  --annulus-width-deg 2.0 \
  --annulus-max-inner-deg 4.5 \
  --roi-surface-order 2 \
  --batch-size 500000 \
  --workers 18 \
  --print-every 10 \
  --npz-name background_v6_64670_annnorm.npz \
  --metadata-name background_v6_64670_annnorm_metadata.json \
  --summary-csv-name background_v6_64670_annnorm_summary.csv \
  --summary-md-name background_v6_64670_annnorm_summary.md \
  --overwrite-run-dir
```

Build Stage E with the v4 containment-1 aperture contract. If the existing
`05_signal.py` does not expose this as a first-class option, copy the current v4
clone/patch pattern into a v6 helper and document the exact transformation in
metadata. Do not reuse the old v4 `signal_v4_containment1_annnorm.npz`.

Expected v6 Stage E output:

```text
apply/output/stage_e_v6_64670_containment1_annnorm/runs/v6_stage_e_64670_containment1_annnorm/signal_v6_64670_containment1_annnorm.npz
apply/output/stage_e_v6_64670_containment1_annnorm/runs/v6_stage_e_64670_containment1_annnorm/signal_v6_64670_containment1_annnorm_metadata.json
```

Acceptance gate:

- Stage D/E metadata points to v6 Stage C and v6 PSF;
- Stage E contains the containment-1 contract used by baselinev4;
- total Crab signal scale is plausible for a half-year run. As a rough sanity
  check, it should be much larger than the old two-month Stage E scale and should
  not differ by an order of magnitude from exposure scaling.

## Phase 6: Stage F/G Baselinev4 Fit And SED

Run Stage F with the v6 aperture-conditioned response, v6 containment-1 signal,
v6 Stage C, and v6 drop4 selector:

```bash
python apply/stages/06_fit.py \
  --response-npz apply/output/stage_a_v6_64670_aperture_conditioned/response_2d_v6_64670_aperture_conditioned.npz \
  --response-metadata apply/output/stage_a_v6_64670_aperture_conditioned/response_2d_v6_64670_aperture_conditioned_metadata.json \
  --signal-npz apply/output/stage_e_v6_64670_containment1_annnorm/runs/v6_stage_e_64670_containment1_annnorm/signal_v6_64670_containment1_annnorm.npz \
  --signal-metadata apply/output/stage_e_v6_64670_containment1_annnorm/runs/v6_stage_e_64670_containment1_annnorm/signal_v6_64670_containment1_annnorm_metadata.json \
  --stage-c-dir apply/output/stage_c_v6_64670/runs/v6_stage_c_64670_halfyear \
  --cell-subset-csv apply/config/cell_selector_v6_drop4_psfborrow.csv \
  --output-dir apply/output/stage_f_v6_64670_baselinev4 \
  --run-id v6_stage_f_64670_baselinev4 \
  --npz-name fit_v6_64670_baselinev4.npz \
  --metadata-name fit_v6_64670_baselinev4_metadata.json \
  --summary-csv-name fit_v6_64670_baselinev4_summary.csv \
  --summary-md-name fit_v6_64670_baselinev4_summary.md \
  --report-html apply/report/stage_f_v6_64670_baselinev4_report.html \
  --overwrite-run-dir
```

Run Stage G:

```bash
python apply/stages/07_sed_points.py \
  --response-npz apply/output/stage_a_v6_64670_aperture_conditioned/response_2d_v6_64670_aperture_conditioned.npz \
  --response-metadata apply/output/stage_a_v6_64670_aperture_conditioned/response_2d_v6_64670_aperture_conditioned_metadata.json \
  --signal-npz apply/output/stage_e_v6_64670_containment1_annnorm/runs/v6_stage_e_64670_containment1_annnorm/signal_v6_64670_containment1_annnorm.npz \
  --signal-metadata apply/output/stage_e_v6_64670_containment1_annnorm/runs/v6_stage_e_64670_containment1_annnorm/signal_v6_64670_containment1_annnorm_metadata.json \
  --stage-f-npz apply/output/stage_f_v6_64670_baselinev4/runs/v6_stage_f_64670_baselinev4/fit_v6_64670_baselinev4.npz \
  --stage-f-metadata apply/output/stage_f_v6_64670_baselinev4/runs/v6_stage_f_64670_baselinev4/fit_v6_64670_baselinev4_metadata.json \
  --output-dir apply/output/stage_g_v6_64670_baselinev4 \
  --run-id v6_stage_g_64670_baselinev4 \
  --baseline-name v6_64670_baselinev4_drop4 \
  --required-cell-ids "" \
  --excluded-cell-ids "4,17,39,43" \
  --skip-expected-stage-f-validation \
  --npz-name sed_points_v6_64670_baselinev4.npz \
  --metadata-name sed_points_v6_64670_baselinev4_metadata.json \
  --summary-csv-name sed_points_v6_64670_baselinev4_summary.csv \
  --summary-json-name sed_points_v6_64670_baselinev4_summary.json \
  --summary-md-name sed_points_v6_64670_baselinev4_summary.md \
  --report-html apply/report/stage_g_v6_64670_baselinev4_report.html \
  --overwrite-run-dir
```

Acceptance gate:

- Stage F reference-count preflight passes;
- Stage F/G metadata paths are all v6 or `_64670`;
- Stage G points and fit curve are compared against the same official references
  used by baselinev4;
- the report clearly states that v6 is not directly comparable to v1-v5 unless
  MC and observation were regenerated with the same model.

## Phase 7: Final Report And Commit Hygiene

Create a compact v6 report under:

```text
apply/report/crab_sed_v6_64670_baselinev4_report.html
```

It should include:

- dataset coverage and eval/time file counts;
- model path and model-generation warning;
- MC binned cache provenance;
- Stage A/B response and PSF summaries;
- Stage C live time and ROI coverage;
- Stage D/E background and signal diagnostics;
- Stage F fit summary, covariance/correlation, model-count vs excess, pull grids;
- Stage G SED overlay and ratio plot;
- a direct comparison table against baselinev4 with the caveat that the model
  generation differs.

Before committing:

```bash
git status --short
git diff -- apply/v6roadmap.md apply/devlog.md
```

Only stage intended v6 files. Do not stage unrelated dirty files already present
in the ETO worktree.

## Stop-And-Ask Conditions

The server agent must stop and ask one concrete question if any of these happen:

1. `/mnt/mydisk/WCDA_observation_eval_64670/recovered_time` is missing or does
   not match eval ROOT stems.
2. The intended half-year date range is not discoverable from files or metadata.
3. The `_64670` model path is missing or its config is incompatible with
   `simulation_all_bin.py` / `apply_observation_energy.py`.
4. Rebuilding MC with `_64670` is impossible because raw MC, denominator hist, or
   binned-cache disk space is missing.
5. The v4 drop4 included cells fail `_64670` MC prefit quality in a way that
   would make Stage A/B invalid.
6. Stage E containment-1 behavior is not reproducible from existing scripts.
7. Any step would require overwriting v1-v5 outputs or deleting data.
