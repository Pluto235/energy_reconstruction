# Codex/Claude project handoff

This file is the operational handoff for using Claude or Codex on the WCDA
energy-reconstruction `apply` workflow.

## Project identity

- Project: LHAASO-WCDA Crab SED analysis for validating ML single-event energy
  reconstruction.
- Local working copy: `/Users/luoji/Documents/projects/energy/apply`
- Main working server: `ETO:/home/server/projects/energy_reconstruction/apply`
- Git repository: `git@github.com:Pluto235/energy_reconstruction.git`
- Local Mac role: edit, inspect results, run small smoke checks only.
- ETO role: authoritative working tree for heavy reconstruction, Slurm jobs,
  commits, and pushes.
- IHEP role: original observation data storage and source-side filtering/time
  recovery support.

The local `apply/` directory is not necessarily a Git checkout. When a change is
made locally, sync the intended files to ETO and commit/push from
`/home/server/projects/energy_reconstruction` if the local tree has no `.git`.

## Server access

Use SSH aliases from the local Mac:

```bash
ssh ETO
ssh ihep
```

Known alias details:

- `ETO`: user `server`, host `10.148.240.220`, main project root
  `/home/server/projects/energy_reconstruction`.
- `ihep`: user `liushijie`, host `lxlogin.ihep.ac.cn`.

For non-interactive commands, rsync, and scripted SSH, disable configured remote
interactive commands:

```bash
ssh -o RemoteCommand=none -o RequestTTY=no ETO 'cd /home/server/projects/energy_reconstruction && pwd'
ssh -o RemoteCommand=none -o RequestTTY=no ihep 'pwd'
```

Do not put passwords, private keys, or one-time codes in project files.

## Sync conventions

Push local `apply/` changes to ETO before heavy work:

```bash
rsync -av --delete -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /Users/luoji/Documents/projects/energy/apply/ \
  ETO:/home/server/projects/energy_reconstruction/apply/
```

Pull ETO results back to the Mac, preserving the local devlog:

```bash
rsync -av --delete --exclude devlog.md -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  ETO:/home/server/projects/energy_reconstruction/apply/ \
  /Users/luoji/Documents/projects/energy/apply/
```

For a small documentation/code edit, prefer syncing only the intended files to
avoid touching generated outputs:

```bash
rsync -av -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  /Users/luoji/Documents/projects/energy/apply/codex-claude.md \
  /Users/luoji/Documents/projects/energy/apply/devlog.md \
  ETO:/home/server/projects/energy_reconstruction/apply/
```

After any meaningful code, documentation, workflow, or configuration change:

1. Prepend one entry to `apply/devlog.md`, newest first.
2. Commit only intended files from the Git checkout.
3. Run `git push`.

## Runtime on ETO

Start heavy work on ETO from the repository root:

```bash
cd /home/server/projects/energy_reconstruction
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate py310
export PYTHONPATH=/home/server/projects/energy_reconstruction:${PYTHONPATH}
```

Run expensive MC inference, observation inference, Stage C-G scans, report
generation over large outputs, and large plotting through Slurm. Local Mac runs
should be limited to lightweight inspection or tiny smoke tests.

## Data locations

ETO observation/eval data:

- Filtered observation ROOT: `/mnt/mydisk/WCDA_observation/<MMDD>/Esg*.root`
- Legacy two-month eval ROOT: `/mnt/mydisk/WCDA_observation_eval/<MMDD>/Esg*.root`
- Legacy recovered-time friends:
  `/mnt/mydisk/WCDA_observation_eval/recovered_time/<MMDD>/*.time.root`
- Half-year `_64670` eval ROOT:
  `/mnt/mydisk/WCDA_observation_eval_64670/<MMDD>/Esg*.root`
- Half-year `_64670` recovered-time friends:
  `/mnt/mydisk/WCDA_observation_eval_64670/recovered_time/<MMDD>/*.time.root`

ETO MC/model data:

- Raw MC: `/mnt/mydisk/WCDA_simulation`
- Primary denominator histogram:
  `/mnt/mydisk/WCDA_simulation_primary_hist/primary_denominator_stage_a.npz`
- Legacy response cache: `/mnt/mydisk/WCDA_simulation_binned_response_v1`
- v3 candidate cache: `/mnt/mydisk/WCDA_simulation_binned_response_v3_candidate`
- v4 split56 cache: `/mnt/mydisk/WCDA_simulation_binned_response_v4_split56_ridge`
- v6 `_64670` cache:
  `/mnt/mydisk/WCDA_simulation_binned_response_v6_64670`
- v6 `_64670` split56 cache:
  `/mnt/mydisk/WCDA_simulation_binned_response_v6_64670_split56_candidate`
- Legacy model: `/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666`
- `_64670` no-core-cut model:
  `/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670`

IHEP source data:

- Full observation ROOT:
  `/eos/lhaaso/ai/wcda/raw/2022/<MMDD>/EsgYYYYMMDD_HH.root`
- Reduced time source:
  `/eos/lhaaso/raw/wcda/2022/<MMDD>/*.reduced.root`
- Existing IHEP eval sample: `/home/lhaaso/liushijie/WCDA_observation_eval`
- Existing IHEP recovered-time sample:
  `/home/lhaaso/liushijie/WCDA_observation_eval/recovered_time`
- IHEP filtering workspace: `/home/lhaaso/liushijie/WCDA_reconstruction`
- Large scratch area: `/scratchfs/lhaaso`

Project outputs:

- Stage outputs: `apply/output/stage_*`
- Reports: `apply/report/*.html` and `apply/report/*.md`
- Configs/selectors: `apply/config/*.csv` and `apply/config/*.json`

## Scientific and analysis contract

The analysis reconstructs the Crab Nebula SED using a two-dimensional
observation grid:

- Shower-size axis: `Nhit` from the ROOT `nv` branch.
- ML energy axis: `log10(E_pred / GeV)`.
- Source coordinates: Crab `RA=83.63 deg`, `Dec=22.01 deg`.
- Site constants used in fits/backgrounds: LHAASO latitude `29.45 deg`,
  longitude `100.14 deg`.
- Default event cuts: `pincness < 1.1`, `fitstat == 0`, `theta < 50 deg`,
  `dcedge > 20 m`.
- Stage C reads eval ROOT tree `t_eventout` and time friend tree
  `t_recovered_time`.

Important data-contract rule: recovered-time friend trees are aligned to eval
ROOT files by same file stem and same entry index. Do not pair a recovered-time
friend made from a different entry sequence with an eval ROOT. If the inference
filtering changes, time recovery must be regenerated or Stage C must be changed
to join by event identifiers.

## Main workflow

End-to-end source flow:

```text
IHEP full Esg ROOT
  -> IHEP branch/pincness reduction
  -> transfer filtered observation ROOT to ETO
  -> ETO ML energy inference
  -> recover time and RA/Dec against the eval ROOT entry sequence
  -> transfer recovered_time friend ROOT to ETO
  -> ETO Stage A-G Crab SED analysis
  -> reports and selected outputs back to local Mac
```

Stage A-G:

- Stage A, `apply/stages/01_build_response.py`: build MC response
  `eta_b(E_true, theta)` and effective area for configured `(Nhit, predE)`
  cells. Can build aperture-conditioned response using a Stage B PSF NPZ.
- Stage B, `apply/stages/02_build_psf.py` and
  `apply/stages/02e_build_psf_v5_compare.py`: build Crab-declination per-cell
  PSF and optimal aperture/containment tables.
- Stage C, `apply/stages/03_reduce_obs.py`: reduce eval ROOT plus recovered-time
  friend tree to configured-cell parquet, including ROI coverage diagnostics.
- Stage D, `apply/stages/04_background.py`: build background. Current baseline
  uses Crab ROI-local annulus/quadratic background with annulus normalization.
- Stage E, `apply/stages/05_signal.py`: count on-source events, combine with
  Stage D background, and output excess/error/significance per cell.
- Stage F, `apply/stages/06_fit.py`: forward-folding chi-square fit using PL
  and LogPar spectral models with `iminuit`.
- Stage G, `apply/stages/07_sed_points.py`: diagnostic SED points, fit curves,
  external-reference overlays, and summary reports.

## Core code map

- `apply/simulation_all_bin.py`: run all-event MC inference, apply optional cuts,
  and write binned MC ROOT/cache directories by `Nhit` and predicted energy.
- `apply/apply_observation_energy.py`: run observation-data model inference,
  mirror ROOT outputs, add predicted-energy branches, and drop hit-level
  branches. Defaults to legacy output unless `--output-root` is overridden.
- `apply/stages/00_prepare_v3_inputs.py`: build v3 candidate ledgers/selectors
  and optional candidate MC cache.
- `apply/stages/00_prepare_v5_predbin_ablation.py`: build v5 PredE-binning
  ablation ledgers/selectors and caches for `gap025`, `gap1`, and `split56`.
- `apply/stages/00a_validate_v6_mc_cache.py`: validate `_64670` MC cache
  provenance before downstream v6 work.
- `apply/stages/00c_prepare_v6_split56_inputs.py`: build v6 `_64670` split56
  candidate cache and prefit selector.
- `apply/stages/00b_enforce_v6_drop4_selector.py` and
  `apply/stages/00d_enforce_v6_split56_drop4_selector.py`: enforce the v4
  drop4/baselinev4 selector contract on v6 selectors.
- `apply/report/build_*.py`: report builders; use them after Stage outputs are
  produced rather than editing generated HTML by hand.
- `apply/config/cell_ledger_*.csv`: cell ledgers/candidate pools.
- `apply/config/cell_selector_*.csv`: fit selectors and included/excluded cells.
- `apply/devlog.md`: project modification log, newest first.

Primary wrapper scripts:

- `apply/run_v4_aperture_conditioned_response_pipeline.sh`: v4 primary
  aperture-conditioned Stage F/G/report flow.
- `apply/run_v4_split56_ridge_pipeline.sh`: resumable split56 pipeline from MC
  cache prep through Stage G.
- `apply/run_v5_psf_comparison_pipeline.sh`: PSF method comparison pipeline.
- `apply/run_v4_r68_aperture_pipeline.sh`: empirical r68 aperture control flow.

## Current baselines and important variants

The v4 baseline contract is the main comparison contract preserved by later
v6 work:

- Selector: `apply/config/cell_selector_v4_drop4_psfborrow.csv`
- Excluded drop4 control cells: `4,17,39,43`
- Included v4 fit cells:
  `1,2,3,14,15,16,26,27,28,29,30,40,41,42,52,53,54,55,65,66,67,68,69,81,82,83`
- Primary v4 aperture-conditioned response:
  `apply/output/stage_a_v4_aperture_conditioned/response_2d_v4_aperture_conditioned.npz`
- Primary v4 containment-1 signal:
  `apply/output/stage_e_v4_containment1_annnorm/runs/v4_stage_e_annnorm_containment1_from_psfborrow/signal_v4_containment1_annnorm.npz`
- Primary v4 Stage F/G:
  `apply/output/stage_f_v4_aperture_conditioned` and
  `apply/output/stage_g_v4_aperture_conditioned`

The half-year v6 `_64670` branch must not reuse v1-v5 MC/model-dependent
artifacts. It should use:

- Observation input: `/mnt/mydisk/WCDA_observation_eval_64670`
- Recovered time: `/mnt/mydisk/WCDA_observation_eval_64670/recovered_time`
- Model:
  `/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670`
- MC cache: `/mnt/mydisk/WCDA_simulation_binned_response_v6_64670`
- Split56 candidate cache:
  `/mnt/mydisk/WCDA_simulation_binned_response_v6_64670_split56_candidate`
- Split56 selector:
  `apply/config/cell_selector_v6_split56_drop4_psfborrow.csv`
- Latest split56 report:
  `apply/report/crab_sed_v6_64670_split56_baselinev4_report.html`

## Operational guardrails

- Do not run large computation locally on the Mac.
- Do not mix model generations. `_64670` observation inputs require `_64670`
  MC binned caches and rebuilt Stage A/B response/PSF.
- Do not write `_64670` eval ROOT or recovered-time friends into legacy
  `/mnt/mydisk/WCDA_observation_eval`.
- Do not run `rsync --delete` against `/mnt/mydisk/WCDA_observation_eval*`
  unless the source side is known to contain the full intended tree.
- Do not delete `/mnt/mydisk/WCDA_observation/<MMDD>` scratch inputs until eval
  ROOT, recovered-time friends, summaries, and Stage C smoke/full reads have all
  been verified.
- Stop and ask one concrete question if the target path, overwrite direction,
  Slurm command, data dependency, or compute size is unclear.

Useful references already in the repo:

- `apply/report/half_year_crab_observation_workflow.md`
- `apply/v6roadmap.md`
- `apply/report/roadmap.md`
- `apply/report/roadmap_v2.md`
- `apply/report/roadmap_v3.md`
