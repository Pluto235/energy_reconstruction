#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_stage_module(name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage04 = load_stage_module("stage04_background", "apply/stages/04_background.py")
stage05 = load_stage_module("stage05_signal", "apply/stages/05_signal.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v3 annulus-normalized background branch from an existing full "
            "Stage D counts-map product, without rescanning Stage C events."
        )
    )
    parser.add_argument(
        "--source-stage-d-npz",
        type=str,
        default=(
            "apply/output/stage_d_v3_candidate_psfborrow/runs/"
            "v3_stage_d_psfborrow_slurm_42029/background_v3_candidate_psfborrow.npz"
        ),
    )
    parser.add_argument(
        "--source-stage-d-metadata",
        type=str,
        default=(
            "apply/output/stage_d_v3_candidate_psfborrow/runs/"
            "v3_stage_d_psfborrow_slurm_42029/background_v3_candidate_psfborrow_metadata.json"
        ),
    )
    parser.add_argument(
        "--source-stage-e-npz",
        type=str,
        default=(
            "apply/output/stage_e_v3_candidate_psfborrow/runs/"
            "v3_stage_e_psfborrow_slurm_42029/signal_v3_candidate_psfborrow.npz"
        ),
    )
    parser.add_argument(
        "--source-stage-e-metadata",
        type=str,
        default=(
            "apply/output/stage_e_v3_candidate_psfborrow/runs/"
            "v3_stage_e_psfborrow_slurm_42029/signal_v3_candidate_psfborrow_metadata.json"
        ),
    )
    parser.add_argument("--cell-selection-csv", type=str, default="apply/config/cell_ledger_v3_candidate.csv")
    parser.add_argument("--baseline-selector-csv", type=str, default="apply/config/cell_selector_v3_baseline_psfborrow.csv")
    parser.add_argument(
        "--psf-npz",
        type=str,
        default="",
        help=(
            "Optional Stage B PSF NPZ supplying r_opt/sigma/containment for the derived Stage D product. "
            "When omitted, these arrays are reused from --source-stage-d-npz."
        ),
    )
    parser.add_argument(
        "--psf-metadata",
        type=str,
        default="",
        help="Optional Stage B PSF metadata paired with --psf-npz, recorded for provenance.",
    )
    parser.add_argument("--stage-d-output-dir", type=str, default="apply/output/stage_d_v3_candidate_annnorm")
    parser.add_argument("--stage-d-run-id", type=str, default="v3_stage_d_annnorm_from_psfborrow")
    parser.add_argument("--stage-e-output-dir", type=str, default="apply/output/stage_e_v3_candidate_annnorm")
    parser.add_argument("--stage-e-run-id", type=str, default="v3_stage_e_annnorm_from_psfborrow")
    parser.add_argument("--stage-e-report-html", type=str, default="apply/report/stage_e_v3_candidate_annnorm_report.html")
    parser.add_argument("--overwrite-run-dir", action="store_true", default=False)
    parser.add_argument("--no-promote-current", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)

    parser.add_argument("--roi-fiducial-deg", type=float, default=None)
    parser.add_argument("--annulus-default-inner-deg", type=float, default=None)
    parser.add_argument("--annulus-width-deg", type=float, default=None)
    parser.add_argument("--annulus-source-mask-min-deg", type=float, default=None)
    parser.add_argument("--annulus-source-mask-r-opt-factor", type=float, default=None)
    parser.add_argument("--annulus-source-mask-margin-deg", type=float, default=None)
    parser.add_argument("--annulus-max-inner-deg", type=float, default=None)
    parser.add_argument("--roi-surface-order", type=int, choices=[1, 2], default=None)
    parser.add_argument("--surface-condition-max", type=float, default=None)
    parser.add_argument("--surface-min-training-pixels", type=int, default=None)

    parser.add_argument("--quality-min-total-sigma", type=float, default=0.0)
    parser.add_argument("--quality-max-total-sigma", type=float, default=300.0)
    parser.add_argument("--stage-d-npz-name", type=str, default="background_v3_candidate_annnorm.npz")
    parser.add_argument("--stage-d-metadata-name", type=str, default="background_v3_candidate_annnorm_metadata.json")
    parser.add_argument("--stage-d-summary-csv-name", type=str, default="background_v3_candidate_annnorm_summary.csv")
    parser.add_argument("--stage-d-summary-md-name", type=str, default="background_v3_candidate_annnorm_summary.md")
    parser.add_argument("--stage-e-npz-name", type=str, default="signal_v3_candidate_annnorm.npz")
    parser.add_argument("--stage-e-metadata-name", type=str, default="signal_v3_candidate_annnorm_metadata.json")
    parser.add_argument("--stage-e-summary-csv-name", type=str, default="signal_v3_candidate_annnorm_summary.csv")
    parser.add_argument("--stage-e-summary-md-name", type=str, default="signal_v3_candidate_annnorm_summary.md")
    return parser.parse_args()


def path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_json(path_: Path) -> Dict[str, object]:
    with path_.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metadata_float(metadata: Dict[str, object], *keys: str, default: float) -> float:
    current: object = metadata
    for key in keys:
        if not isinstance(current, dict):
            return float(default)
        current = current.get(key)
    number = finite_float(current)
    return float(default if number is None else number)


def metadata_int(metadata: Dict[str, object], *keys: str, default: int) -> int:
    number = finite_float(metadata_float(metadata, *keys, default=float(default)))
    return int(default if number is None else round(number))


def read_selector_ids(selector_csv: Path) -> List[int]:
    ids: List[int] = []
    if not selector_csv.exists():
        return ids
    with selector_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            include_text = str(row.get("include", "1")).strip().lower()
            include = include_text in {"", "1", "true", "yes", "y", "include"}
            if include and row.get("cell_id"):
                ids.append(int(row["cell_id"]))
    return ids


def order_for_cell_ids(source_ids: np.ndarray, cells: Sequence[object]) -> np.ndarray:
    id_to_index = {int(cell_id): idx for idx, cell_id in enumerate(np.asarray(source_ids, dtype=np.int64))}
    missing = [int(cell.cell_id) for cell in cells if int(cell.cell_id) not in id_to_index]
    if missing:
        raise ValueError(f"Source product is missing cells: {missing}")
    return np.asarray([id_to_index[int(cell.cell_id)] for cell in cells], dtype=np.int64)


def load_npz_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def write_json(path_: Path, payload: Dict[str, object]) -> None:
    stage04.write_json(path_, payload)


def json_ready(value):
    return stage04.json_ready(value)


def scalar_array(payload: Dict[str, np.ndarray], key: str, default: float) -> np.ndarray:
    value = payload.get(key)
    if value is None:
        return np.asarray([default], dtype=np.float32)
    return np.asarray(value)


def source_cell_rows(metadata: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    rows = metadata.get("cells")
    out: Dict[int, Dict[str, object]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict) or row.get("cell_id") is None:
            continue
        try:
            out[int(row["cell_id"])] = row
        except (TypeError, ValueError):
            continue
    return out


def build_stage_d(
    args: argparse.Namespace,
    *,
    source_d_npz: Path,
    source_d_metadata_path: Path,
    source_d_metadata: Dict[str, object],
    psf_npz: Optional[Path],
    psf_metadata_path: Optional[Path],
    cells: Sequence[object],
    fit_ids: Sequence[int],
) -> Tuple[Path, Path, Path, Dict[str, object]]:
    start = time.perf_counter()
    source = load_npz_arrays(source_d_npz)
    required = {
        "cell_id",
        "counts_map",
        "x_edges_deg",
        "x_centers_deg",
        "rho_hist_edges_deg",
        "rho_hist_total",
        "r_opt_deg",
        "sigma_deg",
        "containment_r_opt",
    }
    missing = required - set(source)
    if missing:
        raise ValueError(f"{source_d_npz} is missing required arrays: {sorted(missing)}")

    order = order_for_cell_ids(source["cell_id"], cells)
    counts_map = np.asarray(source["counts_map"][order], dtype=np.int64)
    psf_payload = load_npz_arrays(psf_npz) if psf_npz is not None else source
    psf_order = order_for_cell_ids(psf_payload["cell_id"], cells)
    r_opt_deg = np.asarray(psf_payload["r_opt_deg"][psf_order], dtype=np.float64)
    sigma_deg = np.asarray(psf_payload["sigma_deg"][psf_order], dtype=np.float64)
    containment_r_opt = np.asarray(psf_payload["containment_r_opt"][psf_order], dtype=np.float64)
    xy_edges = np.asarray(source["x_edges_deg"], dtype=np.float64)
    xy_centers = np.asarray(source["x_centers_deg"], dtype=np.float64)

    roi_fiducial = float(
        args.roi_fiducial_deg
        if args.roi_fiducial_deg is not None
        else metadata_float(source_d_metadata, "roi", "fiducial_radius_deg", default=6.0)
    )
    background_model = source_d_metadata.get("background_model") if isinstance(source_d_metadata.get("background_model"), dict) else {}
    annulus_default_inner = float(
        args.annulus_default_inner_deg
        if args.annulus_default_inner_deg is not None
        else metadata_float(source_d_metadata, "background_model", "annulus_default_inner_deg", default=1.5)
    )
    annulus_width = float(
        args.annulus_width_deg
        if args.annulus_width_deg is not None
        else metadata_float(source_d_metadata, "background_model", "annulus_width_deg", default=2.0)
    )
    annulus_source_mask_min = float(
        args.annulus_source_mask_min_deg
        if args.annulus_source_mask_min_deg is not None
        else metadata_float(source_d_metadata, "background_model", "annulus_source_mask_min_deg", default=1.5)
    )
    annulus_source_mask_factor = float(
        args.annulus_source_mask_r_opt_factor
        if args.annulus_source_mask_r_opt_factor is not None
        else metadata_float(source_d_metadata, "background_model", "annulus_source_mask_r_opt_factor", default=2.0)
    )
    annulus_source_mask_margin = float(
        args.annulus_source_mask_margin_deg
        if args.annulus_source_mask_margin_deg is not None
        else metadata_float(source_d_metadata, "background_model", "annulus_source_mask_margin_deg", default=0.2)
    )
    annulus_max_inner = float(
        args.annulus_max_inner_deg
        if args.annulus_max_inner_deg is not None
        else metadata_float(source_d_metadata, "background_model", "annulus_max_inner_deg", default=4.5)
    )
    surface_order = int(
        args.roi_surface_order
        if args.roi_surface_order is not None
        else metadata_int(source_d_metadata, "background_model", "surface_order", default=2)
    )
    condition_max = float(
        args.surface_condition_max
        if args.surface_condition_max is not None
        else metadata_float(source_d_metadata, "background_model", "surface_condition_max", default=1.0e8)
    )
    min_training_pixels = int(
        args.surface_min_training_pixels
        if args.surface_min_training_pixels is not None
        else metadata_int(source_d_metadata, "background_model", "surface_min_training_pixels", default=80)
    )

    b_on, background_map, training_mask, on_masks, source_masks, annulus_diagnostics = (
        stage04.estimate_roi_annulus_surface_background(
            counts_map,
            xy_centers,
            r_opt_deg,
            roi_fiducial_deg=roi_fiducial,
            annulus_default_inner_deg=annulus_default_inner,
            annulus_width_deg=annulus_width,
            annulus_source_mask_min_deg=annulus_source_mask_min,
            annulus_source_mask_r_opt_factor=annulus_source_mask_factor,
            annulus_source_mask_margin_deg=annulus_source_mask_margin,
            annulus_max_inner_deg=annulus_max_inner,
            surface_order=surface_order,
            condition_max=condition_max,
            min_training_pixels=min_training_pixels,
            annulus_normalize_surface=True,
        )
    )

    if "fiducial_mask" in source:
        fiducial_mask = np.asarray(source["fiducial_mask"], dtype=bool)
    else:
        x_grid, y_grid = np.meshgrid(xy_centers, xy_centers)
        fiducial_mask = np.hypot(x_grid, y_grid) < roi_fiducial
    if "edge_safe_mask" in source:
        edge_safe_mask = np.asarray(source["edge_safe_mask"], dtype=bool)
    else:
        edge_safe_mask = fiducial_mask.copy()
    excess_map = counts_map.astype(np.float32) - background_map.astype(np.float32)
    excess_map[:, ~fiducial_mask] = np.nan
    known_b_sigma_grid = stage04.known_background_sigma_map(counts_map, background_map, fiducial_mask)
    on_pixels = np.asarray([int(np.count_nonzero(mask)) for mask in on_masks], dtype=np.int64)
    off_counts = np.asarray(annulus_diagnostics["annulus_off_counts"], dtype=np.float64)
    off_pixels = np.asarray(annulus_diagnostics["annulus_off_pixels"], dtype=np.int64)

    output_root = path(args.stage_d_output_dir)
    run_id = stage04.sanitize_run_id(args.stage_d_run_id)
    run_dir = stage04.prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))
    npz_path = run_dir / args.stage_d_npz_name
    metadata_path = run_dir / args.stage_d_metadata_name
    summary_csv_path = run_dir / args.stage_d_summary_csv_name
    summary_md_path = run_dir / args.stage_d_summary_md_name
    source_masks_csv_path = run_dir / "source_masks.csv"

    source_rows = source_cell_rows(source_d_metadata)
    rows: List[Dict[str, object]] = []
    warning_ids: List[int] = []
    for cell in cells:
        idx = int(cell.index)
        cid = int(cell.cell_id)
        source_row = source_rows.get(cid, {})
        fiducial_events = int(np.asarray(source.get("cell_fiducial_events", np.zeros(len(cells), dtype=np.int64)))[order][idx])
        source_masked_events = int(counts_map[idx][source_masks[idx] & fiducial_mask].sum())
        warnings: List[str] = []
        if off_pixels[idx] <= 0:
            warnings.append("no_training_pixels")
        if on_pixels[idx] <= 0:
            warnings.append("no_on_pixels")
        if b_on[idx] <= 0.0 or not np.isfinite(b_on[idx]):
            warnings.append("non_positive_B_on")
        if fiducial_events <= 0:
            warnings.append("no_fiducial_events")
        if bool(annulus_diagnostics["core_extrapolation_warning"][idx]):
            warnings.append("core_extrapolation_warning")
        if not bool(annulus_diagnostics["surface_fit_success"][idx]):
            warnings.append("surface_fit_not_successful")
        if warnings:
            warning_ids.append(cid)

        selected_events = int(np.asarray(source.get("cell_total_events", np.zeros(len(cells), dtype=np.int64)))[order][idx])
        grid_events = int(np.asarray(source.get("cell_map_events", np.zeros(len(cells), dtype=np.int64)))[order][idx])
        out_of_grid_events = int(
            np.asarray(source.get("cell_out_of_map_events", np.zeros(len(cells), dtype=np.int64)))[order][idx]
        )
        edge_events = int(
            np.asarray(source.get("cell_edge_diagnostic_events", np.zeros(len(cells), dtype=np.int64)))[order][idx]
        )
        live_days = finite_float(source_row.get("live_time_days"))
        mean_rate = finite_float(source_row.get("mean_rate_hz"))
        median_rate = finite_float(source_row.get("median_rate_hz"))
        rows.append(
            {
                "cell_index": idx,
                "cell_id": cid,
                "nhit_bin": cell.nhit_bin,
                "predE_bin": cell.predE_bin,
                "selected_events": selected_events,
                "grid_events": grid_events,
                "out_of_grid_events": out_of_grid_events,
                "source_masked_events": source_masked_events,
                "source_masked_fraction": float(source_masked_events) / float(fiducial_events) if fiducial_events > 0 else 0.0,
                "fiducial_events": fiducial_events,
                "edge_diagnostic_events": edge_events,
                "live_time_days": live_days if live_days is not None else 0.0,
                "median_rate_hz": median_rate if median_rate is not None else 0.0,
                "mean_rate_hz": mean_rate if mean_rate is not None else 0.0,
                "r_opt_deg": float(r_opt_deg[idx]),
                "sigma_deg": float(sigma_deg[idx]),
                "containment_r_opt": float(containment_r_opt[idx]),
                "B_on": float(b_on[idx]),
                "N_off": "",
                "alpha": "",
                "max_p_on": 0.0,
                "on_pixels": int(on_pixels[idx]),
                "off_pixels": int(off_pixels[idx]),
                "off_counts": float(off_counts[idx]),
                "r_opt_large_warning": bool(r_opt_deg[idx] > 10.0),
                "r_opt_extreme_warning": bool(r_opt_deg[idx] > 20.0),
                "background_form": "direct_expectation",
                "background_mode": "crab_roi_local",
                "background_method": "annulus_quadratic_annulus_normalized",
                "annulus_inner_deg": float(annulus_diagnostics["annulus_inner_deg"][idx]),
                "annulus_outer_deg": float(annulus_diagnostics["annulus_outer_deg"][idx]),
                "surface_fit_chi2": float(annulus_diagnostics["fit_chi2"][idx]),
                "surface_fit_ndof": int(annulus_diagnostics["fit_ndof"][idx]),
                "surface_condition_number": float(annulus_diagnostics["fit_condition_number"][idx]),
                "annulus_residual_rms": float(annulus_diagnostics["annulus_residual_rms"][idx]),
                "annulus_surface_scale": float(annulus_diagnostics["annulus_surface_scale"][idx]),
                "annulus_count_residual_raw": float(annulus_diagnostics["annulus_count_residual_raw"][idx]),
                "annulus_count_residual_final": float(annulus_diagnostics["annulus_count_residual_final"][idx]),
                "warnings": warnings,
            }
        )

    payload = {key: np.asarray(value) for key, value in source.items()}
    payload.update(
        {
            "cell_id": np.asarray([int(cell.cell_id) for cell in cells], dtype=np.int32),
            "nhit_bin": np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
            "predE_bin": np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
            "fiducial_mask": fiducial_mask.astype(bool),
            "edge_safe_mask": edge_safe_mask.astype(bool),
            "on_mask": on_masks.astype(bool),
            "source_mask": source_masks.astype(bool),
            "training_mask": training_mask.astype(bool),
            "counts_map": counts_map.astype(np.int64),
            "background_map": background_map.astype(np.float32),
            "excess_map": excess_map.astype(np.float32),
            "known_b_sigma_map": known_b_sigma_grid.astype(np.float32),
            "off_counts": off_counts.astype(np.float64),
            "off_pixels": off_pixels.astype(np.int64),
            "on_pixels": on_pixels.astype(np.int64),
            "alpha": np.full(len(cells), np.nan, dtype=np.float32),
            "N_off": np.full(len(cells), np.nan, dtype=np.float64),
            "source_ra_deg": scalar_array(source, "source_ra_deg", metadata_float(source_d_metadata, "roi", "source_ra_deg", default=83.63)),
            "source_dec_deg": scalar_array(source, "source_dec_deg", metadata_float(source_d_metadata, "roi", "source_dec_deg", default=22.01)),
            "source_name": np.asarray(["Crab"], dtype="U32"),
            "r_opt_deg": r_opt_deg.astype(np.float32),
            "sigma_deg": sigma_deg.astype(np.float32),
            "containment_r_opt": containment_r_opt.astype(np.float32),
            "B_on": b_on.astype(np.float64),
        }
    )
    payload.update({key: np.asarray(value) for key, value in annulus_diagnostics.items()})
    np.savez_compressed(npz_path, **payload)

    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        plot_outputs = {
            "roi_coverage_profile_png": str(run_dir / "roi_coverage_profile.png"),
            "roi_counts_grid_png": str(run_dir / "roi_counts_grid.png"),
            "roi_background_grid_png": str(run_dir / "roi_background_grid.png"),
            "roi_excess_grid_png": str(run_dir / "roi_excess_grid.png"),
            "roi_known_b_sigma_grid_png": str(run_dir / "roi_known_b_sigma_grid.png"),
            "roi_mask_summary_png": str(run_dir / "roi_mask_summary.png"),
            "annulus_training_mask_grid_png": str(run_dir / "annulus_training_mask_grid.png"),
            "annulus_residual_grid_png": str(run_dir / "annulus_residual_grid.png"),
            "core_background_grid_png": str(run_dir / "core_background_grid.png"),
            "background_prediction_png": str(run_dir / "background_prediction_grid.png"),
        }
        stage04.plot_roi_coverage_profile(
            np.asarray(source["rho_hist_total"], dtype=np.int64),
            np.asarray(source["rho_hist_edges_deg"], dtype=np.float64),
            Path(plot_outputs["roi_coverage_profile_png"]),
            roi_fiducial_deg=roi_fiducial,
            roi_edge_diagnostic_deg=float(abs(xy_edges[0])),
        )
        stage04.plot_roi_counts_grid(
            counts_map,
            cells,
            xy_edges,
            Path(plot_outputs["roi_counts_grid_png"]),
            title="Stage D ROI-local counts",
            roi_fiducial_deg=roi_fiducial,
        )
        stage04.plot_roi_counts_grid(
            background_map,
            cells,
            xy_edges,
            Path(plot_outputs["roi_background_grid_png"]),
            title="Stage D annulus-normalized quadratic surface background",
            roi_fiducial_deg=roi_fiducial,
        )
        stage04.plot_roi_signed_grid(
            excess_map,
            cells,
            xy_edges,
            Path(plot_outputs["roi_excess_grid_png"]),
            title="Stage D counts minus annulus-normalized background",
            colorbar_label="counts - background",
            roi_fiducial_deg=roi_fiducial,
            r_opt_deg=r_opt_deg,
        )
        stage04.plot_roi_signed_grid(
            known_b_sigma_grid,
            cells,
            xy_edges,
            Path(plot_outputs["roi_known_b_sigma_grid_png"]),
            title="Stage D known-background residual after annulus normalization",
            colorbar_label="known-B sigma",
            roi_fiducial_deg=roi_fiducial,
            r_opt_deg=r_opt_deg,
        )
        stage04.plot_roi_mask_summary(fiducial_mask, training_mask, on_masks, xy_edges, Path(plot_outputs["roi_mask_summary_png"]))
        stage04.plot_roi_annulus_mask_grid(
            training_mask,
            source_masks,
            on_masks,
            xy_edges,
            cells,
            Path(plot_outputs["annulus_training_mask_grid_png"]),
        )
        stage04.plot_roi_signed_grid(
            annulus_diagnostics["annulus_residual_map"],
            cells,
            xy_edges,
            Path(plot_outputs["annulus_residual_grid_png"]),
            title="Stage D annulus fit residuals after total normalization",
            colorbar_label="annulus residual sigma",
            roi_fiducial_deg=roi_fiducial,
            r_opt_deg=r_opt_deg,
        )
        stage04.plot_roi_counts_grid(
            annulus_diagnostics["core_background_map"],
            cells,
            xy_edges,
            Path(plot_outputs["core_background_grid_png"]),
            title="Stage D core extrapolated annulus-normalized background",
            roi_fiducial_deg=roi_fiducial,
        )
        stage04.plot_background_grid(rows, cells, Path(plot_outputs["background_prediction_png"]))

    fit_set = set(int(v) for v in fit_ids)
    fit_warning_ids = sorted(set(warning_ids).intersection(fit_set))
    severe_warnings = [
        f"cell {row['cell_id']}: {','.join(row['warnings'])}"
        for row in rows
        if row.get("warnings")
    ]
    quality_status = "ok" if not fit_warning_ids else "warning"
    quality_reason = (
        "annulus-normalized background built from the full psfborrow counts map; no active fit cells have surface warnings"
        if not fit_warning_ids
        else f"active fit cells have surface warnings: {fit_warning_ids}"
    )
    source_roi = source_d_metadata.get("roi") if isinstance(source_d_metadata.get("roi"), dict) else {}
    metadata: Dict[str, object] = {
        "description": "Derived Stage D ROI-local background with annulus-total-normalized 2D surface.",
        "run_id": run_id,
        "derived_from": {
            "source_stage_d_npz": str(source_d_npz),
            "source_stage_d_metadata": str(source_d_metadata_path),
            "source_stage_d_run_id": source_d_metadata.get("run_id"),
            "psf_npz": str(psf_npz) if psf_npz is not None else str(source_d_npz),
            "psf_metadata": str(psf_metadata_path) if psf_metadata_path is not None else None,
            "note": (
                "Reuses the existing full Stage D counts_map and recomputes the annulus quadratic background. "
                "If psf_npz is supplied, on-region masks and B_on use that PSF aperture."
            ),
        },
        "inputs": {
            "stage_c_dir": source_d_metadata.get("inputs", {}).get("stage_c_dir") if isinstance(source_d_metadata.get("inputs"), dict) else None,
            "obs_events_dir": source_d_metadata.get("inputs", {}).get("obs_events_dir") if isinstance(source_d_metadata.get("inputs"), dict) else None,
            "stage_c_metadata_json": source_d_metadata.get("inputs", {}).get("stage_c_metadata_json") if isinstance(source_d_metadata.get("inputs"), dict) else None,
            "source_files_csv": source_d_metadata.get("inputs", {}).get("source_files_csv") if isinstance(source_d_metadata.get("inputs"), dict) else None,
            "psf_npz": str(psf_npz) if psf_npz is not None else (
                source_d_metadata.get("inputs", {}).get("psf_npz") if isinstance(source_d_metadata.get("inputs"), dict) else None
            ),
            "psf_metadata_json": str(psf_metadata_path) if psf_metadata_path is not None else None,
            "cell_selection_csv": str(path(args.cell_selection_csv)),
            "source_stage_d_npz": str(source_d_npz),
        },
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "current_dir": str(output_root / "current"),
        "latest": str(output_root / "latest"),
        "roi": {
            **source_roi,
            "source": "Crab",
            "source_ra_deg": metadata_float(source_d_metadata, "roi", "source_ra_deg", default=83.63),
            "source_dec_deg": metadata_float(source_d_metadata, "roi", "source_dec_deg", default=22.01),
            "fiducial_radius_deg": roi_fiducial,
        },
        "background_model": {
            **background_model,
            "background_mode": "crab_roi_local",
            "method": "annulus_quadratic",
            "background_form": "direct_expectation",
            "B_on_formula": (
                "integral of annulus-count-normalized weighted least-squares annulus "
                "quadratic surface over on aperture"
            ),
            "surface_order": surface_order,
            "surface_basis": ["1", "x", "y", "x^2", "x*y", "y^2"] if surface_order == 2 else ["1", "x", "y"],
            "annulus_normalize_surface": True,
            "surface_normalization_formula": (
                "scale_b=sum_annulus(counts_b)/sum_annulus(max(B_raw_b,0)); "
                "B_final_b=scale_b*max(B_raw_b,0)"
            ),
            "annulus_default_inner_deg": annulus_default_inner,
            "annulus_width_deg": annulus_width,
            "annulus_max_inner_deg": annulus_max_inner,
            "annulus_source_mask_min_deg": annulus_source_mask_min,
            "annulus_source_mask_r_opt_factor": annulus_source_mask_factor,
            "annulus_source_mask_margin_deg": annulus_source_mask_margin,
            "surface_condition_max": condition_max,
            "surface_min_training_pixels": min_training_pixels,
            "li_ma_applicable": False,
        },
        "processing": {
            "input_rows_scanned": source_d_metadata.get("processing", {}).get("input_rows_scanned") if isinstance(source_d_metadata.get("processing"), dict) else None,
            "processed_batches": source_d_metadata.get("processing", {}).get("processed_batches") if isinstance(source_d_metadata.get("processing"), dict) else None,
            "max_batches": source_d_metadata.get("processing", {}).get("max_batches") if isinstance(source_d_metadata.get("processing"), dict) else None,
            "elapsed_seconds": float(time.perf_counter() - start),
            "reused_counts_map": True,
        },
        "quality": {
            "status": quality_status,
            "promotable": not bool(fit_warning_ids),
            "reason": quality_reason,
            "warnings": severe_warnings,
            "candidate_warning_cell_ids": sorted(set(warning_ids)),
            "active_fit_warning_cell_ids": fit_warning_ids,
        },
        "cells": rows,
        "promotion": {
            "promote_current": not bool(args.no_promote_current),
            "status": "pending",
        },
        "outputs": {
            "npz": str(npz_path),
            "metadata_json": str(metadata_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            "source_masks_csv": str(source_masks_csv_path),
            **plot_outputs,
        },
    }

    stage04.write_summary_csv(summary_csv_path, rows)
    stage04.write_source_masks_csv(
        source_masks_csv_path,
        [
            stage04.SourceMask(
                "Crab",
                float(metadata["roi"]["source_ra_deg"]),  # type: ignore[index]
                float(metadata["roi"]["source_dec_deg"]),  # type: ignore[index]
                metadata_float(source_d_metadata, "roi", "source_mask_min_radius_deg", default=2.0),
            )
        ],
    )
    stage04.write_summary_md(summary_md_path, metadata, rows)
    write_json(metadata_path, metadata)

    if not args.no_promote_current and not fit_warning_ids:
        stage04.promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
        write_json(metadata_path, metadata)
    elif args.no_promote_current:
        metadata["promotion"]["status"] = "skipped"  # type: ignore[index]
        write_json(metadata_path, metadata)
    else:
        metadata["promotion"]["status"] = "blocked_quality_gate"  # type: ignore[index]
        write_json(metadata_path, metadata)

    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {metadata_path}")
    return run_dir, npz_path, metadata_path, metadata


def build_stage_e(
    args: argparse.Namespace,
    *,
    source_e_npz: Path,
    source_e_metadata_path: Path,
    source_e_metadata: Dict[str, object],
    stage_d_npz: Path,
    stage_d_metadata_path: Path,
    stage_d_metadata: Dict[str, object],
    cells: Sequence[object],
) -> Tuple[Path, Path, Path, Dict[str, object]]:
    start = time.perf_counter()
    source_e = load_npz_arrays(source_e_npz)
    stage_d = load_npz_arrays(stage_d_npz)
    order_e = order_for_cell_ids(source_e["cell_id"], cells)
    order_d = order_for_cell_ids(stage_d["cell_id"], cells)
    n_on = np.asarray(source_e["N_on"][order_e], dtype=np.int64)
    arrays = {
        "cell_id": np.asarray(stage_d["cell_id"][order_d], dtype=np.int32),
        "r_opt_deg": np.asarray(stage_d["r_opt_deg"][order_d], dtype=np.float64),
        "sigma_deg": np.asarray(stage_d["sigma_deg"][order_d], dtype=np.float64),
        "containment_r_opt": np.asarray(stage_d["containment_r_opt"][order_d], dtype=np.float64),
        "nhit_bin": np.asarray(stage_d["nhit_bin"][order_d]).astype("U32"),
        "predE_bin": np.asarray(stage_d["predE_bin"][order_d]).astype("U32"),
        "B_on": np.asarray(stage_d["B_on"][order_d], dtype=np.float64),
        "N_off": np.full(len(cells), np.nan, dtype=np.float64),
        "alpha": np.full(len(cells), np.nan, dtype=np.float64),
    }
    roi = stage_d_metadata.get("roi") if isinstance(stage_d_metadata.get("roi"), dict) else {}
    roi_fiducial = finite_float(roi.get("fiducial_radius_deg")) if isinstance(roi, dict) else None
    contract = stage05.BackgroundContract(
        arrays=arrays,
        background_mode="crab_roi_local",
        background_form="direct_expectation",
        statistic_kind="known_background_poisson",
        strict_contract=True,
        promotable_contract=True,
        diagnostic_reasons=[],
        warnings=[],
        roi_fiducial_deg=roi_fiducial,
        roi_config=dict(roi) if isinstance(roi, dict) else {},
        metadata_summary={
            "background_mode": "crab_roi_local",
            "background_form": "direct_expectation",
            "method": "annulus_quadratic_annulus_normalized",
            "stage_d_run_id": stage_d_metadata.get("run_id"),
        },
    )
    processing = source_e_metadata.get("processing") if isinstance(source_e_metadata.get("processing"), dict) else {}
    scan = stage05.ScanResult(
        n_on=n_on,
        input_rows=int(processing.get("input_rows_scanned") or 0),
        valid_cell_rows=int(processing.get("valid_cell_rows") or 0),
        roi_kept_rows=int(processing.get("roi_kept_rows") or 0),
        roi_rejected_rows=int(processing.get("roi_rejected_rows") or 0),
        processed_batches=int(processing.get("processed_batches") or 0),
    )
    stats = stage05.compute_signal_stats(
        scan.n_on,
        contract.arrays["B_on"],
        background_form=contract.background_form,
        n_off=contract.arrays["N_off"],
        alpha=contract.arrays["alpha"],
    )
    quality = stage05.evaluate_quality(
        stats,
        contract,
        min_total_sigma=float(args.quality_min_total_sigma),
        max_total_sigma=float(args.quality_max_total_sigma),
        max_batches=None,
    )
    quality["min_total_sigma"] = float(args.quality_min_total_sigma)
    quality["max_total_sigma"] = float(args.quality_max_total_sigma)

    output_root = path(args.stage_e_output_dir)
    run_id = stage05.sanitize_run_id(args.stage_e_run_id)
    run_dir = stage05.prepare_run_output_dir(output_root, run_id, overwrite_run_dir=bool(args.overwrite_run_dir))
    npz_path = run_dir / args.stage_e_npz_name
    metadata_path = run_dir / args.stage_e_metadata_name
    summary_csv_path = run_dir / args.stage_e_summary_csv_name
    summary_md_path = run_dir / args.stage_e_summary_md_name
    report_html_path = path(args.stage_e_report_html)

    formal_sigma = stats.known_b_sigma
    rows = stage05.build_rows(cells, contract, scan, stats)
    np.savez_compressed(
        npz_path,
        cell_id=np.asarray([cell.cell_id for cell in cells], dtype=np.int32),
        nhit_bin=np.asarray([cell.nhit_bin for cell in cells], dtype="U32"),
        predE_bin=np.asarray([cell.predE_bin for cell in cells], dtype="U32"),
        r_opt_deg=contract.arrays["r_opt_deg"].astype(np.float32),
        sigma_deg=contract.arrays["sigma_deg"].astype(np.float32),
        containment_r_opt=contract.arrays["containment_r_opt"].astype(np.float32),
        N_on=scan.n_on.astype(np.int64),
        B_on=contract.arrays["B_on"].astype(np.float64),
        N_off=contract.arrays["N_off"].astype(np.float64),
        alpha=contract.arrays["alpha"].astype(np.float64),
        excess=stats.excess.astype(np.float64),
        excess_err_stat=stats.excess_err_stat.astype(np.float64),
        excess_err_conservative=stats.excess_err_conservative.astype(np.float64),
        known_b_sigma=stats.known_b_sigma.astype(np.float64),
        li_ma_sigma=stats.li_ma_sigma.astype(np.float64),
        formal_sigma=formal_sigma.astype(np.float64),
        background_mode=np.asarray([contract.background_mode] * len(cells), dtype="U64"),
        background_form=np.asarray([contract.background_form] * len(cells), dtype="U64"),
        statistic_kind=np.asarray([contract.statistic_kind] * len(cells), dtype="U64"),
        roi_fiducial_deg=np.asarray(np.nan if contract.roi_fiducial_deg is None else contract.roi_fiducial_deg, dtype=np.float64),
    )

    plot_outputs: Dict[str, str] = {}
    if not args.no_plots:
        plot_outputs = {
            "formal_sigma_grid_png": str(run_dir / "formal_sigma_grid.png"),
            "known_b_sigma_grid_png": str(run_dir / "known_b_sigma_grid.png"),
            "excess_grid_png": str(run_dir / "excess_grid.png"),
            "on_background_grid_png": str(run_dir / "on_background_grid.png"),
            "on_over_background_grid_png": str(run_dir / "on_over_background_grid.png"),
        }
        stage05.plot_heatmap_grid(formal_sigma, cells, Path(plot_outputs["formal_sigma_grid_png"]), title="Stage E formal significance", colorbar_label="formal sigma", cmap_name="RdBu_r", symmetric=True)
        stage05.plot_heatmap_grid(stats.known_b_sigma, cells, Path(plot_outputs["known_b_sigma_grid_png"]), title="Stage E known-background diagnostic", colorbar_label="known-B sigma", cmap_name="RdBu_r", symmetric=True)
        stage05.plot_heatmap_grid(stats.excess, cells, Path(plot_outputs["excess_grid_png"]), title="Stage E excess by configured cell", colorbar_label="excess", cmap_name="magma")
        stage05.plot_on_background_grid(scan.n_on, contract.arrays["B_on"], cells, Path(plot_outputs["on_background_grid_png"]), title="Stage E Crab on-region counts and annulus-normalized background")
        ratio = np.divide(
            scan.n_on.astype(np.float64),
            contract.arrays["B_on"],
            out=np.full(len(cells), np.nan),
            where=contract.arrays["B_on"] > 0,
        )
        stage05.plot_heatmap_grid(ratio, cells, Path(plot_outputs["on_over_background_grid_png"]), title="Stage E N_on / B_on after annulus normalization", colorbar_label="N_on / B_on", cmap_name="viridis")

    stage_c_dir_text = None
    inputs = source_e_metadata.get("inputs") if isinstance(source_e_metadata.get("inputs"), dict) else {}
    if isinstance(inputs, dict):
        stage_c_dir_text = inputs.get("stage_c_dir")
    stage_c_dir = Path(str(stage_c_dir_text)).resolve() if stage_c_dir_text else path("apply/output/stage_c_v3_candidate/current")
    obs_events_dir = stage_c_dir / "obs_events"
    stage_c_metadata_path = stage_c_dir / "obs_events_metadata.json"
    stage_c_metadata = load_json(stage_c_metadata_path) if stage_c_metadata_path.exists() else {}
    outputs: Dict[str, object] = {
        "npz": str(npz_path),
        "metadata_json": str(metadata_path),
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
        "report_html": str(report_html_path),
        **plot_outputs,
    }
    stage_e_args = SimpleNamespace(
        source_ra_deg=metadata_float(stage_d_metadata, "roi", "source_ra_deg", default=83.63),
        source_dec_deg=metadata_float(stage_d_metadata, "roi", "source_dec_deg", default=22.01),
        batch_size=int(processing.get("batch_size") or 500000),
        max_batches=None,
        mjd_min=processing.get("mjd_min"),
        mjd_max=processing.get("mjd_max"),
        no_promote_current=bool(args.no_promote_current),
    )
    metadata = stage05.make_metadata(
        args=stage_e_args,
        run_id=run_id,
        run_dir=run_dir,
        output_root=output_root,
        stage_c_dir=stage_c_dir,
        obs_events_dir=obs_events_dir,
        stage_c_metadata_path=stage_c_metadata_path,
        stage_c_metadata=stage_c_metadata,
        background_npz=stage_d_npz,
        background_metadata_path=stage_d_metadata_path,
        background_metadata=stage_d_metadata,
        selection_csv=path(args.cell_selection_csv),
        contract=contract,
        scan=scan,
        stats=stats,
        quality=quality,
        rows=rows,
        outputs=outputs,
        elapsed_seconds=time.perf_counter() - start,
    )
    metadata["derived_from"] = {
        "source_stage_e_npz": str(source_e_npz),
        "source_stage_e_metadata": str(source_e_metadata_path),
        "source_stage_e_run_id": source_e_metadata.get("run_id"),
        "note": "Reuses existing full Stage E N_on and recomputes signal statistics with annulus-normalized Stage D B_on.",
    }
    stage05.write_summary_csv(summary_csv_path, rows)
    stage05.write_summary_md(summary_md_path, metadata, rows)
    stage05.write_json(metadata_path, metadata)
    if not args.no_plots:
        stage05.write_report_html(report_html_path, metadata, rows)

    promotable = bool(quality.get("promotable")) and not bool(args.no_promote_current)
    if promotable:
        stage05.promote_successful_run(output_root, run_dir)
        metadata["promotion"]["status"] = "promoted"  # type: ignore[index]
        metadata["promotion"]["current_dir"] = str(output_root / "current")  # type: ignore[index]
        metadata["promotion"]["latest"] = str(output_root / "latest")  # type: ignore[index]
        stage05.write_json(metadata_path, metadata)
    elif args.no_promote_current:
        metadata["promotion"]["status"] = "skipped"  # type: ignore[index]
        stage05.write_json(metadata_path, metadata)
    else:
        metadata["promotion"]["status"] = "blocked_quality_gate"  # type: ignore[index]
        metadata["promotion"]["reason"] = str(quality.get("reason", ""))  # type: ignore[index]
        stage05.write_json(metadata_path, metadata)

    print(f"Wrote {npz_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {metadata_path}")
    return run_dir, npz_path, metadata_path, metadata


def main() -> None:
    args = parse_args()
    source_d_npz = path(args.source_stage_d_npz)
    source_d_metadata_path = path(args.source_stage_d_metadata)
    source_e_npz = path(args.source_stage_e_npz)
    source_e_metadata_path = path(args.source_stage_e_metadata)
    psf_npz = path(args.psf_npz) if str(args.psf_npz or "").strip() else None
    psf_metadata_path = path(args.psf_metadata) if str(args.psf_metadata or "").strip() else None
    required_paths = [source_d_npz, source_d_metadata_path, source_e_npz, source_e_metadata_path]
    if psf_npz is not None:
        required_paths.append(psf_npz)
    if psf_metadata_path is not None:
        required_paths.append(psf_metadata_path)
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    cells = stage04.load_cells(path(args.cell_selection_csv))
    fit_ids = read_selector_ids(path(args.baseline_selector_csv))
    source_d_metadata = load_json(source_d_metadata_path)
    source_e_metadata = load_json(source_e_metadata_path)
    stage_d_run_dir, stage_d_npz, stage_d_metadata_path, stage_d_metadata = build_stage_d(
        args,
        source_d_npz=source_d_npz,
        source_d_metadata_path=source_d_metadata_path,
        source_d_metadata=source_d_metadata,
        psf_npz=psf_npz,
        psf_metadata_path=psf_metadata_path,
        cells=cells,
        fit_ids=fit_ids,
    )
    stage_e_run_dir, _, _, _ = build_stage_e(
        args,
        source_e_npz=source_e_npz,
        source_e_metadata_path=source_e_metadata_path,
        source_e_metadata=source_e_metadata,
        stage_d_npz=stage_d_npz,
        stage_d_metadata_path=stage_d_metadata_path,
        stage_d_metadata=stage_d_metadata,
        cells=cells,
    )
    print(f"Stage D annnorm run: {stage_d_run_dir}")
    print(f"Stage E annnorm run: {stage_e_run_dir}")


if __name__ == "__main__":
    main()
