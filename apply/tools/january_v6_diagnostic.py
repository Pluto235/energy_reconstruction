#!/usr/bin/env python3
"""Preserve failed frozen background fits; never publish them as valid signal/SED."""
import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time


def record_aperture(original, failures, **kwargs):
    try:
        return original(**kwargs)
    except ValueError as exc:
        message = str(exc)
        if not kwargs.get("analytic_required") or not ("non-positive fitted background inside the analytic on aperture" in message or "has invalid analytic B_on=" in message):
            raise
        failures.append({**kwargs, "status": "INVALID_BACKGROUND", "error": message})
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import numpy as np
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from january_v6_inputs import TAG, BASE, load_json, sha256
    from january_v6_validate import run_dir, stage_paths, check_selected_cells
    d = importlib.import_module("apply.stages.04_background")
    e = importlib.import_module("apply.stages.05_signal")
    root = repo / "apply/output" / TAG
    manifest = load_json(root / "analysis_manifest.json")
    selected = manifest["selected_cell_ids"]
    selector = repo / f"apply/config/cell_selector_{BASE}_fit.csv"
    with selector.open() as stream:
        check_selected_cells(selected, list(csv.DictReader(stream)))
    cells = d.load_cells(selector)
    ids = np.array([c.cell_id for c in cells])
    psf_path = stage_paths(root, "psf")[0]
    psf = d.load_psf_by_cell(psf_path, cells)
    radii = np.array([psf[c.cell_id]["r_opt_deg"] for c in cells], dtype=np.float32)
    # Use the same Stage D float32 radius round-trip and grid as the failed production job.
    edges = d.make_offset_roi_edges(8., .1, 0.)
    centers = .5 * (edges[1:] + edges[:-1])
    events = run_dir(root, "c") / "obs_events"
    scan = d.scan_stage_c_roi_events(events, cells, source_ra_deg=83.63, source_dec_deg=22.01,
        x_edges_deg=edges, y_edges_deg=edges, rho_hist_edges_deg=d.make_edges(0, 12, .1),
        roi_fiducial_deg=6., roi_edge_diagnostic_deg=8., batch_size=500000, max_batches=None, print_every=100)
    counts = scan.counts_flat.reshape(len(cells), edges.size - 1, edges.size - 1)
    failures = []
    original = d.select_on_aperture_background
    d.select_on_aperture_background = lambda **kwargs: record_aperture(original, failures, **kwargs)
    try:
        b_on, background, training, on_mask, source_mask, fitted = d.estimate_roi_annulus_surface_background(
            counts, centers, radii.astype(np.float64), cell_ids=ids, analytic_required_mask=np.isin(ids, selected),
            roi_fiducial_deg=6., annulus_default_inner_deg=1.5, annulus_width_deg=2.,
            annulus_source_mask_min_deg=1.5, annulus_source_mask_r_opt_factor=2., annulus_source_mask_margin_deg=.2,
            annulus_max_inner_deg=4.5, surface_order=2, condition_max=1e8, min_training_pixels=80,
            annulus_normalize_surface=True, on_aperture_integration="analytic-quadratic")
    finally:
        d.select_on_aperture_background = original
    if not failures:
        raise ValueError("Expected frozen background failure did not reproduce")
    signal_cells = e.load_cells(selector)
    on = e.scan_on_region(events, signal_cells, radii, 83.63, 22.01, roi_fiducial_deg=6.,
        batch_size=500000, max_batches=None, print_every=100)
    require_same = [c.cell_id for c in signal_cells] == ids.tolist()
    if not require_same or on.input_rows != manifest.get("stage_c_selected_rows", scan.input_rows):
        raise ValueError("Diagnostic signal/background input mismatch")
    valid = np.isfinite(b_on) & (b_on > 0) & (fitted["on_aperture_surface_min"] > 0)
    excess = on.n_on.astype(float) - b_on
    excess[~valid] = np.nan
    errors = np.sqrt(on.n_on + b_on)
    errors[~valid] = np.nan
    npz_path = args.output_dir / "background_signal_diagnostic.npz"
    fitted["on_aperture_integration_by_cell"][np.isin(ids, [r["cell_id"] for r in failures])] = "invalid-no-fallback"
    np.savez_compressed(npz_path, cell_id=ids, selected44=np.isin(ids, selected), r_opt_deg=radii,
        counts_map=counts, background_map=background, training_mask=training, on_mask=on_mask,
        source_mask=source_mask, x_edges_deg=edges, N_on=on.n_on, B_on_valid_or_nan=b_on,
        excess_valid_or_nan=excess, conservative_error_valid_or_nan=errors, **fitted)
    rows = []
    for index, cell in enumerate(cells):
        if cell.cell_id not in selected:
            continue
        finite = lambda value: float(value) if np.isfinite(value) else None
        rows.append({"cell_id": cell.cell_id, "nhit_bin": cell.nhit_bin, "predE_bin": cell.predE_bin,
            "r_opt_deg": float(radii[index]), "N_on": int(on.n_on[index]), "background_valid": bool(valid[index]),
            "B_on": finite(b_on[index]), "raw_analytic_B_on": finite(fitted["B_on_analytic"][index]),
            "surface_min": finite(fitted["on_aperture_surface_min"][index]),
            "annulus_counts": int(fitted["annulus_counts"][index]), "annulus_pixels": int(fitted["annulus_pixels"][index]),
            "annulus_inner_deg": float(fitted["annulus_inner_deg"][index]), "annulus_outer_deg": float(fitted["annulus_outer_deg"][index]),
            "excess": finite(excess[index]), "conservative_error": finite(errors[index]),
            "low_on_lt10": bool(on.n_on[index] < 10), "negative_excess": bool(excess[index] < 0),
            "fit_chi2": finite(fitted["fit_chi2"][index]), "fit_ndof": int(fitted["fit_ndof"][index]),
            "annulus_residual_rms": finite(fitted["annulus_residual_rms"][index])})
    with (args.output_dir / "cell44_diagnostic.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "SCIENTIFIC_BACKGROUND_CONTRACT_FAILED", "diagnostic_only": True,
        "run_tag": TAG, "time_label": "202201", "job_id": os.environ.get("SLURM_JOB_ID"),
        "elapsed_seconds": time.perf_counter() - started, "failures": failures, "cells": rows,
        "all_cell_count": len(cells), "fixed_cell_count": len(rows), "input_rows": scan.input_rows,
        "formal_stages": {"D": "FAILED", "E": "BLOCKED_BY_D", "F": "BLOCKED_BY_D", "G": "BLOCKED_BY_D"},
        "fit_and_sed": None,
        "method": "Original weighted-LS fitter unchanged; record required analytic-aperture exceptions as NaN, without fallback. Existing Stage E spherical event counter used only for diagnostics.",
        "limits": "Per-cell diagnostic excess exists only where the original raw polynomial passes positivity. No aggregate44 signal, spectral fit, or SED is defined.",
        "hashes": {str(npz_path): sha256(npz_path), str(selector): sha256(selector), str(psf_path): sha256(psf_path)}}
    (args.output_dir / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cells", "hashes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
