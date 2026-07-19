#!/usr/bin/env python3
"""Optimize per-cell source apertures from cumulative MC response and Stage-D background."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apply.simulation_all_bin import sanitize_label  # noqa: E402
from apply.stages.empirical_cdf_asimov import (  # noqa: E402
    asimov_significance,
    integrate_centered_disk_density,
    integrate_logpar_flux_bins,
    radius_grid,
    select_smallest_near_maximum,
    signal_counts_from_numerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-response-npz", type=Path, required=True)
    parser.add_argument("--binned-root", type=Path, required=True)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--baseline-psf-npz", type=Path, required=True)
    parser.add_argument("--baseline-psf-metadata", type=Path, required=True)
    parser.add_argument("--background-npz", type=Path, required=True)
    parser.add_argument("--spectrum-metadata", type=Path, required=True)
    parser.add_argument("--spectrum-fit-key", default="logpar_conservative")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scan-cache-input", type=Path)
    parser.add_argument("--scan-cache-output", type=Path)
    parser.add_argument("--tree-name", default="t_eventout")
    parser.add_argument("--weight-branch", default="mc_weight")
    parser.add_argument("--radius-min-deg", type=float, default=0.20)
    parser.add_argument("--radius-max-deg", type=float, default=2.00)
    parser.add_argument("--radius-step-deg", type=float, default=0.01)
    parser.add_argument("--near-max-fraction", type=float, default=0.99)
    parser.add_argument("--pivot-tev", type=float, default=3.0)
    parser.add_argument("--energy-quadrature-points", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]).copy() for name in handle.files}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def selected_ids(selector: Path) -> set[int]:
    rows = csv_rows(selector)
    selected = {int(row["cell_id"]) for row in rows if str(row.get("include", "")).strip() == "1"}
    if len(selected) != 44:
        raise ValueError(f"Expected exactly 44 selected cells, found {len(selected)}")
    return selected


def cell_directory(binned_root: Path, row: Mapping[str, str]) -> Path:
    return (
        binned_root
        / f"nhit_{sanitize_label(str(row['nhit_bin']))}"
        / f"predE_{sanitize_label(str(row['predE_bin']))}"
    )


def tree_arrays(path: Path, tree_name: str, branches: list[str]) -> dict[str, np.ndarray]:
    with uproot.open(path) as root_file:
        tree = root_file[f"{tree_name};1"] if f"{tree_name};1" in root_file else root_file[tree_name]
        return {name: np.asarray(value) for name, value in tree.arrays(branches, library="np").items()}


def scan_cell_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    radii = np.asarray(task["radii_deg"], dtype=np.float64)
    loge_edges = np.asarray(task["loge_edges"], dtype=np.float64)
    theta_edges = np.asarray(task["theta_edges_deg"], dtype=np.float64)
    n_radius = radii.size
    n_energy = loge_edges.size - 1
    n_theta = theta_edges.size - 1
    first_passing = np.zeros((n_radius, n_energy, n_theta), dtype=np.float64)
    full = np.zeros((n_energy, n_theta), dtype=np.float64)
    full_sumw2 = 0.0
    events = 0
    truth_events = 0
    finite_dangle_events = 0

    for file_name in task["files"]:
        arrays = tree_arrays(
            Path(str(file_name)),
            str(task["tree_name"]),
            ["mc_energy", "mc_theta", "mc_dangle", str(task["weight_branch"])],
        )
        energy = np.asarray(arrays["mc_energy"], dtype=np.float64)
        theta_deg = np.degrees(np.asarray(arrays["mc_theta"], dtype=np.float64))
        dangle_deg = np.degrees(np.asarray(arrays["mc_dangle"], dtype=np.float64))
        weight = np.asarray(arrays[str(task["weight_branch"])], dtype=np.float64)
        events += int(energy.size)
        loge = np.full(energy.shape, np.nan, dtype=np.float64)
        positive_energy = np.isfinite(energy) & (energy > 0.0)
        loge[positive_energy] = np.log10(energy[positive_energy])
        e_index = np.searchsorted(loge_edges, loge, side="right") - 1
        t_index = np.searchsorted(theta_edges, theta_deg, side="right") - 1
        valid = (
            positive_energy
            & np.isfinite(theta_deg)
            & np.isfinite(weight)
            & (weight > 0.0)
            & (e_index >= 0)
            & (e_index < n_energy)
            & (t_index >= 0)
            & (t_index < n_theta)
        )
        truth_events += int(np.count_nonzero(valid))
        if np.any(valid):
            flat_truth = e_index[valid] * n_theta + t_index[valid]
            full += np.bincount(
                flat_truth,
                weights=weight[valid],
                minlength=n_energy * n_theta,
            ).reshape(n_energy, n_theta)
            full_sumw2 += float(np.sum(weight[valid] ** 2))

        valid_radius = valid & np.isfinite(dangle_deg) & (dangle_deg >= 0.0)
        finite_dangle_events += int(np.count_nonzero(valid_radius))
        if not np.any(valid_radius):
            continue
        r_index = np.searchsorted(radii, dangle_deg[valid_radius], side="left")
        in_scan = r_index < n_radius
        if not np.any(in_scan):
            continue
        e_valid = e_index[valid_radius][in_scan]
        t_valid = t_index[valid_radius][in_scan]
        r_valid = r_index[in_scan]
        w_valid = weight[valid_radius][in_scan]
        flat = (r_valid * n_energy + e_valid) * n_theta + t_valid
        first_passing += np.bincount(
            flat,
            weights=w_valid,
            minlength=n_radius * n_energy * n_theta,
        ).reshape(n_radius, n_energy, n_theta)

    cumulative = np.cumsum(first_passing, axis=0)
    sumw = float(np.sum(full))
    effective_events = sumw * sumw / full_sumw2 if full_sumw2 > 0.0 else 0.0
    return {
        "cell_id": int(task["cell_id"]),
        "cumulative_numerator_sumw": cumulative,
        "full_numerator_sumw": full,
        "input_files": len(task["files"]),
        "events": events,
        "truth_events": truth_events,
        "finite_dangle_events": finite_dangle_events,
        "effective_events": effective_events,
    }


def scan_response_cache(
    *,
    source_rows: list[dict[str, str]],
    selected: set[int],
    binned_root: Path,
    response: Mapping[str, np.ndarray],
    radii: np.ndarray,
    tree_name: str,
    weight_branch: str,
    workers: int,
) -> dict[str, np.ndarray]:
    tasks: list[dict[str, Any]] = []
    for row in source_rows:
        cell_id = int(row["cell_id"])
        if cell_id not in selected:
            continue
        directory = cell_directory(binned_root, row)
        files = sorted(directory.glob("*.root"))
        if not files:
            raise FileNotFoundError(f"No ROOT files for selected cell {cell_id}: {directory}")
        tasks.append(
            {
                "cell_id": cell_id,
                "files": [str(path) for path in files],
                "tree_name": tree_name,
                "weight_branch": weight_branch,
                "radii_deg": radii,
                "loge_edges": response["logE_true_edges"],
                "theta_edges_deg": response["theta_true_edges_deg"],
            }
        )

    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(scan_cell_worker, task): int(task["cell_id"]) for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[int(result["cell_id"])] = result
            print(f"[MC cache] {completed}/{len(tasks)} cell={result['cell_id']}", flush=True)

    ordered = np.asarray(sorted(selected), dtype=np.int64)
    cumulative = np.stack([results[int(cell)]["cumulative_numerator_sumw"] for cell in ordered])
    full = np.stack([results[int(cell)]["full_numerator_sumw"] for cell in ordered])
    fields = {
        name: np.asarray([results[int(cell)][name] for cell in ordered])
        for name in ("input_files", "events", "truth_events", "finite_dangle_events", "effective_events")
    }
    return {
        "cell_id": ordered,
        "radii_deg": radii.astype(np.float64),
        "cumulative_numerator_sumw": cumulative.astype(np.float64),
        "full_numerator_sumw": full.astype(np.float64),
        "logE_true_edges": np.asarray(response["logE_true_edges"], dtype=np.float64),
        "theta_true_edges_deg": np.asarray(response["theta_true_edges_deg"], dtype=np.float64),
        **fields,
    }


def validate_cache(cache: Mapping[str, np.ndarray], response: Mapping[str, np.ndarray], selected: set[int]) -> None:
    cache_ids = np.asarray(cache["cell_id"], dtype=np.int64)
    if set(cache_ids.tolist()) != selected:
        raise ValueError("Scan cache does not contain the selected 44-cell set")
    for edge_name in ("logE_true_edges", "theta_true_edges_deg"):
        if not np.array_equal(np.asarray(cache[edge_name]), np.asarray(response[edge_name])):
            raise ValueError(f"Scan cache {edge_name} does not match the full response")
    response_index = {int(cell): index for index, cell in enumerate(response["cell_id"])}
    for cache_index, cell_id in enumerate(cache_ids):
        reference = np.asarray(response["numerator_sumw"][response_index[int(cell_id)]], dtype=np.float64)
        scanned = np.asarray(cache["full_numerator_sumw"][cache_index], dtype=np.float64)
        if not np.allclose(scanned, reference, rtol=2.0e-11, atol=1.0e-9):
            delta = float(np.max(np.abs(scanned - reference)))
            raise ValueError(f"Cell {int(cell_id)} cache/full-response numerator mismatch: max abs {delta}")


def fit_parameters(metadata: Mapping[str, Any], key: str) -> dict[str, float]:
    fit = (metadata.get("fits") or {}).get(key) or {}
    parameters = fit.get("parameters") or {}
    required = ("phi0", "alpha", "beta")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise KeyError(f"Spectrum fit {key!r} is missing parameters: {missing}")
    out = {name: float(parameters[name]) for name in required}
    if not all(math.isfinite(value) for value in out.values()) or out["phi0"] <= 0.0:
        raise ValueError(f"Invalid LogPar parameters: {out}")
    return out


def plot_grid(
    source_rows: list[dict[str, str]],
    values_by_cell: Mapping[int, float],
    output_png: Path,
    output_pdf: Path,
    *,
    title: str,
    colorbar_label: str,
    fmt: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def interval_key(value: str) -> tuple[float, str]:
        text = str(value).replace("[", "").replace("(", "").split(",", 1)[0].replace(">=", "")
        try:
            return float(text), value
        except ValueError:
            return float("inf"), value

    nhit = sorted({row["nhit_bin"] for row in source_rows}, key=interval_key)
    prede = sorted({row["predE_bin"] for row in source_rows}, key=interval_key)
    ymap = {label: index for index, label in enumerate(nhit)}
    xmap = {label: index for index, label in enumerate(prede)}
    matrix = np.full((len(nhit), len(prede)), np.nan, dtype=np.float64)
    for row in source_rows:
        cell_id = int(row["cell_id"])
        if cell_id in values_by_cell:
            matrix[ymap[row["nhit_bin"]], xmap[row["predE_bin"]]] = float(values_by_cell[cell_id])
    fig, ax = plt.subplots(figsize=(13.0, 5.8), dpi=180)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("predE bin")
    ax.set_ylabel("Nhit bin")
    ax.set_xticks(np.arange(len(prede)), prede, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit)), nhit, fontsize=8)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, format(matrix[y, x], fmt), ha="center", va="center", color="white", fontsize=6.4)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_curves(rows: list[dict[str, Any]], radii: np.ndarray, output_png: Path, output_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: int(row["cell_id"]))
    ncols = 6
    nrows = int(math.ceil(len(ordered) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 2.05 * nrows), dpi=170, sharex=True, sharey=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for axis, row in zip(axes_flat, ordered):
        z = np.asarray(row["asimov_z_curve"], dtype=np.float64)
        normalized = z / float(np.nanmax(z))
        axis.plot(radii, normalized, color="#0072B2", linewidth=1.0)
        axis.axhline(0.99, color="#777777", linestyle=":", linewidth=0.7)
        axis.axvline(float(row["adopted_r_opt_deg"]), color="#009E73", linewidth=0.9)
        axis.axvline(float(row["exact_max_r_deg"]), color="#D55E00", linestyle="--", linewidth=0.8)
        axis.set_title(f"cell {int(row['cell_id'])}", fontsize=8)
        axis.grid(alpha=0.18)
    for axis in axes_flat[len(ordered) :]:
        axis.set_axis_off()
    fig.supxlabel("aperture radius [deg]")
    fig.supylabel(r"$Z_A(r)/Z_{A,\max}$")
    fig.suptitle("Empirical-response Asimov aperture optimization", y=0.998, fontsize=12)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.985))
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    required_paths = [
        args.full_response_npz,
        args.source_table,
        args.selector,
        args.baseline_psf_npz,
        args.baseline_psf_metadata,
        args.background_npz,
        args.spectrum_metadata,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing optimizer inputs: {missing}")
    if not args.binned_root.is_dir() and args.scan_cache_input is None:
        raise FileNotFoundError(args.binned_root)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    radii = radius_grid(args.radius_min_deg, args.radius_max_deg, args.radius_step_deg)
    response = load_npz(args.full_response_npz)
    baseline_psf = load_npz(args.baseline_psf_npz)
    background = load_npz(args.background_npz)
    spectrum_metadata = load_json(args.spectrum_metadata)
    baseline_metadata = load_json(args.baseline_psf_metadata)
    source_rows = csv_rows(args.source_table)
    if len(source_rows) != 91:
        raise ValueError(f"Expected 91 source-table rows, found {len(source_rows)}")
    selected = selected_ids(args.selector)

    if args.scan_cache_input is not None:
        cache = load_npz(args.scan_cache_input)
        cache_source = str(args.scan_cache_input.resolve())
    else:
        cache = scan_response_cache(
            source_rows=source_rows,
            selected=selected,
            binned_root=args.binned_root,
            response=response,
            radii=radii,
            tree_name=args.tree_name,
            weight_branch=args.weight_branch,
            workers=args.workers,
        )
        cache_source = "scanned_from_binned_mc"
    validate_cache(cache, response, selected)
    if not np.array_equal(np.asarray(cache["radii_deg"], dtype=np.float64), radii):
        raise ValueError("Scan-cache radius grid does not match requested grid")
    if args.scan_cache_output is not None:
        args.scan_cache_output.parent.mkdir(parents=True, exist_ok=True)
        if args.scan_cache_output.exists():
            raise FileExistsError(args.scan_cache_output)
        np.savez_compressed(args.scan_cache_output, **cache)

    parameters = fit_parameters(spectrum_metadata, args.spectrum_fit_key)
    exposure = np.asarray((spectrum_metadata.get("exposure") or {}).get("theta_exposure_sec"), dtype=np.float64)
    theta_edges = np.asarray(response["theta_true_edges_deg"], dtype=np.float64)
    if exposure.shape != (theta_edges.size - 1,):
        raise ValueError("Spectrum metadata exposure does not match response theta bins")
    flux_integral = integrate_logpar_flux_bins(
        response["logE_true_edges"],
        parameters,
        pivot_tev=args.pivot_tev,
        quadrature_points=args.energy_quadrature_points,
    )

    response_index = {int(cell): index for index, cell in enumerate(response["cell_id"])}
    background_index = {int(cell): index for index, cell in enumerate(background["cell_id"])}
    psf_index = {int(cell): index for index, cell in enumerate(baseline_psf["cell_id"])}
    cache_index = {int(cell): index for index, cell in enumerate(cache["cell_id"])}
    denominator = np.asarray(response["denominator_sumw"], dtype=np.float64)
    s0_m2 = float(np.asarray(response["s0_m2"]).reshape(-1)[0])
    rows: list[dict[str, Any]] = []
    for cell_id in sorted(selected):
        cindex = cache_index[cell_id]
        signal_curve = signal_counts_from_numerator(
            cache["cumulative_numerator_sumw"][cindex],
            denominator,
            theta_edges,
            s0_m2,
            flux_integral,
            exposure,
        )
        full_index = response_index[cell_id]
        full_signal = 1.0e4 * float(
            np.einsum(
                "et,e,t->",
                np.asarray(response["a_eff"][full_index], dtype=np.float64),
                flux_integral,
                exposure,
            )
        )
        if not math.isfinite(full_signal) or full_signal <= 0.0:
            raise ValueError(f"Cell {cell_id} has invalid full-aperture expected signal {full_signal}")
        containment = signal_curve / full_signal
        coefficients = np.asarray(
            background["surface_density_coefficients"][background_index[cell_id]],
            dtype=np.float64,
        )
        background_curve = integrate_centered_disk_density(coefficients, radii)
        z_curve = asimov_significance(signal_curve, background_curve)
        adopted_index, exact_index = select_smallest_near_maximum(radii, z_curve, args.near_max_fraction)
        pindex = psf_index[cell_id]
        rows.append(
            {
                "cell_id": cell_id,
                "nhit_bin": str(baseline_psf["nhit_bin"][pindex]),
                "predE_bin": str(baseline_psf["predE_bin"][pindex]),
                "old_r_opt_deg": float(baseline_psf["r_opt_deg"][pindex]),
                "old_containment": float(baseline_psf["containment_r_opt"][pindex]),
                "exact_max_r_deg": float(radii[exact_index]),
                "adopted_r_opt_deg": float(radii[adopted_index]),
                "adopted_containment": float(containment[adopted_index]),
                "full_expected_signal": full_signal,
                "adopted_expected_signal": float(signal_curve[adopted_index]),
                "adopted_expected_background": float(background_curve[adopted_index]),
                "exact_max_asimov_z": float(z_curve[exact_index]),
                "adopted_asimov_z": float(z_curve[adopted_index]),
                "adopted_z_fraction": float(z_curve[adopted_index] / z_curve[exact_index]),
                "scan_lower_boundary": adopted_index == 0,
                "scan_upper_boundary": exact_index == radii.size - 1,
                "input_files": int(cache["input_files"][cindex]),
                "mc_events": int(cache["events"][cindex]),
                "truth_events": int(cache["truth_events"][cindex]),
                "finite_dangle_events": int(cache["finite_dangle_events"][cindex]),
                "effective_events": float(cache["effective_events"][cindex]),
                "signal_curve": signal_curve,
                "background_curve": background_curve,
                "containment_curve": containment,
                "asimov_z_curve": z_curve,
            }
        )

    n_cells = len(baseline_psf["cell_id"])
    n_radii = radii.size
    selected_mask = np.zeros(n_cells, dtype=bool)
    exact_r = np.full(n_cells, np.nan, dtype=np.float64)
    adopted_r = np.asarray(baseline_psf["r_opt_deg"], dtype=np.float64).copy()
    adopted_containment = np.asarray(baseline_psf["containment_r_opt"], dtype=np.float64).copy()
    z_fraction = np.full(n_cells, np.nan, dtype=np.float64)
    signal_curves = np.full((n_cells, n_radii), np.nan, dtype=np.float64)
    background_curves = np.full_like(signal_curves, np.nan)
    containment_curves = np.full_like(signal_curves, np.nan)
    z_curves = np.full_like(signal_curves, np.nan)
    row_by_cell = {int(row["cell_id"]): row for row in rows}
    for cell_id, row in row_by_cell.items():
        index = psf_index[cell_id]
        selected_mask[index] = True
        exact_r[index] = float(row["exact_max_r_deg"])
        adopted_r[index] = float(row["adopted_r_opt_deg"])
        adopted_containment[index] = float(row["adopted_containment"])
        z_fraction[index] = float(row["adopted_z_fraction"])
        signal_curves[index] = np.asarray(row["signal_curve"])
        background_curves[index] = np.asarray(row["background_curve"])
        containment_curves[index] = np.asarray(row["containment_curve"])
        z_curves[index] = np.asarray(row["asimov_z_curve"])

    payload = {name: np.asarray(value).copy() for name, value in baseline_psf.items()}
    payload["r_opt_deg"] = adopted_r.astype(np.float32)
    payload["r_opt_rad"] = np.radians(adopted_r).astype(np.float32)
    payload["containment_r_opt"] = adopted_containment.astype(np.float32)
    payload["empirical_cdf_asimov_selected"] = selected_mask
    payload["empirical_cdf_asimov_radii_deg"] = radii.astype(np.float32)
    payload["empirical_cdf_asimov_exact_max_r_deg"] = exact_r.astype(np.float32)
    payload["empirical_cdf_asimov_adopted_r_deg"] = adopted_r.astype(np.float32)
    payload["empirical_cdf_asimov_adopted_containment"] = adopted_containment.astype(np.float32)
    payload["empirical_cdf_asimov_adopted_z_fraction"] = z_fraction.astype(np.float32)
    payload["empirical_cdf_asimov_signal_curve"] = signal_curves.astype(np.float64)
    payload["empirical_cdf_asimov_background_curve"] = background_curves.astype(np.float64)
    payload["empirical_cdf_asimov_containment_curve"] = containment_curves.astype(np.float64)
    payload["empirical_cdf_asimov_z_curve"] = z_curves.astype(np.float64)
    payload["aperture_strategy"] = np.asarray(["empirical_cdf_asimov_99pct_smallest"], dtype="U64")

    stem = f"psf_{args.run_id}"
    npz_path = args.output_dir / f"{stem}.npz"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    csv_path = args.output_dir / f"{stem}_summary.csv"
    np.savez_compressed(npz_path, **payload)

    csv_fields = [
        "cell_id", "nhit_bin", "predE_bin", "old_r_opt_deg", "old_containment",
        "exact_max_r_deg", "adopted_r_opt_deg", "adopted_containment",
        "full_expected_signal", "adopted_expected_signal", "adopted_expected_background",
        "exact_max_asimov_z", "adopted_asimov_z", "adopted_z_fraction",
        "scan_lower_boundary", "scan_upper_boundary", "input_files", "mc_events",
        "truth_events", "finite_dangle_events", "effective_events",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in csv_fields})

    r_grid_png = args.output_dir / "psf_r_opt_deg_grid.png"
    r_grid_pdf = args.output_dir / "psf_r_opt_deg_grid.pdf"
    c_grid_png = args.output_dir / "empirical_cdf_asimov_containment_grid.png"
    c_grid_pdf = args.output_dir / "empirical_cdf_asimov_containment_grid.pdf"
    curves_png = args.output_dir / "empirical_cdf_asimov_curves.png"
    curves_pdf = args.output_dir / "empirical_cdf_asimov_curves.pdf"
    plot_grid(
        source_rows,
        {int(cell): float(value) for cell, value in zip(baseline_psf["cell_id"], adopted_r)},
        r_grid_png,
        r_grid_pdf,
        title="Stage B empirical-CDF Asimov aperture radius",
        colorbar_label="adopted r_opt [deg]",
        fmt=".2f",
    )
    plot_grid(
        source_rows,
        {int(row["cell_id"]): float(row["adopted_containment"]) for row in rows},
        c_grid_png,
        c_grid_pdf,
        title="Stage B empirical containment at adopted aperture",
        colorbar_label="F(r_opt)",
        fmt=".3f",
    )
    plot_curves(rows, radii, curves_png, curves_pdf)
    baseline_effective_plot = args.baseline_psf_npz.parent / "psf_effective_events_grid.png"
    if baseline_effective_plot.is_file():
        shutil.copy2(baseline_effective_plot, args.output_dir / baseline_effective_plot.name)

    selected_radii = np.asarray([row["adopted_r_opt_deg"] for row in rows], dtype=np.float64)
    selected_containment = np.asarray([row["adopted_containment"] for row in rows], dtype=np.float64)
    metadata = dict(baseline_metadata)
    metadata.update(
        {
            "description": "Stage B empirical cumulative-response aperture optimized with Asimov significance.",
            "run_id": args.run_id,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "output_dir": str(args.output_dir.resolve()),
            "aperture_optimization": {
                "method": "full cumulative Stage-A MC response plus analytic Stage-D background",
                "objective": "Asimov significance",
                "selection_rule": "smallest radius with Z_A >= near_max_fraction * max(Z_A)",
                "radius_min_deg": float(radii[0]),
                "radius_max_deg": float(radii[-1]),
                "radius_step_deg": float(args.radius_step_deg),
                "near_max_fraction": float(args.near_max_fraction),
                "selected_cell_count": len(selected),
                "excluded_cell_policy": "retain baseline PSF aperture",
                "spectrum_model": "LogPar with natural logarithm",
                "spectrum_fit_key": args.spectrum_fit_key,
                "spectrum_parameters": parameters,
                "spectrum_metadata": str(args.spectrum_metadata.resolve()),
                "background_npz": str(args.background_npz.resolve()),
                "background_formula": "pi*r^2*c0 + pi*r^4*(cxx+cyy)/4",
                "response_scan_cache": cache_source,
                "theta_max_deg": float(theta_edges[-1]),
                "containment_definition": "S(r_opt) / S(full), using the same LogPar and Crab theta exposure",
            },
            "summary": {
                "n_cells": n_cells,
                "n_optimized_cells": len(rows),
                "adopted_r_opt_min_deg": float(np.min(selected_radii)),
                "adopted_r_opt_median_deg": float(np.median(selected_radii)),
                "adopted_r_opt_max_deg": float(np.max(selected_radii)),
                "adopted_containment_min": float(np.min(selected_containment)),
                "adopted_containment_median": float(np.median(selected_containment)),
                "adopted_containment_max": float(np.max(selected_containment)),
                "lower_boundary_cells": [int(row["cell_id"]) for row in rows if row["scan_lower_boundary"]],
                "upper_boundary_cells": [int(row["cell_id"]) for row in rows if row["scan_upper_boundary"]],
            },
            "inputs": {
                "full_response_npz": str(args.full_response_npz.resolve()),
                "baseline_psf_npz": str(args.baseline_psf_npz.resolve()),
                "background_npz": str(args.background_npz.resolve()),
                "spectrum_metadata": str(args.spectrum_metadata.resolve()),
                "selector": str(args.selector.resolve()),
                "source_table": str(args.source_table.resolve()),
                "binned_root": str(args.binned_root.resolve()),
            },
            "input_sha256": {
                "full_response_npz": sha256(args.full_response_npz),
                "baseline_psf_npz": sha256(args.baseline_psf_npz),
                "background_npz": sha256(args.background_npz),
                "spectrum_metadata": sha256(args.spectrum_metadata),
                "selector": sha256(args.selector),
            },
            "outputs": {
                "npz": str(npz_path.resolve()),
                "metadata_json": str(metadata_path.resolve()),
                "summary_csv": str(csv_path.resolve()),
                "r_opt_grid_png": str(r_grid_png.resolve()),
                "r_opt_grid_pdf": str(r_grid_pdf.resolve()),
                "containment_grid_png": str(c_grid_png.resolve()),
                "containment_grid_pdf": str(c_grid_pdf.resolve()),
                "optimization_curves_png": str(curves_png.resolve()),
                "optimization_curves_pdf": str(curves_pdf.resolve()),
                "scan_cache": str(args.scan_cache_output.resolve()) if args.scan_cache_output else cache_source,
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    cell_metadata = {int(row.get("cell_id", -1)): row for row in metadata.get("cells", [])}
    for row in rows:
        target = cell_metadata.get(int(row["cell_id"]))
        if target is None:
            continue
        target["r_opt_deg"] = float(row["adopted_r_opt_deg"])
        target["r_opt_rad"] = math.radians(float(row["adopted_r_opt_deg"]))
        target["containment_r_opt"] = float(row["adopted_containment"])
        target["aperture_strategy"] = "empirical_cdf_asimov_99pct_smallest"
        target["exact_max_r_deg"] = float(row["exact_max_r_deg"])
        target["adopted_z_fraction"] = float(row["adopted_z_fraction"])
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote empirical-CDF Asimov PSF contract: {npz_path}", flush=True)


if __name__ == "__main__":
    main()
