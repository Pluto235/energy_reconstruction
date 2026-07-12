#!/usr/bin/env python3
"""Fit the v6 Crab spectrum after aggregating selected cells by Nhit only."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-npz", type=Path, required=True)
    parser.add_argument("--signal-npz", type=Path, required=True)
    parser.add_argument("--selector-csv", type=Path, required=True)
    parser.add_argument("--source-files-csv", type=Path, required=True)
    parser.add_argument("--stage-f-module", type=Path, required=True)
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="v6_64748_nhit_grouped_fit")
    parser.add_argument("--pivot-tev", type=float, default=3.0)
    parser.add_argument("--quadrature-points", type=int, default=64)
    return parser.parse_args()


def load_stage_f(path: Path):
    spec = importlib.util.spec_from_file_location("v6_stage_f_module", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Stage F module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def interval_key(label: str) -> float:
    if label.startswith("<"):
        return -math.inf
    if label.startswith(">="):
        return math.inf
    return float(label.split(",", 1)[0].lstrip("["))


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
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_f = load_stage_f(args.stage_f_module)

    with args.selector_csv.open("r", encoding="utf-8", newline="") as handle:
        selector = list(csv.DictReader(handle))
    selected_ids = {int(row["cell_id"]) for row in selector if str(row["include"]).strip() == "1"}
    if len(selected_ids) != 44:
        raise ValueError(f"Expected the official 44-cell selector, found {len(selected_ids)}")

    with np.load(args.response_npz, allow_pickle=False) as data:
        response = {key: np.asarray(data[key]) for key in data.files}
    with np.load(args.signal_npz, allow_pickle=False) as data:
        signal = {key: np.asarray(data[key]) for key in data.files}
    if not np.array_equal(response["cell_id"], signal["cell_id"]):
        raise ValueError("Stage A and Stage E cell IDs do not match")
    mask = np.asarray([int(cell_id) in selected_ids for cell_id in signal["cell_id"]], dtype=bool)
    if int(np.count_nonzero(mask)) != 44:
        raise ValueError("Selected cell mask does not contain exactly 44 cells")

    cell_ids = signal["cell_id"][mask].astype(np.int64)
    nhit_labels = signal["nhit_bin"][mask].astype(str)
    a_eff = response["a_eff"][mask].astype(np.float64)
    containment = signal["containment_r_opt"][mask].astype(np.float64)
    excess = signal["excess"][mask].astype(np.float64)
    err_conservative = signal["excess_err_conservative"][mask].astype(np.float64)
    err_sqrt_n = signal["excess_err_stat"][mask].astype(np.float64)
    loge_edges = response["logE_true_edges"].astype(np.float64)
    theta_edges = response["theta_true_edges_deg"].astype(np.float64)

    theta_exposure, exposure_meta = stage_f.compute_theta_exposure(
        args.source_files_csv.resolve(),
        theta_edges,
        source_ra_deg=83.63,
        source_dec_deg=22.01,
        latitude_deg=29.45,
        longitude_east_deg=100.14,
        sample_step_sec=60.0,
    )

    groups = sorted(set(nhit_labels.tolist()), key=interval_key)
    group_masks = [nhit_labels == label for label in groups]
    grouped_a_eff = np.stack(
        [np.sum(a_eff[group_mask] * containment[group_mask, None, None], axis=0) for group_mask in group_masks]
    )
    grouped_containment = np.ones(len(groups), dtype=np.float64)
    grouped_excess = np.asarray([np.sum(excess[group_mask]) for group_mask in group_masks], dtype=np.float64)
    grouped_err_conservative = np.asarray(
        [np.sqrt(np.sum(np.square(err_conservative[group_mask]))) for group_mask in group_masks], dtype=np.float64
    )
    grouped_err_sqrt_n = np.asarray(
        [np.sqrt(np.sum(np.square(err_sqrt_n[group_mask]))) for group_mask in group_masks], dtype=np.float64
    )

    fit_kwargs = {
        "a_eff_m2": grouped_a_eff,
        "containment": grouped_containment,
        "theta_exposure_sec": theta_exposure,
        "loge_edges": loge_edges,
        "pivot_tev": float(args.pivot_tev),
        "quadrature_points": int(args.quadrature_points),
        "start_gamma": 2.69,
        "start_phi0": 2.114e-12,
    }
    pl = stage_f.fit_model(
        model_name="pl",
        error_mode="conservative",
        observed=grouped_excess,
        errors=grouped_err_conservative,
        **fit_kwargs,
    )
    logpar = stage_f.fit_model(
        model_name="logpar",
        error_mode="conservative",
        observed=grouped_excess,
        errors=grouped_err_conservative,
        start_gamma=float(pl.parameters["gamma"]),
        start_phi0=float(pl.parameters["phi0"]),
        **{key: value for key, value in fit_kwargs.items() if key not in {"start_gamma", "start_phi0"}},
    )
    delta_chi2 = float(pl.chi2 - logpar.chi2)
    preferred = logpar if logpar.valid and delta_chi2 >= 4.0 else pl

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(groups):
        group_mask = group_masks[index]
        rows.append(
            {
                "nhit_bin": label,
                "internal_cell_ids": ";".join(str(value) for value in cell_ids[group_mask]),
                "n_cells": int(np.count_nonzero(group_mask)),
                "observed_excess": float(grouped_excess[index]),
                "error_conservative": float(grouped_err_conservative[index]),
                "error_sqrt_n": float(grouped_err_sqrt_n[index]),
                "pl_model_counts": float(pl.model_counts[index]),
                "pl_pull": float(pl.pull[index]),
                "logpar_model_counts": float(logpar.model_counts[index]),
                "logpar_pull": float(logpar.pull[index]),
            }
        )

    baseline = json.loads(args.baseline_metadata.read_text(encoding="utf-8"))
    baseline_fit = (baseline.get("fits") or {}).get("logpar_conservative") or {}
    payload = {
        "description": "Diagnostic forward-folding fit after aggregation of the official 44 cells into seven Nhit bins",
        "diagnostic_only": True,
        "aggregation": {
            "observed": "sum of Stage E excess within each Nhit bin",
            "model": "sum of aperture-conditioned per-cell forward-folded counts within each Nhit bin",
            "error": "sqrt(sum(error_conservative^2)) within each Nhit bin",
            "selected_cells": len(selected_ids),
            "group_count": len(groups),
        },
        "inputs": {
            "response_npz": str(args.response_npz.resolve()),
            "signal_npz": str(args.signal_npz.resolve()),
            "selector_csv": str(args.selector_csv.resolve()),
            "source_files_csv": str(args.source_files_csv.resolve()),
        },
        "exposure": exposure_meta,
        "fits": {
            "pl_conservative": stage_f.result_to_metadata(pl),
            "logpar_conservative": stage_f.result_to_metadata(logpar),
        },
        "preferred_model": preferred.model_name,
        "delta_chi2_pl_minus_logpar": delta_chi2,
        "baseline_2d_logpar": {
            "chi2": baseline_fit.get("chi2"),
            "ndof": baseline_fit.get("ndof"),
            "chi2_over_ndof": baseline_fit.get("chi2_over_ndof"),
            "p_value": baseline_fit.get("p_value"),
            "parameters": baseline_fit.get("parameters"),
        },
        "groups": rows,
    }

    json_path = args.output_dir / f"{args.prefix}.json"
    csv_path = args.output_dir / f"{args.prefix}.csv"
    png_path = args.output_dir / f"{args.prefix}.png"
    json_path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)

    x = np.arange(len(groups), dtype=np.float64)
    fig, (ax_counts, ax_pull) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    ax_counts.errorbar(
        x,
        grouped_excess,
        yerr=grouped_err_conservative,
        fmt="o",
        color="#222222",
        capsize=3,
        label="Grouped Stage E excess",
    )
    ax_counts.plot(x, pl.model_counts, "-o", color="#0072B2", label="Grouped PL fit")
    ax_counts.plot(x, logpar.model_counts, "-o", color="#D55E00", label="Grouped LogPar fit")
    ax_counts.set_yscale("log")
    ax_counts.set_ylabel("Counts per Nhit group")
    ax_counts.set_title("v6 diagnostic: forward-folding fit using seven Nhit groups only")
    ax_counts.grid(alpha=0.24)
    ax_counts.legend()
    ax_pull.axhspan(-2.0, 2.0, color="#009E73", alpha=0.10)
    ax_pull.axhline(0.0, color="#555555", linewidth=0.8)
    ax_pull.plot(x, logpar.pull, "o-", color="#D55E00")
    ax_pull.set_ylabel("LogPar pull")
    ax_pull.set_xlabel("Nhit bin")
    ax_pull.set_xticks(x, groups, rotation=25, ha="right")
    ax_pull.grid(alpha=0.24)
    fig.savefig(png_path)
    plt.close(fig)

    print(json.dumps({
        "pl": stage_f.result_to_metadata(pl),
        "logpar": stage_f.result_to_metadata(logpar),
        "preferred_model": preferred.model_name,
        "delta_chi2": delta_chi2,
        "outputs": [str(json_path), str(csv_path), str(png_path)],
    }, indent=2))


if __name__ == "__main__":
    main()
