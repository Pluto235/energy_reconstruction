#!/usr/bin/env python3
"""Plot cells 1-3 double-spherical-King fits and their weighted components."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

from apply.stages.psf_rayleigh_king import (  # noqa: E402
    spherical_king_bin_probabilities,
    spherical_king_cdf,
)


BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
INPUT_RUN_LABEL = "v6-64748-nhit100-reselect44-split56-miss030-full-sphere-model-comparison"
OUTPUT_RUN_LABEL = "v6-64748-nhit100-reselect44-split56-miss030-double-spherical-king-components"
INPUT_STEM = f"{BASE_RUN_ID}_cell123_full_sphere_psf_model_comparison"
OUTPUT_STEM = f"{BASE_RUN_ID}_cell123_double_spherical_king_components"

DEFAULT_INPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / INPUT_RUN_LABEL
DEFAULT_INPUT_NPZ = DEFAULT_INPUT_DIR / f"{INPUT_STEM}.npz"
DEFAULT_INPUT_METADATA = DEFAULT_INPUT_DIR / f"{INPUT_STEM}_metadata.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apply" / "report" / "assets" / OUTPUT_RUN_LABEL

COLORS = {
    "data": "#0072B2",
    "total": "#D55E00",
    "core": "#E69F00",
    "tail": "#CC79A7",
    "cut": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--input-metadata", type=Path, default=DEFAULT_INPUT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quadrature-order", type=int, default=20)
    return parser.parse_args()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "lines.linewidth": 1.8,
        }
    )
    return plt


def crossover_radius(
    radius_deg: np.ndarray,
    core_contribution: np.ndarray,
    tail_contribution: np.ndarray,
    *,
    positive_only: bool = True,
) -> float:
    radius = np.asarray(radius_deg, dtype=np.float64)
    core = np.asarray(core_contribution, dtype=np.float64)
    tail = np.asarray(tail_contribution, dtype=np.float64)
    valid = np.isfinite(radius) & np.isfinite(core) & np.isfinite(tail)
    if positive_only:
        valid &= (core > 0.0) & (tail > 0.0)
    if not np.any(valid):
        return float("nan")
    log_ratio = np.abs(np.log(np.clip(core[valid], 1.0e-300, None) / np.clip(tail[valid], 1.0e-300, None)))
    return float(radius[valid][int(np.argmin(log_ratio))])


def draw_figure(
    rows: list[dict[str, Any]],
    edges: np.ndarray,
    grid: np.ndarray,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt = setup_matplotlib()
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, len(rows), figsize=(12.8, 7.2), squeeze=False)
    for column, row in enumerate(rows):
        top = axes[0, column]
        bottom = axes[1, column]
        data_density = np.asarray(row["data_probability"], dtype=np.float64) / widths
        total_density = np.asarray(row["total_probability"], dtype=np.float64) / widths
        core_density = np.asarray(row["core_component_probability"], dtype=np.float64) / widths
        tail_density = np.asarray(row["tail_component_probability"], dtype=np.float64) / widths

        top.stairs(
            data_density,
            edges,
            color=COLORS["data"],
            linewidth=1.15,
            label="weighted raw MC",
            zorder=2,
        )
        top.stairs(
            total_density,
            edges,
            color=COLORS["total"],
            linewidth=2.0,
            label="double spherical-King total",
            zorder=4,
        )
        top.stairs(
            core_density,
            edges,
            color=COLORS["core"],
            linewidth=1.55,
            linestyle="--",
            label=r"weighted core: $f_{\rm c}K_{\rm c}$",
            zorder=3,
        )
        top.stairs(
            tail_density,
            edges,
            color=COLORS["tail"],
            linewidth=1.55,
            linestyle=":",
            label=r"weighted tail: $(1-f_{\rm c})K_{\rm t}$",
            zorder=3,
        )
        top.axvline(
            5.0,
            color=COLORS["cut"],
            linewidth=0.85,
            linestyle=(0, (2, 2)),
            label="former 5 deg edge",
        )
        top.set_xlim(0.0, 20.0)
        top.set_yscale("log")
        top.set_ylim(1.0e-6, max(2.0, 1.4 * float(np.nanmax(data_density))))
        top.set_title(f"cell {row['cell_id']}: Nhit {row['nhit_bin']}, predE {row['predE_bin']}")
        boundary_text = ", ".join(row["boundary_flags"]) if row["boundary_flags"] else "none"
        top.text(
            0.97,
            0.96,
            f"f_c={row['core_fraction']:.3f}\n"
            f"core: sigma={row['sigma_core_deg']:.3f} deg, gamma={row['gamma_core']:.3f}\n"
            f"tail: sigma={row['sigma_tail_deg']:.2f} deg, gamma={row['gamma_tail']:.3f}\n"
            f"KL={row['kl_divergence']:.4e}; boundary={boundary_text}",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
        )
        density_cross = row["density_crossover_deg"]
        if np.isfinite(density_cross) and density_cross <= 20.0:
            top.axvline(density_cross, color="#6B7280", linewidth=0.7, linestyle=(0, (1, 3)), alpha=0.75)

        empirical = np.asarray(row["empirical_survival"], dtype=np.float64)
        total_survival = np.asarray(row["total_survival"], dtype=np.float64)
        core_survival = np.asarray(row["core_component_survival"], dtype=np.float64)
        tail_survival = np.asarray(row["tail_component_survival"], dtype=np.float64)
        bottom.plot(
            grid[empirical > 0.0],
            empirical[empirical > 0.0],
            color=COLORS["data"],
            linewidth=1.65,
            label="empirical survival",
            zorder=2,
        )
        bottom.plot(
            grid[total_survival > 0.0],
            total_survival[total_survival > 0.0],
            color=COLORS["total"],
            linewidth=2.0,
            label="double spherical-King survival",
            zorder=4,
        )
        bottom.plot(
            grid[core_survival > 0.0],
            core_survival[core_survival > 0.0],
            color=COLORS["core"],
            linewidth=1.55,
            linestyle="--",
            label="weighted core survival",
            zorder=3,
        )
        bottom.plot(
            grid[tail_survival > 0.0],
            tail_survival[tail_survival > 0.0],
            color=COLORS["tail"],
            linewidth=1.55,
            linestyle=":",
            label="weighted tail survival",
            zorder=3,
        )
        bottom.axvline(5.0, color=COLORS["cut"], linewidth=0.85, linestyle=(0, (2, 2)))
        bottom.set_xscale("log")
        bottom.set_yscale("log")
        bottom.set_xlim(0.03, 180.0)
        bottom.set_ylim(1.0e-7, 1.15)
        bottom.set_xlabel("angular error, r [deg]")
        bottom.text(
            0.04,
            0.06,
            f"density crossover: {row['density_crossover_deg']:.3f} deg\n"
            f"survival crossover: {row['survival_crossover_deg']:.3f} deg\n"
            f"S(90 deg): data {row['empirical_survival_90deg']:.2e}, model {row['model_survival_90deg']:.2e}",
            transform=bottom.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.5},
        )

    axes[0, 0].set_ylabel("weighted radial density [deg$^{-1}$]")
    axes[1, 0].set_ylabel(r"$S(r)=P(\Delta\theta>r)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Cells 1-3 full-sphere double spherical-King component decomposition", fontsize=12, y=0.995)
    fig.text(
        0.5,
        0.962,
        "Dashed component curves include their fitted mixture weights and sum exactly to the solid total model.",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#374151",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in (args.input_npz, args.input_metadata) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing component-plot inputs: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    metadata = json.loads(args.input_metadata.read_text(encoding="utf-8"))
    metadata_by_cell = {int(row["cell_id"]): row for row in metadata["cells"]}
    with np.load(args.input_npz, allow_pickle=False) as handle:
        source = {key: np.asarray(handle[key]) for key in handle.files}
    edges = np.asarray(source["fit_edges_deg"], dtype=np.float64)
    grid = np.asarray(source["survival_grid_deg"], dtype=np.float64)
    thresholds = np.asarray(source["survival_thresholds_deg"], dtype=np.float64)
    threshold_index = {float(value): index for index, value in enumerate(thresholds)}

    rows: list[dict[str, Any]] = []
    for index, cell_value in enumerate(np.asarray(source["cell_id"], dtype=np.int64)):
        cell_id = int(cell_value)
        cell_metadata = metadata_by_cell[cell_id]
        model_metadata = cell_metadata["models"]["double_spherical_king"]
        fit = model_metadata["fit"]
        core_fraction = float(fit["core_fraction"])
        sigma_core_deg = float(fit["sigma_core_deg"])
        gamma_core = float(fit["gamma_core"])
        sigma_tail_deg = float(fit["sigma_tail_deg"])
        gamma_tail = float(fit["gamma_tail"])

        core_probability = spherical_king_bin_probabilities(
            edges,
            sigma_core_deg,
            gamma_core,
            quadrature_order=args.quadrature_order,
        )
        tail_probability = spherical_king_bin_probabilities(
            edges,
            sigma_tail_deg,
            gamma_tail,
            quadrature_order=args.quadrature_order,
        )
        core_component_probability = core_fraction * core_probability
        tail_component_probability = (1.0 - core_fraction) * tail_probability
        total_probability = core_component_probability + tail_component_probability
        source_total_probability = np.asarray(
            source["double_spherical_king_probability"][index],
            dtype=np.float64,
        )
        if not np.allclose(total_probability, source_total_probability, rtol=0.0, atol=2.0e-12):
            raise ValueError(f"cell {cell_id}: reconstructed density components do not sum to saved total")

        core_survival_shape = 1.0 - spherical_king_cdf(
            grid,
            sigma_core_deg,
            gamma_core,
            quadrature_order=args.quadrature_order,
        )
        tail_survival_shape = 1.0 - spherical_king_cdf(
            grid,
            sigma_tail_deg,
            gamma_tail,
            quadrature_order=args.quadrature_order,
        )
        core_component_survival = core_fraction * core_survival_shape
        tail_component_survival = (1.0 - core_fraction) * tail_survival_shape
        total_survival = np.clip(core_component_survival + tail_component_survival, 0.0, 1.0)
        source_total_survival = np.asarray(
            source["double_spherical_king_survival"][index],
            dtype=np.float64,
        )
        if not np.allclose(total_survival, source_total_survival, rtol=0.0, atol=2.0e-12):
            raise ValueError(f"cell {cell_id}: reconstructed survival components do not sum to saved total")

        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        density_cross = crossover_radius(
            centers,
            core_component_probability / widths,
            tail_component_probability / widths,
        )
        survival_cross = crossover_radius(grid, core_component_survival, tail_component_survival)
        model_at_thresholds = np.interp(thresholds, grid, total_survival)
        empirical_at_thresholds = np.asarray(
            source["empirical_survival_at_thresholds"][index],
            dtype=np.float64,
        )
        rows.append(
            {
                "cell_id": cell_id,
                "nhit_bin": str(source["nhit_bin"][index]),
                "predE_bin": str(source["predE_bin"][index]),
                "data_probability": np.asarray(source["data_probability"][index], dtype=np.float64),
                "empirical_survival": np.asarray(source["empirical_survival"][index], dtype=np.float64),
                "core_fraction": core_fraction,
                "sigma_core_deg": sigma_core_deg,
                "gamma_core": gamma_core,
                "sigma_tail_deg": sigma_tail_deg,
                "gamma_tail": gamma_tail,
                "kl_divergence": float(model_metadata["kl_divergence"]),
                "boundary_flags": list(fit["boundary_flags"]),
                "core_component_probability": core_component_probability,
                "tail_component_probability": tail_component_probability,
                "total_probability": total_probability,
                "core_component_survival": core_component_survival,
                "tail_component_survival": tail_component_survival,
                "total_survival": total_survival,
                "density_crossover_deg": density_cross,
                "survival_crossover_deg": survival_cross,
                "empirical_survival_90deg": float(empirical_at_thresholds[threshold_index[90.0]]),
                "model_survival_90deg": float(model_at_thresholds[threshold_index[90.0]]),
            }
        )
    rows.sort(key=lambda row: int(row["cell_id"]))

    outputs = {
        "png": args.output_dir / f"{OUTPUT_STEM}.png",
        "pdf": args.output_dir / f"{OUTPUT_STEM}.pdf",
        "npz": args.output_dir / f"{OUTPUT_STEM}.npz",
        "csv": args.output_dir / f"{OUTPUT_STEM}_summary.csv",
        "metadata": args.output_dir / f"{OUTPUT_STEM}_metadata.json",
    }
    draw_figure(rows, edges, grid, outputs["png"], outputs["pdf"])

    np.savez_compressed(
        outputs["npz"],
        cell_id=np.asarray([row["cell_id"] for row in rows], dtype=np.int32),
        nhit_bin=np.asarray([row["nhit_bin"] for row in rows], dtype="U32"),
        predE_bin=np.asarray([row["predE_bin"] for row in rows], dtype="U32"),
        fit_edges_deg=edges,
        survival_grid_deg=grid,
        data_probability=np.vstack([row["data_probability"] for row in rows]),
        empirical_survival=np.vstack([row["empirical_survival"] for row in rows]),
        core_component_probability=np.vstack(
            [row["core_component_probability"] for row in rows]
        ),
        tail_component_probability=np.vstack(
            [row["tail_component_probability"] for row in rows]
        ),
        total_probability=np.vstack([row["total_probability"] for row in rows]),
        core_component_survival=np.vstack(
            [row["core_component_survival"] for row in rows]
        ),
        tail_component_survival=np.vstack(
            [row["tail_component_survival"] for row in rows]
        ),
        total_survival=np.vstack([row["total_survival"] for row in rows]),
    )

    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "core_fraction",
        "tail_fraction",
        "sigma_core_deg",
        "gamma_core",
        "sigma_tail_deg",
        "gamma_tail",
        "kl_divergence",
        "density_crossover_deg",
        "survival_crossover_deg",
        "empirical_survival_90deg",
        "model_survival_90deg",
        "boundary_flags",
    ]
    with outputs["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row.get(name) for name in fieldnames},
                    "tail_fraction": 1.0 - row["core_fraction"],
                    "boundary_flags": ";".join(row["boundary_flags"]),
                }
            )

    compact_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "data_probability",
                "empirical_survival",
                "core_component_probability",
                "tail_component_probability",
                "total_probability",
                "core_component_survival",
                "tail_component_survival",
                "total_survival",
            }
        }
        for row in rows
    ]
    output_metadata = {
        "description": "Cells 1-3 full-sphere double spherical-King fits with weighted core and tail component decomposition",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "component_definition": {
            "density": "p_total = f_core * K_core + (1-f_core) * K_tail",
            "survival": "S_total = f_core * S_core + (1-f_core) * S_tail",
            "normalization_support_deg": [0.0, 180.0],
        },
        "quadrature": f"Gauss-Legendre order {args.quadrature_order} per radial bin",
        "inputs": {
            "npz": str(args.input_npz.resolve()),
            "npz_sha256": path_sha256(args.input_npz),
            "metadata": str(args.input_metadata.resolve()),
            "metadata_sha256": path_sha256(args.input_metadata),
        },
        "cells": compact_rows,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": compact_rows, "outputs": output_metadata["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
