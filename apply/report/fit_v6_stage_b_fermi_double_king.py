#!/usr/bin/env python3
"""Fit all v6 Stage B profiles with a Fermi-style double-King model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apply.stages.psf_double_king import (  # noqa: E402
    DoubleKingFit,
    fit_double_king_profile,
    kl_divergence,
    profile_probability,
    rayleigh_bin_probabilities,
)


BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
RUN_ID = f"{BASE_RUN_ID}_fermi_double_king"
BASE_STAGE_B_RUN = (
    REPO_ROOT
    / "apply"
    / "output"
    / f"stage_b_{BASE_RUN_ID}"
    / "runs"
    / f"{BASE_RUN_ID}_stage_b_psf"
)
DEFAULT_FORMAL_NPZ = BASE_STAGE_B_RUN / f"psf_{BASE_RUN_ID}.npz"
DEFAULT_FORMAL_METADATA = BASE_STAGE_B_RUN / f"psf_{BASE_RUN_ID}_metadata.json"
DEFAULT_DIAGNOSTIC_NPZ = BASE_STAGE_B_RUN / f"psf_{BASE_RUN_ID}_unfiltered_diagnostic.npz"
DEFAULT_SELECTOR = REPO_ROOT / "apply" / "config" / f"cell_selector_{BASE_RUN_ID}_fit.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "apply"
    / "output"
    / f"stage_b_{RUN_ID}"
    / "runs"
    / RUN_ID
)
DEFAULT_ASSET_DIR = REPO_ROOT / "apply" / "report" / "assets" / RUN_ID.replace("_", "-")
TARGET_CONTAINMENT = 1.0 - math.exp(-0.5 * 1.58**2)

MODEL_COLOR = "#D55E00"
FORMAL_PROFILE_COLOR = "#1F4E79"
DIAGNOSTIC_COLOR = "#7C3AED"
DIAGNOSTIC_MODEL_COLOR = "#A855F7"
FIT_FACE_COLOR = "#ECFDF5"
FIT_EDGE_COLOR = "#059669"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-npz", type=Path, default=DEFAULT_FORMAL_NPZ)
    parser.add_argument("--formal-metadata", type=Path, default=DEFAULT_FORMAL_METADATA)
    parser.add_argument("--diagnostic-npz", type=Path, default=DEFAULT_DIAGNOSTIC_NPZ)
    parser.add_argument("--fit-selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--random-starts", type=int, default=16)
    parser.add_argument("--gamma-min", type=float, default=1.05)
    parser.add_argument("--target-containment", type=float, default=TARGET_CONTAINMENT)
    parser.add_argument("--cell-ids", type=str, default=None, help="Comma-separated smoke-test subset.")
    parser.add_argument("--no-plot", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_fit_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(float(row["cell_id"]))
            for row in csv.DictReader(handle)
            if row.get("cell_id") and truthy(row.get("include"))
        }


def parse_interval(label: str) -> tuple[float | None, float | None]:
    text = label.strip()
    if text.lower() in {"all", "*"}:
        return None, None
    if text.startswith("[") and text.endswith(")"):
        low, high = text[1:-1].split(",", 1)
        return float(low), float(high)
    if text.startswith("<"):
        return None, float(text[1:])
    if text.startswith(">="):
        return float(text[2:]), None
    raise ValueError(f"Unsupported interval label: {label}")


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None and high is None:
        return 1.0e30
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def display_cell_id(cell_id: int, pred_bin: str) -> int | None:
    if pred_bin.strip().startswith(">="):
        return None
    return int(cell_id) - ((int(cell_id) - 1) // 13)


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _positive_profile(density: np.ndarray, edges: np.ndarray) -> bool:
    values = np.asarray(density, dtype=np.float64)
    return values.shape == (edges.size - 1,) and np.isfinite(values).any() and float(np.nansum(values)) > 0.0


def _fit_task(task: dict[str, object]) -> tuple[int, DoubleKingFit, np.ndarray, float]:
    cell_id = int(task["cell_id"])
    density = np.asarray(task["density"], dtype=np.float64)
    edges = np.asarray(task["edges"], dtype=np.float64)
    fit, model_density = fit_double_king_profile(
        density,
        edges,
        target_containment=float(task["target_containment"]),
        gamma_min=float(task["gamma_min"]),
        random_seed=int(task["seed"]),
        random_starts=int(task["random_starts"]),
    )
    rayleigh_sigma_deg = float(task["rayleigh_sigma_deg"])
    rayleigh_probability = rayleigh_bin_probabilities(edges, rayleigh_sigma_deg)
    rayleigh_kl = kl_divergence(profile_probability(density, edges), rayleigh_probability)
    return cell_id, fit, model_density, rayleigh_kl


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[int, dict[str, object]], set[int]]:
    with np.load(args.formal_npz, allow_pickle=False) as formal_npz:
        formal = {key: np.asarray(formal_npz[key]) for key in formal_npz.files}
    required = {
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "profile_edges_deg",
        "profile_density",
        "sigma_deg",
        "r_opt_deg",
        "events",
    }
    missing = required.difference(formal)
    if missing:
        raise KeyError(f"Formal Stage B NPZ is missing arrays: {sorted(missing)}")

    edges = np.asarray(formal["profile_edges_deg"], dtype=np.float64)
    cell_ids = np.asarray(formal["cell_id"], dtype=np.int64)
    if np.unique(cell_ids).size != cell_ids.size:
        raise ValueError("Formal Stage B NPZ contains duplicate cell IDs")
    if np.asarray(formal["profile_density"]).shape != (cell_ids.size, edges.size - 1):
        raise ValueError("Formal Stage B profile array has incompatible dimensions")

    diagnostic_by_cell: dict[int, dict[str, object]] = {}
    if args.diagnostic_npz.exists():
        with np.load(args.diagnostic_npz, allow_pickle=False) as diagnostic_npz:
            diagnostic = {key: np.asarray(diagnostic_npz[key]) for key in diagnostic_npz.files}
        diagnostic_edges = np.asarray(diagnostic["profile_edges_deg"], dtype=np.float64)
        if diagnostic_edges.shape != edges.shape or not np.allclose(diagnostic_edges, edges):
            raise ValueError("Formal and unfiltered diagnostic profile edges do not match")
        diagnostic_ids = np.asarray(diagnostic["cell_id"], dtype=np.int64)
        for index, cell_id in enumerate(diagnostic_ids):
            diagnostic_by_cell[int(cell_id)] = {
                "density": np.asarray(diagnostic["profile_density"][index], dtype=np.float64),
                "sigma_deg": float(diagnostic["sigma_deg"][index]),
                "r_opt_deg": float(diagnostic["r_opt_deg"][index]),
                "events": int(diagnostic["events"][index]),
                "status": str(diagnostic["status"][index]),
            }
    fit_ids = load_fit_ids(args.fit_selector)
    return formal, diagnostic_by_cell, fit_ids


def fit_all_profiles(args: argparse.Namespace) -> tuple[list[dict[str, object]], np.ndarray, set[int]]:
    formal, diagnostic_by_cell, fit_ids = _load_inputs(args)
    edges = np.asarray(formal["profile_edges_deg"], dtype=np.float64)
    requested_ids = None
    if args.cell_ids:
        requested_ids = {int(value.strip()) for value in args.cell_ids.split(",") if value.strip()}

    rows: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    row_by_cell: dict[int, dict[str, object]] = {}
    for index, cell_id_value in enumerate(np.asarray(formal["cell_id"], dtype=np.int64)):
        cell_id = int(cell_id_value)
        if requested_ids is not None and cell_id not in requested_ids:
            continue
        formal_density = np.asarray(formal["profile_density"][index], dtype=np.float64)
        diagnostic = diagnostic_by_cell.get(cell_id)
        if _positive_profile(formal_density, edges):
            source = "formal"
            density = formal_density
            rayleigh_sigma_deg = float(formal["sigma_deg"][index])
            original_r_opt_deg = float(formal["r_opt_deg"][index])
            events = int(formal["events"][index])
        elif (
            diagnostic is not None
            and diagnostic["status"] == "ok"
            and _positive_profile(np.asarray(diagnostic["density"]), edges)
        ):
            source = "unfiltered_diagnostic"
            density = np.asarray(diagnostic["density"], dtype=np.float64)
            rayleigh_sigma_deg = float(diagnostic["sigma_deg"])
            original_r_opt_deg = float(diagnostic["r_opt_deg"])
            events = int(diagnostic["events"])
        else:
            source = "empty"
            density = np.zeros(edges.size - 1, dtype=np.float64)
            rayleigh_sigma_deg = float("nan")
            original_r_opt_deg = float("nan")
            events = int(diagnostic["events"]) if diagnostic is not None else 0

        row: dict[str, object] = {
            "cell_id": cell_id,
            "nhit_bin": str(formal["nhit_bin"][index]),
            "predE_bin": str(formal["predE_bin"][index]),
            "included_in_final_sed_fit": cell_id in fit_ids,
            "source": source,
            "events": events,
            "rayleigh_sigma_deg": rayleigh_sigma_deg,
            "original_r_opt_deg": original_r_opt_deg,
            "profile_density": density,
            "fit_status": "empty" if source == "empty" else "pending",
            "fit_error": "",
            "double_king_model_density": np.full(edges.size - 1, np.nan, dtype=np.float64),
        }
        rows.append(row)
        row_by_cell[cell_id] = row
        if source != "empty":
            tasks.append(
                {
                    "cell_id": cell_id,
                    "density": density,
                    "edges": edges,
                    "rayleigh_sigma_deg": rayleigh_sigma_deg,
                    "target_containment": args.target_containment,
                    "gamma_min": args.gamma_min,
                    "random_starts": args.random_starts,
                    "seed": 20260718 + cell_id + (10000 if source == "unfiltered_diagnostic" else 0),
                }
            )

    def record(result: tuple[int, DoubleKingFit, np.ndarray, float]) -> None:
        cell_id, fit, model_density, rayleigh_kl = result
        row = row_by_cell[cell_id]
        row.update(fit.to_dict())
        row["rayleigh_kl_divergence"] = rayleigh_kl
        row["kl_improvement_factor"] = rayleigh_kl / fit.kl_divergence if fit.kl_divergence > 0.0 else float("inf")
        row["double_king_model_density"] = model_density
        row["fit_status"] = "ok"

    workers = max(1, int(args.workers))
    if workers == 1:
        for done, task in enumerate(tasks, start=1):
            try:
                result = _fit_task(task)
                record(result)
                print(f"[{done}/{len(tasks)}] cell={result[0]} fitted", flush=True)
            except Exception as exc:
                row = row_by_cell[int(task["cell_id"])]
                row["fit_status"] = "failed"
                row["fit_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{done}/{len(tasks)}] cell={task['cell_id']} failed: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fit_task, task): task for task in tasks}
            for done, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                    record(result)
                    print(f"[{done}/{len(tasks)}] cell={result[0]} fitted", flush=True)
                except Exception as exc:
                    row = row_by_cell[int(task["cell_id"])]
                    row["fit_status"] = "failed"
                    row["fit_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[{done}/{len(tasks)}] cell={task['cell_id']} failed: {exc}", flush=True)

    rows.sort(key=lambda item: int(item["cell_id"]))
    return rows, edges, fit_ids


def _string_array(rows: list[dict[str, object]], key: str, width: int = 128) -> np.ndarray:
    return np.asarray([str(row.get(key, "")) for row in rows], dtype=f"U{width}")


def _float_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        values.append(number)
    return np.asarray(values, dtype=np.float64)


def write_npz(rows: list[dict[str, object]], edges: np.ndarray, path: Path) -> None:
    np.savez_compressed(
        path,
        cell_id=np.asarray([int(row["cell_id"]) for row in rows], dtype=np.int32),
        nhit_bin=_string_array(rows, "nhit_bin", width=32),
        predE_bin=_string_array(rows, "predE_bin", width=32),
        included_in_final_sed_fit=np.asarray(
            [bool(row["included_in_final_sed_fit"]) for row in rows], dtype=np.bool_
        ),
        source=_string_array(rows, "source", width=32),
        fit_status=_string_array(rows, "fit_status", width=32),
        fit_error=_string_array(rows, "fit_error", width=256),
        boundary_flags=np.asarray(
            [";".join(row.get("boundary_flags", ())) for row in rows], dtype="U128"
        ),
        events=np.asarray([int(row["events"]) for row in rows], dtype=np.int64),
        profile_edges_deg=np.asarray(edges, dtype=np.float32),
        profile_density=np.vstack([np.asarray(row["profile_density"], dtype=np.float64) for row in rows]).astype(
            np.float32
        ),
        double_king_model_density=np.vstack(
            [np.asarray(row["double_king_model_density"], dtype=np.float64) for row in rows]
        ).astype(np.float32),
        rayleigh_sigma_deg=_float_array(rows, "rayleigh_sigma_deg").astype(np.float32),
        original_r_opt_deg=_float_array(rows, "original_r_opt_deg").astype(np.float32),
        conditional_core_fraction=_float_array(rows, "conditional_core_fraction").astype(np.float32),
        physical_core_fraction=_float_array(rows, "physical_core_fraction").astype(np.float32),
        sigma_core_deg=_float_array(rows, "sigma_core_deg").astype(np.float32),
        gamma_core=_float_array(rows, "gamma_core").astype(np.float32),
        sigma_tail_deg=_float_array(rows, "sigma_tail_deg").astype(np.float32),
        gamma_tail=_float_array(rows, "gamma_tail").astype(np.float32),
        conditional_r_target_deg=_float_array(rows, "conditional_r_target_deg").astype(np.float32),
        rayleigh_kl_divergence=_float_array(rows, "rayleigh_kl_divergence").astype(np.float64),
        double_king_kl_divergence=_float_array(rows, "kl_divergence").astype(np.float64),
        kl_improvement_factor=_float_array(rows, "kl_improvement_factor").astype(np.float64),
    )


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "included_in_final_sed_fit",
        "source",
        "events",
        "fit_status",
        "fit_error",
        "conditional_core_fraction",
        "physical_core_fraction",
        "sigma_core_deg",
        "gamma_core",
        "sigma_tail_deg",
        "gamma_tail",
        "conditional_r_target_deg",
        "rayleigh_sigma_deg",
        "original_r_opt_deg",
        "rayleigh_kl_divergence",
        "double_king_kl_divergence",
        "kl_improvement_factor",
        "optimizer_success",
        "optimizer_message",
        "boundary_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(row.get(key, ())) if key == "boundary_flags" else row.get(key, "")
                    for key in fieldnames
                }
            )


def draw_grid(
    rows: list[dict[str, object]],
    edges: np.ndarray,
    png_path: Path,
    pdf_path: Path,
    target_containment: float,
) -> None:
    if len(rows) != 91:
        raise ValueError("The full shaded grid requires all 91 internal cells")
    ordered_nhit = sorted({str(row["nhit_bin"]) for row in rows}, key=interval_key)
    ordered_pred = sorted({str(row["predE_bin"]) for row in rows}, key=interval_key)
    index_by_key = {(str(row["nhit_bin"]), str(row["predE_bin"])): row for row in rows}
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt = setup_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(1.78 * len(ordered_pred), 1.55 * len(ordered_nhit)),
        dpi=150,
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for i, nhit_bin in enumerate(ordered_nhit):
        for j, pred_bin in enumerate(ordered_pred):
            ax = axes[i, j]
            row = index_by_key.get((nhit_bin, pred_bin))
            if row is None:
                ax.set_axis_off()
                continue
            cell_id = int(row["cell_id"])
            shown_cell_id = display_cell_id(cell_id, pred_bin)
            included = bool(row["included_in_final_sed_fit"])
            source = str(row["source"])
            status = str(row["fit_status"])
            boundary_flags = tuple(row.get("boundary_flags", ()))
            density = np.asarray(row["profile_density"], dtype=np.float64)
            model_density = np.asarray(row["double_king_model_density"], dtype=np.float64)

            if included:
                ax.set_facecolor(FIT_FACE_COLOR)
                for spine in ax.spines.values():
                    spine.set_color(FIT_EDGE_COLOR)
                    spine.set_linewidth(1.25)
                ax.text(
                    0.03,
                    0.94,
                    "fit",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.8,
                    color="#047857",
                    fontweight="bold",
                )
            if shown_cell_id is not None:
                ax.text(
                    0.97,
                    0.94,
                    f"cell {shown_cell_id}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.2,
                    color="#0F172A",
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.7},
                )

            if status == "ok" and source == "formal":
                ax.step(centers, density, where="mid", color=FORMAL_PROFILE_COLOR, linewidth=0.9)
                ax.plot(centers, model_density, color=MODEL_COLOR, linewidth=0.9, alpha=0.95)
                ax.axvline(
                    float(row["conditional_r_target_deg"]),
                    color="#444444",
                    linewidth=0.7,
                    linestyle="--",
                )
            elif status == "ok" and source == "unfiltered_diagnostic":
                ax.step(centers, density, where="mid", color=DIAGNOSTIC_COLOR, linewidth=0.9)
                ax.plot(
                    centers,
                    model_density,
                    color=DIAGNOSTIC_MODEL_COLOR,
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.95,
                )
                ax.axvline(
                    float(row["conditional_r_target_deg"]),
                    color=DIAGNOSTIC_COLOR,
                    linewidth=0.7,
                    linestyle=":",
                )
                ax.text(
                    0.03,
                    0.80,
                    f"diag no E cut\nN={int(row['events'])}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.2,
                    color="#6D28D9",
                )
            else:
                label = "no MC events" if int(row["events"]) == 0 else "double-King fit failed"
                ax.text(
                    0.5,
                    0.45,
                    label,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#6B7280",
                )
            if boundary_flags:
                ax.text(
                    0.97,
                    0.78,
                    "bnd",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.0,
                    color="#B91C1C",
                    fontweight="bold",
                )

            ax.set_title(pred_bin, fontsize=6.4)
            ax.set_xlim(float(edges[0]), float(edges[-1]))
            ax.tick_params(labelsize=6, length=2)
            ax.grid(alpha=0.22, linewidth=0.35)
            if j == 0:
                ax.set_ylabel(nhit_bin, fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("r (deg)", fontsize=6.7)

    handles = [
        Line2D([0], [0], color=FORMAL_PROFILE_COLOR, linewidth=0.9, label="MC histogram"),
        Line2D([0], [0], color=MODEL_COLOR, linewidth=0.9, label="Fermi-style double-King fit"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=0.8,
            linestyle="--",
            label=f"conditional r_target ({target_containment:.3f})",
        ),
        Line2D([0], [0], color=DIAGNOSTIC_COLOR, linewidth=0.9, label="unfiltered diagnostic only"),
        Patch(facecolor=FIT_FACE_COLOR, edgecolor=FIT_EDGE_COLOR, label="included in final SED fit"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle(
        f"Stage B {BASE_RUN_ID} Fermi-style double-King radial PSF profiles: fit cells shaded",
        fontsize=11,
        y=0.999,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.963])
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if row["fit_status"] == "ok"]
    active_rows = [row for row in ok_rows if bool(row["included_in_final_sed_fit"])]
    improvement = np.asarray([float(row["kl_improvement_factor"]) for row in ok_rows], dtype=np.float64)
    boundary_rows = [row for row in ok_rows if row.get("boundary_flags")]
    return {
        "n_cells": len(rows),
        "n_fit_ok": len(ok_rows),
        "n_optimizer_success": sum(bool(row.get("optimizer_success")) for row in ok_rows),
        "n_fit_failed": sum(row["fit_status"] == "failed" for row in rows),
        "n_empty": sum(row["fit_status"] == "empty" for row in rows),
        "n_formal": sum(row["source"] == "formal" for row in rows),
        "n_unfiltered_diagnostic": sum(row["source"] == "unfiltered_diagnostic" for row in rows),
        "n_included_in_final_sed_fit": sum(bool(row["included_in_final_sed_fit"]) for row in rows),
        "n_active_cells_improved_over_rayleigh": sum(
            float(row["kl_improvement_factor"]) > 1.0 for row in active_rows
        ),
        "n_boundary_flagged": len(boundary_rows),
        "boundary_flag_counts": {
            flag: sum(flag in row.get("boundary_flags", ()) for row in boundary_rows)
            for flag in ("conditional_core_fraction", "sigma_core", "sigma_ratio", "gamma_core", "gamma_tail")
        },
        "rayleigh_to_double_king_kl_improvement": {
            "min": float(np.nanmin(improvement)) if improvement.size else None,
            "median": float(np.nanmedian(improvement)) if improvement.size else None,
            "max": float(np.nanmax(improvement)) if improvement.size else None,
        },
    }


def main() -> None:
    args = parse_args()
    output_files = {
        "npz": args.output_dir / f"psf_{RUN_ID}.npz",
        "csv": args.output_dir / f"psf_{RUN_ID}_summary.csv",
        "metadata": args.output_dir / f"psf_{RUN_ID}_metadata.json",
        "png": args.output_dir / f"{BASE_RUN_ID}_fermi_double_king_stage_b_radial_psf_profiles_fit_shaded.png",
        "pdf": args.output_dir / f"{BASE_RUN_ID}_fermi_double_king_stage_b_radial_psf_profiles_fit_shaded.pdf",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output files already exist; pass --overwrite: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, edges, fit_ids = fit_all_profiles(args)
    write_npz(rows, edges, output_files["npz"])
    write_csv(rows, output_files["csv"])
    if not args.no_plot:
        draw_grid(rows, edges, output_files["png"], output_files["pdf"], float(args.target_containment))
        args.asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_files["png"], args.asset_dir / output_files["png"].name)
        shutil.copy2(output_files["pdf"], args.asset_dir / output_files["pdf"].name)

    metadata = {
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "model": {
            "name": "fermi_style_double_king",
            "surface_density_formula": (
                "K(r;sigma,gamma)=(1/(2*pi*sigma^2))*(1-1/gamma)*"
                "(1+r^2/(2*gamma*sigma^2))^(-gamma)"
            ),
            "mixture": "f_core*K_core + (1-f_core)*K_tail",
            "fit_likelihood": "exact_bin_integrated_multinomial_shape_likelihood",
            "fit_window_deg": [float(edges[0]), float(edges[-1])],
            "normalization": "conditional_on_profile_fit_window",
            "target_containment": float(args.target_containment),
            "r_target_contract": "conditional_fit_window_diagnostic_not_production_aperture",
            "gamma_min": float(args.gamma_min),
            "random_starts": int(args.random_starts),
        },
        "inputs": {
            "formal_npz": str(args.formal_npz),
            "formal_npz_sha256": path_sha256(args.formal_npz),
            "formal_metadata": str(args.formal_metadata),
            "formal_metadata_sha256": path_sha256(args.formal_metadata),
            "diagnostic_npz": str(args.diagnostic_npz),
            "diagnostic_npz_sha256": path_sha256(args.diagnostic_npz),
            "fit_selector": str(args.fit_selector),
            "fit_selector_sha256": path_sha256(args.fit_selector),
        },
        "summary": summarize(rows),
        "fit_cell_ids": sorted(fit_ids),
        "outputs": {key: str(path) for key, path in output_files.items()},
    }
    output_files["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if metadata["summary"]["n_fit_failed"]:
        raise SystemExit(f"Double-King fitting failed for {metadata['summary']['n_fit_failed']} cells")
    print(json.dumps(metadata["summary"], indent=2))
    for path in output_files.values():
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
