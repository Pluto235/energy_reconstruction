#!/usr/bin/env python3
"""Redraw v6 report figures with 84-cell display IDs without changing data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-d-npz", type=Path)
    parser.add_argument("--stage-f-npz", type=Path, required=True)
    parser.add_argument("--stage-f-metadata", type=Path, required=True)
    parser.add_argument("--stage-d-output-dir", type=Path)
    parser.add_argument("--stage-f-output-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--profile-half-width-deg", type=float, default=1.0)
    parser.add_argument(
        "--stage-f-only",
        action="store_true",
        help="Redraw only the Stage F model-count figure without touching shared Stage D outputs.",
    )
    return parser.parse_args()


def interval_key(label: str) -> float:
    if label.startswith("<"):
        return -np.inf
    if label.startswith(">="):
        return np.inf
    return float(label.split(",", 1)[0].lstrip("["))


def display_cell_id(cell_id: int, pred_bin: str) -> int | None:
    if pred_bin.strip().startswith(">="):
        return None
    return int(cell_id) - ((int(cell_id) - 1) // 13)


def plot_counts_grid(
    counts: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    xy_edges: np.ndarray,
    r_opt_deg: np.ndarray,
    output_path: Path,
    *,
    roi_fiducial_deg: float,
) -> None:
    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(2.05 * len(ordered_pred), 1.75 * len(ordered_nhit)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])]
    logged = np.full(counts.shape, np.nan, dtype=np.float64)
    positive = counts > 0
    logged[positive] = np.log10(counts[positive])
    finite = logged[np.isfinite(logged)]
    vmin = float(np.percentile(finite, 5.0)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0.0, 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    fiducial_x = roi_fiducial_deg * np.cos(theta)
    fiducial_y = roi_fiducial_deg * np.sin(theta)
    first_image = None

    for i, nhit in enumerate(ordered_nhit):
        for j, pred in enumerate(ordered_pred):
            ax = axes[i, j]
            idx = index_by_key.get((nhit, pred))
            if idx is None:
                ax.set_axis_off()
                continue
            image = ax.imshow(
                logged[idx],
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if first_image is None:
                first_image = image
            ax.plot(fiducial_x, fiducial_y, color="white", linewidth=0.35, alpha=0.85)
            on_radius = float(r_opt_deg[idx])
            ax.plot(on_radius * np.cos(theta), on_radius * np.sin(theta), color="white", linewidth=0.45, alpha=0.9)
            ax.scatter([0.0], [0.0], marker="+", s=18, c="white", linewidths=0.6)
            shown = display_cell_id(int(cell_ids[idx]), pred)
            ax.set_title(f"cell {shown}: {pred}" if shown is not None else pred, fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit}\ny (deg)", fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("x (deg)", fontsize=6.7)

    fig.suptitle("Stage D observed counts skymap before background subtraction", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.982])
    if first_image is not None:
        colorbar = fig.colorbar(first_image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        colorbar.set_label("log10 observed counts", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path)
    plt.close(fig)


def plot_normalized_offset_profiles(
    profile_map: np.ndarray,
    counts_map: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    coordinate_centers: np.ndarray,
    orthogonal_centers: np.ndarray,
    annulus_inner_deg: np.ndarray,
    annulus_outer_deg: np.ndarray,
    fit_ids: set[int],
    output_path: Path,
    *,
    projection: str,
    quantity: str,
    phase: str,
    y_min: float,
    profile_half_width_deg: float,
) -> None:
    if projection not in {"ra", "dec"}:
        raise ValueError(f"Unsupported projection: {projection}")
    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    central_band = np.abs(orthogonal_centers) < float(profile_half_width_deg)
    if not np.any(central_band):
        raise ValueError(
            f"No bins found inside central profile half-width {profile_half_width_deg} deg"
        )
    if projection == "ra":
        profiles = np.nansum(profile_map[:, central_band, :], axis=1)
        count_profiles = np.nansum(counts_map[:, central_band, :], axis=1)
        supported_coordinates = np.any(np.isfinite(profile_map[:, central_band, :]), axis=(0, 1))
        slice_label = f"|Dec offset| < {profile_half_width_deg:g} deg"
    else:
        profiles = np.nansum(profile_map[:, :, central_band], axis=2)
        count_profiles = np.nansum(counts_map[:, :, central_band], axis=2)
        supported_coordinates = np.any(np.isfinite(profile_map[:, :, central_band]), axis=(0, 2))
        slice_label = f"|RA offset cos(dec)| < {profile_half_width_deg:g} deg"
    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(2.05 * len(ordered_pred), 1.75 * len(ordered_nhit)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    if not np.any(supported_coordinates):
        raise ValueError(f"No finite Stage D support for {projection} profile")
    x_min = float(np.min(coordinate_centers[supported_coordinates]))
    x_max = float(np.max(coordinate_centers[supported_coordinates]))
    for i, nhit in enumerate(ordered_nhit):
        for j, pred in enumerate(ordered_pred):
            ax = axes[i, j]
            idx = index_by_key.get((nhit, pred))
            if idx is None:
                ax.set_axis_off()
                continue
            internal_id = int(cell_ids[idx])
            in_fit = internal_id in fit_ids
            if in_fit:
                ax.set_facecolor("#ecfdf5")
                for spine in ax.spines.values():
                    spine.set_color("#15803d")
                    spine.set_linewidth(1.1)
            annulus_inner = float(annulus_inner_deg[idx])
            annulus_outer = float(annulus_outer_deg[idx])
            if np.isfinite(annulus_inner) and np.isfinite(annulus_outer) and annulus_outer > annulus_inner:
                for left, right in ((-annulus_outer, -annulus_inner), (annulus_inner, annulus_outer)):
                    clipped_left = max(float(left), x_min)
                    clipped_right = min(float(right), x_max)
                    if clipped_right > clipped_left:
                        ax.axvspan(
                            clipped_left,
                            clipped_right,
                            color="#9ca3af",
                            alpha=0.24,
                            linewidth=0.0,
                            zorder=0.5,
                        )
            profile = np.asarray(profiles[idx], dtype=np.float64)
            finite = profile[np.isfinite(profile)]
            peak = float(np.max(finite)) if finite.size else 0.0
            normalized = profile / peak if peak > 0.0 else np.zeros_like(profile)
            ax.plot(coordinate_centers, normalized, color="#1f5a91", linewidth=0.85)
            ax.axhline(0.0, color="#64748b", linewidth=0.45)
            shown = display_cell_id(internal_id, pred)
            status = "fit" if in_fit else "diag"
            cell_text = f"cell {shown}" if shown is not None else "tail"
            observed_count = int(np.nansum(count_profiles[idx]))
            ax.set_title(f"{cell_text}: {pred} [{status}]\nN_slice={observed_count:,}", fontsize=5.7)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(float(y_min), 1.08)
            ax.tick_params(labelsize=5.8, length=2, labelleft=(j == 0))
            ax.grid(alpha=0.18, linewidth=0.35)
            if j == 0:
                ax.set_ylabel(f"{nhit}\n{quantity} / peak", fontsize=6.2)
            if i == len(ordered_nhit) - 1:
                xlabel = "RA offset cos(dec) (deg)" if projection == "ra" else "Dec offset (deg)"
                ax.set_xlabel(xlabel, fontsize=6.2)

    axis_name = "RA-offset" if projection == "ra" else "Dec-offset"
    fig.suptitle(
        f"Stage D normalized {axis_name} {quantity} profiles {phase}; {slice_label}; "
        "gray = projected Stage D annulus (green panels enter fit)",
        fontsize=11,
        y=0.997,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.978], w_pad=0.35, h_pad=0.55)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_roi_grid(
    values: np.ndarray,
    cell_ids: np.ndarray,
    nhit_bins: np.ndarray,
    pred_bins: np.ndarray,
    xy_edges: np.ndarray,
    r_opt_deg: np.ndarray,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    roi_fiducial_deg: float,
) -> None:
    ordered_nhit = sorted(set(nhit_bins.tolist()), key=interval_key)
    ordered_pred = sorted(set(pred_bins.tolist()), key=interval_key)
    index_by_key = {(nhit, pred): idx for idx, (nhit, pred) in enumerate(zip(nhit_bins, pred_bins))}
    fig, axes = plt.subplots(
        len(ordered_nhit),
        len(ordered_pred),
        figsize=(2.05 * len(ordered_pred), 1.75 * len(ordered_nhit)),
        dpi=150,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = [float(xy_edges[0]), float(xy_edges[-1]), float(xy_edges[0]), float(xy_edges[-1])]
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    fiducial_x = roi_fiducial_deg * np.cos(theta)
    fiducial_y = roi_fiducial_deg * np.sin(theta)
    first_image = None

    for i, nhit in enumerate(ordered_nhit):
        for j, pred in enumerate(ordered_pred):
            ax = axes[i, j]
            idx = index_by_key.get((nhit, pred))
            if idx is None:
                ax.set_axis_off()
                continue
            image = ax.imshow(
                values[idx],
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            if first_image is None:
                first_image = image
            ax.plot(fiducial_x, fiducial_y, color="#222222", linewidth=0.35, alpha=0.8)
            on_radius = float(r_opt_deg[idx])
            ax.plot(on_radius * np.cos(theta), on_radius * np.sin(theta), color="#111111", linewidth=0.45, alpha=0.9)
            ax.scatter([0.0], [0.0], marker="+", s=18, c="#111111", linewidths=0.6)
            shown = display_cell_id(int(cell_ids[idx]), pred)
            ax.set_title(f"cell {shown}: {pred}" if shown is not None else pred, fontsize=6.7)
            ax.tick_params(labelsize=6, length=2)
            if j == 0:
                ax.set_ylabel(f"{nhit}\ny (deg)", fontsize=6.7)
            if i == len(ordered_nhit) - 1:
                ax.set_xlabel("x (deg)", fontsize=6.7)

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.982])
    if first_image is not None:
        colorbar = fig.colorbar(first_image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_counts(stage_f: dict[str, np.ndarray], preferred_model: str, output_path: Path) -> None:
    cell_ids = stage_f["cell_id"]
    pred_bins = stage_f["predE_bin"].astype(str)
    labels = [str(display_cell_id(int(cell_id), pred)) for cell_id, pred in zip(cell_ids, pred_bins)]
    x = np.arange(cell_ids.size, dtype=np.float64) + 1.0
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.errorbar(
        x,
        stage_f["excess"],
        yerr=stage_f["excess_err_conservative"],
        fmt="o",
        color="#222222",
        markersize=4,
        label="Stage E excess",
    )
    ax.plot(x, stage_f["pl_conservative_model_counts"], "-o", color="#1f77b4", markersize=3, label="PL model")
    ax.plot(x, stage_f["logpar_conservative_model_counts"], "-o", color="#d62728", markersize=3, label="LogPar model")
    ax.set_xticks(x, labels)
    ax.set_xlabel("display cell ID (predE >= 6 tail excluded)")
    ax.set_ylabel("counts")
    ax.set_title(f"Stage F model counts vs excess; preferred={preferred_model}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def copy_to_assets(paths: list[Path], asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, asset_dir / path.name)


def main() -> None:
    args = parse_args()
    args.stage_f_output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.stage_f_npz, allow_pickle=False) as data:
        stage_f = {key: np.asarray(data[key]) for key in data.files}
    metadata = json.loads(args.stage_f_metadata.read_text(encoding="utf-8"))
    preferred_model = str((metadata.get("preferred_fit") or {}).get("model", "unknown"))
    model_counts = args.stage_f_output_dir / "model_counts_vs_excess.png"
    plot_model_counts(stage_f, preferred_model, model_counts)
    if args.stage_f_only:
        copy_to_assets([model_counts], args.asset_dir)
        print("Redrew Stage F display-only cell labels: 1-84; predE >= 6 tail unnumbered")
        return

    if args.stage_d_npz is None or args.stage_d_output_dir is None:
        raise ValueError("--stage-d-npz and --stage-d-output-dir are required unless --stage-f-only is set")
    args.stage_d_output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.stage_d_npz, allow_pickle=False) as data:
        stage_d = {key: np.asarray(data[key]) for key in data.files}
    roi_fiducial_deg = float(np.max(np.abs(stage_d["x_edges_deg"])))

    roi_counts = args.stage_d_output_dir / "roi_counts_grid.png"
    roi_excess = args.stage_d_output_dir / "roi_excess_grid.png"
    ra_counts_profile = args.stage_d_output_dir / "normalized_ra_offset_counts_profiles.png"
    dec_counts_profile = args.stage_d_output_dir / "normalized_dec_offset_counts_profiles.png"
    ra_excess_profile = args.stage_d_output_dir / "normalized_ra_offset_excess_profiles.png"
    dec_excess_profile = args.stage_d_output_dir / "normalized_dec_offset_excess_profiles.png"
    annulus_residual = args.stage_d_output_dir / "annulus_residual_grid.png"
    common = (
        stage_d["cell_id"],
        stage_d["nhit_bin"].astype(str),
        stage_d["predE_bin"].astype(str),
        stage_d["x_edges_deg"],
        stage_d["r_opt_deg"],
    )
    plot_counts_grid(
        stage_d["counts_map"],
        *common,
        roi_counts,
        roi_fiducial_deg=roi_fiducial_deg,
    )
    plot_roi_grid(
        stage_d["excess_map"],
        *common,
        roi_excess,
        title="Stage D ROI-local counts minus annulus quadratic surface background",
        colorbar_label="counts - background",
        roi_fiducial_deg=roi_fiducial_deg,
    )
    fit_ids = {int(cell_id) for cell_id in stage_f["cell_id"]}
    # Use the same valid Stage D support before and after subtraction so the
    # comparison changes only the background treatment, not the projection area.
    counts_on_background_support = np.where(
        np.isfinite(stage_d["excess_map"]),
        stage_d["counts_map"],
        np.nan,
    )
    plot_normalized_offset_profiles(
        counts_on_background_support,
        counts_on_background_support,
        stage_d["cell_id"],
        stage_d["nhit_bin"].astype(str),
        stage_d["predE_bin"].astype(str),
        stage_d["x_centers_deg"],
        stage_d["y_centers_deg"],
        stage_d["annulus_inner_deg"],
        stage_d["annulus_outer_deg"],
        fit_ids,
        ra_counts_profile,
        projection="ra",
        quantity="observed-count",
        phase="before background subtraction",
        y_min=-0.05,
        profile_half_width_deg=float(args.profile_half_width_deg),
    )
    plot_normalized_offset_profiles(
        counts_on_background_support,
        counts_on_background_support,
        stage_d["cell_id"],
        stage_d["nhit_bin"].astype(str),
        stage_d["predE_bin"].astype(str),
        stage_d["y_centers_deg"],
        stage_d["x_centers_deg"],
        stage_d["annulus_inner_deg"],
        stage_d["annulus_outer_deg"],
        fit_ids,
        dec_counts_profile,
        projection="dec",
        quantity="observed-count",
        phase="before background subtraction",
        y_min=-0.05,
        profile_half_width_deg=float(args.profile_half_width_deg),
    )
    plot_normalized_offset_profiles(
        stage_d["excess_map"],
        counts_on_background_support,
        stage_d["cell_id"],
        stage_d["nhit_bin"].astype(str),
        stage_d["predE_bin"].astype(str),
        stage_d["x_centers_deg"],
        stage_d["y_centers_deg"],
        stage_d["annulus_inner_deg"],
        stage_d["annulus_outer_deg"],
        fit_ids,
        ra_excess_profile,
        projection="ra",
        quantity="excess",
        phase="after background subtraction",
        y_min=-0.35,
        profile_half_width_deg=float(args.profile_half_width_deg),
    )
    plot_normalized_offset_profiles(
        stage_d["excess_map"],
        counts_on_background_support,
        stage_d["cell_id"],
        stage_d["nhit_bin"].astype(str),
        stage_d["predE_bin"].astype(str),
        stage_d["y_centers_deg"],
        stage_d["x_centers_deg"],
        stage_d["annulus_inner_deg"],
        stage_d["annulus_outer_deg"],
        fit_ids,
        dec_excess_profile,
        projection="dec",
        quantity="excess",
        phase="after background subtraction",
        y_min=-0.35,
        profile_half_width_deg=float(args.profile_half_width_deg),
    )
    plot_roi_grid(
        stage_d["annulus_residual_map"],
        *common,
        annulus_residual,
        title="Stage D annulus fit residuals",
        colorbar_label="annulus residual sigma",
        roi_fiducial_deg=roi_fiducial_deg,
    )
    copy_to_assets(
        [
            roi_counts,
            roi_excess,
            ra_counts_profile,
            dec_counts_profile,
            ra_excess_profile,
            dec_excess_profile,
            annulus_residual,
            model_counts,
        ],
        args.asset_dir,
    )
    print("Redrew display-only cell labels: 1-84; predE >= 6 tail unnumbered")


if __name__ == "__main__":
    main()
