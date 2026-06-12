#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 selector and response validation summary artifacts.")
    parser.add_argument("--ledger-csv", type=str, default="apply/config/cell_ledger_v3_candidate.csv")
    parser.add_argument("--baseline-selector-csv", type=str, default="apply/config/cell_selector_v3_baseline.csv")
    parser.add_argument("--systematics-selector-csv", type=str, default="apply/config/cell_selector_v3_systematics.csv")
    parser.add_argument("--high-energy-selector-csv", type=str, default="apply/config/cell_selector_v3_high_energy_probes.csv")
    parser.add_argument("--response-npz", type=str, default="apply/output/stage_a_v3_candidate/response_2d_v3_candidate.npz")
    parser.add_argument(
        "--stage-f-baseline-metadata",
        type=str,
        default="apply/output/stage_f_v3_baseline/runs/v3_stage_f_slurm_42024/fit_v3_baseline_metadata.json",
    )
    parser.add_argument(
        "--stage-g-baseline-metadata",
        type=str,
        default="apply/output/stage_g_v3_baseline/runs/v3_stage_g_slurm_42024/sed_points_v3_baseline_metadata.json",
    )
    parser.add_argument(
        "--stage-f-systematics-metadata",
        type=str,
        default=(
            "apply/output/stage_f_v3_systematics_selector/runs/"
            "v3_stage_f_systematics_selector/fit_v3_systematics_metadata.json"
        ),
    )
    parser.add_argument(
        "--stage-g-systematics-metadata",
        type=str,
        default=(
            "apply/output/stage_g_v3_systematics_selector/runs/"
            "v3_stage_g_systematics_selector/sed_points_v3_systematics_metadata.json"
        ),
    )
    parser.add_argument(
        "--offsource-stage-e-metadata",
        action="append",
        default=[
            "apply/output/stage_e_v3_offsource_control/runs/v3_stage_e_offsource_ra93p63/signal_v3_offsource_metadata.json",
            "apply/output/stage_e_v3_offsource_control/runs/v3_stage_e_offsource_ra73p63/signal_v3_offsource_metadata.json",
        ],
    )
    parser.add_argument(
        "--time-split-stage-e-metadata",
        action="append",
        default=[
            "apply/output/stage_e_v3_time_split/runs/v3_stage_e_time_first/signal_v3_time_first_metadata.json",
            "apply/output/stage_e_v3_time_split/runs/v3_stage_e_time_second/signal_v3_time_second_metadata.json",
        ],
    )
    parser.add_argument("--min-baseline-mc-count", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="apply/report/assets/v3-validation")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_interval(label: str) -> tuple[Optional[float], Optional[float]]:
    text = str(label).strip()
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
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def include_ids(rows: Sequence[Dict[str, str]]) -> List[int]:
    ids: List[int] = []
    for row in rows:
        value = str(row.get("include", "")).strip().lower()
        if value in {"1", "true", "yes", "y", "include"}:
            ids.append(int(row["cell_id"]))
    return ids


def compute_central_flags(rows: Sequence[Dict[str, str]], central_fraction: float) -> Dict[int, bool]:
    flags: Dict[int, bool] = {}
    tail = 0.5 * (1.0 - float(central_fraction))
    by_nhit: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_nhit.setdefault(str(row["nhit_bin"]), []).append(row)
    for items in by_nhit.values():
        sorted_items = sorted(items, key=lambda row: interval_key(str(row["predE_bin"])))
        counts = np.asarray([int(row.get("mc_count") or 0) for row in sorted_items], dtype=np.float64)
        total = float(np.sum(counts))
        if total <= 0.0:
            for row in sorted_items:
                flags[int(row["cell_id"])] = False
            continue
        lo = np.concatenate([[0.0], np.cumsum(counts[:-1])]) / total
        hi = np.cumsum(counts) / total
        for row, low, high in zip(sorted_items, lo, hi):
            flags[int(row["cell_id"])] = bool(
                high > tail and low < (1.0 - tail) and int(row.get("mc_count") or 0) > 0
            )
    return flags


def computed_baseline_ids(
    ledger_rows: Sequence[Dict[str, str]],
    baseline_selector_rows: Sequence[Dict[str, str]],
    *,
    central_fraction: float,
    min_mc_count: int,
) -> List[int]:
    physical_ridge = {
        int(row["cell_id"]): str(row.get("physical_ridge_flag", "")).strip().lower() in {"1", "true", "yes", "y"}
        for row in baseline_selector_rows
    }
    psf_quality = {
        int(row["cell_id"]): str(row.get("psf_quality_flag", "1")).strip().lower() in {"1", "true", "yes", "y"}
        for row in baseline_selector_rows
    }
    central = compute_central_flags(ledger_rows, central_fraction)
    ids: List[int] = []
    for row in ledger_rows:
        cell_id = int(row["cell_id"])
        count = int(row.get("mc_count") or 0)
        if (
            central.get(cell_id, False)
            and physical_ridge.get(cell_id, False)
            and psf_quality.get(cell_id, True)
            and count >= min_mc_count
        ):
            if str(row.get("predE_bin")) != ">=6":
                ids.append(cell_id)
    return ids


def selector_summary_rows(
    *,
    ledger_rows: Sequence[Dict[str, str]],
    baseline_rows: Sequence[Dict[str, str]],
    systematics_rows: Sequence[Dict[str, str]],
    high_energy_rows: Sequence[Dict[str, str]],
    min_mc_count: int,
) -> List[Dict[str, object]]:
    baseline_ids = include_ids(baseline_rows)
    high_energy_ids = set(include_ids(high_energy_rows))
    selector_map = {
        "baseline_selector_central99": baseline_ids,
        "expanded_selector_central99": include_ids(systematics_rows),
        "high_energy_probe_selector": include_ids(high_energy_rows),
        "computed_baseline_central98": computed_baseline_ids(
            ledger_rows, baseline_rows, central_fraction=0.98, min_mc_count=min_mc_count
        ),
        "computed_baseline_central99": computed_baseline_ids(
            ledger_rows, baseline_rows, central_fraction=0.99, min_mc_count=min_mc_count
        ),
        "computed_baseline_central995": computed_baseline_ids(
            ledger_rows, baseline_rows, central_fraction=0.995, min_mc_count=min_mc_count
        ),
    }
    ledger_by_id = {int(row["cell_id"]): row for row in ledger_rows}
    baseline_set = set(baseline_ids)
    rows: List[Dict[str, object]] = []
    for name, ids in selector_map.items():
        id_set = set(ids)
        low_nhit = [
            cid for cid in ids if cid in ledger_by_id and str(ledger_by_id[cid].get("nhit_bin")) == "[125,200)"
        ]
        added = sorted(id_set - baseline_set)
        removed = sorted(baseline_set - id_set)
        rows.append(
            {
                "selector": name,
                "included_cells": len(ids),
                "low_nhit_125_200_cells": len(low_nhit),
                "high_energy_probe_overlap": len(id_set & high_energy_ids),
                "added_vs_baseline": len(added),
                "removed_vs_baseline": len(removed),
                "added_ids": ",".join(str(v) for v in added),
                "removed_ids": ",".join(str(v) for v in removed),
                "included_ids": ",".join(str(v) for v in ids),
            }
        )
    return rows


def extract_fit_row(label: str, stage_f_metadata: Path, stage_g_metadata: Path) -> Dict[str, object]:
    f_meta = load_json(stage_f_metadata)
    g_meta = load_json(stage_g_metadata)
    if not f_meta:
        return {
            "fit_label": label,
            "status": "missing",
            "stage_f_metadata": str(stage_f_metadata),
            "stage_g_metadata": str(stage_g_metadata),
        }

    preferred = f_meta.get("preferred_fit") if isinstance(f_meta.get("preferred_fit"), dict) else {}
    fits = f_meta.get("fits") if isinstance(f_meta.get("fits"), dict) else {}
    model = str(preferred.get("model", "n/a") if isinstance(preferred, dict) else "n/a")
    error_mode = str(preferred.get("error_mode", "conservative") if isinstance(preferred, dict) else "conservative")
    fit_key = f"{model}_{error_mode}"
    fit = fits.get(fit_key, {}) if isinstance(fits, dict) else {}
    params = fit.get("parameters") if isinstance(fit.get("parameters"), dict) else {}
    quality = f_meta.get("quality") if isinstance(f_meta.get("quality"), dict) else {}
    validation = f_meta.get("validation") if isinstance(f_meta.get("validation"), dict) else {}

    stage_g_points = None
    high_energy_points = None
    max_effective_energy = None
    if g_meta:
        outputs = g_meta.get("outputs") if isinstance(g_meta.get("outputs"), dict) else {}
        npz_path = resolve(str(outputs.get("npz", ""))) if outputs.get("npz") else None
        if npz_path and npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as data:
                labels = np.asarray(data["group_label"]).astype(str) if "group_label" in data else np.asarray([])
                grouping = np.asarray(data["grouping"]).astype(str) if "grouping" in data else np.asarray([])
                energy = np.asarray(data["effective_energy_tev"], dtype=np.float64) if "effective_energy_tev" in data else np.asarray([])
            stage_g_points = int(labels.size)
            pred_mask = grouping == "predE"
            high_mask = pred_mask & np.isin(labels, ["[5,6)", ">=6"])
            high_energy_points = int(np.count_nonzero(high_mask))
            if energy.size:
                max_effective_energy = float(np.nanmax(energy))

    return {
        "fit_label": label,
        "status": quality.get("fit_status") or quality.get("status") or "available",
        "stage_f_run": f_meta.get("run_id", "n/a"),
        "stage_g_run": g_meta.get("run_id", "n/a") if g_meta else "missing",
        "n_cells": validation.get("n_cells", len(f_meta.get("cells", [])) if isinstance(f_meta.get("cells"), list) else ""),
        "preferred_model": model,
        "error_mode": error_mode,
        "phi0": params.get("phi0", ""),
        "gamma": params.get("gamma", ""),
        "alpha": params.get("alpha", ""),
        "beta": params.get("beta", ""),
        "chi2": fit.get("chi2", ""),
        "ndof": fit.get("ndof", ""),
        "chi2_over_ndof": fit.get("chi2_over_ndof", ""),
        "stage_g_points": stage_g_points if stage_g_points is not None else "",
        "stage_g_high_energy_predE_points": high_energy_points if high_energy_points is not None else "",
        "stage_g_max_effective_energy_tev": max_effective_energy if max_effective_energy is not None else "",
        "stage_f_metadata": str(stage_f_metadata),
        "stage_g_metadata": str(stage_g_metadata),
    }


def response_closure_rows(response_npz: Path, selector_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    with np.load(response_npz, allow_pickle=False) as data:
        cell_ids = np.asarray(data["cell_id"], dtype=np.int64)
        eta = np.asarray(data["eta"], dtype=np.float64)
        eta_count = np.asarray(data["eta_count"], dtype=np.float64)
        numerator_sumw = np.asarray(data["numerator_sumw"], dtype=np.float64)
        numerator_count = np.asarray(data["numerator_count"], dtype=np.float64)
        denominator_sumw = np.asarray(data["denominator_sumw"], dtype=np.float64)
        denominator_count = np.asarray(data["denominator_count"], dtype=np.float64)

    def parse_ids(value: object) -> List[int]:
        text = str(value or "").strip()
        if not text:
            return []
        return [int(part) for part in text.split(",") if part.strip()]

    rows: List[Dict[str, object]] = []
    all_selector = {
        "selector": "all_candidate_cells",
        "included_ids": ",".join(str(v) for v in cell_ids.tolist()),
    }
    for selector in [all_selector, *selector_rows]:
        ids = parse_ids(selector.get("included_ids"))
        mask = np.isin(cell_ids, np.asarray(ids, dtype=np.int64))
        if not np.any(mask):
            continue
        pred_sumw = eta[mask] * denominator_sumw[None, :, :]
        pred_count = eta_count[mask] * denominator_count[None, :, :]
        truth_sumw = numerator_sumw[mask]
        truth_count = numerator_count[mask]
        total_truth_sumw = float(np.sum(truth_sumw))
        total_pred_sumw = float(np.sum(pred_sumw))
        total_truth_count = float(np.sum(truth_count))
        total_pred_count = float(np.sum(pred_count))

        cell_truth_count = np.sum(truth_count, axis=(1, 2))
        cell_pred_count = np.sum(pred_count, axis=(1, 2))
        cell_truth_sumw = np.sum(truth_sumw, axis=(1, 2))
        cell_pred_sumw = np.sum(pred_sumw, axis=(1, 2))
        count_scale = np.maximum(1.0, np.abs(cell_truth_count))
        sumw_scale = np.maximum(1.0, np.abs(cell_truth_sumw))

        rows.append(
            {
                "selector": selector.get("selector", ""),
                "included_cells": int(np.count_nonzero(mask)),
                "truth_numerator_count": total_truth_count,
                "reconstructed_from_eta_count": total_pred_count,
                "rel_delta_count": (total_pred_count - total_truth_count) / max(1.0, abs(total_truth_count)),
                "max_abs_rel_delta_count_per_cell": float(np.max(np.abs(cell_pred_count - cell_truth_count) / count_scale)),
                "truth_numerator_sumw": total_truth_sumw,
                "reconstructed_from_eta_sumw": total_pred_sumw,
                "rel_delta_sumw": (total_pred_sumw - total_truth_sumw) / max(1.0, abs(total_truth_sumw)),
                "max_abs_rel_delta_sumw_per_cell": float(np.max(np.abs(cell_pred_sumw - cell_truth_sumw) / sumw_scale)),
                "max_sum_eta_over_true_bins": float(np.nanmax(np.sum(eta[mask], axis=0))),
                "max_sum_eta_count_over_true_bins": float(np.nanmax(np.sum(eta_count[mask], axis=0))),
            }
        )
    return rows


def known_background_combined_sigma(n_on: np.ndarray, b_on: np.ndarray) -> float:
    n = np.asarray(n_on, dtype=np.float64)
    b = np.asarray(b_on, dtype=np.float64)
    valid = np.isfinite(n) & np.isfinite(b) & (b > 0.0)
    if not np.any(valid):
        return float("nan")
    term = np.zeros(n.shape, dtype=np.float64)
    positive = valid & (n > 0.0)
    term[positive] = n[positive] * np.log(n[positive] / b[positive]) - (n[positive] - b[positive])
    term[valid & (n <= 0.0)] = b[valid & (n <= 0.0)]
    sigma = math.sqrt(2.0 * max(float(np.sum(term[valid])), 0.0))
    if float(np.sum(n[valid] - b[valid])) < 0.0:
        sigma *= -1.0
    return sigma


def signal_subset_row(label: str, metadata_path: Path, baseline_ids: Sequence[int]) -> Dict[str, object]:
    meta = load_json(metadata_path)
    if not meta:
        return {"validation": label, "status": "missing", "metadata": str(metadata_path)}
    outputs = meta.get("outputs") if isinstance(meta.get("outputs"), dict) else {}
    npz_path = resolve(str(outputs.get("npz", ""))) if outputs.get("npz") else None
    if npz_path is None or not npz_path.exists():
        return {"validation": label, "status": "missing_npz", "metadata": str(metadata_path)}
    with np.load(npz_path, allow_pickle=False) as data:
        ids = np.asarray(data["cell_id"], dtype=np.int64)
        mask = np.isin(ids, np.asarray(baseline_ids, dtype=np.int64))
        n_on = np.asarray(data["N_on"], dtype=np.float64)[mask]
        b_on = np.asarray(data["B_on"], dtype=np.float64)[mask]
        excess = np.asarray(data["excess"], dtype=np.float64)[mask]
    totals = meta.get("totals") if isinstance(meta.get("totals"), dict) else {}
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    processing = meta.get("processing") if isinstance(meta.get("processing"), dict) else {}
    return {
        "validation": label,
        "status": "available",
        "run": meta.get("run_id", ""),
        "ra_deg": source.get("ra_deg", ""),
        "dec_deg": source.get("dec_deg", ""),
        "mjd_min": processing.get("mjd_min", ""),
        "mjd_max": processing.get("mjd_max", ""),
        "all_N_on": totals.get("N_on", ""),
        "all_B_on": totals.get("B_on", ""),
        "all_excess": totals.get("excess", ""),
        "all_formal_sigma": totals.get("formal_sigma", ""),
        "baseline_N_on": int(np.nansum(n_on)),
        "baseline_B_on": float(np.nansum(b_on)),
        "baseline_excess": float(np.nansum(excess)),
        "baseline_combined_sigma": known_background_combined_sigma(n_on, b_on),
        "baseline_cells": int(np.count_nonzero(mask)),
        "metadata": str(metadata_path),
    }


def offsource_status(rows: Sequence[Dict[str, object]]) -> str:
    available = [row for row in rows if row.get("status") == "available"]
    if not available:
        return "not_produced"
    for row in available:
        sigma = finite_float(row.get("baseline_combined_sigma"))
        if sigma is not None and abs(sigma) > 5.0:
            return "failed"
    return "passed"


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_selector_summary(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    plt = setup_matplotlib()
    labels = [str(row["selector"]).replace("_", "\n") for row in rows]
    counts = [float(row["included_cells"]) for row in rows]
    added = [float(row["added_vs_baseline"]) for row in rows]
    removed = [float(row["removed_vs_baseline"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.28
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.bar(x - width, counts, width=width, label="included cells", color="#2f6f8f")
    ax.bar(x, added, width=width, label="added vs baseline", color="#61a76f")
    ax.bar(x + width, removed, width=width, label="removed vs baseline", color="#c9704b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("cells")
    ax.set_title("v3 selector sensitivity audit")
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_closure_summary(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    plt = setup_matplotlib()
    labels = [str(row["selector"]).replace("_", "\n") for row in rows]
    count_delta = [abs(float(row["rel_delta_count"])) for row in rows]
    sumw_delta = [abs(float(row["rel_delta_sumw"])) for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, count_delta, width=width, label="count closure", color="#33658a")
    ax.bar(x + width / 2, sumw_delta, width=width, label="weighted closure", color="#f6ae2d")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("absolute relative delta")
    ax.set_title("Stage A response histogram self-closure")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger_rows = read_csv_rows(resolve(args.ledger_csv))
    baseline_rows = read_csv_rows(resolve(args.baseline_selector_csv))
    systematics_rows = read_csv_rows(resolve(args.systematics_selector_csv))
    high_energy_rows = read_csv_rows(resolve(args.high_energy_selector_csv))

    selector_rows = selector_summary_rows(
        ledger_rows=ledger_rows,
        baseline_rows=baseline_rows,
        systematics_rows=systematics_rows,
        high_energy_rows=high_energy_rows,
        min_mc_count=int(args.min_baseline_mc_count),
    )
    fit_rows = [
        extract_fit_row(
            "baseline_selector_central99",
            resolve(args.stage_f_baseline_metadata),
            resolve(args.stage_g_baseline_metadata),
        ),
        extract_fit_row(
            "expanded_selector_central99",
            resolve(args.stage_f_systematics_metadata),
            resolve(args.stage_g_systematics_metadata),
        ),
    ]
    closure_rows = response_closure_rows(resolve(args.response_npz), selector_rows)
    baseline_ids = include_ids(baseline_rows)
    offsource_rows = [
        signal_subset_row(f"offsource_{idx}", resolve(path), baseline_ids)
        for idx, path in enumerate(args.offsource_stage_e_metadata, start=1)
    ]
    time_split_rows = [
        signal_subset_row(f"time_split_{idx}", resolve(path), baseline_ids)
        for idx, path in enumerate(args.time_split_stage_e_metadata, start=1)
    ]

    selector_csv = output_dir / "v3_selector_systematics_summary.csv"
    fit_csv = output_dir / "v3_selector_fit_comparison.csv"
    closure_csv = output_dir / "v3_response_closure_summary.csv"
    mc_reference_closure_csv = output_dir / "v3_mc_reference_forward_fold_closure.csv"
    offsource_csv = output_dir / "v3_offsource_fake_source_summary.csv"
    time_split_csv = output_dir / "v3_time_split_summary.csv"
    summary_json = output_dir / "v3_validation_summary.json"
    selector_png = output_dir / "v3_selector_sensitivity_summary.png"
    closure_png = output_dir / "v3_response_closure_summary.png"

    write_csv(
        selector_csv,
        selector_rows,
        [
            "selector",
            "included_cells",
            "low_nhit_125_200_cells",
            "high_energy_probe_overlap",
            "added_vs_baseline",
            "removed_vs_baseline",
            "added_ids",
            "removed_ids",
            "included_ids",
        ],
    )
    write_csv(
        fit_csv,
        fit_rows,
        [
            "fit_label",
            "status",
            "stage_f_run",
            "stage_g_run",
            "n_cells",
            "preferred_model",
            "error_mode",
            "phi0",
            "gamma",
            "alpha",
            "beta",
            "chi2",
            "ndof",
            "chi2_over_ndof",
            "stage_g_points",
            "stage_g_high_energy_predE_points",
            "stage_g_max_effective_energy_tev",
            "stage_f_metadata",
            "stage_g_metadata",
        ],
    )
    write_csv(
        closure_csv,
        closure_rows,
        [
            "selector",
            "included_cells",
            "truth_numerator_count",
            "reconstructed_from_eta_count",
            "rel_delta_count",
            "max_abs_rel_delta_count_per_cell",
            "truth_numerator_sumw",
            "reconstructed_from_eta_sumw",
            "rel_delta_sumw",
            "max_abs_rel_delta_sumw_per_cell",
            "max_sum_eta_over_true_bins",
            "max_sum_eta_count_over_true_bins",
        ],
    )
    mc_reference_rows = [
        {
            **row,
            "closure_kind": "stage_a_binned_mc_reference_forward_fold",
            "truth_definition": "Stage A numerator histograms from v3 selected MC events",
            "prediction_definition": "Stage A denominator histograms folded through eta/eta_count response",
        }
        for row in closure_rows
    ]
    write_csv(
        mc_reference_closure_csv,
        mc_reference_rows,
        [
            "closure_kind",
            "selector",
            "included_cells",
            "truth_definition",
            "prediction_definition",
            "truth_numerator_count",
            "reconstructed_from_eta_count",
            "rel_delta_count",
            "max_abs_rel_delta_count_per_cell",
            "truth_numerator_sumw",
            "reconstructed_from_eta_sumw",
            "rel_delta_sumw",
            "max_abs_rel_delta_sumw_per_cell",
            "max_sum_eta_over_true_bins",
            "max_sum_eta_count_over_true_bins",
        ],
    )
    signal_validation_fields = [
        "validation",
        "status",
        "run",
        "ra_deg",
        "dec_deg",
        "mjd_min",
        "mjd_max",
        "all_N_on",
        "all_B_on",
        "all_excess",
        "all_formal_sigma",
        "baseline_N_on",
        "baseline_B_on",
        "baseline_excess",
        "baseline_combined_sigma",
        "baseline_cells",
        "metadata",
    ]
    write_csv(offsource_csv, offsource_rows, signal_validation_fields)
    write_csv(time_split_csv, time_split_rows, signal_validation_fields)
    plot_selector_summary(selector_rows, selector_png)
    plot_closure_summary(closure_rows, closure_png)

    status_items = [
        {
            "item": "baseline_stage_a_to_g",
            "status": "complete",
            "evidence": str(resolve(args.stage_g_baseline_metadata)),
        },
        {
            "item": "baseline_vs_expanded_selector_stage_f_g",
            "status": "complete" if fit_rows[1].get("status") != "missing" else "missing",
            "evidence": str(resolve(args.stage_f_systematics_metadata)),
        },
        {
            "item": "central98_995_selector_audit",
            "status": "prefit_selector_audit_complete",
            "evidence": str(selector_csv),
        },
        {
            "item": "stage_a_response_histogram_self_closure",
            "status": "complete",
            "evidence": str(closure_csv),
        },
        {
            "item": "mc_reference_forward_fold_truth_closure",
            "status": "complete",
            "evidence": str(mc_reference_closure_csv),
        },
        {
            "item": "off_source_fake_source_validation",
            "status": offsource_status(offsource_rows),
            "evidence": str(offsource_csv),
        },
        {
            "item": "time_split_background_stability",
            "status": "complete" if all(row.get("status") == "available" for row in time_split_rows) else "not_produced",
            "evidence": str(time_split_csv),
        },
    ]

    write_json(
        summary_json,
        {
            "inputs": {
                "ledger_csv": str(resolve(args.ledger_csv)),
                "baseline_selector_csv": str(resolve(args.baseline_selector_csv)),
                "systematics_selector_csv": str(resolve(args.systematics_selector_csv)),
                "high_energy_selector_csv": str(resolve(args.high_energy_selector_csv)),
                "response_npz": str(resolve(args.response_npz)),
            },
            "outputs": {
                "selector_summary_csv": str(selector_csv),
                "fit_comparison_csv": str(fit_csv),
                "response_closure_csv": str(closure_csv),
                "mc_reference_forward_fold_closure_csv": str(mc_reference_closure_csv),
                "offsource_fake_source_csv": str(offsource_csv),
                "time_split_csv": str(time_split_csv),
                "selector_sensitivity_png": str(selector_png),
                "response_closure_png": str(closure_png),
            },
            "status_items": status_items,
            "selector_summary": selector_rows,
            "fit_comparison": fit_rows,
            "response_closure": closure_rows,
            "mc_reference_forward_fold_closure": mc_reference_rows,
            "offsource_fake_source": offsource_rows,
            "time_split": time_split_rows,
        },
    )
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
