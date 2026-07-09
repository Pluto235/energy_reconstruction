#!/usr/bin/env python
import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import uproot

from src.common.EdgeConv import process_features
from src.common.hit_coordinate_transform import build_hit_points
from src.theta.ParticleDataset_theta import _safe_mean_std
from src.theta.ParticleRegressor_theta import ParticleNetRegressor


PROCESSING_CONDITIONS_WITH_NAMES = [
    {"name": "vx", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
    {"name": "vy", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
    {"name": "vt", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
    {"name": "vq", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
]
PROCESSING_CONDITIONS = [{k: v for k, v in item.items() if k != "name"} for item in PROCESSING_CONDITIONS_WITH_NAMES]
MODEL_BRANCHES = [item["name"] for item in PROCESSING_CONDITIONS_WITH_NAMES]

NHIT_FORMAL_BINS: List[Tuple[int, int]] = [
    (30, 60),
    (60, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 800),
    (800, 1100),
    (1100, 2000),
]

PRED_LOGE_BINS: List[Tuple[Optional[float], Optional[float], str]] = [
    (None, 2.0, "<2"),
    (2.0, 3.0, "[2,3)"),
    (3.0, 3.25, "[3,3.25)"),
    (3.25, 3.5, "[3.25,3.5)"),
    (3.5, 3.75, "[3.5,3.75)"),
    (3.75, 4.0, "[3.75,4.0)"),
    (4.0, 4.25, "[4.0,4.25)"),
    (4.25, 4.5, "[4.25,4.5)"),
    (4.5, 4.75, "[4.5,4.75)"),
    (4.75, 5.0, "[4.75,5.0)"),
    (5.0, None, ">=5"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all-event inference on WCDA simulation files, then bin events by nhit and predicted log10(E/GeV)."
    )
    parser.add_argument("--input-root", type=str, default="/mnt/mydisk/WCDA_simulation")
    parser.add_argument("--run-dir", type=str, default="/home/server/projects/energy_reconstruction/runs/no_core_cut_2724")
    parser.add_argument("--output-root", type=str, default="/mnt/mydisk/WCDA_simulation_binned")
    parser.add_argument("--tree-name", type=str, default="t_eventout")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-files", type=int, default=None, help="Only process the first N ROOT files. Useful for dry-run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU ids for multi-GPU sharding, e.g. 0,1,2,3")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--summary-dirname", type=str, default="summary")
    parser.add_argument("--reader-workers", type=int, default=0, help="CPU workers for file-level read/preprocess prefetch. 0 keeps sequential mode.")
    parser.add_argument("--prefetch-files", type=int, default=None, help="Maximum number of prepared files kept inflight. Defaults to 2 * reader_workers.")
    parser.add_argument("--reader-backend", type=str, default="thread", choices=["thread", "process"], help="Parallel backend for file prefetch/preprocess.")
    parser.add_argument("--apply-event-cuts", action="store_true", default=False, help="Apply event-level cuts before inference.")
    parser.add_argument("--cut-pinc-max", type=float, default=1.1)
    parser.add_argument("--cut-dangle-max-deg", type=float, default=3.0)
    parser.add_argument("--no-cut-dangle", action="store_true", default=False, help="Disable the MC truth dangle cut.")
    parser.add_argument("--cut-theta-max-deg", type=float, default=30.0)
    parser.add_argument("--cut-fitstat-equals", type=int, default=0)
    parser.add_argument("--cut-dcedge-min", type=float, default=None, help="Optional reconstructed core edge-distance cut.")
    parser.add_argument(
        "--core-box",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=None,
        help="Optional reconstructed core box cut on xc/yc.",
    )
    parser.add_argument(
        "--keep-nhit-bins",
        type=str,
        default=None,
        help="Comma-separated nhit labels to keep on disk and pass through inference, e.g. '>=2000'. "
        "If omitted, all nhit bins are processed as before.",
    )
    return parser.parse_args()


def choose_device(device_arg: str, gpu_id: int) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{gpu_id}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def parse_gpu_ids(gpu_ids_arg: Optional[str], fallback_gpu_id: int, device_arg: str) -> List[int]:
    if device_arg == "cpu":
        return []
    if gpu_ids_arg is None or gpu_ids_arg.strip() == "":
        return [int(fallback_gpu_id)]
    gpu_ids = []
    for part in gpu_ids_arg.split(","):
        part = part.strip()
        if not part:
            continue
        gpu_ids.append(int(part))
    if not gpu_ids:
        raise ValueError("`--gpu-ids` was provided but no valid GPU id was parsed.")
    return gpu_ids


def load_run_config(run_dir: str) -> Dict[str, object]:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(config: Dict[str, object], checkpoint_path: str, device: torch.device) -> ParticleNetRegressor:
    model = ParticleNetRegressor(
        input_dims=2,
        conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
        fc_params=[(256, 0.1), (128, 0.1)],
        use_fusion=True,
        theta_embed_dim=int(config.get("theta_embed_dim", 16)),
        theta_embed_dropout=float(config.get("theta_embed_dropout", 0.0)),
        core_embed_dim=int(config.get("core_embed_dim", 0)),
        core_embed_dropout=float(config.get("core_embed_dropout", 0.0)),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def select_hits(
    points: np.ndarray,
    vq: np.ndarray,
    vt: np.ndarray,
    *,
    max_points: int,
    sample_mode: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = int(points.shape[0])
    if n_points <= max_points:
        return points, vq, vt

    k = max_points
    if sample_mode == "random":
        idxs = rng.choice(n_points, k, replace=False)
    elif sample_mode == "firstk":
        idxs = np.arange(k)
    elif sample_mode == "topk_q":
        idxs = np.argsort(vq)[-k:]
    elif sample_mode == "weighted_q":
        w = np.clip(vq, a_min=0.0, a_max=None)
        s = float(np.sum(w))
        if s <= 0:
            idxs = rng.choice(n_points, k, replace=False)
        else:
            idxs = rng.choice(n_points, k, replace=False, p=w / s)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    return points[idxs], vq[idxs], vt[idxs]


def normalize_features(vq: np.ndarray, vt: np.ndarray, norm_mode: str) -> np.ndarray:
    if norm_mode == "none":
        return np.column_stack([vq, vt]).astype(np.float32)
    if norm_mode == "per_event":
        vq_m, vq_s = _safe_mean_std(vq)
        vt_m, vt_s = _safe_mean_std(vt)
        vq_n = (vq - vq_m) / vq_s
        vt_n = (vt - vt_m) / vt_s
        return np.column_stack([vq_n, vt_n]).astype(np.float32)
    raise ValueError(
        "This apply script currently supports the run's original per_event/none normalization paths. "
        f"Got norm_mode={norm_mode!r}."
    )


def preprocess_event(
    arrays: Dict[str, np.ndarray],
    event_idx: int,
    *,
    max_points: int,
    sample_mode: str,
    norm_mode: str,
    core_scale: Tuple[float, float],
    rng: np.random.Generator,
) -> Optional[Dict[str, np.ndarray]]:
    raw_features = np.column_stack([arrays[name][event_idx] for name in MODEL_BRANCHES])
    processed = process_features(raw_features, PROCESSING_CONDITIONS)
    if processed.shape[0] == 0:
        return None

    vt = processed[:, 2].astype(np.float32)
    vq = processed[:, 3].astype(np.float32)
    detector_ids = np.asarray(arrays["vidmc"][event_idx], dtype=np.int64)
    points = build_hit_points(
        processed,
        detector_ids=detector_ids,
        coordinate_system="global",
    )

    points, vq, vt = select_hits(
        points,
        vq,
        vt,
        max_points=max_points,
        sample_mode=sample_mode,
        rng=rng,
    )
    features = normalize_features(vq, vt, norm_mode=norm_mode)
    n_points = int(points.shape[0])

    if n_points < max_points:
        pad_len = max_points - n_points
        points = np.vstack([points, np.zeros((pad_len, 2), dtype=np.float32)]).astype(np.float32)
        features = np.vstack([features, np.zeros((pad_len, 2), dtype=np.float32)]).astype(np.float32)
        mask = np.concatenate([np.ones(n_points, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
    else:
        mask = np.ones(max_points, dtype=np.float32)

    theta = float(arrays["theta"][event_idx])
    costheta = np.float32(math.cos(theta))
    reco_core_xy = np.array(
        [
            float(arrays["xc"][event_idx]) / float(core_scale[0]),
            float(arrays["yc"][event_idx]) / float(core_scale[1]),
        ],
        dtype=np.float32,
    )

    return {
        "points": points.T.astype(np.float32),
        "features": features.T.astype(np.float32),
        "mask": mask.reshape(1, max_points).astype(np.float32),
        "costheta": np.array([costheta], dtype=np.float32),
        "reco_core_xy": reco_core_xy,
    }


def infer_batch(
    model: ParticleNetRegressor,
    device: torch.device,
    batch_points: np.ndarray,
    batch_features: np.ndarray,
    batch_mask: np.ndarray,
    batch_costheta: np.ndarray,
    batch_reco_core_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    points = torch.from_numpy(batch_points).to(device, non_blocking=True)
    features = torch.from_numpy(batch_features).to(device, non_blocking=True)
    mask = torch.from_numpy(batch_mask).to(device, non_blocking=True)
    costheta = torch.from_numpy(batch_costheta).to(device, non_blocking=True)
    reco_core_xy = torch.from_numpy(batch_reco_core_xy).to(device, non_blocking=True) if batch_reco_core_xy is not None else None

    with torch.no_grad():
        pred = model(points, features, mask, costheta, reco_core_xy)
    return pred.detach().cpu().numpy().reshape(-1)


def get_nhit_value(arrays: Dict[str, np.ndarray], event_idx: int) -> int:
    if "nv" in arrays:
        return int(arrays["nv"][event_idx])
    return int(len(arrays["vx"][event_idx]))


def nhit_bin_label(nhit: int) -> Tuple[str, bool]:
    if nhit < 30:
        return "<30", False
    if nhit >= 2000:
        return ">=2000", False
    for low, high in NHIT_FORMAL_BINS:
        if low <= nhit < high:
            return f"[{low},{high})", True
    return "unclassified", False


def pred_bin_label(loge_pred: float) -> str:
    for low, high, label in PRED_LOGE_BINS:
        if low is None and loge_pred < high:
            return label
        if high is None and loge_pred >= low:
            return label
        if low is not None and high is not None and low <= loge_pred < high:
            return label
    return "unclassified"


def sanitize_label(label: str) -> str:
    return (
        label.replace(">=", "ge_")
        .replace("<", "lt_")
        .replace("[", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def stats_level(count: int) -> str:
    if count < 100:
        return "very low statistics"
    if count < 1000:
        return "low statistics"
    return "acceptable"


def all_summary_rows(bin_counts: Dict[Tuple[str, str], int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    nhit_labels = ["<30"] + [f"[{low},{high})" for low, high in NHIT_FORMAL_BINS] + [">=2000"]
    pred_labels = [label for _, _, label in PRED_LOGE_BINS]
    for nhit_label in nhit_labels:
        for pred_label in pred_labels:
            count = int(bin_counts.get((nhit_label, pred_label), 0))
            is_formal = nhit_label not in {"<30", ">=2000"}
            rows.append(
                {
                    "nhit_bin": nhit_label,
                    "predE_bin": pred_label,
                    "count": count,
                    "formal_nhit_bin": is_formal,
                    "statistics_level": stats_level(count),
                }
            )
    return rows


def write_summary(
    summary_dir: str,
    bin_counts: Dict[Tuple[str, str], int],
    *,
    total_events: int,
    inferred_events: int,
    out_of_range_events: int,
    per_file_counts: Dict[str, int],
    run_metadata: Dict[str, object],
) -> None:
    os.makedirs(summary_dir, exist_ok=True)
    rows = all_summary_rows(bin_counts)

    csv_path = os.path.join(summary_dir, "bin_counts.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nhit_bin", "predE_bin", "count", "formal_nhit_bin", "statistics_level"],
        )
        writer.writeheader()
        writer.writerows(rows)

    low_stats_rows = [row for row in rows if row["count"] > 0 and row["statistics_level"] != "acceptable"]
    md_path = os.path.join(summary_dir, "bin_counts.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Bin Counts\n\n")
        f.write(f"- Total events: {total_events}\n")
        f.write(f"- Successfully inferred events: {inferred_events}\n")
        f.write(f"- Events outside formal nhit range (<30 or >=2000): {out_of_range_events}\n")
        f.write(f"- Processed ROOT files: {len(per_file_counts)}\n\n")
        f.write("| nhit_bin | predE_bin | count | formal_nhit_bin | statistics_level |\n")
        f.write("| --- | --- | ---: | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['nhit_bin']} | {row['predE_bin']} | {row['count']} | "
                f"{row['formal_nhit_bin']} | {row['statistics_level']} |\n"
            )

        f.write("\n## Low-Statistics Bins\n\n")
        if low_stats_rows:
            for row in low_stats_rows:
                f.write(
                    f"- nhit `{row['nhit_bin']}`, predE `{row['predE_bin']}`: "
                    f"{row['count']} events, {row['statistics_level']}\n"
                )
        else:
            f.write("- None\n")

    meta_path = os.path.join(summary_dir, "run_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_metadata": run_metadata,
                "total_events": total_events,
                "inferred_events": inferred_events,
                "out_of_range_events": out_of_range_events,
                "processed_files": len(per_file_counts),
                "per_file_event_counts": per_file_counts,
                "bin_counts": {
                    f"{nhit_label}__{pred_label}": int(count)
                    for (nhit_label, pred_label), count in sorted(bin_counts.items())
                },
            },
            f,
            indent=2,
        )


def flush_file_outputs(
    source_path: str,
    arrays: Dict[str, np.ndarray],
    grouped_indices: Dict[Tuple[str, str], List[int]],
    grouped_preds: Dict[Tuple[str, str], List[float]],
    output_root: str,
    tree_name: str,
) -> None:
    base_name = Path(source_path).stem
    for bin_key, indices in grouped_indices.items():
        if not indices:
            continue
        nhit_label, pred_label = bin_key
        out_dir = os.path.join(
            output_root,
            f"nhit_{sanitize_label(nhit_label)}",
            f"predE_{sanitize_label(pred_label)}",
        )
        os.makedirs(out_dir, exist_ok=True)

        sel = np.asarray(indices, dtype=np.int64)
        payload = {name: arrays[name][sel] for name in arrays.keys()}
        payload["ml_logE_pred"] = np.asarray(grouped_preds[bin_key], dtype=np.float32)
        payload["nhit_bin"] = np.asarray([nhit_label] * len(indices), dtype=object)
        payload["predE_bin"] = np.asarray([pred_label] * len(indices), dtype=object)

        out_path = os.path.join(out_dir, f"{base_name}.root")
        with uproot.recreate(out_path) as fout:
            fout[tree_name] = payload


def discover_root_files(input_root: str, max_files: Optional[int]) -> List[str]:
    files = [
        os.path.join(input_root, name)
        for name in os.listdir(input_root)
        if name.endswith(".root") and os.path.isfile(os.path.join(input_root, name))
    ]
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return files


def _stack_or_empty(chunks: List[np.ndarray], shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if chunks:
        return np.stack(chunks, axis=0).astype(dtype, copy=False)
    return np.empty(shape, dtype=dtype)


def filter_event_indices(
    arrays: Dict[str, np.ndarray],
    *,
    apply_event_cuts: bool,
    cut_pinc_max: float,
    cut_dangle_max_deg: Optional[float],
    cut_theta_max_deg: float,
    cut_fitstat_equals: int,
    cut_dcedge_min: Optional[float],
    core_box: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    n_events = len(next(iter(arrays.values())) if arrays else [])
    if not apply_event_cuts or n_events == 0:
        return np.arange(n_events, dtype=np.int64)

    required = ["pincness", "fitstat", "theta"]
    if cut_dangle_max_deg is not None:
        required.append("mc_dangle")
    if cut_dcedge_min is not None:
        required.append("dcedge")
    if core_box is not None:
        required.extend(["xc", "yc"])
    missing = [name for name in required if name not in arrays]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Event cuts requested, but required branches are missing: {missing_str}")

    mask = np.ones(n_events, dtype=bool)
    mask &= np.asarray(arrays["pincness"] < float(cut_pinc_max))
    mask &= np.asarray(arrays["fitstat"] == int(cut_fitstat_equals))
    mask &= np.asarray(arrays["theta"] < (float(cut_theta_max_deg) * np.pi / 180.0))
    if cut_dangle_max_deg is not None:
        mask &= np.asarray(arrays["mc_dangle"] < (float(cut_dangle_max_deg) * np.pi / 180.0))
    if cut_dcedge_min is not None:
        mask &= np.asarray(arrays["dcedge"] > float(cut_dcedge_min))
    if core_box is not None:
        x_min, x_max, y_min, y_max = core_box
        mask &= np.asarray(arrays["xc"] >= float(x_min))
        mask &= np.asarray(arrays["xc"] <= float(x_max))
        mask &= np.asarray(arrays["yc"] >= float(y_min))
        mask &= np.asarray(arrays["yc"] <= float(y_max))
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def load_and_prepare_file(
    file_path: str,
    *,
    tree_name: str,
    max_points: int,
    sample_mode: str,
    norm_mode: str,
    seed: int,
    apply_event_cuts: bool,
    cut_pinc_max: float,
    cut_dangle_max_deg: Optional[float],
    cut_theta_max_deg: float,
    cut_fitstat_equals: int,
    cut_dcedge_min: Optional[float],
    core_box: Optional[Tuple[float, float, float, float]],
    core_scale: Tuple[float, float],
    keep_nhit_bins: Optional[set] = None,
) -> Dict[str, object]:
    with uproot.open(file_path) as f:
        tree = f[f"{tree_name};1"] if f"{tree_name};1" in f else f[tree_name]
        arrays = tree.arrays(list(tree.keys()), library="np")

    n_events = len(next(iter(arrays.values())) if arrays else [])
    selected_event_indices = filter_event_indices(
        arrays,
        apply_event_cuts=apply_event_cuts,
        cut_pinc_max=cut_pinc_max,
        cut_dangle_max_deg=cut_dangle_max_deg,
        cut_theta_max_deg=cut_theta_max_deg,
        cut_fitstat_equals=cut_fitstat_equals,
        cut_dcedge_min=cut_dcedge_min,
        core_box=core_box,
    )
    rng = np.random.default_rng(seed)

    prepared_points: List[np.ndarray] = []
    prepared_features: List[np.ndarray] = []
    prepared_mask: List[np.ndarray] = []
    prepared_costheta: List[np.ndarray] = []
    prepared_reco_core: List[np.ndarray] = []
    event_indices: List[int] = []
    nhit_labels: List[str] = []
    is_formal_flags: List[bool] = []

    for event_idx in selected_event_indices.tolist():
        nhit_value = get_nhit_value(arrays, event_idx)
        nhit_label, is_formal_bin = nhit_bin_label(nhit_value)
        if keep_nhit_bins is not None and nhit_label not in keep_nhit_bins:
            continue

        preprocessed = preprocess_event(
            arrays,
            event_idx,
            max_points=max_points,
            sample_mode=sample_mode,
            norm_mode=norm_mode,
            core_scale=core_scale,
            rng=rng,
        )
        if preprocessed is None:
            continue

        prepared_points.append(preprocessed["points"])
        prepared_features.append(preprocessed["features"])
        prepared_mask.append(preprocessed["mask"])
        prepared_costheta.append(preprocessed["costheta"])
        prepared_reco_core.append(preprocessed["reco_core_xy"])
        event_indices.append(int(event_idx))
        nhit_labels.append(nhit_label)
        is_formal_flags.append(bool(is_formal_bin))

    return {
        "file_path": file_path,
        "arrays": arrays,
        "total_events": int(n_events),
        "points": _stack_or_empty(prepared_points, (0, 2, max_points), np.float32),
        "features": _stack_or_empty(prepared_features, (0, 2, max_points), np.float32),
        "mask": _stack_or_empty(prepared_mask, (0, 1, max_points), np.float32),
        "costheta": _stack_or_empty(prepared_costheta, (0, 1), np.float32),
        "reco_core_xy": _stack_or_empty(prepared_reco_core, (0, 2), np.float32),
        "event_indices": np.asarray(event_indices, dtype=np.int64),
        "nhit_labels": np.asarray(nhit_labels, dtype=object),
        "is_formal_flags": np.asarray(is_formal_flags, dtype=np.bool_),
    }


def iter_prepared_files(
    root_files: Sequence[str],
    *,
    tree_name: str,
    max_points: int,
    sample_mode: str,
    norm_mode: str,
    seed: int,
    reader_workers: int,
    prefetch_files: Optional[int],
    reader_backend: str,
    apply_event_cuts: bool,
    cut_pinc_max: float,
    cut_dangle_max_deg: Optional[float],
    cut_theta_max_deg: float,
    cut_fitstat_equals: int,
    cut_dcedge_min: Optional[float],
    core_box: Optional[Tuple[float, float, float, float]],
    core_scale: Tuple[float, float],
    keep_nhit_bins: Optional[set] = None,
):
    if reader_workers <= 0:
        for idx, file_path in enumerate(root_files):
            yield load_and_prepare_file(
                file_path,
                tree_name=tree_name,
                max_points=max_points,
                sample_mode=sample_mode,
                norm_mode=norm_mode,
                seed=seed + idx,
                apply_event_cuts=apply_event_cuts,
                cut_pinc_max=cut_pinc_max,
                cut_dangle_max_deg=cut_dangle_max_deg,
                cut_theta_max_deg=cut_theta_max_deg,
                cut_fitstat_equals=cut_fitstat_equals,
                cut_dcedge_min=cut_dcedge_min,
                core_box=core_box,
                core_scale=core_scale,
                keep_nhit_bins=keep_nhit_bins,
            )
        return

    max_inflight = prefetch_files if prefetch_files is not None else max(1, reader_workers * 2)
    max_inflight = max(1, max_inflight)

    executor_cls = (
        concurrent.futures.ThreadPoolExecutor
        if reader_backend == "thread"
        else concurrent.futures.ProcessPoolExecutor
    )

    with executor_cls(max_workers=reader_workers) as executor:
        submitted: Dict[int, concurrent.futures.Future] = {}
        next_submit = 0
        next_yield = 0

        while next_submit < len(root_files) and len(submitted) < max_inflight:
            submitted[next_submit] = executor.submit(
                load_and_prepare_file,
                root_files[next_submit],
                tree_name=tree_name,
                max_points=max_points,
                sample_mode=sample_mode,
                norm_mode=norm_mode,
                seed=seed + next_submit,
                apply_event_cuts=apply_event_cuts,
                cut_pinc_max=cut_pinc_max,
                cut_dangle_max_deg=cut_dangle_max_deg,
                cut_theta_max_deg=cut_theta_max_deg,
                cut_fitstat_equals=cut_fitstat_equals,
                cut_dcedge_min=cut_dcedge_min,
                core_box=core_box,
                core_scale=core_scale,
                keep_nhit_bins=keep_nhit_bins,
            )
            next_submit += 1

        while next_yield < len(root_files):
            future = submitted.pop(next_yield)
            prepared = future.result()
            while next_submit < len(root_files) and len(submitted) < max_inflight:
                submitted[next_submit] = executor.submit(
                    load_and_prepare_file,
                    root_files[next_submit],
                    tree_name=tree_name,
                    max_points=max_points,
                    sample_mode=sample_mode,
                    norm_mode=norm_mode,
                    seed=seed + next_submit,
                    apply_event_cuts=apply_event_cuts,
                    cut_pinc_max=cut_pinc_max,
                    cut_dangle_max_deg=cut_dangle_max_deg,
                    cut_theta_max_deg=cut_theta_max_deg,
                    cut_fitstat_equals=cut_fitstat_equals,
                    cut_dcedge_min=cut_dcedge_min,
                    core_box=core_box,
                    core_scale=core_scale,
                    keep_nhit_bins=keep_nhit_bins,
                )
                next_submit += 1
            yield prepared
            next_yield += 1


def process_one_file(
    prepared: Dict[str, object],
    *,
    model: ParticleNetRegressor,
    device: torch.device,
    tree_name: str,
    batch_size: int,
    output_root: str,
    global_bin_counts: Dict[Tuple[str, str], int],
) -> Dict[str, int]:
    file_path = str(prepared["file_path"])
    arrays = prepared["arrays"]
    points_all = prepared["points"]
    features_all = prepared["features"]
    mask_all = prepared["mask"]
    costheta_all = prepared["costheta"]
    reco_core_all = prepared["reco_core_xy"]
    event_indices_all = prepared["event_indices"]
    nhit_labels_all = prepared["nhit_labels"]
    is_formal_all = prepared["is_formal_flags"]

    file_total = int(prepared["total_events"])
    file_inferred = 0
    file_out_of_range = 0

    grouped_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    grouped_preds: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    n_prepared = int(event_indices_all.shape[0])
    for start in range(0, n_prepared, batch_size):
        stop = min(start + batch_size, n_prepared)
        preds = infer_batch(
            model,
            device,
            points_all[start:stop],
            features_all[start:stop],
            mask_all[start:stop],
            costheta_all[start:stop],
            reco_core_all[start:stop],
        )
        for local_idx, pred in enumerate(preds.tolist()):
            pos = start + local_idx
            event_idx = int(event_indices_all[pos])
            nhit_label = str(nhit_labels_all[pos])
            is_formal_bin = bool(is_formal_all[pos])
            pred_label = pred_bin_label(float(pred))
            global_bin_counts[(nhit_label, pred_label)] += 1
            file_inferred += 1
            if not is_formal_bin:
                file_out_of_range += 1
                # Keep the high-nhit overflow bin on disk for downstream plots.
                if nhit_label != ">=2000":
                    continue
            bin_key = (nhit_label, pred_label)
            grouped_indices[bin_key].append(event_idx)
            grouped_preds[bin_key].append(float(pred))

    flush_file_outputs(file_path, arrays, grouped_indices, grouped_preds, output_root, tree_name)

    return {
        "total_events": file_total,
        "inferred_events": file_inferred,
        "out_of_range_events": file_out_of_range,
    }


def combine_bin_counts(
    target: Dict[Tuple[str, str], int],
    source: Dict[Tuple[str, str], int],
) -> None:
    for key, value in source.items():
        target[key] += int(value)


def run_file_loop(
    *,
    root_files: Sequence[str],
    run_dir: str,
    output_root: str,
    tree_name: str,
    batch_size: int,
    seed: int,
    device_arg: str,
    gpu_id: int,
    print_every: int,
    reader_workers: int,
    prefetch_files: Optional[int],
    reader_backend: str,
    apply_event_cuts: bool,
    cut_pinc_max: float,
    cut_dangle_max_deg: Optional[float],
    cut_theta_max_deg: float,
    cut_fitstat_equals: int,
    cut_dcedge_min: Optional[float],
    core_box: Optional[Tuple[float, float, float, float]],
    core_scale: Tuple[float, float],
    keep_nhit_bins: Optional[set] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = load_run_config(run_dir)
    device = choose_device(device_arg, gpu_id)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
    model = build_model(config, checkpoint_path, device)

    max_points = int(config["max_points"])
    sample_mode = str(config.get("sample_mode", "weighted_q"))
    norm_mode = str(config.get("norm_mode", "per_event"))
    core_scale = (
        float(config.get("core_scale_x", 130.0)),
        float(config.get("core_scale_y", 110.0)),
    )
    if norm_mode == "global":
        raise RuntimeError("Run config requests global normalization, but no persisted scaler artifact is available in this project.")

    total_events = 0
    inferred_events = 0
    out_of_range_events = 0
    per_file_counts: Dict[str, int] = {}
    bin_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    loop_start = time.perf_counter()
    for idx, prepared in enumerate(
        iter_prepared_files(
            root_files,
            tree_name=tree_name,
            max_points=max_points,
            sample_mode=sample_mode,
            norm_mode=norm_mode,
            seed=seed,
            reader_workers=reader_workers,
            prefetch_files=prefetch_files,
            reader_backend=reader_backend,
            apply_event_cuts=apply_event_cuts,
            cut_pinc_max=cut_pinc_max,
            cut_dangle_max_deg=cut_dangle_max_deg,
            cut_theta_max_deg=cut_theta_max_deg,
            cut_fitstat_equals=cut_fitstat_equals,
            cut_dcedge_min=cut_dcedge_min,
            core_box=core_box,
            core_scale=core_scale,
            keep_nhit_bins=keep_nhit_bins,
        ),
        start=1,
    ):
        result = process_one_file(
            prepared,
            model=model,
            device=device,
            tree_name=tree_name,
            batch_size=batch_size,
            output_root=output_root,
            global_bin_counts=bin_counts,
        )
        file_name = os.path.basename(str(prepared["file_path"]))
        total_events += int(result["total_events"])
        inferred_events += int(result["inferred_events"])
        out_of_range_events += int(result["out_of_range_events"])
        per_file_counts[file_name] = int(result["total_events"])

        if idx % print_every == 0 or idx == len(root_files):
            print(
                f"[gpu {gpu_id}] [{idx}/{len(root_files)}] files processed | "
                f"events={total_events} inferred={inferred_events} out_of_range={out_of_range_events}"
            )

    return {
        "elapsed_seconds": float(time.perf_counter() - loop_start),
        "total_events": total_events,
        "inferred_events": inferred_events,
        "out_of_range_events": out_of_range_events,
        "per_file_counts": per_file_counts,
        "bin_counts": dict(bin_counts),
        "gpu_id": gpu_id,
        "n_files": len(root_files),
    }


def shard_root_files(root_files: Sequence[str], n_shards: int) -> List[List[str]]:
    shards: List[List[str]] = [[] for _ in range(n_shards)]
    for idx, file_path in enumerate(root_files):
        shards[idx % n_shards].append(file_path)
    return shards


def run_shard_worker(kwargs: Dict[str, object]) -> Dict[str, object]:
    return run_file_loop(**kwargs)


def print_overall_summary(
    total_events: int,
    inferred_events: int,
    out_of_range_events: int,
    bin_counts: Dict[Tuple[str, str], int],
) -> None:
    print(f"总 event 数: {total_events}")
    print(f"成功推理的 event 数: {inferred_events}")
    print(f"被排除到 nhit range 之外的 event 数: {out_of_range_events}")
    print("每个 2D bin 的数量:")
    for row in all_summary_rows(bin_counts):
        if row["count"] == 0:
            continue
        print(
            f"  nhit={row['nhit_bin']}, predE={row['predE_bin']}, "
            f"count={row['count']}, stats={row['statistics_level']}"
        )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cut_dangle_max_deg = None if args.no_cut_dangle else float(args.cut_dangle_max_deg)
    core_box = tuple(args.core_box) if args.core_box is not None else None

    keep_nhit_bins = None
    if args.keep_nhit_bins is not None and args.keep_nhit_bins.strip() != "":
        keep_nhit_bins = {item.strip() for item in args.keep_nhit_bins.split(",") if item.strip()}
        if not keep_nhit_bins:
            raise ValueError("--keep-nhit-bins was provided but no valid nhit label was parsed.")

    run_dir = os.path.realpath(args.run_dir)
    output_root = os.path.realpath(args.output_root)
    summary_dir = os.path.join(output_root, args.summary_dirname)
    os.makedirs(output_root, exist_ok=True)

    config = load_run_config(run_dir)
    print(f"Run dir: {run_dir}")
    root_files = discover_root_files(args.input_root, args.max_files)
    print(f"Discovered {len(root_files)} ROOT files under {args.input_root}")
    gpu_ids = parse_gpu_ids(args.gpu_ids, args.gpu_id, args.device)
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
    run_metadata: Dict[str, object] = {
        "input_root": os.path.realpath(args.input_root),
        "output_root": output_root,
        "run_dir": run_dir,
        "config_path": os.path.join(run_dir, "config.json"),
        "checkpoint_path": checkpoint_path,
        "tree_name": args.tree_name,
        "processed_file_target": None if args.max_files is None else int(args.max_files),
        "device_arg": args.device,
        "gpu_id": int(args.gpu_id),
        "gpu_ids": [int(value) for value in gpu_ids],
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "reader_workers": int(args.reader_workers),
        "prefetch_files": args.prefetch_files,
        "reader_backend": args.reader_backend,
        "event_cuts": {
            "apply_event_cuts": bool(args.apply_event_cuts),
            "pinc_max": float(args.cut_pinc_max),
            "dangle_max_deg": cut_dangle_max_deg,
            "theta_max_deg": float(args.cut_theta_max_deg),
            "fitstat_equals": int(args.cut_fitstat_equals),
            "dcedge_min": None if args.cut_dcedge_min is None else float(args.cut_dcedge_min),
            "core_box": None if core_box is None else [float(value) for value in core_box],
        },
        "keep_nhit_bins": None if keep_nhit_bins is None else sorted(keep_nhit_bins),
        "model_config": {
            "max_points": config.get("max_points"),
            "sample_mode": config.get("sample_mode"),
            "norm_mode": config.get("norm_mode"),
            "theta_embed_dim": config.get("theta_embed_dim"),
            "theta_embed_dropout": config.get("theta_embed_dropout"),
            "core_embed_dim": config.get("core_embed_dim", 0),
            "core_embed_dropout": config.get("core_embed_dropout", 0.0),
            "core_scale_x": config.get("core_scale_x", 130.0),
            "core_scale_y": config.get("core_scale_y", 110.0),
        },
    }

    total_events = 0
    inferred_events = 0
    out_of_range_events = 0
    per_file_counts: Dict[str, int] = {}
    bin_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    loop_start = time.perf_counter()

    if args.device == "cpu" or len(gpu_ids) <= 1:
        selected_gpu_id = gpu_ids[0] if gpu_ids else 0
        device = choose_device(args.device, selected_gpu_id)
        print(f"Using device: {device}")
        result = run_file_loop(
            root_files=root_files,
            run_dir=run_dir,
            output_root=output_root,
            tree_name=args.tree_name,
            batch_size=args.batch_size,
            seed=args.seed,
            device_arg=args.device,
            gpu_id=selected_gpu_id,
            print_every=args.print_every,
            reader_workers=args.reader_workers,
            prefetch_files=args.prefetch_files,
            reader_backend=args.reader_backend,
            apply_event_cuts=args.apply_event_cuts,
            cut_pinc_max=args.cut_pinc_max,
            cut_dangle_max_deg=cut_dangle_max_deg,
            cut_theta_max_deg=args.cut_theta_max_deg,
            cut_fitstat_equals=args.cut_fitstat_equals,
            cut_dcedge_min=args.cut_dcedge_min,
            core_box=core_box,
            core_scale=(
                float(config.get("core_scale_x", 130.0)),
                float(config.get("core_scale_y", 110.0)),
            ),
            keep_nhit_bins=keep_nhit_bins,
        )
        total_events = int(result["total_events"])
        inferred_events = int(result["inferred_events"])
        out_of_range_events = int(result["out_of_range_events"])
        per_file_counts.update(result["per_file_counts"])
        combine_bin_counts(bin_counts, result["bin_counts"])
        print(f"Single-device elapsed seconds: {float(result['elapsed_seconds']):.2f}")
    else:
        if args.device == "cpu":
            raise RuntimeError("Multi-GPU execution requires --device cuda or auto with available GPUs.")
        print(f"Using multi-GPU sharding over GPU ids: {gpu_ids}")
        shards = shard_root_files(root_files, len(gpu_ids))
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as executor:
            futures = []
            for shard_idx, (gpu_id, shard_files) in enumerate(zip(gpu_ids, shards)):
                if not shard_files:
                    continue
                worker_kwargs = dict(
                    root_files=shard_files,
                    run_dir=run_dir,
                    output_root=output_root,
                    tree_name=args.tree_name,
                    batch_size=args.batch_size,
                    seed=args.seed + shard_idx * 100000,
                    device_arg="cuda",
                    gpu_id=gpu_id,
                    print_every=args.print_every,
                    reader_workers=args.reader_workers,
                    prefetch_files=args.prefetch_files,
                    reader_backend=args.reader_backend,
                    apply_event_cuts=args.apply_event_cuts,
                    cut_pinc_max=args.cut_pinc_max,
                    cut_dangle_max_deg=cut_dangle_max_deg,
                    cut_theta_max_deg=args.cut_theta_max_deg,
                    cut_fitstat_equals=args.cut_fitstat_equals,
                    cut_dcedge_min=args.cut_dcedge_min,
                    core_box=core_box,
                    core_scale=(
                        float(config.get("core_scale_x", 130.0)),
                        float(config.get("core_scale_y", 110.0)),
                    ),
                    keep_nhit_bins=keep_nhit_bins,
                )
                futures.append(executor.submit(run_shard_worker, worker_kwargs))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(
                    f"Shard finished on gpu {result['gpu_id']} | "
                    f"files={result['n_files']} elapsed={float(result['elapsed_seconds']):.2f}s"
                )
                total_events += int(result["total_events"])
                inferred_events += int(result["inferred_events"])
                out_of_range_events += int(result["out_of_range_events"])
                per_file_counts.update(result["per_file_counts"])
                combine_bin_counts(bin_counts, result["bin_counts"])

    write_summary(
        summary_dir,
        bin_counts,
        total_events=total_events,
        inferred_events=inferred_events,
        out_of_range_events=out_of_range_events,
        per_file_counts=per_file_counts,
        run_metadata=run_metadata,
    )
    print(f"Elapsed seconds: {time.perf_counter() - loop_start:.2f}")
    print_overall_summary(total_events, inferred_events, out_of_range_events, bin_counts)


if __name__ == "__main__":
    main()
