#!/usr/bin/env python
import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import uproot
from uproot.source.file import MemmapSource

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
REQUIRED_EVENT_BRANCHES = MODEL_BRANCHES + ["vidmc", "theta", "xc", "yc", "pincness", "fitstat", "dcedge"]
DROP_BRANCHES = {"vx", "vy", "vq", "vt"}
LEGACY_LAYOUT_PREFIX = "obs_filtered_"
DAY_DIR_RE = re.compile(r"^\d{4}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run observation-data inference with the final theta+position embedding model, "
            "write predicted energy branches into mirrored ROOT outputs, and drop hit-level branches."
        )
    )
    parser.add_argument("--input-root", type=str, default="/mnt/mydisk/WCDA_observation")
    parser.add_argument("--output-root", type=str, default="/mnt/mydisk/WCDA_observation_eval")
    parser.add_argument(
        "--day-prefix",
        type=str,
        default=None,
        help="Only process observation files whose MMDD directory starts with this prefix, e.g. '02'.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional path for the final JSON summary. Defaults to <output-root>/apply_summary.json.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/home/server/projects/energy_reconstruction/runs/theta_recoxy_position_embed_midenergy_8666",
    )
    parser.add_argument("--tree-name", type=str, default="t_eventout")
    parser.add_argument("--header-tree-name", type=str, default="t_eventout_h")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-files", type=int, default=None, help="Only process the first N ROOT files after sorting.")
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=None,
        help="Only use the first N cut-passing events from each file. Intended for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU ids for multi-GPU sharding, e.g. 0,1,2,3")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--reader-workers", type=int, default=0, help="CPU workers for uproot decompression and interpretation.")
    parser.add_argument("--step-size", type=str, default="100 MB", help="Chunk size for streaming ROOT reads, e.g. '10 MB'.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output ROOT files that already satisfy the observation eval branch contract.",
    )
    parser.add_argument("--cut-pinc-max", type=float, default=1.1)
    parser.add_argument("--cut-fitstat-equals", type=int, default=0)
    parser.add_argument("--cut-theta-max-deg", type=float, default=50.0)
    parser.add_argument("--cut-dcedge-min", type=float, default=20.0)
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


def discover_root_files(input_root: str, max_files: Optional[int]) -> Tuple[List[str], str]:
    input_path = Path(input_root)
    if input_path.is_file():
        if input_path.suffix.lower() != ".root":
            raise ValueError(f"Expected a ROOT file, got: {input_root}")
        root_files = [str(input_path)]
        base_dir = str(input_path)
    elif input_path.is_dir():
        root_files = sorted(str(path) for path in input_path.rglob("*.root") if path.is_file())
        base_dir = str(input_path)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    if max_files is not None:
        root_files = root_files[:max_files]
    return root_files, base_dir


def filter_root_files_by_day_prefix(
    root_files: Sequence[str],
    *,
    input_root: str,
    day_prefix: Optional[str],
) -> Tuple[List[str], List[str]]:
    if day_prefix is None:
        return list(root_files), sorted(
            {
                normalize_observation_relative_path(file_path, input_root).split(os.sep, 1)[0]
                for file_path in root_files
            }
        )

    normalized_prefix = day_prefix.strip()
    if normalized_prefix == "":
        return list(root_files), sorted(
            {
                normalize_observation_relative_path(file_path, input_root).split(os.sep, 1)[0]
                for file_path in root_files
            }
        )

    filtered_files: List[str] = []
    matched_day_dirs = set()
    for file_path in root_files:
        day_dir = normalize_observation_relative_path(file_path, input_root).split(os.sep, 1)[0]
        if day_dir.startswith(normalized_prefix):
            filtered_files.append(file_path)
            matched_day_dirs.add(day_dir)

    return filtered_files, sorted(matched_day_dirs)


def is_day_dir_name(name: str) -> bool:
    return bool(DAY_DIR_RE.fullmatch(name))


def ensure_supported_observation_layout(input_root: str) -> None:
    input_path = Path(input_root).resolve()
    if any(part.startswith(LEGACY_LAYOUT_PREFIX) for part in input_path.parts):
        raise RuntimeError(
            "Legacy observation layout is not supported. "
            "Please flatten /mnt/mydisk/WCDA_observation so files live under MMDD directories."
        )
    if input_path.is_dir():
        legacy_dirs = sorted(
            child.name
            for child in input_path.iterdir()
            if child.is_dir() and child.name.startswith(LEGACY_LAYOUT_PREFIX)
        )
        if legacy_dirs:
            raise RuntimeError(
                "Legacy observation layout detected under input-root: "
                f"{', '.join(legacy_dirs[:5])}. "
                "Please run scripts/data/flatten_observation_layout.py --apply first."
            )


def normalize_observation_relative_path(file_path: str, input_root: str) -> str:
    file_path_obj = Path(file_path).resolve()
    input_root_obj = Path(input_root).resolve()

    if input_root_obj.is_file():
        if file_path_obj != input_root_obj:
            raise ValueError(f"Input file mismatch: {file_path_obj} vs {input_root_obj}")
        day_dir = file_path_obj.parent.name
        if not is_day_dir_name(day_dir):
            raise ValueError(
                "Single-file observation input must be located under an MMDD directory, "
                f"got parent={file_path_obj.parent}"
            )
        return os.path.join(day_dir, file_path_obj.name)

    if input_root_obj.is_dir():
        try:
            rel_parts = file_path_obj.relative_to(input_root_obj).parts
        except ValueError as exc:
            raise ValueError(f"File {file_path_obj} is not under input-root {input_root_obj}") from exc

        if not rel_parts:
            raise ValueError(f"Could not derive a relative path for {file_path_obj}")

        if input_root_obj.name.startswith(LEGACY_LAYOUT_PREFIX):
            raise ValueError(
                "Legacy observation layout is not supported. "
                "Please flatten /mnt/mydisk/WCDA_observation first."
            )

        if is_day_dir_name(input_root_obj.name):
            return os.path.join(input_root_obj.name, *rel_parts)

        first_part = rel_parts[0]
        if first_part.startswith(LEGACY_LAYOUT_PREFIX):
            raise ValueError(
                "Legacy observation layout detected under input-root. "
                "Please run scripts/data/flatten_observation_layout.py --apply first."
            )
        if not is_day_dir_name(first_part):
            raise ValueError(
                "Observation ROOT files must live under MMDD directories directly beneath input-root. "
                f"Got relative path: {os.path.join(*rel_parts)}"
            )
        return os.path.join(*rel_parts)

    raise FileNotFoundError(f"Input path does not exist: {input_root}")


def load_run_config(run_dir: str) -> Dict[str, object]:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_input_root(
    file_path: str,
    *,
    decompression_executor: Optional[concurrent.futures.Executor] = None,
    interpretation_executor: Optional[concurrent.futures.Executor] = None,
):
    # Prefer direct local-file access for ROOT inputs; the default fsspec path
    # can hang on basket reads in this environment.
    return uproot.open(
        file_path,
        handler=MemmapSource,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
    )


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    return checkpoint


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
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
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

    if sample_mode == "random":
        idxs = rng.choice(n_points, max_points, replace=False)
    elif sample_mode == "firstk":
        idxs = np.arange(max_points)
    elif sample_mode == "topk_q":
        idxs = np.argsort(vq)[-max_points:]
    elif sample_mode == "weighted_q":
        weights = np.clip(vq, a_min=0.0, a_max=None)
        denom = float(np.sum(weights))
        if denom <= 0:
            idxs = rng.choice(n_points, max_points, replace=False)
        else:
            idxs = rng.choice(n_points, max_points, replace=False, p=weights / denom)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    return points[idxs], vq[idxs], vt[idxs]


def normalize_features(vq: np.ndarray, vt: np.ndarray, norm_mode: str) -> np.ndarray:
    if norm_mode == "none":
        return np.column_stack([vq, vt]).astype(np.float32)
    if norm_mode == "per_event":
        vq_mean, vq_std = _safe_mean_std(vq)
        vt_mean, vt_std = _safe_mean_std(vt)
        return np.column_stack([(vq - vq_mean) / vq_std, (vt - vt_mean) / vt_std]).astype(np.float32)
    raise ValueError(
        "Observation apply currently supports the run's original per_event/none normalization paths only. "
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
    points = build_hit_points(processed, detector_ids=detector_ids, coordinate_system="global")
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
    batch_reco_core_xy: np.ndarray,
) -> np.ndarray:
    points = torch.from_numpy(batch_points).to(device, non_blocking=True)
    features = torch.from_numpy(batch_features).to(device, non_blocking=True)
    mask = torch.from_numpy(batch_mask).to(device, non_blocking=True)
    costheta = torch.from_numpy(batch_costheta).to(device, non_blocking=True)
    reco_core_xy = torch.from_numpy(batch_reco_core_xy).to(device, non_blocking=True)
    with torch.no_grad():
        pred = model(points, features, mask, costheta, reco_core_xy)
    return pred.detach().cpu().numpy().reshape(-1)


def _stack_or_empty(chunks: List[np.ndarray], shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if chunks:
        return np.stack(chunks, axis=0).astype(dtype, copy=False)
    return np.empty(shape, dtype=dtype)


def filter_event_indices(
    arrays: Dict[str, np.ndarray],
    *,
    cut_pinc_max: float,
    cut_fitstat_equals: int,
    cut_theta_max_deg: float,
    cut_dcedge_min: float,
) -> np.ndarray:
    n_events = len(next(iter(arrays.values())) if arrays else [])
    if n_events == 0:
        return np.empty((0,), dtype=np.int64)

    missing = [name for name in REQUIRED_EVENT_BRANCHES if name not in arrays]
    if missing:
        raise KeyError(f"Required branches missing: {', '.join(missing)}")

    mask = np.ones(n_events, dtype=bool)
    mask &= np.asarray(arrays["pincness"] < float(cut_pinc_max))
    mask &= np.asarray(arrays["fitstat"] == int(cut_fitstat_equals))
    mask &= np.asarray(arrays["theta"] < (float(cut_theta_max_deg) * np.pi / 180.0))
    mask &= np.asarray(arrays["dcedge"] > float(cut_dcedge_min))
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def open_tree(root_file: uproot.ReadOnlyDirectory, tree_name: str):
    tree_key = f"{tree_name};1" if f"{tree_name};1" in root_file else tree_name
    return root_file[tree_key]


def read_header_arrays(root_file: uproot.ReadOnlyDirectory, header_tree_name: str) -> Optional[Dict[str, np.ndarray]]:
    if f"{header_tree_name};1" not in root_file and header_tree_name not in root_file:
        return None
    header_tree = open_tree(root_file, header_tree_name)
    return header_tree.arrays(list(header_tree.keys()), library="np")


def prepare_chunk(
    arrays: Dict[str, np.ndarray],
    selected_event_indices: np.ndarray,
    *,
    max_points: int,
    sample_mode: str,
    norm_mode: str,
    core_scale: Tuple[float, float],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    prepared_points: List[np.ndarray] = []
    prepared_features: List[np.ndarray] = []
    prepared_mask: List[np.ndarray] = []
    prepared_costheta: List[np.ndarray] = []
    prepared_reco_core: List[np.ndarray] = []
    survivor_event_indices: List[int] = []

    for event_idx in selected_event_indices.tolist():
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
        survivor_event_indices.append(int(event_idx))

    return {
        "points": _stack_or_empty(prepared_points, (0, 2, max_points), np.float32),
        "features": _stack_or_empty(prepared_features, (0, 2, max_points), np.float32),
        "mask": _stack_or_empty(prepared_mask, (0, 1, max_points), np.float32),
        "costheta": _stack_or_empty(prepared_costheta, (0, 1), np.float32),
        "reco_core_xy": _stack_or_empty(prepared_reco_core, (0, 2), np.float32),
        "survivor_event_indices": np.asarray(survivor_event_indices, dtype=np.int64),
    }


def infer_prepared(
    prepared: Dict[str, np.ndarray],
    *,
    model: ParticleNetRegressor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    survivor_event_indices = prepared["survivor_event_indices"]
    n_prepared = int(survivor_event_indices.shape[0])
    preds: List[np.ndarray] = []
    for start in range(0, n_prepared, batch_size):
        stop = min(start + batch_size, n_prepared)
        preds.append(
            infer_batch(
                model,
                device,
                prepared["points"][start:stop],
                prepared["features"][start:stop],
                prepared["mask"][start:stop],
                prepared["costheta"][start:stop],
                prepared["reco_core_xy"][start:stop],
            )
        )
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def build_output_payload(
    arrays: Dict[str, np.ndarray],
    survivor_event_indices: np.ndarray,
    loge_preds: np.ndarray,
) -> Dict[str, np.ndarray]:
    payload = {
        name: arrays[name][survivor_event_indices]
        for name in arrays.keys()
        if name not in DROP_BRANCHES
    }
    payload["ml_logE_pred"] = np.asarray(loge_preds, dtype=np.float32)
    payload["ml_energy_pred"] = np.power(10.0, np.asarray(loge_preds, dtype=np.float64)).astype(np.float32)
    return payload


def build_empty_payload(template_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    payload = {name: template_arrays[name][:0] for name in template_arrays.keys() if name not in DROP_BRANCHES}
    payload["ml_logE_pred"] = np.empty((0,), dtype=np.float32)
    payload["ml_energy_pred"] = np.empty((0,), dtype=np.float32)
    return payload


def output_path_for_file(file_path: str, input_root: str, output_root: str) -> str:
    relative_path = normalize_observation_relative_path(file_path, input_root)
    return os.path.join(output_root, relative_path)


def validate_existing_output(path: str, tree_name: str, header_tree_name: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with uproot.open(path) as root_file:
            if header_tree_name not in root_file or tree_name not in root_file:
                return False
            branches = set(root_file[tree_name].keys())
            if not {"ml_logE_pred", "ml_energy_pred"}.issubset(branches):
                return False
            if DROP_BRANCHES.intersection(branches):
                return False
            return True
    except Exception:
        return False


def process_one_file(
    file_path: str,
    *,
    model: ParticleNetRegressor,
    device: torch.device,
    tree_name: str,
    header_tree_name: str,
    input_root: str,
    output_root: str,
    max_points: int,
    sample_mode: str,
    norm_mode: str,
    seed: int,
    batch_size: int,
    cut_pinc_max: float,
    cut_fitstat_equals: int,
    cut_theta_max_deg: float,
    cut_dcedge_min: float,
    core_scale: Tuple[float, float],
    reader_workers: int,
    max_events_per_file: Optional[int],
    step_size: str = "100 MB",
) -> Dict[str, object]:
    out_path = output_path_for_file(file_path, input_root, output_root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=reader_workers) if reader_workers > 0 else None
    total_events = 0
    cut_pass_events = 0
    written_events = 0
    wrote_tree = False
    template_arrays: Optional[Dict[str, np.ndarray]] = None
    remaining = max_events_per_file
    rng = np.random.default_rng(seed)

    try:
        with open_input_root(
            file_path,
            decompression_executor=executor,
            interpretation_executor=executor,
        ) as root_file:
            tree = open_tree(root_file, tree_name)
            header_arrays = read_header_arrays(root_file, header_tree_name)
            total_events = int(tree.num_entries)

            with uproot.recreate(out_path) as fout:
                if header_arrays is not None:
                    fout[header_tree_name] = header_arrays

                for arrays in tree.iterate(
                    list(tree.keys()),
                    step_size=step_size,
                    library="np",
                    decompression_executor=executor,
                    interpretation_executor=executor,
                ):
                    if template_arrays is None:
                        template_arrays = arrays

                    selected_event_indices = filter_event_indices(
                        arrays,
                        cut_pinc_max=cut_pinc_max,
                        cut_fitstat_equals=cut_fitstat_equals,
                        cut_theta_max_deg=cut_theta_max_deg,
                        cut_dcedge_min=cut_dcedge_min,
                    )
                    if remaining is not None:
                        if remaining <= 0:
                            break
                        selected_event_indices = selected_event_indices[:remaining]
                        remaining -= int(selected_event_indices.shape[0])

                    cut_pass_events += int(selected_event_indices.shape[0])
                    if selected_event_indices.shape[0] == 0:
                        if remaining is not None and remaining <= 0:
                            break
                        continue

                    prepared = prepare_chunk(
                        arrays,
                        selected_event_indices,
                        max_points=max_points,
                        sample_mode=sample_mode,
                        norm_mode=norm_mode,
                        core_scale=core_scale,
                        rng=rng,
                    )
                    loge_preds = infer_prepared(
                        prepared,
                        model=model,
                        device=device,
                        batch_size=batch_size,
                    )
                    event_payload = build_output_payload(
                        arrays,
                        prepared["survivor_event_indices"],
                        loge_preds,
                    )
                    if not wrote_tree:
                        fout[tree_name] = event_payload
                        wrote_tree = True
                    else:
                        fout[tree_name].extend(event_payload)
                    written_events += int(prepared["survivor_event_indices"].shape[0])

                    if remaining is not None and remaining <= 0:
                        break

                if not wrote_tree:
                    if template_arrays is None:
                        template_arrays = tree.arrays(list(tree.keys()), entry_start=0, entry_stop=0, library="np")
                    fout[tree_name] = build_empty_payload(template_arrays)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return {
        "file_path": file_path,
        "out_path": out_path,
        "total_events": int(total_events),
        "cut_pass_events": int(cut_pass_events),
        "written_events": int(written_events),
    }


def build_final_summary(
    *,
    args: argparse.Namespace,
    input_root_path: str,
    checkpoint_path: str,
    max_points: int,
    sample_mode: str,
    norm_mode: str,
    core_scale: Tuple[float, float],
    config: Dict[str, object],
    files_total: int,
) -> Dict[str, object]:
    return {
        "input_root": args.input_root,
        "output_root": args.output_root,
        "day_prefix": args.day_prefix,
        "input_root_path": input_root_path,
        "run_dir": args.run_dir,
        "checkpoint_path": checkpoint_path,
        "summary_path": args.summary_path,
        "tree_name": args.tree_name,
        "header_tree_name": args.header_tree_name,
        "cuts": {
            "pincness_lt": float(args.cut_pinc_max),
            "fitstat_equals": int(args.cut_fitstat_equals),
            "theta_deg_lt": float(args.cut_theta_max_deg),
            "dcedge_gt": float(args.cut_dcedge_min),
        },
        "run_config": {
            "max_points": max_points,
            "sample_mode": sample_mode,
            "norm_mode": norm_mode,
            "theta_embed_dim": int(config.get("theta_embed_dim", 0)),
            "theta_embed_dropout": float(config.get("theta_embed_dropout", 0.0)),
            "core_embed_dim": int(config.get("core_embed_dim", 0)),
            "core_embed_dropout": float(config.get("core_embed_dropout", 0.0)),
            "core_scale_x": float(core_scale[0]),
            "core_scale_y": float(core_scale[1]),
        },
        "files_total": int(files_total),
        "max_events_per_file": args.max_events_per_file,
        "files_completed": 0,
        "files_failed": 0,
        "events_total": 0,
        "events_cut_pass": 0,
        "events_written": 0,
        "failures": [],
        "per_file": {},
    }


def build_worker_summary(root_files: Sequence[str], gpu_id: int) -> Dict[str, object]:
    return {
        "gpu_id": int(gpu_id),
        "n_files": int(len(root_files)),
        "files_completed": 0,
        "files_failed": 0,
        "events_total": 0,
        "events_cut_pass": 0,
        "events_written": 0,
        "failures": [],
        "per_file": {},
    }


def merge_worker_summary(target: Dict[str, object], worker: Dict[str, object]) -> None:
    target["files_completed"] += int(worker["files_completed"])
    target["files_failed"] += int(worker["files_failed"])
    target["events_total"] += int(worker["events_total"])
    target["events_cut_pass"] += int(worker["events_cut_pass"])
    target["events_written"] += int(worker["events_written"])
    target["failures"].extend(worker["failures"])
    target["per_file"].update(worker["per_file"])


def shard_root_files(root_files: Sequence[str], n_shards: int) -> List[List[str]]:
    shards: List[List[str]] = [[] for _ in range(n_shards)]
    for idx, file_path in enumerate(root_files):
        shards[idx % n_shards].append(file_path)
    return shards


def run_file_loop(
    *,
    root_files: Sequence[str],
    input_root_path: str,
    run_dir: str,
    output_root: str,
    tree_name: str,
    header_tree_name: str,
    batch_size: int,
    seed: int,
    device_arg: str,
    gpu_id: int,
    print_every: int,
    reader_workers: int,
    max_events_per_file: Optional[int],
    step_size: str,
    cut_pinc_max: float,
    cut_fitstat_equals: int,
    cut_theta_max_deg: float,
    cut_dcedge_min: float,
    skip_existing: bool,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = load_run_config(run_dir)
    norm_mode = str(config.get("norm_mode", "per_event"))
    sample_mode = str(config.get("sample_mode", "weighted_q"))
    if norm_mode == "global":
        raise RuntimeError("Run config requests global normalization, but no persisted scaler artifact is available in this project.")

    max_points = int(config["max_points"])
    core_scale = (
        float(config.get("core_scale_x", 130.0)),
        float(config.get("core_scale_y", 110.0)),
    )

    device = choose_device(device_arg, gpu_id)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    checkpoint_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
    model = build_model(config, checkpoint_path, device)

    summary = build_worker_summary(root_files, gpu_id)
    loop_start = time.perf_counter()
    for idx, file_path in enumerate(root_files, start=1):
        file_name = normalize_observation_relative_path(file_path, input_root_path)
        try:
            out_path = output_path_for_file(file_path, input_root_path, output_root)
            if skip_existing and validate_existing_output(out_path, tree_name, header_tree_name):
                summary["files_completed"] += 1
                summary["per_file"][file_name] = {
                    "skipped_existing": True,
                    "output_path": out_path,
                }
                if idx % print_every == 0 or idx == len(root_files):
                    print(
                        f"[gpu {gpu_id}] [{idx}/{len(root_files)}] files processed | "
                        f"completed={summary['files_completed']} failed={summary['files_failed']} "
                        f"events_total={summary['events_total']} cut_pass={summary['events_cut_pass']} "
                        f"written={summary['events_written']}"
                    )
                continue
            result = process_one_file(
                file_path,
                model=model,
                device=device,
                tree_name=tree_name,
                header_tree_name=header_tree_name,
                input_root=input_root_path,
                output_root=output_root,
                max_points=max_points,
                sample_mode=sample_mode,
                norm_mode=norm_mode,
                seed=seed + idx - 1,
                batch_size=batch_size,
                cut_pinc_max=cut_pinc_max,
                cut_fitstat_equals=cut_fitstat_equals,
                cut_theta_max_deg=cut_theta_max_deg,
                cut_dcedge_min=cut_dcedge_min,
                core_scale=core_scale,
                reader_workers=reader_workers,
                max_events_per_file=max_events_per_file,
                step_size=step_size,
            )
            summary["files_completed"] += 1
            summary["events_total"] += int(result["total_events"])
            summary["events_cut_pass"] += int(result["cut_pass_events"])
            summary["events_written"] += int(result["written_events"])
            summary["per_file"][file_name] = {
                "total_events": int(result["total_events"]),
                "cut_pass_events": int(result["cut_pass_events"]),
                "written_events": int(result["written_events"]),
                "output_path": result["out_path"],
            }
        except Exception as exc:
            summary["files_failed"] += 1
            summary["failures"].append({"file": file_name, "error": str(exc)})

        if idx % print_every == 0 or idx == len(root_files):
            print(
                f"[gpu {gpu_id}] [{idx}/{len(root_files)}] files processed | "
                f"completed={summary['files_completed']} failed={summary['files_failed']} "
                f"events_total={summary['events_total']} cut_pass={summary['events_cut_pass']} "
                f"written={summary['events_written']}"
            )

    summary["elapsed_seconds"] = float(time.perf_counter() - loop_start)
    return summary


def run_shard_worker(kwargs: Dict[str, object]) -> Dict[str, object]:
    return run_file_loop(**kwargs)


def main() -> None:
    args = parse_args()

    ensure_supported_observation_layout(args.input_root)
    root_files, input_root_path = discover_root_files(args.input_root, args.max_files)
    root_files, matched_day_dirs = filter_root_files_by_day_prefix(
        root_files,
        input_root=input_root_path,
        day_prefix=args.day_prefix,
    )
    if not root_files:
        if args.day_prefix:
            raise RuntimeError(
                f"No ROOT files found under {args.input_root} for day-prefix={args.day_prefix!r}"
            )
        raise RuntimeError(f"No ROOT files found under {args.input_root}")

    config = load_run_config(args.run_dir)
    norm_mode = str(config.get("norm_mode", "per_event"))
    sample_mode = str(config.get("sample_mode", "weighted_q"))
    if norm_mode == "global":
        raise RuntimeError("Run config requests global normalization, but no persisted scaler artifact is available in this project.")

    max_points = int(config["max_points"])
    core_scale = (
        float(config.get("core_scale_x", 130.0)),
        float(config.get("core_scale_y", 110.0)),
    )
    checkpoint_path = os.path.join(args.run_dir, "checkpoints", "best_model.pt")
    gpu_ids = parse_gpu_ids(args.gpu_ids, args.gpu_id, args.device)

    print(f"Discovered {len(root_files)} ROOT files under {args.input_root}")
    if args.day_prefix:
        print(f"Day-prefix filter: {args.day_prefix}")
        print(f"Matched MMDD directories ({len(matched_day_dirs)}): {matched_day_dirs}")
    print(f"Using run dir: {args.run_dir}")
    print(f"Output root: {args.output_root}")
    print(
        "Cuts: "
        f"pincness<{args.cut_pinc_max}, "
        f"fitstat=={args.cut_fitstat_equals}, "
        f"theta<{args.cut_theta_max_deg} deg, "
        f"dcedge>{args.cut_dcedge_min} m"
    )
    print(
        "Inference config: "
        f"max_points={max_points}, sample_mode={sample_mode}, norm_mode={norm_mode}, "
        f"theta_embed_dim={config.get('theta_embed_dim')}, core_embed_dim={config.get('core_embed_dim', 0)}"
    )
    if args.device == "cpu":
        print("Using device: cpu")
    elif len(gpu_ids) <= 1:
        selected_gpu_id = gpu_ids[0] if gpu_ids else args.gpu_id
        print(f"Using single-device execution on gpu_id={selected_gpu_id}")
    else:
        print(f"Using multi-GPU sharding over GPU ids: {gpu_ids}")

    summary_path = args.summary_path or os.path.join(args.output_root, "apply_summary.json")
    args.summary_path = summary_path

    summary = build_final_summary(
        args=args,
        input_root_path=input_root_path,
        checkpoint_path=checkpoint_path,
        max_points=max_points,
        sample_mode=sample_mode,
        norm_mode=norm_mode,
        core_scale=core_scale,
        config=config,
        files_total=len(root_files),
    )

    os.makedirs(args.output_root, exist_ok=True)
    loop_start = time.perf_counter()
    if args.device == "cpu" or len(gpu_ids) <= 1:
        selected_gpu_id = gpu_ids[0] if gpu_ids else args.gpu_id
        result = run_file_loop(
            root_files=root_files,
            input_root_path=input_root_path,
            run_dir=args.run_dir,
            output_root=args.output_root,
            tree_name=args.tree_name,
            header_tree_name=args.header_tree_name,
            batch_size=args.batch_size,
            seed=args.seed,
            device_arg=args.device,
            gpu_id=selected_gpu_id,
            print_every=args.print_every,
            reader_workers=args.reader_workers,
            max_events_per_file=args.max_events_per_file,
            step_size=args.step_size,
            cut_pinc_max=args.cut_pinc_max,
            cut_fitstat_equals=args.cut_fitstat_equals,
            cut_theta_max_deg=args.cut_theta_max_deg,
            cut_dcedge_min=args.cut_dcedge_min,
            skip_existing=args.skip_existing,
        )
        merge_worker_summary(summary, result)
        print(f"Single-device elapsed seconds: {float(result['elapsed_seconds']):.2f}")
    else:
        shards = shard_root_files(root_files, len(gpu_ids))
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as executor:
            futures = []
            for shard_idx, (gpu_id, shard_files) in enumerate(zip(gpu_ids, shards)):
                if not shard_files:
                    continue
                worker_kwargs = dict(
                    root_files=shard_files,
                    input_root_path=input_root_path,
                    run_dir=args.run_dir,
                    output_root=args.output_root,
                    tree_name=args.tree_name,
                    header_tree_name=args.header_tree_name,
                    batch_size=args.batch_size,
                    seed=args.seed + shard_idx * 100000,
                    device_arg="cuda",
                    gpu_id=gpu_id,
                    print_every=args.print_every,
                    reader_workers=args.reader_workers,
                    max_events_per_file=args.max_events_per_file,
                    step_size=args.step_size,
                    cut_pinc_max=args.cut_pinc_max,
                    cut_fitstat_equals=args.cut_fitstat_equals,
                    cut_theta_max_deg=args.cut_theta_max_deg,
                    cut_dcedge_min=args.cut_dcedge_min,
                    skip_existing=args.skip_existing,
                )
                futures.append(executor.submit(run_shard_worker, worker_kwargs))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(
                    f"Shard finished on gpu {result['gpu_id']} | "
                    f"files={result['n_files']} elapsed={float(result['elapsed_seconds']):.2f}s"
                )
                merge_worker_summary(summary, result)

    summary["matched_day_dirs"] = matched_day_dirs
    summary["elapsed_seconds"] = float(time.perf_counter() - loop_start)
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Observation inference finished")
    print(f"Summary: {summary_path}")
    print(
        f"Files completed={summary['files_completed']}, failed={summary['files_failed']}, "
        f"events written={summary['events_written']}"
    )
    if int(summary["files_failed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
