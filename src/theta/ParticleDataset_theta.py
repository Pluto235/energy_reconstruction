import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import uproot
from multiprocessing import Pool
from typing import Dict, Any, Optional, List, Tuple

from src.common.EdgeConv import process_features


class MissingRequiredBranchError(RuntimeError):
    """Raised when a required ROOT branch for theta training is absent."""


def _safe_mean_std(x: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) with numerical safety."""
    if x.size == 0:
        return 0.0, 1.0
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(s) or s < 1e-8:
        s = 1.0
    return m, s


def _default_cuts() -> Dict[str, Any]:
    """
    Default cuts designed to match your current hard-coded logic.
    All fields are optional; if not provided, they won't be applied.
    """
    return dict(
        Emin=100.0,
        pinc_max=1.1,
        dangle_max_rad=0.05236,  # 3 deg
        theta_max_rad=0.524,     # 30 deg
        dcedge_min=20.0,
        use_core_box=False,
        core_box=(-130.0, 130.0, -110.0, 110.0),  # (xmin, xmax, ymin, ymax) for mc_xc, mc_yc
        vqsamp_ratio_min=None,   # e.g. 0.2
        require_fitstat0=True,
        fitstat_equals=0,
    )


class ParticleDataset(Dataset):
    """
    Compatible with your main() call:
      ParticleDataset(root_files, branches, target_branch, processing_conditions, max_points=500)

    New optional args for Slurm param sweep:
      - cuts: dict for event-level filtering
      - norm_mode: 'per_event' | 'global' | 'none'
      - sample_mode: 'random' | 'topk_q' | 'firstk' | 'weighted_q'
      - io_workers: multiprocessing workers for ROOT reading
      - keep_raw: store raw branches (VERY memory heavy, default False)
      - scaler: {'vq': {'mean':..., 'std':...}, 'vt': {...}} used for norm_mode='global'
      - compute_scaler: if True and norm_mode='global' and scaler is None, compute scaler on loaded events
      - save_stats_path: write dataset stats (counts + logE distribution) to json
      - seed: controls deterministic sampling (firstk) and random operations
      - verbose: print file-level info (default False to avoid Slurm log explosion)
    """

    def __init__(
        self,
        root_files: List[str],
        branches: List[str],
        target_branch: List[str],
        processing_conditions: List[Dict[str, Any]],
        max_points: int = 256,
        *,
        cuts: Optional[Dict[str, Any]] = None,
        norm_mode: str = "per_event",
        sample_mode: str = "random",
        io_workers: Optional[int] = None,
        keep_raw: bool = False,
        scaler: Optional[Dict[str, Dict[str, float]]] = None,
        compute_scaler: bool = False,
        save_stats_path: Optional[str] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.branches = branches
        self.target_branch = target_branch
        self.processing_conditions = processing_conditions
        self.max_points = int(max_points)

        self.cuts = _default_cuts()
        if cuts:
            self.cuts.update(cuts)

        self.norm_mode = norm_mode  # 'per_event' | 'global' | 'none'
        self.sample_mode = sample_mode  # 'random' | 'topk_q' | 'firstk' | 'weighted_q'
        self.keep_raw = keep_raw
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Global scaler for vq/vt
        self.scaler = scaler

        # Decide workers: do NOT default to 64; it kills shared FS and Slurm runs.
        if io_workers is None:
            # A safe default for ROOT I/O; tweak via args in Slurm
            io_workers = min(8, os.cpu_count() or 8)
        self.io_workers = int(max(1, io_workers))

        # Load data in parallel
        load_args = [
            (f, self.branches, self.target_branch, self.processing_conditions, self.cuts, self.keep_raw, self.verbose)
            for f in root_files
        ]

        # Note: uproot + multiprocessing can be heavy; keep pool size modest.
        with Pool(self.io_workers) as pool:
            out = pool.starmap(self._load_file, load_args)

        # out is list of tuples: (records, stats)
        records_all: List[Dict[str, Any]] = []
        stats_all: List[Dict[str, Any]] = []
        for recs, st in out:
            records_all.extend(recs)
            stats_all.append(st)

        self.data = records_all

        # Shuffle indices once (so __getitem__ can be deterministic w.r.t. idx)
        self.indices = torch.randperm(len(self.data)).tolist()

        # Compute scaler if requested
        if self.norm_mode == "global":
            if self.scaler is None and compute_scaler:
                self.scaler = self._compute_global_scaler(self.data)
            if self.scaler is None:
                raise ValueError(
                    "norm_mode='global' requires `scaler` or set compute_scaler=True (recommended only on train_dataset)."
                )

        # Write dataset-level stats if requested
        if save_stats_path is not None:
            self._save_dataset_stats(save_stats_path, stats_all, self.data)

        if self.verbose:
            print(f"✅ Loaded {len(self.data)} events from {len(root_files)} files (io_workers={self.io_workers})")

    @staticmethod
    def _load_file(
        file_path: str,
        branches: List[str],
        target_branch: List[str],
        processing_conditions: List[Dict[str, Any]],
        cuts: Dict[str, Any],  # cut字典用来叠加数据筛选
        keep_raw: bool,
        verbose: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load a single ROOT file and return:
          - list of event records
          - stats dict
        """
        stats = {
            "file": file_path,
            "n_total": 0,
            "n_kept": 0,
            "ok": True,
            "error": None,
        }

        # Keep this list as you had it (used for raw_info if keep_raw=True)
        full_branches = [
            "n", "nfit0", "nfit", "nfitb", "vnfit", "fitstat", "nrange",
            "nv", "vflag", "vidmc", "vx", "vy", "vt", "vnpe", "vq",
            "vqsamp", "theta", "phi", "xc", "yc", "dcedge", "dcedgepool",
            "istationcore", "ccindex", "chi2", "rmds", "pincness",
            "compactness", "f5w", "mc_weight", "mc_pid", "mc_energy",
            "mc_theta", "mc_phi", "mc_xc", "mc_yc", "mc_dangle",
            "mc_dcore"
        ]

        try:
            with uproot.open(file_path) as f:
                tree = f["t_eventout;1"]
                available_branches = set(tree.keys())
                if "fitstat" not in available_branches:
                    raise MissingRequiredBranchError(
                        f"Required branch 'fitstat' not found in ROOT tree for file: {file_path}"
                    )
                arrays = tree.arrays(full_branches, library="np")

            # n_total
            n_total = len(next(iter(arrays.values())))
            stats["n_total"] = int(n_total)
            if n_total == 0:
                return [], stats

            # Event-level cut ingredients
            mc_energy = arrays[target_branch[0]]
            pincness = arrays["pincness"]
            dcedge = arrays["dcedge"]
            mc_dangle = arrays["mc_dangle"]
            theta = arrays["theta"]
            mc_xc = arrays["mc_xc"]
            mc_yc = arrays["mc_yc"]
            fitstat = arrays["fitstat"]

            # vqsamp non-zero ratio (optional cut)
            vqsamp_ratio_min = cuts.get("vqsamp_ratio_min", None)
            if vqsamp_ratio_min is not None:
                vqsamp = arrays["vqsamp"]
                vqsamp_nonzero_ratio = np.array([
                    (np.count_nonzero(v > 0) / len(v)) if len(v) > 0 else 0.0
                    for v in vqsamp
                ], dtype=np.float32)
            else:
                vqsamp_nonzero_ratio = None

            # Build mask_evt
            mask_evt = np.ones(n_total, dtype=bool)

            Emin = cuts.get("Emin", None)
            if Emin is not None:
                mask_evt &= (mc_energy > float(Emin))

            Emax = cuts.get("Emax", None)
            if Emax is not None:
                mask_evt &= (mc_energy < float(Emax))

            pinc_max = cuts.get("pinc_max", None)
            if pinc_max is not None:
                mask_evt &= (pincness < float(pinc_max))

            dangle_max_rad = cuts.get("dangle_max_rad", None)
            if dangle_max_rad is not None:
                mask_evt &= (mc_dangle < float(dangle_max_rad))

            theta_max_rad = cuts.get("theta_max_rad", None)
            if theta_max_rad is not None:
                mask_evt &= (theta < float(theta_max_rad))

            dcedge_min = cuts.get("dcedge_min", None)
            if dcedge_min is not None:
                mask_evt &= (dcedge > float(dcedge_min))

            require_fitstat0 = bool(cuts.get("require_fitstat0", True))
            fitstat_equals = cuts.get("fitstat_equals", 0)
            if require_fitstat0:
                mask_evt &= (fitstat == int(fitstat_equals))

            use_core_box = bool(cuts.get("use_core_box", False))
            if use_core_box:
                xmin, xmax, ymin, ymax = cuts.get("core_box", (-130.0, 130.0, -110.0, 110.0))
                mask_evt &= (mc_xc >= xmin) & (mc_xc <= xmax) & (mc_yc >= ymin) & (mc_yc <= ymax)

            if vqsamp_nonzero_ratio is not None:
                mask_evt &= (vqsamp_nonzero_ratio >= float(vqsamp_ratio_min))

            # Apply mask to all branches
            for key in arrays.keys():
                arrays[key] = arrays[key][mask_evt]

            n_kept = int(np.sum(mask_evt))
            stats["n_kept"] = n_kept

            if verbose:
                msg = (
                    f"🔹 {file_path}: kept {n_kept}/{n_total} "
                    f"(E>{cuts.get('Emin')} pinc<{cuts.get('pinc_max')} dcedge>{cuts.get('dcedge_min')} "
                    f"dangle<{cuts.get('dangle_max_rad')} theta<{cuts.get('theta_max_rad')} "
                    f"fitstat={'0' if require_fitstat0 else 'ALL'} core_box={cuts.get('use_core_box')} "
                    f"vqsamp_ratio>={cuts.get('vqsamp_ratio_min')})"
                )
                print(msg)

            if n_kept == 0:
                return [], stats

            # Build per-event records
            n_events = len(next(iter(arrays.values())))
            records: List[Dict[str, Any]] = []

            for i in range(n_events):
                # (1) raw features for the selected branches
                features = np.column_stack([arrays[b][i] for b in branches])  # shape (Nhits, nfeat)
                features = process_features(features, processing_conditions)  # outlier removal / clipping etc.

                # If process_features removed everything -> skip event
                if features.shape[0] == 0:
                    continue

                # (2) points: (vx,vy) centered by reconstructed (xc,yc)
                vx, vy = features[:, 0], features[:, 1]
                xc, yc = arrays["xc"][i], arrays["yc"][i]
                points = np.column_stack([vx - xc, vy - yc]).astype(np.float32)

                # (3) features: use (vq, vt) -> columns [3] and [2] in your current design
                vq = features[:, 3].astype(np.float32)
                vt = features[:, 2].astype(np.float32)

                # (4) label log10(E)
                target = float(arrays["mc_energy"][i])
                log_energy = float(np.log10(target)) if target > 0 else 0.0

                # (5) mc_weight 
                mc_weight = float(arrays["mc_weight"][i])

                # (6) costheta (theta in rad)
                theta_evt = float(arrays["theta"][i])
                costheta_evt = float(np.cos(theta_evt))

                record = {
                    "file": file_path,
                    "event_idx": i,
                    "processed": {
                        "points": points,        # (Nhits,2)
                        "vq": vq,                # (Nhits,)
                        "vt": vt,                # (Nhits,)
                        "log_energy": log_energy,
                        "mc_weight": mc_weight,
                        "costheta": costheta_evt,
                    }
                }

                if keep_raw:
                    raw_info = {key: arrays[key][i] for key in full_branches}
                    record["raw"] = raw_info

                records.append(record)

            return records, stats

        except MissingRequiredBranchError:
            raise
        except Exception as e:
            stats["ok"] = False
            stats["error"] = str(e)
            if verbose:
                print(f"⚠️ File failed: {file_path} err={e}")
            return [], stats

    @staticmethod
    def _compute_global_scaler(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Compute global mean/std for vq/vt over all hits in `data`.
        Use ONLY on train_dataset (then pass scaler to val/test).
        """
        vq_all = []
        vt_all = []
        for rec in data:
            proc = rec["processed"]
            vq_all.append(proc["vq"])
            vt_all.append(proc["vt"])

        vq_cat = np.concatenate(vq_all, axis=0) if len(vq_all) else np.array([0.0], dtype=np.float32)
        vt_cat = np.concatenate(vt_all, axis=0) if len(vt_all) else np.array([0.0], dtype=np.float32)

        vq_m, vq_s = _safe_mean_std(vq_cat)
        vt_m, vt_s = _safe_mean_std(vt_cat)

        return {
            "vq": {"mean": float(vq_m), "std": float(vq_s)},
            "vt": {"mean": float(vt_m), "std": float(vt_s)},
        }

    @staticmethod
    def _save_dataset_stats(save_path: str, stats_all: List[Dict[str, Any]], data: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n_files = len(stats_all)
        n_fail = sum(0 if s.get("ok", True) else 1 for s in stats_all)
        n_total = sum(int(s.get("n_total", 0)) for s in stats_all)
        n_kept = sum(int(s.get("n_kept", 0)) for s in stats_all)

        # logE distribution
        logE = np.array([rec["processed"]["log_energy"] for rec in data], dtype=np.float32) if len(data) else np.array([], dtype=np.float32)
        if logE.size > 0:
            logE_stats = dict(
                count=int(logE.size),
                min=float(np.min(logE)),
                max=float(np.max(logE)),
                mean=float(np.mean(logE)),
                std=float(np.std(logE)),
            )
        else:
            logE_stats = dict(count=0, min=None, max=None, mean=None, std=None)

        payload = dict(
            files=dict(n_files=n_files, n_fail=n_fail),
            events=dict(n_total=n_total, n_kept=n_kept, keep_ratio=(float(n_kept) / float(n_total) if n_total > 0 else None)),
            log_energy=logE_stats,
        )

        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)

    def __len__(self) -> int:
        return len(self.data)
    
    # 不同定义的标准化
    def _normalize(self, vq: np.ndarray, vt: np.ndarray) -> np.ndarray:
        """
        Return features array of shape (Nhits, 2) as [vq_norm, vt_norm].
        """
        if self.norm_mode == "none":
            return np.column_stack([vq, vt]).astype(np.float32)

        if self.norm_mode == "per_event": # 逐个事例的标准化
            vq_m, vq_s = _safe_mean_std(vq)
            vt_m, vt_s = _safe_mean_std(vt)
            vq_n = (vq - vq_m) / vq_s
            vt_n = (vt - vt_m) / vt_s
            return np.column_stack([vq_n, vt_n]).astype(np.float32)

        if self.norm_mode == "global": # 用训练集全集的标准化
            vq_m = float(self.scaler["vq"]["mean"])
            vq_s = float(self.scaler["vq"]["std"])
            vt_m = float(self.scaler["vt"]["mean"])
            vt_s = float(self.scaler["vt"]["std"])
            vq_n = (vq - vq_m) / (vq_s + 1e-8)
            vt_n = (vt - vt_m) / (vt_s + 1e-8)
            return np.column_stack([vq_n, vt_n]).astype(np.float32)

        raise ValueError(f"Unknown norm_mode: {self.norm_mode}")
    
    # 定义不同模式的hit截断
    def _select_hits(self, points: np.ndarray, vq: np.ndarray, vt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply truncation strategy when Nhits > max_points.
        """
        n_points = points.shape[0]
        if n_points <= self.max_points:
            return points, vq, vt

        k = self.max_points

        if self.sample_mode == "random": # 随机采样
            idxs = self.rng.choice(n_points, k, replace=False)
        elif self.sample_mode == "firstk": # 取前k个值
            idxs = np.arange(k)
        elif self.sample_mode == "topk_q": # 按vq取最大的k个
            idxs = np.argsort(vq)[-k:]
        elif self.sample_mode == "weighted_q": # 按vq做概率采样
            # sample with prob proportional to positive vq (fallback to uniform)
            w = np.clip(vq, a_min=0.0, a_max=None)
            s = float(np.sum(w))
            if s <= 0:
                idxs = self.rng.choice(n_points, k, replace=False)
            else:
                p = w / s
                idxs = self.rng.choice(n_points, k, replace=False, p=p)
        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")

        return points[idxs], vq[idxs], vt[idxs]

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        record = self.data[real_idx]

        proc = record["processed"]
        points = proc["points"]     # (Nhits,2)
        vq = proc["vq"]             # (Nhits,)
        vt = proc["vt"]             # (Nhits,)
        log_energy = proc["log_energy"]
        mc_weight = proc["mc_weight"]
        costheta = proc["costheta"]

        # Truncate strategy if too many hits
        points, vq, vt = self._select_hits(points, vq, vt)

        # Normalize features -> (Nhits,2)
        features = self._normalize(vq, vt)

        n_points = points.shape[0]

        # Padding if too few hits
        if n_points < self.max_points:
            pad_len = self.max_points - n_points
            pad_points = np.zeros((pad_len, 2), dtype=np.float32)
            pad_features = np.zeros((pad_len, 2), dtype=np.float32)

            points = np.vstack([points, pad_points]).astype(np.float32)
            features = np.vstack([features, pad_features]).astype(np.float32)
            mask = np.concatenate([np.ones(n_points, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
        else:
            mask = np.ones(self.max_points, dtype=np.float32)

        # To torch: transpose to (C,N)
        points_t = torch.tensor(points, dtype=torch.float32).T          # (2, N)
        features_t = torch.tensor(features, dtype=torch.float32).T      # (2, N)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)   # (1, N)
        log_energy_t = torch.tensor(log_energy, dtype=torch.float32).unsqueeze(-1)  # (1,)
        mc_weight_t = torch.tensor(mc_weight, dtype=torch.float32).unsqueeze(-1)  # (1,)
        costheta_t = torch.tensor(costheta, dtype=torch.float32).unsqueeze(-1)  # (1,)
        
        return points_t, features_t, mask_t, costheta_t, log_energy_t, mc_weight_t
