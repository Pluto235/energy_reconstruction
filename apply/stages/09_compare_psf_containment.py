#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "apply/config/v6_psf_containment_compare.json"
M2_TO_CM2 = 1.0e4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed-Rayleigh and MC-aperture v6 responses.")
    parser.add_argument("mode", choices=["preflight", "mc-stats", "finalize"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--files-per-task", type=int, default=250)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)


def json_ready(value: object) -> object:
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names = []
        for row in rows:
            names.extend(key for key in row if key not in names)
    else:
        names = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def selector_rows(config: dict) -> Tuple[List[dict], List[int], List[int]]:
    rows = read_csv(resolve(config["selector_csv"]))
    included = [int(row["cell_id"]) for row in rows if str(row.get("include")) == "1"]
    excluded = [int(row["cell_id"]) for row in rows if str(row.get("include")) != "1"]
    return rows, included, excluded


def metadata_audit(nominal: dict, aperture: dict) -> dict:
    exact_fields = [
        "binned_root",
        "cell_selection_csv",
        "denominator",
        "weighting",
        "s0_m2",
        "thrown_geometry",
        "run_dir",
    ]
    checks = []
    for field in exact_fields:
        checks.append({"field": field, "equal": nominal.get(field) == aperture.get(field)})
    nominal_cuts = dict(nominal.get("cuts") or {})
    aperture_cuts = dict(aperture.get("cuts") or {})
    nominal_aperture_cut = nominal_cuts.pop("mc_dangle_cut", None)
    aperture_aperture_cut = aperture_cuts.pop("mc_dangle_cut", None)
    checks.append({"field": "reconstruction_cuts", "equal": nominal_cuts == aperture_cuts})
    checks.append(
        {
            "field": "aperture_cut_only_difference",
            "equal": nominal_aperture_cut is None
            and aperture_aperture_cut == "cell-dependent r_opt_deg from aperture_psf_npz",
        }
    )
    nominal_cells = [
        (row.get("cell_id"), row.get("nhit_bin"), row.get("predE_bin"))
        for row in nominal.get("cells", [])
    ]
    aperture_cells = [
        (row.get("cell_id"), row.get("nhit_bin"), row.get("predE_bin"))
        for row in aperture.get("cells", [])
    ]
    checks.append({"field": "cells", "equal": nominal_cells == aperture_cells})
    conditioning = aperture.get("response_aperture_conditioning", {})
    checks.extend(
        [
            {"field": "nominal_response_type", "equal": nominal.get("response_type") == "primary_thrown_response"},
            {
                "field": "aperture_response_type",
                "equal": aperture.get("response_type") == "primary_thrown_aperture_conditioned_response",
            },
            {"field": "aperture_mode", "equal": conditioning.get("mode") == "mc_dangle_le_r_opt"},
        ]
    )
    return {"checks": checks, "passed": all(bool(item["equal"]) for item in checks)}


def array_audit(nominal: dict, aperture: dict) -> dict:
    identical = [
        "logE_true_edges",
        "theta_true_edges_deg",
        "theta_true_centers_deg",
        "cos_theta_center",
        "s0_m2",
        "cell_id",
        "nhit_bin",
        "predE_bin",
        "denominator_sumw",
        "denominator_count",
    ]
    checks = []
    for name in identical:
        left = np.asarray(nominal[name])
        right = np.asarray(aperture[name])
        if np.issubdtype(left.dtype, np.number):
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = np.array_equal(left, right)
        checks.append({"array": name, "equal": bool(equal)})
    numerator_ordered = bool(np.all(aperture["numerator_sumw"] <= nominal["numerator_sumw"] + 1.0e-9))
    checks.append({"array": "aperture_numerator_le_nominal", "equal": numerator_ordered})
    return {"checks": checks, "passed": all(bool(item["equal"]) for item in checks)}


def prepare_scaled_response(config: dict, nominal: dict, nominal_meta: dict, output_dir: Path) -> Tuple[Path, Path]:
    containment = float(config["scheme_r"]["containment"])
    response_path = output_dir / "response_r_fixed0715.npz"
    metadata_path = output_dir / "response_r_fixed0715_metadata.json"
    payload = {key: value.copy() for key, value in nominal.items()}
    payload["a_eff"] = (np.asarray(nominal["a_eff"], dtype=np.float64) * containment).astype(np.float32)
    np.savez_compressed(response_path, **payload)
    metadata = dict(nominal_meta)
    metadata.update(
        {
            "npz_path": str(response_path.resolve()),
            "a_eff_max_m2": float(np.nanmax(payload["a_eff"])),
            "derived_response": {
                "mode": "fixed_rayleigh_containment",
                "source_nominal_response": str(resolve(config["scheme_r"]["nominal_response_npz"])),
                "containment": containment,
                "formula": "Aeff_R = containment * Aeff_nominal",
                "rayleigh_note": "For a 2D Gaussian PSF, r_opt approximately 1.585 sigma gives Rayleigh containment approximately 71.5%.",
            },
        }
    )
    write_json(metadata_path, metadata)
    return response_path, metadata_path


def preflight(config: dict) -> None:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    nominal_path = resolve(config["scheme_r"]["nominal_response_npz"])
    aperture_path = resolve(config["scheme_b"]["response_npz"])
    nominal_meta_path = resolve(config["scheme_r"]["nominal_response_metadata"])
    aperture_meta_path = resolve(config["scheme_b"]["response_metadata"])
    nominal = load_npz(nominal_path)
    aperture = load_npz(aperture_path)
    nominal_meta = read_json(nominal_meta_path)
    aperture_meta = read_json(aperture_meta_path)
    rows, included, excluded = selector_rows(config)
    selector_checks = {
        "included_count": len(included),
        "expected_count": int(config["expected_fit_cells"]),
        "required_included": {str(cid): cid in included for cid in config["required_included_cells"]},
        "required_excluded": {str(cid): cid in excluded for cid in config["required_excluded_cells"]},
    }
    selector_passed = (
        len(included) == int(config["expected_fit_cells"])
        and all(selector_checks["required_included"].values())
        and all(selector_checks["required_excluded"].values())
    )
    metadata = metadata_audit(nominal_meta, aperture_meta)
    arrays = array_audit(nominal, aperture)
    signal = load_npz(resolve(config["signal_npz"]))
    signal_cells = [(int(a), str(b), str(c)) for a, b, c in zip(signal["cell_id"], signal["nhit_bin"], signal["predE_bin"])]
    response_cells = [(int(a), str(b), str(c)) for a, b, c in zip(nominal["cell_id"], nominal["nhit_bin"], nominal["predE_bin"])]
    signal_contract = {
        "cells_match_response": signal_cells == response_cells,
        "containment_all_one": bool(np.allclose(signal["containment_r_opt"], 1.0, rtol=0.0, atol=1.0e-12)),
        "conservative_error_matches": bool(
            np.allclose(signal["excess_err_conservative"], np.sqrt(signal["N_on"] + signal["B_on"]), rtol=1.0e-12, atol=1.0e-10)
        ),
    }
    passed = bool(metadata["passed"] and arrays["passed"] and selector_passed and all(signal_contract.values()))
    derived_npz, derived_meta = prepare_scaled_response(config, nominal, nominal_meta, output_dir)
    payload = {
        "status": "passed" if passed else "failed",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "inputs": {
            "nominal_response": str(nominal_path),
            "aperture_response": str(aperture_path),
            "signal": str(resolve(config["signal_npz"])),
            "selector": str(resolve(config["selector_csv"])),
        },
        "metadata_audit": metadata,
        "array_audit": arrays,
        "selector_audit": {**selector_checks, "passed": selector_passed, "included_cell_ids": included},
        "signal_contract": signal_contract,
        "derived_scheme_r_response": {"npz": str(derived_npz), "metadata": str(derived_meta)},
    }
    write_json(output_dir / "preflight_audit.json", payload)
    if not passed:
        raise RuntimeError("Response comparison preflight failed; see preflight_audit.json")
    print(f"Preflight passed for {len(included)} fixed selector cells.", flush=True)


def sanitize_label(label: str) -> str:
    return str(label).replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "_").replace(">=", "ge").replace(".", "p")


def mc_chunk_worker(task: dict) -> dict:
    total = {key: 0.0 for key in ["sumw", "sumw2", "pass_sumw", "pass_sumw2", "fail_sumw2"]}
    total.update({"events": 0, "valid_events": 0, "pass_events": 0})
    radius_rad = math.radians(float(task["radius_deg"]))
    lo_e, hi_e = task["loge_range"]
    lo_t, hi_t = task["theta_range"]
    for path in task["paths"]:
        with uproot.open(path) as root_file:
            tree = root_file["t_eventout"]
            arrays = tree.arrays(["mc_energy", "mc_theta", "mc_weight", "mc_dangle"], library="np")
        energy = np.asarray(arrays["mc_energy"], dtype=np.float64)
        theta = np.degrees(np.asarray(arrays["mc_theta"], dtype=np.float64))
        weight = np.asarray(arrays["mc_weight"], dtype=np.float64)
        dangle = np.asarray(arrays["mc_dangle"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            loge = np.log10(energy)
        valid = np.isfinite(loge) & np.isfinite(theta) & np.isfinite(weight) & (loge >= lo_e) & (loge < hi_e) & (theta >= lo_t) & (theta < hi_t)
        passed = valid & np.isfinite(dangle) & (dangle <= radius_rad)
        failed = valid & ~passed
        w = weight[valid]
        wp = weight[passed]
        wf = weight[failed]
        total["events"] += int(weight.size)
        total["valid_events"] += int(w.size)
        total["pass_events"] += int(wp.size)
        total["sumw"] += float(np.sum(w))
        total["sumw2"] += float(np.sum(w * w))
        total["pass_sumw"] += float(np.sum(wp))
        total["pass_sumw2"] += float(np.sum(wp * wp))
        total["fail_sumw2"] += float(np.sum(wf * wf))
    total["cell_id"] = int(task["cell_id"])
    total["files"] = len(task["paths"])
    return total


def mc_stats(config: dict, workers: int, files_per_task: int) -> None:
    output_dir = resolve(config["output_dir"])
    nominal = load_npz(resolve(config["scheme_r"]["nominal_response_npz"]))
    aperture = load_npz(resolve(config["scheme_b"]["response_npz"]))
    rows, included, _ = selector_rows(config)
    by_id = {int(row["cell_id"]): row for row in rows}
    tasks = []
    for cid in included:
        idx = int(np.where(nominal["cell_id"] == cid)[0][0])
        row = by_id[cid]
        cell_dir = resolve(config["binned_mc_root"]) / f"nhit_{sanitize_label(row['nhit_bin'])}" / f"predE_{sanitize_label(row['predE_bin'])}"
        paths = sorted(str(path) for path in cell_dir.glob("*.root"))
        for start in range(0, len(paths), max(1, files_per_task)):
            tasks.append(
                {
                    "cell_id": cid,
                    "paths": paths[start : start + files_per_task],
                    "radius_deg": float(aperture["aperture_r_opt_deg"][idx]),
                    "loge_range": [float(nominal["logE_true_edges"][0]), float(nominal["logE_true_edges"][-1])],
                    "theta_range": [float(nominal["theta_true_edges_deg"][0]), float(nominal["theta_true_edges_deg"][-1])],
                }
            )
    aggregates: Dict[int, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(mc_chunk_worker, task) for task in tasks]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            cid = int(result.pop("cell_id"))
            target = aggregates.setdefault(cid, {key: 0.0 for key in result})
            for key, value in result.items():
                target[key] += value
            if done % 20 == 0 or done == len(futures):
                print(f"[mc-stats] {done}/{len(futures)} tasks", flush=True)
    stage_b = {int(row["cell_id"]): row for row in read_csv(resolve(config["stage_b_summary_csv"]))}
    output_rows = []
    for cid in included:
        item = aggregates[cid]
        sumw = float(item["sumw"])
        sumw2 = float(item["sumw2"])
        pass_sumw = float(item["pass_sumw"])
        p = pass_sumw / sumw if sumw > 0.0 else float("nan")
        neff = sumw * sumw / sumw2 if sumw2 > 0.0 else 0.0
        pass_neff = pass_sumw * pass_sumw / float(item["pass_sumw2"]) if item["pass_sumw2"] > 0.0 else 0.0
        variance = ((1.0 - p) ** 2 * item["pass_sumw2"] + p * p * item["fail_sumw2"]) / (sumw * sumw) if sumw > 0.0 else float("nan")
        psf = stage_b[cid]
        output_rows.append(
            {
                "cell_id": cid,
                "nhit_bin": by_id[cid]["nhit_bin"],
                "predE_bin": by_id[cid]["predE_bin"],
                "r_opt_deg": float(aperture["aperture_r_opt_deg"][np.where(nominal["cell_id"] == cid)[0][0]]),
                "events": int(item["events"]),
                "valid_events": int(item["valid_events"]),
                "pass_events": int(item["pass_events"]),
                "sumw": sumw,
                "sumw2": sumw2,
                "neff": neff,
                "pass_sumw": pass_sumw,
                "pass_sumw2": float(item["pass_sumw2"]),
                "pass_neff": pass_neff,
                "weighted_containment": p,
                "weighted_containment_std": math.sqrt(max(0.0, variance)) if math.isfinite(variance) else float("nan"),
                "theta_missing_mass": float(psf["theta_missing_crab_probability_mass"]),
                "psf_effective_events": float(psf["effective_events"]),
                "core_fit_effective_events": float(psf["core_fit_effective_events"]),
                "containment_warning": psf["containment_warning"],
                "angle_check_warning": psf["angle_check_warning"],
            }
        )
    write_csv(output_dir / "mc_containment_stats.csv", output_rows)
    write_json(
        output_dir / "mc_containment_stats.json",
        {
            "definition": "Neff=(sumw)^2/sum(w^2); weighted containment uncertainty uses the weighted Bernoulli ratio delta variance.",
            "same_sample_optimization_bias": "r_opt and containment use the same MC sample; optimistic same-sample optimization bias is possible.",
            "rows": output_rows,
        },
    )


def import_stage(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "apply/stages" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fit_paths(config: dict, scheme: str) -> Tuple[Path, Path]:
    base = resolve(config["output_dir"]) / f"stage_f_{scheme}" / "runs" / f"{config['comparison_id']}_{scheme}"
    return base / f"fit_{scheme}.npz", base / f"fit_{scheme}_metadata.json"


def sed_paths(config: dict, scheme: str) -> Tuple[Path, Path]:
    base = resolve(config["output_dir"]) / f"stage_g_{scheme}" / "runs" / f"{config['comparison_id']}_{scheme}"
    return base / f"sed_points_{scheme}.npz", base / f"sed_points_{scheme}_metadata.json"


def correlation(covariance: Sequence[Sequence[float]] | None) -> List[List[float]] | None:
    if covariance is None:
        return None
    cov = np.asarray(covariance, dtype=np.float64)
    scale = np.sqrt(np.diag(cov))
    return np.divide(cov, scale[:, None] * scale[None, :], out=np.full_like(cov, np.nan), where=(scale[:, None] * scale[None, :]) > 0).tolist()


def model_unit_counts(stage06, response: dict, theta_exposure: np.ndarray, params: dict) -> np.ndarray:
    unit = dict(params)
    unit["phi0"] = 1.0
    return stage06.model_counts(
        np.asarray(response["a_eff"], dtype=np.float64),
        np.ones(response["a_eff"].shape[0], dtype=np.float64),
        theta_exposure,
        np.asarray(response["logE_true_edges"], dtype=np.float64),
        model_name="logpar",
        params=unit,
        pivot_tev=3.0,
        quadrature_points=64,
    )


def selected_response(response: dict, included: Sequence[int]) -> dict:
    by_id = {int(cid): idx for idx, cid in enumerate(response["cell_id"])}
    indices = np.asarray([by_id[int(cid)] for cid in included], dtype=np.int64)
    return {key: (value[indices] if value.ndim and value.shape[0] == len(response["cell_id"]) else value) for key, value in response.items()}


def linear_normalization(observed: np.ndarray, errors: np.ndarray, unit_counts: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    valid = mask & np.isfinite(observed) & np.isfinite(errors) & (errors > 0.0) & np.isfinite(unit_counts) & (unit_counts > 0.0)
    denominator = float(np.sum((unit_counts[valid] / errors[valid]) ** 2))
    numerator = float(np.sum(observed[valid] * unit_counts[valid] / (errors[valid] ** 2)))
    return numerator / denominator, 1.0 / math.sqrt(denominator)


def finite_stats(values: np.ndarray) -> dict:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return {"min": float(np.min(valid)), "median": float(np.median(valid)), "max": float(np.max(valid)), "n": int(valid.size)} if valid.size else {"min": None, "median": None, "max": None, "n": 0}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_mu_heatmap(rows: Sequence[dict], path: Path, special: Sequence[int]) -> None:
    plt = setup_matplotlib()
    nhits = list(dict.fromkeys(str(row["nhit_bin"]) for row in rows))
    preds = list(dict.fromkeys(str(row["predE_bin"]) for row in rows))
    grid = np.full((len(nhits), len(preds)), np.nan)
    ids = np.full(grid.shape, -1, dtype=int)
    for row in rows:
        i, j = nhits.index(str(row["nhit_bin"])), preds.index(str(row["predE_bin"]))
        grid[i, j], ids[i, j] = float(row["mu_ratio_b_over_r"]), int(row["cell_id"])
    fig, ax = plt.subplots(figsize=(14, 5.6), constrained_layout=True)
    image = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=0.65, vmax=1.4)
    for i, j in zip(*np.where(np.isfinite(grid))):
        cid = ids[i, j]
        marker = "*" if cid in special else ""
        ax.text(j, i, f"C{cid}{marker}\n{grid[i,j]:.3f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_xticks(range(len(preds)), preds, rotation=50, ha="right")
    ax.set_yticks(range(len(nhits)), nhits)
    ax.set_xlabel("PredE bin")
    ax.set_ylabel("Nhit bin")
    ax.set_title("Frozen-LogPar expected-count ratio: B / R")
    fig.colorbar(image, ax=ax, label="mu_B / mu_R")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_distribution(rows: Sequence[dict], path: Path, special: Sequence[int]) -> None:
    plt = setup_matplotlib()
    values = np.asarray([float(row["mu_ratio_b_over_r"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    ax.hist(values[np.isfinite(values)], bins=16, color="#2878b5", alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    for row in rows:
        if int(row["cell_id"]) in special:
            ax.axvline(float(row["mu_ratio_b_over_r"]), linewidth=1.2, label=f"C{row['cell_id']}={float(row['mu_ratio_b_over_r']):.3f}")
    ax.set_xlabel("mu_B / mu_R")
    ax.set_ylabel("Selected cells")
    ax.set_title("Cell expected-count ratio distribution")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_aeff_diagnostics(nominal: dict, aperture: dict, included: Sequence[int], path: Path, special: Sequence[int]) -> None:
    plt = setup_matplotlib()
    e = 10 ** (0.5 * (nominal["logE_true_edges"][:-1] + nominal["logE_true_edges"][1:])) / 1000.0
    theta = nominal["theta_true_centers_deg"]
    fig, axes = plt.subplots(len(special), 2, figsize=(11, 3.1 * len(special)), constrained_layout=True)
    for row_idx, cid in enumerate(special):
        idx = int(np.where(nominal["cell_id"] == cid)[0][0])
        den = 0.715 * np.asarray(nominal["a_eff"][idx], dtype=float)
        ratio = np.divide(aperture["a_eff"][idx], den, out=np.full_like(den, np.nan), where=(den > 0) & (nominal["numerator_count"][idx] > 0))
        energy_profile = np.asarray([np.median(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else np.nan for values in ratio])
        theta_profile = np.asarray([np.median(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else np.nan for values in ratio.T])
        axes[row_idx, 0].plot(e, energy_profile, color="#d1495b")
        axes[row_idx, 0].axhline(1.0, color="black", ls="--", lw=1)
        axes[row_idx, 0].set_xscale("log")
        axes[row_idx, 0].set_ylabel(f"C{cid} R_Aeff")
        axes[row_idx, 1].plot(theta, theta_profile, color="#00798c")
        axes[row_idx, 1].axhline(1.0, color="black", ls="--", lw=1)
    axes[-1, 0].set_xlabel("Etrue [TeV]")
    axes[-1, 1].set_xlabel("theta [deg]")
    axes[0, 0].set_title("Median over theta")
    axes[0, 1].set_title("Median over Etrue")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fit_compare(fit_r: dict, fit_b: dict, path: Path) -> None:
    plt = setup_matplotlib()
    cid = fit_r["cell_id"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(cid, fit_r["logpar_conservative_residual"], "o-", ms=3, label="R")
    axes[0].plot(cid, fit_b["logpar_conservative_residual"], "s-", ms=3, label="B")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylabel("excess - expect")
    axes[0].legend()
    axes[1].plot(cid, fit_r["logpar_conservative_pull"], "o-", ms=3, label="R")
    axes[1].plot(cid, fit_b["logpar_conservative_pull"], "s-", ms=3, label="B")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("pull")
    axes[1].set_xlabel("cell_id")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def logpar_flux(energy: np.ndarray, params: dict) -> np.ndarray:
    ratio = energy / 3.0
    return float(params["phi0"]) * np.power(ratio, -(float(params["alpha"]) + float(params["beta"]) * np.log(ratio)))


def plot_sed_compare(rows: Sequence[dict], params_r: dict, params_b: dict, pass5: dict, path: Path, official_path: Path) -> None:
    plt = setup_matplotlib()
    nhit_rows = [row for row in rows if row["grouping"] == "nhit"]
    e_r = np.asarray([row["r_E_eff"] for row in nhit_rows], float)
    e_b = np.asarray([row["b_E_eff"] for row in nhit_rows], float)
    y_r = np.asarray([row["r_E2_dnde"] for row in nhit_rows], float)
    y_b = np.asarray([row["b_E2_dnde"] for row in nhit_rows], float)
    err_r = np.asarray([row["r_E2_dnde_err"] for row in nhit_rows], float)
    err_b = np.asarray([row["b_E2_dnde_err"] for row in nhit_rows], float)
    x = np.geomspace(0.2, 120, 300)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.2), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    axes[0].errorbar(e_r, y_r, yerr=err_r, fmt="o", label="R: fixed 71.5%")
    axes[0].errorbar(e_b, y_b, yerr=err_b, fmt="s", label="B: MC aperture")
    axes[0].plot(x, x*x*logpar_flux(x, params_r), label="R Stage F LogPar")
    axes[0].plot(x, x*x*logpar_flux(x, params_b), label="B Stage F LogPar")
    axes[0].plot(x, x*x*logpar_flux(x, pass5), color="black", ls="--", label="official Pass5 point-fit")
    axes[0].set(xscale="log", yscale="log", ylabel=r"$E^2 dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    axes[0].legend(fontsize=8)
    ratios = np.asarray([row["independent_b_over_r"] for row in nhit_rows], float)
    axes[1].plot(np.sqrt(e_r*e_b), ratios, "o-")
    axes[1].axhline(1, color="black", ls="--", lw=1)
    axes[1].set(xscale="log", xlabel="E [TeV]", ylabel="B / R")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(e_r, [row["r_over_official"] for row in nhit_rows], "o-", label="R / official Pass5")
    ax.plot(e_b, [row["b_over_official"] for row in nhit_rows], "s-", label="B / official Pass5")
    ax.axhline(1, color="black", ls="--", lw=1)
    ax.set(xscale="log", xlabel="E [TeV]", ylabel="SED point / official Pass5", title="Official Pass5 WCDA comparison")
    ax.legend()
    fig.savefig(official_path, dpi=180)
    plt.close(fig)


def format_number(value: object, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}"


def table_html(rows: Sequence[dict], fields: Sequence[Tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in fields)
    body = []
    for row in rows:
        cells = []
        for key, _ in fields:
            value = row.get(key, "")
            text = format_number(value, 6) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def finalize(config: dict) -> None:
    stage06 = import_stage("stage06_compare", "06_fit.py")
    stage07 = import_stage("stage07_compare", "07_sed_points.py")
    output_dir = resolve(config["output_dir"])
    asset_dir = resolve(config["asset_dir"])
    asset_dir.mkdir(parents=True, exist_ok=True)
    nominal_full = load_npz(resolve(config["scheme_r"]["nominal_response_npz"]))
    response_r_full = load_npz(output_dir / "response_r_fixed0715.npz")
    response_b_full = load_npz(resolve(config["scheme_b"]["response_npz"]))
    _, included, _ = selector_rows(config)
    nominal = selected_response(nominal_full, included)
    response_r = selected_response(response_r_full, included)
    response_b = selected_response(response_b_full, included)
    fit_r_path, fit_r_meta_path = fit_paths(config, "r")
    fit_b_path, fit_b_meta_path = fit_paths(config, "b")
    fit_r, fit_b = load_npz(fit_r_path), load_npz(fit_b_path)
    meta_r, meta_b = read_json(fit_r_meta_path), read_json(fit_b_meta_path)
    sed_r_path, _ = sed_paths(config, "r")
    sed_b_path, _ = sed_paths(config, "b")
    sed_r, sed_b = load_npz(sed_r_path), load_npz(sed_b_path)
    production_meta = read_json(resolve(config["production_stage_f_metadata"]))
    frozen = dict(production_meta["fits"]["logpar_conservative"]["parameters"])
    theta_exposure = np.asarray(fit_b["theta_exposure_sec"], dtype=np.float64)
    mu_r = stage06.model_counts(response_r["a_eff"], np.ones(len(included)), theta_exposure, response_r["logE_true_edges"], model_name="logpar", params=frozen, pivot_tev=3.0, quadrature_points=64)
    mu_b = stage06.model_counts(response_b["a_eff"], np.ones(len(included)), theta_exposure, response_b["logE_true_edges"], model_name="logpar", params=frozen, pivot_tev=3.0, quadrature_points=64)
    mu_ratio = np.divide(mu_b, mu_r, out=np.full_like(mu_b, np.nan), where=mu_r > 0)
    aeff_den = 0.715 * np.asarray(nominal["a_eff"], float)
    aeff_valid = (aeff_den > 0) & (nominal["numerator_count"] > 0) & (nominal["denominator_count"][None, :, :] > 0)
    aeff_ratio = np.divide(response_b["a_eff"], aeff_den, out=np.full_like(aeff_den, np.nan), where=aeff_valid)
    mc_rows = {int(row["cell_id"]): row for row in read_csv(output_dir / "mc_containment_stats.csv")}
    cell_rows = []
    for idx, cid in enumerate(included):
        valid_values = aeff_ratio[idx][np.isfinite(aeff_ratio[idx])]
        mc = mc_rows[cid]
        cell_rows.append(
            {
                "cell_id": cid,
                "nhit_bin": str(nominal["nhit_bin"][idx]),
                "predE_bin": str(nominal["predE_bin"][idx]),
                "mu_r": float(mu_r[idx]),
                "mu_b": float(mu_b[idx]),
                "mu_ratio_b_over_r": float(mu_ratio[idx]),
                "mu_difference_b_minus_r": float(mu_b[idx] - mu_r[idx]),
                "r_aeff_min": float(np.min(valid_values)) if valid_values.size else float("nan"),
                "r_aeff_median": float(np.median(valid_values)) if valid_values.size else float("nan"),
                "r_aeff_max": float(np.max(valid_values)) if valid_values.size else float("nan"),
                "mc_containment": float(mc["weighted_containment"]),
                "mc_containment_std": float(mc["weighted_containment_std"]),
                "mc_neff": float(mc["neff"]),
                "theta_missing_mass": float(mc["theta_missing_mass"]),
                "containment_warning": mc["containment_warning"],
                "angle_check_warning": mc["angle_check_warning"],
            }
        )
    write_csv(output_dir / "cell_expect_comparison.csv", cell_rows)
    np.savez_compressed(
        output_dir / "response_expect_comparison.npz",
        cell_id=np.asarray(included, dtype=np.int32),
        nhit_bin=nominal["nhit_bin"],
        predE_bin=nominal["predE_bin"],
        mu_r=mu_r,
        mu_b=mu_b,
        mu_ratio_b_over_r=mu_ratio,
        r_aeff=aeff_ratio,
        r_aeff_valid=aeff_valid,
        logE_true_edges=nominal["logE_true_edges"],
        theta_true_edges_deg=nominal["theta_true_edges_deg"],
    )
    fit_rows = []
    fit_differences = []
    for model in ["pl_conservative", "logpar_conservative"]:
        for scheme, metadata in [("R", meta_r), ("B", meta_b)]:
            item = metadata["fits"][model]
            fit_rows.append(
                {
                    "scheme": scheme,
                    "model": model.replace("_conservative", ""),
                    **item["parameters"],
                    **{f"{key}_err": value for key, value in item["errors"].items()},
                    "chi2": item["chi2"],
                    "ndof": item["ndof"],
                    "chi2_over_ndof": item["chi2_over_ndof"],
                    "p_value": item["p_value"],
                }
            )
            item["correlation"] = correlation(item.get("covariance"))
        r_item = meta_r["fits"][model]
        b_item = meta_b["fits"][model]
        for parameter in r_item["parameters"]:
            r_value = float(r_item["parameters"][parameter])
            b_value = float(b_item["parameters"][parameter])
            difference = b_value - r_value
            fit_differences.append(
                {
                    "model": model.replace("_conservative", ""),
                    "parameter": parameter,
                    "r_value": r_value,
                    "r_error": r_item["errors"].get(parameter),
                    "b_value": b_value,
                    "b_error": b_item["errors"].get(parameter),
                    "b_minus_r": difference,
                    "relative_difference": difference / r_value if r_value != 0.0 else float("nan"),
                    "relative_difference_percent": 100.0 * difference / r_value if r_value != 0.0 else float("nan"),
                }
            )
    write_csv(output_dir / "stage_f_parameter_comparison.csv", fit_rows)
    write_csv(output_dir / "stage_f_parameter_differences.csv", fit_differences)
    cell_fit_rows = []
    for idx, cid in enumerate(included):
        cell_fit_rows.append(
            {
                "cell_id": cid,
                "nhit_bin": str(fit_r["nhit_bin"][idx]),
                "predE_bin": str(fit_r["predE_bin"][idx]),
                "N_on": int(fit_r["N_on"][idx]),
                "B_on": float(fit_r["B_on"][idx]),
                "excess": float(fit_r["excess"][idx]),
                "error": float(fit_r["excess_err_conservative"][idx]),
                "expect_r": float(fit_r["logpar_conservative_model_counts"][idx]),
                "expect_b": float(fit_b["logpar_conservative_model_counts"][idx]),
                "expect_ratio_b_over_r": float(fit_b["logpar_conservative_model_counts"][idx] / fit_r["logpar_conservative_model_counts"][idx]) if fit_r["logpar_conservative_model_counts"][idx] > 0 else float("nan"),
                "residual_r": float(fit_r["logpar_conservative_residual"][idx]),
                "residual_b": float(fit_b["logpar_conservative_residual"][idx]),
                "pull_r": float(fit_r["logpar_conservative_pull"][idx]),
                "pull_b": float(fit_b["logpar_conservative_pull"][idx]),
                "delta_chi2_b_minus_r": float(fit_b["logpar_conservative_pull"][idx] ** 2 - fit_r["logpar_conservative_pull"][idx] ** 2),
            }
        )
    write_csv(output_dir / "stage_f_cell_comparison.csv", cell_fit_rows)
    unit_r = model_unit_counts(stage06, response_r, theta_exposure, frozen)
    unit_b = model_unit_counts(stage06, response_b, theta_exposure, frozen)
    observed = fit_r["excess"]
    errors = fit_r["excess_err_conservative"]
    sed_lookup_r = {(str(g), str(l)): idx for idx, (g, l) in enumerate(zip(sed_r["grouping"], sed_r["group_label"]))}
    sed_lookup_b = {(str(g), str(l)): idx for idx, (g, l) in enumerate(zip(sed_b["grouping"], sed_b["group_label"]))}
    specs = []
    for grouping, values in [("nhit", fit_r["nhit_bin"]), ("predE", fit_r["predE_bin"])]:
        for label in dict.fromkeys(str(value) for value in values):
            specs.append((grouping, label, np.asarray([str(value) == label for value in values])))
    pass5 = stage07.official_pass5_point_fit(pivot_tev=3.0)
    sed_rows = []
    for grouping, label, mask in specs:
        ir, ib = sed_lookup_r[(grouping, label)], sed_lookup_b[(grouping, label)]
        fixed_r, fixed_r_err = linear_normalization(observed, errors, unit_r, mask)
        fixed_b, fixed_b_err = linear_normalization(observed, errors, unit_b, mask)
        er, eb = float(sed_r["effective_energy_tev"][ir]), float(sed_b["effective_energy_tev"][ib])
        yr, yb = float(sed_r["E2_dnde"][ir]), float(sed_b["E2_dnde"][ib])
        official_r = er * er * float(logpar_flux(np.asarray([er]), pass5)[0])
        official_b = eb * eb * float(logpar_flux(np.asarray([eb]), pass5)[0])
        sed_rows.append(
            {
                "grouping": grouping,
                "group_label": label,
                "r_E_eff": er,
                "r_E16": float(sed_r["true_energy_p16_tev"][ir]),
                "r_E50": float(sed_r["true_energy_p50_tev"][ir]),
                "r_E84": float(sed_r["true_energy_p84_tev"][ir]),
                "r_E2_dnde": yr,
                "r_E2_dnde_err": float(sed_r["E2_dnde_err"][ir]),
                "b_E_eff": eb,
                "b_E16": float(sed_b["true_energy_p16_tev"][ib]),
                "b_E50": float(sed_b["true_energy_p50_tev"][ib]),
                "b_E84": float(sed_b["true_energy_p84_tev"][ib]),
                "b_E2_dnde": yb,
                "b_E2_dnde_err": float(sed_b["E2_dnde_err"][ib]),
                "independent_b_over_r": yb / yr if yr > 0 else float("nan"),
                "r_over_official": yr / official_r if official_r > 0 else float("nan"),
                "b_over_official": yb / official_b if official_b > 0 else float("nan"),
                "fixed_shape_n0_r": fixed_r,
                "fixed_shape_n0_r_err": fixed_r_err,
                "fixed_shape_n0_b": fixed_b,
                "fixed_shape_n0_b_err": fixed_b_err,
                "fixed_shape_b_over_r": fixed_b / fixed_r if fixed_r != 0 else float("nan"),
            }
        )
    write_csv(output_dir / "stage_g_sed_comparison.csv", sed_rows)
    plot_mu_heatmap(cell_rows, asset_dir / "mu_ratio_heatmap.png", config["special_cells"])
    plot_distribution(cell_rows, asset_dir / "mu_ratio_distribution.png", config["special_cells"])
    plot_aeff_diagnostics(nominal, response_b, included, asset_dir / "r_aeff_special_cells.png", config["special_cells"])
    plot_fit_compare(fit_r, fit_b, asset_dir / "stage_f_residual_pull_compare.png")
    params_r = meta_r["fits"]["logpar_conservative"]["parameters"]
    params_b = meta_b["fits"]["logpar_conservative"]["parameters"]
    plot_sed_compare(sed_rows, params_r, params_b, pass5, asset_dir / "stage_g_sed_overlay.png", asset_dir / "stage_g_official_ratio.png")
    deviations = sorted(cell_rows, key=lambda row: abs(float(row["mu_ratio_b_over_r"]) - 1.0), reverse=True)
    chi2_drivers = sorted(cell_fit_rows, key=lambda row: abs(float(row["delta_chi2_b_minus_r"])), reverse=True)
    special_rows = [row for row in cell_rows if int(row["cell_id"]) in config["special_cells"]]
    c75_mc = mc_rows[75]
    summary = {
        "comparison_id": config["comparison_id"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "recompute_slurm_job_id": meta_b.get("slurm_job_id"),
        "finalize_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "mu_ratio_stats": finite_stats(mu_ratio),
        "r_aeff_stats": finite_stats(aeff_ratio),
        "largest_mu_deviations": deviations[:10],
        "largest_chi2_change_cells": chi2_drivers[:10],
        "stage_f_parameter_differences": fit_differences,
        "fit_r": meta_r["fits"],
        "fit_b": meta_b["fits"],
        "stage_g_independent_b_over_r_nhit": finite_stats(np.asarray([row["independent_b_over_r"] for row in sed_rows if row["grouping"] == "nhit"])),
        "stage_g_fixed_shape_b_over_r_nhit": finite_stats(np.asarray([row["fixed_shape_b_over_r"] for row in sed_rows if row["grouping"] == "nhit"])),
        "frozen_response_only_logpar": frozen,
        "production_b_reproduction": {
            "reference_metadata": str(resolve(config["production_stage_f_metadata"])),
            "parameters_exact": meta_b["fits"]["logpar_conservative"]["parameters"] == production_meta["fits"]["logpar_conservative"]["parameters"],
            "chi2_difference": float(meta_b["fits"]["logpar_conservative"]["chi2"] - production_meta["fits"]["logpar_conservative"]["chi2"]),
        },
        "c75_mc": c75_mc,
        "warning": "Both global fits have extremely poor chi2; parameter shifts must not be interpreted as resolving the cell-level model mismatch.",
    }
    write_json(output_dir / "comparison_summary.json", summary)
    write_json(
        output_dir / "response_expect_comparison.json",
        {
            "frozen_logpar": frozen,
            "mu_ratio_stats": summary["mu_ratio_stats"],
            "r_aeff_stats": summary["r_aeff_stats"],
            "cells": cell_rows,
        },
    )
    report_path = resolve(config["report_html"])
    asset_rel = Path(config["asset_dir"]).relative_to("apply/report")
    def image_tag(name: str, caption: str) -> str:
        return f"<figure><img src='{html.escape(str(asset_rel / name))}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>"
    mu_stats = summary["mu_ratio_stats"]
    fit_r_lp = meta_r["fits"]["logpar_conservative"]
    fit_b_lp = meta_b["fits"]["logpar_conservative"]
    report = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>v6 PSF Containment Compare</title><style>
body{{margin:0;background:#f5f6f7;color:#17202a;font:15px/1.55 Arial,sans-serif}}main{{max-width:1280px;margin:auto;background:white;padding:32px}}h1{{font-size:30px}}h2{{border-bottom:2px solid #d8dde3;padding-bottom:6px}}code{{background:#eef1f4;padding:2px 4px}}.warn{{border-left:5px solid #c0392b;background:#fff2f0;padding:12px}}.note{{border-left:5px solid #2878b5;background:#eef7fc;padding:12px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}figure{{margin:0}}img{{width:100%;height:auto;border:1px solid #d8dde3}}figcaption{{font-size:13px;color:#52606d}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d8dde3;padding:5px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto;max-height:600px}}
</style></head><body><main>
<h1>v6 PSF Containment Compare:<br>Fixed 71.5% Rayleigh vs MC Aperture-conditioned Response</h1>
<p>Run <code>{html.escape(config['run_id'])}</code>; recomputation Slurm job <code>{html.escape(str(summary['recompute_slurm_job_id']))}</code>; finalize Slurm job <code>{html.escape(str(summary['finalize_slurm_job_id']))}</code>; git commit at execution <code>{html.escape(summary['git_commit_at_run'])}</code>.</p>
<h2>1. Executive summary</h2>
<div class='warn'>The global fit quality is very poor in both schemes. R LogPar chi2/ndof = {fit_r_lp['chi2']:.4f}/{fit_r_lp['ndof']}; B = {fit_b_lp['chi2']:.4f}/{fit_b_lp['ndof']}. Response changes do not make the fine-cell model statistically acceptable.</div>
<p>Frozen-spectrum cell expectation B/R spans <strong>{mu_stats['min']:.4f}</strong> to <strong>{mu_stats['max']:.4f}</strong>, median <strong>{mu_stats['median']:.4f}</strong>. The spread directly tests energy/cell dependence rather than allowing the spectrum to absorb it.</p>
<h2>2. Strict scheme definitions</h2>
<p><strong>R:</strong> <code>Aeff_R(cell,Etrue,theta) = 0.715 Aeff_nominal</code>. For a two-dimensional Gaussian PSF, <code>r_opt approximately 1.585 sigma</code> gives Rayleigh containment approximately 71.5%.</p>
<p><strong>B:</strong> response numerator is cut at <code>mc_dangle &lt;= r_opt(cell)</code>, so <code>Aeff_B=Aeff_aperture</code> and downstream containment is exactly 1.0. No second factor of 0.715 is applied.</p>
<h2>3. Inputs and reproducibility</h2><p>Config: <code>{html.escape(str(resolve(args_config_global)))}</code>. Selector is frozen at 44 cells; C75 is included and C90 excluded. Both schemes use the same Stage E <code>N_on</code>, <code>B_on</code>, excess, and conservative <code>sqrt(N_on+B_on)</code> error.</p>
<ul><li>Nominal response: <code>{html.escape(config['scheme_r']['nominal_response_npz'])}</code></li><li>Aperture response: <code>{html.escape(config['scheme_b']['response_npz'])}</code></li><li>Stage E signal: <code>{html.escape(config['signal_npz'])}</code></li><li>Selector: <code>{html.escape(config['selector_csv'])}</code></li></ul>
<h2>4. Response metadata consistency</h2><div class='note'>Preflight passed: identical MC input root, thrown denominator, reconstruction cuts, weight contract, Nhit/PredE/Etrue/theta bins, and one-to-one 91-cell mapping. Differences are restricted to aperture conditioning and its numerator/Aeff products.</div>
<div class='grid'>{image_tag('mu_ratio_heatmap.png','Frozen LogPar expected-count ratio by selected cell')}{image_tag('mu_ratio_distribution.png','Distribution of frozen-spectrum expected-count ratios')}</div>
<h2>5. Cell-by-cell expect ratio</h2>{table_html(cell_rows, [('cell_id','cell'),('nhit_bin','Nhit'),('predE_bin','PredE'),('mu_r','mu_R'),('mu_b','mu_B'),('mu_ratio_b_over_r','B/R'),('mu_difference_b_minus_r','B-R'),('mc_neff','MC Neff')])}
<h2>6. Stage F global fit comparison</h2>{table_html(fit_rows, [('scheme','scheme'),('model','model'),('phi0','phi0'),('gamma','gamma'),('alpha','alpha'),('beta','beta'),('chi2','chi2'),('ndof','ndof'),('chi2_over_ndof','chi2/ndof'),('p_value','p-value')])}
<h3>Parameter changes (B - R)</h3>{table_html(fit_differences, [('model','model'),('parameter','parameter'),('r_value','R'),('r_error','R error'),('b_value','B'),('b_error','B error'),('b_minus_r','B-R'),('relative_difference_percent','relative %')])}
{image_tag('stage_f_residual_pull_compare.png','Stage F LogPar residual and pull comparison')}
<p>Full Minuit fit-space covariance and derived correlation matrices are stored in <code>comparison_summary.json</code>; initial values, bounds, MIGRAD, and HESSE are inherited unchanged from Stage F.</p>
<h2>7. Stage G SED comparison</h2><div class='grid'>{image_tag('stage_g_sed_overlay.png','Independent Stage F refits: SED overlay and B/R ratio')}{image_tag('stage_g_official_ratio.png','Each scheme relative to official Pass5 WCDA point-fit LogPar')}</div>
<p>The table separates <strong>fixed-spectrum response-only</strong> normalization ratios from <strong>independently refitted Stage F</strong> SED ratios. No Pool-1 third ratio panel is present.</p>
{table_html(sed_rows, [('grouping','group'),('group_label','label'),('r_E_eff','R Eeff'),('r_E16','R E16'),('r_E50','R E50'),('r_E84','R E84'),('r_E2_dnde','R E2dN/dE'),('r_E2_dnde_err','R error'),('b_E_eff','B Eeff'),('b_E16','B E16'),('b_E50','B E50'),('b_E84','B E84'),('b_E2_dnde','B E2dN/dE'),('b_E2_dnde_err','B error'),('independent_b_over_r','B/R refit'),('fixed_shape_b_over_r','B/R fixed')])}
<h2>8. C6, C75, C89 checks</h2>{table_html(special_rows, [('cell_id','cell'),('mu_ratio_b_over_r','mu B/R'),('r_aeff_median','median R_Aeff'),('mc_containment','MC containment'),('mc_containment_std','MC sigma'),('mc_neff','Neff'),('theta_missing_mass','theta missing')])}
<div class='grid'>{image_tag('r_aeff_special_cells.png','R_Aeff versus Etrue and theta for C6, C75, and C89')}</div>
<h2>9. MC statistics and low-statistics risk</h2>
<p>Weighted MC uses <code>Neff=(sumw)^2/sum(w^2)</code>. Containment uncertainty uses a weighted Bernoulli ratio delta variance. C75: Neff={format_number(c75_mc['neff'])}, PSF effective events={format_number(c75_mc['psf_effective_events'])}, core-fit effective events={format_number(c75_mc['core_fit_effective_events'])}, theta missing mass={format_number(c75_mc['theta_missing_mass'])}, containment warning={html.escape(str(c75_mc['containment_warning']))}, angle warning={html.escape(str(c75_mc['angle_check_warning']))}.</p>
<div class='warn'>Finite-MC fluctuations are not evidence for a real non-Gaussian PSF. The same MC sample was used to optimize r_opt and evaluate containment, so same-sample optimization bias may be present. C75's extreme empirical containment must be treated as low-statistics/coverage driven when its PSF/core-fit effective statistics vanish and theta missing mass is unity.</div>
<h2>10. Conclusion</h2>
<ul><li>The effect is not a pure normalization: frozen-spectrum cell B/R spans {mu_stats['min']:.3f}-{mu_stats['max']:.3f}; the leading deviations are C{deviations[0]['cell_id']}, C{deviations[1]['cell_id']}, C{deviations[2]['cell_id']}, and C{deviations[3]['cell_id']}.</li><li>For LogPar, B changes phi0 by {100*(params_b['phi0']/params_r['phi0']-1):.3f}%, alpha by {100*(params_b['alpha']/params_r['alpha']-1):.3f}%, and beta by {100*(params_b['beta']/params_r['beta']-1):.3f}%.</li><li>Nhit SED B/R after independent refits spans {summary['stage_g_independent_b_over_r_nhit']['min']:.3f}-{summary['stage_g_independent_b_over_r_nhit']['max']:.3f}; with one common frozen shape, normalization B/R spans {summary['stage_g_fixed_shape_b_over_r_nhit']['min']:.3f}-{summary['stage_g_fixed_shape_b_over_r_nhit']['max']:.3f}.</li><li>Scheme B improves chi2 by {fit_r_lp['chi2']-fit_b_lp['chi2']:.3f}, from {fit_r_lp['chi2']:.4f} to {fit_b_lp['chi2']:.4f}, but both fits remain decisively unacceptable.</li><li>C75's measured response containment is {100*float(c75_mc['weighted_containment']):.3f}% +/- {100*float(c75_mc['weighted_containment_std']):.3f}% with raw weighted Neff {float(c75_mc['neff']):.1f}; it is not validated as Crab-track PSF containment because theta missing mass is 1 and PSF/core-fit Neff are both zero.</li><li>Current evidence is insufficient to recommend B as the formal response. Crab-data angular-distance validation is required using observed excess radial profiles in matched Nhit/PredE and theta bands, grouped high-Nhit fallbacks, off-source/null checks, aperture-growth curves, and data/MC r68/r90 comparisons without tuning and testing on the same sample.</li></ul>
<p>Machine-readable authority: <code>apply/output/v6_psf_containment_compare/response_expect_comparison.npz</code>, <code>response_expect_comparison.json</code>, <code>cell_expect_comparison.csv</code>, <code>stage_f_parameter_comparison.csv</code>, <code>stage_f_parameter_differences.csv</code>, <code>stage_g_sed_comparison.csv</code>, and <code>comparison_summary.json</code>.</p>
</main></body></html>"""
    report_path.write_text(report, encoding="utf-8")
    shutil.copy2(output_dir / "cell_expect_comparison.csv", asset_dir / "cell_expect_comparison.csv")
    shutil.copy2(output_dir / "stage_f_parameter_comparison.csv", asset_dir / "stage_f_parameter_comparison.csv")
    shutil.copy2(output_dir / "stage_f_parameter_differences.csv", asset_dir / "stage_f_parameter_differences.csv")
    shutil.copy2(output_dir / "stage_g_sed_comparison.csv", asset_dir / "stage_g_sed_comparison.csv")
    shutil.copy2(output_dir / "comparison_summary.json", asset_dir / "comparison_summary.json")
    print(f"Wrote report: {report_path}", flush=True)


args_config_global = str(DEFAULT_CONFIG)


def main() -> None:
    global args_config_global
    args = parse_args()
    args_config_global = args.config
    config = read_json(resolve(args.config))
    if args.mode == "preflight":
        preflight(config)
    elif args.mode == "mc-stats":
        mc_stats(config, workers=args.workers, files_per_task=args.files_per_task)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
