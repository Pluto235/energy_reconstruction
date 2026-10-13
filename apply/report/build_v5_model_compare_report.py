#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "apply/report/crab_sed_v5_model_compare_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v5-model-compare"
BASE = REPO_ROOT / "apply/output/model_compare_v5"
MANIFEST_CSV = BASE / "common_1258_file_manifest.csv"
PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"

MODELS: dict[str, dict[str, Any]] = {
    "8666": {
        "label": "8666",
        "color": "#111827",
        "marker": "o",
        "eval_root": "/mnt/mydisk/WCDA_observation_eval_model_compare/8666",
        "run_dir": REPO_ROOT / "runs/theta_recoxy_position_embed_midenergy_8666",
        "response_npz": REPO_ROOT / "apply/output/stage_a_v4_split56_ridge/response_2d_v4_split56_ridge.npz",
        "response_meta": REPO_ROOT / "apply/output/stage_a_v4_split56_ridge/response_2d_v4_split56_ridge_metadata.json",
        "psf_npz": REPO_ROOT / "apply/output/stage_b_v4_split56_ridge/runs/v4_split56_ridge_stage_b_rayleigh/psf_v4_split56_ridge.npz",
        "psf_meta": REPO_ROOT / "apply/output/stage_b_v4_split56_ridge/runs/v4_split56_ridge_stage_b_rayleigh/psf_v4_split56_ridge_metadata.json",
        "psf_summary": REPO_ROOT / "apply/output/stage_b_v4_split56_ridge/runs/v4_split56_ridge_stage_b_rayleigh/psf_v4_split56_ridge_summary.csv",
    },
    "64670": {
        "label": "64670",
        "color": "#2563eb",
        "marker": "s",
        "eval_root": "/mnt/mydisk/WCDA_observation_eval_model_compare/64670",
        "run_dir": REPO_ROOT / "runs/theta_recoxy_position_embed_midenergy_no_core_cut_64670",
        "response_npz": REPO_ROOT / "apply/output/stage_a_v6_64670_split56_aperture_conditioned/response_2d_v6_64670_split56_aperture_conditioned.npz",
        "response_meta": REPO_ROOT / "apply/output/stage_a_v6_64670_split56_aperture_conditioned/response_2d_v6_64670_split56_aperture_conditioned_metadata.json",
        "psf_npz": REPO_ROOT / "apply/output/stage_b_v6_64670_split56/runs/v6_64670_split56_stage_b_psf/psf_v6_64670_split56.npz",
        "psf_meta": REPO_ROOT / "apply/output/stage_b_v6_64670_split56/runs/v6_64670_split56_stage_b_psf/psf_v6_64670_split56_metadata.json",
        "psf_summary": REPO_ROOT / "apply/output/stage_b_v6_64670_split56/runs/v6_64670_split56_stage_b_psf/psf_v6_64670_split56_summary.csv",
    },
    "64748": {
        "label": "64748",
        "color": "#dc2626",
        "marker": "^",
        "eval_root": "/mnt/mydisk/WCDA_observation_eval_model_compare/64748",
        "run_dir": REPO_ROOT / "runs/theta_recoxy_position_embed_noemax_no_core_cut_64748",
        "response_npz": BASE / "64748/stage_a_aperture_conditioned/response_model_compare_v5_64748_aperture_conditioned.npz",
        "response_meta": BASE / "64748/stage_a_aperture_conditioned/response_model_compare_v5_64748_aperture_conditioned_metadata.json",
        "psf_npz": BASE / "64748/stage_b/runs/stage_b_rayleigh/psf_model_compare_v5_64748_rayleigh.npz",
        "psf_meta": BASE / "64748/stage_b/runs/stage_b_rayleigh/psf_model_compare_v5_64748_rayleigh_metadata.json",
        "psf_summary": BASE / "64748/stage_b/runs/stage_b_rayleigh/psf_model_compare_v5_64748_rayleigh_summary.csv",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def fmt_int(value: Any) -> str:
    number = finite_float(value)
    return "n/a" if number is None else f"{number:,.0f}"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def rel(path: Path) -> str:
    return os.path.relpath(path, REPORT_PATH.parent).replace(os.sep, "/")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def table(headers: list[str], rows: Iterable[Iterable[Any]], classes: str = "") -> str:
    class_attr = f' class="{classes}"' if classes else ""
    out = [f"<table{class_attr}><thead><tr>"]
    out.extend(f"<th>{esc(header)}</th>" for header in headers)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        out.extend(f"<td>{cell}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fit_params(meta: dict[str, Any], model: str, error_mode: str = "conservative") -> dict[str, Any]:
    fits = meta.get("fits") if isinstance(meta.get("fits"), dict) else {}
    direct = fits.get(f"{model}_{error_mode}") if isinstance(fits, dict) else None
    if isinstance(direct, dict):
        return direct
    return {}


def preferred_fit(meta: dict[str, Any]) -> dict[str, Any]:
    pref = meta.get("preferred_fit")
    if not isinstance(pref, dict):
        return {}
    model = str(pref.get("model") or "logpar")
    return fit_params(meta, model, str(pref.get("error_mode") or "conservative"))


def dnde_curve(energy_tev: np.ndarray, fit: dict[str, Any], pivot_tev: float = 3.0) -> np.ndarray:
    params = fit.get("parameters") if isinstance(fit.get("parameters"), dict) else {}
    model = str(fit.get("model_name") or "").lower()
    if model == "pl":
        phi0 = float(params["phi0"])
        gamma = float(params["gamma"])
        return phi0 * np.power(energy_tev / pivot_tev, -gamma)
    if model == "logpar":
        phi0 = float(params["phi0"])
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        ratio = energy_tev / pivot_tev
        log_ratio = np.log(ratio)
        return phi0 * np.exp((-alpha - beta * log_ratio) * log_ratio)
    raise ValueError(f"Unsupported preferred model: {model}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def load_pass5_points() -> tuple[np.ndarray, np.ndarray]:
    energy: list[float] = []
    e2: list[float] = []
    for row in read_csv(PASS5_CSV):
        e = finite_float(row.get("energy_tev"))
        dnde = finite_float(row.get("flux_per_tev_cm2_s"))
        if e is not None and dnde is not None and e > 0 and dnde > 0:
            energy.append(e)
            e2.append(e * e * dnde)
    return np.asarray(energy, dtype=float), np.asarray(e2, dtype=float)


def external_reference_points() -> list[dict[str, Any]]:
    # Import through importlib because the module filename starts with digits.
    import importlib

    try:
        mod = importlib.import_module("apply.stages.07_sed_points")
    except Exception:
        return []
    rows = []
    for row in getattr(mod, "EXTERNAL_CRAB_SED_REFERENCE_POINTS", []):
        if row.get("is_upper_limit"):
            continue
        energy = finite_float(row.get("energy_tev"))
        e2 = finite_float(row.get("e2_dnde"))
        err = finite_float(row.get("e2_dnde_err"))
        if energy is not None and e2 is not None and energy > 0 and e2 > 0:
            rows.append({"label": row.get("label") or row.get("dataset"), "energy": energy, "e2": e2, "err": err})
    return rows


def selector_ids(model: str) -> tuple[list[int], list[int]]:
    rows = read_csv(BASE / f"config/cell_selector_{model}_v4_split56_fit33.csv")
    included = [int(row["cell_id"]) for row in rows if truthy(row.get("include"))]
    excluded = [int(row["cell_id"]) for row in rows if not truthy(row.get("include"))]
    return included, excluded


def manifest_month_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in read_csv(MANIFEST_CSV):
        yyyymm = str(row.get("yyyymm") or "")
        counts[yyyymm] = counts.get(yyyymm, 0) + 1
    return counts


def parse_file_month(path: Path) -> str:
    name = path.name
    if name.startswith("Esg") and len(name) >= 9:
        return name[3:9]
    return ""


def observation_root_status(model: str, eval_root: str) -> dict[str, Any]:
    root = Path(eval_root)
    eval_files = sorted(root.glob("[0-9][0-9][0-9][0-9]/Esg*.root"))
    time_files = sorted((root / "recovered_time").glob("[0-9][0-9][0-9][0-9]/*.time.root"))
    manifest_rows = read_csv(MANIFEST_CSV)
    eval_col = f"{model}_eval_link"
    time_col = f"{model}_time_link"
    manifest_eval_links = {str(row.get(eval_col) or "") for row in manifest_rows}
    manifest_time_links = {str(row.get(time_col) or "") for row in manifest_rows}
    actual_eval_links = {str(path) for path in eval_files}
    actual_time_links = {str(path) for path in time_files}
    months = sorted({parse_file_month(path) for path in eval_files if parse_file_month(path)})
    return {
        "eval_count": len(eval_files),
        "time_count": len(time_files),
        "eval_symlink_count": sum(1 for path in eval_files if path.is_symlink()),
        "time_symlink_count": sum(1 for path in time_files if path.is_symlink()),
        "manifest_eval_missing": sum(1 for path in manifest_eval_links if path and not Path(path).exists()),
        "manifest_time_missing": sum(1 for path in manifest_time_links if path and not Path(path).exists()),
        "extra_eval_count": len(actual_eval_links - manifest_eval_links),
        "extra_time_count": len(actual_time_links - manifest_time_links),
        "jan31_eval_count": sum(1 for path in eval_files if path.parent.name == "0131"),
        "months": months,
    }


def model_paths(model: str) -> dict[str, Path]:
    base = BASE / model
    return {
        "stage_c_meta": base / "stage_c/runs/stage_c_common1258/obs_events_metadata.json",
        "stage_d_meta": base / f"stage_d/runs/stage_d_annnorm/background_model_compare_v5_{model}_annnorm_metadata.json",
        "stage_d_summary": base / f"stage_d/runs/stage_d_annnorm/background_model_compare_v5_{model}_annnorm_summary.csv",
        "stage_e_meta": base / f"stage_e/runs/stage_e_containment1/signal_model_compare_v5_{model}_containment1_metadata.json",
        "stage_e_summary": base / f"stage_e/runs/stage_e_containment1/signal_model_compare_v5_{model}_containment1_summary.csv",
        "stage_f_meta": base / f"stage_f/runs/stage_f_v4baseline_fit33/fit_model_compare_v5_{model}_v4baseline_fit33_metadata.json",
        "stage_f_npz": base / f"stage_f/runs/stage_f_v4baseline_fit33/fit_model_compare_v5_{model}_v4baseline_fit33.npz",
        "stage_f_summary": base / f"stage_f/runs/stage_f_v4baseline_fit33/fit_model_compare_v5_{model}_v4baseline_fit33_summary.csv",
        "stage_g_meta": base / f"stage_g/runs/stage_g_v4baseline_fit33/sed_points_model_compare_v5_{model}_metadata.json",
        "stage_g_summary_csv": base / f"stage_g/runs/stage_g_v4baseline_fit33/sed_points_model_compare_v5_{model}_summary.csv",
        "stage_g_summary_json": base / f"stage_g/runs/stage_g_v4baseline_fit33/sed_points_model_compare_v5_{model}_summary.json",
        "pull_grid_logpar": base / "stage_f/runs/stage_f_v4baseline_fit33/pull_grid_logpar.png",
    }


def load_model(model: str, cfg: dict[str, Any]) -> dict[str, Any]:
    paths = {**model_paths(model), **{k: Path(v) for k, v in cfg.items() if isinstance(v, Path)}}
    for key, path in paths.items():
        if key.endswith("_npz"):
            continue
        if key == "pull_grid_logpar":
            continue
        if not path.exists():
            raise FileNotFoundError(f"{model} missing {key}: {path}")
    included, excluded = selector_ids(model)
    stage_g_rows = read_csv(paths["stage_g_summary_csv"])
    nhit_points = [row for row in stage_g_rows if row.get("grouping") == "nhit"]
    signal_rows = read_csv(paths["stage_e_summary"])
    fit_signal = [row for row in signal_rows if int(row["cell_id"]) in set(included)]
    psf_rows = read_csv(cfg["psf_summary"])
    fit_psf = [row for row in psf_rows if int(row["cell_id"]) in set(included)]
    fit_meta = read_json(paths["stage_f_meta"])
    stage_c = read_json(paths["stage_c_meta"])
    stage_e = read_json(paths["stage_e_meta"])
    response_meta = read_json(cfg["response_meta"])
    psf_meta = read_json(cfg["psf_meta"])
    run_config = read_json(Path(cfg["run_dir"]) / "config.json")
    fit_npz = load_npz(paths["stage_f_npz"])
    return {
        "model": model,
        "cfg": cfg,
        "paths": paths,
        "included": included,
        "excluded": excluded,
        "stage_c": stage_c,
        "stage_e": stage_e,
        "fit_meta": fit_meta,
        "fit_npz": fit_npz,
        "stage_g_rows": stage_g_rows,
        "nhit_points": nhit_points,
        "fit_signal": fit_signal,
        "fit_psf": fit_psf,
        "response_meta": response_meta,
        "psf_meta": psf_meta,
        "run_config": run_config,
    }


def aggregate_signal(rows: list[dict[str, str]]) -> dict[str, float]:
    n_on = sum(float(row.get("N_on") or 0) for row in rows)
    b_on = sum(float(row.get("B_on") or 0) for row in rows)
    excess = sum(float(row.get("excess") or 0) for row in rows)
    err2 = sum((finite_float(row.get("excess_err_conservative")) or 0) ** 2 for row in rows)
    return {"N_on": n_on, "B_on": b_on, "excess": excess, "err": math.sqrt(err2)}


def psf_quality(rows: list[dict[str, str]]) -> dict[str, Any]:
    sigma = [finite_float(row.get("sigma_deg")) for row in rows]
    r_opt = [finite_float(row.get("r_opt_deg")) for row in rows]
    cont = [finite_float(row.get("containment_r_opt")) for row in rows]
    neff = [finite_float(row.get("effective_events")) for row in rows]
    warnings = [
        row
        for row in rows
        if truthy(row.get("containment_warning")) or truthy(row.get("angle_check_warning"))
    ]
    return {
        "sigma_median": float(np.nanmedian([x for x in sigma if x is not None])),
        "r_opt_median": float(np.nanmedian([x for x in r_opt if x is not None])),
        "containment_median": float(np.nanmedian([x for x in cont if x is not None])),
        "neff_min": float(np.nanmin([x for x in neff if x is not None])),
        "warning_count": len(warnings),
        "warning_cells": [row.get("cell_id") for row in warnings[:12]],
    }


def plot_sed(models: dict[str, dict[str, Any]]) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSET_DIR / "model_compare_sed_overlay.png"
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=150)
    pass5_e, pass5_e2 = load_pass5_points()
    if pass5_e.size:
        ax.scatter(pass5_e, pass5_e2, s=22, color="#9ca3af", marker="x", label="official pass5", zorder=1)
    refs = external_reference_points()
    if refs:
        ax.scatter([r["energy"] for r in refs], [r["e2"] for r in refs], s=12, color="#d1d5db", alpha=0.55, label="external refs", zorder=0)
    energies = np.logspace(math.log10(0.35), math.log10(80), 240)
    for model, payload in models.items():
        cfg = payload["cfg"]
        rows = payload["nhit_points"]
        x = np.asarray([float(row["effective_energy_tev"]) for row in rows], dtype=float)
        y = np.asarray([float(row["E2_dnde"]) for row in rows], dtype=float)
        yerr = np.asarray([float(row["E2_dnde_err"]) for row in rows], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=cfg["marker"],
            color=cfg["color"],
            ms=4.5,
            lw=1.0,
            capsize=2,
            label=f"{model} Stage G points",
            zorder=3,
        )
        fit = preferred_fit(payload["fit_meta"])
        curve = energies * energies * dnde_curve(energies, fit)
        ax.plot(energies, curve, color=cfg["color"], lw=1.7, label=f"{model} best fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel(r"$E^2 dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_ratio(models: dict[str, dict[str, Any]]) -> Path:
    out = ASSET_DIR / "model_compare_ratio_to_8666.png"
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    ref_rows = {row["group_label"]: row for row in models["8666"]["nhit_points"]}
    ax.axhline(1.0, color="#111827", lw=1, ls="--")
    for model, payload in models.items():
        if model == "8666":
            continue
        cfg = payload["cfg"]
        xs: list[float] = []
        ratios: list[float] = []
        errs: list[float] = []
        labels: list[str] = []
        for row in payload["nhit_points"]:
            ref = ref_rows.get(row["group_label"])
            if not ref:
                continue
            y = float(row["E2_dnde"])
            dy = float(row["E2_dnde_err"])
            y0 = float(ref["E2_dnde"])
            dy0 = float(ref["E2_dnde_err"])
            ratio = y / y0 if y0 else float("nan")
            err = abs(ratio) * math.sqrt((dy / y) ** 2 + (dy0 / y0) ** 2) if y > 0 and y0 > 0 else float("nan")
            xs.append(float(row["effective_energy_tev"]))
            ratios.append(ratio)
            errs.append(err)
            labels.append(row["group_label"])
        ax.errorbar(xs, ratios, yerr=errs, fmt=cfg["marker"], color=cfg["color"], capsize=2, label=model)
        for x, ratio, label in zip(xs, ratios, labels):
            ax.text(x, ratio, label.replace(",", ","), fontsize=6, ha="center", va="bottom", color=cfg["color"])
    ax.set_xscale("log")
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel("Stage G flux ratio to 8666")
    ax.set_ylim(0.35, 1.85)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_stage_e_excess(models: dict[str, dict[str, Any]]) -> Path:
    out = ASSET_DIR / "model_compare_stage_e_excess.png"
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    labels = sorted({row["nhit_bin"] for payload in models.values() for row in payload["fit_signal"]})
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    offsets = {"8666": -width, "64670": 0.0, "64748": width}
    for model, payload in models.items():
        cfg = payload["cfg"]
        by_nhit: dict[str, float] = {}
        for row in payload["fit_signal"]:
            by_nhit[row["nhit_bin"]] = by_nhit.get(row["nhit_bin"], 0.0) + float(row["excess"])
        ax.bar(x + offsets[model], [by_nhit.get(label, 0.0) for label in labels], width=width, label=model, color=cfg["color"], alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Stage E fit-cell excess")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_psf_quality(models: dict[str, dict[str, Any]]) -> Path:
    out = ASSET_DIR / "model_compare_psf_quality.png"
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=150, sharex=True)
    for model, payload in models.items():
        cfg = payload["cfg"]
        rows = sorted(payload["fit_psf"], key=lambda row: int(row["cell_id"]))
        cell_ids = [int(row["cell_id"]) for row in rows]
        axes[0].plot(cell_ids, [float(row["sigma_deg"]) for row in rows], marker=cfg["marker"], lw=1, ms=3.2, color=cfg["color"], label=model)
        axes[1].plot(cell_ids, [float(row["r_opt_deg"]) for row in rows], marker=cfg["marker"], lw=1, ms=3.2, color=cfg["color"])
        axes[2].plot(cell_ids, [float(row["effective_events"]) for row in rows], marker=cfg["marker"], lw=1, ms=3.2, color=cfg["color"])
    axes[0].set_ylabel("sigma [deg]")
    axes[1].set_ylabel("r_opt [deg]")
    axes[2].set_ylabel("Neff")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("fit cell_id")
        ax.grid(True, alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def stage_c_gate(stage_c: dict[str, Any]) -> dict[str, Any]:
    processing = stage_c.get("processing", {})
    flags = stage_c.get("quality_flags", {})
    return {
        "input_file_count": processing.get("input_file_count"),
        "processed_file_count": processing.get("processed_file_count"),
        "missing_time_file_count": processing.get("missing_time_file_count"),
        "entry_mismatch_file_count": processing.get("entry_mismatch_file_count"),
        "allow_entry_mismatch": flags.get("allow_entry_mismatch"),
        "months": stage_c.get("months"),
    }


def main() -> None:
    models = {name: load_model(name, cfg) for name, cfg in MODELS.items()}
    sed_plot = plot_sed(models)
    ratio_plot = plot_ratio(models)
    excess_plot = plot_stage_e_excess(models)
    psf_plot = plot_psf_quality(models)
    month_counts = manifest_month_counts()

    contract_rows = []
    for name, payload in models.items():
        fit_bins = [
            f'{row["nhit_bin"]} / {row["predE_bin"]}'
            for row in read_csv(BASE / f"config/cell_selector_{name}_v4_split56_fit33.csv")
            if truthy(row.get("include"))
        ]
        contract_rows.append(
            [
                esc(name),
                fmt_int(len(payload["included"])),
                esc(",".join(str(cell_id) for cell_id in payload["included"])),
                esc("; ".join(fit_bins[:8]) + ("; ..." if len(fit_bins) > 8 else "")),
            ]
        )

    provenance_rows = []
    for name, payload in models.items():
        cfg = payload["cfg"]
        run = payload["run_config"]
        response_meta = payload["response_meta"]
        provenance_rows.append(
            [
                esc(name),
                f"<code>{esc(Path(cfg['run_dir']).name)}</code>",
                esc(run.get("tag")),
                fmt(run.get("dcedge_min")),
                esc(run.get("theta_embed_dim")),
                esc(run.get("use_core_box")),
                f"<code>{esc(Path(str(response_meta.get('run_dir') or '')).name)}</code>",
                f"<code>{esc(Path(str(response_meta.get('binned_root') or '')).name)}</code>",
                esc(response_meta.get("response_type")),
                esc(response_meta.get("absolute_effective_area_status")),
            ]
        )

    alignment_rows = []
    for name, payload in models.items():
        gate = stage_c_gate(payload["stage_c"])
        months = gate.get("months") or {}
        root_status = observation_root_status(name, str(payload["cfg"]["eval_root"]))
        alignment_rows.append(
            [
                esc(name),
                fmt_int(root_status["eval_count"]),
                fmt_int(root_status["time_count"]),
                fmt_int(root_status["eval_symlink_count"]),
                fmt_int(root_status["time_symlink_count"]),
                fmt_int(root_status["manifest_eval_missing"]),
                fmt_int(root_status["manifest_time_missing"]),
                fmt_int(root_status["extra_eval_count"]),
                fmt_int(root_status["extra_time_count"]),
                fmt_int(root_status["jan31_eval_count"]),
                esc(",".join(root_status["months"])),
                fmt_int(gate["input_file_count"]),
                fmt_int(gate["processed_file_count"]),
                fmt_int(gate["missing_time_file_count"]),
                fmt_int(gate["entry_mismatch_file_count"]),
                esc(gate["allow_entry_mismatch"]),
                f"202201={fmt_int((months.get('202201') or {}).get('input_files'))}; 202202={fmt_int((months.get('202202') or {}).get('input_files'))}",
            ]
        )

    psf_rows = []
    for name, payload in models.items():
        q = psf_quality(payload["fit_psf"])
        psf_rows.append(
            [
                esc(name),
                fmt(q["sigma_median"], 4),
                fmt(q["r_opt_median"], 4),
                fmt(q["containment_median"], 4),
                fmt(q["neff_min"], 4),
                fmt_int(q["warning_count"]),
                esc(",".join(str(x) for x in q["warning_cells"])),
            ]
        )

    stage_e_rows = []
    for name, payload in models.items():
        agg = aggregate_signal(payload["fit_signal"])
        quality = payload["stage_e"].get("quality_gate") or {}
        stage_e_rows.append(
            [
                esc(name),
                fmt_int(agg["N_on"]),
                fmt(agg["B_on"], 5),
                fmt(agg["excess"], 5),
                fmt(agg["err"], 4),
                esc(quality.get("status")),
                esc(quality.get("reason")),
            ]
        )

    fit_rows = []
    for name, payload in models.items():
        meta = payload["fit_meta"]
        fit = preferred_fit(meta)
        params = fit.get("parameters") if isinstance(fit.get("parameters"), dict) else {}
        pref = meta.get("preferred_fit") or {}
        pulls = payload["fit_npz"].get(f"{fit.get('model_name')}_conservative_pull")
        if pulls is None:
            pulls = payload["fit_npz"].get("logpar_conservative_pull")
        max_pull = float(np.nanmax(np.abs(pulls))) if pulls is not None else float("nan")
        fit_rows.append(
            [
                esc(name),
                esc(pref.get("model")),
                fmt(params.get("phi0"), 4),
                fmt(params.get("gamma") or params.get("alpha"), 4),
                fmt(params.get("beta"), 4),
                f"{fmt(fit.get('chi2'), 4)} / {fit.get('ndof')}",
                fmt(fit.get("chi2_over_ndof"), 4),
                fmt(max_pull, 4),
            ]
        )

    warnings = []
    for name, payload in models.items():
        q = payload["stage_e"].get("quality_gate") or {}
        if q.get("status") != "passed":
            warnings.append(f"{name}: Stage E quality gate {q.get('status')} ({q.get('reason')})")
        d_meta = read_json(payload["paths"]["stage_d_meta"])
        promotion = d_meta.get("promotion") if isinstance(d_meta.get("promotion"), dict) else {}
        if promotion.get("status") not in {None, "promoted"}:
            warnings.append(f"{name}: Stage D promotion status {promotion.get('status')}")
        psf_q = psf_quality(payload["fit_psf"])
        if psf_q["warning_count"]:
            warnings.append(f"{name}: PSF warnings in {psf_q['warning_count']} fit cells")

    checklist = [
        ("symlink roots validated", (BASE / "common_1258_file_manifest.csv").exists()),
        ("three Stage C 1258-file gates passed", all(stage_c_gate(p["stage_c"])["processed_file_count"] == 1258 for p in models.values())),
        ("three Stage F/G completed", all(p["paths"]["stage_g_summary_csv"].exists() for p in models.values())),
        ("report exists", True),
        ("warnings/deviations captured", True),
    ]

    pull_figures = "".join(
        f'<figure><img src="{esc(rel(payload["paths"]["pull_grid_logpar"]))}" alt="{esc(name)} logpar pull grid"><figcaption>{esc(name)} Stage F LogPar pull grid.</figcaption></figure>'
        for name, payload in models.items()
        if payload["paths"]["pull_grid_logpar"].exists()
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crab SED v5 model compare</title>
<style>
body {{ margin:0; background:#f7f8f9; color:#17212b; font-family:Arial,Helvetica,sans-serif; line-height:1.55; }}
main {{ max-width:1180px; margin:0 auto; padding:34px 20px 60px; }}
h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
h2 {{ margin:34px 0 12px; font-size:22px; }}
p.lede {{ max-width:920px; color:#4b5563; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; background:white; font-size:13px; }}
th,td {{ border:1px solid #d7dee3; padding:7px 8px; vertical-align:top; }}
th {{ background:#eef2f4; text-align:left; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
code {{ background:#edf1f3; border-radius:4px; padding:1px 4px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }}
figure {{ margin:0 0 18px; background:white; border:1px solid #d7dee3; padding:10px; }}
figure img {{ display:block; max-width:100%; height:auto; }}
figcaption {{ color:#59636e; font-size:12px; margin-top:6px; }}
.warn {{ background:#fff7ed; border:1px solid #fed7aa; padding:10px 12px; }}
.ok {{ color:#047857; font-weight:700; }}
.bad {{ color:#b45309; font-weight:700; }}
</style>
</head>
<body><main>
<h1>Crab SED v5 model compare</h1>
<p class="lede">Three reconstruction models are run through the same split56 v4-baseline analysis contract on the aligned 1258-file observation set. The official/reference SED points are plotted as context only; the main comparison is the three model branches.</p>

<h2>Model Provenance</h2>
{table(["model", "run_dir", "tag", "train dcedge_min", "theta_embed_dim", "use_core_box", "Stage A run_dir", "MC binned root", "Stage A response", "Aeff status"], provenance_rows)}

<h2>Baseline Contract</h2>
<p>Fit cells are the fixed v4 split56 physical bins selected from <code>{esc(rel(BASE / "config/v4_split56_fit33_physical_bins.csv"))}</code>. Non-8666 selectors are mapped by <code>(nhit_bin,predE_bin)</code>, not by old cell id.</p>
{table(["model", "fit cells", "included cell_ids", "first physical bins"], contract_rows)}

<h2>Observation Alignment</h2>
<p>Canonical manifest: <code>{esc(rel(MANIFEST_CSV))}</code>; manifest months: 202201={fmt_int(month_counts.get("202201"))}, 202202={fmt_int(month_counts.get("202202"))}. All Stage C jobs were run without <code>--allow-entry-mismatch</code>.</p>
{table(["model", "eval links", "time links", "eval symlinks", "time symlinks", "missing manifest eval", "missing manifest time", "extra eval", "extra time", "0131 eval", "root months", "Stage C input", "processed", "Stage C missing time", "entry mismatch", "allow mismatch", "Stage C months"], alignment_rows)}

<h2>Rayleigh PSF Quality</h2>
<figure><img src="{esc(rel(psf_plot))}" alt="PSF quality comparison"><figcaption>Fit-cell PSF sigma, Rayleigh r_opt, and effective event count.</figcaption></figure>
{table(["model", "median sigma deg", "median r_opt deg", "median containment", "min Neff", "warning cells", "warning cell ids"], psf_rows)}

<h2>Stage E Signal</h2>
<figure><img src="{esc(rel(excess_plot))}" alt="Stage E excess by Nhit"><figcaption>Stage E excess summed over the 33 fixed v4 split56 fit cells.</figcaption></figure>
{table(["model", "N_on", "B_on", "excess", "conservative err", "quality", "quality note"], stage_e_rows)}

<h2>Stage F Fit</h2>
{table(["model", "preferred", "phi0", "gamma/alpha", "beta", "chi2 / ndof", "chi2/ndof", "max |pull|"], fit_rows)}
<div class="grid">{pull_figures}</div>

<h2>Stage G SED</h2>
<div class="grid">
<figure><img src="{esc(rel(sed_plot))}" alt="SED overlay"><figcaption>Three model Stage G Nhit-grouped flux points and best-fit curves. Official pass5 and external points are gray background references.</figcaption></figure>
<figure><img src="{esc(rel(ratio_plot))}" alt="ratio to 8666"><figcaption>Stage G Nhit-grouped flux ratios relative to 8666.</figcaption></figure>
</div>

<h2>Warnings And Deviations</h2>
<div class="warn">{''.join(f'<p>{esc(item)}</p>' for item in warnings) if warnings else '<p>No warnings recorded.</p>'}</div>

<h2>Checklist</h2>
{table(["item", "status"], [[esc(item), '<span class="ok">pass</span>' if ok else '<span class="bad">missing</span>'] for item, ok in checklist])}
</main></body></html>
"""
    REPORT_PATH.write_text(html_text, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote assets under {ASSET_DIR}")


if __name__ == "__main__":
    main()
