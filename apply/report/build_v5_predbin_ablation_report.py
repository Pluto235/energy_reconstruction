#!/usr/bin/env python
from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_HTML = REPO_ROOT / "apply/report/crab_sed_v5_predbin_ablation_report.html"
ASSET_DIR = REPO_ROOT / "apply/report/assets/v5-predbin-ablation"
OFFICIAL_PASS5_CSV = REPO_ROOT / "apply/report/assets/official-pass5/wcda_crab_sed_pass5_20260616_104941.csv"


STRATEGIES = {
    "baseline_v4": {
        "label": "baseline_v4",
        "selector": REPO_ROOT / "apply/config/cell_selector_v4_drop4_psfborrow.csv",
        "stage_b_summary": REPO_ROOT / "apply/output/stage_b_v3_candidate_psfborrow/runs/v3_psfborrow_from_nominal/psf_v3_candidate_summary.csv",
        "stage_f_npz": REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4.npz",
        "stage_f_meta": REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4_metadata.json",
        "stage_f_summary": REPO_ROOT / "apply/output/stage_f_v4_aperture_conditioned/runs/v4_stage_f_aperture_conditioned_drop4/fit_v4_aperture_conditioned_drop4_summary.csv",
        "stage_g_summary": REPO_ROOT / "apply/output/stage_g_v4_aperture_conditioned/runs/v4_stage_g_aperture_conditioned_drop4/sed_points_v4_aperture_conditioned_drop4_summary.csv",
    },
    "gap025": {
        "label": "gap025",
        "selector": REPO_ROOT / "apply/config/cell_selector_v5_predbin_gap025.csv",
        "stage_b_summary": REPO_ROOT / "apply/output/stage_b_v5_predbin_gap025/current/psf_v5_predbin_gap025_summary.csv",
        "stage_f_npz": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap025_aperture_conditioned/current/fit_v5_predbin_gap025_aperture_conditioned.npz",
        "stage_f_meta": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap025_aperture_conditioned/current/fit_v5_predbin_gap025_aperture_conditioned_metadata.json",
        "stage_f_summary": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap025_aperture_conditioned/current/fit_v5_predbin_gap025_aperture_conditioned_summary.csv",
        "stage_g_summary": REPO_ROOT / "apply/output/stage_g_v5_predbin_gap025_aperture_conditioned/current/sed_points_v5_predbin_gap025_aperture_conditioned_summary.csv",
    },
    "gap1": {
        "label": "gap1",
        "selector": REPO_ROOT / "apply/config/cell_selector_v5_predbin_gap1.csv",
        "stage_b_summary": REPO_ROOT / "apply/output/stage_b_v5_predbin_gap1/current/psf_v5_predbin_gap1_summary.csv",
        "stage_f_npz": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap1_aperture_conditioned/current/fit_v5_predbin_gap1_aperture_conditioned.npz",
        "stage_f_meta": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap1_aperture_conditioned/current/fit_v5_predbin_gap1_aperture_conditioned_metadata.json",
        "stage_f_summary": REPO_ROOT / "apply/output/stage_f_v5_predbin_gap1_aperture_conditioned/current/fit_v5_predbin_gap1_aperture_conditioned_summary.csv",
        "stage_g_summary": REPO_ROOT / "apply/output/stage_g_v5_predbin_gap1_aperture_conditioned/current/sed_points_v5_predbin_gap1_aperture_conditioned_summary.csv",
    },
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: Path) -> Dict[str, object]:
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


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def html_escape(value: object) -> str:
    return html.escape(str(value))


def relative_path(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start=REPORT_HTML.parent.resolve())
    except OSError:
        return str(path)


def latest_run_artifact(path: Path) -> Path:
    if path.exists():
        return path
    if path.parent.name != "current":
        return path
    runs_dir = path.parent.parent / "runs"
    if not runs_dir.exists():
        return path
    candidates = [candidate for candidate in runs_dir.glob(f"*/{path.name}") if candidate.exists()]
    if not candidates:
        return path
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def resolve_strategy_paths(config: Dict[str, Path]) -> Dict[str, Path]:
    resolved = dict(config)
    for key in ["stage_b_summary", "stage_f_npz", "stage_f_meta", "stage_f_summary", "stage_g_summary"]:
        resolved[key] = latest_run_artifact(config[key])
    return resolved


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_fit_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def fit_params_from_meta(meta: Dict[str, object], model: str) -> Dict[str, object]:
    fits = meta.get("fits")
    if not isinstance(fits, dict):
        return {}
    for value in fits.values():
        if not isinstance(value, dict):
            continue
        if str(value.get("model_name") or value.get("model") or "").lower() != model.lower():
            continue
        params = value.get("parameters") if isinstance(value.get("parameters"), dict) else {}
        errors = value.get("errors") if isinstance(value.get("errors"), dict) else {}
        return {
            "valid": value.get("valid"),
            "chi2": value.get("chi2"),
            "ndof": value.get("ndof"),
            "p_value": value.get("p_value"),
            "parameters": params,
            "errors": errors,
        }
    direct = fits.get(model) or fits.get(model.lower()) or fits.get(model.upper())
    return direct if isinstance(direct, dict) else {}


def max_abs_pull(fit_npz: Dict[str, np.ndarray], preferred: str = "logpar") -> Optional[float]:
    candidates = [
        f"{preferred}_conservative_pull",
        f"{preferred}_sqrt_n_pull",
        "logpar_conservative_pull",
        "logpar_pull",
    ]
    for key in candidates:
        if key in fit_npz:
            values = np.asarray(fit_npz[key], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                return float(np.nanmax(np.abs(finite)))
    return None


def load_strategy(name: str, config: Dict[str, Path]) -> Dict[str, object]:
    config = resolve_strategy_paths(config)
    selector_rows = read_csv_rows(config["selector"])
    included = [row for row in selector_rows if truthy(row.get("include"))]
    psf_rows = read_csv_rows(config["stage_b_summary"])
    fit_meta = read_json(config["stage_f_meta"])
    fit_npz = load_fit_npz(config["stage_f_npz"])
    stage_g_rows = read_csv_rows(config["stage_g_summary"])
    logpar = fit_params_from_meta(fit_meta, "logpar")
    pl = fit_params_from_meta(fit_meta, "pl")
    risk_rows = []
    for row in psf_rows:
        missing_mass = finite_float(row.get("theta_missing_crab_probability_mass")) or 0.0
        neff = finite_float(row.get("effective_events")) or 0.0
        containment_warning = truthy(row.get("containment_warning"))
        risk = missing_mass > 0.0 or neff < 200.0 or containment_warning
        if risk:
            risk_rows.append(row)
    return {
        "name": name,
        "config": {key: str(value) for key, value in config.items()},
        "selector_rows": selector_rows,
        "included_cells": included,
        "psf_rows": psf_rows,
        "psf_risk_rows": risk_rows,
        "fit_meta": fit_meta,
        "fit_npz": fit_npz,
        "stage_g_rows": stage_g_rows,
        "logpar": logpar,
        "pl": pl,
        "max_pull": max_abs_pull(fit_npz),
        "low_nhit_pass5_ratio": low_nhit_pass5_ratio(stage_g_rows),
        "status": "complete" if config["stage_f_meta"].exists() and config["stage_g_summary"].exists() else "pending",
    }


def official_pass5_points() -> List[Dict[str, float]]:
    rows = read_csv_rows(OFFICIAL_PASS5_CSV)
    out: List[Dict[str, float]] = []
    for row in rows:
        energy = finite_float(row.get("energy_tev") or row.get("E_TeV") or row.get("e_ref_tev"))
        flux = finite_float(row.get("e2_dnde") or row.get("E2_dnde") or row.get("E2dnde"))
        err = finite_float(row.get("e2_dnde_err") or row.get("E2_dnde_err") or row.get("E2dnde_err"))
        dnde = finite_float(row.get("flux_per_tev_cm2_s"))
        dnde_err = finite_float(row.get("flux_per_tev_cm2_s_err") or row.get("flux_err_per_tev_cm2_s"))
        if flux is None and energy is not None and dnde is not None:
            flux = energy * energy * dnde
            err = None if dnde_err is None else energy * energy * dnde_err
        if energy is not None and flux is not None:
            out.append({"energy": energy, "flux": flux, "err": err if err is not None else 0.0})
    return out


def low_nhit_pass5_ratio(stage_g_rows: Sequence[Dict[str, str]]) -> Optional[float]:
    nhit_rows = [row for row in stage_g_rows if row.get("grouping") == "nhit"]
    if not nhit_rows:
        return None
    nhit_rows = sorted(
        nhit_rows,
        key=lambda row: interval_key(str(row.get("group_label") or row.get("nhit_bin") or "")),
    )
    low_row = nhit_rows[0]
    energy = finite_float(low_row.get("effective_energy_tev"))
    flux = finite_float(low_row.get("E2_dnde"))
    official = official_pass5_points()
    if energy is None or flux is None or not official:
        return None
    nearest = min(official, key=lambda point: abs(math.log(point["energy"] / energy)))
    if nearest["flux"] <= 0.0:
        return None
    return float(flux / nearest["flux"])


def logpar_flux(energy_tev: np.ndarray, params: Dict[str, object], pivot_tev: float = 3.0) -> Optional[np.ndarray]:
    values = params.get("parameters") if isinstance(params.get("parameters"), dict) else {}
    phi0 = finite_float(values.get("phi0"))
    alpha = finite_float(values.get("alpha"))
    beta = finite_float(values.get("beta"))
    if phi0 is None or alpha is None or beta is None:
        return None
    x = np.asarray(energy_tev, dtype=np.float64) / float(pivot_tev)
    dnde = phi0 * np.power(x, -(alpha + beta * np.log(x)))
    return energy_tev * energy_tev * dnde


def plot_sed_overlay(strategies: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=150)
    colors = {"baseline_v4": "#334155", "gap025": "#0f766e", "gap1": "#b45309"}
    energy = np.logspace(math.log10(0.2), math.log10(120.0), 240)
    for name, payload in strategies.items():
        curve = logpar_flux(energy, payload.get("logpar", {}))
        if curve is not None:
            ax.plot(energy, curve, color=colors.get(name, "black"), linewidth=1.8, label=f"{name} LogPar")
        rows = [row for row in payload.get("stage_g_rows", []) if row.get("grouping") == "nhit"]
        x: List[float] = []
        y: List[float] = []
        yerr: List[float] = []
        for row in rows:
            ex = finite_float(row.get("effective_energy_tev"))
            ey = finite_float(row.get("E2_dnde"))
            ee = finite_float(row.get("E2_dnde_err"))
            if ex is not None and ey is not None:
                x.append(ex)
                y.append(ey)
                yerr.append(0.0 if ee is None else ee)
        if x:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                markersize=4.2,
                capsize=2,
                color=colors.get(name, "black"),
                alpha=0.9,
                label=f"{name} Nhit points",
            )
    official = official_pass5_points()
    if official:
        ax.errorbar(
            [p["energy"] for p in official],
            [p["flux"] for p in official],
            yerr=[p["err"] for p in official],
            fmt="s",
            markersize=3.2,
            capsize=1.5,
            color="#6b7280",
            alpha=0.65,
            label="official pass5",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel(r"$E^2 dN/dE$ (TeV cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Crab SED v5 PredE Binning Ablation")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_psf_heatmap(payload: Dict[str, object], output_path: Path) -> None:
    rows = payload.get("psf_rows", [])
    if not rows:
        return
    plt = setup_matplotlib()
    nhit_bins = sorted({row["nhit_bin"] for row in rows}, key=interval_key)
    pred_bins = sorted({row["predE_bin"] for row in rows}, key=interval_key)
    values = np.full((len(nhit_bins), len(pred_bins)), np.nan, dtype=np.float64)
    for row in rows:
        i = nhit_bins.index(row["nhit_bin"])
        j = pred_bins.index(row["predE_bin"])
        neff = finite_float(row.get("effective_events"))
        missing = finite_float(row.get("theta_missing_crab_probability_mass")) or 0.0
        residual = finite_float(row.get("tail_weight_fraction_above_core_fit")) or 0.0
        score = 0.0
        if neff is None or neff <= 0.0:
            score = 3.0
        else:
            score += max(0.0, math.log10(200.0 / neff))
        score += 5.0 * missing + residual
        values[i, j] = score
    fig, ax = plt.subplots(figsize=(1.25 * len(pred_bins) + 2.4, 0.58 * len(nhit_bins) + 2.0), dpi=150)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(pred_bins)))
    ax.set_xticklabels(pred_bins, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(nhit_bins)))
    ax.set_yticklabels(nhit_bins, fontsize=7)
    ax.set_title(f"{payload['name']} PSF risk score")
    ax.set_xlabel("log10(E_pred / GeV) bin")
    ax.set_ylabel("Nhit bin")
    fig.colorbar(im, ax=ax, shrink=0.82, label="risk score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_interval(label: str) -> Tuple[Optional[float], Optional[float]]:
    label = label.strip()
    if label.startswith("[") and label.endswith(")"):
        low, high = label[1:-1].split(",", 1)
        return float(low), float(high)
    if label.startswith("<"):
        return None, float(label[1:])
    if label.startswith(">="):
        return float(label[2:]), None
    return None, None


def interval_key(label: str) -> float:
    low, high = parse_interval(label)
    if low is None and high is None:
        return 1.0e30
    if low is None:
        return -1.0e30
    if high is None:
        return 1.0e30
    return low


def fmt_float(value: object, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "pending"
    return f"{number:.{digits}g}"


def strategy_summary_table(strategies: Dict[str, Dict[str, object]]) -> str:
    rows: List[str] = []
    for name, payload in strategies.items():
        logpar = payload.get("logpar", {})
        params = logpar.get("parameters") if isinstance(logpar, dict) and isinstance(logpar.get("parameters"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{html_escape(name)}</td>"
            f"<td>{html_escape(payload.get('status'))}</td>"
            f"<td class=\"num\">{len(payload.get('included_cells', []))}</td>"
            f"<td class=\"num\">{len(payload.get('psf_risk_rows', []))}</td>"
            f"<td class=\"num\">{fmt_float(params.get('phi0'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(params.get('alpha'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(params.get('beta'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(logpar.get('chi2'), 5)}/{fmt_float(logpar.get('ndof'), 5)}</td>"
            f"<td class=\"num\">{fmt_float(payload.get('max_pull'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(payload.get('low_nhit_pass5_ratio'), 4)}</td>"
            "</tr>"
        )
    return "".join(rows)


def psf_table(payload: Dict[str, object], limit: int = 24) -> str:
    rows: List[str] = []
    for row in list(payload.get("psf_risk_rows", []))[:limit]:
        rows.append(
            "<tr>"
            f"<td>{html_escape(row.get('cell_id'))}</td>"
            f"<td>{html_escape(row.get('nhit_bin'))}</td>"
            f"<td>{html_escape(row.get('predE_bin'))}</td>"
            f"<td class=\"num\">{fmt_float(row.get('effective_events'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(row.get('theta_missing_crab_probability_mass'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(row.get('sigma_deg'), 4)}</td>"
            f"<td class=\"num\">{fmt_float(row.get('r_opt_deg'), 4)}</td>"
            "</tr>"
        )
    return "".join(rows) if rows else "<tr><td colspan=\"7\">No PSF risk rows found or Stage B pending.</td></tr>"


def write_report(strategies: Dict[str, Dict[str, object]], figures: Dict[str, Path]) -> None:
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    generated = time_now_string()
    psf_sections: List[str] = []
    for name, payload in strategies.items():
        fig = figures.get(f"psf_{name}")
        fig_html = f'<figure><img src="{html_escape(relative_path(fig))}" alt="{html_escape(name)} PSF risk"></figure>' if fig and fig.exists() else "<p>PSF heatmap pending.</p>"
        profiles = figures.get(f"psf_profiles_{name}")
        profiles_html = (
            f'<figure><img src="{html_escape(relative_path(profiles))}" alt="{html_escape(name)} PSF radial profiles"></figure>'
            if profiles and profiles.exists()
            else ""
        )
        psf_sections.append(
            f"""
<section>
<h3>{html_escape(name)}</h3>
{fig_html}
{profiles_html}
<div class="table-wrap"><table>
<thead><tr><th>cell</th><th>Nhit</th><th>predE</th><th class="num">Neff</th><th class="num">missing theta</th><th class="num">sigma deg</th><th class="num">r_opt deg</th></tr></thead>
<tbody>{psf_table(payload)}</tbody>
</table></div>
</section>
"""
        )
    sed_fig = figures.get("sed_overlay")
    sed_html = f'<figure><img src="{html_escape(relative_path(sed_fig))}" alt="SED overlay"></figure>' if sed_fig and sed_fig.exists() else "<p>SED overlay pending.</p>"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crab SED v5 PredE 分箱消融报告</title>
<style>
body {{ margin:0; background:#f7f8f9; color:#17212b; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.62; }}
main {{ max-width:1240px; margin:0 auto; padding:38px 20px 64px; }}
h1 {{ margin:0 0 10px; font-size:36px; line-height:1.15; }}
h2 {{ margin:38px 0 12px; font-size:25px; border-bottom:1px solid #d7dee3; padding-bottom:8px; }}
h3 {{ margin:26px 0 10px; font-size:19px; }}
.lead {{ color:#53606a; max-width:960px; font-size:16px; }}
.table-wrap {{ overflow-x:auto; border:1px solid #d7dee3; border-radius:8px; background:white; margin:14px 0; }}
table {{ width:100%; border-collapse:collapse; min-width:920px; font-size:14px; }}
th,td {{ border-bottom:1px solid #d7dee3; padding:9px 11px; text-align:left; vertical-align:top; }}
th {{ background:#eef2f4; white-space:nowrap; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
figure {{ margin:16px 0; padding:12px; border:1px solid #d7dee3; border-radius:8px; background:white; }}
figure img {{ display:block; width:100%; height:auto; }}
code {{ background:#edf1f3; border-radius:4px; padding:1px 4px; }}
.note {{ border-left:4px solid #0f766e; background:white; border-radius:8px; padding:14px 16px; margin:16px 0; }}
</style>
</head>
<body><main>
<h1>Crab SED v5 PredE 分箱消融报告</h1>
<p class="lead">对比 baseline_v4、gap025 与 gap1 三套 <code>log10(E_pred/GeV)</code> 分箱。gap025/gap1 使用独立 prefit MC selector，主 fit 排除 <code>&lt;2</code> 和 <code>&gt;=6</code> 尾箱；PSF 风险仅标注，不后验删除。</p>
<p class="lead">Generated: {html_escape(generated)}</p>

<section>
<h2>汇总</h2>
<div class="table-wrap"><table>
<thead><tr><th>strategy</th><th>status</th><th class="num">fit cells</th><th class="num">PSF risk cells</th><th class="num">LogPar phi0</th><th class="num">alpha</th><th class="num">beta</th><th class="num">chi2/ndof</th><th class="num">max pull</th><th class="num">low-Nhit pass5 ratio</th></tr></thead>
<tbody>{strategy_summary_table(strategies)}</tbody>
</table></div>
<div class="note">Status 为 pending 时表示对应 Slurm 全量产物还没生成；脚本会在产物出现后自动纳入同一张报告。</div>
</section>

<section>
<h2>SED Overlay</h2>
{sed_html}
</section>

<section>
<h2>Stage B Rayleigh PSF Diagnostics</h2>
{''.join(psf_sections)}
</section>

<section>
<h2>Artifacts</h2>
<div class="table-wrap"><table>
<thead><tr><th>strategy</th><th>selector</th><th>Stage F</th><th>Stage G</th></tr></thead>
<tbody>
{''.join(
    '<tr>'
    f'<td>{html_escape(name)}</td>'
    f'<td><code>{html_escape(payload["config"]["selector"])}</code></td>'
    f'<td><code>{html_escape(payload["config"]["stage_f_meta"])}</code></td>'
    f'<td><code>{html_escape(payload["config"]["stage_g_summary"])}</code></td>'
    '</tr>'
    for name, payload in strategies.items()
)}
</tbody>
</table></div>
</section>
</main></body></html>
"""
    REPORT_HTML.write_text(html_text, encoding="utf-8")


def time_now_string() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    strategies = {name: load_strategy(name, config) for name, config in STRATEGIES.items()}
    figures: Dict[str, Path] = {}
    sed_overlay = ASSET_DIR / "v5_predbin_ablation_sed_overlay.png"
    plot_sed_overlay(strategies, sed_overlay)
    figures["sed_overlay"] = sed_overlay
    for name, payload in strategies.items():
        out = ASSET_DIR / f"{name}_psf_risk_heatmap.png"
        plot_psf_heatmap(payload, out)
        if out.exists():
            figures[f"psf_{name}"] = out
        stage_b_summary = Path(str(payload["config"]["stage_b_summary"]))
        profiles = stage_b_summary.with_name("psf_radial_profiles_grid.png")
        if profiles.exists():
            figures[f"psf_profiles_{name}"] = profiles
    write_report(strategies, figures)
    print(f"Wrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
