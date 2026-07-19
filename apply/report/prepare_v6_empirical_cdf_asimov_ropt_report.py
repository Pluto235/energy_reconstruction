#!/usr/bin/env python3
"""Add empirical-CDF/Asimov aperture evidence to the standard v6 Stage A-G report."""

from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
import html
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--iteration0-psf-metadata", type=Path, required=True)
    parser.add_argument("--iteration0-summary-csv", type=Path, required=True)
    parser.add_argument("--final-psf-npz", type=Path, required=True)
    parser.add_argument("--final-psf-metadata", type=Path, required=True)
    parser.add_argument("--final-summary-csv", type=Path, required=True)
    parser.add_argument("--response-npz", type=Path, required=True)
    parser.add_argument("--stage-d-npz", type=Path, required=True)
    parser.add_argument("--stage-e-npz", type=Path, required=True)
    parser.add_argument("--stage-f-metadata", type=Path, required=True)
    parser.add_argument("--stage-g-npz", type=Path, required=True)
    parser.add_argument("--stage-g-overlay", type=Path, required=True)
    parser.add_argument("--implementation-sha", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]).copy() for name in handle.files}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def relative(report: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), start=report.parent.resolve())


def figure(report: Path, path: Path, caption: str) -> str:
    return (
        '<figure class="figure">'
        f'<img src="{esc(relative(report, path))}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


def plot_iteration_stability(
    iteration0: list[dict[str, str]],
    final: list[dict[str, str]],
    png: Path,
    pdf: Path,
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = {int(row["cell_id"]): row for row in iteration0}
    second = {int(row["cell_id"]): row for row in final}
    cells = sorted(set(first) & set(second))
    r0 = np.asarray([float(first[cell]["adopted_r_opt_deg"]) for cell in cells])
    r1 = np.asarray([float(second[cell]["adopted_r_opt_deg"]) for cell in cells])
    c0 = np.asarray([float(first[cell]["adopted_containment"]) for cell in cells])
    c1 = np.asarray([float(second[cell]["adopted_containment"]) for cell in cells])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), dpi=180)
    lo = min(float(np.min(r0)), float(np.min(r1))) - 0.03
    hi = max(float(np.max(r0)), float(np.max(r1))) + 0.03
    axes[0].plot([lo, hi], [lo, hi], color="#777777", linewidth=0.8)
    axes[0].scatter(r0, r1, color="#0072B2", s=24)
    axes[0].set(xlabel="iteration 0 r_opt [deg]", ylabel="final r_opt [deg]", xlim=(lo, hi), ylim=(lo, hi))
    axes[0].set_title("Aperture-radius stability")
    axes[0].grid(alpha=0.2)

    delta = r1 - r0
    axes[1].bar(np.arange(len(cells)), delta, color=np.where(delta >= 0.0, "#009E73", "#D55E00"), width=0.78)
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(cells)), [str(cell) for cell in cells], rotation=90, fontsize=6)
    axes[1].set_xlabel("cell id")
    axes[1].set_ylabel("final - iteration 0 r_opt [deg]")
    axes[1].set_title("One-step LogPar/background update")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "max_abs_delta_r_deg": float(np.max(np.abs(delta))),
        "median_abs_delta_r_deg": float(np.median(np.abs(delta))),
        "max_abs_delta_containment": float(np.max(np.abs(c1 - c0))),
        "median_abs_delta_containment": float(np.median(np.abs(c1 - c0))),
    }


class ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "img":
            source = dict(attrs).get("src")
            if source:
                self.sources.append(source)


def main() -> None:
    args = parse_args()
    required = [
        args.report,
        args.iteration0_psf_metadata,
        args.iteration0_summary_csv,
        args.final_psf_npz,
        args.final_psf_metadata,
        args.final_summary_csv,
        args.response_npz,
        args.stage_d_npz,
        args.stage_e_npz,
        args.stage_f_metadata,
        args.stage_g_npz,
        args.stage_g_overlay,
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")
    args.asset_dir.mkdir(parents=True, exist_ok=True)

    iteration0_meta = load_json(args.iteration0_psf_metadata)
    final_meta = load_json(args.final_psf_metadata)
    fit_meta = load_json(args.stage_f_metadata)
    iteration0_rows = load_csv(args.iteration0_summary_csv)
    final_rows = load_csv(args.final_summary_csv)
    final_psf = load_npz(args.final_psf_npz)
    response = load_npz(args.response_npz)
    stage_d = load_npz(args.stage_d_npz)
    stage_e = load_npz(args.stage_e_npz)
    stage_g = load_npz(args.stage_g_npz)

    if len(iteration0_rows) != 44 or len(final_rows) != 44:
        raise ValueError("Both optimizer iterations must contain exactly 44 cells")
    first_ids = {int(row["cell_id"]) for row in iteration0_rows}
    final_ids = {int(row["cell_id"]) for row in final_rows}
    if first_ids != final_ids:
        raise ValueError("Iteration 0 and final optimized cell sets differ")

    copied: dict[str, Path] = {}
    for name, source in {
        "optimization_curves": args.final_psf_npz.parent / "empirical_cdf_asimov_curves.png",
        "optimization_curves_pdf": args.final_psf_npz.parent / "empirical_cdf_asimov_curves.pdf",
        "containment_grid": args.final_psf_npz.parent / "empirical_cdf_asimov_containment_grid.png",
        "containment_grid_pdf": args.final_psf_npz.parent / "empirical_cdf_asimov_containment_grid.pdf",
    }.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.asset_dir / source.name
        shutil.copy2(source, target)
        copied[name] = target

    stability_png = args.asset_dir / "empirical_cdf_asimov_iteration_stability.png"
    stability_pdf = args.asset_dir / "empirical_cdf_asimov_iteration_stability.pdf"
    stability = plot_iteration_stability(iteration0_rows, final_rows, stability_png, stability_pdf)

    psf_index = {int(cell): index for index, cell in enumerate(final_psf["cell_id"])}
    response_index = {int(cell): index for index, cell in enumerate(response["cell_id"])}
    stage_d_index = {int(cell): index for index, cell in enumerate(stage_d["cell_id"])}
    selected = sorted(final_ids)
    response_match = all(
        math.isclose(
            float(response["aperture_r_opt_deg"][response_index[cell]]),
            float(final_psf["r_opt_deg"][psf_index[cell]]),
            rel_tol=0.0,
            abs_tol=5.0e-7,
        )
        for cell in selected
    )
    stage_d_match = all(
        math.isclose(
            float(stage_d["r_opt_deg"][stage_d_index[cell]]),
            float(final_psf["r_opt_deg"][psf_index[cell]]),
            rel_tol=0.0,
            abs_tol=5.0e-7,
        )
        for cell in selected
    )
    containment_one = bool(np.array_equal(np.asarray(stage_e["containment_r_opt"]), np.ones_like(stage_e["containment_r_opt"])))
    all_near_max = all(float(row["adopted_z_fraction"]) >= 0.99 - 1.0e-12 for row in final_rows)
    upper_boundary = [int(row["cell_id"]) for row in final_rows if row["scan_upper_boundary"] == "True"]
    lower_boundary = [int(row["cell_id"]) for row in final_rows if row["scan_lower_boundary"] == "True"]
    groupings = {str(value) for value in stage_g["grouping"]}
    logpar = ((fit_meta.get("fits") or {}).get("logpar_conservative") or {})
    logpar_parameters = logpar.get("parameters") or {}

    final_r = np.asarray([float(row["adopted_r_opt_deg"]) for row in final_rows])
    final_c = np.asarray([float(row["adopted_containment"]) for row in final_rows])
    checks = [
        {"name": "44-cell selector frozen", "passed": len(final_ids) == 44 and first_ids == final_ids, "detail": sorted(final_ids)},
        {"name": "aperture response matches final Stage B", "passed": response_match, "detail": "selected-cell radii"},
        {"name": "Stage D aperture matches final Stage B", "passed": stage_d_match, "detail": "selected-cell radii"},
        {"name": "Stage E containment applied exactly zero extra times", "passed": containment_one, "detail": "all containment_r_opt == 1"},
        {"name": "99% smallest-radius rule", "passed": all_near_max, "detail": "all adopted Z/max(Z) >= 0.99"},
        {"name": "no upper scan-boundary optimum", "passed": not upper_boundary, "detail": upper_boundary},
        {"name": "Stage G contains both groupings", "passed": {"nhit", "predE"} <= groupings, "detail": sorted(groupings)},
        {"name": "final LogPar fit valid", "passed": logpar.get("valid") is True, "detail": logpar_parameters},
        {"name": "required Stage G overlay exists", "passed": args.stage_g_overlay.stat().st_size > 0, "detail": str(args.stage_g_overlay)},
    ]
    failed = [check["name"] for check in checks if not check["passed"]]
    if failed:
        raise RuntimeError(f"Empirical-CDF Asimov report validation failed: {failed}")

    initial_parameters = (iteration0_meta.get("aperture_optimization") or {}).get("spectrum_parameters") or {}
    final_input_parameters = (final_meta.get("aperture_optimization") or {}).get("spectrum_parameters") or {}
    summary_rows = [
        ["Radius scan", "0.20-2.00 deg in 0.01 deg steps"],
        ["Production rule", "smallest r with Z_A(r) >= 0.99 max Z_A"],
        ["Optimized cells", "44; selector unchanged"],
        ["Final r_opt min / median / max [deg]", f"{fmt(np.min(final_r))} / {fmt(np.median(final_r))} / {fmt(np.max(final_r))}"],
        ["Final containment min / median / max", f"{fmt(np.min(final_c))} / {fmt(np.median(final_c))} / {fmt(np.max(final_c))}"],
        ["Lower-bound cells", ", ".join(map(str, lower_boundary)) if lower_boundary else "none"],
        ["Iteration max / median |delta r| [deg]", f"{fmt(stability['max_abs_delta_r_deg'])} / {fmt(stability['median_abs_delta_r_deg'])}"],
        ["Initial LogPar (phi0, alpha, beta)", f"{fmt(initial_parameters.get('phi0'))}, {fmt(initial_parameters.get('alpha'))}, {fmt(initial_parameters.get('beta'))}"],
        ["Updated LogPar input (phi0, alpha, beta)", f"{fmt(final_input_parameters.get('phi0'))}, {fmt(final_input_parameters.get('alpha'))}, {fmt(final_input_parameters.get('beta'))}"],
        ["Final fitted LogPar (phi0, alpha, beta)", f"{fmt(logpar_parameters.get('phi0'))}, {fmt(logpar_parameters.get('alpha'))}, {fmt(logpar_parameters.get('beta'))}"],
    ]
    cell_rows = [
        [
            esc(row["cell_id"]),
            esc(row["nhit_bin"]),
            esc(row["predE_bin"]),
            fmt(row["old_r_opt_deg"]),
            fmt(row["exact_max_r_deg"]),
            fmt(row["adopted_r_opt_deg"]),
            fmt(row["adopted_containment"]),
            fmt(row["adopted_z_fraction"]),
            fmt(row["adopted_expected_signal"]),
            fmt(row["adopted_expected_background"]),
        ]
        for row in final_rows
    ]
    check_rows = [
        [
            esc(check["name"]),
            '<span class="status-pass">PASSED</span>' if check["passed"] else '<span class="status-fail">FAILED</span>',
            f"<code>{esc(check['detail'])}</code>",
        ]
        for check in checks
    ]
    section = f"""
  <section id="empirical-cdf-asimov-method">
    <h2>Empirical CDF + Actual-Background Aperture Optimization</h2>
    <div class="okbox"><strong>Production definition:</strong> each selected cell uses the smallest radius satisfying <code>Z_A(r) &gt;= 0.99 max Z_A</code>. The signal curve is forward-folded from the cumulative <code>r x E_true x theta</code> MC response with the current LogPar spectrum and the measured Crab exposure. The background curve is the analytic centered-disk integral of the cell's Stage D continuous Poisson surface.</div>
    <p>The complete MC angular-error distribution is retained. Catastrophic reconstruction tails therefore reduce <code>F(r)</code>, but the optimizer is not forced to enlarge the aperture to reach a fixed 68% or 71.3% containment. The resulting empirical containment is an output, not a target. Stage A is rebuilt with the adopted angular cut and Stage E containment is fixed to one, so the aperture efficiency enters the SED response exactly once.</p>
    {table(["Quantity", "Value"], summary_rows)}
    <div class="figgrid">
      {figure(args.report, copied['optimization_curves'], 'Per-cell Asimov-significance curves; green is the adopted 99%-plateau radius and dashed orange is the exact grid maximum')}
      {figure(args.report, copied['containment_grid'], 'Spectrum- and exposure-weighted empirical containment at each adopted aperture')}
      {figure(args.report, stability_png, 'One-step stability after updating both the LogPar spectrum and Stage D background')}
      {figure(args.report, args.stage_g_overlay, 'Final Stage G SED overlay with both v6 Nhit-grouped and predE-grouped points')}
    </div>
  </section>

  <section id="empirical-cdf-asimov-cells">
    <h2>Final Per-Cell Aperture Contract</h2>
    {table(["Cell", "Nhit", "predE", "old r_opt", "exact max r", "adopted r_opt", "F(r_opt)", "Z/max Z", "S(r_opt)", "B(r_opt)"], cell_rows)}
  </section>

  <section id="empirical-cdf-asimov-validation">
    <h2>Empirical-Aperture Validation</h2>
    {table(["Check", "Status", "Evidence"], check_rows)}
    <p>Implementation commit: <code>{esc(args.implementation_sha)}</code>. Iteration-0 optimizer: <code>{esc(args.iteration0_psf_metadata)}</code>. Final optimizer: <code>{esc(args.final_psf_metadata)}</code>.</p>
  </section>
"""

    report_text = args.report.read_text(encoding="utf-8")
    marker = "  </header>\n"
    if marker not in report_text:
        raise ValueError("Could not locate report header insertion point")
    report_text = report_text.replace(marker, marker + section, 1)
    temporary = args.report.with_suffix(".tmp.html")
    temporary.write_text(report_text, encoding="utf-8")
    temporary.replace(args.report)

    parser = ImageParser()
    parser.feed(args.report.read_text(encoding="utf-8"))
    missing_images = [source for source in parser.sources if not (args.report.parent / source).resolve().is_file()]
    validation = {
        "description": "Empirical-CDF Asimov r_opt final report validation",
        "implementation_sha": args.implementation_sha,
        "passed": not missing_images and not failed,
        "checks": checks,
        "iteration_stability": stability,
        "final_radius_summary_deg": {
            "min": float(np.min(final_r)),
            "median": float(np.median(final_r)),
            "max": float(np.max(final_r)),
        },
        "final_containment_summary": {
            "min": float(np.min(final_c)),
            "median": float(np.median(final_c)),
            "max": float(np.max(final_c)),
        },
        "lower_boundary_cells": lower_boundary,
        "upper_boundary_cells": upper_boundary,
        "image_count": len(parser.sources),
        "missing_images": missing_images,
        "report": str(args.report.resolve()),
        "stage_g_overlay": str(args.stage_g_overlay.resolve()),
    }
    validation_path = args.asset_dir / "empirical_cdf_asimov_report_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError(f"Final report validation failed: {validation}")
    print(f"Prepared empirical-CDF Asimov report: {args.report}")


if __name__ == "__main__":
    main()
