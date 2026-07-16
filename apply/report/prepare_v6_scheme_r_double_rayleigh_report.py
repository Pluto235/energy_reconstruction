#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "apply/report"
BASE_RUN_ID = "v6_64748_nhit100_reselect44_split56_miss030"
PSF_RUN_ID = f"{BASE_RUN_ID}_double_rayleigh"
EXPERIMENT_ID = f"{PSF_RUN_ID}_scheme_R_fixed712979"
TARGET_CONTAINMENT = 0.7129790300890827
REPORT = REPORT_DIR / "crab_sed_v6_64748_nhit100_reselect44_scheme_R_double_rayleigh_report.html"
ASSET_DIR = REPORT_DIR / "assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh-scheme-R"
COMPARISON_JSON = ASSET_DIR / "scheme_R_double_rayleigh_comparison.json"
SELECTOR = REPO_ROOT / f"apply/config/cell_selector_{BASE_RUN_ID}_fit.csv"
R1_ASSET_DIR = REPORT_DIR / "assets/v6-64748-nhit100-reselect44-split56-miss030/scheme-R"
B2_ASSET_DIR = REPORT_DIR / "assets/v6-64748-nhit100-reselect44-split56-miss030-double-rayleigh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 5) -> str:
    number = finite(value)
    if number is None:
        return "n/a"
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e4 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def selector_rows() -> dict[int, dict[str, str]]:
    with SELECTOR.open(newline="", encoding="utf-8") as handle:
        return {int(float(row["cell_id"])): row for row in csv.DictReader(handle)}


def copy_reference_images() -> dict[str, str]:
    sources = {
        "r1_pull": R1_ASSET_DIR / "pull_grid_logpar.png",
        "b2_pull": B2_ASSET_DIR / "pull_grid_logpar.png",
        "r1_ratio": R1_ASSET_DIR / "sed_points_ratio.png",
        "b2_ratio": B2_ASSET_DIR / "sed_points_ratio.png",
    }
    names = {
        "r1_pull": "reference_scheme_R_single_rayleigh_pull_grid_logpar.png",
        "b2_pull": "reference_scheme_B_double_rayleigh_pull_grid_logpar.png",
        "r1_ratio": "reference_scheme_R_single_rayleigh_sed_points_ratio.png",
        "b2_ratio": "reference_scheme_B_double_rayleigh_sed_points_ratio.png",
    }
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for key, source in sources.items():
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing comparison image: {source}")
        shutil.copy2(source, ASSET_DIR / names[key])
    prefix = ASSET_DIR.relative_to(REPORT_DIR).as_posix()
    return {key: f"{prefix}/{name}" for key, name in names.items()}


def build_insert(payload: dict[str, Any], images: dict[str, str]) -> str:
    records = payload["comparison"]
    r1_record, b2_record, r2_record = records
    comparison_rows = [
        [
            esc(record["label"]),
            esc(record["response_contract"]),
            fmt(record["phi0"], 7),
            fmt(record["alpha"], 6),
            fmt(record["beta"], 6),
            f"{fmt(record['chi2'], 5)}/{esc(record['ndof'])}",
            fmt(record["chi2_over_ndof"], 5),
            fmt(record["max_abs_pull"], 5),
            fmt(record["total_obs_over_pass5"], 5),
        ]
        for record in records
    ]

    ids = [record["branch_id"] for record in records]
    by_cell = selector_rows()
    pull_rows = []
    for row in payload["large_pull_cells"]:
        cell_id = int(row["cell_id"])
        cell = by_cell[cell_id]
        pull_rows.append(
            [
                esc(cell_id),
                f"<code>{esc(cell.get('nhit_bin'))}</code>",
                f"<code>{esc(cell.get('predE_bin'))}</code>",
                fmt(row.get(ids[0]), 5),
                fmt(row.get(ids[1]), 5),
                fmt(row.get(ids[2]), 5),
                fmt(row.get("max_abs_pull"), 5),
            ]
        )

    r2_large_pull_count = sum(
        finite(row.get(ids[2])) is not None and abs(float(row[ids[2]])) >= float(payload["large_pull_threshold"])
        for row in payload["cell_pulls"]
    )
    best_reference_chi2 = min(float(r1_record["chi2"]), float(b2_record["chi2"]))
    global_fit_result = "does not improve" if float(r2_record["chi2"]) >= best_reference_chi2 else "improves"

    shared = payload["shared_artifacts"]
    provenance_rows = [
        [esc(name), f"<code>{esc(item['path'])}</code>", f"<code>{esc(item['sha256'])}</code>"]
        for name, item in shared.items()
    ]
    return f"""<section id="scheme-r-double-rayleigh-contract">
    <h2>Scheme R Double-Rayleigh Contract</h2>
    <div class="okbox"><strong>Active R-2R contract:</strong>
      Stage B uses the constrained double-Rayleigh radial PSF and solves
      <code>F(r_opt)={TARGET_CONTAINMENT}</code>. The Stage D background and Stage E signal are reused byte-for-byte from
      Scheme B double-Rayleigh because the aperture, Stage C events, and annnorm configuration are identical.
      Stage F/G replace only the response contract with
      <code>Aeff_R={TARGET_CONTAINMENT}*Aeff_nominal</code>, applied exactly once; downstream containment remains 1 and the
      aperture-conditioned response is not used.
    </div>
    <p>This is an isolated experiment named <code>{esc(EXPERIMENT_ID)}</code>. It does not overwrite either
      <a href="crab_sed_v6_64748_nhit100_reselect44_scheme_R_report.html">Scheme R single-Rayleigh</a> or
      <a href="crab_sed_v6_64748_nhit100_reselect44_scheme_B_report.html">Scheme B double-Rayleigh</a>.</p>
    <h3>Shared-artifact provenance</h3>
    {html_table(["Artifact", "Path", "SHA256"], provenance_rows)}
  </section>

  <section id="three-branch-comparison">
    <h2>Three-Branch Controlled Comparison</h2>
    <p>R-1R versus R-2R isolates the PSF/aperture change under the same fixed response. B-2R versus R-2R isolates the
      response contract under the same double-Rayleigh aperture and the same Stage D/E arrays. The reported
      <code>total obs/pass5</code> forward-folds the official Pass5 spectrum through each branch response and its unchanged exposure.</p>
    {html_table(["Branch", "Response contract", "phi0", "alpha", "beta", "chi2/ndof", "chi2/ndof value", "max |pull|", "total obs/pass5"], comparison_rows)}
    <div class="callout"><strong>Global-fit conclusion:</strong> Scheme R double-Rayleigh {global_fit_result} the global fit:
      <code>chi2/ndof={fmt(r2_record['chi2'], 5)}/{esc(r2_record['ndof'])}</code>, compared with
      <code>{fmt(r1_record['chi2'], 5)}/{esc(r1_record['ndof'])}</code> for R-1R and
      <code>{fmt(b2_record['chi2'], 5)}/{esc(b2_record['ndof'])}</code> for B-2R. Its maximum absolute pull is
      <code>{fmt(r2_record['max_abs_pull'], 5)}</code>, with {r2_large_pull_count} selected cells at
      <code>|pull| &gt;= {fmt(payload['large_pull_threshold'], 1)}</code>; <code>total obs/pass5</code> rises to
      <code>{fmt(r2_record['total_obs_over_pass5'], 5)}</code>.</div>
    <h3>Large-Pull Migration</h3>
    <p>The table includes the union of selected cells with <code>|pull| &gt;= 5</code> in any of the three branches.</p>
    {html_table(["Cell", "Nhit", "predE", "R-1R pull", "B-2R pull", "R-2R pull", "max |pull|"], pull_rows)}
    <h3>Reference pull grids</h3>
    <div class="figgrid">
      <figure class="figure"><img src="{esc(images['r1_pull'])}" alt="Scheme R single-Rayleigh LogPar pull grid"><figcaption>Scheme R single-Rayleigh LogPar pull grid</figcaption></figure>
      <figure class="figure"><img src="{esc(images['b2_pull'])}" alt="Scheme B double-Rayleigh LogPar pull grid"><figcaption>Scheme B double-Rayleigh LogPar pull grid</figcaption></figure>
    </div>
    <h3>Reference SED ratios</h3>
    <div class="figgrid">
      <figure class="figure"><img src="{esc(images['r1_ratio'])}" alt="Scheme R single-Rayleigh SED ratios"><figcaption>Scheme R single-Rayleigh SED ratios</figcaption></figure>
      <figure class="figure"><img src="{esc(images['b2_ratio'])}" alt="Scheme B double-Rayleigh SED ratios"><figcaption>Scheme B double-Rayleigh SED ratios</figcaption></figure>
    </div>
    <p>The R-2R pull grid and SED ratio are the active Stage F/G figures in the diagnostics below.</p>
  </section>
"""


def main() -> None:
    if not REPORT.exists():
        raise FileNotFoundError(f"Missing generated R-2R report: {REPORT}")
    if not COMPARISON_JSON.exists():
        raise FileNotFoundError(f"Missing R-2R comparison payload: {COMPARISON_JSON}")
    text = REPORT.read_text(encoding="utf-8")
    if "scheme-r-double-rayleigh-contract" in text:
        raise ValueError("R-2R report already contains prepared comparison sections")
    payload = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
    images = copy_reference_images()
    insert = build_insert(payload, images)
    marker = "<section>\n    <h2>Stage C Time Audit</h2>"
    if marker not in text:
        raise ValueError("Could not find Stage C insertion marker in generated report")
    text = text.replace(marker, insert + "\n" + marker, 1)
    tmp = REPORT.with_suffix(REPORT.suffix + ".scheme_r_double_rayleigh_tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(REPORT)
    payload.update({"report_path": str(REPORT), "report_sha256": sha256(REPORT)})
    COMPARISON_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Scheme R double-Rayleigh report: {REPORT}")
    print(f"Comparison payload: {COMPARISON_JSON}")


if __name__ == "__main__":
    main()
