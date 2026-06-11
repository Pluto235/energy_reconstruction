#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the integrated v2 Crab SED Stage A-G HTML report.")
    parser.add_argument("--output-html", type=str, default="apply/report/crab_sed_v2_stage_a_to_g_report.html")
    parser.add_argument("--roadmap-html", type=str, default="apply/report/roadmap_v2.html")
    parser.add_argument("--skymap-html", type=str, default="apply/report/crab_v2_cell_skymaps.html")
    parser.add_argument("--stage-a-dir", type=str, default="apply/output/stage_a_v2_raw65")
    parser.add_argument("--stage-b-dir", type=str, default="apply/output/stage_b_v2_raw65/current")
    parser.add_argument("--stage-c-dir", type=str, default="apply/output/stage_c_v2_raw65/current")
    parser.add_argument("--stage-d-dir", type=str, default="apply/output/stage_d_v2_raw65/current")
    parser.add_argument("--stage-e-dir", type=str, default="apply/output/stage_e_v2_raw65/current")
    parser.add_argument("--stage-f-dir", type=str, default="apply/output/stage_f_v2_baseline26/current")
    parser.add_argument("--stage-g-dir", type=str, default="apply/output/stage_g_v2_baseline26/current")
    parser.add_argument("--stage-f-metadata-name", type=str, default="fit_v2_baseline26_metadata.json")
    parser.add_argument("--stage-g-metadata-name", type=str, default="sed_points_v2_baseline26_metadata.json")
    parser.add_argument("--stage-f-report-html", type=str, default="apply/report/stage_f_v2_baseline26_report.html")
    parser.add_argument("--stage-g-report-html", type=str, default="apply/report/stage_g_v2_baseline26_report.html")
    parser.add_argument(
        "--fit-cell-counts-skymap",
        type=str,
        default="",
    )
    parser.add_argument("--raw-ledger-csv", type=str, default="apply/config/cell_ledger_v2_raw65.csv")
    parser.add_argument("--baseline-selector-csv", type=str, default="apply/config/cell_selector_v2_baseline26.csv")
    parser.add_argument("--baseline-name", type=str, default="v2_baseline26")
    return parser.parse_args()


def rel(path: str | Path, start: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    try:
        return os.path.relpath(p, start=start.resolve())
    except ValueError:
        return str(p)


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def fmt(value: object, digits: int = 5) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a"


def h(value: object) -> str:
    return html.escape(str(value))


def spectrum_label(model_name: object) -> str:
    value = str(model_name or "").lower()
    if value == "pl":
        return "PL"
    if value == "logpar":
        return "LogPar"
    return str(model_name or "n/a")


def stage_card(stage: str, title: str, meta: Dict[str, object], artifact: Path, notes: Sequence[str]) -> str:
    run_id = meta.get("run_id") or meta.get("slurm_job_id") or meta.get("run_dir") or "n/a"
    promotion = meta.get("promotion") if isinstance(meta.get("promotion"), dict) else {}
    quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
    quality_gate = meta.get("quality_gate") if isinstance(meta.get("quality_gate"), dict) else {}
    status = (
        promotion.get("status")
        if isinstance(promotion, dict)
        else None
    ) or (
        quality.get("status")
        if isinstance(quality, dict)
        else None
    ) or (
        quality_gate.get("status")
        if isinstance(quality_gate, dict)
        else None
    ) or meta.get("absolute_effective_area_status") or "n/a"
    outputs = meta.get("outputs") if isinstance(meta.get("outputs"), dict) else {}
    links = []
    if artifact.exists():
        links.append(f'<a href="{h(rel(artifact, REPORT_DIR))}">metadata</a>')
    for key in ["summary_md", "summary_csv", "report_html", "npz"]:
        value = outputs.get(key) if isinstance(outputs, dict) else None
        if value and Path(str(value)).exists():
            links.append(f'<a href="{h(rel(str(value), REPORT_DIR))}">{h(key)}</a>')
    links_html = " · ".join(links) if links else "n/a"
    note_html = "".join(f"<li>{h(note)}</li>" for note in notes)
    return f"""
    <article class="stage-card">
      <div class="stage-label">{h(stage)}</div>
      <h3>{h(title)}</h3>
      <dl>
        <dt>Run</dt><dd><code>{h(run_id)}</code></dd>
        <dt>Status</dt><dd><code>{h(status)}</code></dd>
        <dt>Artifacts</dt><dd>{links_html}</dd>
      </dl>
      <ul>{note_html}</ul>
    </article>
    """


def figure(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    return f'<figure><img src="{h(rel(path, REPORT_DIR))}" alt="{h(caption)}"><figcaption>{h(caption)}</figcaption></figure>'


def wide_figure(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    return f'<figure class="wide"><img src="{h(rel(path, REPORT_DIR))}" alt="{h(caption)}"><figcaption>{h(caption)}</figcaption></figure>'


def table_from_rows(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    head = "".join(f"<th>{h(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{h(row.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def selector_ids(rows: Sequence[Dict[str, str]], include: bool) -> List[int]:
    values: List[int] = []
    for row in rows:
        raw = str(row.get("include", "")).strip().lower()
        row_include = raw in {"1", "true", "yes", "y", "include"}
        if row_include == include:
            values.append(int(row["cell_id"]))
    return values


REPORT_DIR = Path(".")


def main() -> None:
    global REPORT_DIR
    args = parse_args()
    output_html = (REPO_ROOT / args.output_html).resolve()
    REPORT_DIR = output_html.parent

    stage_a_dir = (REPO_ROOT / args.stage_a_dir).resolve()
    stage_b_dir = (REPO_ROOT / args.stage_b_dir).resolve()
    stage_c_dir = (REPO_ROOT / args.stage_c_dir).resolve()
    stage_d_dir = (REPO_ROOT / args.stage_d_dir).resolve()
    stage_e_dir = (REPO_ROOT / args.stage_e_dir).resolve()
    stage_f_dir = (REPO_ROOT / args.stage_f_dir).resolve()
    stage_g_dir = (REPO_ROOT / args.stage_g_dir).resolve()

    raw_rows = read_csv_rows((REPO_ROOT / args.raw_ledger_csv).resolve())
    selector_rows = read_csv_rows((REPO_ROOT / args.baseline_selector_csv).resolve())
    baseline_name = str(args.baseline_name or "v2_baseline26")
    included_ids = selector_ids(selector_rows, True)
    excluded_ids = selector_ids(selector_rows, False)
    role_counts: Dict[str, int] = {}
    for row in raw_rows:
        role = row.get("cell_role", "")
        role_counts[role] = role_counts.get(role, 0) + 1

    stage_a_meta_path = stage_a_dir / "response_2d_v2_raw65_metadata.json"
    stage_b_meta_path = stage_b_dir / "psf_v2_raw65_metadata.json"
    stage_c_meta_path = stage_c_dir / "obs_events_metadata.json"
    stage_d_meta_path = stage_d_dir / "background_v2_raw65_metadata.json"
    stage_e_meta_path = stage_e_dir / "signal_v2_raw65_metadata.json"
    stage_f_meta_path = stage_f_dir / args.stage_f_metadata_name
    stage_g_meta_path = stage_g_dir / args.stage_g_metadata_name

    stage_a = load_json(stage_a_meta_path)
    stage_b = load_json(stage_b_meta_path)
    stage_c = load_json(stage_c_meta_path)
    stage_d = load_json(stage_d_meta_path)
    stage_e = load_json(stage_e_meta_path)
    stage_f = load_json(stage_f_meta_path)
    stage_g = load_json(stage_g_meta_path)

    totals_e = stage_e.get("totals") if isinstance(stage_e.get("totals"), dict) else {}
    preferred = stage_f.get("preferred_fit") if isinstance(stage_f.get("preferred_fit"), dict) else {}
    fits = stage_f.get("fits") if isinstance(stage_f.get("fits"), dict) else {}
    preferred_model = str(preferred.get("model", "pl") if isinstance(preferred, dict) else "pl").lower()
    preferred_error = str(preferred.get("error_mode", "conservative") if isinstance(preferred, dict) else "conservative").lower()
    preferred_key = f"{preferred_model}_{preferred_error}"
    fit_preferred = fits.get(preferred_key, {}) if isinstance(fits, dict) else {}
    fit_params = fit_preferred.get("parameters") if isinstance(fit_preferred.get("parameters"), dict) else {}
    fit_pl = {}
    if isinstance(fits, dict):
        fit_pl = fits.get("pl_conservative", {}) if isinstance(fits.get("pl_conservative"), dict) else {}
    fit_pl_params = fit_pl.get("parameters") if isinstance(fit_pl.get("parameters"), dict) else {}
    quality_e = stage_e.get("quality_gate") if isinstance(stage_e.get("quality_gate"), dict) else {}
    contract_e = stage_e.get("stage_d_contract") if isinstance(stage_e.get("stage_d_contract"), dict) else {}
    frozen_g = stage_g.get("frozen_spectrum") if isinstance(stage_g.get("frozen_spectrum"), dict) else {}
    frozen_g_label = spectrum_label(frozen_g.get("model") if isinstance(frozen_g, dict) else preferred_model)
    stage_b_warnings = stage_b.get("warning_rows", stage_b.get("warnings", []))

    sed_points = stage_g.get("points", []) if isinstance(stage_g.get("points"), list) else []
    sed_table_rows = []
    for point in sed_points:
        if not isinstance(point, dict):
            continue
        sed_table_rows.append(
            {
                "grouping": point.get("grouping", ""),
                "group": point.get("group_label", ""),
                "cells": ",".join(str(v) for v in point.get("cell_ids", [])) if isinstance(point.get("cell_ids"), list) else "",
                "E_eff TeV": fmt(point.get("effective_energy_tev"), 5),
                "E2 dN/dE": fmt(point.get("E2_dnde"), 5),
                "err": fmt(point.get("E2_dnde_err"), 4),
                "ratio StageF model": fmt(point.get("ratio_to_stage_f_model", point.get("ratio_to_stage_f_pl")), 4),
            }
        )

    role_table = [{"role": key, "cells": value} for key, value in sorted(role_counts.items())]
    stage_cards = [
        stage_card(
            "A",
            "raw65 response",
            stage_a,
            stage_a_meta_path,
            [
                f"Cells: {fmt_int(len(raw_rows))}",
                f"Response type: {stage_a.get('response_type', 'n/a')}",
                f"S0: {fmt(stage_a.get('s0_m2'), 6)} m2",
            ],
        ),
        stage_card(
            "B",
            "raw65 PSF",
            stage_b,
            stage_b_meta_path,
            [
                f"Cells: {fmt_int(len(stage_b.get('cells', [])) if isinstance(stage_b.get('cells'), list) else len(raw_rows))}",
                "Crab declination PSF, r_opt and containment.",
                f"Warning rows: {fmt_int(len(stage_b_warnings) if isinstance(stage_b_warnings, list) else 0)}",
            ],
        ),
        stage_card(
            "C",
            "raw65 observation reduction",
            stage_c,
            stage_c_meta_path,
            [
                f"Output rows: {fmt_int(stage_c.get('processing', {}).get('selected_rows') if isinstance(stage_c.get('processing'), dict) else None)}",
                f"Source files: {fmt_int(stage_c.get('processing', {}).get('processed_file_count') if isinstance(stage_c.get('processing'), dict) else None)}",
                "Crab-centered ROI coverage retained for Stage D.",
            ],
        ),
        stage_card(
            "D",
            "ROI-local direct expectation",
            stage_d,
            stage_d_meta_path,
            [
                f"Background mode: {stage_d.get('background_model', {}).get('background_mode', 'n/a') if isinstance(stage_d.get('background_model'), dict) else 'n/a'}",
                f"Background form: {stage_d.get('background_model', {}).get('background_form', 'n/a') if isinstance(stage_d.get('background_model'), dict) else 'n/a'}",
            ],
        ),
        stage_card(
            "E",
            "raw65 signal table",
            stage_e,
            stage_e_meta_path,
            [
                f"Quality: {quality_e.get('status', 'n/a') if isinstance(quality_e, dict) else 'n/a'}",
                f"Formal sigma: {fmt(totals_e.get('formal_sigma'), 5) if isinstance(totals_e, dict) else 'n/a'}",
                f"Excess: {fmt(totals_e.get('excess'), 5) if isinstance(totals_e, dict) else 'n/a'}",
                "N_on/B_on and excess are diagnostics only for v2.0.",
            ],
        ),
        stage_card(
            "F",
            f"{baseline_name} forward folding",
            stage_f,
            stage_f_meta_path,
            [
                f"Included cells: {','.join(str(v) for v in included_ids)}",
                f"Preferred model: {spectrum_label(preferred_model)} / {preferred_error}",
                f"Preferred phi0: {fmt(fit_params.get('phi0'), 6) if isinstance(fit_params, dict) else 'n/a'}",
                f"PL gamma: {fmt(fit_pl_params.get('gamma'), 5) if isinstance(fit_pl_params, dict) else 'n/a'}",
            ],
        ),
        stage_card(
            "G",
            f"{baseline_name} diagnostic SED",
            stage_g,
            stage_g_meta_path,
            [
                f"SED points: {fmt_int(len(sed_points))}",
                f"Fixed Stage F {frozen_g_label} shape; per-group normalization only.",
            ],
        ),
    ]

    report_links = [
        ("Roadmap v2", REPO_ROOT / args.roadmap_html),
        ("v2 skymaps", REPO_ROOT / args.skymap_html),
        ("Stage E report", REPO_ROOT / "apply/report/stage_e_v2_raw65_report.html"),
        ("Stage F report", REPO_ROOT / args.stage_f_report_html),
        ("Stage G report", REPO_ROOT / args.stage_g_report_html),
    ]
    link_html = " · ".join(
        f'<a href="{h(rel(path, REPORT_DIR))}">{h(label)}</a>' for label, path in report_links if Path(path).exists()
    )

    figures = [
        wide_figure(
            REPO_ROOT / args.fit_cell_counts_skymap,
            f"{baseline_name} fit-cell Stage D counts skymap",
        )
        if args.fit_cell_counts_skymap
        else "",
        figure(stage_f_dir / "model_counts_vs_excess.png", "Stage F model counts vs excess"),
        figure(
            stage_f_dir / ("pull_grid_logpar.png" if preferred_model == "logpar" else "pull_grid_pl.png"),
            f"Stage F {spectrum_label(preferred_model)} conservative pulls",
        ),
        figure(stage_g_dir / "sed_points_stage_f_fullarray_pool1.png", "Stage G SED points"),
        figure(stage_g_dir / "sed_points_ratio.png", "Stage G SED ratios"),
        figure(stage_g_dir / "sed_point_cell_counts.png", "Stage G cell counts per point"),
    ]

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crab SED v2 Stage A-G Report</title>
<style>
:root {{ --bg:#f6f7f8; --fg:#1f2933; --muted:#5d6b76; --panel:#ffffff; --panel2:#eef2f5; --border:#d8e0e6; --accent:#006c67; --warn:#b7791f; --code:#edf2f7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans CJK SC","Microsoft YaHei",sans-serif; line-height:1.62; }}
main {{ max-width:1240px; margin:0 auto; padding:42px 20px 72px; }}
header {{ border-bottom:1px solid var(--border); padding-bottom:24px; margin-bottom:30px; }}
.eyebrow {{ color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
h1 {{ margin:8px 0 12px; font-size:clamp(32px,5vw,52px); line-height:1.08; }}
h2 {{ margin:38px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--border); font-size:24px; }}
h3 {{ margin:4px 0 12px; font-size:20px; }}
p {{ margin:10px 0; }}
a {{ color:var(--accent); }}
code {{ background:var(--code); border-radius:4px; padding:2px 5px; font-size:13px; }}
.lead {{ max-width:960px; color:var(--muted); font-size:17px; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.metric {{ min-height:112px; padding:16px; border:1px solid var(--border); background:var(--panel); border-radius:8px; }}
.label {{ color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }}
.value {{ margin-top:8px; font-size:25px; font-weight:800; overflow-wrap:anywhere; }}
.note {{ margin-top:7px; color:var(--muted); font-size:13px; }}
.callout {{ margin:18px 0; padding:16px 18px; border:1px solid var(--border); border-left:4px solid var(--warn); background:var(--panel); border-radius:8px; }}
.stage-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.stage-card {{ position:relative; padding:18px 18px 16px 76px; border:1px solid var(--border); background:var(--panel); border-radius:8px; min-height:190px; }}
.stage-label {{ position:absolute; left:18px; top:18px; width:40px; height:40px; border-radius:50%; display:grid; place-items:center; background:var(--accent); color:#fff; font-weight:900; }}
dl {{ display:grid; grid-template-columns:86px 1fr; gap:4px 10px; margin:0 0 10px; }}
dt {{ color:var(--muted); }}
dd {{ margin:0; overflow-wrap:anywhere; }}
ul {{ margin:10px 0 0 18px; padding:0; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--border); background:var(--panel); border-radius:8px; margin:16px 0; }}
table {{ width:100%; min-width:880px; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
th {{ background:var(--panel2); white-space:nowrap; }}
.figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
figure {{ margin:0; padding:12px; border:1px solid var(--border); background:var(--panel); border-radius:8px; }}
.wide {{ grid-column:1 / -1; }}
img {{ display:block; width:100%; height:auto; background:#fff; border-radius:4px; }}
figcaption {{ margin-top:8px; color:var(--muted); font-size:13px; }}
footer {{ margin-top:48px; padding-top:18px; border-top:1px solid var(--border); color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
@media (max-width:900px) {{ .metric-grid,.stage-grid,.figure-grid {{ grid-template-columns:1fr; }} .stage-card {{ padding-left:18px; padding-top:68px; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">LHAASO-WCDA · Crab SED v2</div>
    <h1>Stage A-G 完整结果报告</h1>
    <p class="lead">本页把 roadmap v2 的 raw65 ledger、{h(baseline_name)} selector、Stage A-E raw artifacts、Stage F forward-folding 和 Stage G diagnostic SED 串成一个可追踪结果页。</p>
  </header>

  <section>
    <h2>Run Summary</h2>
    <div class="metric-grid">
      <div class="metric"><div class="label">raw ledger</div><div class="value">{fmt_int(len(raw_rows))}</div><div class="note">v2_raw65 cells</div></div>
      <div class="metric"><div class="label">baseline</div><div class="value">{fmt_int(len(included_ids))}</div><div class="note">{h(baseline_name)} cells</div></div>
      <div class="metric"><div class="label">Stage E sigma</div><div class="value">{fmt(totals_e.get('formal_sigma'), 4) if isinstance(totals_e, dict) else 'n/a'}</div><div class="note">{h(contract_e.get('background_form', 'n/a') if isinstance(contract_e, dict) else 'n/a')}</div></div>
      <div class="metric"><div class="label">Stage F preferred</div><div class="value">{h(spectrum_label(preferred_model))}</div><div class="note">chi2/ndof {fmt(fit_preferred.get('chi2'), 4) if isinstance(fit_preferred, dict) else 'n/a'} / {h(fit_preferred.get('ndof', 'n/a') if isinstance(fit_preferred, dict) else 'n/a')}</div></div>
    </div>
    <div class="callout">
      v2.0 使用 <code>{h(contract_e.get('background_mode', 'n/a') if isinstance(contract_e, dict) else 'n/a')}</code> / <code>{h(contract_e.get('background_form', 'n/a') if isinstance(contract_e, dict) else 'n/a')}</code>。若 background form 是 direct expectation，则 Li-Ma 不适用；Stage E/F/G 的显著性、pull 和 SED residual 只作为 diagnostic 信息，不反向修改 selector。
    </div>
    <p>{link_html}</p>
  </section>

  <section>
    <h2>Ledger And Selector</h2>
    <p>raw65 role 分布：</p>
    <div class="table-wrap">{table_from_rows(role_table, ['role', 'cells'])}</div>
    <p>{h(baseline_name)} included cells: <code>{h(','.join(str(v) for v in included_ids))}</code></p>
    <p>excluded / diagnostic cells: <code>{h(','.join(str(v) for v in excluded_ids))}</code></p>
  </section>

  <section>
    <h2>Stages</h2>
    <div class="stage-grid">{''.join(stage_cards)}</div>
  </section>

  <section>
    <h2>Stage G SED Points</h2>
    <div class="table-wrap">{table_from_rows(sed_table_rows, ['grouping', 'group', 'cells', 'E_eff TeV', 'E2 dN/dE', 'err', 'ratio StageF model'])}</div>
  </section>

  <section>
    <h2>Figures</h2>
    <div class="figure-grid">{''.join(figures)}</div>
  </section>

  <footer>
    Generated from Stage metadata under <code>{h(str(REPO_ROOT / 'apply/output'))}</code>.
  </footer>
</main>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
