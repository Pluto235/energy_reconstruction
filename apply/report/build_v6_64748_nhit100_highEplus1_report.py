#!/usr/bin/env python3
from __future__ import annotations

import csv
from html.parser import HTMLParser
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "v6_64748_nhit100_highEplus1_split56"
REPORT_DIR = REPO_ROOT / "apply" / "report"
REPORT_PATH = REPORT_DIR / "crab_sed_v6_64748_nhit100_highEplus1_stage_a_to_g_report.html"
ASSET_DIR = REPORT_DIR / "assets" / "v6-64748-nhit100-highEplus1"
VALIDATION_JSON = ASSET_DIR / "report_validation.json"

LEDGER = REPO_ROOT / f"apply/config/cell_ledger_{RUN_ID}_candidate.csv"
PREFIT_SELECTOR = REPO_ROOT / f"apply/config/cell_selector_{RUN_ID}_prefit.csv"
FIT_SELECTOR = REPO_ROOT / f"apply/config/cell_selector_{RUN_ID}_fit.csv"
SELECTOR_META = REPO_ROOT / f"apply/config/cell_selector_{RUN_ID}_fit_metadata.json"
HIGH_E_DECISIONS = REPO_ROOT / f"apply/config/cell_selector_{RUN_ID}_highEplus1_decisions.csv"

STAGE_A = REPO_ROOT / f"apply/output/stage_a_{RUN_ID}"
STAGE_A_AP = REPO_ROOT / f"apply/output/stage_a_{RUN_ID}_aperture_conditioned"
STAGE_B = REPO_ROOT / f"apply/output/stage_b_{RUN_ID}/runs/{RUN_ID}_stage_b_psf"
STAGE_C = REPO_ROOT / f"apply/output/stage_c_{RUN_ID}/runs/{RUN_ID}_stage_c_halfyear"
STAGE_D = REPO_ROOT / f"apply/output/stage_d_{RUN_ID}_annnorm/runs/{RUN_ID}_stage_d_annnorm"
STAGE_E = REPO_ROOT / f"apply/output/stage_e_{RUN_ID}_containment1_annnorm/runs/{RUN_ID}_stage_e_containment1_annnorm"
STAGE_F = REPO_ROOT / f"apply/output/stage_f_{RUN_ID}/runs/{RUN_ID}_stage_f"
STAGE_G = REPO_ROOT / f"apply/output/stage_g_{RUN_ID}/runs/{RUN_ID}_stage_g"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def rel(path: Path | str) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    return html.escape(os.path.relpath(target, start=REPORT_PATH.parent))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 4) -> str:
    out = finite_float(value)
    if out is None:
        return "n/a"
    if out == 0:
        return "0"
    if abs(out) >= 1.0e5 or abs(out) < 1.0e-3:
        return f"{out:.{digits}e}"
    return f"{out:.{digits}g}"


def fmt_int(value: Any) -> str:
    out = finite_float(value)
    return "n/a" if out is None else f"{int(round(out)):,}"


def table(headers: list[str], rows: Iterable[Iterable[Any]], classes: str = "") -> str:
    class_attr = f' class="{classes}"' if classes else ""
    parts = [f"<table{class_attr}><thead><tr>"]
    parts.extend(f"<th>{esc(header)}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{cell}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def figure(path: Path, caption: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected report image: {path}")
    return (
        '<figure class="figure">'
        f'<img src="{rel(path)}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption>"
        "</figure>"
    )


def global_cutflow_map(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        if row.get("scope") != "global":
            continue
        try:
            out[str(row["step"])] = int(float(row["count"]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def sed_rows_by_group(path: Path, grouping: str) -> list[dict[str, str]]:
    return [row for row in load_csv(path) if row.get("grouping") == grouping]


def fit_metric(meta: dict[str, Any], fit_key: str, metric: str) -> Any:
    return ((meta.get("fits") or {}).get(fit_key) or {}).get(metric)


def parse_pipeline_jobs() -> list[tuple[str, str]]:
    raw = os.environ.get("PIPELINE_JOB_IDS", "").strip()
    pairs: list[tuple[str, str]] = []
    for item in re.split(r"[;,]", raw):
        item = item.strip()
        if not item or ":" not in item:
            continue
        label, job_id = item.split(":", 1)
        if job_id and job_id != "PENDING":
            pairs.append((label.strip(), job_id.strip()))
    if not pairs and os.environ.get("SLURM_JOB_ID"):
        pairs.append(("current_report_job", str(os.environ["SLURM_JOB_ID"])))
    return pairs


def sacct_rows(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, job_id in pairs:
        record = {"stage": label, "job_id": job_id, "state": "unknown", "elapsed": "unknown", "exit_code": "unknown", "job_name": ""}
        try:
            result = subprocess.run(
                ["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,Elapsed,ExitCode,JobName"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            rows.append(record)
            continue
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 5 or parts[0] != job_id:
                continue
            record.update({"state": parts[1], "elapsed": parts[2], "exit_code": parts[3], "job_name": parts[4]})
            break
        rows.append(record)
    return rows


def collect_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.extend(collect_strings(item, child))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(collect_strings(item, f"{prefix}[{idx}]"))
        return out
    if isinstance(value, str):
        return [(prefix, value)]
    return []


LEGACY_INPUT_RE = re.compile(
    r"(64670|8666|WCDA_simulation_binned_response_v[1-5]|stage_[a-g]_v[1-5]|/v[1-5][_/.-]|_v[1-5]_)",
    re.IGNORECASE,
)


def contamination_audit(metadata_files: list[Path]) -> dict[str, Any]:
    offenders: list[dict[str, str]] = []
    for path in metadata_files:
        payload = load_json(path)
        for key, value in collect_strings(payload):
            if LEGACY_INPUT_RE.search(value):
                offenders.append({"metadata": str(path), "field": key, "value": value})
    return {
        "status": "passed" if not offenders else "failed",
        "metadata_files": [str(path) for path in metadata_files],
        "offenders": offenders,
    }


class ImageRefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr_map = {key.lower(): value for key, value in attrs}
        src = attr_map.get("src")
        if src:
            self.images.append(src)


def validate_html_images(path: Path) -> dict[str, Any]:
    parser = ImageRefParser()
    text = path.read_text(encoding="utf-8")
    parser.feed(text)
    missing: list[str] = []
    for src in parser.images:
        if re.match(r"^[a-z]+://", src):
            continue
        image_path = (path.parent / src).resolve()
        if not image_path.exists():
            missing.append(src)
    return {"image_refs": parser.images, "missing_image_refs": missing, "image_ref_count": len(parser.images)}


def status_class(value: str) -> str:
    normalized = value.lower()
    if normalized in {"pass", "passed", "completed", "completed+"}:
        return "pass"
    if normalized in {"warning", "running", "unknown"}:
        return "warn"
    return "fail"


def main() -> None:
    stage_a_meta = load_json(STAGE_A / f"response_2d_{RUN_ID}_metadata.json")
    stage_a_ap_meta = load_json(STAGE_A_AP / f"response_2d_{RUN_ID}_aperture_conditioned_metadata.json")
    stage_b_meta = load_json(STAGE_B / f"psf_{RUN_ID}_metadata.json")
    stage_c_meta = load_json(STAGE_C / "obs_events_metadata.json")
    stage_d_meta = load_json(STAGE_D / f"background_{RUN_ID}_annnorm_metadata.json")
    stage_e_meta = load_json(STAGE_E / f"signal_{RUN_ID}_containment1_annnorm_metadata.json")
    stage_f_meta = load_json(STAGE_F / f"fit_{RUN_ID}_metadata.json")
    stage_g_meta = load_json(STAGE_G / f"sed_points_{RUN_ID}_metadata.json")
    stage_g_summary = load_json(STAGE_G / f"sed_points_{RUN_ID}_summary.json")
    selector_meta = load_json(SELECTOR_META)

    selector_rows = load_csv(FIT_SELECTOR)
    prefit_rows = load_csv(PREFIT_SELECTOR)
    decision_rows = load_csv(HIGH_E_DECISIONS)
    source_rows = load_csv(STAGE_C / "source_files.csv")
    cutflow = global_cutflow_map(load_csv(STAGE_C / "obs_events_cutflow.csv"))
    fit_rows = [row for row in selector_rows if truthy(row.get("include"))]
    tail_rows = [row for row in selector_rows if row.get("predE_bin") == ">=6"]
    tail_included = [row for row in tail_rows if truthy(row.get("include"))]
    high_inc = [row for row in selector_rows if truthy(row.get("highEplus1_included_flag"))]
    high_rej = [row for row in selector_rows if truthy(row.get("highEplus1_rejected_flag"))]
    original_ridge = [row for row in selector_rows if truthy(row.get("original_ridge_fit_flag"))]

    processing = stage_c_meta.get("processing") or {}
    stage_c_files = int(processing.get("processed_file_count") or 0)
    missing_time = int(processing.get("missing_time_file_count") or 0)
    entry_mismatch = int(processing.get("entry_mismatch_file_count") or 0)
    selected_rows = int(processing.get("selected_rows") or 0)
    rough_live_days = float(((stage_c_meta.get("live_time_basis") or {}).get("rough_live_time_days_sum_files")) or 0.0)
    bad_time_rows = [
        row
        for row in source_rows
        if row.get("status") == "missing_time_skipped" or str(row.get("entry_mismatch")).strip().lower() == "true"
    ]

    e_totals = stage_e_meta.get("totals") or {}
    f_pref = stage_f_meta.get("preferred_fit") or {}
    f_quality = stage_f_meta.get("quality") or {}
    v6_logpar = (stage_f_meta.get("fits") or {}).get("logpar_conservative") or {}
    g_quality = stage_g_meta.get("quality") or {}
    g_frozen = stage_g_summary.get("frozen_spectrum") or {}
    g_csv = STAGE_G / f"sed_points_{RUN_ID}_summary.csv"
    g_nhit_rows = sed_rows_by_group(g_csv, "nhit")
    g_pred_rows = sed_rows_by_group(g_csv, "predE")
    job_rows = sacct_rows(parse_pipeline_jobs())

    metadata_files = [
        STAGE_A / f"response_2d_{RUN_ID}_metadata.json",
        STAGE_A_AP / f"response_2d_{RUN_ID}_aperture_conditioned_metadata.json",
        STAGE_B / f"psf_{RUN_ID}_metadata.json",
        STAGE_C / "obs_events_metadata.json",
        STAGE_D / f"background_{RUN_ID}_annnorm_metadata.json",
        STAGE_E / f"signal_{RUN_ID}_containment1_annnorm_metadata.json",
        STAGE_F / f"fit_{RUN_ID}_metadata.json",
        STAGE_G / f"sed_points_{RUN_ID}_metadata.json",
        SELECTOR_META,
    ]
    contamination = contamination_audit(metadata_files)

    validation_rows = [
        ("run id", "pass" if RUN_ID in str(stage_a_meta.get("npz_path")) else "warning", RUN_ID),
        ("Nhit binning", "pass" if selector_rows and selector_rows[0].get("nhit_bin") == "[100,200)" else "fail", selector_rows[0].get("nhit_bin") if selector_rows else "missing"),
        ("tail policy", "pass" if not tail_included else "fail", f"{len(tail_rows)} >=6 tail cells, {len(tail_included)} included"),
        ("selector", "pass", f"{len(original_ridge)} original ridge cells, {len(high_inc)} highEplus1 included, {len(high_rej)} rejected probes"),
        ("Stage C files", "pass" if stage_c_files > 3000 else "warning", f"{stage_c_files:,} processed, missing time {missing_time}, entry mismatch {entry_mismatch}"),
        ("Stage E signal", "pass" if (stage_e_meta.get("quality_gate") or {}).get("status") == "passed" else "warning", f"formal sigma {fmt(e_totals.get('formal_sigma'), 5)}"),
        ("Stage F fit", "pass" if f_quality.get("fit_status") == "passed" else "warning", f"preferred {f_pref.get('model')}"),
        ("Stage G points", "pass" if g_nhit_rows and g_pred_rows else "warning", f"{len(g_nhit_rows)} Nhit points, {len(g_pred_rows)} predE points"),
        ("metadata pollution", contamination["status"], f"{len(contamination['offenders'])} legacy main-input path/token offenders"),
    ]

    stage_rows = [
        ("Prepare/cache", "/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate"),
        ("Stage A response", f"apply/output/stage_a_{RUN_ID}"),
        ("Stage B PSF", f"apply/output/stage_b_{RUN_ID}"),
        ("Stage A aperture response", f"apply/output/stage_a_{RUN_ID}_aperture_conditioned"),
        ("Stage C observation", f"apply/output/stage_c_{RUN_ID}"),
        ("Stage D background", f"apply/output/stage_d_{RUN_ID}_annnorm"),
        ("Stage E signal", f"apply/output/stage_e_{RUN_ID}_containment1_annnorm"),
        ("Stage F fit", f"apply/output/stage_f_{RUN_ID}"),
        ("Stage G SED", f"apply/output/stage_g_{RUN_ID}"),
    ]

    expected_figures = [
        (STAGE_B / "psf_r_opt_deg_grid.png", "Stage B r_opt by cell"),
        (STAGE_B / "psf_effective_events_grid.png", "Stage B effective events by cell"),
        (STAGE_B / "psf_radial_profiles_grid.png", "Stage B radial PSF profiles"),
        (STAGE_D / "roi_excess_grid.png", "Stage D ROI excess map grid"),
        (STAGE_D / "annulus_residual_grid.png", "Stage D annulus residuals"),
        (STAGE_E / "formal_sigma_grid.png", "Stage E formal sigma grid"),
        (STAGE_E / "on_background_grid.png", "Stage E on/background grid"),
        (STAGE_F / "model_counts_vs_excess.png", "Stage F model counts versus excess"),
        (STAGE_F / "pull_grid_logpar.png", "Stage F LogPar pull grid"),
        (STAGE_G / "sed_points_stage_f_fullarray_pool1.png", "Stage G SED overlay"),
        (STAGE_G / "sed_points_ratio.png", "Stage G SED ratio plot"),
    ]
    figure_html = "".join(figure(path, caption) for path, caption in expected_figures)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crab SED v6 64748 nhit100 highEplus1 Stage A-G</title>
  <style>
    :root {{
      --ink:#17202a; --muted:#5e6875; --line:#d7dde5; --panel:#f6f8fb;
      --ok:#126b45; --warn:#93620c; --fail:#aa2e25; --accent:#005eb8;
    }}
    body {{ margin:0; color:var(--ink); background:#fff; font-family:Arial,Helvetica,sans-serif; line-height:1.48; }}
    main {{ max-width:1220px; margin:0 auto; padding:32px 24px 58px; }}
    header {{ border-bottom:3px solid var(--ink); padding-bottom:18px; margin-bottom:26px; }}
    h1,h2,h3 {{ margin:0; line-height:1.18; letter-spacing:0; }}
    h1 {{ font-size:32px; }}
    h2 {{ font-size:21px; margin-top:34px; padding-top:14px; border-top:1px solid var(--line); }}
    h3 {{ font-size:16px; margin-top:18px; color:#2f3b48; }}
    p {{ margin:10px 0; }}
    code {{ background:#eef2f6; padding:1px 4px; border-radius:3px; font-size:12px; }}
    table {{ border-collapse:collapse; width:100%; margin:14px 0 22px; font-size:13px; }}
    th,td {{ border:1px solid var(--line); padding:6px 7px; text-align:right; vertical-align:top; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ background:#edf1f6; font-weight:700; }}
    .lede {{ font-size:16px; color:#2f3b48; max-width:960px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:16px 0; }}
    .metric {{ border:1px solid var(--line); border-radius:6px; padding:12px; background:var(--panel); min-height:78px; }}
    .metric .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .metric .value {{ font-size:23px; font-weight:700; margin-top:4px; }}
    .metric .sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
    .status-pass {{ color:var(--ok); font-weight:700; }}
    .status-warn,.status-warning {{ color:var(--warn); font-weight:700; }}
    .status-fail,.status-failed {{ color:var(--fail); font-weight:700; }}
    .okbox {{ border-left:5px solid var(--ok); background:#edf9f1; padding:12px 14px; margin:18px 0; }}
    .callout {{ border-left:5px solid var(--warn); background:#fff8ec; padding:12px 14px; margin:18px 0; }}
    .figgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; margin:14px 0 22px; }}
    .figure {{ margin:0; border:1px solid var(--line); border-radius:6px; padding:10px; background:#fff; }}
    .figure img {{ display:block; width:100%; height:auto; }}
    .figure figcaption {{ margin-top:8px; color:var(--muted); font-size:12px; }}
    .pathlist td {{ text-align:left; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Crab SED v6 <code>64748</code> nhit100 highEplus1 Stage A-G</h1>
    <p class="lede">New v6 chain for <code>{RUN_ID}</code>. The first Nhit bin is <code>[100,200)</code>, predE split56 keeps <code>[5,5.5)</code> and <code>[5.5,6)</code>, and <code>&gt;=6</code> is retained only as a diagnostic tail outside Stage F/G.</p>
  </header>

  <section>
    <h2>Executive Check</h2>
    <div class="grid">
      <div class="metric"><div class="label">candidate cells</div><div class="value">{len(selector_rows)}</div><div class="sub">7 Nhit x 13 predE</div></div>
      <div class="metric"><div class="label">fit cells</div><div class="value">{len(fit_rows)}</div><div class="sub">{len(original_ridge)} ridge + {len(high_inc)} highEplus1</div></div>
      <div class="metric"><div class="label">Stage C files</div><div class="value">{stage_c_files:,}</div><div class="sub">missing time {missing_time}; entry mismatch {entry_mismatch}</div></div>
      <div class="metric"><div class="label">selected rows</div><div class="value">{selected_rows:,}</div><div class="sub">rough live {rough_live_days:.3f} d</div></div>
      <div class="metric"><div class="label">Stage E signal</div><div class="value">{fmt(e_totals.get('formal_sigma'), 5)}</div><div class="sub">formal sigma</div></div>
      <div class="metric"><div class="label">Stage F preferred</div><div class="value">{esc(f_pref.get('model')).upper()}</div><div class="sub">chi2/ndof {fmt(v6_logpar.get('chi2'), 4)}/{esc(v6_logpar.get('ndof'))}</div></div>
      <div class="metric"><div class="label">Stage G points</div><div class="value">{len(g_nhit_rows)}/{len(g_pred_rows)}</div><div class="sub">Nhit / predE groupings</div></div>
      <div class="metric"><div class="label">metadata audit</div><div class="value">{esc(contamination['status']).upper()}</div><div class="sub">{len(contamination['offenders'])} offenders</div></div>
    </div>
    {table(["Gate", "Status", "Evidence"], [[esc(name), f'<span class="status-{status_class(status)}">{esc(status)}</span>', esc(evidence)] for name, status, evidence in validation_rows])}
  </section>

  <section>
    <h2>Slurm Jobs</h2>
    <p>All heavy recomputation stages are intended to run through the Slurm dependency chain. This table is populated from <code>PIPELINE_JOB_IDS</code> when available.</p>
    {table(["Stage", "Job", "State", "Elapsed", "Exit", "Name"], [[esc(row["stage"]), esc(row["job_id"]), f'<span class="status-{status_class(row["state"])}">{esc(row["state"])}</span>', esc(row["elapsed"]), esc(row["exit_code"]), esc(row["job_name"])] for row in job_rows]) if job_rows else '<div class="callout">No Slurm job id manifest was provided to the report builder.</div>'}
  </section>

  <section>
    <h2>Inputs And Outputs</h2>
    <p>The main chain uses the 64748 observation eval root and its recovered-time tree. The response, PSF, aperture response, fit, and SED products are all under the new run namespace.</p>
    {table(["Field", "Value"], [
        ["Run id", f"<code>{RUN_ID}</code>"],
        ["Observation root", f"<code>{esc(stage_c_meta.get('obs_root'))}</code>"],
        ["Recovered time root", f"<code>{esc(stage_c_meta.get('time_root'))}</code>"],
        ["MC candidate cache", "<code>/mnt/mydisk/WCDA_simulation_binned_response_v6_64748_nhit100_highEplus1_split56_candidate</code>"],
        ["Model run dir", f"<code>{esc(stage_a_meta.get('run_dir'))}</code>"],
        ["Fit selector", f"<code>{rel(FIT_SELECTOR)}</code>"],
        ["Stage C input entries", f"{cutflow.get('input_entries', 0):,}"],
        ["After configured-cell selection", f"{cutflow.get('after_cell_selection', 0):,}"],
    ], "pathlist")}
    {table(["Stage", "Primary output"], [[esc(name), f"<code>{esc(path)}</code>"] for name, path in stage_rows], "pathlist")}
  </section>

  <section>
    <h2>Selector Rule</h2>
    <p>The final selector preserves original MC-ridge fit cells, then considers exactly one adjacent higher predE bin per Nhit band. New high-side cells enter Stage F/G only when both MC statistics and Stage B PSF quality gates pass. No lower-energy expansion is applied.</p>
    {table(["Nhit", "Status", "Candidate", "MC count", "PSF gate", "Reasons"], [[f"<code>{esc(row.get('nhit_bin'))}</code>", esc(row.get("status")), f"<code>{esc(row.get('candidate_predE_bin'))}</code> cell {esc(row.get('candidate_cell_id'))}", fmt_int(row.get("mc_count")), esc(row.get("psf_quality_flag")), esc(row.get("psf_quality_reasons"))] for row in decision_rows])}
    <h3>Included highEplus1 cells</h3>
    {table(["Cell", "Nhit", "predE", "MC count", "PSF reason"], [[esc(row.get("cell_id")), f"<code>{esc(row.get('nhit_bin'))}</code>", f"<code>{esc(row.get('predE_bin'))}</code>", fmt_int(row.get("mc_count")), esc(row.get("psf_quality_reasons"))] for row in high_inc]) if high_inc else '<div class="callout">No highEplus1 probe passed the Stage B quality gate.</div>'}
    <h3>Rejected highEplus1 probes</h3>
    {table(["Cell", "Nhit", "predE", "MC count", "Reason"], [[esc(row.get("cell_id")), f"<code>{esc(row.get('nhit_bin'))}</code>", f"<code>{esc(row.get('predE_bin'))}</code>", fmt_int(row.get("mc_count")), esc(row.get("exclusion_source"))] for row in high_rej]) if high_rej else '<div class="okbox">No highEplus1 candidate was rejected by the gate.</div>'}
  </section>

  <section>
    <h2>Stage C Time Audit</h2>
    <p>Stage C used <code>{esc(stage_c_meta.get('time_root'))}</code>. Missing-time and entry-mismatch rows are listed below when present.</p>
    {table(["Metric", "Value"], [
        ["Processed files", f"{stage_c_files:,}"],
        ["Missing time files", f"{missing_time:,}"],
        ["Entry mismatch files", f"{entry_mismatch:,}"],
        ["Selected rows", f"{selected_rows:,}"],
        ["Matched MJD min", fmt((stage_c_meta.get("mjd_coverage") or {}).get("matched_mjd_min"), 8)],
        ["Matched MJD max", fmt((stage_c_meta.get("mjd_coverage") or {}).get("matched_mjd_max"), 8)],
    ])}
    {table(["source_file_id", "status", "relative_path", "event_entries", "time_entries", "selected_rows"], [[esc(row.get("source_file_id")), esc(row.get("status")), esc(row.get("relative_path")), esc(row.get("event_entries")), esc(row.get("time_entries")), esc(row.get("selected_rows"))] for row in bad_time_rows[:50]]) if bad_time_rows else '<div class="okbox">No missing recovered-time files or entry mismatches were recorded.</div>'}
  </section>

  <section>
    <h2>Stage A-B-D-E Diagnostics</h2>
    <p>Stage A nominal response is <code>{esc(stage_a_meta.get('response_type'))}</code>; Stage F/G use <code>{esc(stage_a_ap_meta.get('response_type'))}</code>. Stage B wrote {esc(stage_b_meta.get('n_cells'))} PSF rows.</p>
    <div class="figgrid">{figure_html}</div>
  </section>

  <section>
    <h2>Stage F Fit</h2>
    <p>Stage F uses the new highEplus1 selector and the aperture-conditioned 64748 response. The preferred model recorded by Stage F is <code>{esc(f_pref.get('model'))}</code>.</p>
    {table(["Fit", "Valid", "chi2/ndof", "p", "phi0", "gamma/alpha", "beta"], [
        ["PL conservative", esc(fit_metric(stage_f_meta, "pl_conservative", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_conservative', 'chi2'), 4)}/{esc(fit_metric(stage_f_meta, 'pl_conservative', 'ndof'))}", fmt(fit_metric(stage_f_meta, "pl_conservative", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_conservative", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_conservative", "gamma"), 4), "n/a"],
        ["LogPar conservative", esc(fit_metric(stage_f_meta, "logpar_conservative", "valid")), f"{fmt(v6_logpar.get('chi2'), 4)}/{esc(v6_logpar.get('ndof'))}", fmt(v6_logpar.get("p_value"), 3), fmt(v6_logpar.get("phi0"), 4), fmt(v6_logpar.get("alpha"), 4), fmt(v6_logpar.get("beta"), 4)],
        ["PL sqrt-N", esc(fit_metric(stage_f_meta, "pl_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'pl_sqrt_n', 'chi2'), 4)}/{esc(fit_metric(stage_f_meta, 'pl_sqrt_n', 'ndof'))}", fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "pl_sqrt_n", "gamma"), 4), "n/a"],
        ["LogPar sqrt-N", esc(fit_metric(stage_f_meta, "logpar_sqrt_n", "valid")), f"{fmt(fit_metric(stage_f_meta, 'logpar_sqrt_n', 'chi2'), 4)}/{esc(fit_metric(stage_f_meta, 'logpar_sqrt_n', 'ndof'))}", fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "p_value"), 3), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "phi0"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "alpha"), 4), fmt(fit_metric(stage_f_meta, "logpar_sqrt_n", "beta"), 4)],
    ])}
  </section>

  <section>
    <h2>Stage G SED</h2>
    <p>Stage G freezes the preferred Stage F spectrum with phi0={fmt(g_frozen.get('phi0'), 4)}, alpha={fmt(g_frozen.get('alpha'), 4)}, beta={fmt(g_frozen.get('beta'), 4)} at {fmt(g_frozen.get('pivot_tev'), 3)} TeV. Quality status: <code>{esc(g_quality.get('status'))}</code>.</p>
    {table(["Nhit group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{esc(row['ndof'])}"] for row in g_nhit_rows])}
    {table(["PredE group", "Cells", "E_eff TeV", "E2 dN/dE", "Err", "chi2/ndof"], [[f"<code>{esc(row['group_label'])}</code>", esc(row["cell_ids"]), fmt(row["effective_energy_tev"], 4), fmt(row["E2_dnde"], 4), fmt(row["E2_dnde_err"], 3), f"{fmt(row['chi2'], 4)}/{esc(row['ndof'])}"] for row in g_pred_rows])}
  </section>

  <section>
    <h2>Metadata Audit</h2>
    <p>Main-input metadata files were scanned for legacy cache/model path tokens. Status: <strong>{esc(contamination['status'])}</strong>.</p>
    {table(["Metadata", "Field", "Value"], [[f"<code>{esc(row['metadata'])}</code>", esc(row["field"]), f"<code>{esc(row['value'])}</code>"] for row in contamination["offenders"]], "pathlist") if contamination["offenders"] else '<div class="okbox">No legacy main-input metadata path offenders were found.</div>'}
  </section>
</main>
</body>
</html>
"""

    REPORT_PATH.write_text(html_doc, encoding="utf-8")
    html_validation = validate_html_images(REPORT_PATH)
    validation_payload = {
        "report_path": str(REPORT_PATH),
        "html_image_validation": html_validation,
        "metadata_contamination": contamination,
        "selector_summary": selector_meta,
    }
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8")
    if html_validation["missing_image_refs"]:
        raise SystemExit(f"Report has missing image references: {html_validation['missing_image_refs']}")
    if contamination["offenders"]:
        raise SystemExit("Metadata contamination audit failed; see report and validation JSON")

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {VALIDATION_JSON}")
    print(f"Image refs: {html_validation['image_ref_count']}; missing: {len(html_validation['missing_image_refs'])}")
    print(f"Fit cells: {len(fit_rows)}; highEplus1 included/rejected: {len(high_inc)}/{len(high_rej)}")


if __name__ == "__main__":
    main()
