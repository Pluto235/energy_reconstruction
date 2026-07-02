#!/usr/bin/env python
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUN_TAG = "v6_64670_split56_baselinev4"
SPLIT_CHILD_KEYS = {
    ("[2000,3000)", "[5,5.5)"),
    ("[2000,3000)", "[5.5,6)"),
}
FORBIDDEN = [
    "_8666",
    "theta_recoxy_position_embed_midenergy_8666",
    "WCDA_simulation_binned_response_v1",
    "stage_a_v3",
    "stage_b_v3",
    "stage_a_v4",
    "stage_b_v4",
    "stage_a_v5",
    "stage_b_v5",
]
ALLOWED_COMPARISON_KEYS = {"v4_comparison", "comparison_reference"}


class ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr = dict(attrs)
        src = attr.get("src")
        if src:
            self.images.append(src)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate split56 v6 _64670 Stage A-G outputs and final report.")
    parser.add_argument("--report-html", type=Path, default=Path("apply/report/crab_sed_v6_64670_split56_baselinev4_report.html"))
    parser.add_argument("--selector", type=Path, default=Path("apply/config/cell_selector_v6_split56_drop4_psfborrow.csv"))
    parser.add_argument("--metadata", type=Path, action="append", default=[])
    parser.add_argument("--expect-candidate-cells", type=int, default=91)
    parser.add_argument("--expect-fit-cells", type=int, default=27)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_forbidden(value: Any, *, context: tuple[str, ...] = ()) -> list[str]:
    hits: list[str] = []
    if any(key in ALLOWED_COMPARISON_KEYS for key in context):
        return hits
    if isinstance(value, dict):
        for key, item in value.items():
            hits.extend(find_forbidden(item, context=(*context, str(key))))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            hits.extend(find_forbidden(item, context=(*context, str(idx))))
    else:
        text = str(value)
        for needle in FORBIDDEN:
            if needle in text:
                hits.append(".".join(context) + f": {needle} in {text}")
    return hits


def default_metadata_paths() -> list[Path]:
    return [
        Path("apply/config/v6_64670_split56_strategy_manifest.json"),
        Path("apply/output/stage_a_v6_64670_split56/response_2d_v6_64670_split56_metadata.json"),
        Path("apply/output/stage_b_v6_64670_split56/runs/v6_64670_split56_stage_b_psf/psf_v6_64670_split56_metadata.json"),
        Path("apply/output/stage_a_v6_64670_split56_aperture_conditioned/response_2d_v6_64670_split56_aperture_conditioned_metadata.json"),
        Path("apply/output/stage_c_v6_64670_split56/runs/v6_64670_split56_stage_c_halfyear/obs_events_metadata.json"),
        Path("apply/output/stage_d_v6_64670_split56_annnorm/runs/v6_64670_split56_stage_d_annnorm/background_v6_64670_split56_annnorm_metadata.json"),
        Path("apply/output/stage_e_v6_64670_split56_containment1_annnorm/runs/v6_64670_split56_stage_e_containment1_annnorm/signal_v6_64670_split56_containment1_annnorm_metadata.json"),
        Path("apply/output/stage_f_v6_64670_split56_baselinev4/runs/v6_64670_split56_stage_f_baselinev4/fit_v6_64670_split56_baselinev4_metadata.json"),
        Path("apply/output/stage_g_v6_64670_split56_baselinev4/runs/v6_64670_split56_stage_g_baselinev4/sed_points_v6_64670_split56_baselinev4_metadata.json"),
    ]


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    report = resolve(args.report_html)
    if not report.exists():
        failures.append(f"missing final report: {report}")
    else:
        parser = ImageParser()
        try:
            parser.feed(report.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"HTML parse failed: {exc}")
        for src in parser.images:
            if "://" in src or src.startswith("data:"):
                continue
            img = (report.parent / src).resolve()
            if not img.exists():
                failures.append(f"missing image referenced by report: {src} -> {img}")

    selector = resolve(args.selector)
    if not selector.exists():
        failures.append(f"missing selector: {selector}")
    else:
        rows = read_csv(selector)
        fit = [row for row in rows if str(row.get("include")).strip() == "1"]
        fit_keys = {(row.get("nhit_bin"), row.get("predE_bin")) for row in fit}
        if len(rows) != int(args.expect_candidate_cells):
            failures.append(f"selector candidate rows={len(rows)}, expected {int(args.expect_candidate_cells)}")
        if len(fit) != int(args.expect_fit_cells):
            failures.append(f"selector fit rows={len(fit)}, expected {int(args.expect_fit_cells)}")
        missing_split_children = sorted(SPLIT_CHILD_KEYS - fit_keys)
        if missing_split_children:
            failures.append(f"selector does not include required split child fit bins: {missing_split_children}")
        if any(row.get("predE_bin") == ">=6" for row in fit):
            failures.append("selector includes >=6 tail bin")
        for row in fit:
            if (row.get("nhit_bin"), row.get("predE_bin")) == ("[2000,3000)", "[5.5,6)"):
                if str(row.get("split_child_gate_exception")).strip() != "1":
                    failures.append("right split child is included but selector does not mark split_child_gate_exception=1")
                if int(float(row.get("split_child_effective_min_mc_count") or 0)) > int(float(row.get("mc_count") or 0)):
                    failures.append("right split child mc_count is below its effective split-child threshold")

    metadata_paths = [resolve(path) for path in (args.metadata or default_metadata_paths())]
    for path in metadata_paths:
        if not path.exists():
            failures.append(f"missing metadata: {path}")
            continue
        payload = load_json(path)
        hits = find_forbidden(payload)
        if hits:
            failures.append(f"path pollution in {path}: " + "; ".join(hits[:12]))
        text = json.dumps(payload, sort_keys=True)
        if path.name != "v6_64670_split56_strategy_manifest.json" and "split56" not in text:
            failures.append(f"metadata does not mention split56: {path}")

    key_outputs = [
        "apply/config/cell_ledger_v6_split56_candidate.csv",
        "apply/config/cell_selector_v6_split56_drop4_psfborrow.csv",
        "apply/output/stage_a_v6_64670_split56/response_2d_v6_64670_split56.npz",
        "apply/output/stage_b_v6_64670_split56/runs/v6_64670_split56_stage_b_psf/psf_v6_64670_split56.npz",
        "apply/output/stage_a_v6_64670_split56_aperture_conditioned/response_2d_v6_64670_split56_aperture_conditioned.npz",
        "apply/output/stage_c_v6_64670_split56/runs/v6_64670_split56_stage_c_halfyear/obs_events_metadata.json",
        "apply/output/stage_d_v6_64670_split56_annnorm/runs/v6_64670_split56_stage_d_annnorm/background_v6_64670_split56_annnorm.npz",
        "apply/output/stage_e_v6_64670_split56_containment1_annnorm/runs/v6_64670_split56_stage_e_containment1_annnorm/signal_v6_64670_split56_containment1_annnorm.npz",
        "apply/output/stage_f_v6_64670_split56_baselinev4/runs/v6_64670_split56_stage_f_baselinev4/fit_v6_64670_split56_baselinev4.npz",
        "apply/output/stage_g_v6_64670_split56_baselinev4/runs/v6_64670_split56_stage_g_baselinev4/sed_points_v6_64670_split56_baselinev4.npz",
        "apply/report/crab_sed_v6_64670_split56_baselinev4_report.html",
    ]
    for rel in key_outputs:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"missing key output: {path}")

    if failures:
        raise SystemExit("split56 validation failed:\n- " + "\n- ".join(failures))
    print(
        json.dumps(
            {
                "status": "passed",
                "report_html": str(report),
                "metadata_checked": [str(path) for path in metadata_paths],
                "forbidden_patterns": FORBIDDEN,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
