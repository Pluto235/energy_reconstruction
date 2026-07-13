#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "apply/report"
SOURCE = REPORT_DIR / "crab_sed_v6_64748_nhit100_reselect44_split56_miss030_stage_a_to_g_report.html"
TARGET = REPORT_DIR / "crab_sed_v6_64748_nhit100_reselect44_scheme_B_report.html"
OLD_TITLE = "Crab SED v6 64748 nhit100 reselect44 miss030 Stage A-G"
NEW_TITLE = "Crab SED v6 64748 reselect44 - Scheme B - MC Aperture-conditioned Response"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if SOURCE.exists():
        before = sha256(SOURCE)
        SOURCE.replace(TARGET)
    elif TARGET.exists():
        before = sha256(TARGET)
    else:
        raise FileNotFoundError(f"Missing B report source: {SOURCE}")

    text = TARGET.read_text(encoding="utf-8")
    text = text.replace(OLD_TITLE, NEW_TITLE)
    marker = "</header>"
    contract = (
        '<div class="okbox"><strong>Scheme B response contract:</strong> Uses the MC '
        'aperture-conditioned response; downstream containment is exactly 1 and no additional '
        '0.715 factor is applied.<br><strong>Response:</strong> '
        '<code>apply/output/stage_a_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned/'
        'response_2d_v6_64748_nhit100_reselect44_split56_miss030_aperture_conditioned.npz</code><br>'
        '<strong>Stage E/F/G:</strong> <code>stage_e_v6_64748_nhit100_reselect44_split56_miss030_'
        'containment1_annnorm</code>, <code>stage_f_v6_64748_nhit100_reselect44_split56_miss030</code>, '
        '<code>stage_g_v6_64748_nhit100_reselect44_split56_miss030</code><br>'
        f'<strong>Input commit:</strong> <code>{subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()}</code></div>'
    )
    if "Scheme B response contract:" not in text:
        text = text.replace(marker, marker + "\n" + contract, 1)
    TARGET.write_text(text, encoding="utf-8")
    print(f"B source sha256 before label-only rename: {before}")
    print(f"B report: {TARGET}")


if __name__ == "__main__":
    main()
