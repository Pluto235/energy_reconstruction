#!/usr/bin/env python3
"""Copy January 2022 days only after the IHEP ROOT audit has completed."""
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

STATE = Path("/mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january")
DESTINATION = Path("/mnt/newdisk/shijie/WCDA_obs")
REMOTE = "liushijie@lxlogin.ihep.ac.cn"
AUDITS = "/home/lhaaso/liushijie/WCDA_reconstruction/runs/obs_jan2022_restore_20261013/audit"
SOURCE = "/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013"


def verify(report, directory):
    day = report["day"]
    if day not in {f"202201{n:02d}" for n in range(1, 32)}:
        raise ValueError(f"Unexpected day: {day}")
    names = [record["name"] for record in report["files"]]
    if len(names) != report["file_count"] or len(set(names)) != len(names):
        raise ValueError("Duplicate filenames or inconsistent audit count")
    for record in report["files"]:
        name, size = record["name"], record["output"]["bytes"]
        if not re.fullmatch(rf"Esg{day}_\d{{2}}\.root", name) or size <= 0:
            raise ValueError(f"Invalid audited file: {name}")
        path = directory / name
        if path.stat().st_size != size:
            raise RuntimeError(f"Transferred size mismatch: {path}")
        with path.open("rb") as file:
            if file.read(4) != b"root":
                raise RuntimeError(f"Transferred ROOT signature mismatch: {path}")
    if sum(record["output"]["bytes"] for record in report["files"]) != report["bytes"]:
        raise ValueError("Inconsistent daily byte total")


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-S", str(STATE / "ihep.sock"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]
    command = "import pathlib,json;print(json.dumps([json.loads(p.read_text()) for p in pathlib.Path(" + repr(AUDITS) + ").glob('*.json')]))"
    completed = {}
    deadline = time.monotonic() + 36 * 3600
    while len(completed) < 31:
        if time.monotonic() > deadline:
            raise TimeoutError("January preparation did not complete within 36 hours")
        raw = subprocess.check_output(ssh + [REMOTE, "python3 -c " + shlex.quote(command)], text=True)
        reports = json.loads(raw)
        for report in reports:
            day = report["day"]
            if day in completed:
                continue
            if day not in {f"202201{n:02d}" for n in range(1, 32)}:
                raise ValueError(f"Unexpected audited day: {day}")
            target = DESTINATION / day[4:]
            subprocess.run([
                "rsync", "-rt", "--partial", "--partial-dir=.rsync-partial", "--timeout=1200", "--stats",
                "--include=Esg*.root", "--exclude=*", "-e", shlex.join(ssh),
                f"{REMOTE}:{SOURCE}/{day[4:]}/", str(target) + "/",
            ], check=True)
            verify(report, target)
            completed[day] = report
            (STATE / "progress.json").write_text(json.dumps({
                "completed_days": sorted(completed),
                "files": sum(r["file_count"] for r in completed.values()),
                "bytes": sum(r["bytes"] for r in completed.values()),
            }, indent=2) + "\n")
            print(f"SYNC_VERIFIED {day} files={report['file_count']} bytes={report['bytes']}", flush=True)
        if len(completed) < 31:
            time.sleep(60)
    if sum(r["file_count"] for r in completed.values()) != 681:
        raise RuntimeError("January inventory differs from the 681-file source snapshot")
    (STATE / "completed.json").write_text(json.dumps(list(completed.values()), indent=2) + "\n")
    print("JANUARY_SYNC_COMPLETE files=681", flush=True)


if __name__ == "__main__":
    main()
