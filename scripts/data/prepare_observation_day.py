#!/usr/bin/env python3
"""Reuse or rebuild one day of the established pincness/branch-reduced inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import ROOT

ROOT.gROOT.SetBatch(True)
MACRO = Path(__file__).with_name("filter_observed_pincness_loop_scoped.C")
KEEP = set(re.findall(r'"(\w+)"', MACRO.read_text().split("return {", 1)[1].split("};", 1)[0]))


def inspect(path, expected_entries=None, expected_headers=None):
    try:
        file = ROOT.TFile.Open(str(path), "READ")
    except OSError as exc:
        raise RuntimeError(f"Cannot open ROOT file: {path}") from exc
    if not file or file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered):
        if file:
            file.Close()
        raise RuntimeError(f"Invalid ROOT file: {path}")
    try:
        tree, header = file.Get("t_eventout"), file.Get("t_eventout_h")
        if not tree or not header:
            raise RuntimeError(f"Missing event/header tree: {path}")
        branches = {b.GetName() for b in tree.GetListOfBranches()}
        entries, headers = int(tree.GetEntries()), int(header.GetEntries())
        if expected_entries is not None:
            if branches != KEEP or entries != expected_entries or headers != expected_headers:
                raise RuntimeError(f"Branch/entry/header mismatch: {path}")
            if tree.GetEntries("!(pincness < 1.1)") != 0:
                raise RuntimeError(f"Event cut violated: {path}")
            if not ROOT.readable_observation_tree(tree) or not ROOT.readable_observation_tree(header):
                raise RuntimeError(f"Unreadable retained data: {path}")
        elif not KEEP.issubset(branches):
            raise RuntimeError(f"Missing source branches: {path}")
        return {
            "entries": entries,
            "selected_entries": int(tree.GetEntries("pincness < 1.1")),
            "header_entries": headers,
            "bytes": int(file.GetSize()),
        }
    finally:
        file.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    if args.day not in {f"202201{n:02d}" for n in range(1, 32)}:
        raise ValueError("This restoration is restricted to January 2022")
    staging = Path("/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013")
    if args.output_root.resolve() != staging or (args.output_root / args.day[4:]).is_symlink():
        raise ValueError("Receipt cleanup is restricted to this run's temporary staging tree")
    sources = args.manifest.read_text().splitlines()
    prefix = f"root://eos01.ihep.ac.cn//eos/lhaaso/ai/wcda/raw/2022/{args.day[4:]}/"
    if not sources or len(sources) != len(set(sources)) or any(
        not url.startswith(prefix) or not re.fullmatch(rf"Esg{args.day}_\d{{2}}\.root", url.rsplit("/", 1)[-1])
        for url in sources
    ):
        raise ValueError("Invalid or empty input manifest")
    output_dir = args.output_root / args.day[4:]
    output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    file_audits = args.audit_dir.parent / "audit_files"
    acks = args.audit_dir.parent / "acks"
    file_audits.mkdir(exist_ok=True)
    acks.mkdir(exist_ok=True)
    ROOT.gROOT.ProcessLine(f'.L {MACRO}')
    records = []
    for url in sources:
        name = url.rsplit("/", 1)[-1]
        output = output_dir / name
        if output.is_symlink() or output.with_suffix(".root.partial").is_symlink():
            raise ValueError(f"Unexpected staging symlink: {output}")
        file_report = file_audits / f"{name}.json"
        ack = acks / f"{name}.ack"
        if file_report.exists() and ack.exists():
            record = json.loads(file_report.read_text())["files"][0]
            if ack.read_text().strip() == record["output"]["sha256"]:
                records.append(record)
                output.unlink(missing_ok=True)
                continue
        source = inspect(url)
        archive = args.archive_root / args.day[4:] / name
        if not output.exists() and archive.is_file():
            output.hardlink_to(archive)
        reused = output.exists()
        try:
            if not reused:
                raise RuntimeError("No existing output to reuse")
            result = inspect(output, source["selected_entries"], source["header_entries"])
        except RuntimeError:
            if output.exists():
                output.rename(output.with_name(output.name + f".invalid-{time.time_ns()}"))
            temporary = output.with_suffix(".root.partial")
            rc = ROOT.filter_observed_pincness_loop_scoped(url, str(temporary), 1.1)
            if rc != 0:
                raise RuntimeError(f"Filtering failed ({rc}): {url}")
            result = inspect(temporary, source["selected_entries"], source["header_entries"])
            temporary.replace(output)
            reused = False
        digest = hashlib.sha256()
        with output.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
        record = {"name": name, "source": source, "output": result, "reused": reused}
        records.append(record)
        temporary_report = file_report.with_suffix(".json.partial")
        temporary_report.write_text(json.dumps({
            "day": args.day, "file_count": 1, "bytes": result["bytes"], "files": [record],
        }) + "\n")
        temporary_report.replace(file_report)
        print(f"VERIFIED {args.day} {name} entries={result['entries']} bytes={result['bytes']} reused={reused}", flush=True)
        deadline = time.monotonic() + 6 * 3600
        while not ack.exists() or ack.read_text().strip() != result["sha256"]:
            if time.monotonic() > deadline:
                raise TimeoutError(f"No verified ETO receipt within six hours: {name}")
            time.sleep(10)
        output.unlink()
        print(f"ETO_ACK_CLEANED {name}", flush=True)
    report = {
        "day": args.day, "cut": "pincness < 1.1", "kept_branches": sorted(KEEP),
        "file_count": len(records), "bytes": sum(r["output"]["bytes"] for r in records),
        "files": records,
    }
    temporary = args.audit_dir / f"{args.day}.json.partial"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.audit_dir / f"{args.day}.json")
    print(f"DATE_DONE {args.day} files={len(records)}", flush=True)


if __name__ == "__main__":
    main()
