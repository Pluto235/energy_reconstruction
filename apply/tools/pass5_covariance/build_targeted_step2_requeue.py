#!/usr/bin/env python3
import csv
import json
from pathlib import Path

RUN_DIR = Path(".")
with open(RUN_DIR / "strict_hour_selection/strict_hour_manifest.json") as f:
    hm = json.load(f)
jobs = {j["label"]: j for j in hm["jobs"]}

with open(RUN_DIR / "bad_labels.txt") as f:
    bad_labels = [line.strip() for line in f if line.strip()]

SOURCE_HOUR_ROOT = "/scratchfs/lhaaso/liushijie/pass5_crab_v6_sorted_gti_hours"
TARGET_HOUR_ROOT = "/eos/user/l/liushijie/pass5_crab_v6_sorted_gti/pass5_crab_v6_sorted_gti_hours"
TARGET_MAP_ROOT = "/eos/user/l/liushijie/pass5_crab_v6_sorted_gti/pass5_crab_v6_sorted_gti_map_chunks"
XROOTD_PREFIX = "root://eos01.ihep.ac.cn/"


def xrootd_uri(path):
    return XROOTD_PREFIX.rstrip("/") + "/" + path


def relocate_list(src_list_path, dst_list_path):
    lines_out = []
    with open(src_list_path) as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if raw.startswith(SOURCE_HOUR_ROOT):
                rel = raw[len(SOURCE_HOUR_ROOT):].lstrip("/")
                lines_out.append(xrootd_uri(TARGET_HOUR_ROOT + "/" + rel))
            else:
                lines_out.append(raw)
    Path(dst_list_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_list_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")


rows = []
missing_labels = []
for idx, label in enumerate(bad_labels):
    job = jobs.get(label)
    if job is None:
        missing_labels.append(label)
        continue
    src_list = job["list"]
    dst_list = RUN_DIR / "strict_recovery/step2_hour_lists_retry2" / (label + ".list")
    relocate_list(src_list, dst_list)
    output_bkg = xrootd_uri(TARGET_MAP_ROOT + "/" + label + "_BKG.root")
    output_j2000 = xrootd_uri(TARGET_MAP_ROOT + "/" + label + "_BKG_J2000.root")
    rows.append([idx, job["index"], label, str(dst_list.resolve()), output_bkg, output_j2000])

print("missing_labels:", missing_labels)
print("rows:", len(rows))

with open(RUN_DIR / "strict_recovery/step2_recovery_jobs.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["recovery_index", "original_index", "label", "list_path", "output_bkg", "output_j2000"])
    writer.writerows(rows)

print("wrote strict_recovery/step2_recovery_jobs.tsv")
