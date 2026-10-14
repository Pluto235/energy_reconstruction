#!/usr/bin/env python3
"""Build the actual January execution/acceptance report, including a failed scientific gate."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "apply/tools"))
from january_v6_inputs import TAG, BASE, load_json, sha256, check_stat
from january_v6_validate import check, run_dir, stage_paths, require, check_diagnostic


def table(headers, rows):
    clean = lambda v: str(v).replace("|", " / ").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--jobs", required=True, help="Comma-separated actual Slurm IDs")
    args = parser.parse_args()
    root = REPO / "apply/output" / TAG
    if (root / "final_acceptance.json").exists():
        raise FileExistsError("Existing final acceptance preserved")
    manifest = load_json(root / "analysis_manifest.json")
    gti = load_json(root / "gti/gti_audit.json")
    diagnostic = load_json(root / "background_diagnostic/diagnostic.json")
    compatibility_path = REPO / "apply/docs/JANUARY_V6_MC_COMPATIBILITY.json"
    compatibility = load_json(compatibility_path)
    require(compatibility["decision"]["blockers"] == [] and compatibility["decision"]["nominal_response"] == "reuse_compatibility_passed" and compatibility["decision"]["scheme_R_response"] == "reuse_compatibility_passed", "MC compatibility decisions failed")
    c = load_json(run_dir(root, "c") / "obs_events_metadata.json")
    check("c", root, REPO)
    psf_receipt = check("psf", root, REPO)
    for row in manifest["files"]:
        check_stat(Path(manifest["prediction_root"]) / row["relative_path"], row["prediction_binding"]["stat"])
    require(gti["result"] == "PASS" and gti["scope"] == "month" and gti["file_count"] == 681, "GTI scope failed")
    require(gti["status"] == [69278905, 464, 0, 0, 0, 0, 0], "Original anomaly audit failed")
    require(sha256(root / "analysis_manifest.json") == gti["manifest"]["sha256"], "GTI source manifest changed")
    require(sha256(root / "gti/gti_source_files.csv") == gti["gti_exposure_sha256"], "GTI intervals changed")
    require(diagnostic["diagnostic_only"] and diagnostic["formal_stages"]["D"] == "FAILED" and diagnostic["fit_and_sed"] is None, "Invalid diagnostic status")
    require([row["cell_id"] for row in diagnostic["cells"]] == manifest["selected_cell_ids"], "Diagnostic44 order")
    for name, expected in diagnostic["hashes"].items():
        require(sha256(name) == expected, f"Diagnostic artifact changed: {name}")
    for artifact in compatibility["artifacts"].values():
        if "path" in artifact and "sha256" in artifact:
            require(sha256(artifact["path"]) == artifact["sha256"], f"MC compatibility artifact changed: {artifact['path']}")
    d = np.load(root / "background_diagnostic/background_signal_diagnostic.npz")
    check_diagnostic(diagnostic, d, manifest["selected_cell_ids"], c["processing"]["selected_rows"])
    ids = d["cell_id"].astype(int)
    selected_indices = np.flatnonzero(np.isin(ids, manifest["selected_cell_ids"]))
    failed_ids = [row["cell_id"] for row in diagnostic["failures"]]
    require(failed_ids == [62], "Unexpected frozen failure set; report must be reviewed")
    i62 = ids.tolist().index(62)
    require(d["fit_rank"][i62] == 6 and np.isfinite(d["fit_condition_number"][i62]), "Numerical background fit failed")
    require(c["processing"]["selected_rows"] == 21559258 and gti["merged_interval_count"] == 756 and d["annulus_counts"][i62] == 10 and d["annulus_pixels"][i62] == 5648, "Observed report constants changed")
    f = importlib.import_module("apply.stages.06_fit")
    response_path = REPO / f"apply/output/stage_e_{BASE}_fixed712979_rayleigh_annnorm/response_{BASE}_fixed712979_rayleigh.npz"
    with np.load(response_path) as response:
        theta_edges = response["theta_true_edges_deg"].copy()
    exposure, exposure_metadata = f.compute_theta_exposure(root / "gti/gti_source_files.csv", theta_edges,
        source_ra_deg=83.63, source_dec_deg=22.01, latitude_deg=29.45, longitude_east_deg=100.14, sample_step_sec=60.)
    require(np.isclose(exposure_metadata["total_live_seconds"], gti["merged_live_seconds"], rtol=0, atol=1e-5), "Exposure/GTI live time mismatch")
    require(np.isfinite(exposure).all() and 0 < exposure.sum() <= gti["merged_live_seconds"], "Invalid zenith exposure")
    exposure_metadata["method"] = "Existing Stage F source-theta midpoint integration of audited original-time GTI intervals at 60 s; no per-file density weights"
    exposure_metadata["scope"] = "exposure-only diagnostic; no spectral fit"
    (root / "theta_exposure.json").write_text(json.dumps(exposure_metadata, indent=2) + "\n")
    np.savez_compressed(root / "theta_exposure.npz", theta_edges_deg=theta_edges, theta_exposure_sec=exposure)
    output = subprocess.check_output(["sacct", "-X", "-j", args.jobs, "--parsable2", "--noheader", "-o", "JobID,JobName%40,State,ExitCode,Elapsed,AllocCPUS,ReqMem,Timelimit,NodeList"], text=True)
    jobs = [dict(zip(["id", "name", "state", "exit_code", "elapsed", "cpus", "memory", "limit", "node"], line.split("|"))) for line in output.splitlines() if line]
    by_job = {row["id"]: row for row in jobs}
    for job_id in ("66081", "66082", "66084", "66086", "66087"):
        require(by_job[job_id]["state"] == "COMPLETED" and by_job[job_id]["exit_code"] == "0:0", f"Slurm prerequisite failed: {job_id}")
    require(by_job["66085"]["state"] == "FAILED" and by_job["66085"]["exit_code"] != "0:0", "Missing actual production D failure")
    (root / "slurm_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    import shutil
    for job in jobs:
        for suffix in ("out", "err"):
            source = REPO / "logs/slurm" / f"{job['name']}_{job['id']}.{suffix}"
            if source.exists():
                shutil.copy2(source, logs / source.name)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    ax.stairs(exposure / 86400., theta_edges, fill=True, color="#446c8f")
    ax.set(xlabel="Crab zenith angle (deg)", ylabel="Exposure (days)", title="202201 · audited GTI source exposure")
    fig.savefig(root / "theta_exposure.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), constrained_layout=True)
    axes[0].bar(np.arange(44), d["annulus_counts"][selected_indices], color="#446c8f")
    axes[0].set(yscale="log", ylabel="Annulus events", title="Frozen 44 cells · January background diagnostics")
    axes[1].plot(np.arange(44), d["on_aperture_surface_min"][selected_indices], "o", color="#446c8f", markersize=4)
    axes[1].axhline(0, color="#b34545", lw=1)
    axes[1].set(ylabel="Raw polynomial minimum\n(counts / pixel)", xlabel="Frozen cell ID")
    axes[1].set_xticks(np.arange(44), ids[selected_indices], rotation=90, fontsize=7)
    fig.savefig(root / "cell44_background.png", dpi=150)
    plt.close(fig)
    coefficients = d["surface_coefficients"][i62]
    x = np.linspace(-float(d["r_opt_deg"][i62]), float(d["r_opt_deg"][i62]), 300)
    raw = d["annulus_surface_scale"][i62] * (coefficients[0] + coefficients[1] * x + coefficients[3] * x*x)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    edges = d["x_edges_deg"]
    image = axes[0].imshow(d["counts_map"][i62], origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]], cmap="Blues")
    for radius in (float(d["r_opt_deg"][i62]), float(d["annulus_inner_deg"][i62]), float(d["annulus_outer_deg"][i62])):
        axes[0].add_patch(plt.Circle((0, 0), radius, fill=False, lw=1, color="#b34545"))
    axes[0].set(xlim=(-6, 6), ylim=(-6, 6), xlabel="RA tangent offset (deg)", ylabel="Dec offset (deg)", title="Cell 62 · observed counts")
    fig.colorbar(image, ax=axes[0], label="events / pixel")
    axes[1].plot(x, raw, color="#b34545")
    axes[1].axhline(0, color="#888", lw=1)
    axes[1].set(xlabel="x inside fixed aperture (deg)", ylabel="Raw fitted density (counts / pixel)", title="Cell 62 · y=0 quadratic section")
    fig.savefig(root / "cell62_failure.png", dpi=150)
    plt.close(fig)
    negative = [r["cell_id"] for r in diagnostic["cells"] if r["negative_excess"]]
    low = [r["cell_id"] for r in diagnostic["cells"] if r["low_on_lt10"]]
    cell62 = next(row for row in diagnostic["cells"] if row["cell_id"] == 62)
    cell_rows = [[r["cell_id"], r["nhit_bin"], r["predE_bin"], r["N_on"], r["annulus_counts"], f'{r["raw_analytic_B_on"]:.6g}' if r["raw_analytic_B_on"] is not None else "undefined", f'{r["surface_min"]:.6g}' if r["surface_min"] is not None else "undefined", "valid" if r["background_valid"] else "INVALID", f'{r["excess"]:.6g}' if r["excess"] is not None else "undefined"] for r in diagnostic["cells"]]
    job_rows = [[j["id"], j["name"], j["state"], j["exit_code"], j["elapsed"], j["cpus"], j["memory"]] for j in jobs]
    report = f'''---
title: "202201 Crab · v6 64748 冻结方案复现与失败验收"
subtitle: "双 Rayleigh PSF · Scheme R · analytic B_on · 独立一月"
date: "2026-10-15"
lang: "zh-CN"
toc: true
---

> [!WARNING]
> **最终科学验收未通过。** 输入、时间对齐、全月 Stage C、MC 响应和 PSF 核验完成。固定 cell 62 的原始二次背景在圆孔径内非正，Stage D 正式失败；E–G 受门控阻断，没有可交付的正式44-cell信号、谱拟合或SED。没有删cell、裁剪 analytic B_on、改背景或临时调参。

## 验收结论

{table(["维度", "结论", "证据"], [["数据完整性", "PASS", "681/681；48,726,074预测；对齐输入来源一致"], ["时间/曝光", "PASS（事件间隙定义）", "原始69,279,369条时间；保留464异常；756段GTI"], ["响应/PSF契约", "PASS", "primary分母保留；覆盖率乘一次；全部91半径与历史位级一致"], ["数值求解", "PASS for cell62 linear solve", "rank6；condition65.463；不代表物理有效"], ["科学背景有效性", "FAIL", "cell62 raw B_on=-0.878926；孔径最小值=-0.0014731463"], ["谱拟合/SED", "BLOCKED", "D未通过；没有拟合收敛或科学拟合优度结果"]])}

本次 run ID：`{TAG}`。基线为指定旧 v6 analytic B_on 报告；基线半年结果仅用于参数与孔径来源核验，没有移植其观测计数、显著性、χ²或谱参数。未开展 Pass5 共同GTI比较，未使用v7。

## 输入与事件流

{table(["步骤", "事件数"], [[k, f"{v:,}"] for k,v in c["cutflow"].items()])}

预测已通过旧cuts，故本轮重验各cut不再减少。固定ledger使用nv（Nhit）且Nhit≥100、predE≥2：{c['assignment_audit']['out_of_ledger_after_finite']:,}条落在ledger外，其中Nhit<100为{c['assignment_audit']['nhit_below_candidate_min_after_quality_cuts']:,}，predE<2为{c['assignment_audit']['predE_below_candidate_min_after_quality_cuts']:,}；后者包含于前者，不能相加扣除。全月parquet为21,559,258条，91个诊断cell；固定拟合集合仍是44个，完整顺序保留。

Stage C只读取已验收预测及time_aligned。未使用原始全量friend按行配预测，未允许缺失时间、条目错配或共同前缀截断。Source gate核对全部验收报告哈希、输入stat/UUID、friend哈希及逐文件收据；没有重跑对齐或能量推理。

## GTI 与源轨迹曝光

实际 UTC：`{gti['utc_start']}` 至 `{gti['utc_stop']}`；MJD `{gti['mjd_min']:.12f}` 至 `{gti['mjd_max']:.12f}`。202201为北京时间文件标签，不裁成UTC日历一月。

原始status0为69,278,905；非零status464条保持原始异常文件及审计详情；另146条合法IES fallback保留并计数。TAI−Unix=37秒，MJD/Unix最大差异{gti['max_mjd_unix_error_seconds']:.3g}秒（float64舍入范围）。逐文件排序，按>60秒断开，全局合并重叠/接触区间；保留原逐文件规则的正边界间隙。合并移除重复活时{gti['overlap_removed_seconds']:.6f}秒。

{table(["量", "本次值"], [["GTI段数", gti['merged_interval_count']], ["有效活时（秒）", f"{gti['merged_live_seconds']:.6f}"], ["有效活时（天）", f"{gti['total_live_days']:.8f}"], ["Crab进入MC天顶角范围的曝光（天）", f"{exposure_metadata['source_visible_live_days']:.8f}"], ["Stage C预测稀疏事件rough活时（天，仅对比诊断）", f"{c['live_time_basis']['rough_live_time_days_sum_files']:.8f}"], ["≤60秒文件边界正间隙（保持原逐文件口径）", f"{gti['remaining_positive_gaps_le60_count']}段 / {gti['remaining_positive_gaps_le60_seconds']:.6f}秒"]])}

源轨迹沿用旧v6：RA83.63°、Dec22.01°，台址纬29.45°、经100.14°，原GMST公式和60秒中点积分；响应天顶角bins不变。每个GTI按真实区间长度积分，无半年比例缩放、无31天当活时、无预测稀疏度重权。没有独立硬件死时间记录，故称“事件间隙定义有效活时”，不能声称硬件精确live time。原始文件header保存处理/运行来源但没有独立采集起止或硬件死时间量。

![一月GTI经旧v6 Crab轨迹积分的天顶角曝光](theta_exposure.png)

## MC 响应与 PSF

10,000个原始MC文件清单/尺寸/来源收据已核对；额外复核两个来源样本hash，未重新全量checksum。模型config与响应snapshot一致；原预处理及权重、cuts、nv/predE/真能量/天顶角bins、投放面积和单位保留。恢复NPZ中的primary denominator_sumw/count仍完整，独立分母文件路径失效没有触发替代分母。

复用nominal与Scheme R响应，`Aeff_R={manifest['contract']['target_containment']}×Aeff_nominal`且仅一次；原投放面积/几何因子重算与保存Aeff在float32舍入精度内吻合。没有使用aperture-conditioned响应再乘覆盖率。

双Rayleigh主产物缺失，因此复用原保存的径向profile及原2R算法补建；所有91个r_opt与历史aperture数组位级一致（maxdiff0）。没有MC预测或分箱重跑。固定44中56、61、70、75、86、87沿用基线PSF回退，保留而不删cell。详细参数、路径、hash、源码AST对照在[MC/PSF兼容性证据](mc_compatibility.json)。

## Stage D 失败与44-cell诊断

正式D使用Crab ROI局部背景、weighted-LS annulus-quadratic、二阶、annulus normalization、fiducial6°/edge8°/网格0.1°、默认ring内1.5°/宽2°/最大内4.5°；analytic-quadratic积分，44-cell要求正性。N_on诊断调用旧Stage E逐事件球面距离，孔径固定，未转为像素中心计数。

Cell62：nv `[800,1100)`，predE `[4.75,5.0)`，r_opt={cell62['r_opt_deg']:.12f}°，环区[{cell62['annulus_inner_deg']},{cell62['annulus_outer_deg']})°，实际拟合环区10条/5648像素。六参数无正性约束的二次拟合rank6、condition65.463、χ²/ndof={d['fit_chi2'][i62]:.6f}/{int(d['fit_ndof'][i62])}；原始积分−0.878926与非正最小值使原契约失败。稀疏计数下这个χ²不能当背景可靠性的证据。

这支持“低统计二次曲面向核心外推失效”的解释；核心拟合、积分、正性检查及本次网格与基线一致，未发现数据/实现错配。诊断记录NaN表示无合法B_on，保留raw积分与系数，未将零或像素背景替为正式值。仅有效cell展示诊断excess，未将它们合成43-cell总信号或拟合。负excess cell：{negative}；N_on<10 cell：{low}。没有正式谱pull；背景残差与曲面诊断已保存。

![固定44cell的环区统计和原始孔径最小值；零线以下不满足正性](cell44_background.png)

![cell62的稀疏计数及未裁剪二次曲面孔径截面](cell62_failure.png)

{table(["cell", "Nhit", "predE", "N_on", "N_ann", "raw analytic B_on", "surface min", "validity", "excess diagnostic"], cell_rows)}

## Slurm 与产物

{table(["job", "阶段", "状态", "退出码", "耗时", "CPU", "内存申请"], job_rows)}

66083是相对路径解析失败，未生成PSF；修正绝对路径后66084通过。66085是真实科学门槛失败，未因文件存在忽略退出码。66087完成全部cell诊断，成功退出只说明诊断已完成，不表示D通过。各阶段耗时另存metadata；Slurm MaxRSS采样对这些短作业返回0，不能当作实际内存峰值。所有作业CPU-only；无GPU申请、无集群配置修改或无关任务取消。

本次产物位于`{root}`，各Slurm日志复制至`logs/`。主清单[analysis_manifest.json](analysis_manifest.json)、完成清单[analysis_manifest_completed.json](analysis_manifest_completed.json)、[最终验收收据](final_acceptance.json)、[GTI审计](gti/gti_audit.json)、[GTI区间](gti/gti_source_files.csv)、[曝光JSON](theta_exposure.json)、[44-cell诊断CSV](background_diagnostic/cell44_diagnostic.csv)、[背景/信号诊断JSON](background_diagnostic/diagnostic.json)、[Slurm账本](slurm_jobs.json)。大数据和生成图/报告不进入Git；可版本化执行结论在apply/docs。

## 定义与解释限制

固定44-cell方案的正式一月能谱复现**未完成/未通过**，而数据与依赖已就绪。继续取得有效谱需要新的科学方案授权，当前结果不能作为正式Crab谱，也不能据此评估PL/LogPar收敛、拟合优度或SED。此报告记录冻结方案实际运行的边界，不用旧半年谱参数、显著性或Pass5计数填补缺失结果。当前/latest指向、旧半年观测与报告、模型、已验收预测和时间friend均未覆盖。
'''
    (root / "report.md").write_text(report)
    shutil.copy2(compatibility_path, root / "mc_compatibility.json")
    code_paths = sorted((REPO / "apply/tools").glob("january_v6_*.py")) + [Path(__file__).resolve()] + sorted((REPO / "scripts/slurm").glob("january_v6_*.sbatch"))
    completed = {**manifest, "completion_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "runtime_source_hashes": {str(p): sha256(p) for p in code_paths}, "mc_psf_compatibility_sha256": sha256(compatibility_path), "stages": diagnostic["formal_stages"], "stage_c_selected_rows": c["processing"]["selected_rows"], "gti_audit_sha256": sha256(root / "gti/gti_audit.json"), "zenith_exposure": exposure_metadata, "slurm_jobs": jobs, "science_result": diagnostic["status"]}
    (root / "analysis_manifest_completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    subprocess.run([sys.executable, str(args.renderer), str(root / "report.md"), "--output", str(root / "report_202201.html")], check=True)
    hashes = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file() and p.suffix not in (".parquet",)}
    final = {"result": "SCIENTIFIC_BACKGROUND_CONTRACT_FAILED", "data_integrity": "PASS", "input_compatibility": "PASS", "gti_exposure": "PASS_GAP_DEFINED", "numerical_linear_solve_cell62": "PASS", "scientific_background_validity": "FAIL", "spectral_fit_convergence": "NOT_RUN_D_FAILED", "scientific_fit_goodness": "NOT_AVAILABLE", "sed": "NOT_RUN_D_FAILED", "cell62": cell62, "invalid_required_cells": failed_ids, "negative_excess_cells": negative, "low_n_on_lt10_cells": low, "slurm_jobs": jobs, "receipt_job": os.environ.get("SLURM_JOB_ID"), "commit": completed["completion_commit"], "hashes": hashes, "report": str(root / "report_202201.html"), "limits": [gti["limitation"], diagnostic["limits"]]}
    (root / "final_acceptance.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({k: v for k,v in final.items() if k not in ("hashes", "slurm_jobs", "cell62")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
