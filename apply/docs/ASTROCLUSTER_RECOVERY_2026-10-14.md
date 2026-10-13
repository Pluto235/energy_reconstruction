# AstroCluster energy 恢复与同步 — 2026-10-14

ETO 主工作目录：`/mnt/newdisk/shijie/energy`。

恢复来源：`AstroCluster:/share/home/lsj3906/ETO_restored_20261013/complete/server/projects/`。

恢复副本：`/mnt/newdisk/shijie/energy/.recovery/2026-10-14/astrocluster/`。

## 结果与优先用途

已同步 9 个 energy 相关项目：**24,669 个普通文件，16,948,343,173 字节（16.95 GB），3,543 个软链接**。仅提取 `complete/`；不采用 `incomplete/` 的零填充缺损文件。排除历史 `.git` 指针、`__pycache__` 和 `.pyc`。

50 个完整的 `best_model.pt` 及恢复到的实验配置、诊断文件已通过 `rsync -a --ignore-existing` 补入 ETO `runs/`。已有同名文件保留，全部来源仍可在独立恢复副本中追溯。**49 个模型现在有配套配置**；这些是恢复完整性结论，不代表已经通过推理验证。

| 重点内容 | 恢复情况 | 可用于什么 |
|---|---|---|
| `theta_recoxy_position_embed_noemax_no_core_cut_64748` | 权重完整；原配置缺损，已从 ETO 历史响应元数据快照补回 | 优先恢复当前 v6 SED 模型依赖；部署前做单文件推理检查 |
| `theta_recoxy_position_embed_midenergy_no_core_cut_64670` | 权重和原始配置均完整 | v5/64670 比较实验的模型依赖 |
| 其他 48 个完整权重 | 47 个有原始配置；`proton_only_smoketest_10876` 缺配置 | 历史实验对照；缺配置者只保留权重 |
| `theta_recoxy_position_embed_midenergy_8666` | 权重和配置仍在缺损区；未接回该权重 | 旧默认观察推理流程仍缺模型，不用其他实验替代 |
| Crab audit、empirical run3/4/5 等副本 | 完整文件已逐项目归档，仍存在已知缺损依赖 | 查阅审核、交接、经验 CDF 和 Poisson 分支实现；按实验逐项核对依赖 |

`64748/config.json` 来自 **12 份彼此完全一致的历史 `run_config_snapshot`**，包括：

- `apply/output/model_compare_v5/64748/stage_a_aperture_conditioned/response_model_compare_v5_64748_aperture_conditioned_metadata.json`
- `apply/output/model_compare_v5/64748/stage_a_nominal/response_model_compare_v5_64748_nominal_metadata.json`

保留原始配置中的历史路径和所有训练参数，未猜测或重设参数。配置来源和 SHA-256 记录于 `.recovery/2026-10-14/activation.json`。

## 已同步的项目

代码列包含 Python、Shell、C/C++、头文件和 Notebook，不计 Slurm `.sbatch`；Slurm 文件也已同步。文档列统计 Markdown。

| 项目 | 普通文件 | 代码 | 文档 | 模型 | GB |
|---|---:|---:|---:|---:|---:|
| `energy-reconstruction-poisson` | 605 | 92 | 14 | 0 | 0.126 |
| `energy_reconstruction` | 16,490 | 331 | 191 | 50 | 15.943 |
| `energy_reconstruction-crab-v6-pass5-audit` | 1,311 | 217 | 71 | 0 | 0.129 |
| `energy_reconstruction_double_king_finalize` | 904 | 73 | 44 | 0 | 0.142 |
| `energy_reconstruction_empirical_run` | 675 | 65 | 38 | 0 | 0.128 |
| `energy_reconstruction_empirical_run2` | 895 | 148 | 26 | 0 | 0.108 |
| `energy_reconstruction_empirical_run3` | 1,287 | 202 | 64 | 0 | 0.124 |
| `energy_reconstruction_empirical_run4` | 1,248 | 201 | 53 | 0 | 0.126 |
| `energy_reconstruction_empirical_run5` | 1,254 | 209 | 57 | 0 | 0.123 |

## 与当前 ETO 的关系

同步前对主项目的 600 个代码／Markdown 条目逐项比较（包含 `.C` 宏，不含模型）：276 个相同、319 个在 ETO 缺失、5 个同名内容不同。新增内容包括 `apply/v7_nq03/` 的部分检查与推理工具、一月方向分析工具及测试、Slurm 入口和 HSC 参考／练习代码。

这些代码和其他实验副本保留在恢复目录，没有覆盖 ETO 的活动实现。319 个缺失条目不等于 319 个可直接运行的功能：其中包含历史源码快照、第三方代码和不完整实验的组件。完整缺损摘要见 `.recovery/2026-10-14/astro-incomplete-summary.jsonl`。

以下 5 个冲突保留 ETO 版本，Astro 原件在恢复副本中：

- `AGENTS.md`
- `apply/scripts/slurm/optimize_v6_empirical_cdf_asimov_iter0.sbatch`
- `apply/scripts/slurm/run_v6_empirical_cdf_asimov_final.sbatch`
- `apply/scripts/slurm/run_v6_empirical_cdf_asimov_iter0.sbatch`
- `apply/stages/06_fit.py`

Crab audit 的 10 个缺损代码条目包括 1 个报告生成器和 9 个 Notebook；该副本并非完整可运行项目。empirical run3/4/5 也有缺损代码／文档。完整文件之间可以供后续对照和补缺，本次没有自动混合不同分支。

恢复的 ROOT、Parquet、NPZ、历史输出和 HSC 练习数据保留在副本中；未替换已经核验的一月观测数据或 MC 输入，也未激活其旧磁盘绝对软链接。

## 校验与边界

ETO Slurm 作业 **66037**：`COMPLETED`、退出码 `0:0`、耗时 12 秒，从 `energy/apply` 启动，1 CPU / 2 GB。首个作业 66036 因 `/bin/sh` 不支持 `pipefail` 在读取文件前退出，改用 POSIX `set -eu` 后重提。

- 24,669 个文件的路径、类型、字节数和完整清单一致；3,543 个软链接目标原样保留。
- 2,801 个代码、Markdown 和模型条目的 SHA-256 与 Astro 来源一致。
- 1,042 个 Python 文件全部通过 AST 语法解析。
- 50 个模型全部通过 PyTorch ZIP 容器、CRC、张量存储成员检查；接回 `runs/` 后再次核对 SHA-256。
- 其他科学产物验证路径、类型、大小及 SSH/tar 传输成功；没有逐文件比较来源 SHA-256，也没有逐个验证 ROOT/Parquet 内容。

ETO 的系统 Python 和 `/opt/anaconda3` 环境没有 PyTorch，旧 `/home/server/anaconda3/envs/py310` 路径不存在。本次未安装依赖，未运行 `torch.load`、事件推理、训练或 SED 重算。原始 `complete/` 定义仅保证救援字节覆盖，并不证明故障前文件内容正确。执行生产任务前，还需确认环境、模型／预处理契约、数据路径和相应单文件冒烟结果。

详细证据均在 ETO `.recovery/2026-10-14/`：`astro-source-manifest.json`、`verification.json`、`activation.json`、`source-comparison-preflight.json`、`variant-comparison.json`、缺损摘要、`runs-rsync.log`、审核脚本与 Slurm 日志。模型、恢复副本和生成产物继续留在忽略目录，不进入 Git。

## 已接回的 50 个模型

以下权重均为 `runs/<实验名>/checkpoints/best_model.pt`。配置“有”仅指 JSON 已恢复并能解析。

| 实验 | 配置 |
|---|---|
| `allcuts_relaxed_10pct_2730` | 有配置 |
| `allcuts_relaxed_2728` | 有配置 |
| `allcuts_strict_2728` | 有配置 |
| `corebox_true_2724` | 有配置 |
| `dangle3_2726` | 有配置 |
| `dangle_unlimited_2726` | 有配置 |
| `dc0_core1_14` | 有配置 |
| `dc20_core0_13` | 有配置 |
| `dc20_core0_14` | 有配置 |
| `dcedge20_2724` | 有配置 |
| `fitstat0_2727` | 有配置 |
| `fitstat_all_2727` | 有配置 |
| `huber_clean_31` | 有配置 |
| `huber_noise_32` | 有配置 |
| `huber_raw_31` | 有配置 |
| `loss_huber_24` | 有配置 |
| `loss_mse_24` | 有配置 |
| `loss_rel_24` | 有配置 |
| `mcw_hist_loss_23` | 有配置 |
| `mse_loss_21` | 有配置 |
| `no_core_cut_2724` | 有配置 |
| `pincness_1p1_2725` | 有配置 |
| `pincness_unlimited_2725` | 有配置 |
| `proton_only_smoketest_10876` | 缺配置，保留权重 |
| `sample_random_15` | 有配置 |
| `sample_topkq_15` | 有配置 |
| `sample_weightedq_15` | 有配置 |
| `theta30_10pct_2731` | 有配置 |
| `theta_embed_19` | 有配置 |
| `theta_embed_2707` | 有配置 |
| `theta_embed_2714` | 有配置 |
| `theta_embed_2716` | 有配置 |
| `theta_embed_2717` | 有配置 |
| `theta_embed_edge0_2718` | 有配置 |
| `theta_embed_edge0_2719` | 有配置 |
| `theta_embed_mid_2709` | 有配置 |
| `theta_nv_embed_2740` | 有配置 |
| `theta_nv_embed_smoketest_2739` | 有配置 |
| `theta_position_embed_2735` | 有配置 |
| `theta_position_embed_smoketest_2734` | 有配置 |
| `theta_recoxy_position_embed_2742` | 有配置 |
| `theta_recoxy_position_embed_midenergy_no_core_cut_64670` | 有配置 |
| `theta_recoxy_position_embed_noemax_no_core_cut_64748` | 有配置 |
| `theta_recoxy_position_embed_smoketest_2741` | 有配置 |
| `theta_truecore_points_edge0_2733` | 有配置 |
| `theta_truecore_points_smoketest_2732` | 有配置 |
| `theta_unlimited_10pct_2731` | 有配置 |
| `trueE_1e2_1e4_20` | 有配置 |
| `trueE_ge_1e4_20` | 有配置 |
| `ture_in_pred_2700` | 有配置 |
