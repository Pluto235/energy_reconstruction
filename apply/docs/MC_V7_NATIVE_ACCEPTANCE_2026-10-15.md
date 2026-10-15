# MC v7 friend 原生 Nhit 契约交付

2026-10-15 用户明确确认：权威 Nhit 为原生、无 mc_ 前缀的 `nq03t30s0 + nq03t30s1 + nq03t30s2`；无需逐 hit 重算核验。用户说明上游采用 `abs(vt)<30`。本次按这一来源契约执行，不把此前存储 float32 时间的诊断解释为对上游源码的独立验证。

rmds 仅因 float64/float32 舍入发生 `<=20` 判断翻转时，接受摘要判断；不修改 eventout 原值。超出存储舍入范围的 rmds 差异仍阻断发布。chi2 沿用用户既有豁免，角度边界、唯一匹配、类型、范围、整数求和溢出和 friend 回读仍保留。

## 最终全量结果

| 项目 | 结果 |
| --- | ---: |
| 目标/成功文件 | 10,000 / 10,000 |
| 成功事件 | 38,521,725 |
| 失败/待处理 | 0 / 0 |
| 必需字段覆盖率 | 100%（全部 12 字段） |
| 新验证补发 | 19 文件 / 73,487 事件 |
| 独立回读 | 10,000 文件 / 38,521,725 事件，失败 0 |
| 非有限匹配键、重复、缺失、歧义、摘要行复用 | 全部 0 |
| 未使用摘要记录、重复 eventout 内容哈希 | 0 / 0 |
| poolused=7 | 38,521,725 事件 / 10,000 文件 |
| poolused 其他值 | 0 |
| 接受的 rmds 舍入边界翻转 | 1 |
| theta 三项明确边界翻转 | 全部 0 |

此处“全部 poolused==7”覆盖全部目标事件，不再仅限此前成功部分。所有原生整数及三池求和均验证通过；不将历史逐 hit 重算作为当前接受依据。没有剩余失败文件或待重试文件。

Slurm 66145 已完成（2 分 9 秒，退出码 0:0）；上述结论依据逐字段回读与匹配统计，不仅依据作业退出码。66147 复跑 13 项测试通过，并更新混合旧/新 receipt 汇总中 Nhit<30 的历史比较分母；没有改动 friend 或重复执行逐 hit 计算。历史比较覆盖 38,448,238 事件，73,487 个新验证事件标记 not_compared；这些比较统计不是本次契约的发布门槛。

## 使用目录与版本

使用完整的新版本：

`/mnt/newdisk/shijie/WCDA_simulation/v7_friends/run_d84da6abb98b/`

每个文件位于 `files/<stem>/<stem>.v7friend`，TTree 为 `v7_friend`，字段前缀 `v7_`。其中 `v7_nhit_authoritative` 是权威三池整数的 int64 求和，原生字段整数类型和值保留，`v7_eventout_entry`/`v7_summary_entry` 提供显式映射。按 eventout 全部原始事件顺序保存，无物理筛选；Nhit<30 亦保留。

新目录独立于旧版本：9,981 个既有验证包逐字节复制，19 个旧隔离文件重新匹配、验证和生成。复制包保留原 v1 代码/契约元数据，表示它们在更严格的旧契约下已通过；目录级 v2 契约代表本次接受规则，`lineage.json` 逐文件说明复用或重验及 receipt 哈希。没有软链接依赖。读取时选择此版本目录，不跨版本重复合并。

旧目录 `run_fb615fce12b6` 及旧隔离 receipt 保留原状；旧调查报告的隔离结论属于当时契约，现已由用户明确的新规则取代。原 MC 文件不改写。新旧版本均使用 `.v7friend`，不被递归 `*.root` 发现逻辑作为 eventout 混入。

## rmds 接受案例

文件 `Egr000001180_01_n0010_eventout.root`，eventout entry=3163，摘要 entry=3163：

- eventout rmds：20.000000077668222，原值保留。
- 摘要 rmds：20.0，`<=20` 为真，按用户要求接受。
- 两者在 float32 存储精度下相等，receipt 标记 `accepted_rounding_flips=1`，不豁免超出该精度的差异。

本任务生成 friend，不运行下游物理选择；原 eventout 上已有的筛选代码没有被修改。

## 执行与验证

代码实现提交：`a54a9ad42b2ac3bb1f03a81de6dc994485b8b1a5`，核心 SHA256：`d84da6abb98bd019be450aa76fc13dbee36d7a15fc59c72cc4e1d0c4aafd2ab1`。

Slurm 66143 的恢复测试发现契约中的 tuple 经 JSON 读回为 list，导致重启身份比较失败；已用规范化 JSON 比较修复。Slurm 66144：13 项测试全部通过，覆盖唯一匹配、缺失/重复/重排、精度、失败不发布、非 7 poolused、原生 Nhit、rmds 舍入接受与超范围拒绝，以及新版本复制/恢复/旧版不变。日志中 receipt/hash missing traceback 是传输清理负面测试的预期输出，不是测试失败。

Slurm 66145 执行 `release_native.py` → `audit.py` → `deliver.py`。提交脚本保存于 ETO `/mnt/newdisk/shijie/energy/apply/output/v7_native_accept_20261015/release.sbatch`，使用 main/default-safe、1 CPU、2 GiB、30 分钟。脚本绑定上述代码提交；断点续作必须保持同一代码、输入、契约和版本路径。

19 份摘要采用有界暂存，其中 18 份复用已取回的原生摘要，新增文件 1180 摘要从 IHEP 原路径取回；19 份摘要及 ETO eventout SHA256 均重新与旧 receipt 校验。所有目标 eventout stat 身份再核对，旧版 10,000 份 receipts 哈希前后不变。复制 friend 的输入/输出 SHA256 一致，随后逐文件独立回读。

六个 float32 共同字段键仍是内容对应证据，不等同于独立物理事件 ID 认证。没有恢复上游已滤除事件，没有模型推理、训练或响应/能谱计算。

## 机器可读产物

新版本目录中的 `contract.json`、`lineage.json`、`summary.json`、`final_audit.json`、`delivery_summary.json` 与逐文件 `receipt.json` 共同记录契约、来源、字段覆盖和验证。应用目录 `output/v7_native_accept_20261015/` 保存提交脚本、测试及运行日志、Slurm accounting 和交付结果副本。数据、暂存和生成产物不提交 Git。

旧版逐 hit 统计仅作历史诊断。新补发的 73,487 个事件没有执行逐 hit 比较，汇总显式记为 not_compared，不能把未执行核验计为“重算一致”。原生三池 Nhit 本身仍逐事件验证整数类型、非负范围与求和溢出。
