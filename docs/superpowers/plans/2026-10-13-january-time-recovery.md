# 一月观测数据并行时间恢复实施计划

> **执行方式：**使用 `superpowers:executing-plans` 在当前会话逐项执行；本侧对话禁止使用子代理。本文件只制定计划，尚未部署程序或提交恢复任务。

**Goal:** 尽量复用 IHEP 的旧时间和 J2000 坐标成品，仅补算缺失事件，按新裁剪集的完整事件顺序生成 friend，并与正在运行的 IHEP→ETO 观测同步并行。

**Architecture:** 以 ETO 每个观测文件的持久接收记录为启动条件，运行独立的时间恢复流水线。IHEP 仅读取原始 EOS 的 pincness 和六个事件标量，构造与裁剪集一致的目标序列；关联旧主文件与旧 friend，复用有效结果并调用原算法补缺。成品通过有容量限制的 EOS 暂存队列传到 ETO，接收端逐条比对目标事件序列后发布。

**Tech Stack:** ROOT / C++、旧 Cog 时间转换及 SLALIB 坐标算法、Python 标准库与现有 ETO uproot 环境、IHEP HTCondor / hep_sub、XRootD / EOS、共享 Lustre 状态目录。

---

## 已核实的现状与执行边界

- 计划基于 Git `98ea92d2f44ba38ee64b76f243bc50474a35b44c`。执行前重新核对版本和未提交改动，不覆盖其他任务的代码。
- 2026-10-13 19:57 的观测接收快照为 286 / 681 文件、62,621,272,800 bytes、无当前错误。这是历史检查点，不代表执行计划时的实时进度。
- 原 home 配额阻塞已经由主线迁移处理。现有裁剪控制目录是 `/scratchfs2/lhaaso/liushijie/WCDA_obs_202201_trimmed_20261013/.control`，禁止再写旧 home 运行目录。
- 主线已定位 `lhws247.ihep.ac.cn` 的 CVMFS/ROOT 故障并仅重调度受影响裁剪任务。时间恢复基准和生产沿用该节点排除条件，先做小型运行环境探测；不能仅凭 JobStatus Running 宣称正在有效计算。
- 现有裁剪队列容量为 5.5 GiB。该队列、生产者、接收程序和清理确认协议保持独立，不为时间恢复暂停或替换它们。
- IHEP 旧档案 `/home/lhaaso/liushijie/WCDA_observation_eval_64670_20220101_20220630/` 仍有一月 681 对主文件与 friend。旧 friend 合计 2,507,086,737 bytes。
- 旧一月摘要记录 48,726,074 entries，48,725,681 matched；`0101_22` 有 393 条 missing_reduced，另有 102 条 ies_fallback。摘要只能用于定位，不能代替重新校验旧文件内容。
- `/eos/user/l/liushijie/` 查询到个人配额 1,000,000,000,000 bytes、已用 802,284,381,769 bytes。实际写入、发布和计算节点访问尚未验证，不能把这次余量直接当作永久预算。
- 不训练、不推理、不增加物理筛选、不修改已确认观测 ROOT、原始 EOS、旧档案或其他项目。没有用户新的执行指令时，停留在本计划交付。

## 数据契约和复用判定

1. 新目标序列只采用 `pincness < 1.1`，沿原始事件顺序保留每一行。对 NaN 的行为与当前裁剪宏一致。不使用 fitstat、theta、dcedge 等额外筛选。
2. 事件键为 `(irun, ies, iseq, ievent)`，使用原始整数类型，不经过浮点转换。目标 theta/phi 保持原始 float64 数值和弧度单位。
3. 旧 friend 不包含事件键。先验证旧主树与旧 friend 条数一致，再将同一 entry 的键、theta/phi 和恢复结果关联。为每个目标文件建立同名旧文件索引；同名未命中时查同日其余旧文件的索引，扩大复用范围。只缓存当前日期索引，限制内存，不建立常驻全月大索引。
4. 旧 `match_status == 0`，时间字段有效、字段间自洽，且同键结果无冲突时，复用时间字段。复用字段保持原数值，不重新做时间转换。
5. 只有确认旧坐标确实计算过、坐标有效、theta/phi 完全一致、转换约定兼容时，复用旧 RA/Dec。旧程序关闭 WITH_RA_DEC 时会写零坐标但仍可能 match_status=0，因此不能仅检查 match_status 或有限值。没有运行证据时，用旧时间及旧角度按原算法重算核验，核验不能通过则只复用时间、重算坐标。
6. 相同事件键出现不同有效时间时，报告冲突并停止该文件发布，不采用任意一条。目标集中的重复行仍保留，不能去重或改变顺序。
7. 无有效旧时间的事件，从 `/eos/lhaaso/raw/wcda/2022/<MMDD>/*.reduced.root` 补齐。沿用旧程序的文件定位和 `wcda.ievent` 核对，不放宽 ies fallback：仅精确文件不存在且 `(irun, iseq)` 候选唯一时允许；精确文件不可读时禁止改用别的 ies。旧摘要存在 ies_fallback 的文件须按事件键定位并复核这些行的来源；若现在精确源可得，则重新恢复这些行，旧 fallback 来源无法审计时不直接复用。
8. 补算继续使用 `Ttai_t → Tunix_t / Tmjd_t`，不手工减固定秒数；坐标继续使用 `azimxwcda → hcs2eqa → eqa2eqm`，结果为 J2000 平赤道坐标。
9. `t_recovered_time` 保留旧的 11 个分支及类型：`tai_sec/L, tai_subsec/D, unix_sec/L, unix_subsec/D, mjd/D, mjdns4/I, ra_mean/D, dec_mean/D, ra_mean_deg/D, dec_mean_deg/D, match_status/I`。
10. 同一 friend 增加辅助树 `t_alignment`，保存 `entry_index`、四元事件键、theta、phi，供 ETO 对主文件逐条核验。旧下游仍按树名读取 t_recovered_time；不把这些分支添加到现有 30 分支观测主树。
11. 无法恢复的行仍写占位和非零 match_status，保持完整对齐。区分“681 个文件齐全”和“全部事件恢复”；残留未匹配不能被计为成功恢复，不能静默丢行。

## 文件与运行目录

实现代码归入项目 Git，运行目录只部署对应提交快照：

| 文件 | 职责 |
|---|---|
| `scripts/data/recover_observation_time.cc` | 以现有 `time-recover/src/recover_wcda_time.cc` 为算法来源；标量目标序列、旧结果关联、补缺、双树输出、完整读回检查 |
| `scripts/data/run_observation_time_shard.sh` | HTCondor 工作入口、显式 ROOT/Cog/SLALIB 环境、作业临时盘、有限重试和逐文件运行 |
| `scripts/data/sync_observation_time.py` | 读取观测 receipts、独立请求队列、EOS 容量预留、接收对齐核验、持久确认及任务临时副本清理 |
| `scripts/slurm/sync_observation_time.sbatch` | ETO 接收/对齐检查的 1 CPU / 2 GiB Slurm 入口，避免在登录进程中做全月批量校验 |
| `scripts/data/test_observation_time.py` | 小型 ROOT 回归数据、匹配/复用/补缺/顺序/失败与重启案例 |
| `apply/docs/OBS_JANUARY_TIME_RECOVERY.md` | 部署、任务编号、检查命令、实测性能、异常清单、下游对齐要求 |
| `apply/devlog.md` | 每次有意义改动的简短记录 |

拟采用的独立路径（本计划未创建远端目录）：

```text
ETO 观测输入及其接收记录：
  /mnt/newdisk/shijie/WCDA_obs/<MMDD>/EsgYYYYMMDD_HH.root
  /mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january/receipts/

ETO 时间恢复状态与最终成品：
  /mnt/newdisk/shijie/energy/.recovery/2026-10-13/obs-january-time/
  /mnt/newdisk/shijie/WCDA_obs/recovered_time/<MMDD>/EsgYYYYMMDD_HH.time.root

IHEP 独立状态、请求、确认与部署代码：
  /scratchfs2/lhaaso/liushijie/WCDA_time_recovery_202201_20261013/.control/

IHEP EOS 临时成品：
  /eos/user/l/liushijie/WCDA_time_recovery_202201_20261013/<MMDD>/

每个作业的局部中间文件：
  $_CONDOR_SCRATCH_DIR/wcda_time/
```

## Task 1：执行前检查与独立存储验证

**Files:** 操作文档、运行配置；暂不修改现有裁剪脚本。

- [ ] 重新读取本地、ETO、IHEP 适用 AGENTS.md，核对当前 Git、原始 31 manifests、观测接收记录和相关任务归属。只把真实 manifest 中的 681 名称作为预期集合。
- [ ] 查询 scratch 的个人字节/文件配额、EOS 个人及适用组配额。scratch 的预算必须覆盖现有裁剪队列所有预留、当前非队列用量和新增控制目录；继续保留至少 1 GiB 软限安全量、6 GiB 硬限安全量。预算不够时不启动新恢复，不擅自减少现有裁剪预算。
- [ ] 新时间控制目录预算最多 64 MiB，并预留文件数余量。只持久保存摘要、任务/确认记录和每次失败最多 32 KiB 日志尾部；大日志计入 EOS 暂存预算。额外错误刷屏不得写回 home 或占满控制目录。
- [ ] 在 HTCondor 极小作业中验证本地临时盘、EOS 上传临时名、关闭后读回、发布名称、接收确认、仅删除测试临时文件。明确使用 XRootD/EOS 接口，不把 POSIX fsync/flock 假设套用到 EOS。
- [ ] 起始 EOS 队列上限取 `min(16 GiB, 可用个人/组余量减 20 GiB 安全量)`，若不能容纳两个工作者的预留则降低并发或不启动。队列计入上传中的 partial、已完成待传文件、失败遗留及临时日志；共享 Lustre 上的锁和账本负责并发预留，不能只限制文件数。

执行时的只读检查命令：

```bash
lfs quota -u liushijie /scratchfs2
eos root://eos01.ihep.ac.cn quota ls -m -u 24005 -p /eos/user/l/liushijie/
```

**通过条件：**实际计算节点完成极小文件的上传、读回、接收确认和安全清理；控制目录与两条流水线的总配额预算成立。登录节点的 test -w 或 df 不是通过依据。

## Task 2：建立“旧结果复用＋缺失补齐”核心

**Files:** `recover_observation_time.cc`、`test_observation_time.py`。

- [ ] 将自有旧恢复实现纳入 scripts/data，保留原转换公式、时间类型和 reduced 查找约束；Cog/SLALIB 作为显式外部依赖记录路径、版本及运行环境，不私改第三方库。
- [ ] 先写微型 ROOT 用例：旧主文件 A/B/C 对应有效/失败时间，新目标顺序 C/A/D/A；应输出相同四行，复用成功的 A 两次，补算 C/D，不能按旧行号连接。
- [ ] 增加同名文件未命中、同日其他旧文件命中的用例；相同键冲突、旧主/friend 条数不一致、旧坐标未生成、theta/phi 改变、精确 ies 不可读均不得误认成可完整复用。
- [ ] IHEP 直接读取已确认文件对应的原始 EOS：只扫描 pincness 及匹配所需六个标量，复现目标序列；不读取 hit，不依赖会被接收端清理的临时裁剪文件，也不把完整小时原始文件缓存到 scratch。
- [ ] 对新增 reduced 查询按文件分组/缓存必要的 ievent/sec/subsec 数据，避免同一 reduced 文件被逐事件重复打开。仅对缺失事件恢复时间；仅对缺少有效坐标的事件调用 SLALIB。
- [ ] 按目标 entry_index 写两个树，失败行保留。检查每次 GetEntry/Fill/Write/Close、ROOT 错误和文件结束边界；输出先在作业临时盘写 .partial，完整读回所有输出行后才成为可上传成品。
- [ ] 输出审计计数：目标数、复用时间数、复用坐标数、新恢复时间数、新算坐标数、未匹配数、冲突数和 ies_fallback 数；验证每行去向与总数一致。

**通过条件：**小型回归先复现错误连接/不完整写入，再通过正确实现；复用分支逐值保持旧结果，补算分支符合旧算法，所有失败路径保持行数和顺序。

## Task 3：独立按文件调度、确认与清理

**Files:** `sync_observation_time.py`、`run_observation_time_shard.sh`、上述回归测试。

- [ ] ETO 只读现有观测 receipts。必须有持久记录、正确名称、字节数及主树数量校验，才发布独立时间任务。支持已有 legacy SHA 身份，但不新算数据哈希。
- [ ] 新 ETO 接收/对齐程序通过 Slurm 运行，先验证作业节点到 IHEP 的非交互通道；复用现有 ControlMaster 仅限同一主机可访问的情况，不复制密钥、读取密码或把批量校验回退到登录进程。
- [ ] 时间任务携带原观测 record_id、目标名称/条数、对应原始 EOS URL、配置及 Git 提交。处理前后核对输入身份；名称相同但身份变化时保留旧结果并报告冲突。
- [ ] 共享队列使用每文件锁和原子状态转换：`requested → running → ready → received → cleaned`；失败转入可追踪的 retry 状态。最多三次处理尝试，长期 SSH/传输失败保留成品并背压，不能误计完成。
- [ ] 接收端通过独立 incoming 路径下载 .time.root，按有内存上限的分块完整读取 t_alignment，与 ETO 主文件六标量及连续 entry_index 逐条比较，再检查 t_recovered_time 的 11 个字段、分支类型、数量和状态统计；不读取 hit。仅比较数量不足以验收。
- [ ] 发布顺序：校验字节/ROOT/完整对齐 → fsync ETO 文件 → 原子发布及目录 fsync → 持久化本地时间 receipt → 持久化 IHEP 独立确认 → 删除该任务的 EOS 临时成品。已确认 ETO 成品不覆盖。
- [ ] 作业退出前成品必须已发布到受预算保护的 EOS，不能仅留在会被 HTCondor 清理的本地盘。未成功发布时记录失败，重跑可重建；不能把局部文件消失当成成功。
- [ ] 加入在上传中断、ETO 发布后确认前中断、确认后清理前中断的重启用例，验证不重复计算已确认结果、不提前清理、不碰旧时间档案及观测接收记录。

**通过条件：**微型文件完成真实双向确认；重启能恢复；现有观测同步的代码、进度文件、receipts 和容量账本保持独立。

## Task 4：完整小时基准与逐步并行

**Files:** 新操作文档、任务配置、审计及基准摘要。

- [ ] 首先通过 hep_sub 提交 1 路完整小时基准，选已到达 ETO 的 `0101/Esg20220101_00.root`。新目标已核实 320,258 行、旧同名 friend 摘要 228,604 行；这些数是对照点，不预设全部旧行一定能复用。
- [ ] 同一目标用旧算法全量恢复作为参考，与混合复用结果逐行比较：复用字段必须完全一致，新算字段按相同环境旧算法核验。差异先定位并报告，不擅自放宽时间/坐标精度。
- [ ] 对 `0101_22` 专门验证旧 393 条失败，统计能补齐多少、哪些 reduced 仍不可取得；所有残留必须有对应事件键和失败原因。
- [ ] 每阶段单独计时：目标标量读取、旧索引读取、键匹配、缺失时间恢复、坐标补算、完整校验、EOS 发布、传输、确认/清理、背压等待；记录 CPU、RSS、读字节、复用覆盖率。
- [ ] 一个完整小时达到 ETO、对齐核验通过、确认持久化且本次 EOS 临时副本删除后，再启用 **2 路 IHEP 恢复**。每路初始申请 1 CPU / 4,000 MB / 4 GiB 本地临时盘，实际批处理验证后调整。
- [ ] 与继续运行的观测裁剪同时观察至少一组完整小时：对比同期观测吞吐、EOS/NFS 等待、个人配额、两条队列积压和新增恢复吞吐。若恢复挤占裁剪，则先降为 1 路；没有实测理由不升到 4 路。

提交沿用已经验证的站点入口：加载 `/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/setup_hepjob.sh`，使用 `hep_sub -g lhaaso -os AlmaLinux9 -m 4000 -cpu 1 -schedd schedd11`。临时盘申请和 walltime 参数在该入口实际帮助/提交结果中核实，不照搬未验证的选项，不退回曾经无法匹配的原生提交模板。

**通过条件：**至少一个新时间成品已发布到用户指定目录，并完成独立确认和临时清理；随后真实 2 路运行、记录有效任务编号。ETA 由实测复用率和各阶段吞吐更新，不沿用观测裁剪的 2–3 小时估计。

## Task 5：全月验收、下游约束和交付

- [ ] 最终名称集合严格对应原 31 日期 / 681 文件 manifests，每个 friend 的主树与 alignment 条数均等于当前对应观测主树，所有输入身份匹配。
- [ ] 汇总复用/补算/未匹配/冲突、分日期异常及确认/清理情况；解释保留的非零 match_status，不把文件齐全写成全部事件成功。
- [ ] 确认新的 EOS 临时队列没有未确认输出，确认记录仍保留；不删除原始 reduced、旧主文件、旧 friend 或观测数据。
- [ ] 当前 friend 对应仅 pincness 裁剪集。之后能量推理还会筛选事件及剔除无法准备的样本，必须按推理最终 survivor_event_indices 同步筛选 friend/alignment，或按事件键重新关联；不能直接把当前 friend 挂给条数更少的 eval。
- [ ] 按阶段把实现、必要回归、操作文档及 devlog 提交并 push，部署完全对应的代码快照。报告当前进度、有效任务编号、实际并发、个人配额、队列容量、复用覆盖率、瓶颈及剩余 ETA。

**本计划的停止点：**只交付计划和 Git 文档提交；执行上述测试、创建远端恢复目录、迁移存储和提交 HTCondor 均属于后续实施。
