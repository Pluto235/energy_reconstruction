# Crab SED Roadmap v2（raw65 ledger + frozen selector）

**目标。** v2 的核心不是更早删 cell，而是把 cell 选择推迟到拟合阶段：Stage A-E 尽量保留可追踪的原始 cell、响应、PSF、背景和信号 metadata；Stage F/G 只用预冻结 selector 进入物理拟合。任何基于 Crab on-source 结果的量，例如 `N_on/B_on`、excess、显著性、Stage F pull、Stage G residual，都只能作为诊断 flag，不能作为正式 include/exclude 规则。

参考输入：

- 现有 v1 roadmap: `apply/report/roadmap.md`
- v2 skymap quicklook: `apply/report/crab_v2_cell_skymaps.md`
- 低能异常诊断: `apply/report/low_energy_cell_diagnostics.md`
- 当前 v2 cell 文件: `apply/config/cell_selection_v2.csv`
- 当前 v2p1 文件: `apply/config/cell_selection_v2p1.csv`

---

## 1. v2 设计原则

1. **raw ledger 尽量全。** Stage A/B/C/D/E 都跑同一套 `raw65` 非重叠 cell ledger，不因为某个 cell 当前看起来贡献小、背景大或 residual 差就在上游删除。
2. **fit selector 预冻结。** Stage F/G 使用单独 selector 文件决定 include/exclude。selector 必须基于 MC、几何、PSF、背景稳定性、coverage、预期灵敏度等 prefit 信息；不能基于 Crab 实测 `N_on/B_on`、excess、significance 或 fit residual。
3. **诊断和拟合分离。** Stage E 输出 65-cell 全表和 flags；Stage F/G 默认只读取 frozen baseline cells，但保留 probes/systematics selectors。
4. **非重叠 binning。** 旧 `>=2000` open Nhit bin 不再进入 v2 raw ledger；高端用 v2 skymap 已验证过的 `[2000,3000)` 五个 cell。暂不加 `>=3000`，除非后续 MC/obs 统计单独证明可用。
5. **双里程碑背景路线。** v2.0 先闭环 direct-expectation diagnostic；v2.1 再升级到真实 `N_off/alpha` 和 Li-Ma。

---

## 2. Cell Ledger 与 Selector

### 2.1 Stage A-E raw ledger: `v2_raw65`

`v2_raw65` 是 Stage A/B/C/D/E 的主 cell 输入。定义为：

- 从旧 acceptable pool 保留所有 `statistics_level == acceptable` 且 `Nhit != >=2000` 的 60 个 cell；
- 加入 v2 的 `[2000,3000)` 五个 high-Nhit cell；
- 总计 65 个互斥 cell。

建议新增配置文件：

```text
apply/config/cell_ledger_v2_raw65.csv
```

每行至少包含：

```text
cell_id
nhit_bin
predE_bin
mc_count
raw_ledger_version = v2_raw65
cell_role
role_reason
source_pool
```

推荐 `cell_role`：

| role | 用途 |
|---|---|
| `baseline_fit` | 默认进入 Stage F/G baseline |
| `transition_probe` | 不进 baseline，可做低能/边界 probe |
| `diagnostic_legacy_low_nhit` | `[30,100)` legacy low-Nhit diagnostics，永久不进 v2 baseline |
| `diagnostic_low_stat_high_energy` | v2 high-energy 边界低统计 cells |
| `diagnostic_response_tail` | old acceptable pool 中偏离 physical ridge 的 response-tail cells |

### 2.2 Stage F/G baseline selector: `v2_baseline26`

默认 Stage F/G baseline 是 **26 个预冻结 cells**：

1. 当前 `cell_selection_v2p1.csv` 的 23 个 cells；
2. 加入 `[100,200)` 的 physical ridge 三格：
   - `[100,200) x [2,3)`
   - `[100,200) x [3,3.25)`
   - `[100,200) x [3.25,3.5)`

建议新增 selector：

```text
apply/config/cell_selector_v2_baseline26.csv
```

这个 selector 的定义是 prefit 选择，不是 Crab excess 选择。它允许 `Nhit >= 100`，但 `[100,200)` 只纳入 physical ridge 子集，不整行纳入。`[100,200) x [3.5,3.75)` 作为 `transition_probe`，不进默认 baseline；`[100,200)` 更高 `E_pred` tail 只做 diagnostics。

### 2.3 禁止的正式 selector 规则

以下量只能写入 Stage E/Stage F 诊断表，不能作为正式 baseline selector：

- `N_on/B_on`
- `excess`
- per-cell Crab significance
- Stage F pull
- Stage G SED residual
- “删掉这个 cell 后 chi2 变好”这类 post-fit 决策

如果未来要用这些信息定义新 selector，必须走 split validation、独立月份验证或明确标记为 exploratory/probe，不能回填成 v2 baseline。

---

## 3. v2.0 Milestone: raw65 + direct-expectation diagnostic

v2.0 的目标是快速闭环 65-cell metadata-preserving pipeline，并给出 26-cell diagnostic SED。它不声称 Li-Ma，不作为最终正式统计口径。

### Stage A — Response on raw65

- 输入 `cell_ledger_v2_raw65.csv`，对 65 个 cell 全部构建 `eta_b(E_true, theta)` 和 `A_eff,b(E_true, theta)`。
- 保留 `S0 = 4.0e6 m^2` 的单位修正。
- 输出 metadata 必须记录每个 cell 的 `cell_role`，但不根据 role 删除 response。
- 响应自检要覆盖 raw65 全体，同时单独汇总 baseline26。

### Stage B — PSF on raw65

- 对 65 个 cell 全部建立 Crab 赤纬带 PSF、`sigma_deg`、`r_opt_deg`、containment。
- 对低统计或拟合失败 cell 打 `psf_quality_flag`，但 Stage B 不删除 cell。
- Stage E/F 通过 selector 和 quality flag 决定是否用于拟合。

### Stage C — Observation reduction on raw65

- 从 ROOT + recovered time 规约观测事件到 raw65 cell。
- 输出 parquet 保留 `(ra_mean_deg, dec_mean_deg, mjd, theta, cell_id, nv, ml_logE_pred)` 等下游需要的列。
- 不在 raw65 的事件不进入主 parquet，但 metadata 必须记录 out-of-ledger counts，按 Nhit/predE bin 给出诊断。
- Crab-centered ROI coverage 继续输出 `rho` profile、`rho < 6`、`6 < rho < 8`、per-cell coverage。

### Stage D — ROI-local direct expectation on raw65

- 当前输入被视为 Crab local ROI；v2.0 默认 `background_mode = crab_roi_local`。
- 默认 fiducial ROI 仍为 `rho < 6 deg`，`6 < rho < 8 deg` 只做 edge diagnostics。
- 背景方法使用 ROI-local equal-Dec sideband 或同等 direct expectation：
  - 输出 `B_on,b`
  - `background_form = direct_expectation`
  - `N_off/alpha` 标注为 not applicable
- 对每个 cell 输出 background quality flags：
  - training/off pixels
  - source mask radius
  - edge-safe area
  - sideband/ring stability
  - fiducial radius sensitivity

### Stage E — Signal table on raw65

- 对 65 个 raw cells 全部计算 `N_on,b`、`B_on,b`、excess、known-background Poisson diagnostic。
- 若 Stage D 是 `crab_roi_local`，必须使用同一个 fiducial ROI 和 mask/edge config。
- 输出全量 65-cell signal table，同时给出 baseline26 聚合量。
- `N_on/B_on`、known-B sigma、excess/model 只作为 diagnostic columns/flags，不得反向改变 selector。
- Promotion gate 分两层：
  - raw65 artifact gate: contract/schema/scan completeness
  - baseline26 diagnostic gate: total significance 在合理区间，且 report 明确是 direct-expectation diagnostic

### Stage F — Forward folding on baseline26

- Stage F 读取 raw65 response/signal，但默认 selector 是 `cell_selector_v2_baseline26.csv`。
- 拟合模型顺序仍为 PL，然后 LogPar；LogPar 只有在明确改善时作为 preferred。
- 输出必须记录：
  - raw ledger version: `v2_raw65`
  - fit selector: `v2_baseline26`
  - included/excluded cell ids
  - excluded reasons from selector file
- 不允许在 Stage F 内按 pull 自动删 cell。

### Stage G — Diagnostic SED on baseline26

- 固定 Stage F 谱形，对 Nhit group 和/或 energy-pred group 拟单点归一化。
- 输出 v2.0 diagnostic SED，与 WCDA-1 / HAWC / HESS 比较。
- 报告必须写明：
  - 使用 direct-expectation background
  - Li-Ma 不适用
  - 这是 diagnostic SED，不是最终 formal on/off 统计版

---

## 4. v2.1 Milestone: raw65 + true off counts / Li-Ma

v2.1 在 v2.0 闭环后升级 Stage D/E 的统计口径，目标是让 Stage E 能输出真实 on/off 统计和 Li-Ma significance。

### Stage D upgrade

- 在 `rho < 6 deg` fiducial ROI 内定义真实 off regions。
- 优先实现两种背景 cross-check：
  - equal-Dec sideband off counts
  - ring/annulus off counts with explicit ROI geometry correction
- 每个 cell 输出：
  - `N_off,b`
  - `alpha_b`
  - `B_on,b = alpha_b * N_off,b`
  - off-region geometry/mask/area metadata
  - off acceptance / edge correction diagnostics
- metadata 必须明确：
  - `background_form = off_counts`
  - `li_ma_applicable = true`
  - off definition version

### Stage E upgrade

- 若 Stage D 输出 `off_counts`，Stage E 计算：
  - `excess_b = N_on,b - alpha_b N_off,b`
  - `sigma_stat = sqrt(N_on,b + alpha_b^2 N_off,b)`
  - Li-Ma significance
- direct expectation 不得被解释成 off counts。
- v2.1 的 Stage E report 同时保留 known-B diagnostic 作为对照，但 formal significance 使用 Li-Ma。

### Stage F/G upgrade

- Stage F/G 默认仍使用 `v2_baseline26`，保证 v2.0 和 v2.1 的差异主要来自背景统计升级，而不是 selector 变化。
- 对照输出：
  - v2.0 direct-expectation fit
  - v2.1 Li-Ma/on-off fit
  - 背景方法 sideband vs ring sensitivity

---

## 5. Systematics 与验证

1. **raw65 completeness.** Stage A-E 的 cell id、Nhit bin、predE bin、role 必须一致；任何 stage 不得隐式丢 cell。
2. **selector freeze audit.** Stage F/G metadata 必须证明 selector 文件在 fit 前已存在，且未使用 Stage E source-result columns 自动筛选。
3. **low-Nhit diagnostics.** `[30,100)` 永久 diagnostic-only；`[100,200)` 只允许 physical ridge 三格进入 baseline，其余 tail cells 作为 response/background stress tests。
4. **high-energy diagnostics.** `[2000,3000)` 五格保留在 raw65；默认 baseline 只纳入 v2p1 保留的 `[4.75,5.0)` 和 `>=5` 两格，其余低统计高能边界 cells diagnostic-only。
5. **ROI edge sensitivity.** v2.0/v2.1 都要重跑 `rho < 5.5`、`6.0`、`6.5 deg` 的 Stage D/E 对照，baseline26 excess 和谱参数不能对边界产生不可解释漂移。
6. **background method sensitivity.** v2.1 至少比较 equal-Dec sideband 和 ring/annulus。
7. **1D vs 2D consistency.** raw65 响应沿 `E_pred` 轴 marginalise 后的一维 Nhit fit，应与二维 baseline fit 在统计误差内一致。
8. **response closure.** 用参考谱 forward-folding 回 MC cell counts，raw65 和 baseline26 都要给 closure summary。

---

## 6. 交付物

| 文件 / artifact | 内容 |
|---|---|
| `apply/config/cell_ledger_v2_raw65.csv` | 65 个非重叠 raw cells + role metadata |
| `apply/config/cell_selector_v2_baseline26.csv` | 26 个默认 Stage F/G baseline cells |
| `apply/config/cell_selector_v2_transition_probes.csv` | `[100,200)` transition 和高能边界 probes |
| `apply/output/stage_a/...` | raw65 response + baseline26 summary |
| `apply/output/stage_b/...` | raw65 PSF + quality flags |
| `apply/output/stage_c/...` | raw65 observation parquet + out-of-ledger diagnostics |
| `apply/output/stage_d/...` | v2.0 direct expectation 或 v2.1 off-count background |
| `apply/output/stage_e/...` | raw65 signal table + baseline26 aggregate |
| `apply/output/stage_f/...` | selector-driven forward-folding fits |
| `apply/output/stage_g/...` | diagnostic/formal SED points |
| `apply/report/roadmap_v2.md` | 本文档 |
| `apply/report/roadmap_v2.html` | 本文档 HTML companion |

---

## 7. 当前固定决策

1. v2 raw ledger 使用 **65 个非重叠 cells**：旧 acceptable 去掉 `>=2000` open bin，加 v2 `[2000,3000)` 五格。
2. Stage A/B/C/D/E **全部跑 raw65**。
3. `[30,100)` 保留在 raw ledger，但永久标记为 `diagnostic_legacy_low_nhit`，不进 v2 baseline。
4. Stage F/G 默认 baseline 是 **26 cells**：v2p1 的 23 cells 加 `[100,200)` physical ridge 三格。
5. `[100,200)` 不整行进入 baseline；`[3.5,3.75)` 是 transition probe，更高 `E_pred` tail diagnostic-only。
6. `N_on/B_on < 1.5` 或类似 on-source 结果不能作为正式剔除条件，只能是诊断 flag。
7. v2.0 和 v2.1 写在同一个 roadmap 中：v2.0 先做 direct-expectation diagnostic 闭环，v2.1 再做 true off-counts / Li-Ma 统计升级。
