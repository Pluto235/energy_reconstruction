# Crab SED Roadmap v3（HAWC-style cell selection + annulus 2D surface background）

**目标。** v3 的核心是同时解决两个问题：

1. 用 HAWC `B x Ehat` 的思想重选 cell：先定义完整二维候选网格，再用 MC/几何/PSF/背景稳定性等 **prefit** 信息冻结 selector。
2. 将 Stage D 从 equal-Dec sideband direct expectation 升级为 Crab-local annulus 2D surface background：用 Crab 周围 annulus 训练背景曲面，再外推到源核心区估计 `B_on`。

v3 最终交付物必须和 v2 一样形成完整 Stage A-G HTML 报告；报告至少包含 v2 已有图，同时新增 v3 背景拟合和 cell-selection 诊断图。

参考输入：

- v2 roadmap: `apply/report/roadmap_v2.md`
- v2 integrated report: `apply/report/crab_sed_v2_stage_a_to_g_report.html`
- HAWC reference: *Measurement of the Crab Nebula Spectrum Past 100 TeV with HAWC*, ApJ 881, 134, arXiv:1905.12518
- 当前 v2 raw ledger: `apply/config/cell_ledger_v2_raw65.csv`
- 当前 v2 baseline selector: `apply/config/cell_selector_v2_baseline24.csv`

---

## 1. v3 固定决策

1. **Nhit 下限改为 125。** v3 不再保留 `Nhit < 125`，也不把 `[100,200)` 拆成 `[100,125)` 和 `[125,200)` 两段；最低 Nhit bin 直接使用 `[125,200)`。
2. **Stage C 必须能访问 Nhit。** 当前 v2 Stage C parquet 只有 `cell_id`，没有原始 Nhit；v3 要在 Stage C 事件选择阶段完成 `Nhit >= 125` 和新 cell assignment，或在输出 parquet 中保留足够字段供后续审计。
3. **背景训练环带默认 `1.5 < rho < 3.5 deg`。** 对低 Nhit / 宽 PSF cell，允许按 PSF 尾巴把环带整体外移，例如 `2.0-4.0 deg` 或 `2.5-4.5 deg`，但必须是 prefit 规则。
4. **2D 背景曲面使用二阶模型。** nominal surface basis:
   `1, x, y, x^2, x*y, y^2`。第一版先做加权二乘/Poisson 近似，保留 Poisson log-link 作为系统学交叉检查。
5. **环带覆盖视为完整。** v3 仍记录 coverage diagnostics，但不把环带不完整作为当前主要风险。
6. **selector 仍然冻结。** 禁止用 Crab 实测 `N_on/B_on`、excess、significance、Stage F pull 或 Stage G residual 回头定义 baseline selector。
7. **predE 使用混合网格。** v2 的 `[2,3)` 作为最终 fit bin 太粗，v3 拆成 `[2,2.5)` 和 `[2.5,3)` 两个 0.5 dex 候选 bins；`[3,5)`，包括 `[4,5)`，已经是 0.25 dex，保留现状；`>=5` 数据量少，v3 只拆成 `[5,6)` 和 `>=6` 两个高能宽 bins，作为 baseline 合并高能 bin / upper-limit bin 候选。

---

## 2. HAWC 方法对 v3 cell selection 的借鉴

HAWC 的高能 Crab 分析不是只按 reconstructed energy 分箱，而是使用二维 analysis bins：

```text
shower-size bin B  x  reconstructed-energy bin Ehat
```

他们用 9 个 fraction-hit `B` bins 和 12 个 quarter-decade reconstructed-energy bins，得到 108 个候选 bins。实际 fit 时不是全部使用，而是基于 MC 分布预先选择：

- 去掉 reconstructed energy bias 很大的最低能 bins；
- 对每个 shower-size bin，看 MC 的 estimated-energy 分布，只保留 central 99%；
- 去掉空 bin、低统计尾部 bin、MC/data 不容易建模的 bin；
- gamma/hadron cuts 和 PSF 都在每个 2D bin 上单独优化/建模；
- 最终 GP fit 使用 40 个 bins，NN fit 使用 37 个 bins。

v3 采用同样原则，但变量换成：

```text
Nhit bin  x  predicted-energy / ml_logE_pred bin
```

这意味着 v3 baseline selector 的来源必须是 MC 和 prefit 数据质量，而不是 Crab on-source excess。

---

## 3. v3 cell ledger 与 selector

### 3.1 Candidate ledger: `v3_candidate_grid`

建议新增：

```text
apply/config/cell_ledger_v3_candidate.csv
```

最低 Nhit bin 从 `[125,200)` 开始。后续 Nhit bins 沿用 v2 的非重叠结构，初始建议：

```text
[125,200), [200,300), [300,500), [500,800), [800,1100), [1100,2000), [2000,3000)
```

predicted-energy 轴使用混合网格，而不是全范围强制 0.25 dex。v2 中 `[3,5)` 已经是 0.25 dex，尤其 `[4,5)` 已经对应 `[4.0,4.25)`, `[4.25,4.5)`, `[4.5,4.75)`, `[4.75,5.0)`，不需要额外细分；v2 中 `[2,3)` 太宽，但低 Nhit / 低能端能量分辨率有限，v3 拆为 `[2.0,2.5)` 和 `[2.5,3.0)` 两个 0.5 dex 候选 bins；v2 中 `>=5` 因统计量低，不强制拆成 0.25 dex，v3 改为 `[5.0,6.0)` 和 `>=6.0` 两个高能宽 bins。最终 baseline 仍由 MC central-99%、response/PSF/background quality 和 expected sensitivity 决定，`>=6.0` 可自然落为 upper-limit / diagnostic bin。每行至少包含：

```text
cell_id
nhit_bin
predE_bin
mc_count
candidate_version = v3_candidate
source_pool
cell_role
role_reason
```

### 3.2 HAWC-style prefit selection: `v3_baseline`

建议新增：

```text
apply/config/cell_selector_v3_baseline.csv
apply/config/cell_selector_v3_systematics.csv
apply/config/cell_selector_v3_high_energy_probes.csv
```

baseline selector 只允许使用以下 prefit 信息：

| 输入 | 用途 |
|---|---|
| MC `E_true` vs `E_pred` response | 保留每个 Nhit bin 的 central 99% reconstructed-energy population |
| MC effective area / expected counts | 去掉近零响应、极低 MC 统计、极端 response-tail bins |
| PSF quality | 去掉 PSF 拟合失败、`r_opt` 不稳定、containment 异常 bins |
| Stage C exposure/coverage | 去掉 coverage 明显异常或 event assignment 不稳定 bins |
| Stage D annulus fit quality on background control regions | 去掉背景曲面 rank 不足、condition number 过大、annulus residual 结构异常 bins |
| Reference-spectrum expected sensitivity | 仅可用于 prefit sensitivity ranking，不可使用 Crab measured excess |

### 3.3 禁止作为 selector 的量

以下只作为报告诊断，不得进入正式 include/exclude 规则：

- `N_on/B_on`
- per-cell Crab excess
- per-cell Crab significance
- Stage F pull
- Stage G SED residual
- “删掉某 cell 后 fit 变好”

如果未来确实要用 on-source 结果调 selector，必须单独标为 exploratory，并用 split-time / independent validation 验证，不能回填成 v3 baseline。

---

## 4. Stage A-G v3 pipeline

### Stage A — Response on v3 candidate grid

- 输入 `cell_ledger_v3_candidate.csv`。
- 对所有候选 cell 生成 response `eta_b(E_true, theta)` 和 `A_eff,b(E_true, theta)`。
- 输出每个 cell 的 response quality：
  - MC count
  - expected counts under reference spectra
  - true-energy containment / migration summary
  - central-99% selector flag by Nhit row
- 输出 v3 predE binning 的 MC 能量分布 overlay：
  - 横轴：`log10(E_true / GeV)` 或等价 TeV 能量轴；
  - 纵轴：每个 predE bin 内归一化后的 MC counts / livetime-equivalent counts density；
  - 每条曲线对应一个 v3 predE bin，使用不同颜色叠画在同一张图上；
  - 图中标注 `[2,2.5)`, `[2.5,3)`, `[3,5)` 0.25 dex bins, `[5,6)`, `>=6` 的 bin 边界和 median / 68% containment。
- 不在 Stage A 删除 cell，只写 metadata 和 selection inputs。

### Stage B — PSF on v3 candidate grid

- 对所有候选 cell 生成 Crab declination PSF。
- 保留：
  - `sigma_deg`
  - `r_opt_deg`
  - containment at `r_opt`
  - 68% containment
  - PSF fit quality flag
- 对 `[125,200)` 和低 reconstructed-energy bins 特别记录 PSF tail risk。
- 输出用于 Stage D annulus placement 的 `source_mask_radius_deg` 建议。

### Stage C — Observation reduction with `Nhit >= 125`

- Stage C 必须从原始观测事件重新赋予 v3 `cell_id`。
- `Nhit < 125` 直接排除，并在 metadata 记录排除计数。
- 输出 parquet 至少保留：

```text
ra_mean_deg
dec_mean_deg
mjd
theta
cell_id
nv or Nhit audit field
ml_logE_pred
source_file_id
```

- metadata 必须包含：
  - Nhit cut summary
  - out-of-ledger counts
  - per-cell event counts
  - Crab-centered `rho` coverage profile
  - annulus coverage profile for `1.5-3.5 deg` and shifted annuli

### Stage D — Annulus 2D surface background

Stage D v3 是核心变化。

#### 4.4.1 Training region

默认训练区：

```text
1.5 deg < rho < 3.5 deg
```

其中 `rho = sqrt(x^2 + y^2)`，`x = RA offset * cos(Crab Dec)`，`y = Dec offset`。

对每个 cell 定义 prefit annulus placement：

```text
source_mask_radius_b = max(1.5 deg, 2.0 * r_opt_b)
annulus_inner_b = max(1.5 deg, source_mask_radius_b + 0.2 deg)
annulus_outer_b = annulus_inner_b + 2.0 deg
```

如果 `annulus_inner_b <= 1.5 deg`，使用默认 `1.5-3.5 deg`。如果低 Nhit / 宽 PSF cell 的 `annulus_inner_b` 更大，则允许外移到 `2.0-4.0 deg` 或 `2.5-4.5 deg`。所有外移规则必须由 PSF 决定，不能由 Crab excess 决定。

#### 4.4.2 Surface model

对每个 cell 独立拟合二阶背景曲面：

```text
B(x, y) = a0 + ax*x + ay*y + axx*x^2 + axy*x*y + ayy*y^2
```

实现约束：

- training pixels 来自 annulus；
- 每个 training pixel 的观测 counts 作为拟合输入；
- 初版可用加权 least squares，权重 `1 / max(counts, 1)` 或 Poisson 方差近似；
- 背景预测必须非负，负值用质量门阻断或使用正值约束/log-link 交叉检查；
- 输出 `background_map` 覆盖 `rho < 6 deg`，核心区用 annulus 拟合曲面外推得到；
- `B_on,b` 通过把 `background_map_b` 积分到 cell 的 on aperture 得到。

#### 4.4.3 Stage D outputs

Stage D v3 NPZ 必须继续兼容 Stage E：

```text
cell_id
nhit_bin
predE_bin
r_opt_deg
sigma_deg
containment_r_opt
B_on
counts_map
background_map
excess_map
known_b_sigma_map
```

新增 fit diagnostics：

```text
annulus_inner_deg
annulus_outer_deg
surface_coefficients
surface_covariance
fit_chi2
fit_ndof
fit_condition_number
annulus_counts
annulus_pixels
annulus_residual_mean
annulus_residual_rms
core_extrapolation_warning
```

metadata：

```text
background_mode = crab_roi_local
background_form = direct_expectation
background_method = annulus_2d_quadratic_surface
li_ma_applicable = false
```

#### 4.4.4 Stage D quality gates

每个 cell 至少检查：

- annulus pixel count 足够；
- design matrix rank = 6；
- condition number 不超过阈值；
- fitted background 在 `rho < 6 deg` 内非负；
- annulus residual 没有明显 RA/Dec 单调结构；
- `B_on` 对 annulus placement 的变化不过大；
- quadratic surface 与 low-order alternatives 的差异可解释。

### Stage E — Signal table on v3 candidate grid

- 读取 Stage D v3 `B_on`。
- 对所有 candidate cells 输出：
  - `N_on`
  - `B_on`
  - excess
  - known-background Poisson diagnostic
  - conservative error
  - annulus-fit quality flags
- Stage E 不删除 cell。
- baseline aggregate 只通过 `cell_selector_v3_baseline.csv` 汇总。

### Stage F — Forward folding on v3 baseline selector

- Stage F 默认读取：

```text
response: v3 candidate response
signal: v3 candidate signal table
selector: cell_selector_v3_baseline.csv
```

- 拟合仍使用 PL 和 LogPar。
- selector 必须在 metadata 中完整记录：
  - included cells
  - excluded cells
  - exclusion source: MC / PSF / background-fit quality / systematics
- 不允许 Stage F 按 pull 自动删 cell。

### Stage G — Diagnostic SED on v3 baseline selector

- 固定 Stage F preferred shape，对 reconstructed-energy groups 拟单点 normalization。
- SED points 使用 HAWC-style energy grouping：按 `predE` / reconstructed energy group 合并贡献的 Nhit cells。
- 对高能点必须输出：
  - contributing cell list
  - effective energy from response / reference spectrum
  - TS or diagnostic significance
  - bin purity / migration warning if available

---

## 5. v3 final integrated report

新增：

```text
apply/report/crab_sed_v3_stage_a_to_g_report.html
```

报告至少包含 v2 final report 已有内容：

| v2 figure / section | v3 保留要求 |
|---|---|
| Run Summary | 保留，并增加 `v3_candidate` / `v3_baseline` 计数 |
| Stage cards A-G | 保留 |
| Stage G SED table | 保留 |
| fit-cell Stage D counts skymap | 保留，叠加 `rho=6 deg` 圈 |
| fit-cell Stage D excess skymap | 保留 |
| RA normalized counts profiles | 保留，fit cells 高亮 |
| Dec normalized counts profiles | 保留，fit cells 高亮 |
| Stage F model counts vs excess | 保留 |
| Stage F pull grid | 保留 |
| Stage G SED points | 保留 |
| Stage G SED ratios | 保留 |
| Stage G cell counts per point | 保留 |

v3 新增必备图：

| 新图 | 用途 |
|---|---|
| v3 cell selection matrix | 显示 candidate / baseline / probe / excluded cells |
| MC central-99% selection mask | 对应 HAWC-style prefit selection |
| MC normalized energy-distribution overlay | 用 MC 模拟数据按 v3 predE bin 画归一化 true-energy 分布；横轴为能量，纵轴为归一化 counts / livetime-equivalent counts density，每个 predE bin 一条不同颜色曲线 |
| annulus training mask grid | 每个 fit cell 的训练环带和 source mask |
| fitted 2D background surface grid | 展示 `background_map` |
| annulus residual grid | 检查 fit residual 是否有 RA/Dec 结构 |
| core extrapolated background grid | 展示外推到 Crab 核心区的背景 |
| before/after Dec profile comparison | 验证 Dec 方向背景不平衡是否改善 |
| background-method sensitivity summary | 比较 default annulus、shifted annulus、surface order |

---

## 6. Systematics and validation

### 6.1 Cell selection systematics

- baseline selector vs expanded selector；
- central-99% vs central-98% / central-99.5%；
- low Nhit `[125,200)` 是否进入 baseline；
- high-energy low-stat probes 是否只作为 diagnostic。

### 6.2 Background model systematics

必须至少比较：

```text
default annulus: 1.5-3.5 deg
shifted annulus: PSF-driven inner/outer radii
surface order 1: 1, x, y
surface order 2: 1, x, y, x^2, xy, y^2
Poisson log-link / positive constrained variant
```

输出对比：

- total `B_on`
- baseline excess
- Stage F PL / LogPar parameters
- Stage G high-energy SED points
- RA/Dec profiles

### 6.3 Closure tests

- 用 MC reference spectrum forward-fold 回 v3 cells；
- 检查 response-predicted counts 和 MC truth counts 的 closure；
- 用 off-source/control regions 测试 annulus 2D surface 是否给出接近零的 fake source excess；
- 对不同 time split 重复 Stage D/E，检查背景面稳定性。

---

## 7. Implementation phases

### Phase 0 — v3 design artifacts

交付：

```text
apply/report/roadmap_v3.md
apply/report/roadmap_v3.html
```

### Phase 1 — v3 cell grid and selector prototype

交付：

```text
apply/config/cell_ledger_v3_candidate.csv
apply/config/cell_selector_v3_baseline.csv
apply/config/cell_selector_v3_systematics.csv
apply/report/v3_cell_selection_diagnostics.html
```

完成标准：

- `[125,200)` 起步；
- MC central-99% selection mask 可视化；
- MC normalized energy-distribution overlay 可视化，并复用/升级已有 `plot_true_energy_distributions.py` 或 `plot_acceptable_true_energy_grid.py` 风格；
- selector 不依赖 Crab on-source results。

### Phase 2 — Stage C v3 observation reduction

完成标准：

- Stage C 可以执行 `Nhit >= 125`；
- v3 `cell_id` assignment 可审计；
- output metadata 记录 out-of-ledger 和 annulus coverage。

### Phase 3 — Stage D annulus 2D surface background

完成标准：

- 输出兼容 Stage E 的 `B_on/background_map/excess_map`；
- 每个 fit cell 有 annulus mask、surface、residual、quality metadata；
- low Nhit broad-PSF cells 自动使用 PSF-driven annulus placement。

### Phase 4 — Stage E/F/G v3 full chain

完成标准：

- v3 candidate signal table；
- v3 baseline forward-folding fit；
- v3 SED points；
- high-energy points 输出 contributing cells 和 migration warnings。

### Phase 5 — v3 integrated report and cross-checks

完成标准：

- `crab_sed_v3_stage_a_to_g_report.html` 包含 v2 全部关键图；
- 新增 annulus 2D surface diagnostics；
- 新增 MC normalized energy-distribution overlay，说明 v3 predE 分箱对应的 true-energy 覆盖和重叠；
- 报告中明确列出 selector freeze audit 和 background systematics。

---

## 8. 当前已确认决策与待验证问题

以下是 2026-06-12 已确认或仍需实现前验证的点：

1. **Stage C 原始 Nhit 字段：已接受推荐。** 从 ROOT/recovered event source 重新生成 Stage C v3，并在 parquet 或 metadata 中保留 Nhit audit 字段。
2. **predE binning：已更新推荐。** `[2,3)` 一整 dex 太宽，但拆到 0.25 dex 对低能端可能过细；v3 推荐拆成 `[2,2.5)` 和 `[2.5,3)` 两个 0.5 dex 候选 bins。`[3,5)` 的 0.25 dex binning 合理，`[4,5)` 已经细分，不需要额外拆。`>=5` 统计量低，v3 推荐拆成 `[5,6)` 和 `>=6` 两个高能宽 bins；`>=6` 主要作为 upper-limit / diagnostic 候选，不默认要求显著探测。
3. **annulus 外移上限：已接受推荐。** 默认允许到 `4.5 deg`，必要时到 `5.0 deg`，但必须保持在 `rho < 6 deg` fiducial coverage 内，并写入 systematics。
4. **nominal surface fit 统计形式：已接受推荐。** v3.0 用二阶加权 least squares 快速闭环，同时输出 Poisson log-link cross-check 作为 v3.1 或 systematics。
