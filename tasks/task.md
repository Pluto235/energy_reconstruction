# Task1
项目路径：
/home/server/projects/energy_reconstruction

任务：
做一个“严格筛选 vs 放开筛选”的对比实验，并提交运行。

背景：
我现在常用的数据筛选条件是：

- 真实能量 > 100 GeV
- fitstat = 0
- dcedge_min = 20
- dangle_max_deg = 3
- theta_max_deg = 30
- pinc_max = 1.1

现在我要做一个对比实验：

实验1：严格筛选
使用上面这整套筛选条件

实验2：放开筛选
尽可能放开这些筛选条件，看看训练效果差别

放开筛选建议：
- 不限制 fitstat
- dcedge_min = 0
- dangle_max_deg 设成很大，例如 999
- theta_max_deg 放宽到 50（或如果代码允许，也可以更大）
- pinc_max 设成很大，例如 9999
- Emin 保持为 100 GeV，除非你判断也应该一起放开；如果改动，请明确说明

要求：

1. 先确认代码里这些筛选参数目前分别对应什么命令行参数/配置项
尤其确认：
- fitstat 筛选是否已经支持开关
- Emin
- dcedge_min
- dangle_max_deg
- theta_max_deg
- pinc_max

2. 新建 sbatch
在 scripts/slurm/ 下创建一个新的 sbatch，例如：
allcuts_compare.sbatch

3. 在同一个 sbatch 中依次运行两个实验

实验1 run 名：
allcuts_strict

实验2 run 名：
allcuts_relaxed

4. 训练参数
除筛选条件外，其他训练参数保持和最近成功实验一致。

5. 提交运行
用 sbatch 提交这个脚本。

6. 最后告诉我：
- 新建的 sbatch 文件路径
- slurm job id
- 两个实验的 run 目录
- 使用的具体筛选参数
- slurm 日志路径
- 是否成功启动训练

注意：
- 不要覆盖已有 runs
- 优先复用现有成功 sbatch 的写法
- 如果“放开筛选”里的某一项在当前代码中不能直接关闭，请选择最合理的近似放宽方式，并告诉我

# Task2
项目路径：
/home/server/projects/energy_reconstruction

任务：
针对一个已经训练好的模型，只在“评估阶段”放宽数据筛选条件，比较评估结果变化。

模型路径：
/home/server/projects/energy_reconstruction/runs/fitstat0_2727/checkpoints/best_model.pt

对应 run_dir：
/home/server/projects/energy_reconstruction/runs/fitstat0_2727

背景：
这个模型训练时使用了严格筛选条件，训练后的评估也用了相同筛选。
现在要做的是：保持模型不变，只在评估时放宽部分筛选条件，看看结果是否明显变差。

需要做的对比：

1. dcedge
- 基线评估：dcedge_min = 20
- 放宽评估：dcedge_min = 0

2. dangle
- 基线评估：dangle_max_deg = 3
- 放宽评估：dangle_max_deg = 999

3. pincness
- 基线评估：pinc_max = 1.1
- 放宽评估：pinc_max = 9999

要求：

1. 写一个正式的 Python 评估脚本
- 放在合适位置，例如：
  src/theta/eval_compare_relaxed.py
- 输入：
  - run_dir
  - 可选 out_dir_name
  - 评估 override 参数
- 自动读取：
  - run_dir/config.json
  - run_dir/checkpoints/best_model.pt
- 保持模型和训练配置不变，只覆盖评估数据筛选条件
- 对每一种放宽条件分别生成一个独立评估输出目录，例如：
  - fig_eval_dcedge20_baseline
  - fig_eval_dcedge0_relaxed
  - fig_eval_dangle3_baseline
  - fig_eval_dangle_relaxed
  - fig_eval_pinc1p1_baseline
  - fig_eval_pinc_relaxed

2. 复用当前 theta 主线代码
- 尽量复用已有 dataset / evaluate / evaluate_only 逻辑
- 不要重写整套评估流程
- 如果当前代码不支持“只在评估时 override cut”，就最小修改实现这个能力

3. 输出内容
每次评估至少要生成：
- metrics.json
- preds.npz
- 评估图（沿用现有输出）
- 一个 effective_eval_config.json，记录这次评估实际使用的参数

4. 写一个 sbatch 脚本并提交
- 放到：
  scripts/slurm/
- 文件名例如：
  eval_relaxed_cuts_fitstat0_2727.sbatch
- 在一个 sbatch 里依次跑：
  - dcedge baseline + relaxed
  - dangle baseline + relaxed
  - pincness baseline + relaxed

5. 提交运行
- 用 sbatch 提交
- 检查作业是否成功启动
- 如果失败，查看日志并修到至少能正常开始评估

6. 最后告诉我：
- 新写的 Python 脚本路径
- 新写的 sbatch 路径
- slurm job id
- 每个评估输出目录路径
- 修改了哪些代码
- 是否成功运行
- 如果有失败，失败在哪一步

# Task 3
项目路径：
/home/server/projects/energy_reconstruction

在你已经完成“只在评估阶段放宽 cuts”的脚本和 sbatch 之后，继续做一件事：

任务：
把三个变量的 baseline / relaxed 评估结果自动汇总到一个 notebook 里，方便直接比较。

背景：
现在有三组评估对比：

1. dcedge
- baseline: dcedge_min = 20
- relaxed: dcedge_min = 0

2. dangle
- baseline: dangle_max_deg = 3
- relaxed: dangle_max_deg = 999

3. pincness
- baseline: pinc_max = 1.1
- relaxed: pinc_max = 9999

要求：

1. 新建 notebook
在 notebook/ 下创建一个新的 ipynb，例如：
eval_relaxed_cuts_fitstat0_2727.ipynb

2. 自动读取结果
从这次评估生成的输出目录中自动读取结果，不要手工抄数据。
优先使用：
- preds.npz
- metrics.json
- 以及现有评估图所依赖的数据

3. notebook 需要做的图
对每一组变量（dcedge / dangle / pincness），分别把 baseline 和 relaxed 画在同一张图上比较：

- resolution_weighted
- logRMS_weighted
- bias_weighted

也就是说，至少要生成 3 组 × 3 张 = 9 张图。

4. notebook 内容要求
- 开头说明这个 notebook 分析的是哪个模型：
  fitstat0_2727
- 写清楚 baseline 和 relaxed 分别代表什么
- 每段代码前补简短注释
- 自动识别并显示每组评估结果对应的目录
- 图例清楚，标题和坐标轴写清楚
- 保持 notebook 整洁，不要太乱

5. 路径适配
适配当前项目结构：
- src/common
- src/theta
- notebook
- runs

如果需要 import 项目代码，请使用当前路径结构。

6. 最后告诉我：
- 新建 notebook 的路径
- 读取了哪些评估输出目录
- 是否成功生成 9 张对比图
- 如果某些评估目录没找到，请明确说明缺了哪些