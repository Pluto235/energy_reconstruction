# ETO 推理环境 — 2026-10-14

项目根目录：`/mnt/newdisk/shijie/energy`。复用已有 Anaconda；不使用 sudo，不修改系统 Conda，不需要另装 Miniconda。

## 路径与激活

- Conda 24.1.2：`/opt/anaconda3/bin/conda`
- Python：`/mnt/newdisk/shijie/conda/envs/py310/bin/python`
- 环境目录：`/mnt/newdisk/shijie/conda/envs/py310`
- Conda 包缓存：`/mnt/newdisk/shijie/conda/pkgs`
- pip 缓存：`/mnt/newdisk/shijie/conda/pip-cache`

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/newdisk/shijie/conda/envs/py310
```

使用完整环境路径激活，不依赖旧的 `/home/server/anaconda3` 或环境名搜索配置。

## 安装与兼容性

ETO 有两张 Tesla V100-SXM2-16GB（计算能力 7.0），驱动 550.163.01，`nvidia-smi` 报告 CUDA 12.4。选择 PyTorch 2.6.0+cu124，CUDA 运行库随 wheel 安装，无需安装系统 CUDA Toolkit。

版本选择参照 [PyTorch 官方历史版本安装说明](https://docs.pytorch.org/get-started/previous-versions/) 和 [NVIDIA CUDA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。实际可用性由下述 Slurm 验证确认。

创建环境与安装命令（仅在需要重建时执行）：

```bash
mkdir -p /mnt/newdisk/shijie/conda/envs /mnt/newdisk/shijie/conda/pkgs
CONDA_PKGS_DIRS=/mnt/newdisk/shijie/conda/pkgs \
CONDA_ENVS_PATH=/mnt/newdisk/shijie/conda/envs \
/opt/anaconda3/bin/conda create -y \
  -p /mnt/newdisk/shijie/conda/envs/py310 python=3.10 pip
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/newdisk/shijie/conda/envs/py310
export PIP_CACHE_DIR=/mnt/newdisk/shijie/conda/pip-cache
cd /mnt/newdisk/shijie/energy
python -m pip install 'torch==2.6.0+cu124' \
  --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt awkward \
  -c apply/config/eto_py310_constraints.txt
python -m pip check
```

项目 `requirements.txt` 保留原样；`awkward` 显式安装，约束文件固定本机适配的 Torch 构建。其他包按 requirements 解析，完整安装快照保存在 ETO `logs/` 中，不作为跨平台锁文件使用。

## Slurm 轻量验证

```bash
mkdir -p /mnt/newdisk/shijie/energy/logs/slurm
cd /mnt/newdisk/shijie/energy/apply
sbatch tools/check_eto_environment.sbatch
```

资源：`main` 分区，2 GPU / 2 CPU / 4 GB，最长 5 分钟，从 `energy/apply` 启动。按当前两张卡配置，脚本执行 `pip check`、全部项目依赖和 awkward 导入，分别在两张卡上执行 CUDA 小矩阵运算，并调用 `apply.apply_observation_energy.load_run_config` 和 `build_model` 严格加载以下配置与权重到每张 GPU：

```text
runs/theta_recoxy_position_embed_noemax_no_core_cut_64748/config.json
runs/theta_recoxy_position_embed_noemax_no_core_cut_64748/checkpoints/best_model.pt
```

检查模型处于 eval 模式、所有参数位于 CUDA 且有限，以及配置／权重加载前后 SHA-256 一致。只输出日志，不读取观测数据，不调用模型 forward，不训练、不运行整月任务。

### 本次实测

Slurm **66040**：2026-10-14 01:03:42–01:03:47（Asia/Shanghai），`COMPLETED`，退出码 `0:0`，耗时 5 秒。两张 V100 分别通过 CUDA 运算和现有严格权重加载；全部依赖导入成功，`pip check` 无冲突，stderr 为空。首轮单卡验证 66039 同样通过。

| 组件 | 已安装版本 |
|---|---|
| Python | 3.10.22 |
| PyTorch / CUDA runtime | 2.6.0+cu124 / 12.4 |
| uproot / awkward | 5.7.7 / 2.14.0 |
| numpy / pandas | 2.2.6 / 2.3.3 |
| scikit-learn / matplotlib | 1.7.2 / 3.10.9 |
| pyarrow / iminuit | 25.0.1 / 2.33.0 |
| tqdm / psutil | 4.70.1 / 7.2.2 |

ETO 验证日志：`logs/slurm/energy_env_check_66040.out` 和 `.err`。安装快照：`logs/eto_py310_2026-10-14.freeze.txt`、`logs/eto_py310_2026-10-14.conda-explicit.txt`。

配置和权重在验证前后保持以下 SHA-256：

```text
config.json: cc0d3a57fdc047bd9e0170a996d4d35808e636d2a882402c7e91ed3b84a8e8af
best_model.pt: 0934001d358b4b0a9b9100977df8e4259df16e69894caafb12137b2877f53a64
```

## 使用边界

64748 的配置由历史快照恢复，严格加载成功只能证明结构与权重匹配，不能证明物理结果正确。历史配置内的输入路径未改写。旧生产 sbatch 仍包含旧 Conda／项目路径及 4 GPU 请求；当前机器只有 2 GPU，后续生产应申请 `--gres=gpu:2` 并使用 `--gpu-ids 0,1`，同时核对新路径和输入输出位置。此次只新增环境约束和轻量验证入口，不修改或启动生产脚本。
