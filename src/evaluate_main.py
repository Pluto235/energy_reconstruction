# 评估模型主函数
import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.abspath(".."))  # 添加上一级目录

from src import utils
from src.evaluate import evaluate_model
from src.ParticleDataset import ParticleDataset
from src.ParticleRegressor import ParticleNetRegressor
import random
# 主函数
def main():
    """
    使用已训练好的模型评估能量重建性能，
    并提取预测集中在 logE ∈ [2,3] 的事件保存到 pd.DataFrame。
    """
   
	 # ======== 读取数据文件夹 ========
    root_path = "/home/server/mydisk/WCDA_simulation/" # WCDA_simulation folder
    file_path = []
    for filename in os.listdir(root_path):
           full_path = os.path.join(root_path, filename)
           file_path.append(full_path)
    
	 # 随机提取100个进行评估
    # test_files = random.sample(file_path, 1000) # 500*20%
    test_files = file_path[:-2000]
    print(f"📁 本次随机选取使用 {len(test_files)} 个ROOT文件数据进行评估")

	 # ======== 预处理条件 ========       
    processing_conditions = [
            {"name": "vx",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0},
            {"name": "vy",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0},  
            {"name": "vt",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0},
            {"name": "vq",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0}
        ]
    branches = [d["name"] for d in processing_conditions] # 提取name列表，便于后续读取相关的值 ，输入特征
    target_branch = ["mc_energy"] # 定义一个标签列表， 预测目标是能量
    processing_conditions = [
        {k: v for k, v in d.items() if k != "name"} # 删除掉name字段，只保留预处理操作字段
        for d in processing_conditions
    ]
    
   #  # ======== 先检查GPU状态 ========
    utils.check_gpu_status()
   
    # # ======== 只构建test数据集 ========
    batch_size = 256
    test_dataset = ParticleDataset(
        root_files=test_files,
        branches=branches,
        target_branch=target_branch,
        processing_conditions=processing_conditions,
        max_points=500
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ======== 初始化模型 ========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = ParticleNetRegressor(
        input_dims=2,  # vq, vt 两个特征
        conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
        fc_params=[(256, 0.1), (128, 0.1)],
        use_fusion=True
    )
    
    # === 加载保存的模型参数 ===
    print("评估模型...")
    ckpt_path = "/home/server/projects/energy_reconstruction/best_model_full_0104.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    # 兼容 DataParallel 保存的权重 在多GPU上训练，在单GPU上评估时遇到
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        print("⚙️ 检测到 DataParallel 模型，已移除 'module.' 前缀")
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"✅ 成功加载模型权重: {ckpt_path}")
    
	 # === 进行评估 ===
    predictions = []
    true_energies = []
    with torch.no_grad():
        for points, features, mask, logE_true in test_loader:
            points = points.to(device)
            features = features.to(device)
            mask = mask.to(device)
            logE_true = logE_true.to(device)

            # 模型预测 log(E)
            logE_pred = model(points, features, mask)


            predictions.extend(logE_pred.detach().cpu().numpy())
            true_energies.extend(logE_true.detach().cpu().numpy())

    predictions = np.array(predictions).flatten()
    true_energies = np.array(true_energies).flatten() # in log scale

    # ===== 异常值检查 =====
    if len(predictions) == 0 or np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        print("⚠️ 预测结果中存在异常值")
        return None, None

    # ===== 计算性能指标 =====
    rel_error = (predictions - true_energies)
    resolution = np.std(rel_error)
    bias = np.mean(rel_error)
    # r2 = r2_score(true_energies, predictions)

    print("\n=== 模型评估结果 ===")
    print(f"能量分辨率 (resolution): {resolution:.4f}")
    print(f"能量偏差 (bias): {bias:.4f}")
    # print(f"R² Score: {r2:.4f}")

    utils.plot_resolution(true_energies, predictions)
    utils.plot_pred_vs_true_heatmap(true_energies, predictions)

    # true_logE, pred_logE 都是 log10(E/GeV)
    stats1 = utils.plot_pred_distributions_in_true_bins(true_energies, predictions, use_delta=True)
    stats2 = utils.plot_pred_distributions_in_true_bins68(true_energies, predictions, use_delta=True)  # 推荐：直接看分辨率
    
    return stats1, stats2


def plot_resolution_vs_energy(stats,  # 用 stats2（delta-logE）
                              Emin=1e2, Emax=1e6,
                              use_fractional=True,
                              save_name=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    # 从stats里取 bin center（用true能量bin中心）
    log_lo = np.array([s["logE_true_lo"] for s in stats], dtype=float)
    log_hi = np.array([s["logE_true_hi"] for s in stats], dtype=float)
    log_center = 0.5 * (log_lo + log_hi)
    E_center = 10 ** log_center  # GeV

    mu = np.array([s["mu"] for s in stats], dtype=float)      # ΔlogE mean
    sig = np.array([s["sigma"] for s in stats], dtype=float)  # ΔlogE std
    N = np.array([s.get("N", 0) for s in stats], dtype=float)

    # 过滤无效bin
    ok = np.isfinite(E_center) & np.isfinite(sig) & (N > 10) & (sig > 0)
    E_center = E_center[ok]
    mu = mu[ok]
    sig = sig[ok]
    N = N[ok]

    # 分辨率定义
    if use_fractional:
        # σE/E ≈ ln(10) * σ(Δlog10E) 线性分辨率
        y = np.log(10.0) * sig
        y_label = r"Energy resolution  $\sigma_E/E \approx \ln(10)\,\sigma_{\Delta\log_{10}E}$"
        title = "Energy resolution vs True Energy"
    else:
        # 直接画 σ(Δlog10E) log分辨率
        y = sig
        y_label = r"$\sigma(\Delta \log_{10}E)$"
        title = r"Log-energy resolution vs True Energy"

    # 误差条（粗略）：sigma 的统计误差 ~ sigma/sqrt(2(N-1))
    yerr = y / np.sqrt(2 * (N - 1))

    plt.figure(figsize=(6, 4.8))
    plt.errorbar(E_center, y, yerr=yerr, fmt="o", capsize=3)

    plt.xscale("log")
    plt.xlabel("True Energy bin center [GeV]")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    # 保存
    save_dir = "/home/server/projects/energy_reconstruction/fig/"
    os.makedirs(save_dir, exist_ok=True)
    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "frac" if use_fractional else "log"
        save_name = f"resolution_vs_energy_{tag}_{timestamp}.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 分辨率曲线已保存到: {save_path}")


if __name__ == "__main__":
     stats1, stats2 = main()
     plot_resolution_vs_energy(stats2, use_fractional=True)  # 推荐：σE/E
     plot_resolution_vs_energy(stats2, use_fractional=False) # 也可以：σ(ΔlogE)

    