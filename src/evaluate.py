import torch
import copy
import numpy as np
from sklearn.metrics import r2_score
import psutil
from .utils import plot_resolution, plot_pred_vs_true

# 直接加载best_model_2进行评估，不需要进行训练
def evaluate_model(model, test_loader, checkpoint_path=None):
    """
    评估 ParticleNet 模型性能（针对 log(E) 回归任务）
    
    模型输出和标签均为 log(E)，评估时也是 log。
    输出指标：
        - 能量分辨率 σ = std((E_pred - E_true)/E_true)
        - 能量偏差 bias = mean((E_pred - E_true)/E_true)

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # === 加载保存的模型参数 ===
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 兼容 DataParallel 保存的权重 在多GPU上训练，在单GPU上评估时遇到
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        print("⚙️ 检测到 DataParallel 模型，已移除 'module.' 前缀")
        
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"✅ 成功加载模型权重: {checkpoint_path}")

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

            # 反 log 得到能量
            # E_pred = torch.pow(10, logE_pred).cpu().numpy().flatten()
            # E_true = torch.pow(10, logE_true).cpu().numpy().flatten()

            predictions.extend(logE_pred.detach().cpu().numpy())
            true_energies.extend(logE_true.detach().cpu().numpy())

    predictions = np.array(predictions)
    true_energies = np.array(true_energies)

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

    if checkpoint_path is None:
        print("⚠️ 没有保存模型的路径")
    
    plot_resolution(true_energies, predictions)
    plot_pred_vs_true(true_energies, predictions)

    return predictions, true_energies # in log scale