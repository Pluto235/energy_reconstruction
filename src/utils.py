import torch

def check_gpu_status():
    """检查GPU状态"""
    print("=" * 50)
    print("GPU状态检查")
    print("=" * 50)
    
    # 检查CUDA是否可用
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 检查GPU数量
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        # 检查每个GPU的状态
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  内存缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        
        # 检查当前设备
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("❌ 没有检测到可用的GPU")
    
    print("=" * 50)



def plot_resolution(true_E, pred_E, bins=20, skip=0, save_name=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    true_E = np.array(true_E)
    pred_E = np.array(pred_E)

    # 计算 log 能量 bin
    bin_edges = np.logspace(np.log10(true_E.min()), np.log10(true_E.max()), bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    resolution = []
    for i in range(bins):
        mask = (true_E >= bin_edges[i]) & (true_E < bin_edges[i + 1])
        if mask.sum() > 10:
            rel = pred_E[mask] - true_E[mask]
            resolution.append(np.std(rel))
        else:
            resolution.append(np.nan)

    # 剔除前 skip 个 bin
    bin_centers = bin_centers[skip:]
    resolution = resolution[skip:]

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, resolution, 'o-')
    plt.xlabel("True Energy (log scale)")
    plt.ylabel("Resolution (std of residual)")
    plt.title("Energy Resolution vs True Energy")
    plt.tight_layout()

    # 保存图片
    save_dir = "/home/server/projects/energy_reconstruction/fig/"
    os.makedirs(save_dir, exist_ok=True)

    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"resolution_{timestamp}.png"

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Resolution 图像已保存到: {save_path}")

    plt.show()



def plot_pred_vs_true(true, pred, log_scale=True, save_name=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime

    """
    绘制预测能量与真实能量的散点对比图（支持 log–log 形式）
    
    参数:
        true (array-like): 真实能量（或 logE）
        pred (array-like): 预测能量（或 logE）
        log_scale (bool): 是否使用对数坐标（默认 True）
        title (str): 图标题
        save_name (str): 文件名（例如 "pred_vs_true.png"），不含路径
    """
    true = np.array(true)
    pred = np.array(pred)

    true = 10 ** true
    pred = 10 ** pred

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, s=6, alpha=0.4, label="Events")

    mn = min(true.min(), pred.min())
    mx = max(true.max(), pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='y = x')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel('True Energy [GeV]')
    plt.ylabel('Predicted Energy [GeV]')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.title("Predicted vs True Energy (log-log)")
    plt.tight_layout()

    # 自动保存路径
    save_dir = "/home/server/projects/energy_reconstruction/fig/"
    os.makedirs(save_dir, exist_ok=True)

    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"pred_vs_true_{timestamp}.png"

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ LogE_pred vs logE_true 图像已保存到: {save_path}")

