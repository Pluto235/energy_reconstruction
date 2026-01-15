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

def plot_bias(ture_E, pred_E, bins=20, skip=0 ,save_name=None):
    

def plot_log_RMSerror(true_E, pred_E, bins=20, skip=0, save_name=None):
    '''
    输出的是log下的resolution: std(pred_E - true_E) in log scale
    
    :param true_E: 真实能量值log
    :param pred_E: 预测能量值log
    :param bins: 能量分bin
    :param skip: 选择跳过前几个bin
    :param save_name: 保存名字
    '''
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



def plot_pred_vs_true_heatmap(true, pred, bins=120, save_name=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from matplotlib.colors import LogNorm
    from datetime import datetime

    """
    使用二维直方图绘制 Pred vs True 能量响应矩阵（logE–logE）
    true, pred: log10(E / GeV)
    """

    true = np.asarray(true)
    pred = np.asarray(pred)

    # 可选：限制能量范围（防止极端outlier拉伸colorbar）
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]

    # 2D histogram（在 logE 空间）
    H, xedges, yedges = np.histogram2d(
        true, pred,
        bins=bins
    )

    plt.figure(figsize=(6, 6))

    im = plt.pcolormesh(
        xedges, yedges, H.T,
        norm=LogNorm(vmin=1, vmax=H.max()),
        cmap="viridis"
    )

    # y = x 参考线
    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)

    plt.xlabel(r"log$_{10}$(E$_{\mathrm{true}}$ / GeV)")
    plt.ylabel(r"log$_{10}$(E$_{\mathrm{pred}}$ / GeV)")
    plt.title("Energy Response Matrix")

    cbar = plt.colorbar(im)
    cbar.set_label("Counts")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存
    save_dir = "/home/server/projects/energy_reconstruction/fig/"
    os.makedirs(save_dir, exist_ok=True)

    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"pred_vs_true_heatmap_{timestamp}.png"

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Energy response heatmap 已保存到: {save_path}")



def plot_pred_distributions_in_true_bins(true_logE, pred_logE,
                                         Emin=1e2, Emax=1e6, nbins_true=10,
                                         bins_pred=60, use_delta=False,
                                         save_name=None):
    """
    true_logE, pred_logE: log10(E/GeV)
    nbins_true: true energy 分10个log bin（100-1e6 GeV）
    use_delta:
        False -> 画 pred_logE 分布
        True  -> 画 ΔlogE = pred_logE - true_logE 分布（更像高斯，直接给分辨率）
    运用所有数据
        bias: mu 即mean(x)
        resolution: sigma 即std(x)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    true_logE = np.asarray(true_logE)
    pred_logE = np.asarray(pred_logE)
    m = np.isfinite(true_logE) & np.isfinite(pred_logE)
    true_logE = true_logE[m]
    pred_logE = pred_logE[m]

    # true energy 的log-bin边界：100->1e6 GeV 对应 log10: 2->6
    edges_log = np.linspace(np.log10(Emin), np.log10(Emax), nbins_true + 1)

    # 画图：每个bin一个子图
    ncols = 5
    nrows = int(np.ceil(nbins_true / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 3.0*nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    stats = []  # 存每个bin的均值/方差等

    for i in range(nbins_true):
        lo, hi = edges_log[i], edges_log[i+1]
        sel = (true_logE >= lo) & (true_logE < hi)
        ax = axes[i]

        if sel.sum() < 10:
            ax.text(0.5, 0.5, f"Too few events\nN={sel.sum()}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]")
            continue

        if use_delta:
            x = pred_logE[sel] - true_logE[sel]          # ΔlogE
            x_label = r"$\Delta \log_{10}E$  (pred-true)"
            ref = 0.0
        else:
            x = pred_logE[sel]                           # pred_logE
            x_label = r"$\log_{10}(E_{\mathrm{pred}}/\mathrm{GeV})$"
            ref = 0.5*(lo+hi)                            # 参考：bin中心(log)

        mu = np.mean(x)
        sig = np.std(x)

        ax.hist(x, bins=bins_pred, histtype="step", linewidth=1.5, density=True)
        ax.axvline(mu, linestyle="--", linewidth=1) # 能量bin里预测能量的均值
        ax.axvline(ref, linestyle=":", linewidth=1) # log下bin的中心

        ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]  N={sel.sum()}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("PDF")

        # 记录统计量
        stats.append({
            "bin": i,
            "logE_true_lo": lo,
            "logE_true_hi": hi,
            "N": int(sel.sum()),
            "mu": float(mu),
            "sigma": float(sig),
            "ref": float(ref),
        })

        # 在图上写数值
        ax.text(0.03, 0.95, f"μ={mu:.3f}\nσ={sig:.3f}",
                transform=ax.transAxes, va="top")

    # 多余子图关掉
    for j in range(nbins_true, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Predicted energy distribution in true-energy bins", y=1.02)
    plt.tight_layout()

    # 保存
    save_dir = "/home/server/projects/energy_reconstruction/fig/"
    os.makedirs(save_dir, exist_ok=True)

    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "delta" if use_delta else "predlogE"
        save_name = f"pred_dist_in_true_bins_{tag}_{timestamp}.png"

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 分布图已保存到: {save_path}")
    return stats


def plot_pred_distributions_in_true_bins68(true_logE, pred_logE,
                                         Emin=1e2, Emax=1e6, nbins_true=10,
                                         bins_pred=60, use_delta=False,
                                         save_name=None):
    """
    true_logE, pred_logE: log10(E/GeV)
    nbins_true: true energy 分10个log bin（100-1e6 GeV）
    use_delta:
        False -> 画 pred_logE 分布
        True  -> 画 ΔlogE = pred_logE - true_logE 分布（更像高斯，直接给分辨率）
    统计量采用鲁棒的 68% containment:
        bias = median(x)
        resolution = (q84 - q16)/2
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    true_logE = np.asarray(true_logE)
    pred_logE = np.asarray(pred_logE)
    m = np.isfinite(true_logE) & np.isfinite(pred_logE)
    true_logE = true_logE[m]
    pred_logE = pred_logE[m]

    # true energy 的log-bin边界：100->1e6 GeV 对应 log10: 2->6
    edges_log = np.linspace(np.log10(Emin), np.log10(Emax), nbins_true + 1)

    # 画图：每个bin一个子图
    ncols = 5
    nrows = int(np.ceil(nbins_true / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 3.0*nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    stats = []

    for i in range(nbins_true):
        lo, hi = edges_log[i], edges_log[i+1]
        sel = (true_logE >= lo) & (true_logE < hi)
        ax = axes[i]

        N = int(sel.sum())
        if N < 20:
            ax.text(0.5, 0.5, f"Too few events\nN={N}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]")
            continue

        if use_delta:
            x = pred_logE[sel] - true_logE[sel]          # ΔlogE
            x_label = r"$\Delta \log_{10}E$  (pred-true)"
            ref = 0.0
        else:
            x = pred_logE[sel]                           # pred_logE
            x_label = r"$\log_{10}(E_{\mathrm{pred}}/\mathrm{GeV})$"
            ref = 0.5*(lo+hi)                            # true bin中心(log)

        # ====== 68% containment 统计量（鲁棒）======
        # bias 用 median，resolution 用 (q84-q16)/2
        med = float(np.median(x))
        q16, q84 = np.percentile(x, [16, 84])
        q16 = float(q16)
        q84 = float(q84)
        sig68 = 0.5 * (q84 - q16)

        # 画直方图
        ax.hist(x, bins=bins_pred, histtype="step", linewidth=1.5, density=True)

        # 画参考线
        # ax.axvline(med, linestyle="--", linewidth=1.2, label="median")
        # ax.axvline(ref, linestyle=":", linewidth=1.2, label="ref")
        ax.axvspan(q16, q84, color="gray", alpha=0.3, label="68% containment")

        # 画 68% containment 区间（可选但很直观）
        ax.axvline(q16, linestyle="-.", linewidth=1.0)
        ax.axvline(q84, linestyle="-.", linewidth=1.0)

        ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]  N={N}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("PDF")

        # 保存统计量（字段名沿用你之前的 mu/sigma，便于后续画曲线不改太多）
        stats.append({
            "bin": i,
            "logE_true_lo": float(lo),
            "logE_true_hi": float(hi),
            "N": N,

            # 这里的 mu/sigma 现在是“鲁棒定义”
            "mu": med,               # bias: median(x)
            "sigma": float(sig68),   # resolution: 68% containment half-width

            "q16": q16,
            "q84": q84,
            "ref": float(ref),
        })

        # 图中文字
        ax.text(0.03, 0.95, f"med={med:.3f}\nσ68={sig68:.3f}",
                transform=ax.transAxes, va="top")

    # 多余子图关掉
    for j in range(nbins_true, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Predicted energy distribution in true-energy bins (68% containment)", y=1.02)
    plt.tight_layout()

    # 保存
    save_dir = "/home/server/projects/energy_reconstruction/fig/"
    os.makedirs(save_dir, exist_ok=True)

    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "delta" if use_delta else "predlogE"
        save_name = f"pred_dist_in_true_bins_{tag}_contain68_{timestamp}.png"

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 分布图已保存到: {save_path}")
    return stats

  