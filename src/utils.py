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


import os
from datetime import datetime

_DEFAULT_FIG_DIR = "/home/server/projects/energy_reconstruction/fig/"

def _resolve_save_path(out_dir, save_name, prefix):
    save_dir = out_dir if out_dir is not None else _DEFAULT_FIG_DIR
    os.makedirs(save_dir, exist_ok=True)

    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{prefix}_{timestamp}.png"

    return os.path.join(save_dir, save_name)

import numpy as np

def _weighted_mean(x, w):
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.sum(w)
    if s <= 0:
        return np.nan
    return np.sum(w * x) / s

def _weighted_var(x, w):
    m = _weighted_mean(x, w)
    if not np.isfinite(m):
        return np.nan
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.sum(w)
    if s <= 0:
        return np.nan
    return np.sum(w * (x - m) ** 2) / s

def _weighted_std(x, w):
    v = _weighted_var(x, w)
    return np.sqrt(v) if np.isfinite(v) and v >= 0 else np.nan

def _weighted_rms(x, w):
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.sum(w)
    if s <= 0:
        return np.nan
    return np.sqrt(np.sum(w * x**2) / s)

def plot_bias(true_E, pred_E, weights=None, bins=20, skip=0, save_name=None, out_dir=None, space="log"):
    """
    Energy bias:
      b = < logE_pred - logE_true >   (per true-E bin)

    weights:
      - None: unweighted mean
      - array: weighted mean in each bin
    """
    import numpy as np
    import matplotlib.pyplot as plt

    true_E = np.asarray(true_E).squeeze()
    pred_E = np.asarray(pred_E).squeeze()
    if weights is not None:
        weights = np.asarray(weights).squeeze()

    if space != "log":
        raise ValueError("plot_bias currently implemented in log space. Use space='log' or implement linear branch.")

    # valid
    if weights is None:
        m0 = np.isfinite(true_E) & np.isfinite(pred_E)
    else:
        m0 = np.isfinite(true_E) & np.isfinite(pred_E) & np.isfinite(weights) & (weights > 0)

    true_E = true_E[m0]
    pred_E = pred_E[m0]
    if weights is not None:
        weights = weights[m0]

    bin_edges = np.linspace(true_E.min(), true_E.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bias = []
    for i in range(bins):
        m = (true_E >= bin_edges[i]) & (true_E < bin_edges[i + 1])
        if m.sum() > 10:
            residual = pred_E[m] - true_E[m]
            if weights is None:
                bias.append(float(np.mean(residual)))
            else:
                bias.append(float(_weighted_mean(residual, weights[m])))
        else:
            bias.append(np.nan)

    bin_centers = bin_centers[skip:]
    bias = np.asarray(bias)[skip:]

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, bias, "o-", lw=1.8)
    plt.axhline(0, color="gray", ls="--", lw=1)
    plt.xlabel(r"True Energy $\log_{10}(E/\mathrm{GeV})$")
    plt.ylabel("Bias")
    title = "Energy Bias vs True Energy" + ("" if weights is None else " (mc-weighted)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prefix = "bias" if weights is None else "bias_weighted"
    save_path = _resolve_save_path(out_dir, save_name, prefix=prefix)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Bias 图像已保存到: {save_path}")


def plot_resolution(true_E, pred_E, weights=None, bins=20, skip=0, save_name=None, out_dir=None, space="log"):
    """
    Energy resolution (your definition):
      resolution = std(logE_pred)  (at fixed true energy)

    weights:
      - None: unweighted std
      - array: weighted std in each bin
    """
    import numpy as np
    import matplotlib.pyplot as plt

    true_E = np.asarray(true_E).squeeze()
    pred_E = np.asarray(pred_E).squeeze()
    if weights is not None:
        weights = np.asarray(weights).squeeze()

    if space != "log":
        raise ValueError("plot_resolution currently implemented in log space. Use space='log' or implement linear branch.")

    if weights is None:
        m0 = np.isfinite(true_E) & np.isfinite(pred_E)
    else:
        m0 = np.isfinite(true_E) & np.isfinite(pred_E) & np.isfinite(weights) & (weights > 0)

    true_E = true_E[m0]
    pred_E = pred_E[m0]
    if weights is not None:
        weights = weights[m0]

    bin_edges = np.linspace(true_E.min(), true_E.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    resolution = []
    for i in range(bins):
        m = (true_E >= bin_edges[i]) & (true_E < bin_edges[i + 1])
        if m.sum() > 10:
            if weights is None:
                resolution.append(float(np.std(pred_E[m])))
            else:
                resolution.append(float(_weighted_std(pred_E[m], weights[m])))
        else:
            resolution.append(np.nan)

    bin_centers = bin_centers[skip:]
    resolution = np.asarray(resolution)[skip:]

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, resolution, "o-", lw=1.8)
    plt.xlabel(r"True Energy $\log_{10}(E/\mathrm{GeV})$")
    plt.ylabel("Resolution")
    title = "Energy Resolution vs True Energy" + ("" if weights is None else " (mc-weighted)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prefix = "resolution" if weights is None else "resolution_weighted"
    save_path = _resolve_save_path(out_dir, save_name, prefix=prefix)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Resolution 图像已保存到: {save_path}")

def plot_log_RMSerror(true_E, pred_E, weights=None, bins=20, skip=0, save_name=None, out_dir=None, space="log"):
    """
    log RMS error:
      rho = sqrt( < (logE_pred - logE_true)^2 > )  (per true-E bin)

    weights:
      - None: unweighted RMS
      - array: weighted RMS in each bin
    """
    import numpy as np
    import matplotlib.pyplot as plt

    true_E = np.asarray(true_E).squeeze()
    pred_E = np.asarray(pred_E).squeeze()
    if weights is not None:
        weights = np.asarray(weights).squeeze()

    if space != "log":
        raise ValueError("plot_log_RMSerror currently implemented in log space. Use space='log' or implement linear branch.")

    if weights is None:
        m0 = np.isfinite(true_E) & np.isfinite(pred_E)
    else:
        m0 = np.isfinite(true_E) & np.isfinite(pred_E) & np.isfinite(weights) & (weights > 0)

    true_E = true_E[m0]
    pred_E = pred_E[m0]
    if weights is not None:
        weights = weights[m0]

    bin_edges = np.linspace(true_E.min(), true_E.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    log_rms = []
    for i in range(bins):
        m = (true_E >= bin_edges[i]) & (true_E < bin_edges[i + 1])
        if m.sum() > 10:
            residual = pred_E[m] - true_E[m]
            if weights is None:
                log_rms.append(float(np.sqrt(np.mean(residual ** 2))))
            else:
                log_rms.append(float(_weighted_rms(residual, weights[m])))
        else:
            log_rms.append(np.nan)

    bin_centers = bin_centers[skip:]
    log_rms = np.asarray(log_rms)[skip:]

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, log_rms, "o-", lw=1.8)
    plt.xlabel(r"True Energy $\log_{10}(E/\mathrm{GeV})$")
    plt.ylabel("log RMS error")
    title = "Log RMS Error vs True Energy" + ("" if weights is None else " (mc-weighted)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prefix = "logRMS" if weights is None else "logRMS_weighted"
    save_path = _resolve_save_path(out_dir, save_name, prefix=prefix)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ log RMS error 图像已保存到: {save_path}")


def plot_pred_vs_true_heatmap(
    true,
    pred,
    weights=None,
    bins=120,
    save_name=None,
    out_dir=None,
    space="log",):
    """
    使用二维直方图绘制 Pred vs True 能量响应矩阵（logE–logE）

    Parameters
    ----------
    true, pred : array-like
        log10(E / GeV)
    weights : array-like or None
        事件权重（如 mc_weight）。
        - None: 普通 counts heatmap（等权）
        - array: 加权 heatmap（Crab 等效）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    true = np.asarray(true).squeeze()
    pred = np.asarray(pred).squeeze()

    if weights is not None:
        weights = np.asarray(weights).squeeze()

    # 目前 heatmap 只定义在 log space
    if space != "log":
        raise ValueError(
            "plot_pred_vs_true_heatmap is implemented for log space only."
        )

    # ========= valid mask =========
    if weights is None:
        m = np.isfinite(true) & np.isfinite(pred)
    else:
        m = (
            np.isfinite(true)
            & np.isfinite(pred)
            & np.isfinite(weights)
            & (weights > 0)
        )

    true = true[m]
    pred = pred[m]
    if weights is not None:
        weights = weights[m]

    # ========= 2D histogram =========
    H, xedges, yedges = np.histogram2d(
        true,
        pred,
        bins=bins,
        weights=weights,  # None 或 mc_weight
    )

    # 防止 LogNorm vmin=0
    H_plot = H.copy()
    H_plot[H_plot <= 0] = np.nan

    # ========= plot =========
    plt.figure(figsize=(6, 6))
    im = plt.pcolormesh(
        xedges,
        yedges,
        H_plot.T,
        norm=LogNorm(vmin=np.nanmin(H_plot), vmax=np.nanmax(H_plot)),
        cmap="viridis",
    )

    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)

    plt.xlabel(r"log$_{10}$(E$_{\mathrm{true}}$ / GeV)")
    plt.ylabel(r"log$_{10}$(E$_{\mathrm{pred}}$ / GeV)")

    if weights is None:
        plt.title("Energy Response Matrix (counts)")
        cbar_label = "Counts"
        prefix = "pred_vs_true_heatmap"
    else:
        plt.title("Energy Response Matrix (mc-weighted)")
        cbar_label = "Weighted counts"
        prefix = "pred_vs_true_heatmap_weighted"

    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = _resolve_save_path(
        out_dir,
        save_name,
        prefix=prefix,
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Energy response heatmap 已保存到: {save_path}")


def plot_pred_distributions_in_true_bins(true_logE, pred_logE,
                                         Emin=1e2, Emax=1e6, nbins_true=10,
                                         bins_pred=60, use_delta=False,
                                         save_name=None, out_dir=None):
    """
    true_logE, pred_logE: log10(E/GeV)
    nbins_true: true energy 分 nbins_true 个 log bin（Emin->Emax）
    use_delta:
        False -> 画 pred_logE 分布
        True  -> 画 ΔlogE = pred_logE - true_logE 分布
    运用所有数据
    if use_delta = False  x=pred_logE
    if use_delta = True   x=ΔlogE
        bias: mu 即mean(x) 同上面bias的定义
        resolution: sigma 即std(x)  同上面的resolution的定义
    返回 stats：每个true能量bin的均值/方差等
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    true_logE = np.asarray(true_logE).squeeze()
    pred_logE = np.asarray(pred_logE).squeeze()
    m = np.isfinite(true_logE) & np.isfinite(pred_logE)
    true_logE = true_logE[m]
    pred_logE = pred_logE[m]

    # true energy 的log-bin边界：Emin->Emax
    edges_log = np.linspace(np.log10(Emin), np.log10(Emax), nbins_true + 1)

    ncols = 5
    nrows = int(np.ceil(nbins_true / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    stats = []

    for i in range(nbins_true):
        lo, hi = edges_log[i], edges_log[i + 1]
        sel = (true_logE >= lo) & (true_logE < hi)
        ax = axes[i]

        if sel.sum() < 10:
            ax.text(0.5, 0.5, f"Too few events\nN={sel.sum()}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]")
            continue

        if use_delta:
            x = pred_logE[sel] - true_logE[sel]
            x_label = r"$\Delta \log_{10}E$  (pred-true)"
            ref = 0.0
        else:
            x = pred_logE[sel]
            x_label = r"$\log_{10}(E_{\mathrm{pred}}/\mathrm{GeV})$"
            ref = 0.5 * (lo + hi)

        mu = float(np.mean(x))
        sig = float(np.std(x))

        ax.hist(x, bins=bins_pred, histtype="step", linewidth=1.5, density=True)
        ax.axvline(mu, linestyle="--", linewidth=1)
        ax.axvline(ref, linestyle=":", linewidth=1)

        ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]  N={sel.sum()}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("PDF")

        stats.append({
            "bin": int(i),
            "logE_true_lo": float(lo),
            "logE_true_hi": float(hi),
            "N": int(sel.sum()),
            "mu": mu,
            "sigma": sig,
            "ref": float(ref),
        })

        ax.text(0.03, 0.95, f"μ={mu:.3f}\nσ={sig:.3f}",
                transform=ax.transAxes, va="top")

    for j in range(nbins_true, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Predicted energy distribution in true-energy bins", y=1.02)
    plt.tight_layout()

    # ===== 保存（由 out_dir 控制）=====
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "delta" if use_delta else "predlogE"
            save_name = f"pred_dist_in_true_bins_{tag}_{timestamp}.png"
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ 分10能量bin的分布图已保存到: {save_path}")

    plt.close()
    return stats


def plot_pred_distributions_in_true_bins68(true_logE, pred_logE,
                                           Emin=1e2, Emax=1e6, nbins_true=10,
                                           bins_pred=60, use_delta=False,
                                           save_name=None, out_dir=None):
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

    true_logE = np.asarray(true_logE).squeeze()
    pred_logE = np.asarray(pred_logE).squeeze()
    m = np.isfinite(true_logE) & np.isfinite(pred_logE)
    true_logE = true_logE[m]
    pred_logE = pred_logE[m]

    edges_log = np.linspace(np.log10(Emin), np.log10(Emax), nbins_true + 1)

    ncols = 5
    nrows = int(np.ceil(nbins_true / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    stats = []

    for i in range(nbins_true):
        lo, hi = edges_log[i], edges_log[i + 1]
        sel = (true_logE >= lo) & (true_logE < hi)
        ax = axes[i]

        N = int(sel.sum())
        if N < 20:
            ax.text(0.5, 0.5, f"Too few events\nN={N}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]")
            continue

        if use_delta:
            x = pred_logE[sel] - true_logE[sel]
            x_label = r"$\Delta \log_{10}E$  (pred-true)"
            ref = 0.0
        else:
            x = pred_logE[sel]
            x_label = r"$\log_{10}(E_{\mathrm{pred}}/\mathrm{GeV})$"
            ref = 0.5 * (lo + hi)

        med = float(np.median(x))
        q16, q84 = np.percentile(x, [16, 84])
        q16, q84 = float(q16), float(q84)
        sig68 = float(0.5 * (q84 - q16))

        ax.hist(x, bins=bins_pred, histtype="step", linewidth=1.5, density=True)
        ax.axvspan(q16, q84, color="gray", alpha=0.3, label="68% containment")
        ax.axvline(q16, linestyle="-.", linewidth=1.0)
        ax.axvline(q84, linestyle="-.", linewidth=1.0)

        ax.set_title(f"logE_true ∈ [{lo:.2f},{hi:.2f}]  N={N}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("PDF")

        stats.append({
            "bin": int(i),
            "logE_true_lo": float(lo),
            "logE_true_hi": float(hi),
            "N": N,
            "mu": med,           # median
            "sigma": sig68,      # 68% half-width
            "q16": q16,
            "q84": q84,
            "ref": float(ref),
        })

        ax.text(0.03, 0.95, f"med={med:.3f}\nσ68={sig68:.3f}",
                transform=ax.transAxes, va="top")

    for j in range(nbins_true, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Predicted energy distribution in true-energy bins (68% containment)", y=1.02)
    plt.tight_layout()

    # ===== 保存（由 out_dir 控制）=====
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "delta" if use_delta else "predlogE"
            save_name = f"pred_dist_in_true_bins_{tag}_contain68_{timestamp}.png"
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ 分10能量bin68%比例分布图已保存到: {save_path}")

    plt.close()
    return stats

def plot_true_distributions_in_pred_bins68(true_logE, pred_logE,
                                           Emin=1e2, Emax=1e6, nbins_pred=10,
                                           bins_x=60, use_delta=False,
                                           save_name=None, out_dir=None):
    """
    以预测能量 pred_logE 分 bin（Emin~Emax, nbins_pred 个 log bin），
    在每个 pred bin 内看分布并用鲁棒 68% containment 统计：
        bias = median(x)
        resolution = (q84 - q16)/2

    参数
    ----
    true_logE, pred_logE : array-like
        log10(E/GeV)
    nbins_pred : int
        按 pred 能量分 bin 的数量
    bins_x : int
        直方图的 bins 数（每个子图内部）
    use_delta :
        False -> 画 true_logE 分布（在 pred bin 里）
        True  -> 画 ΔlogE = pred_logE - true_logE 分布（在 pred bin 里）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    true_logE = np.asarray(true_logE).squeeze()
    pred_logE = np.asarray(pred_logE).squeeze()
    m = np.isfinite(true_logE) & np.isfinite(pred_logE)
    true_logE = true_logE[m]
    pred_logE = pred_logE[m]

    edges_log = np.linspace(np.log10(Emin), np.log10(Emax), nbins_pred + 1)

    ncols = 5
    nrows = int(np.ceil(nbins_pred / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    stats = []

    for i in range(nbins_pred):
        lo, hi = edges_log[i], edges_log[i + 1]
        # ✅ 关键：按 pred_logE 分 bin
        sel = (pred_logE >= lo) & (pred_logE < hi)
        ax = axes[i]

        N = int(sel.sum())
        if N < 20:
            ax.text(0.5, 0.5, f"Too few events\nN={N}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"logE_pred ∈ [{lo:.2f},{hi:.2f}]")
            continue

        if use_delta:
            x = pred_logE[sel] - true_logE[sel]
            x_label = r"$\Delta \log_{10}E$  (pred-true)"
            ref = 0.0
        else:
            # ✅ 在 pred bin 内看 true_logE 的分布（你也可以改成看 pred_logE 自己，但那就没意义了）
            x = true_logE[sel]
            x_label = r"$\log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$"
            ref = 0.5 * (lo + hi)   # ✅ 参考值用 pred bin 中心更合理

        med = float(np.median(x))
        q16, q84 = np.percentile(x, [16, 84])
        q16, q84 = float(q16), float(q84)
        sig68 = float(0.5 * (q84 - q16))

        ax.hist(x, bins=bins_x, histtype="step", linewidth=1.5, density=True)
        ax.axvspan(q16, q84, color="gray", alpha=0.3, label="68% containment")
        ax.axvline(q16, linestyle="-.", linewidth=1.0)
        ax.axvline(q84, linestyle="-.", linewidth=1.0)

        ax.set_title(f"logE_pred ∈ [{lo:.2f},{hi:.2f}]  N={N}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("PDF")

        stats.append({
            "bin": int(i),
            "logE_pred_lo": float(lo),
            "logE_pred_hi": float(hi),
            "N": N,
            "mu": med,           # median
            "sigma": sig68,      # 68% half-width
            "q16": q16,
            "q84": q84,
            "ref": float(ref),
        })

        ax.text(0.03, 0.95, f"med={med:.3f}\nσ68={sig68:.3f}",
                transform=ax.transAxes, va="top")

    for j in range(nbins_pred, len(axes)):
        axes[j].axis("off")

    fig.suptitle("True/Δ distributions in predicted-energy bins (68% containment)", y=1.02)
    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "delta" if use_delta else "truelogE"
            save_name = f"true_dist_in_pred_bins_{tag}_contain68_{timestamp}.png"
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ 以预测能量分bin的68%分布图已保存到: {save_path}")

    plt.close()
    return stats


def plot_true_distributions_in_pred_bins(true_logE, pred_logE,
                                         Emin=1e2, Emax=1e6, nbins_pred=10,
                                         bins_x=60, use_delta=False,
                                         save_name=None, out_dir=None):
    """
    以预测能量 pred_logE 分 nbins_pred 个 log bin（Emin->Emax），在每个 pred bin 内画分布并统计均值/标准差。

    use_delta:
        False -> 画 true_logE 分布（条件：落在该 pred bin 内）
        True  -> 画 ΔlogE = pred_logE - true_logE 分布（条件：落在该 pred bin 内）

    统计量（使用所有数据，不用 68%）：
        bias:  mu = mean(x)
        resolution: sigma = std(x)

    返回 stats：每个 pred 能量 bin 的均值/方差等
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    true_logE = np.asarray(true_logE).squeeze()
    pred_logE = np.asarray(pred_logE).squeeze()
    m = np.isfinite(true_logE) & np.isfinite(pred_logE)
    true_logE = true_logE[m]
    pred_logE = pred_logE[m]

    # ✅ pred energy 的log-bin边界：Emin->Emax
    edges_log = np.linspace(np.log10(Emin), np.log10(Emax), nbins_pred + 1)

    ncols = 5
    nrows = int(np.ceil(nbins_pred / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    stats = []

    for i in range(nbins_pred):
        lo, hi = edges_log[i], edges_log[i + 1]
        # ✅ 关键：按 pred_logE 分 bin
        sel = (pred_logE >= lo) & (pred_logE < hi)
        ax = axes[i]

        N = int(sel.sum())
        if N < 10:
            ax.text(0.5, 0.5, f"Too few events\nN={N}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"logE_pred ∈ [{lo:.2f},{hi:.2f}]")
            continue

        if use_delta:
            x = pred_logE[sel] - true_logE[sel]
            x_label = r"$\Delta \log_{10}E$  (pred-true)"
            ref = 0.0
        else:
            # ✅ 在 pred bin 内看 true_logE 的分布（更有意义；看 pred_logE 自己会“人为截断”）
            x = true_logE[sel]
            x_label = r"$\log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$"
            ref = 0.5 * (lo + hi)   # ✅ 参考用 pred bin 中心

        mu = float(np.mean(x))
        sig = float(np.std(x))  # 如需无偏估计可改 ddof=1

        ax.hist(x, bins=bins_x, histtype="step", linewidth=1.5, density=True)
        ax.axvline(mu, linestyle="--", linewidth=1)
        ax.axvline(ref, linestyle=":", linewidth=1)

        ax.set_title(f"logE_pred ∈ [{lo:.2f},{hi:.2f}]  N={N}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("PDF")

        stats.append({
            "bin": int(i),
            "logE_pred_lo": float(lo),
            "logE_pred_hi": float(hi),
            "N": N,
            "mu": mu,
            "sigma": sig,
            "ref": float(ref),
        })

        ax.text(0.03, 0.95, f"μ={mu:.3f}\nσ={sig:.3f}",
                transform=ax.transAxes, va="top")

    for j in range(nbins_pred, len(axes)):
        axes[j].axis("off")

    fig.suptitle("True/Δ distributions in predicted-energy bins (mean/std)", y=1.02)
    plt.tight_layout()

    # ===== 保存（由 out_dir 控制）=====
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "delta" if use_delta else "truelogE"
            save_name = f"true_dist_in_pred_bins_{tag}_{timestamp}.png"
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ 以预测能量分bin的分布图已保存到: {save_path}")

    plt.close()
    return stats
