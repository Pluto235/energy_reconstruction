import os
import json
import torch
import numpy as np
from sklearn.metrics import r2_score
from src import utils


def evaluate_model(
    model,
    test_loader,
    checkpoint_path=None,
    out_dir=None,
    space="log",          # "log" or "linear" (仅影响你 utils 画图口径；本函数两套指标都会给)
    save_arrays=True,
):
    """
    Evaluate model for log10(E) regression.

    - 输出两套 overall metrics：
        (1) unweighted: 事件等权（与你当前 utils 图一致）
        (2) weighted:   mc_weight 加权（Crab 等效分布整体表现）
    - 不改 utils 的画图：仍然用等权的 true/pred

    Args:
        out_dir: directory to save figures + arrays + metrics.json
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # === load checkpoint ===
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # DataParallel compatibility
        if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            print("⚙️ 检测到 DataParallel 模型，已移除 'module.' 前缀")

        model.load_state_dict(checkpoint)
        print(f"✅ 成功加载模型权重: {checkpoint_path}")
    else:
        print("⚠️ checkpoint_path=None，将直接评估当前 model 权重")

    model.to(device)
    model.eval()

    preds, trues, ws = [], [], []

    with torch.no_grad():
        for points, features, mask, logE_true, mc_weight in test_loader:
            points = points.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logE_true = logE_true.to(device, non_blocking=True)

            # mc_weight 仅用于统计，留在 CPU 也行；这里直接 detach->cpu
            logE_pred = model(points, features, mask)

            preds.append(logE_pred.detach().cpu().numpy())
            trues.append(logE_true.detach().cpu().numpy())
            ws.append(mc_weight.detach().cpu().numpy())

    if len(preds) == 0:
        print("⚠️ test_loader 为空，未产生预测")
        return None, None

    logE_pred = np.concatenate(preds, axis=0).squeeze()
    logE_true = np.concatenate(trues, axis=0).squeeze()
    w = np.concatenate(ws, axis=0).squeeze().astype(np.float64)

    # ===== clean invalid =====
    valid = np.isfinite(logE_pred) & np.isfinite(logE_true) & np.isfinite(w) & (w > 0)
    if np.sum(~valid) > 0:
        print(f"⚠️ 清理掉无效事件: {int(np.sum(~valid))} / {int(valid.size)} (NaN/Inf 或 w<=0)")
    logE_pred = logE_pred[valid]
    logE_true = logE_true[valid]
    w = w[valid]

    if logE_true.size == 0:
        print("⚠️ 清理后无有效事件")
        return None, None

    # 可选：归一化权重到 mean=1（不改变加权均值的相对含义，数值更稳）
    w = w / (np.mean(w) + 1e-12)

    # ===== weighted helpers =====
    def wmean(x, ww):
        return float(np.sum(ww * x) / (np.sum(ww) + 1e-12))

    def wstd(x, ww):
        m = np.sum(ww * x) / (np.sum(ww) + 1e-12)
        v = np.sum(ww * (x - m) ** 2) / (np.sum(ww) + 1e-12)
        return float(np.sqrt(max(v, 0.0)))

    def wrmse(err, ww):
        return float(np.sqrt(np.sum(ww * (err ** 2)) / (np.sum(ww) + 1e-12)))

    # ===== metrics in log space =====
    dlogE = logE_pred - logE_true

    # unweighted
    log_bias = float(np.mean(dlogE))
    log_sigma = float(np.std(dlogE))
    log_rmse = float(np.sqrt(np.mean(dlogE ** 2)))

    # weighted (Crab 等效)
    w_log_bias = wmean(dlogE, w)
    w_log_sigma = wstd(dlogE, w)
    w_log_rmse = wrmse(dlogE, w)

    try:
        r2 = float(r2_score(logE_true, logE_pred))
    except Exception:
        r2 = None

    # ===== metrics in linear space =====
    E_pred = np.power(10.0, logE_pred)
    E_true = np.power(10.0, logE_true)
    rel = (E_pred - E_true) / (E_true + 1e-12)

    # unweighted
    rel_bias = float(np.mean(rel))
    rel_sigma = float(np.std(rel))

    # weighted
    w_rel_bias = wmean(rel, w)
    w_rel_sigma = wstd(rel, w)

    print("\n=== 模型评估结果（overall）===")
    print(f"[log][unweighted]   bias={log_bias:.4f}, sigma={log_sigma:.4f}, rmse={log_rmse:.4f}")
    print(f"[log][mc_weighted] bias={w_log_bias:.4f}, sigma={w_log_sigma:.4f}, rmse={w_log_rmse:.4f}")
    print(f"[lin][unweighted]  rel_bias={rel_bias:.4f}, rel_sigma={rel_sigma:.4f}")
    print(f"[lin][mc_weighted] rel_bias={w_rel_bias:.4f}, rel_sigma={w_rel_sigma:.4f}")
    if r2 is not None:
        print(f"[log]             R²={r2:.4f}")

    # ===== output dir =====
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        metrics = {
            "n": int(logE_true.size),
            "checkpoint": checkpoint_path,
            "space_used_for_plots": space,

            # unweighted
            "log_bias": log_bias,
            "log_sigma": log_sigma,
            "log_rmse": log_rmse,
            "rel_bias": rel_bias,
            "rel_sigma": rel_sigma,
            "r2_log": r2,

            # weighted
            "w_log_bias": w_log_bias,
            "w_log_sigma": w_log_sigma,
            "w_log_rmse": w_log_rmse,
            "w_rel_bias": w_rel_bias,
            "w_rel_sigma": w_rel_sigma,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if save_arrays:
            np.savez(
                os.path.join(out_dir, "preds.npz"),
                logE_pred=logE_pred,
                logE_true=logE_true,
                E_pred=E_pred,
                E_true=E_true,
                rel=rel,
                dlogE=dlogE,
                mc_weight=w,   # ✅ 保存清理/归一化后的权重
            )

    # ===== plots (保持不改 utils：仍然等权口径) =====
    utils.plot_resolution(logE_true, logE_pred, space=space, out_dir=out_dir)
    utils.plot_pred_vs_true_heatmap(logE_true, logE_pred, space=space, out_dir=out_dir)
    utils.plot_bias(logE_true, logE_pred, space=space, out_dir=out_dir)
    utils.plot_log_RMSerror(logE_true, logE_pred, space=space, out_dir=out_dir)
    # plot weighted maps
    utils.plot_resolution(logE_true, logE_pred, weights=w, space=space, out_dir=out_dir)
    utils.plot_pred_vs_true_heatmap(logE_true, logE_pred, weights=w, space=space, out_dir=out_dir)
    utils.plot_bias(logE_true, logE_pred, weights=w, space=space, out_dir=out_dir)
    utils.plot_log_RMSerror(logE_true, logE_pred, weights=w, space=space, out_dir=out_dir)

    utils.plot_pred_distributions_in_true_bins(logE_true, logE_pred, out_dir=out_dir)
    utils.plot_pred_distributions_in_true_bins68(logE_true, logE_pred, out_dir=out_dir)

    return logE_pred, logE_true
