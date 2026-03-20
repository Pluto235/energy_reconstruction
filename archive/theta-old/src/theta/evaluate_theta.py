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
    space="log",          # "log" or "linear"
    save_arrays=True,
):
    """
    Evaluate model for log10(E) regression.

    Args:
        space:
            - "log": all metrics/plots use log10(E) space (dlogE)
            - "linear": metrics/plots use linear energy space (relative error etc.)
        out_dir:
            directory to save figures + arrays + metrics.json
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

    preds = []
    trues = []

    with torch.no_grad():
        for points, features, mask, costheta, logE_true in test_loader:
            points = points.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            costheta = costheta.to(device, non_blocking=True)
            logE_true = logE_true.to(device, non_blocking=True)

            logE_pred = model(points, features, mask, costheta)

            preds.append(logE_pred.detach().cpu().numpy())
            trues.append(logE_true.detach().cpu().numpy())

    if len(preds) == 0:
        print("⚠️ test_loader 为空，未产生预测")
        return None, None

    logE_pred = np.concatenate(preds, axis=0).squeeze()
    logE_true = np.concatenate(trues, axis=0).squeeze()

    # ===== sanity check =====
    if np.any(~np.isfinite(logE_pred)) or np.any(~np.isfinite(logE_true)):
        print("⚠️ 预测/标签存在 NaN/Inf")
        return None, None

    # ===== metrics in log space =====
    dlogE = logE_pred - logE_true
    log_bias = float(np.mean(dlogE))
    log_sigma = float(np.std(dlogE))
    log_rmse = float(np.sqrt(np.mean(dlogE ** 2)))

    try:
        r2 = float(r2_score(logE_true, logE_pred))
    except Exception:
        r2 = None

    # ===== metrics in linear space (optional) =====
    E_pred = np.power(10.0, logE_pred)
    E_true = np.power(10.0, logE_true)
    rel = (E_pred - E_true) / (E_true + 1e-12)
    rel_bias = float(np.mean(rel))
    rel_sigma = float(np.std(rel))

    print("\n=== 模型评估结果 ===")
    print(f"[log]   bias={log_bias:.4f}, sigma={log_sigma:.4f}, rmse={log_rmse:.4f}")
    print(f"[linear] rel_bias={rel_bias:.4f}, rel_sigma={rel_sigma:.4f}")
    if r2 is not None:
        print(f"[log]   R²={r2:.4f}")

    # ===== output dir =====
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        # save metrics
        metrics = {
            "n": int(logE_true.size),
            "log_bias": log_bias,
            "log_sigma": log_sigma,
            "log_rmse": log_rmse,
            "r2_log": r2,
            "rel_bias": rel_bias,
            "rel_sigma": rel_sigma,
            "space_used_for_plots": space,
            "checkpoint": checkpoint_path,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # save arrays
        if save_arrays:
            np.savez(
                os.path.join(out_dir, "preds.npz"),
                logE_pred=logE_pred,
                logE_true=logE_true,
                E_pred=E_pred,
                E_true=E_true,
                rel=rel,
                dlogE=dlogE,
            )

    # ===== plots =====
    # 你说 utils 已经支持 log/linear 选项，那这里建议显式传 space + out_dir（按你utils签名微调即可）
    utils.plot_resolution(logE_true, logE_pred, space=space, out_dir=out_dir)
    utils.plot_pred_vs_true_heatmap(logE_true, logE_pred,weights=w, space=space, out_dir=out_dir)
    utils.plot_bias(logE_true, logE_pred, space=space, out_dir=out_dir)
    utils.plot_log_RMSerror(logE_true, logE_pred, space=space, out_dir=out_dir)

    utils.plot_pred_distributions_in_true_bins(logE_true, logE_pred, out_dir=out_dir)
    utils.plot_pred_distributions_in_true_bins68(logE_true, logE_pred, out_dir=out_dir)

    return logE_pred, logE_true
