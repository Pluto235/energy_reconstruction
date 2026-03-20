# 用保存到preds.npz进行快速评估画图
import os
import argparse
import numpy as np

from src.utils import (
    plot_bias,
    plot_resolution,
    plot_log_RMSerror,
    plot_pred_vs_true_heatmap,
    plot_pred_distributions_in_true_bins,
    plot_pred_distributions_in_true_bins68,
)

def get_any(npz, *keys):
    for k in keys:
        if k in npz:
            return npz[k]
    return None

def main():
    '''
    conda activate py310
    cd /home/server/projects/energy_reconstruction/src
    
    python analyze_preds.py \
      --npz /home/server/projects/energy_reconstruction/runs/*****/preds.npz \
      --out_dir /home/server/projects/energy_reconstruction/runs/****/plots \
      --tag sample_random
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="preds.npz / pred.npz path")
    ap.add_argument("--out_dir", required=True, help="directory to save figures")
    ap.add_argument("--tag", default="exp", help="used in figure names")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--emin", type=float, default=1e2)
    ap.add_argument("--emax", type=float, default=1e6)
    ap.add_argument("--nbins_true", type=int, default=10)
    ap.add_argument("--bins_pred", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)

    # 兼容不同字段名：优先用 logE；没有则用 E 转 log10
    true_logE = get_any(d, "true_logE", "true_log10E", "true_log_e")
    pred_logE = get_any(d, "pred_logE", "pred_log10E", "pred_log_e")

    if true_logE is None or pred_logE is None:
        true_E = get_any(d, "true_E", "true_energy", "true_e")
        pred_E = get_any(d, "pred_E", "pred_energy", "pred_e")
        if true_E is None or pred_E is None:
            raise KeyError(f"npz keys not found. available keys: {list(d.keys())}")
        true_logE = np.log10(np.asarray(true_E).squeeze())
        pred_logE = np.log10(np.asarray(pred_E).squeeze())
    else:
        true_logE = np.asarray(true_logE).squeeze()
        pred_logE = np.asarray(pred_logE).squeeze()

    # 统一过滤 NaN/inf
    m = np.isfinite(true_logE) & np.isfinite(pred_logE)
    true_logE, pred_logE = true_logE[m], pred_logE[m]

    # 你 utils 的 bias/resolution/logRMS 都是 log space，直接喂 logE
    plot_bias(true_logE, pred_logE, bins=args.bins, skip=args.skip,
              save_name=f"{args.tag}_bias.png", out_dir=args.out_dir, space="log")

    plot_resolution(true_logE, pred_logE, bins=args.bins, skip=args.skip,
                    save_name=f"{args.tag}_resolution.png", out_dir=args.out_dir, space="log")

    plot_log_RMSerror(true_logE, pred_logE, bins=args.bins, skip=args.skip,
                      save_name=f"{args.tag}_logRMS.png", out_dir=args.out_dir, space="log")

    # 响应矩阵 heatmap
    plot_pred_vs_true_heatmap(true_logE, pred_logE, bins=120,
                              save_name=f"{args.tag}_heatmap.png", out_dir=args.out_dir, space="log")

    # 分 true 能量 bin 的分布：pred_logE & delta
    plot_pred_distributions_in_true_bins(
        true_logE, pred_logE,
        Emin=args.emin, Emax=args.emax, nbins_true=args.nbins_true,
        bins_pred=args.bins_pred, use_delta=False,
        save_name=f"{args.tag}_pred_in_true_bins.png", out_dir=args.out_dir
    )
    plot_pred_distributions_in_true_bins(
        true_logE, pred_logE,
        Emin=args.emin, Emax=args.emax, nbins_true=args.nbins_true,
        bins_pred=args.bins_pred, use_delta=True,
        save_name=f"{args.tag}_delta_in_true_bins.png", out_dir=args.out_dir
    )

    # 68% containment 版本
    plot_pred_distributions_in_true_bins68(
        true_logE, pred_logE,
        Emin=args.emin, Emax=args.emax, nbins_true=args.nbins_true,
        bins_pred=args.bins_pred, use_delta=True,
        save_name=f"{args.tag}_delta_in_true_bins_68.png", out_dir=args.out_dir
    )

    print(f"✅ Done. Saved to: {args.out_dir}")

if __name__ == "__main__":
    main()


