import os
import json
import time
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .ParticleDataset_theta import ParticleDataset
from .ParticleRegressor_theta import ParticleNetRegressor
from .train_theta import train_model
from .evaluate_theta import evaluate_model
from src.common import utils


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_argparser():
    p = argparse.ArgumentParser("WCDA Energy Reconstruction")

    # ===== data =====
    p.add_argument("--root_path", type=str, default="/home/server/mydisk/WCDA_simulation/")
    p.add_argument("--n_files", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.1)  # val fraction of train split

    # ===== dataset / dataloader =====
    p.add_argument("--max_points", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)

    # ROOT I/O workers inside ParticleDataset multiprocessing
    p.add_argument("--io_workers", type=int, default=8)

    # ===== train cuts (event-level) =====
    p.add_argument("--Emin", type=float, default=None, help="mc_energy lower bound (GeV), inclusive")
    p.add_argument("--Emax", type=float, default=None, help="mc_energy upper bound (GeV), exclusive; None means no upper bound")
    p.add_argument("--pinc_max", type=float, default=1.1)
    p.add_argument("--dcedge_min", type=float, default=20.0)
    p.add_argument("--dangle_max_deg", type=float, default=3.0)
    p.add_argument("--theta_max_deg", type=float, default=30.0)
    p.add_argument("--use_core_box", action="store_true", default=False)
    p.add_argument("--core_box", type=float, nargs=4, default=[-130.0, 130.0, -110.0, 110.0])
    p.add_argument("--vqsamp_ratio_min", type=float, default=None) # events 层面的vqsamp剔除
    p.add_argument("--require_fitstat0", type=_str2bool, default=True, help="Whether to keep only fitstat == 0 events")
    p.add_argument("--fitstat_equals", type=int, default=0, help="Keep only events with this fitstat value")

    # ===== eval cuts (event-level) =====
    p.add_argument("--eval_Emin", type=float, default=None)
    p.add_argument("--eval_Emax", type=float, default=None)
    p.add_argument("--eval_pinc_max", type=float, default=None)
    p.add_argument("--eval_dcedge_min", type=float, default=None)
    p.add_argument("--eval_dangle_max_deg", type=float, default=None)
    p.add_argument("--eval_theta_max_deg", type=float, default=None)
    p.add_argument("--eval_use_core_box", action="store_true", default=False)
    p.add_argument("--eval_core_box", type=float, nargs=4, default=None)
    p.add_argument("--eval_vqsamp_ratio_min", type=float, default=None)
    p.add_argument("--eval_require_fitstat0", type=_str2bool, default=None)
    p.add_argument("--eval_fitstat_equals", type=int, default=None)

    # ===== feature processing behavior =====
    p.add_argument("--norm_mode", type=str, default="per_event", choices=["per_event", "global", "none"])
    p.add_argument("--sample_mode", type=str, default="random", choices=["random", "topk_q", "firstk", "weighted_q"])

    # ===== training =====
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)

    # ===== imbalance weighting (bin weights) =====
    p.add_argument("--bins_hist", type=int, default=50)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--max_weight", type=float, default=None, help="Optional clip for bin weights, e.g. 10.0")

    # ===== loss mode =====
    p.add_argument("--loss_mode", type=str, default="huber", choices=["mse", "huber", "rel"])
    p.add_argument("--huber_delta", type=float, default=0.2)
    p.add_argument("--rel_delta", type=float, default=0.3)
    p.add_argument("--rel_squared", action="store_true", default=False)

    # ===== outputs =====
    p.add_argument("--run_dir", type=str, default=None,
                   help="If None, auto create under /home/server/projects/energy_reconstruction/runs/")
    p.add_argument("--tag", type=str, default="exp")

    # ===== evaluation plotting =====
    p.add_argument("--eval_space", type=str, default="log", choices=["log", "linear"])
    p.add_argument("--save_arrays", action="store_true", default=True)

    # ===== model: theta embedding =====
    p.add_argument("--theta_embed_dim", type=int, default=16, help="embedding dim for costheta (0 disables)")
    p.add_argument("--theta_embed_dropout", type=float, default=0.0, help="dropout in theta embedding MLP")

    return p


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    """
    returns:
      model, log_pred, log_true, train_losses, val_losses, run_dir
    """
    set_seed(args.seed)

    # ===== run_dir =====
    if args.run_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = (
            f"{ts}_{args.tag}_bs{args.batch_size}_mp{args.max_points}_"
            f"{args.norm_mode}_{args.sample_mode}_Emin{args.Emin}_seed{args.seed}_"
            f"loss{args.loss_mode}"
        )
        run_dir = os.path.join("/home/server/projects/energy_reconstruction/runs", run_id)
    else:
        run_dir = args.run_dir

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    fig_dir = os.path.join(run_dir, "fig")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ===== save config =====
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"📦 run_dir: {run_dir}")

    # ===== list ROOT files =====
    root_path = args.root_path
    file_path = []
    for filename in os.listdir(root_path):
        if not filename.endswith(".root"):
            continue
        full_path = os.path.join(root_path, filename)
        if os.path.isfile(full_path):
            file_path.append(full_path)

    file_path.sort()
    root_files = file_path[: args.n_files]
    print(f"📁 本次使用 {len(root_files)} 个ROOT文件数据")

    # ===== processing conditions =====
    processing_conditions = [
        {"name": "vx", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
        {"name": "vy", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
        {"name": "vt", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
        {"name": "vq", "subtract": 0, "multiply": 1, "min": -100000.0, "max": 100000.0},
    ]
    branches = [d["name"] for d in processing_conditions]
    target_branch = ["mc_energy"]
    processing_conditions = [{k: v for k, v in d.items() if k != "name"} for d in processing_conditions]

    # ===== GPU check =====
    utils.check_gpu_status()

    # ===== split files =====
    train_files, test_files = train_test_split(root_files, test_size=args.test_size, random_state=args.seed)
    train_files, val_files = train_test_split(train_files, test_size=args.val_size, random_state=args.seed)

    print(f"训练集文件数: {len(train_files)}")  # 72%
    print(f"验证集文件数: {len(val_files)}")    # 8%
    print(f"测试集文件数: {len(test_files)}")   # 20%
    fitstat_desc = f"fitstat == {args.fitstat_equals}" if args.require_fitstat0 else "fitstat unrestricted"
    print(f"事件筛选: {fitstat_desc}")

    # ===== train cuts dict (degrees -> radians) =====
    cuts_train = dict(
        Emin=args.Emin,
        Emax=args.Emax,
        pinc_max=args.pinc_max,
        dcedge_min=args.dcedge_min,
        dangle_max_rad=args.dangle_max_deg * np.pi / 180.0,
        theta_max_rad=args.theta_max_deg * np.pi / 180.0,
        use_core_box=args.use_core_box,
        core_box=tuple(args.core_box),
        vqsamp_ratio_min=args.vqsamp_ratio_min,
        require_fitstat0=args.require_fitstat0,
        fitstat_equals=args.fitstat_equals,
    )

    # ===== eval cuts dict (degrees -> radians) =====
    def _fallback(x, y):
        return y if x is None else x

    cuts_eval = dict(
        Emin=_fallback(args.eval_Emin, args.Emin),
        Emax=_fallback(args.eval_Emax, args.Emax),
        pinc_max=_fallback(args.eval_pinc_max, args.pinc_max),
        dcedge_min=_fallback(args.eval_dcedge_min, args.dcedge_min),
        dangle_max_rad=_fallback(args.eval_dangle_max_deg, args.dangle_max_deg) * np.pi / 180.0,
        theta_max_rad=_fallback(args.eval_theta_max_deg, args.theta_max_deg) * np.pi / 180.0,
        use_core_box=(args.eval_use_core_box or args.use_core_box),
        core_box=tuple(_fallback(args.eval_core_box, args.core_box)),
        vqsamp_ratio_min=_fallback(args.eval_vqsamp_ratio_min, args.vqsamp_ratio_min),
        require_fitstat0=_fallback(args.eval_require_fitstat0, args.require_fitstat0),
        fitstat_equals=_fallback(args.eval_fitstat_equals, args.fitstat_equals),
    )

    # ===== datasets =====
    train_dataset = ParticleDataset(
        root_files=train_files,
        branches=branches,
        target_branch=target_branch,
        processing_conditions=processing_conditions,
        max_points=args.max_points,
        cuts=cuts_train,
        norm_mode=args.norm_mode,
        sample_mode=args.sample_mode,
        io_workers=args.io_workers,
        compute_scaler=(args.norm_mode == "global"),
        save_stats_path=os.path.join(run_dir, "dataset_train_stats.json"),
        seed=args.seed,
        verbose=False,
    )

    val_dataset = ParticleDataset(
        root_files=val_files,
        branches=branches,
        target_branch=target_branch,
        processing_conditions=processing_conditions,
        max_points=args.max_points,
        cuts=cuts_train,
        norm_mode=args.norm_mode,
        sample_mode=args.sample_mode,
        io_workers=args.io_workers,
        scaler=train_dataset.scaler if args.norm_mode == "global" else None,
        save_stats_path=os.path.join(run_dir, "dataset_val_stats.json"),
        seed=args.seed,
        verbose=False,
    )

    test_dataset = ParticleDataset(
        root_files=test_files,
        branches=branches,
        target_branch=target_branch,
        processing_conditions=processing_conditions,
        max_points=args.max_points,
        cuts=cuts_eval,
        norm_mode=args.norm_mode,
        sample_mode=args.sample_mode,
        io_workers=args.io_workers,
        scaler=train_dataset.scaler if args.norm_mode == "global" else None,
        save_stats_path=os.path.join(run_dir, "dataset_test_stats.json"),
        seed=args.seed,
        verbose=False,
    )

    # ===== dataloaders =====
    print(f"Batch Size: {args.batch_size}")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    # ===== model =====
    model = ParticleNetRegressor(
        input_dims=2,
        conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
        fc_params=[(256, 0.1), (128, 0.1)],
        use_fusion=True,
        theta_embed_dim=args.theta_embed_dim,
        theta_embed_dropout=args.theta_embed_dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已移动到: {device}")

    # ===== sanity forward =====
    print("测试GPU运行...")
    with torch.no_grad():
        batch = next(iter(train_loader))
        if len(batch) == 6:
            test_points, test_features, test_mask,test_costheta, test_energies, _w = batch
        else:
            test_points, test_features, test_mask,test_costheta, test_energies = batch

        test_points = test_points.to(device, non_blocking=True)
        test_features = test_features.to(device, non_blocking=True)
        test_mask = test_mask.to(device, non_blocking=True)
        test_costheta = test_costheta.to(device, non_blocking=True)

        out = model(test_points, test_features, test_mask, test_costheta)
        print(f"测试输出形状: {out.shape}")
        print(f"测试输出值范围: {out.min().item():.4f} ~ {out.max().item():.4f}")

    # ===== training =====
    save_path = os.path.join(ckpt_dir, "best_model.pt")
    print("开始训练...")
    train_losses, val_losses, save_path = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=args.grad_clip,
        bins_hist=args.bins_hist,
        # min_count=args.min_count,
        # max_weight=args.max_weight,
        save_path=save_path,
        loss_mode=args.loss_mode,
        huber_delta=args.huber_delta,
        rel_delta=args.rel_delta,
        rel_squared=args.rel_squared,
    )

    # ===== save loss json + loss fig =====
    with open(os.path.join(run_dir, "loss_log.json"), "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_fig_path = os.path.join(fig_dir, "loss_curve.png")
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()
    print(f"📈 已保存图像: {loss_fig_path}")

    # ===== evaluation =====
    print("评估模型...")
    log_pred, log_true = evaluate_model(
        model,
        test_loader,
        checkpoint_path=save_path,
        out_dir=fig_dir,          # ✅ 所有评估图都进 run_dir/fig
        space=args.eval_space,
        save_arrays=args.save_arrays,
    )

    return model, log_pred, log_true, train_losses, val_losses, run_dir


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    model, pred, true, train_losses, val_losses, run_dir = main(args)

    # 可选：模型结构 summary（不建议在 Slurm 大规模扫参时开）
    try:
        from torchsummary import summary
        summary(model, input_size=[(2, args.max_points), (2, args.max_points), (1, args.max_points)])
    except Exception as e:
        print(f"torchsummary skipped: {e}")

    print(f"✅ 完成。所有输出已写入: {run_dir}")
