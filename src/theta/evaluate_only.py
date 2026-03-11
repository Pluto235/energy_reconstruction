import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .ParticleDataset_theta import ParticleDataset
from .ParticleRegressor_theta import ParticleNetRegressor
from .evaluate_theta import evaluate_model
from src.common import utils


def main(run_dir):

    config_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pt")

    with open(config_path, "r") as f:
        args = json.load(f)

    print(f"📦 使用配置: {config_path}")
    print(f"📦 加载模型: {ckpt_path}")

    utils.check_gpu_status()

    # ===== ROOT 文件 =====
    root_path = args["root_path"]
    files = sorted(os.listdir(root_path))
    files = [os.path.join(root_path, f) for f in files]
    root_files = files[: args["n_files"]]

    # ===== 保持和训练一致的 test split =====
    train_files, test_files = train_test_split(
        root_files,
        test_size=args["test_size"],
        random_state=args["seed"]
    )

    # ===== cuts =====
    cuts = dict(
        Emin=args["Emin"],
        Emax=args["Emax"],
        pinc_max=args["pinc_max"],
        dcedge_min=args["dcedge_min"],
        dangle_max_rad=args["dangle_max_deg"] * np.pi / 180.0,
        theta_max_rad=args["theta_max_deg"] * np.pi / 180.0,
        use_core_box=args["use_core_box"],
        core_box=tuple(args["core_box"]),
        vqsamp_ratio_min=args["vqsamp_ratio_min"],
    )

    # ===== dataset =====
    processing_conditions = [
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
    ]

    test_dataset = ParticleDataset(
        root_files=test_files,
        branches=["vx", "vy", "vt", "vq"],
        target_branch=["mc_energy"],
        processing_conditions=processing_conditions,
        max_points=args["max_points"],
        cuts=cuts,
        norm_mode=args["norm_mode"],
        sample_mode=args["sample_mode"],
        io_workers=args["io_workers"],
        compute_scaler=False,
        seed=args["seed"],
        verbose=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=args["pin_memory"]
    )

    # ===== model =====
    model = ParticleNetRegressor(
        input_dims=2,
        conv_params=[
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256))
        ],
        fc_params=[(256, 0.1), (128, 0.1)],
        use_fusion=True,
        theta_embed_dim=args["theta_embed_dim"],
        theta_embed_dropout=args["theta_embed_dropout"],
    )

    # ===== 评估 =====
    out_dir = os.path.join(run_dir, "fig_eval_only")
    os.makedirs(out_dir, exist_ok=True)

    evaluate_model(
        model,
        test_loader,
        checkpoint_path=ckpt_path,
        out_dir=out_dir,
        space=args["eval_space"],
        save_arrays=True,
    )

    print("✅ 评估完成")
    print(f"📊 图像输出: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.run_dir)
