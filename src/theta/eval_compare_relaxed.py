import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .ParticleDataset_theta import ParticleDataset
from .ParticleRegressor_theta import ParticleNetRegressor
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


def _fallback(x, y):
    return y if x is None else x


def build_argparser():
    p = argparse.ArgumentParser("Evaluate theta model with relaxed/baseline eval cuts")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--out_dir_name", type=str, required=True)

    p.add_argument("--eval_Emin", type=float, default=None)
    p.add_argument("--eval_Emax", type=float, default=None)
    p.add_argument("--eval_pinc_max", type=float, default=None)
    p.add_argument("--eval_dcedge_min", type=float, default=None)
    p.add_argument("--eval_dangle_max_deg", type=float, default=None)
    p.add_argument("--eval_theta_max_deg", type=float, default=None)
    p.add_argument("--eval_use_core_box", type=_str2bool, default=None)
    p.add_argument("--eval_core_box", type=float, nargs=4, default=None)
    p.add_argument("--eval_vqsamp_ratio_min", type=float, default=None)
    p.add_argument("--eval_require_fitstat0", type=_str2bool, default=None)
    p.add_argument("--eval_fitstat_equals", type=int, default=None)

    p.add_argument("--space", type=str, default=None, choices=["log", "linear"])
    p.add_argument("--save_arrays", type=_str2bool, default=True)
    return p


def load_training_config(run_dir: str):
    config_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pt")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config, config_path, ckpt_path


def build_root_files(config):
    root_path = config["root_path"]
    files = sorted(
        os.path.join(root_path, f)
        for f in os.listdir(root_path)
        if f.endswith(".root") and os.path.isfile(os.path.join(root_path, f))
    )
    return files[: config["n_files"]]


def build_eval_cuts(config, overrides):
    eval_dangle_deg = _fallback(overrides.get("eval_dangle_max_deg"), _fallback(config.get("eval_dangle_max_deg"), config["dangle_max_deg"]))
    eval_theta_deg = _fallback(overrides.get("eval_theta_max_deg"), _fallback(config.get("eval_theta_max_deg"), config["theta_max_deg"]))

    cuts = dict(
        Emin=_fallback(overrides.get("eval_Emin"), _fallback(config.get("eval_Emin"), config["Emin"])),
        Emax=_fallback(overrides.get("eval_Emax"), _fallback(config.get("eval_Emax"), config["Emax"])),
        pinc_max=_fallback(overrides.get("eval_pinc_max"), _fallback(config.get("eval_pinc_max"), config["pinc_max"])),
        dcedge_min=_fallback(overrides.get("eval_dcedge_min"), _fallback(config.get("eval_dcedge_min"), config["dcedge_min"])),
        dangle_max_rad=eval_dangle_deg * np.pi / 180.0,
        theta_max_rad=eval_theta_deg * np.pi / 180.0,
        use_core_box=_fallback(overrides.get("eval_use_core_box"), bool(config.get("eval_use_core_box") or config["use_core_box"])),
        core_box=tuple(_fallback(overrides.get("eval_core_box"), _fallback(config.get("eval_core_box"), config["core_box"]))),
        vqsamp_ratio_min=_fallback(overrides.get("eval_vqsamp_ratio_min"), _fallback(config.get("eval_vqsamp_ratio_min"), config["vqsamp_ratio_min"])),
        require_fitstat0=_fallback(overrides.get("eval_require_fitstat0"), _fallback(config.get("eval_require_fitstat0"), config.get("require_fitstat0", True))),
        fitstat_equals=_fallback(overrides.get("eval_fitstat_equals"), _fallback(config.get("eval_fitstat_equals"), config.get("fitstat_equals", 0))),
    )

    effective_eval_config = deepcopy(config)
    effective_eval_config.update(
        {
            "effective_eval_Emin": cuts["Emin"],
            "effective_eval_Emax": cuts["Emax"],
            "effective_eval_pinc_max": cuts["pinc_max"],
            "effective_eval_dcedge_min": cuts["dcedge_min"],
            "effective_eval_dangle_max_deg": float(eval_dangle_deg),
            "effective_eval_theta_max_deg": float(eval_theta_deg),
            "effective_eval_use_core_box": bool(cuts["use_core_box"]),
            "effective_eval_core_box": list(cuts["core_box"]),
            "effective_eval_vqsamp_ratio_min": cuts["vqsamp_ratio_min"],
            "effective_eval_require_fitstat0": bool(cuts["require_fitstat0"]),
            "effective_eval_fitstat_equals": int(cuts["fitstat_equals"]),
        }
    )

    return cuts, effective_eval_config


def build_test_dataset(config, test_files, cuts):
    processing_conditions = [
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
        {"subtract": 0, "multiply": 1, "min": -1e5, "max": 1e5},
    ]

    return ParticleDataset(
        root_files=test_files,
        branches=["vx", "vy", "vt", "vq"],
        target_branch=["mc_energy"],
        processing_conditions=processing_conditions,
        max_points=config["max_points"],
        cuts=cuts,
        norm_mode=config["norm_mode"],
        sample_mode=config["sample_mode"],
        io_workers=config["io_workers"],
        compute_scaler=False,
        seed=config["seed"],
        verbose=False,
        nv_scale=config.get("nv_scale", 3000.0),
    )


def build_model(config):
    return ParticleNetRegressor(
        input_dims=2,
        conv_params=[
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
        ],
        fc_params=[(256, 0.1), (128, 0.1)],
        use_fusion=True,
        theta_embed_dim=config["theta_embed_dim"],
        theta_embed_dropout=config["theta_embed_dropout"],
        nv_embed_dim=config.get("nv_embed_dim", 0),
        nv_embed_dropout=config.get("nv_embed_dropout", 0.0),
    )


def main():
    parser = build_argparser()
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    config, config_path, ckpt_path = load_training_config(run_dir)

    overrides = {
        "eval_Emin": args.eval_Emin,
        "eval_Emax": args.eval_Emax,
        "eval_pinc_max": args.eval_pinc_max,
        "eval_dcedge_min": args.eval_dcedge_min,
        "eval_dangle_max_deg": args.eval_dangle_max_deg,
        "eval_theta_max_deg": args.eval_theta_max_deg,
        "eval_use_core_box": args.eval_use_core_box,
        "eval_core_box": args.eval_core_box,
        "eval_vqsamp_ratio_min": args.eval_vqsamp_ratio_min,
        "eval_require_fitstat0": args.eval_require_fitstat0,
        "eval_fitstat_equals": args.eval_fitstat_equals,
    }

    print(f"📦 使用训练配置: {config_path}")
    print(f"📦 加载模型权重: {ckpt_path}")
    print(f"📊 评估输出目录名: {args.out_dir_name}")

    utils.check_gpu_status()

    root_files = build_root_files(config)
    train_files, test_files = train_test_split(
        root_files,
        test_size=config["test_size"],
        random_state=config["seed"],
    )

    cuts, effective_eval_config = build_eval_cuts(config, overrides)

    print("🧪 Effective eval cuts:")
    print(json.dumps(
        {
            "Emin": cuts["Emin"],
            "Emax": cuts["Emax"],
            "pinc_max": cuts["pinc_max"],
            "dcedge_min": cuts["dcedge_min"],
            "dangle_max_deg": effective_eval_config["effective_eval_dangle_max_deg"],
            "theta_max_deg": effective_eval_config["effective_eval_theta_max_deg"],
            "use_core_box": cuts["use_core_box"],
            "core_box": cuts["core_box"],
            "vqsamp_ratio_min": cuts["vqsamp_ratio_min"],
            "require_fitstat0": cuts["require_fitstat0"],
            "fitstat_equals": cuts["fitstat_equals"],
        },
        indent=2,
    ))

    test_dataset = build_test_dataset(config, test_files, cuts)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    model = build_model(config)

    out_dir = os.path.join(run_dir, args.out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    effective_eval_config.update(
        {
            "source_run_dir": run_dir,
            "checkpoint_path": ckpt_path,
            "out_dir": out_dir,
            "out_dir_name": args.out_dir_name,
            "space": args.space or config["eval_space"],
            "save_arrays": bool(args.save_arrays),
        }
    )
    with open(os.path.join(out_dir, "effective_eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(effective_eval_config, f, indent=2)

    evaluate_model(
        model=model,
        test_loader=test_loader,
        checkpoint_path=ckpt_path,
        out_dir=out_dir,
        space=args.space or config["eval_space"],
        save_arrays=bool(args.save_arrays),
    )

    print("✅ 评估完成")
    print(f"📊 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
