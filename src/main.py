import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import torch.nn.functional as F
import time
import uproot
from tqdm import tqdm
import logging
import argparse
from sklearn.metrics import r2_score
from typing import Tuple, List
import copy
import random

from .EdgeConv import EdgeConvBlock
from .ParticleDataset import ParticleDataset
from .ParticleRegressor import ParticleNetRegressor
from .train import train_model
from .evaluate import evaluate_model
from . import utils


def main():
    '''
    - 读取数据文件夹列表root_files
    - 提供预处理条件branches, target_branch, processing_conditions
    - 检查GPU状态
    - 划分数据集（从文件层面）
    - 创建数据集dataset, 分批读root_files, branches, target_branch, processing_conditions, max_points=256
    - 加载数据dataloader
    - model initial
    - device to cuda
    - test 一个迭代器
    - train函数
    - evaluate函数
    - return model, predictions, true_energies, train_losses, val_losses
     '''
   
	 # ======== 读取数据文件夹 ========
    root_path = "/home/server/mydisk/WCDA_split/nv_60_150/" # nv_150_500  nv_500_3000  nv_60_150
    file_path = []
    for filename in os.listdir(root_path):
           full_path = os.path.join(root_path, filename)
           file_path.append(full_path)
    

    root_files = random.sample(file_path, 1000) 
    print(f"📁 本次使用 {len(root_files)} 个ROOT文件数据")

	 # ======== 预处理条件 ========       
    processing_conditions = [
            {"name": "vx",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0},
            {"name": "vy",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0},  
            {"name": "vt",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0},
            {"name": "vq",        "subtract": 0,    "multiply": 1,    "min": -100000.0, "max": 100000.0}
        ]
    branches = [d["name"] for d in processing_conditions] # 提取name列表，便于后续读取相关的值 ，输入特征
    target_branch = ["mc_energy"] # 定义一个标签列表， 预测目标是能量
    processing_conditions = [
        {k: v for k, v in d.items() if k != "name"} # 删除掉name字段，只保留预处理操作字段
        for d in processing_conditions
    ]
    

    # ======== 先检查GPU状态 ========
    utils.check_gpu_status()
    
    # ======== 从文件层面划分数据集 =======
    from sklearn.model_selection import train_test_split
    
    # 分割数据
    train_files, test_files = train_test_split(root_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)
    
	# train_model 使用数据train 2728 和 val 304
   # evaluate_model使用数据 test 758
    print(f"训练集文件数: {len(train_files)}") # 72%
    print(f"验证集文件数: {len(val_files)}")   # 8%
    print(f"测试集文件数: {len(test_files)}")  # 20%
    
    # ======== 构建数据集 ========
    train_dataset = ParticleDataset(
        root_files=train_files,
        branches=branches,
        target_branch=target_branch,
        processing_conditions=processing_conditions,
        max_points=500
    )
    
    val_dataset = ParticleDataset(
        root_files=val_files,
        branches=branches,
        target_branch=target_branch,
        processing_conditions=processing_conditions,
        max_points=500
    )

    test_dataset = ParticleDataset(
        root_files=test_files,
        branches=branches,
        target_branch=target_branch,
        processing_conditions=processing_conditions,
        max_points=500
    )
    
    # ======== 构建 DataLoader ========
    batch_size = 256
    print(f"Batch Size: {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ======== 初始化模型 ========
    model = ParticleNetRegressor(
        input_dims=2,  # vq, vt 两个特征
        conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
        fc_params=[(256, 0.1)],
        use_fusion=True
    )
    
    # 立即将模型移动到GPU进行测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型已移动到: {device}")
    
    # ======== 测试一次前向传播，确保模型正常 ========
    print("测试GPU运行...")
    with torch.no_grad():
        test_points, test_features, test_mask, test_energies = next(iter(train_loader))
        test_points = test_points.to(device)
        test_features = test_features.to(device)
        test_mask = test_mask.to(device)
        output = model(test_points, test_features, test_mask)
        print(f"测试输出形状: {output.shape}") # 应该输出(batch_size, 1)
        # print(output)
        print(f"测试输出值范围: {output.min().item():.4f} ~ {output.max().item():.4f}") # log_energy


    # ======== 训练模型 ========
    save_path = "/home/server/projects/energy_reconstruction/best_model_60_150_1110.pt"

    print("开始训练...")
    train_losses, val_losses, save_path = train_model(model, train_loader, val_loader, num_epochs=500, save_path=save_path)
    
    # ======== 评估模型 ========
    print("评估模型...")
   #  model.load_state_dict(torch.load("best_model.pt")) #不需要重新train，直接evaluate
   #  model.eval()
    log_pred, log_energies = evaluate_model(model, test_loader, save_path) 

    return model, log_pred, log_energies, train_losses, val_losses
   
# 运行
if __name__ == "__main__":
    model, pred, true, train_losses, val_losses = main()

    # 保存loss曲线
    import json
    with open("loss_log_60_150_1110.json", "w") as f:
        json.dump({
            "train_loss": train_losses,
            "val_loss": val_losses
        }, f)
    print("✅ 已保存训练日志到 loss_log_60_150_1110.json")

    # 或者立刻画图（可选）
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/server/projects/energy_reconstruction/fig/loss_curve.png", dpi=300)
    print("📈 已保存图像 loss_curve.png")
