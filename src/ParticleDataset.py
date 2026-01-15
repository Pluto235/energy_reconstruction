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
from multiprocessing import Pool
import psutil

from .EdgeConv import process_features

class ParticleDataset(Dataset):
    '''
    Dateset 需要做到
    - ✅并行加载root文件
    - ✅提取branch[vx vy vt vq]和target_branch[mc_energy]
    - ✅根据process_conditions 处理数据,主要是为了剔除离群值
    - ✅由于每个event的hit数不一样，需要填充/随机采样数据到max_points
    - ✅对点云做去中心化处理(vx, vy) - (xc, yc)
    - 对数据进行标准化，(vq, vt)需要标准化,全局标准化? 还是逐事例的标准化！
    - ✅对能量标签取log10，但是不做标准化
    - 是否要剔除部分超高能>1TeV和超低能<20GeV的数据？？
    
    input:
    - root_files:    文件根目录，文件名列表
    - branch:        需要读取的特征[vx vy vt vq]
    - target_branch: 标签[mc_energy]
    - processing_conditions:预处理条件
    - max_points:     固定的数据长度
    
    return
    - points:     去中心化之后的点云的二维坐标(vx-xc, vy-yc)
    - features:   预处理和归一化后的点云特征(vq, vt)
    - mask:       padding和截断造成的mask
    - log_energy: 预处理和log之后的真实能量 mc_energy
    '''
    def __init__(self, root_files, branches, target_branch, processing_conditions,  max_points=256):
        self.branches = branches
        self.target_branch = target_branch
        self.processing_conditions = processing_conditions
        self.max_points = max_points

        # 并行加载 ROOT 文件 增加了进程数到16或32,64
        with Pool(min(64, os.cpu_count())) as pool:
            
            self.data = pool.starmap(
                self._load_file,
                [(f, branches, target_branch, processing_conditions) for f in root_files]
            )

        # 展平所有文件的数据
        self.data = [item for sublist in self.data for item in sublist]
        self.indices = torch.randperm(len(self.data)).tolist()
        print(f"✅ Loaded {len(self.data)} events from {len(root_files)} files")

    @staticmethod
    def _load_file(file_path, branches, target_branch, processing_conditions):
        """从单个ROOT文件中提取数据, 并筛选vqsamp>0 hits + mc_energy>100"""
        try:
            with uproot.open(file_path) as f:
                tree = f["t_eventout;1"]
                
                full_branches = [
                "n", "nfit0", "nfit", "nfitb", "vnfit", "fitstat", "nrange",
                "nv", "vflag", "vidmc", "vx", "vy", "vt", "vnpe", "vq",
                "vqsamp", "theta", "phi", "xc", "yc", "dcedge", "dcedgepool",
                "istationcore", "ccindex", "chi2", "rmds", "pincness",
                "compactness", "f5w", "mc_weight", "mc_pid", "mc_energy",
                "mc_theta", "mc_phi", "mc_xc", "mc_yc", "mc_dangle",
                "mc_dcore"
                  ]
    
                # ✅ 增加读取 vqsamp 分支
                #　read_branches = branches + target_branch + ["xc", "yc", "vqsamp"]
                arrays = tree.arrays(full_branches, library="np")
                
                n_total = len(next(iter(arrays.values())))
                if n_total == 0:
                    print(f"⚠️ 文件 {file_path} 无事件，跳过。")
                    return []
    
                # ✅ 计算每个事件的 vqsamp 非零比例（忽略 0）
                vqsamp_nonzero_ratio = np.array([
                    np.count_nonzero(v > 0) / len(v) if len(v) > 0 else 0
                    for v in arrays["vqsamp"]
                ])
    
                # ✅ 取出 mc_energy（注意 target_branch 是列表）
                mc_energy = arrays[target_branch[0]]

                # 取出pincness
                pincness = arrays["pincness"]
                compactness = arrays["compactness"]

                # 筛选重建芯位距离水池
                dcedge = arrays["dcedge"]

                # 筛选mc_dangle
                mc_dangle = arrays["mc_dangle"]

                # 筛选重建zenith 即 theta 
                theta = arrays["theta"]
    
                # ✅ 构造联合筛选条件
                mask_evt = (vqsamp_nonzero_ratio > 0.5) & (mc_energy > 100) & (pincness < 1.1)  & (dcedge >= 20) & (mc_dangle < 0.05236) & (theta<0.524)
    
                # ✅ 应用到所有分支
                for key in arrays.keys():
                    arrays[key] = arrays[key][mask_evt]
    
                n_kept = mask_evt.sum()
                print(f"🔹 {file_path}: 保留 {n_kept}/{n_total} 个事件 (vqsamp>0.5 & E>100 & pincness<1.1 & dcedge 20m & mc_dangle 3° & theta 30°)")

                n_events = len(next(iter(arrays.values())))
                results = []
    
                for i in range(n_events):  # 对每个event
                    # 保存完整的原始信息
                    raw_info = {key: arrays[key][i] for key in full_branches}

                    # (1) 原始特征矩阵
                    features = np.column_stack([arrays[b][i] for b in branches])
                    features = process_features(features, processing_conditions)  # 剔除离群值
    
                    # (2) 点云中心化: (vx, vy) - (xc, yc)
                    vx, vy = features[:, 0], features[:, 1]
                    xc, yc = arrays["xc"][i], arrays["yc"][i]
                    points = np.column_stack([vx - xc, vy - yc])
    
                    # (3) 对 (vq, vt) 做标准化（列 2,3）
                    vq = features[:, 3]
                    vt = features[:, 2]
                    vq = (vq - np.mean(vq)) / (np.std(vq) + 1e-8)
                    vt = (vt - np.mean(vt)) / (np.std(vt) + 1e-8)
                    norm_features = np.column_stack([vq, vt])
    
                    # (4) 能量标签取 log10
                    target = arrays["mc_energy"][i]
                    log_energy = np.log10(target) if target > 0 else 0.0
    
                    # results.append((points, norm_features, log_energy))
                    results.append({
                        "file": file_path,
                        "event_idx": i,
                    
                        # ===== 原始数据 =====
                        "raw": raw_info,
                    
                        # ===== 预处理数据 =====
                        "processed": {
                            "points": points,
                            "features": norm_features,
                            "log_energy": log_energy
                        }
                    })

    
                return results
    
        except Exception as e:
            print(f"⚠️ 文件 {file_path} 读取失败: {e}")
            return []


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        record = self.data[real_idx]

        proc = record['processed'] # 只提取processed的返回
        points = proc['points']
        features = proc['features']
        log_energy = proc['log_energy']

        n_points = len(points)

        # padding 或随机采样
        if n_points > self.max_points:
            idxs = np.random.choice(n_points, self.max_points, replace=False)
            points = points[idxs]
            features = features[idxs]
            mask = np.ones(self.max_points)
        else:
            pad_len = self.max_points - n_points
            pad_points = np.zeros((pad_len, 2))
            pad_features = np.zeros((pad_len, features.shape[1]))
            points = np.vstack([points, pad_points])
            features = np.vstack([features, pad_features]) # 这里的features已经是load-file输出的norm_features了
            mask = np.concatenate([np.ones(n_points), np.zeros(pad_len)])
         
        points = torch.tensor(points, dtype=torch.float32).T # 转置维度: (N, 2) → (2, N)
        features = torch.tensor(features, dtype=torch.float32).T # 转置维度: (N, 2) → (2, N)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # 修复mask的维度(B, N) -> (B, 1, N)
        log_energy = torch.tensor(log_energy, dtype=torch.float32).unsqueeze(-1) # [B, 1]
        
        return points, features, mask, log_energy
        
