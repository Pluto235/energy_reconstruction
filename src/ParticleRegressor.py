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

from .EdgeConv import EdgeConvBlock

class ParticleNetRegressor(nn.Module):
    '''
    主要网络
    input 
    - points(B, 2, N) features(B, C, N) mask(B, 1, N)
    
    model:
    EdgeConv block *3
    fusion block
    Global avgPool
    FCs
    
    output:
    pred_log_energy
    '''
    def __init__(self, input_dims: int, 
                 conv_params: List[Tuple[int, Tuple[int, ...]]] = [(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                 fc_params: List[Tuple[int, float]] = [(256, 0.1), (128, 0.1)],
                 use_fusion: bool = True):
        super().__init__()
        self.use_fusion = use_fusion

        # 输入特征批归一化 input_dims=2 (vq, vt)
        self.bn_fts = nn.BatchNorm1d(input_dims)
        
        # EdgeConv 块
        self.edge_convs = nn.ModuleList()
        for idx, (k, channels) in enumerate(conv_params):  # (B, 2, N, k) -> (B, 64, N, k) -> (B, 128, N, k) -> (B, 256, N, k) 
            in_feat = input_dims if idx == 0 else conv_params[idx-1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k, in_feat, channels))
        
        # 特征融合
        if use_fusion:
            fusion_in = sum(x[1][-1] for x in conv_params)
            self.fusion_block = nn.Sequential(
                nn.Conv1d(fusion_in, 256, kernel_size=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU() # 这个relu层没有可能出问题
            )
            fc_input_dim = 256
        else:
            fc_input_dim = conv_params[-1][1][-1]
        
        # 回归头(全连接层）
        fcs = []
        for idx, (out_dim, dropout) in enumerate(fc_params):
            in_dim = fc_input_dim if idx == 0 else fc_params[idx-1][0]
            fcs.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        # 输出层 - 回归任务输出1个值（能量）
        fcs.append(nn.Linear(fc_params[-1][0], 1))
        self.fc = nn.Sequential(*fcs)

        # # 去除relu层
        # fcs = []
        # for idx, (out_dim, dropout) in enumerate(fc_params):
        #     in_dim = fc_input_dim if idx == 0 else fc_params[idx - 1][0]
        #     fcs.append(nn.Linear(in_dim, out_dim))
        #     # 只在不是最后一层时添加激活
        #     if idx != len(fc_params) - 1:
        #         fcs.append(nn.ReLU())
        #         fcs.append(nn.Dropout(dropout))
        
        # # 最后一层线性输出 log10(E)
        # fcs.append(nn.Linear(fc_params[-1][0], 1))
        # self.fc = nn.Sequential(*fcs)


        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, points: torch.Tensor, features: torch.Tensor, mask: torch.Tensor):
        """
        前向传播
        输入:
            points: (B, 2, N) - 坐标 [x, y]
            features: (B, 2, N) - 特征 [电荷, 时间]
            mask: (B, 1, N) - 掩码
        输出:
            energy: (B, 1) - 预测能量
        """
        if mask.dim() == 2:  # 防止mask是(B, N)
            print("mask维度有问题")
            mask = mask.unsqueeze(1)
        # 应用掩码
        points = points * mask
        features = features * mask
        coord_shift = (mask == 0) * 1e9  # 填充点坐标加1e9，使其在KNN中被忽略
        
        # 特征归一化
        fts = self.bn_fts(features) * mask
        outputs = []
        
        # 多层EdgeConv
        for idx, conv in enumerate(self.edge_convs):
            # 第一层用坐标构建图，后续用特征
            pts = (points if idx == 0 else fts) + coord_shift # 坐标加上偏移
            fts = conv(pts, fts) * mask
            # pts: (B, 2, N) - 用于构建KNN图
				# fts: (B, 2, N) -> (B, 64, N) -> (B, 128, N)  -> (B, 256, N) - 三层卷积层，特征维度提升
            if self.use_fusion:
                outputs.append(fts)
        
        # 拼接三个层次的输出, 特征融合
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask # (B, 64+128+256, N) -> (B, 256, N)
        
        # 全局平均池化 - 只对有效点求平均
        x = fts.sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1) # (B, 256, N) -> 输出: (B, 256)
        
        # 回归预测
        logE_pred = self.fc(x) # (B, 256) -> (B, 256) -> (B, 128) -> (B, 1)
        return logE_pred