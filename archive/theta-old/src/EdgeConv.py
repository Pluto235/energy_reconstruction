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

# 1. KNN 和图特征函数
def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """计算K近邻索引"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k+1, dim=-1)[1][:, :, 1:]

def get_graph_feature(x: torch.Tensor, k: int, idx: torch.Tensor):
    """为每个点构建局部图特征"""
    B, D, N = x.size() # B是批次大小，D是特征维度，N是点的数量
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    
    x = x.transpose(2, 1).contiguous()
    neighbor_features = x.view(B*N, D)[idx, :].view(B, N, k, D) # 根据索引获取邻居特征，(B, N, k, D) -> (B, D, N, k)
    neighbor_features = neighbor_features.permute(0, 3, 1, 2).contiguous()
    
    x = x.permute(0, 2, 1).unsqueeze(-1).repeat(1, 1, 1, k) # copy k times x: (B, D, N, k)
    return torch.cat([x, neighbor_features - x], dim=1) # cat: (B, 2*D, N, k)

# 2. EdgeConvBlock 定义
class EdgeConvBlock(nn.Module):
    '''
    input              (B, 2, N)
    knn               -> (B, 2, N, k)
    get_graph_feature -> (B, 4, N, k)
    Edgeconv          ->(B, 4, N, k)->(B, 64, N, k)->(B, 64, N, k)->(B, 64, N, k)
    pooling           -> (B, 64, N)
     '''
    def __init__(self, k: int, in_feat: int, out_feats: Tuple[int, ...]):
        super().__init__()
        self.k = k
        self.num_layers = len(out_feats)
        
      #   # 创建卷积层
      #   self.convs = nn.ModuleList()
      #   for i in range(self.num_layers): # 每一个
      #       in_ch = 2 * in_feat if i == 0 else out_feats[i-1]
      #       self.convs.append(nn.Conv2d(in_ch, out_feats[i], kernel_size=1, bias=False)) 
        
      #   # 创建BatchNorm和激活层
      #   self.bns = nn.ModuleList()
      #   self.acts = nn.ModuleList()
      #   for i in range(self.num_layers):
      #       self.bns.append(nn.BatchNorm2d(out_feats[i])) # BN和Relu不改变维度
      #       self.acts.append(nn.ReLU(inplace=True))

        # 创建卷积层 ，BN层，激活层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        
        for i in range(self.num_layers): #与上面写在两个循环下的写法等价
            in_ch = 2 * in_feat if i == 0 else out_feats[i-1]
            self.convs.append(nn.Conv2d(in_ch, out_feats[i], kernel_size=1, bias=False)) # 2dconv (B, 2*D, N, k) -> (B, 64, N, k)
            self.bns.append(nn.BatchNorm2d(out_feats[i])) # BN和Relu不改变维度
            self.acts.append(nn.ReLU(inplace=True))


        # 跳跃连接
        if in_feat != out_feats[-1]:
            self.sc = nn.Sequential(
                nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False),
                nn.BatchNorm1d(out_feats[-1])
            )
        else:
            self.sc = None
        
        self.sc_act = nn.ReLU(inplace=True)

    def forward(self, points: torch.Tensor, features: torch.Tensor):
        # 构建KNN图
        idx = knn(points, self.k)
        # 提取图特征
        x = get_graph_feature(features, self.k, idx) # 2dconv (B, 2*D, N, k) 
        
        # 多层卷积处理，每个edgeconv下进行3次卷积是为了增强非线性表达能力（因为有多次的激活函数）
        for i in range(self.num_layers): # (B, 64, N, k) -> (B, 64, N, k) -> (B, 64, N, k) 
            x = self.convs[i](x)  
            x = self.bns[i](x)
            x = self.acts[i](x)
        
        # 池化: 对k维度求平均
        fts = x.mean(dim=-1) # (B, 64, N, k) -> (B, 64, N)
        
        # 跳跃连接
        sc = self.sc(features) if self.sc else features
        return self.sc_act(sc + fts)


def process_features(features, processing_conditions):
    features = np.array(features, dtype=np.float32)
    for i, cond in enumerate(processing_conditions):
        #features[:, i] = (features[:, i] - cond.get('subtract', 0)) * cond.get('multiply', 1)
        #features[:, i] = np.clip(features[:, i], cond.get('min', -5), cond.get('max', 5))
        col = features[:, i].copy()
        
        # 对坐标特征（vx, vy）仅做线性变换，不做截断和标准化
        if i < 2:  # 前两列为坐标
            col = (col - cond['subtract']) * cond['multiply']
        else:
            # 其他特征：线性变换 + 截断 + 标准化
            col = (col - cond['subtract']) * cond['multiply']
            col = np.clip(col, cond['min'], cond['max'])
            #col = (col - np.mean(col)) / (np.std(col) + 1e-8)
        
        features[:, i] = col
    return features