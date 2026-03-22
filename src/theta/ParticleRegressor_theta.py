import torch
import torch.nn as nn
from typing import Tuple, List

from src.common.EdgeConv import EdgeConvBlock

class ParticleNetRegressor(nn.Module):
    '''
    主要网络
    input 
    - points(B, 2, N) features(B, C, N) mask(B, 1, N)
    + costheta(B,) or (B,1)
    + true_core_xy(B, 2)
    model:
    EdgeConv block *3
    fusion block
    Global avgPool
    (theta embedding + concat)
    FCs
    output:
    pred_log_energy
    '''
    def __init__(self, input_dims: int,
                 conv_params: List[Tuple[int, Tuple[int, ...]]] = [(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))],
                 fc_params: List[Tuple[int, float]] = [(256, 0.1), (128, 0.1)],
                 use_fusion: bool = True,
                 theta_embed_dim: int = 16,
                 theta_embed_dropout: float = 0.0,
                 core_embed_dim: int = 16,
                 core_embed_dropout: float = 0.0,
                 ):
        super().__init__()
        self.use_fusion = use_fusion
        self.theta_embed_dim = theta_embed_dim
        self.core_embed_dim = core_embed_dim

        # 输入特征批归一化 input_dims=2 (vq, vt)
        self.bn_fts = nn.BatchNorm1d(input_dims)

        # EdgeConv 块
        self.edge_convs = nn.ModuleList()
        for idx, (k, channels) in enumerate(conv_params):
            in_feat = input_dims if idx == 0 else conv_params[idx-1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k, in_feat, channels))

        # 特征融合
        if use_fusion:
            fusion_in = sum(x[1][-1] for x in conv_params)
            self.fusion_block = nn.Sequential(
                nn.Conv1d(fusion_in, 256, kernel_size=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
            base_fc_input_dim = 256
        else:
            base_fc_input_dim = conv_params[-1][1][-1]

        # Event-level theta embedding
        if theta_embed_dim > 0:
            self.theta_mlp = nn.Sequential(
                nn.Linear(1, theta_embed_dim),
                nn.GELU(),
                nn.Dropout(theta_embed_dropout),
            )
        else:
            self.theta_mlp = None

        # Event-level true-core embedding using normalized (mc_xc, mc_yc).
        if core_embed_dim > 0:
            self.core_mlp = nn.Sequential(
                nn.Linear(2, core_embed_dim),
                nn.GELU(),
                nn.Dropout(core_embed_dropout),
            )
        else:
            self.core_mlp = None

        fc_input_dim = base_fc_input_dim + max(theta_embed_dim, 0) + max(core_embed_dim, 0)

        # 回归头(全连接层）
        fcs = []
        for idx, (out_dim, dropout) in enumerate(fc_params):
            in_dim = fc_input_dim if idx == 0 else fc_params[idx-1][0]
            fcs.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        fcs.append(nn.Linear(fc_params[-1][0], 1))
        self.fc = nn.Sequential(*fcs)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                points: torch.Tensor,
                features: torch.Tensor,
                mask: torch.Tensor,
                costheta: torch.Tensor = None,
                true_core_xy: torch.Tensor = None,
                ):
        """
        前向传播
        输入:
            points:   (B, 2, N) - 坐标 [x, y]
            features: (B, C, N) - 特征 [电荷, 时间] (你这里 C=2)
            mask:     (B, 1, N) - 掩码
            costheta: (B,) or (B,1) - 事件级全局变量
            true_core_xy: (B, 2) - 归一化后的真实芯位坐标
        输出:
            logE_pred: (B, 1) - 预测能量(log)
        """
        if mask.dim() == 2:  # 防止mask是(B, N)
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
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)

        # 拼接三个层次的输出, 特征融合
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        # 全局平均池化 - 只对有效点求平均
        x = fts.sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)  # (B, 256)

        cond_embeddings = []

        if self.theta_mlp is not None:
            if costheta is None:
                raise ValueError("theta_embed_dim>0 but costheta is None. Please pass costheta to forward().")
            if costheta.dim() == 1:
                costheta = costheta.unsqueeze(1)  # (B,1)

            cond_embeddings.append(self.theta_mlp(costheta.float()))

        if self.core_mlp is not None:
            if true_core_xy is None:
                raise ValueError("core_embed_dim>0 but true_core_xy is None. Please pass true_core_xy to forward().")
            if true_core_xy.dim() == 1:
                true_core_xy = true_core_xy.unsqueeze(0)
            true_core_xy = true_core_xy.reshape(true_core_xy.shape[0], -1)
            if true_core_xy.shape[1] != 2:
                raise ValueError(f"Expected true_core_xy with shape (B,2), got {tuple(true_core_xy.shape)}")

            cond_embeddings.append(self.core_mlp(true_core_xy.float()))

        if cond_embeddings:
            x = torch.cat([x] + cond_embeddings, dim=1)

        # 回归预测
        logE_pred = self.fc(x)
        return logE_pred
