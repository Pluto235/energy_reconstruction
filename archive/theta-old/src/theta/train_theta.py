import torch
import torch.nn as nn
import copy
import numpy as np

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-3,
    patience=6,
    min_delta=1e-4,
    grad_clip=5.0,
    alpha=0.7,       # 能量平衡权重强度
    beta=0.5,        # Huber loss 的 β
    save_path=None
):
    """
    ParticleNet 训练函数（带能量分布平衡权重 + 早停 + 学习率调度 + 梯度裁剪）
    模型输入与输出均为 log10(E)
    """

    # ---------- 初始化 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    if torch.cuda.device_count() > 1:
        print(f"🔸 使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)
    model.to(device)

    # ---------- 从训练数据估计 logE 分布 ----------
    print("📊 估计训练集 log(E) 分布 ...")
    all_logE = []
    for _, _, _, _, energies in train_loader: # 在整个train数据上的分布
        all_logE.append(energies)
    all_logE = torch.cat(all_logE, dim=0).cpu().numpy()
    mu, sigma = np.mean(all_logE), np.std(all_logE)
    print(f"   → μ = {mu:.4f}, σ = {sigma:.4f}")

    # ---------- 定义损失函数（带权重） ----------
    # def weighted_mse(pred, true, mu, sigma, alpha=0.7):
    #     """
    #     能量加权版:
    #     w = exp(α * (x - μ)^2 / (2σ^2))
    #     """
    #     weights = torch.exp(alpha * ((true - mu) ** 2) / (2 * sigma ** 2))
    #     weights = weights / weights.mean()  # 归一化
    #     return (weights * (pred - true) **2).mean()

    # ---------- 定义损失函数（基于直方图分布反权重） ----------
    print("⚖️ 构建基于直方图的能量加权损失 ...")
    
    # 计算 logE 的统计直方图
    hist, edges = np.histogram(all_logE, bins=50)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prob = hist / np.sum(hist)
    inv_prob = 1.0 / (prob + 1e-8)
    inv_prob = inv_prob / np.mean(inv_prob)  # 归一化，使平均权重为 1
    
    # 转为 tensor 并放到设备上
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
    inv_prob_t = torch.tensor(inv_prob, dtype=torch.float32, device=device)
    
    def histogram_weighted_mse(pred, true):
        """
        基于 logE 的直方图反频率加权均方误差损失
        pred, true: log10(E)
        """
        x = true.squeeze(-1)
        idx = torch.bucketize(x, centers_t)
        idx = torch.clamp(idx, 1, len(centers_t) - 1)
    
        x0 = centers_t[idx - 1]
        x1 = centers_t[idx]
        y0 = inv_prob_t[idx - 1]
        y1 = inv_prob_t[idx]
    
        # 线性插值获得权重
        weights = y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-8)
        weights = weights / (weights.mean() + 1e-8)  # 归一化
    
        return torch.mean(weights * (pred - true) ** 2)
    
    criterion = histogram_weighted_mse

    # ---------- 优化器与调度 ----------
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=10, factor=0.5, verbose=True
    # )
    # 余弦学习率改进方案
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # ---------- 训练循环 ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, (points, features, mask, costheta, logE_true) in enumerate(train_loader):
            points, features, mask, costheta, logE_true = (
                points.to(device, non_blocking=True),
                features.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                costheta.to(device, non_blocking=True),
                logE_true.to(device, non_blocking=True),
            )

            optimizer.zero_grad(set_to_none=True)
            logE_pred = model(points, features, mask, costheta)
            loss = criterion(logE_pred, logE_true)
            loss.backward()

            # 梯度裁剪：控制梯度爆炸
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 200 == 0:
                print(
                    f"Epoch[{epoch}/{num_epochs}] Batch[{batch_idx}/{len(train_loader)}] "
                    f"TrainLoss: {loss.item():.6f}"
                )

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, features, mask, costheta, logE_true in val_loader:
                points, features, mask, costheta, logE_true = (
                    points.to(device, non_blocking=True),
                    features.to(device, non_blocking=True),
                    mask.to(device, non_blocking=True),
                    costheta.to(device, non_blocking=True),
                    logE_true.to(device, non_blocking=True),
                )
                logE_pred = model(points, features, mask, costheta)
                val_loss += criterion(logE_pred, logE_true).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step() # 更新学习率
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"📘 Epoch {epoch:03d}/{num_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.6e}"
        )

        # ---------- Early Stopping ----------
        if save_path is None:
            print("⚠️ 没有正确传入保存路径！")
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_wts, save_path)
            print(f"✅ Val loss improved to {val_loss:.6f}, model saved to '{save_path}'")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(best_model_wts)
    print(f"🏁 Training completed. Best Val Loss: {best_val_loss:.6f}")
    return train_losses, val_losses, save_path # logE训练的损失
