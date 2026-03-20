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
    bins_hist=50,
    eps=1e-8,
    save_path=None
):
    """
    ParticleNet 训练函数
    - 使用 mc_weight 构建 weighted histogram (Crab 等效能谱)
    - loss 使用 inv_prob_weighted(logE_true) 作为权重（不再额外乘 mc_weight，避免双重加权爆炸）
    - 修复 CosineAnnealingLR.step 用法
    - 避免 batch 内归一化导致的权重抖动
    """

    # ---------- 初始化 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    if torch.cuda.device_count() > 1:
        print(f"🔸 使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)
    model.to(device)

    # ---------- 从训练数据估计 logE 分布（用 mc_weight 做 weighted histogram） ----------
    print("📊 估计训练集 log(E) 分布（mc_weight 加权 -> Crab 等效能谱）...")

    all_logE = []
    all_w = []
    for batch in train_loader:
        # batch: points, features, mask, logE_true, mc_weight
        logE = batch[3]
        w = batch[4]
        all_logE.append(logE)
        all_w.append(w)

    all_logE = torch.cat(all_logE, dim=0).cpu().numpy().reshape(-1)
    all_w    = torch.cat(all_w, dim=0).cpu().numpy().reshape(-1)

    # 保险：去掉非正权重/NaN
    valid = np.isfinite(all_logE) & np.isfinite(all_w) & (all_w > 0)
    all_logE = all_logE[valid]
    all_w    = all_w[valid]

    mu, sigma = np.mean(all_logE), np.std(all_logE)
    print(f"   → μ = {mu:.4f}, σ = {sigma:.4f}, N(valid)={len(all_logE)}")

    # weighted histogram：用 mc_weight 当作 weights
    hist_w, edges = np.histogram(all_logE, bins=bins_hist, weights=all_w)
    prob_w = hist_w / (np.sum(hist_w) + eps)

    # 反频率：稀有(在 Crab 等效谱里)的能段给更大权重
    inv_prob_w = 1.0 / (prob_w + eps)

    # 全局归一化：让平均权重约为 1（在“bin 概率意义”上）
    inv_prob_w = inv_prob_w / (np.mean(inv_prob_w) + eps)

    # 转 tensor
    edges_t = torch.tensor(edges, dtype=torch.float32, device=device)            # (bins+1,)
    inv_prob_t = torch.tensor(inv_prob_w, dtype=torch.float32, device=device)    # (bins,)

    def inv_prob_weight_from_logE(logE_true: torch.Tensor) -> torch.Tensor:
        """
        根据 logE_true 在 weighted histogram 的 bin 中取 inv_prob 权重
        logE_true: (B,1) or (B,)
        return: (B,) 权重
        """
        x = logE_true.squeeze(-1)  # (B,)
        # bucketize 返回 [0..len(edges)]，我们要映射到 bin index [0..bins-1]
        # torch.bucketize: index i 满足 edges[i-1] < x <= edges[i]
        idx = torch.bucketize(x, edges_t) - 1  # 转成 bin index
        idx = torch.clamp(idx, 0, inv_prob_t.numel() - 1)
        return inv_prob_t[idx]

    def histogram_weighted_mse_mcshape(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        loss = mean( w_bin(logE_true) * (pred-true)^2 )
        w_bin 来自 mc_weight 加权后的能谱(等效 Crab)
        """
        w = inv_prob_weight_from_logE(true)  # (B,)
        diff2 = (pred.squeeze(-1) - true.squeeze(-1)) ** 2
        # 注意：不做 batch 内 w.mean() 归一化，避免抖动；全局已归一化过
        return torch.mean(w * diff2)

    criterion = histogram_weighted_mse_mcshape

    # ---------- 优化器与调度 ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # ---------- 训练循环 ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, (points, features, mask, logE_true, mc_weight) in enumerate(train_loader):
            points = points.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logE_true = logE_true.to(device, non_blocking=True)
            # mc_weight 此处不进 loss（我们已经用它构建了等效能谱），留着不用也行
            # mc_weight = mc_weight.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logE_pred = model(points, features, mask)
            loss = criterion(logE_pred, logE_true)
            loss.backward()

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
            for points, features, mask, logE_true, mc_weight in val_loader:
                points = points.to(device, non_blocking=True)
                features = features.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                logE_true = logE_true.to(device, non_blocking=True)

                logE_pred = model(points, features, mask)
                val_loss += criterion(logE_pred, logE_true).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ✅ 修复：CosineAnnealingLR 不吃 val_loss
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"📘 Epoch {epoch:03d}/{num_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.6e}"
        )

        # ---------- Early Stopping ----------
        if save_path is None:
            print("⚠️ 没有正确传入保存路径！")
        if (save_path is not None) and (val_loss < best_val_loss - min_delta):
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
    return train_losses, val_losses, save_path
