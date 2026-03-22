import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # ===== bin-weight for imbalance =====
    bins_hist=50,
    eps=1e-8,
    min_count=5,            # 防止极低计数 bin 权重爆炸
    max_weight=None,        # 可选：对 w_bin 做 clip，例如 10.0；None 不 clip

    save_path=None,

    # ===== loss mode =====
    loss_mode="huber",      # "mse" | "huber" | "rel"
    huber_delta=0.2,        # huber on logE 的 delta（logE误差量级常用 0.2~0.3）
    rel_delta=0.3,          # huber on relative error 的 delta（rel 量级常用 0.2~0.5）
    rel_squared=False,      # True: rel^2; False: huber(rel)
):
    """
    训练函数：先用训练集 logE_true 构建分bin逆频权重 w_bin（修正能谱不均），
    再在此基础上选择不同的 loss 形式（mse / huber / rel）。

    - histogram: unweighted counts on logE_true
    - w_bin = 1 / P(bin)  (并做归一化，使概率意义下平均权重 ~ 1)
    - loss = sum(w_bin * per_event_loss) / sum(w_bin)

    注意：兼容旧 batch 结构，也兼容新结构
    (points, features, mask, costheta, true_core_xy, logE_true, mc_weight)。
    """

    # ---------- device & DP ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    use_dp = False
    if torch.cuda.device_count() > 1:
        print(f"🔸 使用 {torch.cuda.device_count()} 个GPU (DataParallel)")
        model = nn.DataParallel(model)
        use_dp = True
    model.to(device)

    # ---------- build logE histogram weights ----------
    print("📊 估计训练集 log(E) 分布（unweighted histogram）并构建分bin逆频权重...")

    def _unpack_batch(batch):
        if len(batch) == 7:
            points, features, mask, costheta, true_core_xy, logE_true, mc_weight = batch
        elif len(batch) == 6:
            points, features, mask, costheta, logE_true, mc_weight = batch
            true_core_xy = None
        elif len(batch) == 5:
            points, features, mask, logE_true, mc_weight = batch
            costheta = None
            true_core_xy = None
        else:
            raise ValueError(f"Unexpected batch size: got {len(batch)} items, expect 5, 6 or 7.")

        return points, features, mask, costheta, true_core_xy, logE_true, mc_weight

    all_logE = []
    for batch in train_loader:
        logE = _unpack_batch(batch)[5]
        all_logE.append(logE)

    all_logE = torch.cat(all_logE, dim=0).cpu().numpy().reshape(-1)
    all_logE = all_logE[np.isfinite(all_logE)]

    if all_logE.size == 0:
        raise RuntimeError("No valid logE in train_loader to build histogram weights.")

    mu, sigma = float(np.mean(all_logE)), float(np.std(all_logE))
    print(f"   → μ = {mu:.4f}, σ = {sigma:.4f}, N(valid)={int(all_logE.size)}")

    hist, edges = np.histogram(all_logE, bins=bins_hist)
    hist = hist.astype(np.float64)

    # clamp low-count bins to avoid huge weights
    hist = np.maximum(hist, float(min_count))

    prob = hist / (np.sum(hist) + eps)      # P(bin)
    inv_prob = 1.0 / (prob + eps)           # 1/P(bin)

    # normalize so that average weight in "probability sense" is ~1:
    # E[w] = sum P(bin) * (1/P(bin)) = bins (if no eps/clamp). With eps/clamp, normalize explicitly.
    mean_w = float(np.sum(prob * inv_prob))
    if not np.isfinite(mean_w) or mean_w <= 0:
        mean_w = 1.0
    inv_prob = inv_prob / mean_w

    edges_t = torch.tensor(edges, dtype=torch.float32, device=device)          # (bins+1,)
    inv_prob_t = torch.tensor(inv_prob, dtype=torch.float32, device=device)    # (bins,)

    def w_bin_from_logE(logE_true: torch.Tensor) -> torch.Tensor:
        """
        logE_true: (B,1) or (B,)
        return w_bin: (B,)
        """
        x = logE_true.squeeze(-1)
        idx = torch.bucketize(x, edges_t) - 1
        idx = torch.clamp(idx, 0, inv_prob_t.numel() - 1)
        w = inv_prob_t[idx]
        if max_weight is not None:
            w = torch.clamp(w, max=float(max_weight))
        return w

    def weighted_reduce(per_event: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.sum(w * per_event) / (torch.sum(w) + 1e-12)
    
    # ---------- choose loss ----------
    loss_mode = str(loss_mode).lower().strip()
    if loss_mode not in ["mse", "huber", "rel"]:
        raise ValueError(f"loss_mode must be one of ['mse','huber','rel'], got: {loss_mode}")

    print(f"🧪 loss_mode={loss_mode} | huber_delta={huber_delta} | rel_delta={rel_delta} | rel_squared={rel_squared}")
    if max_weight is not None:
        print(f"🧯 max_weight clip = {max_weight}")

    def criterion(pred_logE: torch.Tensor, true_logE: torch.Tensor) -> torch.Tensor:
        """
        pred_logE/true_logE: (B,1) or (B,)
        """
        w = w_bin_from_logE(true_logE)

        if loss_mode == "mse":
            err = pred_logE.squeeze(-1) - true_logE.squeeze(-1)
            per = err ** 2
            return weighted_reduce(per, w)

        if loss_mode == "huber":
            err = pred_logE.squeeze(-1) - true_logE.squeeze(-1)
            per = F.smooth_l1_loss(err, torch.zeros_like(err), beta=huber_delta, reduction="none")
            return weighted_reduce(per, w)

        # loss_mode == "rel"
        pred_logE_ = pred_logE.squeeze(-1)
        true_logE_ = true_logE.squeeze(-1)
        E_pred = torch.pow(10.0, pred_logE_)
        E_true = torch.pow(10.0, true_logE_)

        rel = (E_pred - E_true) / (E_true + 1e-12)  # relative error

        if rel_squared:
            per = rel ** 2
        else:
            per = F.smooth_l1_loss(rel, torch.zeros_like(rel), beta=rel_delta, reduction="none")

        return weighted_reduce(per, w)

    # ---------- optimizer & scheduler ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # ---------- training loop ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            points, features, mask, costheta, true_core_xy, logE_true, _ = _unpack_batch(batch)

            points = points.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            costheta = costheta.to(device, non_blocking=True) if costheta is not None else None
            true_core_xy = true_core_xy.to(device, non_blocking=True) if true_core_xy is not None else None
            logE_true = logE_true.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logE_pred = model(points, features, mask, costheta, true_core_xy)

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

        # ---------- validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                points, features, mask, costheta, true_core_xy, logE_true, _ = _unpack_batch(batch)

                points = points.to(device, non_blocking=True)
                features = features.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                costheta = costheta.to(device, non_blocking=True) if costheta is not None else None
                true_core_xy = true_core_xy.to(device, non_blocking=True) if true_core_xy is not None else None
                logE_true = logE_true.to(device, non_blocking=True)

                logE_pred = model(points, features, mask, costheta, true_core_xy)
                val_loss += criterion(logE_pred, logE_true).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"📘 Epoch {epoch:03d}/{num_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.6e}"
        )

        # ---------- early stopping + save ----------
        if save_path is None:
            print("⚠️ 没有正确传入保存路径！(save_path=None)")

        if (save_path is not None) and (val_loss < best_val_loss - min_delta):
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            state_to_save = model.module.state_dict() if use_dp else model.state_dict()
            torch.save(state_to_save, save_path)
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
