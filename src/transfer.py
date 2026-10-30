"""迁移学习对比：手写模型 vs 预训练 ResNet18（冻结线性头 / 全量微调）。"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18 as tv_resnet18

DATA = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
RESULTS = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEAN_T = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

# 已完成的从零训练结果（Kaggle T4）
FROM_SCRATCH = dict(name="从零手写 ResNet18", test_acc=0.9761, best_val_acc=0.9730,
                    epochs=30, params=11178186, train_time_s=565, device="Kaggle T4")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def prep(X, gidx, img_size: int):
    xb = torch.from_numpy(X[gidx]).float().div_(255.0).permute(0, 3, 1, 2)
    if img_size != xb.shape[-1]:
        xb = F.interpolate(xb, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return (xb - MEAN_T) / STD_T


@torch.no_grad()
def extract_features(backbone, X, y, idx, img_size: int, bs: int = 256):
    backbone.eval()
    feats, labels = [], []
    for i in range(0, len(idx), bs):
        gidx = idx[i:i + bs]
        feats.append(backbone(prep(X, gidx, img_size)))
        labels.append(torch.from_numpy(y[gidx]))
    return torch.cat(feats), torch.cat(labels)


def eval_head(head, feat, y) -> float:
    head.eval()
    with torch.no_grad():
        return float((head(feat).argmax(1) == y).float().mean())


def run_frozen(X, y, tr, va, te, img_size, epochs=60, lr=1e-2):
    """B) 冻结预训练主干，只训练线性分类头。"""
    print("[B] 冻结主干 + 线性头：提取特征中 ...")
    t0 = time.time()
    backbone = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    ftr, ytr = extract_features(backbone, X, y, tr, img_size)
    fva, yva = extract_features(backbone, X, y, va, img_size)
    fte, yte = extract_features(backbone, X, y, te, img_size)
    print(f"[B] 特征维度={ftr.shape}，提取耗时 {time.time() - t0:.0f}s")

    head = nn.Linear(ftr.shape[1], 10)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_state, best_val = None, -1.0
    for e in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(ftr))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            crit(head(ftr[idx]), ytr[idx]).backward()
            opt.step()
        va_acc = eval_head(head, fva, yva)
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if e % 10 == 0 or e == epochs:
            print(f"[B] epoch {e:02d} val_acc={va_acc:.4f}")
    head.load_state_dict(best_state)
    test_acc = eval_head(head, fte, yte)
    return dict(name="预训练冻结 + 线性头", test_acc=test_acc, best_val_acc=best_val,
                epochs=epochs, train_time_s=time.time() - t0, img_size=img_size,
                params=int(sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters())),
                device="CPU")


def run_finetune(X, y, tr, va, te, img_size, epochs, lr, bs, device="cpu"):
    """C) 加载预训练权重，全量微调。"""
    print("[C] 预训练全量微调 ...")
    t0 = time.time()
    net = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, 10)
    net.to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = max(1, int(np.ceil(len(tr) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(idx):
        net.eval()
        correct = total = 0
        for i in range(0, len(idx), bs):
            gidx = idx[i:i + bs]
            xb = prep(X, gidx, img_size).to(device)
            pred = net(xb).argmax(1).cpu().numpy()
            correct += int((pred == y[gidx]).sum())
            total += len(gidx)
        return correct / total

    rows = []
    best_val, best_state = -1.0, None
    for e in range(1, epochs + 1):
        net.train()
        t_ep = time.time()
        perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            gidx = tr[sel]
            xb = prep(X, gidx, img_size).to(device)
            yb = torch.from_numpy(y[gidx]).to(device)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        val_acc = evaluate(va)
        rows.append(dict(epoch=e, val_acc=val_acc, epoch_time=time.time() - t_ep))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        print(f"[C] epoch {e:02d} val_acc={val_acc:.4f} ({time.time() - t_ep:.0f}s)")
    net.load_state_dict(best_state)
    test_acc = evaluate(te)
    return dict(name="预训练全量微调", test_acc=test_acc, best_val_acc=best_val,
                epochs=epochs, train_time_s=time.time() - t0, img_size=img_size,
                params=int(sum(p.numel() for p in net.parameters())), device="CPU"), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    set_seed(42)

    z = np.load(DATA / "eurosat_rgb.npz")
    X, y, si = z["X"], z["y"].astype(np.int64), z["split_idx"]
    tr = np.where(si == 0)[0]
    va = np.where(si == 1)[0]
    te = np.where(si == 2)[0]
    print(f"train={len(tr)} val={len(va)} test={len(te)} img_size={args.img_size}")

    res_frozen = run_frozen(X, y, tr, va, te, args.img_size)
    res_ft, log_rows = run_finetune(X, y, tr, va, te, args.img_size, args.epochs, args.lr, args.batch_size, args.device)

    results = [FROM_SCRATCH, res_frozen, res_ft]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "transfer_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(RESULTS / "transfer_log.csv", index=False, encoding="utf-8")

    # 对比柱状图
    names = [r["name"] for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, accs, color=colors)
    for b, a in zip(bars, accs):
        ax.text(a + 0.15, b.get_y() + b.get_height() / 2, f"{a:.2f}%", va="center", fontsize=10)
    ax.set_xlim(90, 100)
    ax.set_xlabel("测试集准确率 (%)")
    ax.set_title("迁移学习对照：手写 vs 预训练")
    plt.tight_layout()
    fig.savefig(FIG / "transfer_compare.png", dpi=150)
    plt.close(fig)

    # 微调曲线
    log = pd.DataFrame(log_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(log["epoch"], log["val_acc"] * 100, marker="o", color="#c44e52")
    ax.set_xlabel("epoch"); ax.set_ylabel("验证集准确率 (%)")
    ax.set_title("预训练 ResNet18 全量微调曲线")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "transfer_curves.png", dpi=150)
    plt.close(fig)

    print("\n=== 对照结果 ===")
    for r in results:
        print(f"{r['name']:20s} test_acc={r['test_acc']:.4f}  epochs={r['epochs']}")
    print("saved ->", RESULTS / "transfer_results.json", "|", FIG / "transfer_compare.png")


if __name__ == "__main__":
    main()
