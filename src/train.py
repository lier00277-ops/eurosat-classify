"""手写 ResNet18 训练脚本（EuroSAT，CPU 可跑）。

用法（在项目根目录）：
    python -m src.train --epochs 10 --batch-size 128
    python -m src.train --smoke          # 小样本过拟合自检，验证代码正确性

产物（默认 D:\\eurosat\\results，可用 EUROSAT_RESULTS 覆盖）：
    {tag}_best.pt        验证集最优权重
    log_{tag}.csv        每个 epoch 的训练日志
    summary_{tag}.json   训练总结
    figures/train_curves_{tag}.png
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EuroSATDataset
from src.models.resnet import resnet18

DATA_ROOT = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
OUT_DIR = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
FIG_DIR = Path("figures")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_arrays():
    npz = np.load(DATA_ROOT / "eurosat_rgb.npz")
    return npz["X"], npz["y"].astype(np.int64)


def split_indices(name: str) -> np.ndarray:
    df = pd.read_csv(DATA_ROOT / "splits" / f"{name}.csv")
    return df["row_id"].to_numpy()


def compute_stats(X: np.ndarray, indices: np.ndarray):
    sub = X[indices].astype(np.float32) / 255.0
    return sub.mean(axis=(0, 1, 2)).tolist(), sub.std(axis=(0, 1, 2)).tolist()


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += crit(logits, yb).item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total


def save_curves(log: pd.DataFrame, tag: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(log["epoch"], log["train_loss"], label="train")
    axes[0].plot(log["epoch"], log["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(log["epoch"], log["train_acc"], label="train")
    axes[1].plot(log["epoch"], log["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"train_curves_{tag}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="只用前 N 个训练样本（调试用）")
    ap.add_argument("--smoke", action="store_true", help="小样本过拟合自检")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="resnet18")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 30
        args.subset = 320
        args.tag = args.tag + "_smoke"

    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_arrays()
    classes = json.loads((DATA_ROOT / "classes.json").read_text(encoding="utf-8"))["classes"]
    train_idx = split_indices("train")
    val_idx = split_indices("val")
    if args.subset:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, size=min(args.subset, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, size=min(max(64, args.subset // 4), len(val_idx)), replace=False)
    mean, std = compute_stats(X, train_idx)
    (OUT_DIR / f"{args.tag}_stats.json").write_text(json.dumps({"mean": mean, "std": std}), encoding="utf-8")

    train_ds = EuroSATDataset(X, y, train_idx, mean, std, train=True)
    val_ds = EuroSATDataset(X, y, val_idx, mean, std, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = resnet18(num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model=ResNet18 params={n_params / 1e6:.2f}M device={device} epochs={args.epochs} "
          f"train={len(train_idx)} val={len(val_idx)}")

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    steps = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)

    rows = []
    best_acc = -1.0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        tr_loss, tr_acc = loss_sum / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, crit, device)
        dt = time.time() - t0
        rows.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=val_loss,
                         val_acc=val_acc, lr=sched.get_last_lr()[0], epoch_time=dt))
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                        "classes": classes, "args": vars(args)}, OUT_DIR / f"{args.tag}_best.pt")
        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} {dt:.0f}s{' *best' if improved else ''}")

    log = pd.DataFrame(rows)
    log.to_csv(OUT_DIR / f"log_{args.tag}.csv", index=False, encoding="utf-8")
    save_curves(log, args.tag)
    summary = dict(tag=args.tag, best_val_acc=float(best_acc), final_train_acc=float(rows[-1]["train_acc"]),
                   epochs=args.epochs, params=int(n_params), total_time_s=time.time() - t_start, args=vars(args))
    (OUT_DIR / f"summary_{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] done best_val_acc={best_acc:.4f} total_time={time.time() - t_start:.0f}s")
    print(f"[train] artifacts -> {OUT_DIR}")
    print(f"[train] curves -> {FIG_DIR / ('train_curves_' + args.tag + '.png')}")


if __name__ == "__main__":
    main()
