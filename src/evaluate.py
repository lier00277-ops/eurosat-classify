"""在 test 集上评估模型：准确率、每类 P/R/F1、混淆矩阵。

用法：
    python -m src.evaluate --tag resnet18
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EuroSATDataset
from src.models.resnet import resnet18

DATA_ROOT = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
OUT_DIR = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
FIG_DIR = Path("figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="resnet18")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    classes = json.loads((DATA_ROOT / "classes.json").read_text(encoding="utf-8"))["classes"]
    stats = json.loads((OUT_DIR / f"{args.tag}_stats.json").read_text(encoding="utf-8"))
    npz = np.load(DATA_ROOT / "eurosat_rgb.npz")
    X, y = npz["X"], npz["y"].astype(np.int64)
    test_idx = pd.read_csv(DATA_ROOT / "splits" / "test.csv")["row_id"].to_numpy()

    ds = EuroSATDataset(X, y, test_idx, stats["mean"], stats["std"], train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model = resnet18(num_classes=len(classes))
    ckpt = torch.load(OUT_DIR / f"{args.tag}_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb).argmax(1).numpy())
    preds = np.concatenate(preds)
    y_true = y[test_idx]
    acc = float((preds == y_true).mean())

    K = len(classes)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, preds):
        cm[t, p] += 1

    rows = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(class_name=c, precision=round(prec, 4), recall=round(rec, 4),
                        f1=round(f1, 4), support=int(cm[i, :].sum())))
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print(f"[eval] TEST ACC = {acc:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_DIR / f"metrics_{args.tag}.csv", index=False, encoding="utf-8")

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion Matrix (acc={acc:.4f})")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_matrix_{args.tag}.png", dpi=150)
    plt.close(fig)

    json.dump(dict(acc=acc, per_class=rows),
              open(OUT_DIR / f"eval_{args.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[eval] saved -> {OUT_DIR / ('metrics_' + args.tag + '.csv')}")
    print(f"[eval] saved -> {FIG_DIR / ('confusion_matrix_' + args.tag + '.png')}")


if __name__ == "__main__":
    main()
