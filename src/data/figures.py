"""画类别分布、样本网格、亮度分布图。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# 中文标题用系统字体，失败则忽略
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    meta = pd.read_csv(DATA_ROOT / "metadata.csv")
    classes = json.loads((DATA_ROOT / "classes.json").read_text(encoding="utf-8"))["classes"]
    npz = np.load(DATA_ROOT / "eurosat_rgb.npz")
    X, y = npz["X"], npz["y"]

    # 1) 类别分布
    counts = meta.groupby("class_name", observed=True).size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("EuroSAT 各类别样本数")
    ax.set_ylabel("样本数")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2) 样本网格 3x4
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax in axes.flat:
        i = int(rng.integers(0, len(X)))
        ax.imshow(X[i])
        ax.set_title(classes[int(y[i])], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=150)
    plt.close(fig)

    # 3) 亮度均值分布
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in classes:
        sub = meta[meta["class_name"] == c]
        ax.hist(sub["brightness_mean"], bins=30, alpha=0.5, label=c)
    ax.set_title("各类别图像亮度均值分布")
    ax.set_xlabel("brightness mean")
    ax.set_ylabel("数量")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "brightness_distribution.png", dpi=150)
    plt.close(fig)

    print("saved:", [p.name for p in sorted(FIG_DIR.glob("*.png"))])


if __name__ == "__main__":
    main()
