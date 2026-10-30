"""重新生成迁移学习对比图。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    res = json.loads((RESULTS / "transfer_results.json").read_text(encoding="utf-8"))
    log = pd.read_csv(RESULTS / "transfer_log.csv")

    names = [r["name"] for r in res]
    accs = [r["test_acc"] * 100 for r in res]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, accs, color=colors)
    for b, a in zip(bars, accs):
        ax.text(a + 0.12, b.get_y() + b.get_height() / 2, f"{a:.2f}%", va="center", fontsize=10)
    ax.set_xlim(88, 100)
    ax.set_xlabel("测试集准确率 (%)")
    ax.set_title("迁移学习对照：手写 vs 预训练（64×64 输入）")
    plt.tight_layout()
    fig.savefig(FIG / "transfer_compare.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(log["epoch"], log["val_acc"] * 100, marker="o", color="#c44e52")
    ax.set_xlabel("epoch")
    ax.set_ylabel("验证集准确率 (%)")
    ax.set_title("预训练 ResNet18 全量微调曲线（64×64）")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "transfer_curves.png", dpi=150)
    plt.close(fig)
    print("regenerated:", FIG / "transfer_compare.png", "|", FIG / "transfer_curves.png")


if __name__ == "__main__":
    main()
