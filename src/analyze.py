"""测试集数据分析 + 错误分析。

产出：
    results/analyze_metrics.csv        每类 P/R/F1（按样本数排序）
    results/analyze_predictions.csv    每个测试样本的预测明细
    results/analyze_top_confusions.csv 最常混淆的类别对
    figures/confusion_matrix_analysis.png  行归一化混淆矩阵
    figures/per_class_f1.png           每类 F1 条形图
    figures/misclassified_samples.png  高置信度错分样本
    figures/class_imbalance_vs_recall.png  类别样本量 vs 召回率
    figures/top_confusions.png         最易混淆类别对

用法：
    python -m src.analyze
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.resnet import resnet18

DATA = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
RESULTS = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def infer(model, X, idx, mean_t, std_t, bs=256):
    preds = np.empty(len(idx), dtype=np.int64)
    probs = np.empty(len(idx), dtype=np.float32)
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        xb = torch.from_numpy(X[chunk]).float().div_(255.0).permute(0, 3, 1, 2)
        xb = (xb - mean_t) / std_t
        with torch.no_grad():
            logit = model(xb)
            prob = torch.softmax(logit, dim=1)
        preds[i:i + len(chunk)] = logit.argmax(1).numpy()
        probs[i:i + len(chunk)] = prob.max(1).values.numpy()
    return preds, probs


def main() -> None:
    ck = torch.load(RESULTS / "resnet18_best.pt", map_location="cpu")
    classes = ck["classes"]
    mean, std = ck["mean"], ck["std"]
    z = np.load(DATA / "eurosat_rgb.npz")
    X, y, si = z["X"], z["y"].astype(np.int64), z["split_idx"]
    te = np.where(si == 2)[0]
    tr = np.where(si == 0)[0]

    model = resnet18(num_classes=len(classes))
    model.load_state_dict(ck["model"])
    model.eval()
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    preds, conf = infer(model, X, te, mean_t, std_t)
    y_true = y[te]
    acc = float((preds == y_true).mean())
    K = len(classes)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, preds):
        cm[t, p] += 1
    print(f"test acc = {acc:.4f}  (n={len(te)})")

    # ---------- 每类指标 ----------
    rows = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(class_name=c, precision=round(prec, 4), recall=round(rec, 4),
                        f1=round(f1, 4), support=int(cm[i, :].sum())))
    report = pd.DataFrame(rows).sort_values("support")
    report.to_csv(RESULTS / "analyze_metrics.csv", index=False, encoding="utf-8")
    print("\n每类指标（按样本数升序）:")
    print(report.to_string(index=False))

    # ---------- 预测明细 ----------
    pd.DataFrame({
        "row_id": te, "true": y_true, "pred": preds,
        "true_name": [classes[i] for i in y_true],
        "pred_name": [classes[i] for i in preds],
        "confidence": np.round(conf, 4),
        "correct": preds == y_true,
    }).to_csv(RESULTS / "analyze_predictions.csv", index=False, encoding="utf-8")

    # ---------- 混淆对 ----------
    pairs = []
    for i in range(K):
        for j in range(K):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j]), round(cm[i, j] / cm[i, :].sum(), 4)))
    pairs.sort(key=lambda x: -x[2])
    pd.DataFrame(pairs, columns=["true", "pred", "count", "rate_in_true"]).to_csv(
        RESULTS / "analyze_top_confusions.csv", index=False, encoding="utf-8")
    print("\nTop 10 混淆对:")
    for p in pairs[:10]:
        print(f"  {p[0]:24s} -> {p[1]:24s}  {p[2]:3d}  ({p[3]:.2%})")

    # ---------- 图1：归一化混淆矩阵 ----------
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"归一化混淆矩阵（test acc={acc:.4f}）")
    for i in range(K):
        for j in range(K):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); fig.savefig(FIG / "confusion_matrix_analysis.png", dpi=150); plt.close(fig)

    # ---------- 图2：每类 F1 ----------
    rep = pd.DataFrame(rows).sort_values("f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(rep["class_name"], rep["f1"], color="steelblue")
    ax.set_xlim(0.90, 1.0)
    ax.set_xlabel("F1"); ax.set_title("各类别 F1 分数")
    for i, (name, f1) in enumerate(zip(rep["class_name"], rep["f1"])):
        ax.text(f1 + 0.001, i, f"{f1:.3f}", va="center", fontsize=8)
    plt.tight_layout(); fig.savefig(FIG / "per_class_f1.png", dpi=150); plt.close(fig)

    # ---------- 图3：高置信度错分样本 ----------
    wrong = np.where(preds != y_true)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:12]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, wi in zip(axes.flat, wrong):
        gidx = te[wi]
        ax.imshow(X[gidx])
        ax.set_title(f"真: {classes[y_true[wi]]}\n预: {classes[preds[wi]]} ({conf[wi]:.2f})", fontsize=8)
        ax.axis("off")
    plt.suptitle("高置信度错分样本（真值 → 预测）", fontsize=12)
    plt.tight_layout(); fig.savefig(FIG / "misclassified_samples.png", dpi=150); plt.close(fig)

    # ---------- 图4：类别样本量 vs 召回率 ----------
    tr_counts = pd.Series(y[tr]).value_counts().reindex(range(K)).fillna(0).astype(int)
    recs = [r["recall"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tr_counts.values, recs, s=80, color="darkorange")
    for i, c in enumerate(classes):
        ax.annotate(c, (tr_counts.iloc[i], recs[i]), fontsize=7,
                    textcoords="offset points", xytext=(5, 4))
    ax.set_xlabel("训练集样本数"); ax.set_ylabel("召回率")
    ax.set_title("类别样本量 vs 召回率（不均衡分析）")
    ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(FIG / "class_imbalance_vs_recall.png", dpi=150); plt.close(fig)

    # ---------- 图5：Top 混淆对 ----------
    top = pairs[:10]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{t} → {p}" for t, p, _, _ in top][::-1]
    counts = [c for _, _, c, _ in top][::-1]
    ax.barh(labels, counts, color="indianred")
    ax.set_xlabel("错分数量"); ax.set_title("最易混淆的类别对（Top 10）")
    plt.tight_layout(); fig.savefig(FIG / "top_confusions.png", dpi=150); plt.close(fig)

    json.dump(dict(test_acc=acc, n_test=int(len(te)), top_confusions=pairs[:10]),
              open(RESULTS / "analyze_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved analysis artifacts ->", RESULTS, "|", FIG)


if __name__ == "__main__":
    main()
