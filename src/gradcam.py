"""Grad-CAM：看模型判断类别时关注图片的哪个区域。python -m src.gradcam"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.infer import load_model

DATA = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
RESULTS = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class GradCAM:
    """基于 layer4 特征图的 Grad-CAM。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._hook = target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.activations = out

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad()
        logits = self.model(x)  # 不能 no_grad，否则没有梯度
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        score = logits[0, class_idx]
        grad = torch.autograd.grad(score, self.activations)[0][0]  # (C,H,W)
        act = self.activations[0].detach()                          # (C,H,W)
        weights = grad.mean(dim=(1, 2))                             # (C,)
        cam = torch.relu((weights[:, None, None] * act).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), int(class_idx)

    def close(self):
        self._hook.remove()


def upsample(cam: np.ndarray, size: int = 64) -> np.ndarray:
    t = torch.tensor(cam, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def main() -> None:
    model, classes, mean, std, device = load_model()
    z = np.load(DATA / "eurosat_rgb.npz")
    X, y, si = z["X"], z["y"].astype(np.int64), z["split_idx"]
    te = np.where(si == 2)[0]
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    # 先跑一遍，拿预测与置信度
    preds = np.empty(len(te), dtype=np.int64)
    conf = np.empty(len(te), dtype=np.float32)
    bs = 256
    for i in range(0, len(te), bs):
        idx = te[i:i + bs]
        xb = torch.from_numpy(X[idx]).float().div_(255.0).permute(0, 3, 1, 2)
        xb = (xb - mean_t) / std_t
        with torch.no_grad():
            prob = torch.softmax(model(xb), dim=1)
        preds[i:i + len(idx)] = prob.argmax(1).numpy()
        conf[i:i + len(idx)] = prob.max(1).values.numpy()
    y_true = y[te]
    correct = preds == y_true
    print(f"[gradcam] test acc = {correct.mean():.4f}")

    cam = GradCAM(model, model.layer4)

    def cam_for(pos: int, class_idx: int):
        gidx = te[pos]
        x = torch.from_numpy(X[gidx]).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        x = (x - mean_t) / std_t
        c, _ = cam(x, class_idx)
        return c, X[gidx]

    # ---------- 图1：每类一个正确样本 ----------
    picks = []
    for ci in range(len(classes)):
        pos = np.where((y_true == ci) & correct)[0]
        if len(pos):
            picks.append(int(pos[0]))
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for ax, pos in zip(axes.flat, picks):
        c, img = cam_for(pos, int(preds[pos]))
        ax.imshow(img)
        ax.imshow(upsample(c), cmap="jet", alpha=0.45)
        ax.set_title(f"{classes[y_true[pos]]}  {conf[pos]:.2f}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    plt.suptitle("Grad-CAM：每类一个正确样本（注意力落在预测类别上）")
    plt.tight_layout()
    fig.savefig(FIG / "gradcam_per_class.png", dpi=150)
    plt.close(fig)

    # ---------- 图2：高置信度错分样本 ----------
    wrong = np.where(~correct)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:10]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for ax, pos in zip(axes.flat, wrong):
        c, img = cam_for(int(pos), int(preds[pos]))
        ax.imshow(img)
        ax.imshow(upsample(c), cmap="jet", alpha=0.45)
        ax.set_title(f"真: {classes[y_true[pos]]}\n预: {classes[preds[pos]]} ({conf[pos]:.2f})", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong):]:
        ax.axis("off")
    plt.suptitle("Grad-CAM：高置信度错分样本（热力图对应“预测类别”的注意力）")
    plt.tight_layout()
    fig.savefig(FIG / "gradcam_errors.png", dpi=150)
    plt.close(fig)

    # ---------- 图3：原图 vs 热力图 ----------
    def pick(name: str, want_correct: bool):
        ci = classes.index(name)
        mask = (y_true == ci) & (correct if want_correct else ~correct)
        pos = np.where(mask)[0]
        return int(pos[0]) if len(pos) else None

    rows = []
    for name, wc in [("Forest", True), ("River", False), ("Residential Buildings", True), ("Permanent Crop", False)]:
        p = pick(name, wc)
        if p is not None:
            rows.append((p, name, wc))
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.5, 3.4 * max(1, len(rows))))
    if len(rows) == 1:
        axes = axes[None, :]
    meta = []
    for r, (pos, name, wc) in enumerate(rows):
        c, img = cam_for(pos, int(preds[pos]))
        cu = upsample(c)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"原图：{name}", fontsize=9)
        axes[r, 0].axis("off")
        axes[r, 1].imshow(img)
        axes[r, 1].imshow(cu, cmap="jet", alpha=0.5)
        axes[r, 1].set_title(f"Grad-CAM（预测：{classes[preds[pos]]}）", fontsize=9)
        axes[r, 1].axis("off")
        meta.append(dict(sample=name, correct=bool(wc), true_class=name,
                         pred_class=classes[preds[pos]], confidence=float(conf[pos]),
                         row_id=int(te[pos])))
    plt.tight_layout()
    fig.savefig(FIG / "gradcam_samples.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(meta).to_csv(RESULTS / "gradcam_meta.csv", index=False, encoding="utf-8")
    cam.close()
    print("[gradcam] saved ->", FIG / "gradcam_per_class.png", "|", FIG / "gradcam_errors.png", "|", FIG / "gradcam_samples.png")


if __name__ == "__main__":
    main()
