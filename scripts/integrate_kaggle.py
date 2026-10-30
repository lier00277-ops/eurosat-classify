"""把 Kaggle 的 checkpoint（body.N 命名）转成项目模型结构（layer1~4），并验证精度一致。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.models.resnet import resnet18

DATA = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
RESULTS = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
KAGGLE_OUT = Path(os.environ.get("KAGGLE_OUT", r"D:\eurosat\kaggle_output"))
SRC_CKPT = KAGGLE_OUT / "resnet18_best.pt"


def remap_state_dict(sd: dict) -> dict:
    new = {}
    for k, v in sd.items():
        if k.startswith("body."):
            parts = k.split(".")
            i = int(parts[1])
            new[f"layer{1 + i // 2}.{i % 2}." + ".".join(parts[2:])] = v
        else:
            new[k] = v
    return new


def main() -> None:
    ck = torch.load(SRC_CKPT, map_location="cpu")
    classes = ck["classes"]
    model = resnet18(num_classes=len(classes))
    model.load_state_dict(remap_state_dict(ck["model"]), strict=True)
    print(f"[integrate] checkpoint key 重映射 + strict load 通过 (val_acc={ck['val_acc']:.4f})")

    z = np.load(DATA / "eurosat_rgb.npz")
    X, y, split_idx = z["X"], z["y"].astype(np.int64), z["split_idx"]
    tr = np.where(split_idx == 0)[0]
    mean = (X[tr].astype(np.float32).mean(axis=(0, 1, 2)) / 255.0).tolist()
    std = (X[tr].astype(np.float32).std(axis=(0, 1, 2)) / 255.0).tolist()
    print(f"[integrate] mean={np.round(mean, 4)} std={np.round(std, 4)} (应与 Kaggle 日志一致)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "val_acc": ck.get("val_acc"), "classes": classes,
                "mean": mean, "std": std, "source": "kaggle/kaggle_train.py"},
               RESULTS / "resnet18_best.pt")
    (RESULTS / "resnet18_stats.json").write_text(json.dumps({"mean": mean, "std": std}), encoding="utf-8")
    print(f"[integrate] saved -> {RESULTS / 'resnet18_best.pt'}")

    # 复现性校验
    model.eval()
    te = np.where(split_idx == 2)[0]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    correct = 0
    bs = 256
    with torch.no_grad():
        for i in range(0, len(te), bs):
            idx = te[i:i + bs]
            xb = torch.from_numpy(X[idx]).float().div_(255.0).permute(0, 3, 1, 2)
            xb = (xb - mean_t) / std_t
            pred = model(xb).argmax(1).numpy()
            correct += int((pred == y[idx]).sum())
    acc = correct / len(te)
    print(f"[integrate] local re-eval acc = {acc:.4f}  (Kaggle reported 0.9761)")
    assert abs(acc - 0.9761) < 1e-3, "精度不一致，检查数据/归一化"
    print("[integrate] REPRODUCIBILITY OK")


if __name__ == "__main__":
    main()
