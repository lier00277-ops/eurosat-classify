"""模型加载、图片预处理和预测。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.models.resnet import resnet18

RESULTS = Path(os.environ.get("EUROSAT_RESULTS", r"D:\eurosat\results"))
IMG_SIZE = 64


def load_model(ckpt_path: str | None = None, device: str = "cpu"):
    """加载 checkpoint，返回 model/classes/mean/std/device。"""
    ckpt_path = Path(ckpt_path) if ckpt_path else RESULTS / "resnet18_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)
    classes = ck["classes"]
    model = resnet18(num_classes=len(classes)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, classes, ck["mean"], ck["std"], device


def preprocess(img: Image.Image, mean, std) -> torch.Tensor:
    """缩放并归一化到 (1, 3, 64, 64)。"""
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    m = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    s = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - m) / s
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


@torch.no_grad()
def predict_pil(img: Image.Image, model, classes, mean, std, device="cpu", topk: int = 3):
    """返回 top-k 类别和置信度。"""
    x = preprocess(img, mean, std).to(device)
    prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    order = prob.argsort()[::-1][:topk]
    return [{"class": classes[int(i)], "confidence": float(prob[i])} for i in order]
