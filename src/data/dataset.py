"""内存版 EuroSAT Dataset：整个数据集预加载进 RAM，训练时只做轻量增强。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EuroSATDataset(Dataset):
    """X: (N,64,64,3) uint8 全量数组；indices 指定当前 split。

    数据量只有 ~265MB，整个数组常驻内存，避免每个 batch 读盘/解码。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, mean, std, train: bool = False):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = self.X[idx].astype(np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]   # 水平翻转
            if np.random.rand() < 0.5:
                img = img[::-1, :, :]   # 垂直翻转（遥感图像近似旋转不变）
        img = (img - self.mean) / self.std
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()
        return img, int(self.y[idx])
