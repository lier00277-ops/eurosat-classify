"""检查数据完整性，统计类别分布。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))


def main() -> int:
    meta = pd.read_csv(DATA_ROOT / "metadata.csv")
    classes = json.loads((DATA_ROOT / "classes.json").read_text(encoding="utf-8"))["classes"]
    npz = np.load(DATA_ROOT / "eurosat_rgb.npz")
    X, y = npz["X"], npz["y"]

    errors: list[str] = []
    if len(meta) != len(X):
        errors.append(f"metadata 行数 {len(meta)} != 图像张量 {len(X)}")
    if len(y) != len(X):
        errors.append("标签长度与图像张量不一致")
    if (X.dtype != np.uint8):
        errors.append(f"图像 dtype 不是 uint8: {X.dtype}")
    if X.min() < 0 or X.max() > 255:
        errors.append("像素值越界")
    if meta.isna().any().any():
        errors.append("metadata 存在缺失值")

    print("== 完整性检查 ==")
    print(f"样本数: {len(meta)}   图像张量: {X.shape} ({X.dtype})   类别数: {len(classes)}")
    print(f"像素范围: [{X.min()}, {X.max()}]   缺失值: {int(meta.isna().sum().sum())}")
    print(f"结论: {'通过' if not errors else '发现问题 -> ' + '; '.join(errors)}")

    print("\n== 类别分布（按官方划分）==")
    g = meta.groupby(["source_split", "class_name"], observed=True).size().unstack(fill_value=0)
    print(g.to_string())

    print("\n== 各类总数（升序）==")
    total = meta.groupby("class_name", observed=True).size().sort_values()
    print(total.to_string())
    print(f"不均衡比 max/min = {total.max() / total.min():.2f}")

    print("\n== 图像属性 ==")
    sizes = set(zip(meta["width"].tolist(), meta["height"].tolist()))
    print(f"尺寸集合: {sorted(sizes)}")
    print(f"亮度均值范围: {meta['brightness_mean'].min():.1f} ~ {meta['brightness_mean'].max():.1f}")
    print(f"亮度标准差范围: {meta['brightness_std'].min():.1f} ~ {meta['brightness_std'].max():.1f}")

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
