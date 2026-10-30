"""下载 EuroSAT RGB，导出为 npz、metadata.csv 和 classes.json。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

DATA_ROOT = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
HF_CACHE = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME")
DATASET_NAME = "blanchon/EuroSAT_RGB"
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[download] dataset={DATASET_NAME} cache={HF_CACHE}")
    ds = load_dataset(DATASET_NAME, cache_dir=HF_CACHE)
    classes = list(ds["train"].features["label"].names)

    split_names = list(SPLIT_MAP)
    X_parts, y_parts, split_idx_parts, rows = [], [], [], []
    row = 0
    for split_name in split_names:
        short = SPLIT_MAP[split_name]
        d = ds[split_name]
        n = len(d)
        print(f"[download] {split_name} ({short}): {n} images")
        for i in range(n):
            ex = d[i]
            arr = np.asarray(ex["image"].convert("RGB"), dtype=np.uint8)
            y = int(ex["label"])
            X_parts.append(arr)
            y_parts.append(y)
            split_idx_parts.append(split_names.index(split_name))
            rows.append(
                {
                    "row_id": row,
                    "source_split": short,
                    "label": y,
                    "class_name": classes[y],
                    "filename": ex.get("filename", ""),
                    "width": int(arr.shape[1]),
                    "height": int(arr.shape[0]),
                    "brightness_mean": float(arr.mean()),
                    "brightness_std": float(arr.std()),
                }
            )
            row += 1

    X = np.stack(X_parts, axis=0)  # (N, H, W, 3) uint8
    y = np.asarray(y_parts, dtype=np.int16)
    split_idx = np.asarray(split_idx_parts, dtype=np.int8)

    npz_path = DATA_ROOT / "eurosat_rgb.npz"
    np.savez(npz_path, X=X, y=y, split_idx=split_idx)
    print(f"[download] saved {npz_path} X={X.shape} dtype={X.dtype}")

    meta = pd.DataFrame(rows)
    meta.to_csv(DATA_ROOT / "metadata.csv", index=False, encoding="utf-8")
    print(f"[download] saved metadata.csv rows={len(meta)}")

    (DATA_ROOT / "classes.json").write_text(
        json.dumps({"classes": classes, "split_names": split_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[download] done")


if __name__ == "__main__":
    main()
