"""按类别做分层划分 train/val/test（60/20/20），保存到 data/splits/。

用法：
    python -m src.data.split
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
SEED = 42
RATIOS = (0.6, 0.2, 0.2)


def stratified_split(meta: pd.DataFrame, seed: int = SEED, ratios=RATIOS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.empty(len(meta), dtype=object)
    for _label, group in meta.groupby("label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        split[idx[:n_train]] = "train"
        split[idx[n_train : n_train + n_val]] = "val"
        split[idx[n_train + n_val :]] = "test"
    return split


def main() -> None:
    meta = pd.read_csv(DATA_ROOT / "metadata.csv")
    meta["split"] = stratified_split(meta)
    out = DATA_ROOT / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        sub = meta[meta["split"] == name]
        sub.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
        print(f"[split] {name}: {len(sub)}")
    meta.to_csv(out / "all.csv", index=False, encoding="utf-8")
    print(f"[split] saved -> {out}")


if __name__ == "__main__":
    main()
