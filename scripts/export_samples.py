"""导出每类一张样例图，放大到 256x256，用来测试推理服务。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA = Path(os.environ.get("EUROSAT_DATA", r"D:\eurosat\data"))
OUT = Path(os.environ.get("SAMPLES_OUT", r"D:\eurosat\samples"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA / "eurosat_rgb.npz")
    X, y, si = z["X"], z["y"], z["split_idx"]
    classes = json.loads((DATA / "classes.json").read_text(encoding="utf-8"))["classes"]
    te = np.where(si == 2)[0]

    rows = []
    for ci, cname in enumerate(classes):
        idx = te[y[te] == ci][:1]
        if len(idx) == 0:
            continue
        arr = X[idx[0]]
        img = Image.fromarray(arr).resize((256, 256), Image.BICUBIC)
        safe = cname.replace(" ", "_")
        path = OUT / f"{safe}.png"
        img.save(path)
        rows.append(dict(file=path.name, class_name=cname, row_id=int(idx[0])))
        print(f"saved {path.name:32s} <- {cname}")

    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False, encoding="utf-8")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
