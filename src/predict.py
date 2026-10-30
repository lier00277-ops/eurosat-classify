"""命令行推理：--image 单张，--dir 文件夹。"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.infer import load_model, predict_pil

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None, help="单张图片路径")
    ap.add_argument("--dir", type=str, default=None, help="图片文件夹")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint 路径")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, classes, mean, std, device = load_model(args.ckpt, args.device)
    print(f"[predict] model loaded: {len(classes)} classes on {device}")

    if args.image:
        res = predict_pil(Image.open(args.image), model, classes, mean, std, device, topk=3)
        print(f"== {args.image}")
        for r in res:
            print(f"  {r['class']:24s} {r['confidence']:.2%}")
    elif args.dir:
        paths = [p for p in sorted(Path(args.dir).iterdir()) if p.suffix.lower() in EXTS]
        if not paths:
            print(f"[predict] 目录里没有图片: {args.dir}")
            return
        for p in paths:
            res = predict_pil(Image.open(p), model, classes, mean, std, device, topk=1)
            print(f"  {p.name:32s} -> {res[0]['class']:24s} {res[0]['confidence']:.2%}")
    else:
        print("请用 --image 或 --dir 指定输入")


if __name__ == "__main__":
    main()
