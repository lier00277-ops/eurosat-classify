"""校验手写 ResNet18 的结构正确性：
1. 参数量与 torchvision 官方 resnet18 对齐（仅 stem 不同）。
2. 前向输出形状正确。
3. 每个 BasicBlock 参数与官方一致。
"""
from __future__ import annotations

import torch
import torchvision

from src.models.resnet import BasicBlock, ResNet, resnet18


def count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ours = resnet18(num_classes=10)
    ref = torchvision.models.resnet18(num_classes=10)

    n_ours, n_ref = count(ours), count(ref)
    print(f"ours params = {n_ours:,}")
    print(f"ref  params = {n_ref:,}")
    print(f"diff        = {n_ref - n_ours:,}")

    # 官方 stem: 7x7 conv = 7*7*3*64 = 9408；我们的 small stem: 3x3 conv = 3*3*3*64 = 1728
    expected_diff = 7 * 7 * 3 * 64 - 3 * 3 * 3 * 64
    assert n_ref - n_ours == expected_diff, f"结构不一致，期望差 {expected_diff}，实际 {n_ref - n_ours}"
    print(f"OK 参数差异 = 官方7x7主干 - 手写3x3主干 = {expected_diff:,}（其余层完全一致）")

    # 前向形状
    x = torch.randn(2, 3, 64, 64)
    out = ours(x)
    assert out.shape == (2, 10), out.shape
    print(f"OK forward: {tuple(x.shape)} -> {tuple(out.shape)}")

    # 逐块核对 BasicBlock 参数量（torchvision 的 BasicBlock 需要显式传 downsample）
    b_ours = BasicBlock(64, 128, stride=2)
    ref_down = torch.nn.Sequential(
        torch.nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        torch.nn.BatchNorm2d(128),
    )
    b_ref = torchvision.models.resnet.BasicBlock(64, 128, stride=2, downsample=ref_down)
    assert count(b_ours) == count(b_ref), (count(b_ours), count(b_ref))
    print(f"OK BasicBlock(64->128,s2) params = {count(b_ours):,} (与官方一致)")

    # 输出尺寸链
    m = resnet18(num_classes=10).eval()
    with torch.no_grad():
        f = m.stem(x)
        print("stem ->", tuple(f.shape))
        f = m.layer1(f); print("layer1 ->", tuple(f.shape))
        f = m.layer2(f); print("layer2 ->", tuple(f.shape))
        f = m.layer3(f); print("layer3 ->", tuple(f.shape))
        f = m.layer4(f); print("layer4 ->", tuple(f.shape))
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
