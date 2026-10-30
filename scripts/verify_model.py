"""对比 torchvision resnet18，检查结构、参数量和输出形状。"""
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

    # 与官方 ResNet18 的两处有意差异：
    #   1) stem 用 3x3 stride1（官方 7x7 stride2）→ 少 7*7*3*64-3*3*3*64 = 7680
    #   2) 64x64 小图让 layer1 也 stride2，因此多一个 1x1 downsample → 多 1*1*64*64+2*64 = 4224
    # 净差 = 7680 - 4224 = 3456，其余层与官方逐层一致
    expected_diff = (7 * 7 * 3 * 64 - 3 * 3 * 3 * 64) - (1 * 1 * 64 * 64 + 2 * 64)
    assert n_ref - n_ours == expected_diff, f"结构不一致，期望差 {expected_diff}，实际 {n_ref - n_ours}"
    print(f"OK 参数净差 = {expected_diff:,}（3x3 主干 -7680，layer1 下采样 +4224，其余层一致）")

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
    print("全部通过")


if __name__ == "__main__":
    main()
