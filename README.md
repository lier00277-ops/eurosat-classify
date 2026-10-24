# EuroSAT 遥感地物分类 · 手写 ResNet 图像分类

第一阶段考核项目：用**手写 ResNet18** 对 EuroSAT 卫星影像做 10 类地物分类，并在同一项目中完成数据清洗、统计分析与结果可视化。

## 数据集

EuroSAT RGB：10 类地物（AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake），约 2.7 万张 64×64 图像。

## 目录结构

```
eurosat-classify/
├── data/            # 数据（raw 不入库，splits 入库）
├── src/
│   ├── data/        # 下载、清洗、划分、统计
│   └── models/      # 手写 BasicBlock + ResNet
├── scripts/         # 一键入口
├── notebooks/       # 探索分析
├── figures/         # 图表输出
├── results/         # 指标与权重（权重不入库）
└── docs/            # 项目说明与笔记
```

## 环境

```bash
uv venv --python 3.12
uv pip install -e .
```

## 进度

- [ ] Day1 环境 + 数据 + 探索
- [ ] Day2 训练管道
- [ ] Day3 手写 ResNet
- [ ] Day4 分析 + 服务化
- [ ] Day5 文档 + 视频
