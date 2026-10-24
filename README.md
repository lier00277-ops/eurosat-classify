# EuroSAT 遥感地物分类 · 手写 ResNet

第一阶段考核项目：用**手写 ResNet18**（从零训练、无预训练权重）对 EuroSAT 卫星影像做 10 类地物分类，并在同一项目中完成数据清洗、统计分析与结果可视化。

## 结果

| 指标 | 值 |
|---|---|
| **测试集准确率** | **97.61%** |
| 最佳验证集准确率 | 97.30% |
| 训练集准确率（最后一轮） | 99.83% |
| 模型参数量 | 11.18 M |
| 训练设备 | Kaggle 免费 GPU（T4） |
| 训练耗时 | 约 9.4 分钟（30 epoch） |

![训练曲线](figures/train_curves_resnet18.png)

![混淆矩阵](figures/confusion_matrix_resnet18.png)

各类别 P/R/F1 见 [`results/metrics_resnet18.csv`](results/metrics_resnet18.csv)。最弱的是 **Permanent Crop（F1 0.953）** 与 **Herbaceous Vegetation（F1 0.956）**，主要是几类植被（Herbaceous / Permanent Crop / Pasture）颜色纹理相近、易互相混淆。

## 数据集

EuroSAT RGB：10 类地物、27,000 张 64×64 图像（train 16200 / val 5400 / test 5400）。

类别分布不均衡（2000～3000，max/min = 1.5），Pasture 最少。详见 [`figures/class_distribution.png`](figures/class_distribution.png) 与 `D:\eurosat\data\metadata.csv`。

## 模型设计（手写 ResNet18）

为适配 64×64 小图，对标准 ResNet18 做了两处有意调整：

- **stem 用 3×3 stride1**（官方为 7×7 stride2），保留更多空间细节 → 少 7,680 参数
- **4 个 stage 均下采样**（layer1 也 stride2），特征图 `64→32→16→8→4` → 多 4,224 参数

与官方 ResNet18 净差 3,456 参数，其余 BasicBlock 逐层一致，`scripts/verify_model.py` 会自动校验：

```
stem -> (2, 64, 64, 64)
layer1 -> (2, 64, 32, 32)
layer2 -> (2, 128, 16, 16)
layer3 -> (2, 256, 8, 8)
layer4 -> (2, 512, 4, 4)
```

关键代码：[`src/models/resnet.py`](src/models/resnet.py)（`BasicBlock` + `ResNet`）。

## 复现流程

```powershell
# 1) 数据下载 + 分层划分 + 完整性检查 + 探索图
python -m src.data.download
python -m src.data.split
python -m src.data.check
python -m src.data.figures

# 2) 训练：在 Kaggle GPU 上运行 kaggle/kaggle_train.py（自包含单单元格）

# 3) 把 Kaggle 产物放到 D:\eurosat\kaggle_output，转成统一 checkpoint 并做复现校验
python -m scripts.integrate_kaggle
# -> local re-eval acc = 0.9761（与 Kaggle 一致）

# 4) 本地评估
python -m src.evaluate --tag resnet18
```

## 目录结构

```
eurosat-classify/
├── data/                 # 数据说明（大文件在 D:\eurosat\data）
├── src/
│   ├── data/             # 下载 / 清洗 / 分层划分 / 检查 / 可视化
│   ├── models/           # 手写 BasicBlock + ResNet
│   ├── train.py          # 本地 CPU 训练脚本
│   └── evaluate.py       # 测试集评估
├── kaggle/kaggle_train.py  # Kaggle GPU 单单元格训练脚本
├── scripts/              # 结构校验 / checkpoint 集成
├── figures/              # 图表输出
├── results/              # 指标、日志（权重在 D 盘，不入库）
├── docs/                 # 项目说明与分析报告
└── notebooks/            # 探索分析
```

## 环境（全部安装到 D 盘，不占 C 盘）

- 虚拟环境：`D:\eurosat\venv`
- 数据集：`D:\eurosat\data`
- 缓存（uv / pip / huggingface）：`D:\eurosat\cache`
- 训练权重：`D:\eurosat\results\resnet18_best.pt`

```powershell
. .\scripts\env.ps1
uv venv D:\eurosat\venv --python 3.12
uv pip install --python D:\eurosat\venv\Scripts\python.exe -e .
```

## 进度

- [x] Day1 数据管道 + 探索分析
- [x] Day2 手写 ResNet + 训练（Kaggle GPU，test 97.61%）
- [x] Day3 数据分析 + 错误分析（见 `docs/数据分析报告.md`）
- [ ] Day4 推理服务 + 文档
- [ ] Day5 视频 + 提交
