# 水稻叶病害图像分类课程项目

## 项目简介

本项目是一个面向课程期末作业的水稻叶病害图像分类项目，目标是识别 4 类水稻叶病害：

- `Bacterialblight`
- `Blast`
- `Brownspot`
- `Tungro`

项目已经完成从数据审计、数据去重、模型训练、超参数调优、测试评估，到图表导出、实验报告和答辩材料整理的完整流程。当前仓库不仅能复现实验，还能直接作为课程展示和答辩使用。

## 项目亮点

- 同时保留 `official` 与 `clean` 两套评测口径。
  - `official`：沿用原始官方划分，便于展示高分结果。
  - `clean`：按图像内容哈希去重并重划分，更能反映真实泛化能力。
- 提供 3 类模型的完整对比：
  - `CustomCNN`
  - `ResNet18`
  - `EfficientNet-B0`
- 已完成 `EfficientNet-B0` 的正式 Optuna 调参。
- 已补充 `clean_resnet18` 的多随机种子稳定性验证。
- 已导出轻量结果目录 `final_assets/`，可直接用于报告和答辩。
- 已生成实际答辩 PPT 文件，可直接打开展示。
- 已准备中文实验报告与中文答辩提纲。

## 环境与依赖

推荐使用 `conda` 环境运行本项目。

### 推荐环境

- Python：`3.11`
- 深度学习框架：`PyTorch 2.10.0 + CUDA 13.0`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`

### 创建环境

```powershell
conda env create -f environment.yml
conda activate rice-leaf-dl
```

### 检查 GPU 是否可用

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 依赖说明

- `environment.yml`：推荐的 Conda 环境定义
- `requirements.txt`：PyTorch、可视化、调参和数据处理等依赖清单
- `环境配置检查说明文档.md`：详细的环境验证与故障排除指南

## 快速开始

### 1. 生成数据清单与数据审计结果

```powershell
python prepare_dataset.py --dataset-root "Rice Leaf Disease Images"
```

### 2. 运行正式训练

```powershell
python train.py --config configs/official_efficientnet_tuned.yaml --skip-eval
python train.py --config configs/clean_resnet18.yaml --skip-eval
```

### 3. 运行调参

```powershell
python tune.py --config configs/official_efficientnet_search.yaml --trials 12
```

### 4. 运行测试评估

```powershell
python evaluate.py --config configs/official_efficientnet_tuned.yaml --checkpoint outputs/runs/official_efficientnet_tuned/best_model.pt
python evaluate.py --config configs/clean_resnet18.yaml --checkpoint outputs/runs/clean_resnet18/best_model.pt
```

### 5. 导出轻量结果

```powershell
python summarize_results.py
```

## 目录结构总览

说明：

- 为保证可读性，以下目录树不展开 `.git/`、`.local-remote.git/`、`__pycache__/`
- 数据集中的逐张图片文件不展开
- `outputs/` 下的海量中间产物不逐项列出，只展示关键结构和代表性层级

### 1. 顶层结构

```text
DeepLearning/
├─ .gitignore                      # Git 忽略规则，排除数据集、outputs 和缓存目录
├─ environment.yml                # 推荐的 Conda 环境定义
├─ requirements.txt               # Python 依赖清单
├─ progress.md                    # 项目全过程变更记录
├─ README.md                      # 当前中文说明文档
├─ prepare_dataset.py             # 数据清单生成与数据审计入口
├─ train.py                       # 单实验训练入口
├─ tune.py                        # EfficientNet-B0 超参数调优入口
├─ evaluate.py                    # 单模型测试评估入口
├─ predict.py                     # 单张图片预测入口
├─ summarize_results.py           # 轻量结果导出与汇总入口
├─ configs/                       # 全部实验 YAML 配置目录
├─ docs/                          # 中文实验报告与答辩文档
├─ src/                           # 项目核心源码目录
├─ final_assets/                  # 轻量结果目录，可直接用于报告与答辩
├─ build_ppt.py                   # 根据现有结果自动生成答辩 PPT 的脚本
├─ 环境配置检查说明文档.md          # 环境验证操作指南（PyTorch / CUDA / GPU）
├─ outputs/                       # 完整中间产物目录，不纳入 Git 跟踪
└─ Rice Leaf Disease Images/      # 原始数据集目录
```

### 2. `configs/` 配置目录

当前仓库中的 YAML 配置文件如下：

```text
configs/
├─ clean_customcnn.yaml              # clean 视图下的 CustomCNN 正式训练配置
├─ clean_efficientnet.yaml           # clean 视图下的 EfficientNet 早期配置
├─ clean_efficientnet_tuned.yaml     # clean 视图下的 EfficientNet 调优后正式配置
├─ clean_resnet18.yaml               # clean 视图下的 ResNet18 正式训练配置
├─ clean_resnet18_seed123.yaml       # clean ResNet18 的 seed=123 稳定性验证配置
├─ clean_resnet18_seed2026.yaml      # clean ResNet18 的 seed=2026 稳定性验证配置
├─ official_customcnn.yaml           # official 视图下的 CustomCNN 正式训练配置
├─ official_efficientnet.yaml        # official 视图下的 EfficientNet 早期配置
├─ official_efficientnet_search.yaml # official EfficientNet 的正式调参配置
├─ official_efficientnet_tuned.yaml  # official 视图下的 EfficientNet 调优后正式配置
├─ official_resnet18.yaml            # official 视图下的 ResNet18 正式训练配置
└─ smoke_official.yaml               # smoke 流程验证配置
```

### 3. `docs/` 文档目录

```text
docs/
├─ report.md              # 中文实验报告正式版
├─ defense.md             # 中文答辩提纲与答辩口述稿
├─ qa_explanations.md     # 答辩常见质疑点集中解释稿
└─ ppt_outline.md         # 逐页答辩 PPT 大纲
```

### 4. `src/rice_leaf_disease/` 源码目录

```text
src/
└─ rice_leaf_disease/
   ├─ __init__.py   # 包入口
   ├─ config.py     # 配置加载与导出
   ├─ data.py       # 数据清单、数据集与数据增强
   ├─ models.py     # CustomCNN 与迁移学习模型构建
   ├─ engine.py     # 训练主循环、优化器和训练曲线生成
   ├─ analysis.py   # 评估、混淆矩阵、分类指标、错误样本导出
   ├─ gradcam.py    # Grad-CAM 可解释性实现
   └─ utils.py      # 通用工具函数
```

### 5. `final_assets/` 轻量结果目录

该目录是最终报告和答辩直接使用的结果目录，建议优先查看。

```text
final_assets/
├─ final_test_metrics.csv                    # 主要正式实验的测试集指标汇总
├─ leaderboard_filtered.csv                  # 过滤后的实验排行榜结果
├─ official_efficientnet_best_params.csv     # official EfficientNet 最优参数表
├─ official_efficientnet_best_params.yaml    # official EfficientNet 最优参数原始 YAML
├─ official_efficientnet_trials.csv          # official EfficientNet 调参 trial 明细
├─ report_table_main.csv                     # 报告主结果表
├─ resnet18_seed_consistency.csv             # clean ResNet18 多 seed 测试结果明细
├─ resnet18_seed_consistency_summary.csv     # clean ResNet18 多 seed 统计汇总
├─ dataset_audit/
│  ├─ clean_summary.json                     # clean 视图数据统计摘要
│  └─ official_summary.json                  # official 视图数据统计摘要
├─ ppt/
│  └─ 水稻叶病害分类答辩.pptx                  # 自动生成的实际答辩 PPT 文件
└─ figures/
   ├─ best_params.png                        # 最优调参结果可视化
   ├─ clean_class_distribution.png           # clean 视图类别分布图
   ├─ clean_customcnn_training_curves.png    # clean / CustomCNN 训练曲线
   ├─ clean_efficientnet_tuned_training_curves.png   # clean / EfficientNet 训练曲线
   ├─ clean_resnet18_class_metrics.png       # clean / ResNet18 分类指标图
   ├─ clean_resnet18_confusion_matrix.png    # clean / ResNet18 混淆矩阵
   ├─ clean_resnet18_gradcam_gallery.png     # clean / ResNet18 Grad-CAM 图集
   ├─ clean_resnet18_misclassified_examples.png      # clean / ResNet18 错误样本图
   ├─ clean_resnet18_seed123_training_curves.png     # clean / ResNet18 seed=123 训练曲线
   ├─ clean_resnet18_seed2026_training_curves.png    # clean / ResNet18 seed=2026 训练曲线
   ├─ clean_resnet18_training_curves.png     # clean / ResNet18 seed=42 训练曲线
   ├─ duplicate_overview.png                 # 官方划分重复图概览图
   ├─ model_comparison_accuracy.png          # 正式模型 Accuracy 对比图
   ├─ model_comparison_macro_f1.png          # 正式模型 Macro F1 对比图
   ├─ official_best_class_metrics.png        # official 主力模型分类指标图
   ├─ official_best_confusion_matrix.png     # official 主力模型混淆矩阵
   ├─ official_best_gradcam_gallery.png      # official 主力模型 Grad-CAM 图集
   ├─ official_best_misclassified_examples.png       # official 主力模型错误样本图
   ├─ official_class_distribution.png        # official 视图类别分布图
   ├─ official_customcnn_training_curves.png # official / CustomCNN 训练曲线
   ├─ official_efficientnet_tuned_training_curves.png # official / EfficientNet 训练曲线
   ├─ official_resnet18_training_curves.png  # official / ResNet18 训练曲线
   ├─ official_vs_clean_accuracy.png         # official 与 clean 结果对比图
   ├─ resnet18_seed_consistency.png          # clean ResNet18 多 seed 稳定性图
   └─ tuning_history.png                     # EfficientNet 调参历史图
```

### 6. `outputs/` 中间产物目录（代表性展开）

`outputs/` 保存完整中间产物，文件较多，因此这里只展示关键结构：

```text
outputs/
├─ manifests/                      # 训练与评估使用的数据清单
│  ├─ official_manifest.csv        # official 视图清单
│  ├─ clean_manifest.csv           # clean 视图清单
│  └─ duplicate_groups.csv         # 重复图哈希分组
├─ dataset_audit/                  # 数据审计结果
│  ├─ official_summary.json        # official 数据统计
│  ├─ clean_summary.json           # clean 数据统计
│  └─ figures/                     # 数据分布与重复图统计图
├─ runs/                           # 所有训练 run 的完整输出
│  ├─ official_customcnn/          # official / CustomCNN 训练产物
│  ├─ official_resnet18/           # official / ResNet18 训练产物
│  ├─ official_efficientnet_tuned/ # official / EfficientNet 正式训练产物
│  ├─ clean_customcnn/             # clean / CustomCNN 训练产物
│  ├─ clean_resnet18/              # clean / ResNet18 训练产物
│  ├─ clean_resnet18_seed123/      # clean / ResNet18 seed=123 训练产物
│  ├─ clean_resnet18_seed2026/     # clean / ResNet18 seed=2026 训练产物
│  ├─ clean_efficientnet_tuned/    # clean / EfficientNet 正式训练产物
│  └─ official_efficientnet_search_trial_*/ # 调参 trial 训练产物
├─ evaluations/                    # 测试评估结果
│  ├─ official_customcnn/          # official / CustomCNN 测试评估
│  ├─ official_resnet18/           # official / ResNet18 测试评估
│  ├─ official_efficientnet_tuned/ # official / EfficientNet 测试评估
│  ├─ clean_customcnn/             # clean / CustomCNN 测试评估
│  ├─ clean_resnet18/              # clean / ResNet18 测试评估
│  ├─ clean_resnet18_seed123/      # clean / ResNet18 seed=123 测试评估
│  ├─ clean_resnet18_seed2026/     # clean / ResNet18 seed=2026 测试评估
│  └─ clean_efficientnet_tuned/    # clean / EfficientNet 测试评估
├─ summary/                        # 原始汇总排行榜与图表
├─ tuning/                         # 调参结果目录
└─ predictions/                    # 单图预测演示结果
```

补充说明：

- `outputs/` 是完整中间产物目录，适合研究实验过程
- `final_assets/` 是轻量结果目录，适合报告和答辩直接引用
- 因中间文件数量较多，README 不再继续逐项展开 `outputs/` 内的全部文件

### 7. 原始数据集目录结构

数据集内部不展开逐张图片，仅展示层级结构：

```text
Rice Leaf Disease Images/
├─ 数据集介绍.txt                    # 数据集文字说明
├─ Rice Leaf Disease Images.rar      # 数据集压缩包副本
├─ train/                            # 训练集
│  ├─ Bacterialblight/              # 白枯病图片目录
│  ├─ Blast/                        # 稻瘟病图片目录
│  ├─ Brownspot/                    # 褐斑病图片目录
│  └─ Tungro/                       # Tungro 病图片目录
├─ validation/                       # 验证集
│  ├─ Bacterialblight/
│  ├─ Blast/
│  ├─ Brownspot/
│  └─ Tungro/
└─ test/                             # 测试集
   ├─ Bacterialblight/
   ├─ Blast/
   ├─ Brownspot/
   └─ Tungro/
```

说明：

- 图片文件数量较多，且不影响仓库导航，因此不逐张列出
- 训练与评估实际使用的是基于该目录生成的 `manifest` 文件

## 关键实验结果

### 正式主结果

| 数据视图 | 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| official | CustomCNN | 99.51% | 99.57% | 99.46% | 99.52% |
| official | ResNet18 | 99.51% | 99.39% | 99.57% | 99.48% |
| official | EfficientNet-B0（调优后） | 99.84% | 99.79% | 99.87% | 99.83% |
| clean | CustomCNN | 99.37% | 99.40% | 99.27% | 99.33% |
| clean | ResNet18 | 100.00% | 100.00% | 100.00% | 100.00% |
| clean | EfficientNet-B0（调优后） | 99.79% | 99.79% | 99.81% | 99.80% |

### ResNet18 多种子稳定性结果

| seed | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| 42 | 100.00% | 100.00% | 100.00% | 100.00% |
| 123 | 99.79% | 99.81% | 99.74% | 99.77% |
| 2026 | 100.00% | 100.00% | 100.00% | 100.00% |

稳定性总结：

- 平均 Accuracy：`99.93%`
- Accuracy 标准差：`0.10%`
- 平均 Macro F1：`99.92%`
- Macro F1 标准差：`0.11%`

这说明 `clean_resnet18` 的高分并不是单次偶然，而是在多随机种子下依然保持稳定高分。

## 结果文件说明

如果你要快速定位关键结果，优先看下面这些文件：

- `final_assets/report_table_main.csv`
  - 报告主表，汇总正式主实验矩阵的测试结果
- `final_assets/final_test_metrics.csv`
  - 正式模型测试结果过滤汇总
- `final_assets/official_efficientnet_best_params.csv`
  - official 主力 EfficientNet 的最优调参参数
- `final_assets/official_efficientnet_trials.csv`
  - official EfficientNet 的调参 trial 结果
- `final_assets/resnet18_seed_consistency.csv`
  - clean ResNet18 多种子测试结果明细
- `final_assets/resnet18_seed_consistency_summary.csv`
  - clean ResNet18 稳定性统计结果
- `docs/report.md`
  - 中文实验报告正式版
- `docs/defense.md`
  - 中文答辩提纲和口述稿

## 常用命令

### 创建并进入环境

```powershell
conda env create -f environment.yml
conda activate rice-leaf-dl
```

### 生成数据清单与数据审计

```powershell
python prepare_dataset.py --dataset-root "Rice Leaf Disease Images"
```

### 训练单个模型

```powershell
python train.py --config configs/official_customcnn.yaml --skip-eval
python train.py --config configs/official_resnet18.yaml --skip-eval
python train.py --config configs/official_efficientnet_tuned.yaml --skip-eval
python train.py --config configs/clean_customcnn.yaml --skip-eval
python train.py --config configs/clean_resnet18.yaml --skip-eval
python train.py --config configs/clean_efficientnet_tuned.yaml --skip-eval
```

### 运行官方主模型调参

```powershell
python tune.py --config configs/official_efficientnet_search.yaml --trials 12
```

### 单独评估某个模型

```powershell
python evaluate.py --config configs/official_efficientnet_tuned.yaml --checkpoint outputs/runs/official_efficientnet_tuned/best_model.pt
python evaluate.py --config configs/clean_resnet18.yaml --checkpoint outputs/runs/clean_resnet18/best_model.pt
```

### 运行 ResNet18 稳定性验证

```powershell
python train.py --config configs/clean_resnet18_seed123.yaml --skip-eval
python train.py --config configs/clean_resnet18_seed2026.yaml --skip-eval

python evaluate.py --config configs/clean_resnet18_seed123.yaml --checkpoint outputs/runs/clean_resnet18_seed123/best_model.pt
python evaluate.py --config configs/clean_resnet18_seed2026.yaml --checkpoint outputs/runs/clean_resnet18_seed2026/best_model.pt
```

### 导出轻量结果

```powershell
python summarize_results.py
```

### 生成答辩 PPT

```powershell
python build_ppt.py
```

### 单图预测演示

```powershell
python predict.py --config configs/official_efficientnet_tuned.yaml --checkpoint outputs/runs/official_efficientnet_tuned/best_model.pt --image "某张图片路径"
```

## 注意事项

- `official` 视图中存在跨 `train / validation / test` 的重复图像，因此其分数更适合做展示，不适合直接代表泛化能力。
- `clean` 视图经过哈希去重，更适合作为模型泛化能力的结论依据。
- `outputs/` 保存完整中间产物，文件多且体积大，通常不建议直接提交。
- `final_assets/` 是已经整理好的轻量结果目录，报告和答辩优先引用这里的文件。
- 模型权重保留在 `outputs/runs/` 下，没有纳入 `final_assets/`。
- 若后续继续扩展实验，建议保持：
  - `configs/` 负责配置
  - `outputs/` 负责完整中间产物
  - `final_assets/` 负责最终轻量结果
  - `docs/` 负责最终文档
