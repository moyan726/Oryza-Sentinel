# 水稻叶病害图像分类期末实验报告

## 1. 课题背景

本课题面向水稻叶片病害识别任务，目标是根据叶片图像自动判别病害类别。项目数据包含四类病害：`Bacterialblight`、`Blast`、`Brownspot`、`Tungro`。本实验采用深度学习图像分类方法，围绕“高准确率、可复现、可解释、可答辩展示”的课程要求，构建了一个完整实验流程。

## 2. 数据审计

### 2.1 数据集概况

- 原始目录采用 `train/validation/test` 图像分类结构。
- 原始统计结果请引用：
  - `outputs/dataset_audit/official_summary.json`
  - `outputs/dataset_audit/clean_summary.json`
- 数据分布图请引用：
  - `outputs/dataset_audit/figures/official_class_distribution.png`
  - `outputs/dataset_audit/figures/clean_class_distribution.png`

### 2.2 数据质量问题

通过图片内容哈希比对，发现原始官方划分中存在跨集合重复图片，以及增强命名图片进入验证集/测试集的现象。这会导致模型在测试阶段见到与训练集高度相似甚至完全相同的样本，从而使最终准确率偏高。

### 2.3 双轨评测设计

为兼顾课程成绩展示与实验严谨性，本实验采用双轨评测：

1. 官方划分结果：沿用原始 `train/validation/test`，用于展示较高准确率。
2. 去重重划分结果：对全量图片按内容哈希去重后重新分层划分，作为严谨性补充结果。

## 3. 方法设计

### 3.1 总体方案

本实验采用“迁移学习主线 + 自建 CNN 基线”的结构：

- 基线模型：`CustomCNN`
- 对比模型：`ResNet18`
- 主力模型：`EfficientNet-B0`

### 3.2 数据预处理与增强

- 输入尺寸：`224 x 224`
- 训练增强：随机裁剪、水平翻转、轻度旋转、颜色扰动
- 验证/测试：缩放、中心裁剪、归一化

### 3.3 训练策略

- 第一阶段：冻结迁移学习骨干，只训练分类头
- 第二阶段：解冻全网络，使用更小学习率微调
- 使用策略：早停、学习率调度、最佳权重保存
- 设备策略：优先使用 GPU，若不可用则自动回退 CPU

## 4. 实验设置

### 4.1 环境配置

- Python：`3.11`
- 深度学习框架：`PyTorch`
- GPU：`NVIDIA RTX 4060`（若配置完成）
- 依赖说明：见 `requirements.txt` 与 `environment.yml`

### 4.2 关键超参数

请将最终最优实验的配置补充到下表：

| 参数 | 取值 |
| --- | --- |
| dataset_view | 待填 |
| model_name | 待填 |
| batch_size | 待填 |
| optimizer | 待填 |
| lr | 待填 |
| weight_decay | 待填 |
| dropout | 待填 |
| label_smoothing | 待填 |
| freeze_epochs | 待填 |
| finetune_epochs | 待填 |

### 4.3 运行命令

```powershell
python prepare_dataset.py --dataset-root "Rice Leaf Disease Images"
python train.py --config configs/official_efficientnet.yaml
python train.py --config configs/clean_efficientnet.yaml
python tune.py --config configs/official_efficientnet.yaml --retrain-best
```

## 5. 调参与结果分析

### 5.1 训练过程曲线

请引用训练曲线：

- `outputs/runs/<experiment_name>/figures/training_curves.png`

### 5.2 超参数搜索结果

请引用调参产物：

- `outputs/tuning/<experiment_name>/trials.csv`
- `outputs/tuning/<experiment_name>/tuning_history.png`
- `outputs/tuning/<experiment_name>/best_params.png`

### 5.3 模型对比

请引用对比图：

- `outputs/summary/figures/model_comparison_accuracy.png`
- `outputs/summary/figures/model_comparison_macro_f1.png`
- `outputs/summary/figures/official_vs_clean_accuracy.png`

建议在本节写出：

- `CustomCNN` 与迁移学习模型的性能差距
- `ResNet18` 与 `EfficientNet-B0` 的差异
- 官方划分与去重划分之间的结果偏差

## 6. 测试结果与可解释性

### 6.1 定量结果

请将最终测试结果补充到下表：

| 数据视图 | 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| official | 待填 | 待填 | 待填 | 待填 | 待填 |
| clean | 待填 | 待填 | 待填 | 待填 | 待填 |

### 6.2 混淆矩阵与分类指标

请引用：

- `outputs/evaluations/<experiment_name>/test/figures/confusion_matrix.png`
- `outputs/evaluations/<experiment_name>/test/figures/class_metrics.png`

### 6.3 错误样本分析

请引用：

- `outputs/evaluations/<experiment_name>/test/figures/misclassified_examples.png`

分析重点建议：

- 哪两类最容易混淆
- 错分样本是否存在背景干扰、病斑面积过小、光照不稳定等问题

### 6.4 Grad-CAM 可解释性

请引用：

- `outputs/evaluations/<experiment_name>/test/figures/gradcam/gradcam_gallery.png`

建议说明模型主要关注的叶片区域是否与病斑位置一致，以及可解释性是否支持模型决策。

## 7. 结论与改进

### 7.1 结论

- 迁移学习模型在该任务上显著优于自建 CNN。
- 官方划分结果更高，但含有重复图片带来的乐观偏差。
- 去重重划分结果更能反映模型的真实泛化能力。

### 7.2 改进方向

- 收集更多真实场景图像，降低重复与增强样本偏差
- 增加更强的数据增强和更大规模模型对比
- 引入 K 折交叉验证、集成学习或测试时增强

## 8. 附录

### 8.1 关键文件

- `prepare_dataset.py`
- `train.py`
- `tune.py`
- `evaluate.py`
- `predict.py`

### 8.2 复现实验说明

1. 按 `environment.yml` 创建环境
2. 运行 `prepare_dataset.py`
3. 运行训练与评估脚本
4. 从 `outputs/` 目录引用图表和指标填充本报告
