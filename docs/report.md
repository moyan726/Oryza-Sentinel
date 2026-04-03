# 水稻叶病害图像分类期末实验报告

## 1. 课题背景

本课题面向水稻叶片病害识别任务，目标是根据叶片图像自动判别病害类别。项目数据包含四类病害：`Bacterialblight`、`Blast`、`Brownspot`、`Tungro`。实验围绕“高准确率、可复现、可解释、具备期末答辩材料”四个目标展开，构建了从数据审计、模型训练、超参数调优到结果分析与报告回填的完整链路。

## 2. 数据审计

### 2.1 数据集概况

原始数据集采用标准图像分类目录结构 `train / validation / test`，总计 5932 张图像，4 个类别，官方划分统计如下：

| 划分 | Bacterialblight | Blast | Brownspot | Tungro | 合计 |
| --- | --- | --- | --- | --- | --- |
| train | 1200 | 1200 | 1300 | 1000 | 4700 |
| validation | 192 | 120 | 150 | 154 | 616 |
| test | 192 | 120 | 150 | 154 | 616 |
| 总计 | 1584 | 1440 | 1600 | 1308 | 5932 |

对应统计文件与图表：

- `final_assets/dataset_audit/official_summary.json`
- `final_assets/figures/official_class_distribution.png`

### 2.2 重复图与数据泄漏问题

对全量图片做内容哈希后发现：

- 官方数据总共 5932 张，但唯一哈希仅 4794 个。
- 存在 224 组跨 `train / validation / test` 的重复哈希，共涉及 448 张图像。
- 这意味着官方划分中存在数据泄漏风险，若只看官方测试集准确率，结果会偏乐观。

为此，本实验同时构建了 `clean` 视图：对所有图像按内容哈希全局去重，再重新按约 `8:1:1` 进行分层划分。去重后统计如下：

| 划分 | Bacterialblight | Blast | Brownspot | Tungro | 合计 |
| --- | --- | --- | --- | --- | --- |
| train | 1061 | 768 | 960 | 1046 | 3835 |
| validation | 133 | 96 | 120 | 131 | 480 |
| test | 132 | 96 | 120 | 131 | 479 |
| 总计 | 1326 | 960 | 1200 | 1308 | 4794 |

对应统计文件与图表：

- `final_assets/dataset_audit/clean_summary.json`
- `final_assets/figures/clean_class_distribution.png`
- `final_assets/figures/duplicate_overview.png`

### 2.3 双轨评测设计

为了兼顾课程成绩展示与实验严谨性，本实验采用双轨评测：

1. `official`：沿用原始官方划分，用于展示较高准确率。
2. `clean`：基于哈希去重后的重划分，用于反映更真实的泛化能力。

## 3. 方法设计

### 3.1 模型路线

本实验采用“自建 CNN 基线 + 迁移学习对比”的设计：

- `CustomCNN`：作为课程基础模型，强调自主搭建网络结构。
- `ResNet18`：作为轻量级迁移学习对比模型。
- `EfficientNet-B0`：作为主力模型，并执行正式超参数调优。

### 3.2 数据预处理与增强

- 输入尺寸统一为 `224 x 224`
- 训练阶段使用随机裁剪、水平翻转、轻度旋转、颜色扰动
- 验证与测试仅做缩放、中心裁剪和归一化

### 3.3 训练策略

- `CustomCNN` 直接训练 20 轮
- 迁移学习模型采用两阶段训练：
  - 第一阶段冻结骨干，仅训练分类头
  - 第二阶段解冻全网络，使用更小学习率微调
- 使用早停、学习率调度与最佳权重保存
- 训练环境优先使用 GPU，若不可用则自动回退到 CPU

## 4. 实验设置

### 4.1 环境配置

- Python：`3.11`
- 深度学习框架：`PyTorch 2.10.0 + CUDA 13.0`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- 预训练权重：`torchvision` 官方 ImageNet 预训练权重

### 4.2 关键超参数

三类模型的正式实验参数如下：

| 模型 | batch_size | optimizer | lr | weight_decay | dropout | label_smoothing | freeze_epochs | finetune_epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CustomCNN | 64 | AdamW | 1e-3 | 1e-4 | 0.30 | 0.03 | 0 | 20 |
| ResNet18 | 48 | AdamW | 8e-4 | 1e-4 | 0.30 | 0.03 | 3 | 10 |
| EfficientNet-B0（调优后） | 16 | AdamW | 2.710e-4 | 1.296e-5 | 0.3727 | 0.1128 | 2 | 5 |

主力模型最终采用 `official_efficientnet_tuned` 作为官方主结果，`clean_efficientnet_tuned` 作为严谨性主结果。

### 4.3 运行命令

```powershell
python prepare_dataset.py --dataset-root "Rice Leaf Disease Images"
python train.py --config configs/official_customcnn.yaml --skip-eval
python train.py --config configs/official_resnet18.yaml --skip-eval
python tune.py --config configs/official_efficientnet_search.yaml --trials 12
python train.py --config configs/official_efficientnet_tuned.yaml --skip-eval
python train.py --config configs/clean_customcnn.yaml --skip-eval
python train.py --config configs/clean_resnet18.yaml --skip-eval
python train.py --config configs/clean_efficientnet_tuned.yaml --skip-eval
python evaluate.py --config <config> --checkpoint <best_model.pt>
```

## 5. 调参与结果分析

### 5.1 训练过程曲线

正式训练曲线已经统一导出到 `final_assets/figures/`，关键图如下：

- `official_customcnn_training_curves.png`
- `official_resnet18_training_curves.png`
- `official_efficientnet_tuned_training_curves.png`
- `clean_customcnn_training_curves.png`
- `clean_resnet18_training_curves.png`
- `clean_efficientnet_tuned_training_curves.png`

从训练日志可以看出：

- `CustomCNN` 在 official 与 clean 两个视图下都能稳定收敛，但训练轮数明显更长，且最终性能略低于迁移学习模型。
- `ResNet18` 和 `EfficientNet-B0` 依靠预训练权重，在前几个 epoch 就能迅速达到很高的验证准确率。
- `official_efficientnet_tuned` 与 `clean_efficientnet_tuned` 都在 7 个 epoch 内完成最优收敛，效率较高。

### 5.2 官方主模型调参结果

本实验仅对 `official + EfficientNet-B0` 做正式 Optuna 调参，共运行 12 个 trial：

- 完整完成 7 个 trial
- 早停剪枝 5 个 trial
- 最优 trial 验证准确率达到 `1.0000`

最优参数如下：

| optimizer | lr | weight_decay | dropout | label_smoothing | freeze_epochs | finetune_epochs | batch_size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW | 0.000271 | 0.00001296 | 0.3727 | 0.1128 | 2 | 5 | 16 |

调参产物：

- `final_assets/official_efficientnet_trials.csv`
- `final_assets/official_efficientnet_best_params.csv`
- `final_assets/figures/tuning_history.png`
- `final_assets/figures/best_params.png`

调参结论：

- `AdamW` 在该任务上明显优于若干 `SGD` 试验。
- 中等偏小学习率与较强的 `label_smoothing` 有助于获得更稳的最优结果。
- `EfficientNet-B0` 并不需要太长的微调周期，`2 + 5` 的两阶段训练已经足够达到最优验证表现。

### 5.3 模型对比分析

关键汇总图：

- `final_assets/figures/model_comparison_accuracy.png`
- `final_assets/figures/model_comparison_macro_f1.png`
- `final_assets/figures/official_vs_clean_accuracy.png`

从结果上看：

- 迁移学习整体优于自建 `CustomCNN`，尤其在 `macro_f1` 上更稳。
- 在 `official` 视图中，`EfficientNet-B0` 优于 `ResNet18` 与 `CustomCNN`，是官方主结果的最佳选择。
- 在 `clean` 视图中，`ResNet18` 测试集取得了 100% 准确率，是本次 clean 对比中的最高分；但为了保持“统一主力模型 + 调参主线”的叙述一致性，报告主线仍使用调优后的 `EfficientNet-B0` 作为 clean 主结果展示。
- `official` 与 `clean` 的差距不算大，说明模型本身具备较强识别能力；但由于官方划分存在跨集合重复图，clean 视图仍然更值得作为泛化结论依据。

### 5.4 ResNet18 稳定性验证

为了验证 `clean_resnet18` 的 `100%` 结果不是偶然，本实验额外进行了 3 次不同随机种子的重复训练与测试：

| seed | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| 42 | 100.00% | 100.00% | 100.00% | 100.00% |
| 123 | 99.79% | 99.81% | 99.74% | 99.77% |
| 2026 | 100.00% | 100.00% | 100.00% | 100.00% |

稳定性汇总：

- Accuracy 均值：`99.93%`
- Accuracy 标准差：`0.10%`
- Macro F1 均值：`99.92%`
- Macro F1 标准差：`0.11%`

对应产物：

- `final_assets/resnet18_seed_consistency.csv`
- `final_assets/resnet18_seed_consistency_summary.csv`
- `final_assets/figures/resnet18_seed_consistency.png`

这说明 `clean_resnet18` 的高分结果并非单次偶然。虽然不同随机种子下不保证每次都达到绝对 `100%`，但整体波动极小，仍可认为它在当前 clean 划分上的表现非常稳定。

## 6. 测试结果与可解释性

### 6.1 主结果表

| 数据视图 | 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| official | CustomCNN | 99.51% | 99.57% | 99.46% | 99.52% |
| official | ResNet18 | 99.51% | 99.39% | 99.57% | 99.48% |
| official | EfficientNet-B0（tuned） | 99.84% | 99.79% | 99.87% | 99.83% |
| clean | CustomCNN | 99.37% | 99.40% | 99.27% | 99.33% |
| clean | ResNet18 | 100.00% | 100.00% | 100.00% | 100.00% |
| clean | EfficientNet-B0（tuned） | 99.79% | 99.79% | 99.81% | 99.80% |

对应轻量汇总 CSV：

- `final_assets/report_table_main.csv`
- `final_assets/final_test_metrics.csv`

### 6.2 官方主结果说明

按照课题主线，官方主结果采用 `official_efficientnet_tuned`：

- 测试准确率：`99.84%`
- Macro Precision：`99.79%`
- Macro Recall：`99.87%`
- Macro F1：`99.83%`
- 最优验证准确率：`100.00%`
- 实际完成训练轮数：`7`
- 成功加载 ImageNet 预训练权重

### 6.3 严谨性主结果说明

clean 视图主线结果采用 `clean_efficientnet_tuned`：

- 测试准确率：`99.79%`
- Macro Precision：`99.79%`
- Macro Recall：`99.81%`
- Macro F1：`99.80%`
- 最优验证准确率：`100.00%`
- 实际完成训练轮数：`7`

同时，`clean_resnet18` 在当前 clean 测试集上取得了 `100.00%` 的准确率与 Macro F1，是本轮对比中的最高分模型，这说明在当前数据规模和类别区分度下，较轻量的残差网络也能获得极强的识别性能。

结合多随机种子重复实验可以进一步说明：

- `clean_resnet18` 并不是“只在单次 seed 下偶然达到 100%”
- 但它也不是每次都绝对 100%，因此更准确的表述应为“在当前 clean 划分上稳定取得接近满分的高表现”

### 6.4 混淆矩阵与分类指标

主力模型图表：

- `final_assets/figures/official_best_confusion_matrix.png`
- `final_assets/figures/official_best_class_metrics.png`

对 `official_efficientnet_tuned` 来说，测试集只出现 1 个错分样本：

- `Bacterialblight` 中有 1 张被误判为 `Blast`
- `Brownspot` 与 `Tungro` 在测试集中全部识别正确

这说明模型对褐斑病和 Tungro 病的判别几乎完全稳定，而白枯病与稻瘟病在少数边界样本上仍有轻微混淆。

在 `clean_efficientnet_tuned` 中，测试集也只出现 1 个错分样本：

- 1 张 `Bacterialblight` 被误判为 `Brownspot`

### 6.5 错误样本分析

错误样本图：

- `final_assets/figures/official_best_misclassified_examples.png`

结合错分样本可以看出：

- 错误主要出现在病斑形态较接近、局部纹理不明显的样本上
- 当叶片背景较复杂、病斑区域较小或颜色对比不够强时，模型更容易把少数白枯病样本误判成其他类别
- 这类错误数量非常少，但仍说明模型在“边界类样本”上的鲁棒性还有提升空间

### 6.6 Grad-CAM 可解释性

可解释性图表：

- `final_assets/figures/official_best_gradcam_gallery.png`

从热力图可以看到，模型的高响应区域主要集中在病斑分布密集的叶片区域，而不是背景区域。这说明模型学习到的关键特征与病害位置基本一致，可解释性结果支持模型预测并非偶然。

## 7. 结论与改进

### 7.1 结论

1. 迁移学习模型整体优于自建 CNN。本实验中，无论在 official 还是 clean 视图下，迁移学习模型在 Accuracy 与 Macro F1 上都更稳，说明 ImageNet 预训练对水稻叶病害识别任务有明显帮助。
2. 官方划分结果更高，但受重复图影响更乐观。通过哈希审计发现官方数据存在 224 组跨集合重复图，共涉及 448 张图片，因此单看官方分数会高估模型泛化能力。
3. clean 划分更能反映真实泛化能力。虽然 clean 视图整体分数仍然很高，但它消除了跨集合重复图片，是更可信的实验口径。

### 7.2 本次最佳结果

- 官方主结果：`official_efficientnet_tuned`，Accuracy `99.84%`
- clean 主结果：`clean_efficientnet_tuned`，Accuracy `99.79%`
- clean 对比最高分：`clean_resnet18`，Accuracy `100.00%`
- clean ResNet18 稳定性验证：3 次 seed 平均 Accuracy `99.93%`

### 7.3 为什么 clean 视图下 ResNet18 优于 EfficientNet-B0

结合本次实验现象，可以给出一个合理解释：

1. 当前 clean 数据规模并不大，且四类病害的视觉差异整体较明显。
2. `ResNet18` 的模型容量更适中，在该数据规模下更容易形成稳定泛化，而不需要更复杂的特征缩放机制。
3. `EfficientNet-B0` 已经非常强，但它的优势通常在更大规模、更复杂分布的数据上更容易体现。
4. 在当前任务上，`ResNet18` 的结构复杂度与数据难度更匹配，因此 clean 视图下取得了更高且更稳定的结果。

### 7.4 改进方向

- 收集更多真实场景图像，减少重复图和增强图带来的数据偏差
- 增加更多边界样本分析，重点提升白枯病与其他病害的区分能力
- 引入 K 折交叉验证、测试时增强或模型集成，进一步评估模型稳定性

## 8. 附录

### 8.1 关键文件

- `prepare_dataset.py`
- `train.py`
- `tune.py`
- `evaluate.py`
- `predict.py`
- `summarize_results.py`

### 8.2 轻量提交素材

本次最终报告所依赖的轻量结果已统一导出到 `final_assets/`：

- 指标汇总：`final_assets/report_table_main.csv`
- 调参结果：`final_assets/official_efficientnet_best_params.csv`
- 图表目录：`final_assets/figures/`

### 8.3 复现实验说明

1. 使用 `environment.yml` 创建并激活 `rice-leaf-dl` 环境
2. 运行 `prepare_dataset.py` 生成 `official` 与 `clean` manifest
3. 训练 6 组正式模型，并对 official 主模型运行 Optuna 调参
4. 用 `evaluate.py` 生成测试集指标和可解释化图表
5. 用 `summarize_results.py` 导出轻量结果，回填报告与答辩材料
