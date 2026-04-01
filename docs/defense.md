# 水稻叶病害分类答辩提纲

## 第 1 页：课题背景与目标

- 研究背景：农业病害识别对减产预警和管理决策有实际价值
- 任务定义：根据叶片图像识别四类水稻病害
- 课题目标：实现高准确率分类模型，并给出完整实验分析

## 第 2 页：数据集与问题发现

- 展示原始数据结构：`train / validation / test`
- 展示四类病害与样本量统计
- 说明关键问题：官方划分中存在跨集合重复图片
- 强调为什么不能只看高准确率，而要补充严谨评测

建议配图：

- `outputs/dataset_audit/figures/official_class_distribution.png`
- `outputs/dataset_audit/figures/duplicate_overview.png`

## 第 3 页：整体技术路线

- 数据审计与哈希去重
- 双轨评测：官方划分 + clean 去重划分
- 模型路线：`CustomCNN`、`ResNet18`、`EfficientNet-B0`
- 训练策略：冻结训练 + 微调训练 + 早停 + 学习率调度

## 第 4 页：模型与训练设计

- 为什么选择迁移学习
- 为什么 `EfficientNet-B0` 作为主力模型
- 数据增强策略如何提升泛化能力
- GPU/CPU 自动切换保证代码可复现

## 第 5 页：超参数调优

- 调参工具：Optuna
- 搜索参数：学习率、优化器、权重衰减、dropout、label smoothing、batch size、阶段轮数
- 展示调参历史和最佳参数

建议配图：

- `outputs/tuning/<experiment_name>/tuning_history.png`
- `outputs/tuning/<experiment_name>/best_params.png`

## 第 6 页：模型效果对比

- 展示不同模型的 Accuracy 和 Macro-F1
- 对比 `CustomCNN`、`ResNet18`、`EfficientNet-B0`
- 说明迁移学习为何明显优于自建 CNN

建议配图：

- `outputs/summary/figures/model_comparison_accuracy.png`
- `outputs/summary/figures/model_comparison_macro_f1.png`

## 第 7 页：官方划分与去重划分对比

- 展示两种评测口径下的结果差异
- 解释为什么官方划分更高
- 强调去重划分更有说服力

建议配图：

- `outputs/summary/figures/official_vs_clean_accuracy.png`

## 第 8 页：测试结果细节

- 展示混淆矩阵
- 展示各类别 Precision / Recall / F1
- 指出最容易混淆的类别及原因

建议配图：

- `outputs/evaluations/<experiment_name>/test/figures/confusion_matrix.png`
- `outputs/evaluations/<experiment_name>/test/figures/class_metrics.png`

## 第 9 页：错误分析与可解释性

- 展示误分类样本
- 分析可能原因：病斑相似、背景复杂、光照差异、图像质量
- 展示 Grad-CAM 热力图，说明模型关注区域

建议配图：

- `outputs/evaluations/<experiment_name>/test/figures/misclassified_examples.png`
- `outputs/evaluations/<experiment_name>/test/figures/gradcam/gradcam_gallery.png`

## 第 10 页：总结与展望

- 本项目完成了一个可复现的水稻叶病害分类系统
- 迁移学习模型取得了更优结果
- 双轨评测避免了只看高分而忽略数据泄漏的问题
- 后续可继续加入更多真实场景数据与更强模型

## 答辩时可直接口述的总结

本项目不仅完成了水稻叶病害四分类任务，还对原始数据做了重复图审计，给出了官方划分和去重重划分两套结果。这样既能展示较高准确率，也能体现实验设计的严谨性。最终主力模型采用 EfficientNet-B0，通过迁移学习、两阶段训练和超参数调优获得了较好的分类效果，同时结合混淆矩阵、错误样本和 Grad-CAM 对模型行为进行了分析。
