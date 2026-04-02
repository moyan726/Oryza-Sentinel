# 水稻叶病害分类答辩提纲

## 第 1 页：课题背景与目标

- 课题目标：根据叶片图像识别 4 类水稻病害
- 课程要求：高准确率、超参数调优、图表分析、完整实验报告
- 实际交付：训练代码、调参代码、可视化图表、中文报告与答辩材料

## 第 2 页：数据集与关键问题

- 原始数据总量：5932 张图像，4 类病害
- 官方划分：`train 4700 / validation 616 / test 616`
- 哈希审计发现：224 组跨集合重复图，涉及 448 张图片
- 因此采用双轨评测：
  - `official` 展示高分结果
  - `clean` 反映更真实的泛化能力

建议展示：

- `final_assets/figures/official_class_distribution.png`
- `final_assets/figures/duplicate_overview.png`

## 第 3 页：整体技术路线

- 数据审计与去重重划分
- 自建 CNN 基线 + 迁移学习对比
- 模型矩阵：
  - `CustomCNN`
  - `ResNet18`
  - `EfficientNet-B0`
- 训练策略：
  - 两阶段微调
  - 早停
  - 学习率调度
  - GPU 加速

## 第 4 页：为什么选择 EfficientNet-B0 作为主线

- 参数量适中，适合课程作业训练预算
- 配合预训练权重收敛快
- 易于进行超参数调优和结果展示
- 作为主力模型，official 与 clean 两条主线都保持统一

## 第 5 页：超参数调优设计

- 调参对象：`official + EfficientNet-B0`
- 调参方法：Optuna
- 总 trial 数：12
- 完成 trial：7
- 提前剪枝：5

最佳参数：

- optimizer：`AdamW`
- lr：`0.000271`
- weight_decay：`1.296e-05`
- dropout：`0.3727`
- label_smoothing：`0.1128`
- freeze_epochs：`2`
- finetune_epochs：`5`
- batch_size：`16`

建议展示：

- `final_assets/figures/tuning_history.png`
- `final_assets/figures/best_params.png`

## 第 6 页：模型对比结果

| 数据视图 | 模型 | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| official | CustomCNN | 99.51% | 99.52% |
| official | ResNet18 | 99.51% | 99.48% |
| official | EfficientNet-B0（tuned） | 99.84% | 99.83% |
| clean | CustomCNN | 99.37% | 99.33% |
| clean | ResNet18 | 100.00% | 100.00% |
| clean | EfficientNet-B0（tuned） | 99.79% | 99.80% |

答辩时重点说明：

- 迁移学习整体优于自建 CNN
- official 主结果采用调优后的 EfficientNet-B0
- clean 视图中 ResNet18 得分最高，但主线模型仍保持 EfficientNet-B0 统一叙述

建议展示：

- `final_assets/figures/model_comparison_accuracy.png`
- `final_assets/figures/model_comparison_macro_f1.png`
- `final_assets/figures/official_vs_clean_accuracy.png`

## 第 7 页：官方主结果分析

官方主结果：`official_efficientnet_tuned`

- Accuracy：`99.84%`
- Macro Precision：`99.79%`
- Macro Recall：`99.87%`
- Macro F1：`99.83%`
- 最优验证准确率：`100.00%`
- 实际完成训练轮数：`7`

混淆矩阵结论：

- 仅 1 张 `Bacterialblight` 被误判为 `Blast`
- `Brownspot` 和 `Tungro` 测试集全部识别正确

建议展示：

- `final_assets/figures/official_best_confusion_matrix.png`
- `final_assets/figures/official_best_class_metrics.png`

## 第 8 页：clean 结果与严谨性说明

clean 主结果：`clean_efficientnet_tuned`

- Accuracy：`99.79%`
- Macro F1：`99.80%`

clean 对比最高分：`clean_resnet18`

- Accuracy：`100.00%`
- Macro F1：`100.00%`

答辩时可这样说：

“我没有只展示官方高分结果，还额外做了去重重划分。这样既能说明模型性能高，也能说明实验设计是严谨的。”

## 第 9 页：错误分析与可解释性

- `official_efficientnet_tuned` 的错误极少，主要是白枯病边界样本与其他病害的局部纹理相似
- `clean_efficientnet_tuned` 也仅有 1 个错分样本
- Grad-CAM 显示模型主要关注叶片病斑区域，而非背景

建议展示：

- `final_assets/figures/official_best_misclassified_examples.png`
- `final_assets/figures/official_best_gradcam_gallery.png`

## 第 10 页：总结

- 本项目完成了从数据审计到模型训练、调参、评估和答辩材料生成的完整流程
- 迁移学习是本任务的最优主路线
- 官方划分结果更高，但 clean 划分更能说明模型真实泛化能力
- 最终结果已经达到课程作业中“高准确率 + 调参 + 图表分析”的完整要求

## 可直接口述的总结稿

本项目针对水稻叶病害四分类任务，构建了一个完整的深度学习实验流程。实验首先对原始数据进行了哈希审计，发现官方划分中存在跨集合重复图片，因此我采用了 official 和 clean 两套评测口径。模型方面，我设计了自建 CNN 基线，并使用 ResNet18 和 EfficientNet-B0 做迁移学习对比。随后对 official 视图下的 EfficientNet-B0 进行了 12 次 Optuna 超参数调优，最终官方主结果达到 99.84% 的测试准确率，clean 主结果达到 99.79%。同时，通过混淆矩阵、错误样本和 Grad-CAM，我进一步验证了模型判断的合理性。整体来看，这个项目既满足了课程对准确率和调参分析的要求，也兼顾了实验设计的严谨性。
