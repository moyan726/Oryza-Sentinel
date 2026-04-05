# 水稻叶病害分类答辩 PPT 大纲

## 第 1 页：封面

- 标题：水稻叶病害图像分类课程项目答辩
- 副标题：
  - 课程：________
  - 姓名：________
  - 学号：________
  - 日期：2026-04-03
- 视觉素材：
  - `final_assets/figures/official_best_gradcam_gallery.png`
- 讲述重点：
  - 简洁说明项目主题
  - 给出“图像分类 + 课程作业 + 正式结果”三个关键词

## 第 2 页：研究背景与任务定义

- 核心内容：
  - 为什么要做水稻叶病害识别
  - 本项目识别 4 类病害
  - 目标是构建高准确率、可解释、可答辩展示的课程项目
- 建议素材：
  - `final_assets/figures/official_class_distribution.png`
- 讲述重点：
  - 明确任务是 4 分类而不是检测或分割

## 第 3 页：数据集与关键问题

- 核心内容：
  - 原始数据 5932 张，官方划分为 train / validation / test
  - 发现 224 组跨集合重复图，涉及 448 张图片
  - 如果只看官方划分，结果会偏乐观
- 建议素材：
  - `final_assets/figures/duplicate_overview.png`
- 讲述重点：
  - 强调自己做了数据审计，而不是直接套模型

## 第 4 页：双轨评测设计

- 核心内容：
  - `official`：保留官方划分，用于展示高分结果
  - `clean`：全局去重后重划分，用于验证真实泛化能力
- 建议素材：
  - `final_assets/figures/official_class_distribution.png`
  - `final_assets/figures/clean_class_distribution.png`
- 讲述重点：
  - 说明这一步是本项目的严谨性来源

## 第 5 页：技术路线总览

- 核心内容：
  - 数据审计
  - 模型设计
  - 超参数调优
  - 测试评估
  - 可解释性分析
- 讲述重点：
  - 展示完整流程，而不是只展示最后分数

## 第 6 页：模型与训练策略

- 核心内容：
  - 基线模型：CustomCNN
  - 对比模型：ResNet18
  - 主力模型：EfficientNet-B0
  - 两阶段微调、早停、学习率调度、GPU 加速
- 讲述重点：
  - EfficientNet 作为调参主线
  - ResNet18 作为亮点结果模型

## 第 7 页：EfficientNet-B0 调参结果

- 核心内容：
  - 调参对象：official + EfficientNet-B0
  - 总 trial 数：12
  - 完成 7 次、剪枝 5 次
  - 最优参数：
    - AdamW
    - lr=0.000271
    - weight_decay=1.296e-05
    - dropout=0.3727
    - label_smoothing=0.1128
    - freeze_epochs=2
    - finetune_epochs=5
    - batch_size=16
- 建议素材：
  - `final_assets/figures/tuning_history.png`
  - `final_assets/figures/best_params.png`
- 讲述重点：
  - 说明调参不是形式，而是有实质提升

## 第 8 页：正式结果总对比

- 核心内容：
  - official / clean 双视图下三模型对比
  - official 主结果：EfficientNet-B0 tuned = 99.84%
  - clean 对比最高分：ResNet18 = 100.00%
- 建议素材：
  - `final_assets/figures/model_comparison_accuracy.png`
  - `final_assets/figures/model_comparison_macro_f1.png`
  - `final_assets/figures/official_vs_clean_accuracy.png`
- 讲述重点：
  - 迁移学习整体优于自建 CNN

## 第 9 页：官方主线结果

- 核心内容：
  - official 主结果：`official_efficientnet_tuned`
  - Accuracy=99.84%
  - Macro F1=99.83%
  - 只有 1 个错分样本
- 建议素材：
  - `final_assets/figures/official_best_confusion_matrix.png`
  - `final_assets/figures/official_best_class_metrics.png`
- 讲述重点：
  - 说明调优后的 EfficientNet 是 official 口径下最优主线

## 第 10 页：clean 亮点结果

- 核心内容：
  - clean 视图下最高分模型：ResNet18
  - Accuracy=100.00%
  - Macro F1=100.00%
  - 模型容量适中，泛化更稳
- 建议素材：
  - `final_assets/figures/clean_resnet18_confusion_matrix.png`
  - `final_assets/figures/clean_resnet18_class_metrics.png`
- 讲述重点：
  - 强调这是答辩亮点页

## 第 11 页：稳定性与可解释性

- 核心内容：
  - 三个 seed 结果：100.00% / 99.79% / 100.00%
  - 平均 Accuracy=99.93%
  - 说明高分不是偶然
  - 模型主要关注叶片病斑区域
- 建议素材：
  - `final_assets/figures/resnet18_seed_consistency.png`
  - `final_assets/figures/clean_resnet18_gradcam_gallery.png`
- 讲述重点：
  - 回答“100% 是否可信”

## 第 12 页：结论与高频问答

- 结论 1：
  - 迁移学习有效，整体优于自建 CNN
- 结论 2：
  - official 结果高，但存在重复图导致的乐观偏差
- 结论 3：
  - clean 更能说明真实泛化能力，ResNet18 在 clean 上表现最亮眼
- 高频问答：
  - 为什么 official 更高？
  - 为什么 ResNet18 在 clean 更强？
  - 100% 是否可信？
  - 为什么仍保留 EfficientNet 作为调参主线？

## 讲述建议

- 总页数控制在 12 页左右，答辩时间可控制在 6-8 分钟
- 以图表讲解为主，每页正文尽量只讲 3-4 个重点
- 讲述逻辑建议：
  1. 先讲问题和数据
  2. 再讲方法与调参
  3. 再讲结果与亮点
  4. 最后讲结论与问答
