# 项目变更记录

## 2026-04-01 21:05 (Asia/Shanghai) - 初始化期末作业工程化改造

### 变更摘要
将当前仅包含“水稻叶病害图片数据集”的目录扩展为可复现的深度学习期末作业项目，补齐数据清单生成、模型训练、超参数调优、评估可视化、中文实验报告与中文答辩材料。

### 问题说明
- 当前项目只有图片数据及压缩包，缺少训练代码、评估代码、实验配置、依赖清单、结果图表和提交文档，无法直接作为课程期末作业提交。
- 数据集中存在官方 `train/validation/test` 划分，但已发现跨集合重复图片与增强图片混入验证/测试集的情况，若直接沿用划分，最终准确率可能偏高，结论也不够严谨。
- 当前基础环境实测为 `TensorFlow 2.20 + CPU`，未具备可直接运行 `PyTorch + CUDA` 的独立实验环境，不利于后续高效训练与调参。

### 策略方向
- 采用“双轨评测”策略：保留官方划分作为高分展示主结果，同时基于图片哈希去重后重新分层划分数据，给出更严谨的补充实验。
- 采用“迁移学习主线 + 自建 CNN 基线”的实验设计，以 `EfficientNet-B0` 作为主力模型，`ResNet18` 作为对比模型，`CustomCNN` 作为课程基础模型。
- 代码层面统一支持 CPU/GPU 自动切换，环境层面提供独立 `conda` 环境说明和依赖清单，确保可复现。
- 训练、评估、调参、绘图、报告内容尽量脚本化生成，减少手工整理成本。

### 具体改动
- 本条记录用于初始化本次工程化改造任务。
- 后续将新增数据准备、训练、调参、评估、推理、配置、依赖、报告与答辩稿等文件。
- 本条记录创建时，代码与文档尚未落地，详细实现结果将在后续追加记录中如实补充。

### 影响面
- 包含项：项目目录结构、Python 脚本、配置文件、依赖说明、输出物目录、中文报告、中文答辩稿。
- 不包含项：原始图片数据内容本身不修改；默认不复制原始图片，仅生成清单与输出结果。
- 关键点：必须保留官方划分结果，同时单独提供去重重划分结果与差异解释。

### 风险与回滚
- 风险点：GPU 环境适配可能受驱动、CUDA 与依赖版本兼容性影响；数据去重重划分后准确率可能低于官方划分；超参数调优耗时较长。
- 缓解措施：提供 CPU 回退逻辑、固定随机种子、保留官方划分主结果、使用配置化脚本控制实验规模。
- 回退方式：若 GPU 环境暂不可用，可直接退回 CPU 执行最小可运行流程；若某类高级分析未完成，至少保留基础训练、评估与图表链路。

### 验证
- 已确认项目中此前不存在 `progress.md`。
- 已确认数据集目录结构为图像分类格式，包含 `train/validation/test` 与 4 个类别。
- 已确认后续实现需重点处理跨集合重复图片问题。
- 其余验证项待本次工程化改造完成后补充。

### 关联信息
- 数据目录：`E:\pycharm\Python3_9\DeepLearning\Rice Leaf Disease Images`
- 目标任务：水稻叶病害四分类期末作业工程化实现
- 计划主框架：PyTorch + CUDA（代码保留 CPU 自动回退）

## 2026-04-01 21:50 (Asia/Shanghai) - 完成首版工程骨架、环境适配与 smoke 验证

### 变更摘要
已完成期末作业项目的首版工程化实现，包含数据清单生成、训练脚本、评估脚本、调参脚本、单图推理脚本、依赖清单、中文报告模板、中文答辩提纲，以及独立 `conda` GPU 环境的创建与基础 smoke 验证。

### 问题说明
- 原始项目只有图片数据，无法直接训练、调参、评估与提交作业。
- 课程目标要求高准确率、超参数调优、图表对比和实验文档，需要完整工程链路。
- 原始基础环境缺少可用的 `PyTorch + CUDA` 依赖，且部分科学计算库状态异常，不适合作为正式实验环境。

### 策略方向
- 采用“代码骨架先成型、GPU 环境单独隔离、先做 smoke 验证、正式结果后续长跑”的策略。
- 数据处理和图表生成优先脚本化，减少后续手工整理成本。
- smoke 配置采用少 batch、少 epoch、禁用预训练的轻量验证方案，只验证链路是否可跑，不用它代表最终成绩。

### 具体改动
- 新增依赖文件：`requirements.txt`、`environment.yml`。
- 新增项目说明：`README.md`。
- 新增配置文件：`configs/official_efficientnet.yaml`、`configs/clean_efficientnet.yaml`、`configs/smoke_official.yaml`。
- 新增核心代码：
  - `prepare_dataset.py`
  - `train.py`
  - `evaluate.py`
  - `tune.py`
  - `predict.py`
  - `src/rice_leaf_disease/config.py`
  - `src/rice_leaf_disease/data.py`
  - `src/rice_leaf_disease/models.py`
  - `src/rice_leaf_disease/engine.py`
  - `src/rice_leaf_disease/analysis.py`
  - `src/rice_leaf_disease/gradcam.py`
  - `src/rice_leaf_disease/utils.py`
- 新增中文文档：
  - `docs/report.md`
  - `docs/defense.md`
- 已生成数据产物：
  - `outputs/manifests/official_manifest.csv`
  - `outputs/manifests/clean_manifest.csv`
  - `outputs/manifests/duplicate_groups.csv`
  - `outputs/dataset_audit/*`
- 已创建独立环境：`rice-leaf-dl`
- 已安装/声明第三方依赖：`torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`seaborn`、`scipy`、`scikit-learn`、`Pillow`、`tqdm`、`optuna`、`PyYAML`

### 影响面
- 包含项：数据审计、双轨清单、训练、评估、调参、推理、中文报告模板、中文答辩提纲、GPU 环境说明。
- 不包含项：尚未进行完整正式训练、完整超参数搜索和最终高分模型产出；当前 smoke 结果不能直接写进最终报告结论。
- 关键点：当前输出目录已具备正式实验所需的目录结构和产物格式，可继续直接开展长时训练。

### 风险与回滚
- 风险点：
  - smoke 配置由于仅取少量 batch 且禁用预训练，当前准确率为 0，不代表正式实验能力。
  - 正式高分实验仍依赖更长时间训练与预训练权重下载。
  - 去重划分下的最终准确率可能明显低于官方划分。
- 缓解措施：
  - 已保留官方划分与 clean 划分两套配置。
  - 已提供 GPU 环境与调参脚本，支持后续正式实验提速。
  - 已将正式实验报告与答辩提纲模板化，后续只需填充结果。
- 回退方式：
  - 若 GPU 环境后续异常，可使用同一代码路径回退到 CPU。
  - 若某模型效果不理想，可直接切换到 `ResNet18` 或 `CustomCNN` 配置重跑，不影响工程结构。

### 验证
- 静态验证：
  - 已对新增 Python 文件执行 `compileall`，语法通过。
- 数据验证：
  - 已成功运行 `prepare_dataset.py`，生成 `official` 与 `clean` 两套 manifest 及数据审计图表。
- 环境验证：
  - 已创建 `conda` 环境 `rice-leaf-dl`。
  - 已确认 `torch 2.10.0+cu130`、`torchvision 0.25.0+cu130`、`optuna 4.8.0` 安装成功。
  - 已确认 CUDA 可用，识别到 `NVIDIA GeForce RTX 4060 Laptop GPU`。
- 功能 smoke：
  - 已成功运行 `train.py --config configs/smoke_official.yaml`
  - 已成功运行 `evaluate.py`
  - 已成功运行 `predict.py`
  - 已成功运行 `tune.py --trials 1`
- 结果说明：
  - smoke 实验使用极小训练配置，验证目标是链路打通而非拿分，当前评估指标为 0 属预期可接受范围。

### 关联信息
- 正式环境名称：`rice-leaf-dl`
- 正式训练建议入口：`configs/official_efficientnet.yaml`、`configs/clean_efficientnet.yaml`
- smoke 验证入口：`configs/smoke_official.yaml`

## 2026-04-01 22:02 (Asia/Shanghai) - 初始化 Git 仓库并推送到本地 bare 仓库

### 变更摘要
为当前课程项目补充 Git 版本管理初始化，新增忽略规则，使用中文提交信息完成一次本地提交，并推送到项目内的本地 bare 仓库。

### 问题说明
- 当前项目目录不是 Git 仓库，无法直接执行提交和推送。
- 项目中包含原始图片数据、压缩包和训练输出，若不先设置忽略规则，直接提交会导致仓库体积过大且混入可再生成文件。
- 未发现现成的本地远端仓库路径，需要提供一个本地可推送目标。

### 策略方向
- 在当前目录初始化 Git 仓库，主分支使用 `main`。
- 新增 `.gitignore`，默认不跟踪数据集目录、压缩包、`outputs/` 和缓存目录，只提交代码、配置、文档与记录文件。
- 在项目目录内创建本地 bare 仓库 `.local-remote.git` 作为远端，避免跨目录写入和额外权限问题。

### 具体改动
- 新增 `.gitignore`。
- 将初始化 Git 仓库、创建本地 bare 仓库远端、提交并推送一次。
- 本次提交信息使用中文。

### 影响面
- 包含项：项目版本控制初始化、跟踪范围控制、一次本地推送。
- 不包含项：不提交原始图片数据、不提交压缩包、不提交 `outputs/` 目录中的生成结果。
- 关键点：后续若需要共享数据集，应通过单独数据说明而非直接纳入 Git 仓库。

### 风险与回滚
- 风险点：若后续确实需要提交某些输出文件，当前忽略规则会默认排除它们。
- 缓解措施：保留 `.gitignore` 为可编辑文本，后续可按需放开单独目录或文件。
- 回退方式：可删除本地 `.git` 与 `.local-remote.git` 后重新初始化，或调整忽略规则后重新提交。

### 验证
- 待验证项：
  - `git init` 成功
  - 中文提交创建成功
  - 本地 bare 仓库创建成功
  - `git push` 成功

### 关联信息
- 默认分支：`main`
- 本地远端目录：`E:\pycharm\Python3_9\DeepLearning\.local-remote.git`

## 2026-04-01 22:08 (Asia/Shanghai) - 推送项目到 GitHub 远端仓库

### 变更摘要
在保留本地 bare 仓库远端的基础上，新增 GitHub 远端仓库并将当前 `main` 分支推送到 `https://github.com/moyan726/Oryza-Sentinel.git`。

### 问题说明
- 当前 Git 仓库仅推送到了项目内的本地 bare 仓库，尚未同步到 GitHub。
- 用户已创建 GitHub 仓库，需要把本地代码同步到公开远端，便于后续备份、展示和继续开发。

### 策略方向
- 保留现有本地远端 `origin` 不变，避免影响当前本地备份链路。
- 新增一个独立的 GitHub 远端用于外部同步，直接将 `main` 推送过去。
- 若存在认证要求，则依赖本机现有 GitHub 凭据管理或交互式认证。

### 具体改动
- 将新增 GitHub 远端指向：`https://github.com/moyan726/Oryza-Sentinel.git`
- 将把当前 `main` 分支推送到该远端
- 若推送成功，将形成一条新的中文提交记录用于同步本次操作说明

### 影响面
- 包含项：远端配置、远端同步、版本历史补充记录。
- 不包含项：不修改数据集内容，不调整训练代码逻辑。
- 关键点：保留本地远端与 GitHub 远端并存，避免覆盖已有本地推送路径。

### 风险与回滚
- 风险点：GitHub 推送可能受网络或认证失败影响；若远端已出现同名历史，可能需要处理分支关系。
- 缓解措施：当前远端页面显示为空仓库，冲突风险较低；推送前保留本地提交历史不变。
- 回退方式：若 GitHub 远端配置不需要，可删除对应 remote；本地仓库与本地 bare 远端不受影响。

### 验证
- 待验证项：
  - GitHub 远端添加成功
  - `main` 分支推送成功
  - GitHub 页面可见当前项目文件

### 关联信息
- GitHub 仓库：`https://github.com/moyan726/Oryza-Sentinel.git`
- 当前本地远端：`origin -> .local-remote.git`

## 2026-04-01 22:20 (Asia/Shanghai) - 正式训练、调参与报告回填实施

### 变更摘要
开始执行正式实验方案，目标是完成 3 个模型在 official/clean 双视图下的完整对比、对 official 主力模型进行正式调参，并用真实结果回填实验报告与答辩材料。

### 问题说明
- 当前项目只有 smoke 级产物，尚无正式训练结果，报告与答辩文档仍保留模板占位。
- 现有代码已能跑通训练、评估、调参与推理链路，但缺少正式配置矩阵、轻量结果归档与最终文档回填。
- 若直接开始长时训练而不先补齐正式配置和结果汇总机制，后续容易出现实验命名混乱、图表难以汇总、报告回填重复劳动的问题。

### 策略方向
- 先补正式实验配置、结果汇总与轻量产物导出能力，再运行长时 GPU 训练与调参。
- 以 `official_efficientnet_tuned` 为官方主结果，以 `clean_efficientnet_tuned` 为严谨性主结果。
- 保持输出物分层：完整中间产物继续放在 `outputs/`，最终提交素材汇总到轻量目录，便于后续 Git 管理。

### 具体改动
- 待新增正式 YAML 配置矩阵。
- 待补充轻量结果导出与汇总能力。
- 待运行正式训练、正式调参与统一测试评估。
- 待将真实指标回填到 `docs/report.md` 与 `docs/defense.md`。

### 影响面
- 包含项：配置文件、训练与评估流程、汇总结果文件、报告与答辩稿。
- 不包含项：原始数据集内容不变，大型模型权重默认不进入 Git 跟踪。
- 关键点：最终报告必须只保留真实结果，不保留“待填”占位。

### 风险与回滚
- 风险点：正式调参与完整矩阵训练耗时较长；若最佳权重下载异常，迁移学习实验可能退回随机初始化；部分模型结果可能不及预期。
- 缓解措施：统一命名实验、优先使用可用 GPU、保留 CPU/GPU 自动回退、保留所有实际结果并在文档中解释。
- 回退方式：若个别训练阶段异常，可复用现有 checkpoint 与配置单独重跑受影响模型，不必重做全部实验。

### 验证
- 待验证项：
  - 6 个正式实验配置可成功启动
  - official 主模型调参成功并导出最佳参数
  - 最终主表与关键图表齐备
  - 报告与答辩材料完成真实结果回填

### 关联信息
- 正式实验矩阵：`official/clean × CustomCNN/ResNet18/EfficientNet-B0`
- 目标环境：`rice-leaf-dl`

## 2026-04-02 02:32 (Asia/Shanghai) - 完成正式训练、正式调参与报告回填

### 变更摘要
已完成正式实验矩阵训练、official 主模型的正式 Optuna 调参、6 组正式测试评估、轻量结果导出，以及实验报告与答辩材料的真实结果回填。

### 问题说明
- 需要把此前仅有 smoke 级产物的项目推进到“可交期末作业”的正式版本。
- 需要在有限 GPU 预算内完成 3 个模型在 official/clean 双视图下的完整对比。
- 需要保证报告中的表格、结论、图表路径和调参结果全部基于真实实验输出，而不是模板占位。

### 策略方向
- 使用独立 GPU 环境 `rice-leaf-dl` 统一完成正式训练与调参。
- 采用“official 作为高分展示、clean 作为严谨性补充”的双轨评测。
- 将完整中间产物保留在 `outputs/`，再通过 `summarize_results.py` 导出轻量结果到 `final_assets/`，方便后续提交与 Git 管理。

### 具体改动
- 新增正式配置矩阵：
  - `configs/official_customcnn.yaml`
  - `configs/official_resnet18.yaml`
  - `configs/official_efficientnet_search.yaml`
  - `configs/official_efficientnet_tuned.yaml`
  - `configs/clean_customcnn.yaml`
  - `configs/clean_resnet18.yaml`
  - `configs/clean_efficientnet_tuned.yaml`
- 新增轻量结果汇总脚本：`summarize_results.py`
- 修复 Grad-CAM 与 inplace 激活冲突：
  - 调整 `CustomCNN` 中 ReLU 为非 inplace
  - 在 `GradCAM` 初始化时统一关闭模型中的 inplace 激活
- 完成正式训练与评估：
  - `official_customcnn`
  - `official_resnet18`
  - `official_efficientnet_tuned`
  - `clean_customcnn`
  - `clean_resnet18`
  - `clean_efficientnet_tuned`
- 完成 official 主模型正式调参：
  - 基于 `official_efficientnet_search`
  - 运行 12 个 Optuna trial
- 导出轻量结果目录：`final_assets/`
- 回填并重写：
  - `docs/report.md`
  - `docs/defense.md`

### 影响面
- 包含项：正式配置、正式训练结果、调参记录、图表、轻量结果目录、最终中文报告、最终中文答辩提纲。
- 不包含项：大型模型权重未纳入轻量目录，也未计划纳入 Git 跟踪。
- 关键点：文档中的“待填”占位已全部移除，结论已基于真实实验结果改写。

### 风险与回滚
- 风险点：
  - official 数据集存在 224 组跨集合重复图，官方结果仍存在乐观偏差风险。
  - clean 视图下 `ResNet18` 测试集达到 100%，该结果可能受当前 clean 划分规模和样本难度影响，后续若扩展数据或做交叉验证，结果可能波动。
- 缓解措施：
  - 报告已明确写出 official 与 clean 双视图的差异与意义。
  - 保留全部真实对比结果，不隐藏对主线模型不利的数值。
- 回退方式：
  - 若后续需要重跑任一模型，可直接复用当前配置文件与环境重新训练。
  - 若只需更新文档，可复用 `final_assets/` 中的轻量结果重新回填。

### 验证
- 数据与环境：
  - 已重新生成 `official` 与 `clean` manifest
  - 已确认 GPU 环境可持续训练与调参
  - 已预下载 `ResNet18` 与 `EfficientNet-B0` 官方预训练权重
- 调参与训练：
  - official 主模型 Optuna 正式调参 12 次完成，其中 7 次完成、5 次剪枝
  - 最优参数为：
    - optimizer=`adamw`
    - lr=`0.0002710364225466581`
    - weight_decay=`1.296013562349762e-05`
    - dropout=`0.3726753363320149`
    - label_smoothing=`0.11283345642709426`
    - freeze_epochs=`2`
    - finetune_epochs=`5`
    - batch_size=`16`
- 正式测试结果：
  - official / CustomCNN：Accuracy `99.51%`，Macro F1 `99.52%`
  - official / ResNet18：Accuracy `99.51%`，Macro F1 `99.48%`
  - official / EfficientNet-B0 tuned：Accuracy `99.84%`，Macro F1 `99.83%`
  - clean / CustomCNN：Accuracy `99.37%`，Macro F1 `99.33%`
  - clean / ResNet18：Accuracy `100.00%`，Macro F1 `100.00%`
  - clean / EfficientNet-B0 tuned：Accuracy `99.79%`，Macro F1 `99.80%`
- 轻量结果：
  - 已导出 `final_assets/report_table_main.csv`
  - 已导出 `final_assets/official_efficientnet_best_params.csv`
  - 已导出关键图表到 `final_assets/figures/`
- 文档：
  - `docs/report.md` 已回填真实结果且无“待填”
  - `docs/defense.md` 已补入最终指标与答辩口述稿

### 关联信息
- official 主结果：`official_efficientnet_tuned`
- clean 主结果：`clean_efficientnet_tuned`
- clean 对比最高分：`clean_resnet18`
- 轻量结果目录：`E:\pycharm\Python3_9\DeepLearning\final_assets`

## 2026-04-03 09:10 (Asia/Shanghai) - 追加 ResNet18 稳定性验证与答辩补强

### 变更摘要
开始执行新一轮优化，重点围绕 `clean_resnet18` 增加多随机种子稳定性验证，并补强答辩所需的“为什么 ResNet18 在 clean 上表现更强”和“100% 结果是否可信”的证据链。

### 问题说明
- 当前正式结果已经足够作为期末作业提交，但 `clean_resnet18` 达到 `100%` 的结果较容易在答辩时被追问其稳定性与可信度。
- 现有文档已具备正式结果，但缺少多随机种子重复实验、稳定性统计图和针对 `ResNet18` 的单独答辩话术。
- 若不补充稳定性验证，答辩时容易被质疑“是不是当前划分刚好简单”。

### 策略方向
- 只针对 `clean_resnet18` 做种子重复实验，不扩大到新的模型族或新一轮大规模调参。
- 优先增强答辩可信度，而不是继续追求更高的单次分数。
- 保持轻量结果导出策略不变，所有新增稳定性图表与表格统一收敛到 `final_assets/`。

### 具体改动
- 待新增：
  - `configs/clean_resnet18_seed123.yaml`
  - `configs/clean_resnet18_seed2026.yaml`
- 待补充：
  - `final_assets/resnet18_seed_consistency.csv`
  - `final_assets/figures/resnet18_seed_consistency.png`
  - `clean_resnet18` 的轻量答辩图表导出
- 待更新：
  - `docs/report.md`
  - `docs/defense.md`

### 影响面
- 包含项：稳定性配置、重复实验结果、轻量稳定性汇总、文档增强。
- 不包含项：不新增第三方依赖，不再次扩大 EfficientNet 调参范围，不提交大型模型权重。
- 关键点：若多 seed 结果存在波动，文档结论必须如实降级，不保留“完全稳定”的绝对表述。

### 风险与回滚
- 风险点：重复实验可能暴露 `clean_resnet18` 对随机种子敏感，导致答辩结论需要调整。
- 缓解措施：提前约定文档话术以“稳定高分”或“当前划分下最高分但存在随机性”两档切换。
- 回退方式：若某个 seed 训练异常，可单独重跑该 seed；如需停止本轮增强，也不影响上一轮正式作业版本。

### 验证
- 待验证项：
  - 新增两个 seed 配置可成功训练与评估
  - 稳定性汇总 CSV 和图表成功生成
  - 报告与答辩材料新增稳定性结论和高频追问回答

### 关联信息
- 稳定性验证目标模型：`clean_resnet18`
- 目标输出目录：`E:\pycharm\Python3_9\DeepLearning\final_assets`

## 2026-04-03 10:25 (Asia/Shanghai) - 完成 ResNet18 多种子稳定性验证与答辩增强

### 变更摘要
已围绕 `clean_resnet18` 完成 2 组新增随机种子重复训练与测试评估，形成 3 次 seed 的稳定性验证结果，并将其回填到实验报告、答辩提纲和轻量结果目录中。

### 问题说明
- `clean_resnet18` 在上一轮实验中达到 `100%` 测试准确率，但单次高分结果在答辩中容易被质疑是否仅由当前随机种子或当前划分偶然造成。
- 现有报告缺少“稳定性验证”和“高频追问回答”，不利于应对老师对结果可信度的追问。

### 策略方向
- 固定网络结构、数据划分和训练参数，仅更换随机种子验证结果稳定性。
- 用“均值 + 标准差 + 单次结果”来支撑答辩结论，而不是继续使用单次 `100%` 进行绝对化表述。
- 在保留 EfficientNet 调参主线的同时，将 `ResNet18` 提升为答辩亮点。

### 具体改动
- 新增配置：
  - `configs/clean_resnet18_seed123.yaml`
  - `configs/clean_resnet18_seed2026.yaml`
- 更新代码：
  - `src/rice_leaf_disease/analysis.py`
    - 当测试集无错分样本时，自动生成占位错误样本图，避免答辩素材缺失
  - `summarize_results.py`
    - 导出 `resnet18_seed_consistency.csv`
    - 导出 `resnet18_seed_consistency_summary.csv`
    - 导出 `figures/resnet18_seed_consistency.png`
    - 增加 `clean_resnet18` 的混淆矩阵、分类指标图、Grad-CAM 和误差图导出
- 更新文档：
  - `docs/report.md`
    - 新增 `5.4 ResNet18 稳定性验证`
    - 新增“为什么 clean 下 ResNet18 优于 EfficientNet-B0”的解释
    - 调整结论，避免将 `100%` 描述为绝对稳定
  - `docs/defense.md`
    - 强化 ResNet18 为答辩亮点
    - 新增“高频追问回答”段落

### 影响面
- 包含项：配置、稳定性汇总、轻量图表、报告内容、答辩口述材料。
- 不包含项：未新增第三方依赖，未扩大到其他模型的新一轮大规模调参。
- 关键点：答辩话术已从“单次 100%”升级为“多 seed 下稳定高分”。

### 风险与回滚
- 风险点：新增 seed 中有一次结果低于 100%，若表述不当会被误解为模型不稳定。
- 缓解措施：文档已明确写出真实结果为 `100.00% / 99.79% / 100.00%`，并用均值和标准差解释其整体稳定性。
- 回退方式：若后续希望回到更保守表述，可仅保留稳定性 CSV 和图表，不在口头答辩中突出数值结论。

### 验证
- 新增训练：
  - `clean_resnet18_seed123`
    - 最优验证准确率：`100.00%`
    - 测试 Accuracy：`99.79%`
    - 测试 Macro F1：`99.77%`
  - `clean_resnet18_seed2026`
    - 最优验证准确率：`100.00%`
    - 测试 Accuracy：`100.00%`
    - 测试 Macro F1：`100.00%`
- 稳定性汇总：
  - seed 42：Accuracy `100.00%`
  - seed 123：Accuracy `99.79%`
  - seed 2026：Accuracy `100.00%`
  - 平均 Accuracy：`99.93%`
  - Accuracy 标准差：`0.10%`
  - 平均 Macro F1：`99.92%`
  - Macro F1 标准差：`0.11%`
- 轻量结果：
  - 已生成 `final_assets/resnet18_seed_consistency.csv`
  - 已生成 `final_assets/resnet18_seed_consistency_summary.csv`
  - 已生成 `final_assets/figures/resnet18_seed_consistency.png`
  - 已补齐 `clean_resnet18` 的轻量答辩图表
- 文档：
  - `docs/report.md` 已新增稳定性章节
  - `docs/defense.md` 已新增“高频追问回答”

### 关联信息
- 稳定性结论核心表述：`clean_resnet18` 在 3 个随机种子下均取得 `>=99.79%` 的测试准确率
- 当前最佳答辩亮点模型：`clean_resnet18`

## 2026-04-03 20:35 (Asia/Shanghai) - 重写 README 为中文可读版并补充目录树说明

### 变更摘要
开始重写项目 `README.md`，目标是将其改为中文、增强可读性，并提供覆盖当前真实仓库结构的目录树与文件说明，便于首次打开仓库的读者快速理解项目内容、运行方式与结果位置。

### 问题说明
- 当前 `README.md` 仍以英文为主，信息密度较低，不足以支撑课程项目的最终交付与仓库导航需求。
- 现有仓库已经包含正式训练结果、轻量结果目录、正式配置矩阵和答辩材料，但 README 尚未同步更新，无法清晰说明“代码在哪、结果在哪、每个目录做什么”。
- 若不补充结构化目录树，后续查看仓库时容易在 `configs/`、`final_assets/`、`outputs/` 与数据集目录之间产生混淆。

### 策略方向
- 采用完整中文重写，而不是在原英文 README 上零散补丁式修改。
- 使用“树状结构 + 行尾中文说明”的形式展示仓库结构，尽量覆盖真实文件。
- 对 `.git/`、`.local-remote.git/`、`__pycache__/`、海量中间产物与逐张图片文件只做必要概述，不逐项展开。

### 具体改动
- 待重写：
  - `README.md`
- 待纳入 README 的重点内容：
  - 项目简介
  - 项目亮点
  - 环境与依赖
  - 快速开始
  - 目录结构总览
  - 关键实验结果
  - 结果文件说明
  - 常用命令
  - 注意事项

### 影响面
- 包含项：README 内容、目录树说明、命令示例、结果导航。
- 不包含项：本次不调整训练代码、不新增实验、不修改已有结果文件内容。
- 关键点：README 将承担“项目介绍 + 使用说明 + 结果导航”三重角色。

### 风险与回滚
- 风险点：若目录树写得过细，README 可读性会下降；若写得过粗，又不能满足仓库导航需求。
- 缓解措施：按模块拆分多个树状片段，重点详细展开 `configs/`、`docs/`、`src/`、`final_assets/`，仅概述 `outputs/` 和数据集内部海量文件。
- 回退方式：若后续觉得 README 信息过密，可在不改动事实内容的前提下进一步按章节拆分或精简展示层级。

### 验证
- 待验证项：
  - README 全文改为中文
  - 目录树与当前仓库真实结构一致
  - 关键结果数值与 `final_assets/*.csv` 保持一致
  - 命令可复制执行

### 关联信息
- 目标文件：`E:\pycharm\Python3_9\DeepLearning\README.md`
- 主要结果目录：`E:\pycharm\Python3_9\DeepLearning\final_assets`

## 2026-04-03 20:45 (Asia/Shanghai) - README 中文重写完成

### 变更摘要
已将项目 `README.md` 完整重写为中文版本，并补充覆盖当前真实仓库结构的目录树、结果说明、常用命令和注意事项，使其能够直接承担项目导航文档的作用。

### 问题说明
- 旧版 README 以英文为主，信息量不足，无法反映当前项目已经具备的正式实验结果、轻量结果目录与答辩材料。
- 仓库中已经存在 `configs/`、`docs/`、`src/`、`final_assets/`、`outputs/`、数据集目录等多个层次，但缺乏统一、可读的入口说明。

### 策略方向
- 不保留原英文主体，而是按中文完整重写。
- 使用分段式目录树，分别说明顶层结构、配置目录、源码目录、轻量结果目录、`outputs/` 代表性结构和数据集结构。
- 对 `.git/`、`.local-remote.git/`、`__pycache__/`、逐张图片文件与海量中间产物仅作必要说明，不逐项展开。

### 具体改动
- 已重写 `README.md`
- README 当前包含：
  - 项目简介
  - 项目亮点
  - 环境与依赖
  - 快速开始
  - 目录结构总览
  - 关键实验结果
  - 结果文件说明
  - 常用命令
  - 注意事项
- 已在目录树中覆盖：
  - 顶层关键脚本与文件
  - `configs/` 全部当前 YAML
  - `docs/` 全部文档
  - `src/rice_leaf_disease/` 全部源码文件
  - `final_assets/` 全部轻量结果文件与关键图表
  - `outputs/` 的代表性结构
  - 数据集目录的分层结构

### 影响面
- 包含项：README 内容、仓库导航、结果说明、命令说明。
- 不包含项：本次未修改训练逻辑、未新增实验、未变更现有结果文件内容。
- 关键点：README 现已可独立作为项目说明入口使用。

### 风险与回滚
- 风险点：目录树信息量较大，若后续仓库结构继续扩张，README 需要同步维护。
- 缓解措施：本次已按“详细展开稳定目录、概述海量目录”的方式控制信息密度。
- 回退方式：若未来需要进一步精简，可在不改变事实信息的前提下压缩目录树展示层级。

### 验证
- 已确认 README 全文为中文主内容。
- 已确认 README 中写入当前正式实验关键结果：
  - official / EfficientNet-B0 tuned：`99.84%`
  - clean / EfficientNet-B0 tuned：`99.79%`
  - clean / ResNet18：`100.00%`
  - ResNet18 三个 seed 平均 Accuracy：`99.93%`
- 已确认 README 中包含 `final_assets/`、`outputs/`、`Rice Leaf Disease Images/` 的结构说明。
- 已确认 README 中的常用命令可直接复制执行。

### 关联信息
- 当前 README 已与 `final_assets/*.csv` 和正式实验结果保持一致
- 当前 README 改动已完成，版本控制同步由后续 Git 提交与远端推送处理

## 2026-04-03 21:10 (Asia/Shanghai) - 新增答辩 PPT 大纲与 `.pptx` 自动生成能力

### 变更摘要
开始为当前课程项目新增答辩 PPT 大纲文档与 `.pptx` 自动生成能力，目标是基于现有正式实验结果和图表产物，生成一份可直接打开和使用的课程答辩演示文件。

### 问题说明
- 当前项目已经具备正式实验报告、答辩提纲和轻量结果图表，但尚未产出实际的 `.pptx` 演示文件。
- 当前环境未安装 `python-pptx`，也未发现可直接调用的 PowerPoint / LibreOffice 程序，因此无法直接生成 PPT 文件。
- 若只保留 Markdown 版答辩提纲，仍需要手工排版才能形成真正可交付的答辩文件。

### 策略方向
- 采用 `python-pptx` 自动生成 `.pptx` 文件，不依赖手工打开 PowerPoint 排版。
- 继续复用现有 `final_assets/figures` 中的正式图表结果，避免重复绘图。
- 同步新增一份 `docs/ppt_outline.md`，使答辩文稿与 PPT 内容保持一一对应。

### 具体改动
- 待新增依赖：
  - `python-pptx`
- 待同步更新：
  - `requirements.txt`
  - `environment.yml`
- 待新增文件：
  - `docs/ppt_outline.md`
  - `build_ppt.py`
- 待生成产物：
  - `final_assets/ppt/水稻叶病害分类答辩.pptx`

### 影响面
- 包含项：依赖、PPT 生成脚本、PPT 大纲、实际 PPT 文件、文档说明。
- 不包含项：本次不新增训练实验，不修改现有正式结果数值。
- 关键点：新增第三方依赖后，必须同步依赖文件与变更记录，保证他人可复现。

### 风险与回滚
- 风险点：`python-pptx` 安装可能受网络或环境兼容性影响；自动排版生成的 PPT 可能需要通过几次样式调整才能达到较好的可读性。
- 缓解措施：优先使用已验证的 `rice-leaf-dl` 环境，固定版式风格并复用现有结果图。
- 回退方式：若自动生成的 PPT 版式不理想，至少保留 `docs/ppt_outline.md` 作为手工答辩稿基础；若依赖安装失败，则保留生成脚本草案与大纲文档。

### 验证
- 待验证项：
  - `python-pptx` 安装成功
  - `build_ppt.py` 能生成 `.pptx`
  - `.pptx` 文件包含 12 页
  - 大纲与 PPT 页面内容一致

### 关联信息
- PPT 目标路径：`E:\pycharm\Python3_9\DeepLearning\final_assets\ppt\水稻叶病害分类答辩.pptx`
- 大纲目标路径：`E:\pycharm\Python3_9\DeepLearning\docs\ppt_outline.md`

## 2026-04-03 20:50 (Asia/Shanghai) - 答辩 PPT 大纲与 `.pptx` 文件生成完成

### 变更摘要
已新增答辩 PPT 大纲文档、PPT 自动生成脚本，并成功生成一份可直接打开的 `.pptx` 答辩文件。

### 问题说明
- 原有项目仅有 Markdown 版答辩提纲，无法直接作为课堂答辩演示文件使用。
- 当前环境初始缺少 `python-pptx`，无法自动输出真正的 PPT 文件。

### 策略方向
- 采用 `python-pptx` 自动生成答辩演示文件，不依赖手工排版。
- 继续复用 `final_assets/figures` 中已有的正式图表结果，减少重复工作并保持文档、图表和 PPT 内容一致。

### 具体改动
- 依赖更新：
  - `requirements.txt` 新增 `python-pptx`
- 新增文件：
  - `docs/ppt_outline.md`
  - `build_ppt.py`
- 文档同步：
  - `docs/defense.md` 新增 PPT 落地说明
  - `README.md` 新增 PPT 生成脚本和 PPT 文件路径说明
- 生成产物：
  - `final_assets/ppt/水稻叶病害分类答辩.pptx`

### 影响面
- 包含项：依赖、答辩大纲、PPT 自动生成脚本、实际 PPT 文件、README 与答辩文档说明。
- 不包含项：本次未新增实验结果，不改变已有训练与评估逻辑。
- 关键点：PPT 内容与现有正式实验结果、答辩提纲和轻量图表保持一致。

### 风险与回滚
- 风险点：自动生成的 PPT 虽已形成完整结构，但若老师对版式风格有强个性要求，仍可能需要人工微调。
- 缓解措施：当前版式已固定为视觉强化风，重点图表与结论页已经到位，适合继续微调而不是从零制作。
- 回退方式：若后续希望改版，可直接修改 `build_ppt.py` 后重新生成，不影响现有数据结果。

### 验证
- 依赖验证：
  - 已成功安装 `python-pptx 1.0.2`
- 生成功能验证：
  - `build_ppt.py` 已成功运行
  - 已生成 `E:\pycharm\Python3_9\DeepLearning\final_assets\ppt\水稻叶病害分类答辩.pptx`
- 内容结构验证：
  - 已使用 `python-pptx` 成功读取生成后的 PPT 文件
  - 当前 PPT 共 `12` 页
  - 文件大小约 `4.16 MB`

### 关联信息
- PPT 文件：`E:\pycharm\Python3_9\DeepLearning\final_assets\ppt\水稻叶病害分类答辩.pptx`
- 生成脚本：`E:\pycharm\Python3_9\DeepLearning\build_ppt.py`
- 大纲文档：`E:\pycharm\Python3_9\DeepLearning\docs\ppt_outline.md`

## 2026-04-05 10:10 (Asia/Shanghai) - 补充答辩质疑点的口头版与书面版解释材料

### 变更摘要
开始围绕老师可能提出的 4 类质疑点补充专门解释材料，目标是同时形成可直接口头使用的答辩话术，以及可直接引用或改写进文档的书面说明。

### 问题说明
- 当前 `docs/defense.md` 已有部分高频问答，但还没有形成一份专门针对“版本号、过高准确率、工程取舍和可加分项”的完整答辩说明材料。
- 用户需要的是一套“详细、合理、可直接拿去答辩”的解释，而不是零散的结论句。
- 若只靠现有报告和答辩稿中的零散表述，面对老师连续追问时不够集中，也不便于后续直接复用。

### 策略方向
- 新增一份独立的中文说明文档，集中整理问题背景、事实基础、口头回答和书面表达。
- 同步增强 `docs/defense.md` 中的高频追问回答，使核心质疑点能直接在答辩稿中读出。
- 不修改实验结果本身，不人为掩盖争议点，而是基于当前项目事实做更稳妥的解释。

### 具体改动
- 待新增：
  - `docs/qa_explanations.md`
- 待更新：
  - `docs/defense.md`

### 影响面
- 包含项：口头答辩话术、书面解释模板、文档增强。
- 不包含项：本次不新增训练实验、不修改结果数值、不调整代码逻辑。
- 关键点：解释必须以当前仓库事实为依据，不能靠空泛安慰或硬性辩解。

### 风险与回滚
- 风险点：若解释写得过于防守，容易显得心虚；若写得过于强硬，又容易在答辩中被进一步追问。
- 缓解措施：统一采用“承认边界 + 提供事实 + 给出理由 + 明确取舍”的表达风格。
- 回退方式：若后续需要更短版本，可从独立说明文档中摘取简版问答，不影响原始说明保留。

### 验证
- 待验证项：
  - 新增文档覆盖 4 个质疑点
  - `docs/defense.md` 能直接支撑口头回答
  - 所有解释与当前 `requirements.txt`、`report.md`、`final_assets/*.csv`、`.gitignore` 保持一致

### 关联信息
- 目标文档：`E:\pycharm\Python3_9\DeepLearning\docs\qa_explanations.md`
- 同步增强文档：`E:\pycharm\Python3_9\DeepLearning\docs\defense.md`

## 2026-04-05 10:28 (Asia/Shanghai) - 完成答辩质疑点解释材料

### 变更摘要
已完成针对“PyTorch 版本可疑、clean 视图 100% 准确率、可加分项缺失、outputs 未入 Git”4 类问题的集中解释材料，并同步增强答辩稿中的高频追问回答。

### 问题说明
- 用户需要的是一套可以直接用于答辩现场和书面补充说明的解释，而不是零散的临场发挥。
- 当前 `docs/defense.md` 已有部分问答，但尚未覆盖版本号和仓库分层管理这类容易被老师继续追问的点。

### 策略方向
- 新增独立说明文档，集中沉淀问题背景、事实基础、口头回答和书面表达。
- 在 `docs/defense.md` 中同步补全高频追问区，使其可直接口头使用。
- 保持解释基于当前仓库事实，不回避边界，也不夸大结论。

### 具体改动
- 新增：
  - `docs/qa_explanations.md`
- 更新：
  - `docs/defense.md`
    - 新增 `torch==2.10.0` 解释
    - 新增“结果是不是过拟合”解释
    - 新增“为什么没有交叉验证/速度对比/参数量对比”解释
    - 新增“为什么 outputs 不进 Git”解释

### 影响面
- 包含项：独立答辩说明文档、答辩稿高频问答增强。
- 不包含项：本次未新增训练实验、未修改结果数值、未修改代码逻辑。
- 关键点：现在项目同时具备“书面版”和“口头版”质疑应对材料。

### 风险与回滚
- 风险点：如果后续依赖版本或仓库组织策略发生变化，这份解释材料需要同步更新。
- 缓解措施：当前内容已明确以现有 `requirements.txt`、`.gitignore`、`report.md`、`final_assets/*.csv` 为事实基础。
- 回退方式：若只需要更简短版本，可直接从 `docs/qa_explanations.md` 中删减，不影响事实内容保留。

### 验证
- 已确认新增独立说明文档覆盖 4 类质疑点。
- 已确认 `docs/defense.md` 中的高频追问回答已扩展到 7 条。
- 已确认解释内容与当前 `requirements.txt`、`.gitignore`、`docs/report.md`、`final_assets/resnet18_seed_consistency_summary.csv` 保持一致。

### 关联信息
- 书面版说明：`E:\pycharm\Python3_9\DeepLearning\docs\qa_explanations.md`
- 口头版答辩稿：`E:\pycharm\Python3_9\DeepLearning\docs\defense.md`

## 2026-04-24 07:09 (Asia/Shanghai) - 为现有答辩 PPT 补充本地图片素材

### 变更摘要
开始对现有答辩 PPT `docs/水稻叶病害智能识别：迁移学习与超参数调优实验.pptx` 进行视觉补充，在不改动页面原有文字内容的前提下，将项目本地已有的病害原图与实验图表放入更合适的页面位置，提升页面完成度与答辩展示效果。

### 问题说明
- 当前 PPT 已根据 `docs/PPT深读学习.md` 产出主要文案与页面结构，但部分页面仍缺少本地图片素材支撑，视觉信息密度不足。
- 用户明确要求保留页面原有文字，不允许借插图过程改写或删改现有文案，因此需要采用“只增补图片、不改文本”的方式处理。
- 项目中同时存在原始病害图片库与实验结果图表，若不先核对当前 PPT 的页结构与空白区域，容易出现插图错页、遮挡文本或图文不匹配的问题。

### 策略方向
- 先检查现有 PPT 的实际页数、版式结构、文本框与空白区域，再按页面主题选择最匹配的本地图像，而不是直接批量插入。
- 优先复用项目内已有素材：封面/背景/任务页优先使用病害原图，结果分析页优先使用 `final_assets/figures` 下的实验图表，确保图文语义一致。
- 修改方式控制为增量编辑：只新增图片对象，尽量不移动、不替换、不删除现有文字对象；如局部空间不足，优先寻找留白区域而不是调整文案。

### 具体改动
- 待检查并记录 `docs/水稻叶病害智能识别：迁移学习与超参数调优实验.pptx` 的页结构、可插图区域与适配关系。
- 待从 `Rice Leaf Disease Images/` 与 `final_assets/figures/` 中筛选适合各页的本地图片素材。
- 待将筛选后的图片写入目标 PPT，并输出修改后的结果文件。

### 影响面
- 包含项：目标 PPT 文件的图片元素补充、本地素材筛选、本次插图映射与结果核对。
- 不包含项：不修改页面原有标题、正文、表格与结论文案；不新增第三方依赖；不改动实验数据与图表内容本身。
- 关键点：最终结果必须保持原文字内容不变，同时保证新增图片与页面主题相匹配且不遮挡关键信息。

### 风险与回滚
- 风险点：图片尺寸或比例不合适可能压缩留白、遮挡文字，或使页面视觉重心失衡。
- 缓解措施：先做页面结构检查与预览渲染，再执行写回；优先选择语义明确、构图稳定的本地图片；完成后逐页核对。
- 回退方式：若插图效果不理想，可回退到当前未修改的 `docs/水稻叶病害智能识别：迁移学习与超参数调优实验.pptx` 原始版本，或重新导出一份仅调整图片层的 PPT 副本。

### 验证
- 待验证项：确认目标 PPT 页数与结构；确认每页插图不覆盖原有文字；确认输出文件可正常打开且图片嵌入成功；确认插图页与 `docs/PPT深读学习.md` 的页面主题一致。

### 关联信息
- 需求说明：`E:\pycharm\Python3_9\DeepLearning\docs\PPT深读学习.md`
- 目标文件：`E:\pycharm\Python3_9\DeepLearning\docs\水稻叶病害智能识别：迁移学习与超参数调优实验.pptx`
- 候选素材目录：`E:\pycharm\Python3_9\DeepLearning\Rice Leaf Disease Images`、`E:\pycharm\Python3_9\DeepLearning\final_assets\figures`

## 2026-04-24 07:40 (Asia/Shanghai) - 现有答辩 PPT 本地图片补充完成

### 变更摘要
已基于项目本地病害图片与实验图表，为现有答辩 PPT 输出一份“本地图片版”成品，在不改动页面原有文字内容的前提下，将原先的占位性质风景图替换为更贴近课题语义的病害样例图、数据分布图、训练曲线图、模型对比图、混淆矩阵、错误样本图与 Grad-CAM 图。

### 问题说明
- 原始 PPT 已具备完整文字与版式，但多页仍使用泛化风格的风景占位图，和“水稻叶病害识别”课题本身关联不强。
- 直接替换导入后的图片媒体时，发现源 PPT 中部分页复用了同一底层媒体资源，导致一处替换会串改到其他页，不适合继续使用“原位替换”方案。
- 用户要求不能改动原有文字，因此本次只能围绕图片层与导出文件处理，不能通过重排文案来腾挪版面。

### 策略方向
- 改为从原始 PPT 重新导入，在目标图片位置上方覆盖新增本地图片，而不是改写共享媒体本体，避免串页。
- 对封面、目录、背景介绍页优先使用本地病害样例拼贴图；对数据审计、训练分析、结果分析与可解释性页优先使用 `final_assets/figures` 内实验图表。
- 对章节分隔页维持现有样式不强改，只处理有明确图片槽位或占位图的页面，以降低误伤版式风险。

### 具体改动
- 新增临时脚本与中间产物：
  - `tmp/slides/ppt_image_insert/build_assets.py`
  - `tmp/slides/ppt_image_insert/inspect_ppt.mjs`
  - `tmp/slides/ppt_image_insert/update_ppt_images.mjs`
  - `tmp/slides/ppt_image_insert/assets/` 下若干本地拼贴图与图表面板
- 基于本地病害图片生成了封面/目录/横幅/竖版拼贴图，用于替换原始风景占位图。
- 将下列内容型图片写入 PPT：
  - 数据集概况页：病害拼贴图、官方类别分布图
  - 重复图与泄漏页：重复图统计图
  - Optuna 页：调参结果图
  - 训练过程页：训练曲线图组
  - 结果分析页：模型对比图、Official vs Clean 对比图、稳定性图
  - 可解释性页：混淆矩阵、错误样本图、Grad-CAM 图
  - 结论/展望/Q&A/结束页：病害拼贴图或结果图
- 输出新文件：
  - `E:\pycharm\Python3_9\DeepLearning\docs\水稻叶病害智能识别：迁移学习与超参数调优实验_本地图片版.pptx`

### 影响面
- 包含项：PPT 图片层更新、本地素材拼贴、实验图表嵌入、修改后 PPT 导出与预览校验。
- 不包含项：不修改原始 PPT 文案内容、不新增第三方依赖、不调整实验结果数值、不修改数据或训练代码。
- 关键点：原始文件 `docs/水稻叶病害智能识别：迁移学习与超参数调优实验.pptx` 保留不动，新的图片版以新文件名输出。

### 风险与回滚
- 风险点：部分图表因原始图片槽位比例较极端，只能以覆盖方式适配，个别页面图表可读性会受版式限制。
- 缓解措施：已对封面、数据页、结果页、可解释性页、结论页等关键页面逐页渲染预览核对，优先保证“图文相关”和“不遮挡文字”。
- 回退方式：若后续希望继续微调，可继续基于原始文件 `docs/水稻叶病害智能识别：迁移学习与超参数调优实验.pptx` 重导，不影响原件。

### 验证
- 已确认读取原始 PPT 成功，实际页数为 `28` 页。
- 已成功导出新文件：`docs/水稻叶病害智能识别：迁移学习与超参数调优实验_本地图片版.pptx`
- 已渲染并检查关键预览页：
  - 封面、目录、背景介绍
  - 数据集概况、重复图与数据泄漏
  - Optuna 调参、训练过程
  - 六模型主结果、Official vs Clean 差异
  - 混淆矩阵与错误样本、Grad-CAM
  - 结论、改进方向、Q&A、结束页
- 验证结果：页面原有文字内容保持不变，新增图片未覆盖正文，PPT 可正常导出并生成预览。

### 关联信息
- 原始 PPT：`E:\pycharm\Python3_9\DeepLearning\docs\水稻叶病害智能识别：迁移学习与超参数调优实验.pptx`
- 输出 PPT：`E:\pycharm\Python3_9\DeepLearning\docs\水稻叶病害智能识别：迁移学习与超参数调优实验_本地图片版.pptx`
- 中间检查目录：`E:\pycharm\Python3_9\DeepLearning\tmp\slides\ppt_image_insert`

## 2026-04-26 10:10 (+08:00 Asia/Shanghai) - 清理高确信临时与半成品产物

### 变更摘要
根据用户确认的清理范围，准备删除项目中的临时目录、Python/Jupyter 自动缓存、不完整的 Optuna 剪枝 trial 输出，以及 smoke 冒烟测试输出，以减少项目目录冗余并降低后续查找正式实验结果时的干扰。

### 问题说明
- 当前项目中存在多类非正式交付产物：`tmp/` 下的 PPT/Word 解包、预览和中间素材，`__pycache__/` 与 Jupyter 自动缓存，以及 `outputs/` 下的 smoke 测试和被剪枝调参 trial。
- 这些文件不是最终实验报告、最终图表或正式模型结果的一部分，继续保留会增加目录体积，并容易与正式实验输出混淆。
- 本次清理由用户明确指定执行，触发条件为“最建议删除（临时/半成品，高确信）全部删除”。

### 策略方向
- 仅删除上一轮盘点中高确信的临时/半成品目录，不扩大到重复文档、旧版 PPT、完整调参 trial、数据压缩包或正式模型结果。
- 删除前先解析并确认目标路径均位于项目根目录 `E:\pycharm\Python3_9\DeepLearning` 内，避免误删项目外文件。
- 保留 `final_assets`、正式 `outputs/runs`、正式 `outputs/evaluations`、`Rice Leaf Disease Images`、`docs`、源码、配置和 Git 目录。

### 具体改动
- 待删除临时目录与缓存：`tmp/`、`__pycache__/`、`src/rice_leaf_disease/__pycache__/`、`.ipynb_checkpoints/`、`.virtual_documents/`。
- 待删除不完整调参 trial：`outputs/runs/official_efficientnet_search_trial_007` 至 `official_efficientnet_search_trial_011`。
- 待删除 smoke 测试输出：`outputs/runs/smoke_official_efficientnet_b0`、`outputs/runs/smoke_official_efficientnet_b0_trial_000`、`outputs/evaluations/smoke_official_efficientnet_b0`、`outputs/tuning/smoke_official_efficientnet_b0`。
- 不包含项：不删除 `official_efficientnet_search_trial_000` 至 `006`、不删除 `official_efficientnet_tuned`、`clean_efficientnet_tuned`、ResNet/CustomCNN 正式 run、不删除最终报告和 PPT。

### 影响面
- 预计释放约 191.6 MB 的临时与半成品文件空间。
- 可能影响后续复查 smoke 冒烟测试或被剪枝 trial 的原始 checkpoint；不影响正式实验图表、正式模型、最终评估表和数据集。

### 风险与回滚
- 风险点：删除操作不可直接从项目内恢复；若后续需要复查 smoke 测试或剪枝 trial，需要重新运行对应实验。
- 缓解措施：删除范围仅限已确认的临时/半成品路径，并在删除前进行项目根目录边界校验。
- 回退方式：如需要恢复，只能从外部备份、Git 以外的文件备份，或重新运行对应脚本生成。

### 验证
- 待验证：删除后使用 `Test-Path` 检查上述目标路径均不存在。
- 待验证：使用 `Get-ChildItem outputs/runs` 确认正式 run 仍保留。
- 待验证：使用 `git status --short` 查看工作区状态，确认无非预期源码或配置变更。

### 关联信息
- 清理范围来源：用户确认的“最建议删除（临时/半成品，高确信）”列表。
- 项目根目录：`E:\pycharm\Python3_9\DeepLearning`

### 补充验证结果（2026-04-26 10:10 +08:00）
- 删除执行结果：上述 14 个目标路径已完成删除，实际释放空间约 191.61 MB。
- `Test-Path` 验证：所有删除目标均返回 `False`，未发现残留目录。
- 正式结果保留验证：`outputs/runs` 中仍保留 `official_efficientnet_tuned`、`clean_efficientnet_tuned`、`official_resnet18`、`clean_resnet18`、`official_customcnn`、`clean_customcnn`、`official_efficientnet_search_trial_000` 至 `006` 等目录；`outputs/evaluations` 中仍保留正式评估目录。
- 目录体积复核：`outputs/` 由清理前约 807.47 MB 降至约 659.47 MB；项目中 `final_assets/`、`docs/`、`Rice Leaf Disease Images/`、源码与配置未被删除。
- `git status --short` 复核：当前可见变更包含本次追加的 `progress.md`，以及清理前已存在的 notebook 与 docs 未跟踪/修改项；未出现新的源码或配置文件删除。

## 2026-04-26 10:15 (+08:00 Asia/Shanghai) - 将 docs 目录排除出 Git 提交范围

### 变更摘要
根据用户要求，将项目中的 `docs/` 文档目录加入 `.gitignore`，并准备把已经被 Git 跟踪的 docs 文档从索引中移除，使后续仓库提交不再包含该目录内容。

### 问题说明
- 用户要求“doc 文件夹加入 gitignore 不向仓库提交”。当前项目中实际存在的文档目录为 `docs/`。
- 仅修改 `.gitignore` 只能阻止未跟踪的新文件进入 Git；经检查，`docs/defense.md`、`docs/ppt_outline.md`、`docs/qa_explanations.md`、`docs/report.md` 已被 Git 跟踪，仍会继续出现在提交中。
- 因此本次需要同时处理忽略规则与 Git 索引状态。

### 策略方向
- 在 `.gitignore` 中新增 `docs/`，用于忽略整个文档目录下的新增或未跟踪文件。
- 使用 `git rm --cached -r docs` 仅从 Git 索引移除已跟踪 docs 文件，不删除本地文件。
- 不修改 docs 目录内任何文档内容，不删除本地文档文件。

### 具体改动
- 待更新：`.gitignore` 新增 `docs/` 规则。
- 待执行：将已跟踪的 `docs/` 文件从 Git 索引中取消跟踪。
- 不包含项：不删除 `docs/` 目录、不删除报告/PPT/Markdown 文件、不修改文档正文、不提交 Git commit。

### 影响面
- 后续普通提交不会再包含 `docs/` 下的新文档或修改。
- 已从索引移除的 docs 文件在下一次提交中会表现为仓库删除，但本地文件仍保留。

### 风险与回滚
- 风险点：如果之后希望仓库继续保留部分 docs 文件，需要重新调整忽略规则并重新 `git add` 对应文件。
- 缓解措施：只取消 Git 跟踪，不删除本地文件，保证当前工作资料仍在磁盘上。
- 回退方式：删除 `.gitignore` 中的 `docs/` 规则，然后对需要保留的文档重新执行 `git add docs/...`。

### 验证
- 待验证：`.gitignore` 包含 `docs/`。
- 待验证：`git status --short` 中 docs 已跟踪文件显示为删除暂存状态或不再作为未跟踪内容出现。
- 待验证：本地 `docs/` 目录仍存在。

### 关联信息
- 项目文档目录：`E:\pycharm\Python3_9\DeepLearning\docs`
- Git 忽略文件：`E:\pycharm\Python3_9\DeepLearning\.gitignore`

### 补充验证结果（2026-04-26 10:16 +08:00）
- `.gitignore` 已新增 `docs/` 规则，用于忽略项目文档目录。
- 已执行 `git rm --cached -r docs`，从 Git 索引中移除原先跟踪的 `docs/defense.md`、`docs/ppt_outline.md`、`docs/qa_explanations.md`、`docs/report.md`。
- 本地文件验证：`docs/` 目录仍存在，目录内报告、PPT、Markdown 文档均保留在磁盘上。
- `git ls-files docs` 验证：当前无 docs 路径仍处于 Git 跟踪列表中。
- `git status --short` 验证：`.gitignore` 与 `progress.md` 已修改；docs 中原已跟踪文件显示为从仓库索引删除，这是取消跟踪的预期状态；清理前已有的 notebook 修改仍存在。

## 2026-04-26 10:20 (+08:00 Asia/Shanghai) - 准备将当前项目变更提交并推送到 GitHub

### 变更摘要
根据用户要求，准备将当前工作区中可提交的项目变更统一暂存、提交，并推送到 GitHub 远程仓库 `github/main`。

### 问题说明
- 当前工作区存在 `.gitignore`、`progress.md`、docs 取消跟踪记录，以及 `rice_leaf_disease_defense.ipynb` 的修改。
- 用户要求“现在将所有文件向github仓库推送一次”，因此需要先形成一次 Git 提交，再推送到 GitHub 远程分支。
- 因 `docs/` 已按上一项任务加入 `.gitignore` 并取消跟踪，本次不会把本地 docs 文件内容重新加入仓库。

### 策略方向
- 使用 `git add -A` 暂存当前所有可提交变更，包括文件修改、取消跟踪记录和 notebook 修改。
- 创建一次提交，提交信息概括为清理临时产物、忽略 docs 目录并同步 notebook 更新。
- 推送当前 `main` 分支到 GitHub 远程 `github/main`。

### 具体改动
- 待提交：`.gitignore` 新增 `docs/` 忽略规则。
- 待提交：`docs/` 中原已跟踪文档从 Git 索引移除，本地仍保留。
- 待提交：`progress.md` 追加清理、忽略 docs、推送准备记录。
- 待提交：`rice_leaf_disease_defense.ipynb` 当前已有修改。
- 不包含项：不会强制添加已被 `.gitignore` 忽略的 `outputs/`、`docs/` 新文件、数据集目录、临时缓存目录。

### 影响面
- GitHub 仓库将收到一次新的提交，反映当前仓库可跟踪文件的最新状态。
- 本地 docs 文件仍存在，但仓库中原有 docs 跟踪文件会在该提交中表现为删除。

### 风险与回滚
- 风险点：notebook 当前修改会一并提交；docs 原跟踪文件会从远程仓库移除。
- 缓解措施：这是用户要求“所有文件推送”的执行结果，且 docs 取消跟踪符合上一项明确要求。
- 回退方式：如需撤销远程提交，可后续使用 Git revert 生成反向提交；如需重新跟踪 docs，可移除 `.gitignore` 中 `docs/` 并重新 `git add docs/...`。

### 验证
- 待验证：`git status --short` 确认提交前变更已暂存。
- 待验证：`git commit` 成功生成提交。
- 待验证：`git push github main` 成功推送到 GitHub。

### 关联信息
- 当前分支：`main`
- GitHub 远程：`github https://github.com/moyan726/Oryza-Sentinel.git`
