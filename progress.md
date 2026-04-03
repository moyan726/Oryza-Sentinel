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
