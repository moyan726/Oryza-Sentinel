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
