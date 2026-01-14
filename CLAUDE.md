# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch深度学习项目模板，专注于情感语音识别任务。采用配置驱动、组件注册、单一入口设计。

## CRITICAL: Paper Reproduction Principle

**复现论文时，优先忠于原文，严格按照原文设置进行实验。**

- **不要擅自"优化"或"改进"**: 即使某些设置看起来不合理、过时或次优，也必须严格遵循原文
- **为什么重要**: 只有复现出与原文一致的结果，才能验证实现的正确性。一旦偏离原文设置，结果无法对比，复现就失去了意义
- **改进在后**: 只有在成功复现出原文结果后，才能进行消融实验或改进尝试

常见需要严格遵循的设置：
- 模型架构细节（层数、隐藏维度、激活函数）
- 数据预处理方式（采样率、归一化方法、数据增强）
- 训练超参数（学习率、batch size、optimizer、scheduler、epochs）
- 数据集划分方式（fold划分、train/val/test比例）
- 评估指标和计算方式

**记录原文设置**: 在 `papers/implementations/` 中详细记录原文的所有关键设置，确保实现完全一致。

## Environment Setup

**Package Manager**: UV only (不使用 pip)

```bash
# 安装依赖
uv sync

# 运行脚本
uv run python src/train.py --config configs/xxx.yaml
```

**Python Version**: >=3.12 (见 pyproject.toml)

**镜像源**: Tsinghua PyPI mirror (已在 pyproject.toml 配置)

## Core Architecture

### Single Entry Point Design

**关键原则**: train.py 是唯一入口，包含训练、验证、测试的完整流程

- **不需要** eval.py、infer.py、test.py 等独立文件
- 通过 `--mode` 参数切换执行模式（train/val/test）
- 训练模式包含完整的 train → val → test 流程

```bash
# 训练（包含验证和测试）
uv run python src/train.py --config configs/baseline.yaml

# 仅测试
uv run python src/train.py --config configs/baseline.yaml --mode test --checkpoint runs/xxx/best_model.pth

# 仅验证
uv run python src/train.py --config configs/baseline.yaml --mode val --checkpoint runs/xxx/best_model.pth
```

### Registry Pattern

组件注册机制用于模型、损失函数、数据集、指标的动态装配：

```python
from src.utils.registry import Registry

MODEL_REGISTRY = Registry("model")

@MODEL_REGISTRY.register("ResNet50")
class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # ...

# 从配置文件中动态创建
model = MODEL_REGISTRY.get(config.model.name)(**config.model.params)
```

**注册机制涵盖**:
- `src/models/` - MODEL_REGISTRY
- `src/losses/` - LOSS_REGISTRY
- `src/data/` - DATASET_REGISTRY
- `src/metrics/` - METRIC_REGISTRY

### Configuration System

YAML配置 + 命令行覆盖：

```bash
# 使用配置文件并覆盖特定参数
uv run python src/train.py \
    --config configs/baseline.yaml \
    train.epochs=50 \
    model.params.num_classes=100
```

配置文件结构示例：
```yaml
model:
  name: ResNet50
  params:
    num_classes: 10

data:
  root: ./data/processed
  batch_size: 32

train:
  epochs: 100
  learning_rate: 0.001
  seed: 42
```

## Code Standards

**300行限制**: 单个文件不超过 300 行代码

- 超过限制时拆分到独立模块（如 `src/trainers/`, `src/evaluators/`）
- train.py 复杂时，将训练循环逻辑抽取到 `src/trainers/trainer.py`

**命名规范**:
- Python: `snake_case` for functions/variables, `PascalCase` for classes
- 最大行长：100 字符

**类型注解**: 必须使用类型注解和文档字符串

## Project Structure

```
deep_learning_template/
├── configs/              # YAML配置文件
├── data/                 # 数据存放区
│   └── labels/          # 情感语音数据集标注（IEMOCAP, MSP-PODCAST, CCSEMO）
├── docs/                # 项目文档、实验报告、技术总结
├── papers/              # 论文PDF与解析笔记
├── pretrained_model/    # 预训练权重（通过下载脚本获取，使用全局缓存+软链接）
├── runs/                # 实验结果（自动生成，不提交Git）
├── scripts/             # 工具脚本
│   ├── download_models_from_hf.py    # 从 Hugging Face 下载模型
│   └── download_models_from_ms.py    # 从 ModelScope 下载模型
└── src/                 # 核心代码
    ├── train.py         # 唯一入口（训练/验证/测试）
    ├── models/          # 模型定义
    ├── data/            # 数据集、transforms、loader
    ├── losses/          # 损失函数
    ├── metrics/         # 评估指标
    └── utils/           # 工具模块（registry, config, logger, checkpoint等）
```

## Dataset Labels

位于 `data/labels/`，包含三个情感语音数据集：

- **IEMOCAP**: 10,039 样本，英文，VAD维度，5折交叉验证（按session）
- **MSP-PODCAST**: 19,898 样本，英文，VAD维度，官方train/test划分
- **CCSEMO**: 7,554 样本，中文，VA维度，5折交叉验证（按说话人，103人）

字段结构：
```python
# IEMOCAP
audio_path,session,dialog,name,start_time,end_time,discrete_emotion,V,A,D,transcript

# MSP-PODCAST
name,audio_path,discrete_emotion,A,V,D,spkid,gender,split_set,duration,transcript

# CCSEMO
audio_path,name,V,A,gender,duration,discrete_emotion,split_set,transcript
```

**重新生成5折划分**:
```bash
cd data/labels
python3 generate_folds.py
```

## Model Downloads

**全局缓存 + 软链接策略**: 模型下载到 `/mnt/shareEEx/liuyang/resources/pretrained_model`，然后软链接到项目 `pretrained_model/`

```bash
# Hugging Face 下载
uv run python scripts/download_models_from_hf.py --model all
uv run python scripts/download_models_from_hf.py -m wavlm-large-finetuned-iemocap
uv run python scripts/download_models_from_hf.py --repo-id facebook/hubert-large-ll60k --name hubert_large

# ModelScope 下载
uv run python scripts/download_models_from_ms.py --model wavlm-large
```

**环境变量** (可选，用于加速 HF 下载):
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=your_token_here
```

## Experiment Results

每次实验必须保存 6 类文件到 `runs/exp_xxx/`:

1. **config.yaml** - 完整配置（用于复现）
2. **train.log** - 训练日志（每epoch的loss/metrics）
3. **best_results.txt** - 最佳性能指标摘要
4. **test_prediction.csv** - 测试集预测结果（sample_id, true_label, predicted_label, confidence）
5. **损失曲线** - loss_curve.png, metrics_curve.png
6. **best_model.pth** - 最佳模型权重

实验命名规范: `exp_{dataset}_{model}_{date}_{time}`

示例:
```
runs/
└── exp_iemocap_wavlm_2025-01-13_14-30/
    ├── config.yaml
    ├── train.log
    ├── best_results.txt
    ├── test_prediction.csv
    ├── loss_curve.png
    ├── metrics_curve.png
    └── best_model.pth
```

## Documentation Management

**重要规则**: 除了项目根目录的 `README.md` 和 `CLAUDE.md`，其他所有 Markdown 文档必须放在 `docs/` 目录下。

### 文档分类

```
docs/
├── setup/           # 环境配置、安装指南
├── experiments/     # 实验记录、结果分析
├── models/          # 模型架构、设计文档
├── tutorials/       # 使用教程、最佳实践
└── references/      # 参考资料、论文笔记
```

### 编写规范（必须遵循）

**文件命名**:
- 使用小写字母和连字符：`model-comparison.md`
- 使用描述性名称：`bert-fine-tuning-guide.md`
- 日期前缀（可选）：`2025-01-13-experiment-report.md`

**内容规范**:
- 使用清晰的标题层级（# ## ###）
- 代码块使用三个反引号标注语言类型
- 避免使用过多装饰性符号和 emoji
- 保持简洁专业的风格

**常见文档类型**:
- **实验报告**: 记录实验配置、结果、分析、结论
- **技术总结**: 问题描述、解决方案、实现细节
- **使用指南**: 功能介绍、使用步骤、参数说明、示例代码

## Development Workflow

### Adding a New Model

1. 在 `src/models/` 创建新文件（如 `my_model.py`）
2. 使用 `@MODEL_REGISTRY.register()` 装饰器注册
3. 在配置文件中通过 `model.name` 引用
4. 确保文件不超过 300 行

### Adding a New Dataset

1. 在 `src/data/` 创建 Dataset 类
2. 使用 `@DATASET_REGISTRY.register()` 注册
3. 实现 `__len__` 和 `__getitem__`
4. 更新 `data/labels/README.md` 说明标签格式

### Modifying Configuration

配置文件位于 `configs/`，分层结构：
- `defaults/` - 基础配置模板
- `experiments/` - 具体实验配置
- `environments/` - 环境相关配置（路径、设备）

修改后通过 CLI 验证：
```bash
uv run python src/train.py --config configs/xxx.yaml --dry-run
```

## Common Commands

```bash
# 安装环境
uv sync

# 训练实验
uv run python src/train.py --config configs/baseline.yaml

# 覆盖配置参数
uv run python src/train.py --config configs/baseline.yaml train.epochs=50 train.learning_rate=0.0001

# 仅测试（加载权重）
uv run python src/train.py --config configs/baseline.yaml --mode test --checkpoint runs/exp_xxx/best_model.pth

# 下载预训练模型
uv run python scripts/download_models_from_hf.py --model all

# 重新生成数据集5折划分
cd data/labels && python3 generate_folds.py
```

## File Modifications

**默认行为**: 提供代码片段，不直接修改文件

**直接修改文件**: 仅在用户明确请求时（如 "直接改", "modify the file", "apply the changes"）

## Documentation

技术文档位于 `docs/`，论文相关位于 `papers/`:

```
docs/
├── setup/           # 环境配置、安装指南
├── experiments/     # 实验记录、结果分析
├── models/          # 模型架构、设计文档
├── tutorials/       # 使用教程、最佳实践
└── references/      # 参考资料、论文笔记

papers/
├── pdfs/            # 论文PDF文件（添加到.gitignore）
├── notes/           # 论文解析笔记
└── implementations/ # 复现笔记、技术细节
```

**文档规范**: Markdown格式，不使用过多装饰性符号

## Important Notes

- **Git管理**: `runs/`, `pretrained_model/`, `papers/pdfs/*.pdf` 已添加到 `.gitignore`
- **随机种子**: 通过配置文件统一管理（`train.seed`）
- **路径处理**: 标签文件中的 `audio_path` 需在运行时动态调整为实际存储位置
- **性能监控**: 验证集用于选择最佳模型，测试集仅用于最终报告
- **错误分析**: 使用 `test_prediction.csv` 分析预测错误样本
