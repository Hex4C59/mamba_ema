# Deep Learning Template

一个轻量、可扩展的深度学习项目模板，基于 PyTorch，提供配置驱动、组件注册、可复现的实验框架。

## 项目概览

本项目采用模块化设计，将数据处理、模型定义、训练流程、配置管理等环节解耦，便于快速迭代与协作开发。

### 核心特性

- **配置驱动**：基于 YAML 的配置系统，支持命令行覆盖（`key.subkey=value`）
- **组件注册**：Registry 机制实现模型、损失函数、数据集等组件的即插即用
- **可复现性**：统一的随机种子管理，配置自动保存
- **代码规范**：类型注解、文档字符串、模块化设计

---

## 目录结构详解

```
deep_learning_template/
├── configs/              # 实验配置文件
├── data/                 # 数据存放区
├── docs/                 # 项目文档与技术总结
├── papers/               # 论文PDF与解析笔记
├── pretrained_model/     # 预训练模型权重
├── runs/                 # 训练产出（日志、权重、可视化）
├── scripts/              # 数据预处理与工具脚本
├── src/                  # 核心源代码
├── pyproject.toml        # 项目依赖与元信息
└── README.md             # 本文件
```

---

### 1. `configs/` - 实验配置

**用途**：集中管理所有实验参数（模型超参、数据路径、训练设置等）

**推荐组织方式**：
```
configs/
├── defaults/          # 基础配置模板
├── experiments/       # 具体实验配置（继承 defaults 并覆盖）
└── environments/      # 环境相关配置（路径、设备、并行策略）
```

**配置文件示例** (YAML)：
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

**使用方式**：
```bash
# 加载配置并通过 CLI 覆盖
uv run python src/train.py --config configs/experiments/baseline.yaml \
    train.epochs=50 model.params.num_classes=100
```

**注意事项**：
- 不要在配置文件中硬编码绝对路径或凭据
- 使用环境变量或 `environments/` 子目录管理敏感信息
- 重要参数变更需更新文档

---

### 2. `docs/` - 项目文档

用途：存放项目相关的技术文档、实验报告、使用指南等 Markdown 文件

推荐组织方式：
```
docs/
├── README.md           # 文档目录说明
├── setup/              # 环境配置、安装指南
├── experiments/        # 实验记录、结果分析
├── models/             # 模型架构、设计文档
├── tutorials/          # 使用教程、最佳实践
└── references/         # 参考资料、论文笔记
```

文档类型示例：
- 实验报告：记录实验配置、结果、分析和结论
- 技术总结：问题解决方案、技术要点总结
- 使用指南：功能说明、使用步骤、示例代码
- 设计文档：架构设计、接口说明

编写规范：
- 使用清晰的标题层级，避免过多装饰符号
- 文件命名使用小写字母和连字符：`model-comparison.md`
- 代码块标注语言类型，表格使用标准格式
- 保持简洁专业的风格

---

### 3. `papers/` - 论文资料

用途：存放论文复现相关的资料，包括原始PDF、解析笔记、复现说明

推荐组织方式：
```
papers/
├── pdfs/               # 原始论文PDF文件
├── notes/              # 论文解析笔记和要点总结
├── implementations/    # 复现实现说明和技术文档
└── README.md           # 说明文档
```

使用流程：
- 将论文PDF放入 `pdfs/` 目录
- 使用 Claude 读取PDF并生成解析笔记到 `notes/`
- 记录复现过程和技术细节到 `implementations/`

文件命名规范：
- PDF：`年份-缩写-主题.pdf`（如 `2017-transformer-attention.pdf`）
- 笔记：`论文简称-关键词.md`（如 `transformer-attention.md`）
- 复现文档：`模型名-implementation.md`

注意事项：
- PDF文件较大，建议添加到 `.gitignore`
- 笔记和复现文档应纳入版本控制
- 使用清晰的目录结构便于查找和管理

---

### 4. `data/` - 数据存放区

**用途**：存放原始数据、中间处理结果、标注文件等

**推荐组织方式**：
```
data/
├── raw/              # 原始数据（下载后的未处理数据）
├── processed/        # 预处理后的训练数据
├── labels/           # 标注文件（JSON/CSV/TXT）
└── splits/           # 训练集/验证集/测试集划分清单
```

**文件类型示例**：
- 图像数据：`data/processed/train/`, `data/processed/val/`
- 音频数据：`data/processed/*.wav`
- 文本数据：`data/processed/corpus.txt`

**注意事项**：
- **不要**将大型数据集提交到 Git 仓库
- 添加 `.gitignore` 规则忽略数据文件
- 数据集由用户手动准备并放置到对应目录

---

### 5. `pretrained_model/` - 预训练模型权重

**用途**：存放从 Hugging Face、ModelScope 等平台下载的预训练权重

**文件命名示例**：
```
pretrained_model/
├── resnet50-imagenet.pth
├── bert-base-uncased/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
└── README.md          # 权重来源、版权、使用说明
```

**下载脚本**：

项目提供了两个预训练模型下载脚本（位于 `scripts/` 目录）：

1. **从 Hugging Face 下载** (`download_models_from_hf.py`)
   ```bash
   # 下载所有预置模型
   uv run python scripts/download_models_from_hf.py --model all

   # 下载特定模型
   uv run python scripts/download_models_from_hf.py -m wavlm-large-finetuned-iemocap

   # 下载自定义模型
   uv run python scripts/download_models_from_hf.py \
       --repo-id facebook/hubert-large-ll60k \
       --name hubert_large

   # 强制重新下载
   uv run python scripts/download_models_from_hf.py -m wavlm-large-finetuned-iemocap --force
   ```

   **环境变量配置**：
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxx          # Hugging Face token
   export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像（可选）
   ```

2. **从 ModelScope 下载** (`download_models_from_ms.py`)
   ```bash
   # 下载预置的 ModelScope 模型
   uv run python scripts/download_models_from_ms.py
   ```

**加载示例**：
```python
from pathlib import Path
import torch

model = MyModel()
checkpoint = torch.load(Path("pretrained_model/resnet50-imagenet.pth"))
model.load_state_dict(checkpoint)
```

**注意事项**：
- 大文件使用 Git LFS 或外部存储
- 可以手动从 Hugging Face 等平台下载，也可使用项目提供的脚本
- 在 README 中注明权重来源、许可证、预期性能

---

### 6. `runs/` - 训练产出

**用途**：保存训练日志、模型权重、TensorBoard 记录、可视化结果等

**必须保存的文件**：
1. `config.yaml` - 实验配置文件（用于复现）
2. `train.log` - 完整训练日志（损失、指标、时间戳）
3. `best_results.txt` - 最佳性能指标记录
4. `test_prediction.csv` - 测试集预测结果（样本ID + 预测值 + 真实值）
5. 损失优化曲线图（如 `loss_curve.png`, `metrics_curve.png`）
6. 最佳权重文件（如 `best_model.pth`）

**自动生成结构示例**：
```
runs/
├── exp_2025-01-13_14-30-45/
│   ├── config.yaml              # 本次实验的完整配置
│   ├── train.log                # 训练日志（损失、指标、时间戳）
│   ├── best_results.txt         # 最佳结果摘要（验证集/测试集最佳指标）
│   ├── test_prediction.csv      # 测试集预测结果
│   ├── loss_curve.png           # 训练/验证损失曲线
│   ├── metrics_curve.png        # 性能指标曲线（准确率、F1等）
│   ├── best_model.pth           # 最佳模型权重
│   ├── checkpoints/             # 其他检查点（可选）
│   │   ├── epoch_10.pth
│   │   └── epoch_20.pth
│   └── tensorboard/             # TensorBoard 事件文件（可选）
└── README.md
```

**注意事项**：
- 添加到 `.gitignore`（产出文件不进入版本控制）
- 定期清理过期实验产出
- 保留关键实验的必要文件（上述6项）以便复现

---

### 7. `scripts/` - 数据处理与工具脚本

**用途**：数据预处理、格式转换、数据集划分、模型下载等工具脚本

**推荐组织方式**：
```
scripts/
├── download_models_from_hf.py   # 从 Hugging Face 下载预训练模型
├── download_models_from_ms.py   # 从 ModelScope 下载预训练模型
├── preprocess.py                # 数据清洗、归一化、增强
├── split_dataset.py             # 生成训练/验证/测试集划分
└── build_manifest.py            # 生成文件清单（路径索引）
```

**关键脚本说明**：

#### `download_models_from_hf.py` - Hugging Face 模型下载

从 Hugging Face Hub 下载语音表征/ASR 预训练模型。

**预置模型**：
- `wav2vec2-large-robust-12-ft-emotion-msp-dim` - 情感维度预测
- `wavlm-large-msp-podcast-emotion-dim` - WavLM 情感模型
- `wav2vec2-large-superb-er` - 情感识别
- `hubert-large-superb-er` - HuBERT 情感识别
- `wavlm-large-finetuned-iemocap` - IEMOCAP 微调版

**使用方式**：
```bash
# 下载所有预置模型
uv run python scripts/download_models_from_hf.py --model all

# 下载指定模型
uv run python scripts/download_models_from_hf.py -m wavlm-large-finetuned-iemocap

# 自定义下载任意 HF 模型
uv run python scripts/download_models_from_hf.py \
    --repo-id facebook/hubert-large-ll60k \
    --name hubert_large

# 强制覆盖已存在的模型
uv run python scripts/download_models_from_hf.py -m wav2vec2-large-superb-er --force
```

**环境变量**：
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx          # HF token（可选）
export HF_ENDPOINT=https://hf-mirror.com  # 使用国内镜像加速
```

#### `download_models_from_ms.py` - ModelScope 模型下载

从魔塔社区下载预训练模型。

**预置模型**：
- `emotion2vec_plus_large` - 情感向量表征模型

**使用方式**：
```bash
uv run python scripts/download_models_from_ms.py
```

**数据处理脚本示例**：
```bash
# 预处理并划分数据
uv run python scripts/preprocess.py --input data/raw --output data/processed
uv run python scripts/split_dataset.py --data data/processed --ratio 0.8:0.1:0.1
```

**编写规范**：
- 支持命令行参数（`argparse` 或 `click`）
- 打印清晰的进度信息（使用 `tqdm`）
- 在脚本头部添加文档字符串说明用途与参数
- 单个文件不超过 300 行

---

### 8. `src/` - 核心源代码

项目的主要代码目录，包含训练入口、模型定义、数据加载等所有核心逻辑。

注意：本项目采用单一入口设计，train.py 包含训练、验证、测试的完整流程，不需要单独的 eval.py 或 infer.py。

#### 7.1 `src/train.py` - 训练入口脚本（唯一入口）

作用：解析配置、初始化模型/数据/优化器、执行训练-验证-测试完整流程、保存权重与日志

功能模式：
- 训练模式：执行完整的训练-验证-测试流程
- 测试模式：仅加载权重执行测试
- 验证模式：仅加载权重执行验证（可选）

典型流程：
```python
# 1. 加载配置
cfg = load_yaml(Path(args.config))
cfg = apply_overrides(cfg, args.overrides)

# 2. 设置随机种子
set_global_seed(cfg["train"]["seed"])

# 3. 构建组件（通过 Registry）
model = build_from_config(cfg["model"], MODEL_REGISTRY)
train_loader = build_from_config(cfg["data"]["train"], DATASET_REGISTRY)
val_loader = build_from_config(cfg["data"]["val"], DATASET_REGISTRY)
test_loader = build_from_config(cfg["data"]["test"], DATASET_REGISTRY)
loss_fn = build_from_config(cfg["loss"], LOSS_REGISTRY)

# 4. 训练与验证循环
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer, loss_fn)
    val_metrics = validate(model, val_loader, metric_fn)
    save_checkpoint(model, f"runs/{exp_name}/epoch_{epoch}.pth")

# 5. 最终测试
test_metrics = test(model, test_loader, metric_fn)
save_results(test_metrics, f"runs/{exp_name}/test_results.json")
```

运行示例：
```bash
# 完整训练+验证+测试流程
uv run python src/train.py \
    --config configs/experiments/baseline.yaml \
    train.epochs=50 \
    data.batch_size=64

# 仅执行测试（加载已有权重）
uv run python src/train.py \
    --config configs/experiments/baseline.yaml \
    --mode test \
    --checkpoint runs/exp_xxx/best.pth

# 仅执行验证
uv run python src/train.py \
    --config configs/experiments/baseline.yaml \
    --mode val \
    --checkpoint runs/exp_xxx/epoch_10.pth
```

重要说明：
- 单个 train.py 文件控制所有训练、验证、测试逻辑
- 不需要单独的 eval.py、infer.py、test.py 文件
- 通过 --mode 参数切换不同执行模式
- 每个文件不超过 300 行，复杂逻辑应拆分到独立模块（如 `src/trainers/`、`src/evaluators/`）

---

#### 7.2 `src/models/` - 模型定义

**用途**：存放所有模型架构定义（CNN、Transformer、自定义网络等）

**文件示例**：
```
src/models/
├── __init__.py           # 导入并注册所有模型
├── model_x.py            # 模型 X 的定义
├── resnet.py             # ResNet 系列模型
└── README.md
```

**代码示例**：
```python
# src/models/resnet.py
from src.utils.registry import Registry

MODEL_REGISTRY = Registry("model")

@MODEL_REGISTRY.register("ResNet50")
class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # 模型定义...

    def forward(self, x: Tensor) -> Tensor:
        # 前向传播...
        return x
```

**使用规范**：
- 使用 `@MODEL_REGISTRY.register` 装饰器注册模型
- 构造函数参数与配置文件 `model.params` 一一对应
- 添加类型注解与文档字符串

---

#### 7.3 `src/data/` - 数据集与数据加载

**用途**：定义 `torch.utils.data.Dataset` 实现、数据增强、数据加载器工厂

**文件示例**：
```
src/data/
├── __init__.py
├── image_dataset.py      # 图像数据集
├── audio_dataset.py      # 音频数据集
├── transforms.py         # 数据增强与预处理
└── README.md
```

**代码示例**：
```python
from torch.utils.data import Dataset
from src.utils.registry import Registry

DATASET_REGISTRY = Registry("dataset")

@DATASET_REGISTRY.register("ImageDataset")
class ImageDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.split = split
        # 加载文件列表...

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        # 返回 (image, label)
        ...

    def __len__(self) -> int:
        return len(self.samples)
```

---

#### 7.4 `src/losses/` - 损失函数

**用途**：实现项目特定的损失函数（交叉熵、对比学习损失、组合损失等）

**文件示例**：
```
src/losses/
├── __init__.py
├── <task>_loss.py        # 任务特定损失（模板文件）
├── focal_loss.py         # Focal Loss 实现
└── README.md
```

**代码示例**：
```python
from torch import nn, Tensor
from src.utils.registry import Registry

LOSS_REGISTRY = Registry("loss")

@LOSS_REGISTRY.register("FocalLoss")
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # 损失计算...
        return loss
```

---

#### 7.5 `src/metrics/` - 评估指标

**用途**：实现准确率、F1、mAP、BLEU 等评估指标

**文件示例**：
```
src/metrics/
├── __init__.py
├── <task>_metric.py      # 任务特定指标（模板文件）
├── accuracy.py           # 准确率
├── f1_score.py           # F1 分数
└── README.md
```

**代码示例**：
```python
from src.utils.registry import Registry

METRIC_REGISTRY = Registry("metric")

@METRIC_REGISTRY.register("Accuracy")
class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, pred: Tensor, target: Tensor) -> None:
        self.correct += (pred.argmax(1) == target).sum().item()
        self.total += target.size(0)

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0
```

---

#### 7.6 `src/utils/` - 工具模块

**用途**：提供配置加载、组件注册、随机种子设置等通用功能

**文件说明**：

| 文件 | 作用 | 主要函数/类 |
|------|------|------------|
| `config.py` | YAML 配置加载与命令行覆盖 | `load_yaml()`, `apply_overrides()`, `save_yaml()` |
| `registry.py` | 组件注册与工厂模式实现 | `Registry` 类, `build_from_config()` |
| `seed.py` | 全局随机种子设置 | `set_global_seed()` |

**`config.py` - 配置管理**

核心功能：
- `load_yaml(path)`: 加载 YAML 配置文件为字典
- `apply_overrides(config, overrides)`: 应用命令行覆盖（支持嵌套键 `model.params.hidden=128`）
- `save_yaml(data, path)`: 保存配置到文件

使用示例：
```python
from pathlib import Path
from src.utils.config import load_yaml, apply_overrides

cfg = load_yaml(Path("configs/default.yaml"))
cfg = apply_overrides(cfg, ["train.epochs=100", "model.name=ResNet50"])
```

**`registry.py` - 组件注册系统**

核心功能：
- `Registry` 类：管理名称到类/函数的映射
- `@registry.register(name)`: 装饰器注册组件
- `build_from_config(cfg, registry)`: 根据配置实例化组件

使用示例：
```python
from src.utils.registry import Registry

MODEL_REGISTRY = Registry("model")

@MODEL_REGISTRY.register("MyModel")
class MyModel(nn.Module):
    def __init__(self, hidden_dim: int):
        ...

# 通过配置实例化
cfg = {"name": "MyModel", "params": {"hidden_dim": 256}}
model = build_from_config(cfg, MODEL_REGISTRY)
```

**`seed.py` - 随机种子管理**

核心功能：
- `set_global_seed(seed)`: 设置 Python、NumPy、PyTorch 的随机种子

使用示例：
```python
from src.utils.seed import set_global_seed

set_global_seed(42)  # 确保实验可复现
```

---

## 快速开始

### 1. 环境配置

```bash
# 使用 UV 管理依赖（统一使用 uv）
uv venv
uv pip install -e .
```

### 2. 准备数据

```bash
# 将数据集放置到对应目录（由用户手动准备）
# data/raw/           - 原始数据
# data/processed/     - 预处理后的数据

# 使用脚本进行预处理和划分
uv run python scripts/preprocess.py --input data/raw --output data/processed
uv run python scripts/split_dataset.py --data data/processed --ratio 0.8:0.1:0.1
```

### 3. 运行训练

```bash
# 使用默认配置
uv run python src/train.py --config configs/default.yaml

# 覆盖特定参数
uv run python src/train.py \
    --config configs/experiments/baseline.yaml \
    train.epochs=50 \
    data.batch_size=64 \
    model.params.dropout=0.2
```

### 4. 监控训练

```bash
# 启动 TensorBoard（如果已集成）
tensorboard --logdir runs/
```

---

## 开发指南

### 添加新模型

1. 在 `src/models/` 创建模型文件（如 `my_model.py`）
2. 定义模型类并使用 `@MODEL_REGISTRY.register` 注册
3. 在 `src/models/__init__.py` 导入模型以触发注册
4. 更新配置文件 `model.name` 为注册名称

### 添加新数据集

1. 在 `src/data/` 创建数据集类（继承 `torch.utils.data.Dataset`）
2. 使用 `@DATASET_REGISTRY.register` 注册
3. 更新配置文件 `data.name` 和 `data.params`

### 添加新损失函数/指标

1. 在 `src/losses/` 或 `src/metrics/` 创建实现文件
2. 使用对应 Registry 注册
3. 在配置文件中引用

---

## 代码规范

- **类型注解**：所有函数参数与返回值需添加类型提示
- **文档字符串**：使用 Google 风格的 docstring
- **命名规范**：
  - 函数/变量：`snake_case`
  - 类：`PascalCase`
  - 常量：`UPPER_SNAKE_CASE`
- **行宽限制**：最大 100 字符
- **文件长度限制**：单个文件**不超过 300 行**，超过需拆分为多个模块
- **导入顺序**：标准库 → 第三方库 → 本地模块
- **环境管理**：统一使用 **UV** 管理 Python 环境和依赖

---

## 依赖管理

主要依赖（见 `pyproject.toml`）：

- **PyTorch** (≥2.3.1)：深度学习框架
- **torchvision** (≥0.18.1)：图像处理与预训练模型
- **torchaudio** (≥2.3.1)：音频处理
- **transformers** (≥4.56.1)：Hugging Face 预训练模型
- **pyyaml** (≥6.0.2)：配置文件解析
- **tqdm** (≥4.67.1)：进度条
- **matplotlib** (≥3.10.6)：可视化
- **pandas** (≥2.3.2)：数据处理

---

## 常见问题

**Q: 如何复现已有实验？**
A: 从 `runs/<实验目录>/config.yaml` 复制配置，使用相同随机种子运行训练脚本。

**Q: 如何使用多 GPU 训练？**
A: 在训练脚本中使用 `torch.nn.DataParallel` 或 `DistributedDataParallel`（需在配置中添加设备参数）。

**Q: 配置文件覆盖规则是什么？**
A: 命令行参数优先级高于配置文件。使用点分隔的键路径（`train.lr=0.01`）覆盖嵌套配置。

---

## 许可证

MIT License（详见 `LICENSE` 文件）

---

## 联系方式

- **作者**：Liu Yang
- **邮箱**：yang.liu6@siat.ac.cn
- **GitHub**：[Hex4C59/deep_learning_template](https://github.com/Hex4C59/deep_learning_template)
