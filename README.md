# Mamba-EMA

基于 Mamba-EMA 模型的情感语音识别项目。

## 项目简介

本项目实现了用于情感维度识别（Valence-Arousal-Dominance）的深度学习模型，采用配置驱动、组件注册、单一入口的设计模式。

**核心特性**：
- 单一入口训练脚本（train/val/test 统一管理）
- 组件注册机制（模型、损失函数、数据集动态装配）
- 配置驱动（YAML 配置 + 命令行覆盖）
- UV 包管理（快速依赖安装和环境隔离）

## 环境要求

- Python >= 3.12
- UV 包管理器
- PyTorch 2.6.0
- CUDA（推荐，用于 GPU 加速）

## 快速开始

### 安装依赖

```bash
# 安装 UV（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步项目依赖
uv sync
```

### 训练模型

```bash
# 使用配置文件训练
uv run python src/train.py --config configs/baseline.yaml

# 覆盖配置参数
uv run python src/train.py \
    --config configs/baseline.yaml \
    train.epochs=50 \
    train.learning_rate=0.0001
```

### 测试模型

```bash
# 加载权重进行测试
uv run python src/train.py \
    --config configs/baseline.yaml \
    --mode test \
    --checkpoint runs/exp_xxx/best_model.pth
```

## 项目结构

```
mamba_ema/
├── configs/              # YAML 配置文件
├── data/                 # 数据集和标注文件
├── docs/                 # 项目文档
├── papers/               # 论文 PDF 与笔记
├── pretrained_model/     # 预训练权重（软链接）
├── runs/                 # 实验结果（自动生成）
├── scripts/              # 工具脚本
└── src/                  # 核心代码
    ├── train.py          # 唯一入口（训练/验证/测试）
    ├── models/           # 模型定义
    ├── data/             # 数据集和数据加载
    ├── losses/           # 损失函数
    ├── metrics/          # 评估指标
    └── utils/            # 工具模块
```

## 支持的数据集

- **IEMOCAP**: 10,039 样本，英文，VAD 维度，5 折交叉验证
- **MSP-PODCAST**: 19,898 样本，英文，VAD 维度
- **CCSEMO**: 7,554 样本，中文，VA 维度

## 下载预训练模型

```bash
# 从 Hugging Face 下载所有模型
uv run python scripts/download_models_from_hf.py --model all

# 下载特定模型
uv run python scripts/download_models_from_hf.py -m wavlm-large-finetuned-iemocap

# 从 ModelScope 下载
uv run python scripts/download_models_from_ms.py --model wavlm-large
```

## 开发指南

详细的开发文档和项目规范请参考：
- `CLAUDE.md` - 项目架构、代码规范、开发工作流
- `docs/` - 技术文档、实验报告、使用教程
- `papers/` - 论文笔记、复现记录

## License

MIT License
