# papers - 论文资料目录

本目录用于存放论文复现相关的资料，包括原始PDF、解析笔记、复现说明等。

## 目录结构

```
papers/
├── pdfs/               # 原始论文PDF文件
├── notes/              # 论文解析笔记和要点总结
├── implementations/    # 复现实现说明和技术文档
└── README.md           # 本文件
```

## 使用流程

### 1. 添加论文PDF

将论文PDF文件放入 `pdfs/` 目录，建议使用规范的命名：

```
pdfs/
├── 2023-bert-attention-is-all-you-need.pdf
├── 2024-resnet-deep-residual-learning.pdf
└── 2025-wavlm-large-scale-self-supervised.pdf
```

命名规范：`年份-简短名称-关键词.pdf`

### 2. 生成解析笔记

通过 Claude 解析PDF后，会在 `notes/` 目录生成对应的笔记文件：

```
notes/
├── attention-is-all-you-need.md
├── deep-residual-learning.md
└── wavlm-large-scale.md
```

笔记内容包括：
- 论文基本信息（标题、作者、发表时间、会议/期刊）
- 研究问题和动机
- 模型架构和方法
- 实验设置（数据集、超参数、训练细节）
- 主要结果和性能指标
- 复现要点和注意事项

### 3. 记录复现过程

在 `implementations/` 目录下创建复现说明文档：

```
implementations/
├── bert-implementation.md
├── resnet-implementation.md
└── wavlm-implementation.md
```

内容包括：
- 环境配置和依赖
- 数据准备步骤
- 模型实现要点
- 训练脚本和配置
- 遇到的问题和解决方案
- 复现结果对比

## 文件命名规范

### PDF文件

格式：`年份-缩写-主题.pdf`

示例：
- `2017-transformer-attention-is-all-you-need.pdf`
- `2020-wav2vec2-self-supervised-speech.pdf`
- `2023-llama-open-efficient-foundation.pdf`

### 笔记文件

格式：`论文简称-关键词.md`

示例：
- `transformer-attention-mechanism.md`
- `wav2vec2-speech-representation.md`
- `llama-language-model.md`

### 复现文档

格式：`模型名-implementation.md` 或 `模型名-reproduction-notes.md`

示例：
- `bert-implementation.md`
- `resnet-reproduction-notes.md`
- `wavlm-training-guide.md`

## 笔记模板

创建论文笔记时可参考以下模板：

```markdown
# 论文标题

## 基本信息

- 作者：
- 发表时间：
- 会议/期刊：
- 论文链接：
- 代码链接：

## 研究问题

简要描述论文要解决的问题和研究动机。

## 方法概述

核心方法和模型架构的简要说明。

## 模型架构

详细的模型结构说明，包括：
- 网络层次结构
- 关键组件
- 参数规模

## 实验设置

### 数据集
- 数据集名称和规模
- 数据预处理方法

### 训练配置
- 优化器和学习率
- Batch size 和 epoch 数
- 正则化方法
- 硬件配置

### 评估指标
- 使用的评估指标
- 对比基线

## 主要结果

关键实验结果和性能指标。

## 复现要点

- 关键超参数
- 容易出错的地方
- 与原论文的差异
- 实现建议

## 参考资料

- 官方代码仓库
- 相关博客文章
- 其他实现参考
```

## 维护建议

- 定期整理和归档不再使用的论文
- 为重要论文添加标签或分类
- 记录复现成功率和遇到的问题
- 更新实现笔记，补充新的发现

## Git 管理

PDF文件通常较大，建议在 `.gitignore` 中忽略：

```gitignore
# 忽略PDF文件
papers/pdfs/*.pdf

# 但保留笔记和文档
!papers/notes/
!papers/implementations/
```

如需版本控制PDF，可使用 Git LFS。

---

最后更新：2025-01-13
