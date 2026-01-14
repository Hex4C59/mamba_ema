# pretrained_model - 预训练模型权重存放目录

本目录用于存放从 **Hugging Face** 等平台下载的预训练模型权重，为项目提供迁移学习和微调的基础。

## 目录用途

- 集中管理预训练模型权重、配置文件、tokenizer 等资源
- 支持离线加载预训练模型，避免运行时重复下载
- 便于版本管理和实验复现

---

## 推荐目录结构

```
pretrained_model/
├── bert-base-uncased/           # BERT 模型
│   ├── config.json              # 模型配置
│   ├── pytorch_model.bin        # PyTorch 权重
│   ├── tokenizer_config.json    # Tokenizer 配置
│   ├── vocab.txt                # 词表
│   └── special_tokens_map.json  # 特殊 token 映射
│
├── resnet50/                    # ResNet50 (torchvision)
│   └── resnet50-imagenet.pth    # ImageNet 预训练权重
│
├── whisper-small/               # Whisper 语音模型
│   ├── config.json
│   ├── model.safetensors        # SafeTensors 格式权重
│   ├── preprocessor_config.json
│   └── tokenizer.json
│
└── README.md                    # 本文件
```

**命名建议**：
- 使用模型在 Hugging Face 上的官方名称（如 `bert-base-uncased`）
- 避免使用空格或特殊字符，统一使用小写和连字符

---

## 在代码中加载预训练模型

### 加载 Transformers 模型

```python
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# 从本地目录加载
model_dir = Path("pretrained_model/bert-base-uncased")
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用示例
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

### 加载 PyTorch 权重（torchvision）

```python
from pathlib import Path
import torch
from torchvision.models import resnet50

# 加载预训练 ResNet50
model = resnet50()
checkpoint = torch.load(
    Path("pretrained_model/resnet50/resnet50-imagenet.pth"),
    map_location="cpu"
)
model.load_state_dict(checkpoint)
```

### 加载 SafeTensors 格式权重

```python
from safetensors.torch import load_file
from pathlib import Path

# 加载 SafeTensors 权重
weights = load_file(Path("pretrained_model/whisper-small/model.safetensors"))

# 应用到模型
model.load_state_dict(weights)
```

---

## 常用模型参考

以下是常见的预训练模型及其 Hugging Face 仓库 ID（用户需自行从 Hugging Face 下载）：

### NLP 模型

| 模型名称 | Hugging Face ID | 用途 |
|---------|----------------|------|
| BERT Base | `bert-base-uncased` | 文本分类、NER、问答 |
| BERT Large | `bert-large-uncased` | 更强的文本理解 |
| RoBERTa | `roberta-base` | BERT 改进版 |
| GPT-2 | `gpt2` | 文本生成 |
| T5 | `t5-base` | 文本到文本转换 |
| BART | `facebook/bart-base` | 摘要、翻译 |

### 视觉模型

| 模型名称 | Hugging Face ID | 用途 |
|---------|----------------|------|
| ViT | `google/vit-base-patch16-224` | 图像分类 |
| CLIP | `openai/clip-vit-base-patch32` | 图文匹配 |
| DINO | `facebook/dino-vitb16` | 自监督视觉表示 |

### 多模态/语音模型

| 模型名称 | Hugging Face ID | 用途 |
|---------|----------------|------|
| Whisper | `openai/whisper-small` | 语音识别 |
| Wav2Vec2 | `facebook/wav2vec2-base-960h` | 语音特征提取 |
| BLIP | `Salesforce/blip-image-captioning-base` | 图像描述生成 |

---

## 存储与管理建议

### Git 管理

预训练模型文件通常较大（几百 MB 到几 GB），**不建议直接提交到 Git 仓库**。

在 `.gitignore` 中添加：
```gitignore
# 忽略预训练模型权重
pretrained_model/*
!pretrained_model/README.md
!pretrained_model/.gitkeep
```

### 使用 Git LFS（可选）

如果需要版本控制权重文件：

```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件
git lfs track "pretrained_model/**/*.bin"
git lfs track "pretrained_model/**/*.safetensors"
git lfs track "pretrained_model/**/*.pth"

# 提交 .gitattributes
git add .gitattributes
git commit -m "Track large model files with Git LFS"
```

---

## 模型信息记录模板

为每个模型创建 `<model_name>/INFO.md` 记录关键信息：

```markdown
# BERT Base Uncased

**来源**：https://huggingface.co/bert-base-uncased
**下载日期**：2025-01-13
**版本/Commit**：`main` (commit: abc123...)
**许可证**：Apache 2.0

## 模型信息
- 参数量：110M
- 训练数据：BooksCorpus + English Wikipedia
- 用途：文本分类、命名实体识别、问答任务

## 文件清单
- `pytorch_model.bin` (440 MB) - PyTorch 权重
- `config.json` - 模型配置
- `vocab.txt` - 词表文件
- `tokenizer_config.json` - Tokenizer 配置

## 使用示例
见项目 `src/models/bert_classifier.py`

## 注意事项
- 需要 transformers >= 4.0
- 推荐使用 fp16 以节省显存
```

---

## 环境变量配置（可选）

设置 Hugging Face 缓存目录：

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export HF_HOME="/mnt/shareEEx/liuyang/resources/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

---

## 常见问题

**Q: 如何验证权重文件完整性？**
A: 检查文件大小或使用 SHA256 校验：
```bash
sha256sum pretrained_model/bert-base-uncased/pytorch_model.bin
```

**Q: 权重文件占用空间太大怎么办？**
A: 考虑：
- 只保留必要的文件（删除不需要的 checkpoint）
- 使用量化版本（如 `int8`、`fp16`）
- 使用 SafeTensors 格式（通常比 `.bin` 更小）

**Q: 如何在代码中优雅地处理模型路径？**
A: 使用 `pathlib.Path` 和项目根目录相对路径：
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "pretrained_model" / "bert-base-uncased"

model = AutoModel.from_pretrained(MODEL_DIR)
```

---

## 许可证与版权

**重要提示**：
- 使用预训练模型前，请仔细阅读模型的许可证（在模型页面或 `LICENSE` 文件中）
- 商业使用需确认模型许可证允许（如 Apache 2.0、MIT、CC-BY）
- 部分模型仅限研究使用（如某些 Meta 的模型）
- 不要分发受限制的模型权重

常见许可证：
- **Apache 2.0**：商业友好，需保留版权声明
- **MIT**：几乎无限制
- **CC-BY-4.0**：需注明出处
- **CC-BY-NC-4.0**：仅限非商业用途

---

## 维护清单

定期执行以下维护任务：

- [ ] 清理不再使用的旧版本模型
- [ ] 检查磁盘空间占用
- [ ] 验证关键模型的可加载性
- [ ] 更新模型信息文档

---

**最后更新**：2025-01-13
