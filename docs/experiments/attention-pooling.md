# 注意力池化实现

## 改动说明

将 Speech Encoder 的池化方式从 **Mean Pooling** 改为 **Attention Pooling**，以更好地利用时序信息。

## 原理对比

### Mean Pooling（原方案）
```python
hidden = [B, T, 1024]  # WavLM 输出
pooled = hidden.mean(dim=1)  # [B, 1024]
# 问题：所有时刻权重相同，情感显著时刻被平均稀释
```

### Attention Pooling（新方案）
```python
hidden = [B, T, 1024]  # WavLM 输出

# 学习每个时刻的重要性
attn_weights = softmax(attention_layer(hidden))  # [B, T, 1]

# 加权求和
pooled = (hidden * attn_weights).sum(dim=1)  # [B, 1024]

# 优势：关注情感爆发点、重读词、韵律变化等关键时刻
```

## 实现细节

### 1. 修改模型定义
**文件**: `src/models/mamba_ema_model.py`

添加 `speech_encoder_pooling` 参数：
```python
def __init__(
    self,
    # ...
    speech_encoder_pooling: str = "mean",  # 新增参数
    # ...
):
    self.speech_encoder = SpeechEncoder(
        # ...
        pooling=speech_encoder_pooling,  # 传递给 SpeechEncoder
    )
```

### 2. 注意力层实现
**文件**: `src/models/encoders/speech_encoder.py`

已有实现（line 67-68, 133-140）：
```python
# 初始化
if pooling == "attention":
    self.attention = nn.Linear(d_output, 1)

# Forward
attn_weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=1)  # [1, T]
pooled = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1).squeeze(0)  # [d_output]
```

### 3. 配置文件
**文件**: `configs/finetune.yaml`, `configs/finetune_full.yaml`

```yaml
model:
  params:
    speech_encoder_pooling: attention  # 启用注意力池化
```

## 预期效果

### 参数量增加
```
Mean Pooling: 0 额外参数
Attention Pooling: 1024 × 1 + 1 = 1025 参数（可忽略）
```

### 性能提升预期
- **CCC-V**: +1-3%（关注 valence 相关的韵律变化时刻）
- **CCC-A**: +1-2%（关注 arousal 相关的能量峰值时刻）

### 计算开销
- 额外计算量：~1% FLOPs（一个线性层 + softmax）
- 训练速度影响：可忽略

## 验证

```bash
# 验证配置正确加载
uv run python -c "
from src.models.mamba_ema_model import MambaEMAModel
import yaml

with open('configs/finetune.yaml') as f:
    config = yaml.safe_load(f)

model = MambaEMAModel(**config['model']['params'])
print(f'Pooling: {model.speech_encoder.pooling}')
print(f'Has attention: {hasattr(model.speech_encoder, \"attention\")}')
"
```

预期输出：
```
Pooling: attention
Has attention: True
```

## 相关文件

- `src/models/mamba_ema_model.py`: 主模型
- `src/models/encoders/speech_encoder.py`: Speech Encoder 实现
- `configs/finetune.yaml`: 快速验证配置
- `configs/finetune_full.yaml`: 完整训练配置

## 后续计划

如果注意力池化效果良好，可以进一步尝试：
1. Multi-head Attention Pooling（多头注意力）
2. 可视化注意力权重（分析模型关注了哪些时刻）
3. 结合 Statistical Pooling（同时使用 mean + std）
