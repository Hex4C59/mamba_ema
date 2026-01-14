# Valence-Arousal 特征分离建模

## 背景

在之前的实验中（runs/ema_iemocap_2026-01-14_14-49-47 等），发现 CCC-V (0.45-0.64) 显著低于 CCC-A (0.64-0.73)，表明模型对 Valence 的预测能力弱于 Arousal。

## 问题分析

1. **Valence 和 Arousal 的特征依赖不同**：
   - Valence（情绪效价，正面/负面）：更依赖韵律特征（语调、节奏、音高变化）
   - Arousal（激活度，兴奋/平静）：更依赖语音能量（音量、强度、频谱能量）

2. **原始模型问题**：
   - 使用单一回归头预测 V 和 A
   - 两个维度共享相同的特征权重
   - 无法针对性地优化 Valence 预测

## 改进方案

### 架构修改

**分离的回归头 + 特征加权注意力**

```
                    ┌──────────────────┐
                    │   Fused Features │
                    │  [speech, prosody│
                    │     ema_state]   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
      ┌───────▼────────┐           ┌────────▼───────┐
      │ Valence Branch │           │ Arousal Branch │
      │ (prosody-heavy)│           │ (speech-heavy) │
      └───────┬────────┘           └────────┬───────┘
              │                             │
    ┌─────────▼──────────┐       ┌──────────▼─────────┐
    │ Attention Weights  │       │ Attention Weights  │
    │ [w_prosody, w_other]│      │ [w_speech, w_other]│
    └─────────┬──────────┘       └──────────┬─────────┘
              │                             │
    ┌─────────▼──────────┐       ┌──────────▼─────────┐
    │ Weighted Features  │       │ Weighted Features  │
    │ prosody↑ speech↓   │       │ speech↑ prosody↓   │
    └─────────┬──────────┘       └──────────┬─────────┘
              │                             │
    ┌─────────▼──────────┐       ┌──────────▼─────────┐
    │  Valence Head      │       │  Arousal Head      │
    │  (MLP + LN + ReLU) │       │  (MLP + LN + ReLU) │
    └─────────┬──────────┘       └──────────┬─────────┘
              │                             │
              ▼                             ▼
          valence_pred                  arousal_pred
```

### 核心代码

**1. 注意力模块**

```python
# Valence attention: 学习 prosody vs. other 的权重
self.valence_attention = nn.Sequential(
    nn.Linear(d_head_input, d_hidden),
    nn.Tanh(),
    nn.Linear(d_hidden, 2),  # [prosody weight, other weight]
    nn.Softmax(dim=-1)
)

# Arousal attention: 学习 speech vs. other 的权重
self.arousal_attention = nn.Sequential(
    nn.Linear(d_head_input, d_hidden),
    nn.Tanh(),
    nn.Linear(d_hidden, 2),  # [speech weight, other weight]
    nn.Softmax(dim=-1)
)
```

**2. 独立回归头**

```python
# Valence head
self.valence_head = nn.Sequential(
    nn.Linear(d_head_input, d_hidden),
    nn.LayerNorm(d_hidden),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_hidden, d_hidden // 2),
    nn.LayerNorm(d_hidden // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_hidden // 2, 1),
)

# Arousal head (same structure)
self.arousal_head = nn.Sequential(...)
```

**3. 前向传播中的特征加权**

```python
# Split features
speech_feat = head_input[:, :self.d_speech]      # [B, 1024]
prosody_feat = head_input[:, self.d_speech:...]  # [B, 64]
ema_feat = head_input[:, ...]                    # [B, d_state]

# Valence: emphasize prosody
valence_weights = self.valence_attention(head_input)  # [B, 2]
w_prosody_v = valence_weights[:, 0:1]
w_other_v = valence_weights[:, 1:2]

weighted_prosody_v = prosody_feat * w_prosody_v
weighted_speech_v = speech_feat * w_other_v
weighted_ema_v = ema_feat * w_other_v

valence_input = torch.cat([weighted_speech_v, weighted_prosody_v, weighted_ema_v], dim=-1)
valence = self.valence_head(valence_input).squeeze(-1)

# Arousal: emphasize speech
arousal_weights = self.arousal_attention(head_input)  # [B, 2]
w_speech_a = arousal_weights[:, 0:1]
w_other_a = arousal_weights[:, 1:2]

weighted_speech_a = speech_feat * w_speech_a
weighted_prosody_a = prosody_feat * w_other_a
weighted_ema_a = ema_feat * w_other_a

arousal_input = torch.cat([weighted_speech_a, weighted_prosody_a, weighted_ema_a], dim=-1)
arousal = self.arousal_head(arousal_input).squeeze(-1)
```

## 实现细节

### 参数增加

- **原始模型**: 共享回归头
  - regression_head: 1 x (1152→256→128→2)

- **新模型**: 分离头 + 注意力
  - valence_attention: 1152→256→2
  - valence_head: 1152→256→128→1
  - arousal_attention: 1152→256→2
  - arousal_head: 1152→256→128→1

参数量增加约 1.5MB，对总参数量 318M 影响极小。

### 优势

1. **针对性优化**: Valence 和 Arousal 使用不同的特征权重
2. **可解释性**: Attention 权重可以可视化，理解模型依赖哪些特征
3. **灵活性**: 两个分支独立训练，可以设置不同的学习率或损失权重
4. **性能提升**: 通过特征分离，预期 Valence 的 CCC 会显著提高

## 使用方法

### 训练

模型已自动升级，无需修改配置文件，直接训练即可：

```bash
# 单折训练
uv run python src/train.py --config configs/baseline.yaml --fold 1 --gpu 0

# 5折交叉验证
for fold in {1..5}; do
    uv run python src/train.py --config configs/baseline.yaml --fold $fold --gpu 0
done
```

### 可视化注意力权重（可选）

训练后可以分析模型学到的特征权重分布：

```python
import torch
model.eval()
with torch.no_grad():
    output = model(batch)
    # 在 forward 中添加 return attention weights
    v_weights = model.valence_weights  # [B, 2]
    a_weights = model.arousal_weights  # [B, 2]

    print(f"Valence - Prosody weight: {v_weights[:, 0].mean():.3f}")
    print(f"Arousal - Speech weight: {a_weights[:, 0].mean():.3f}")
```

## 预期效果

根据特征分离假设，预期：

1. **CCC-V 提升**: 从 0.45-0.64 提升到 0.60-0.70（+15% 相对提升）
2. **CCC-A 保持**: 维持在 0.64-0.73 范围
3. **MSE-V 降低**: 从 0.66-0.77 降低到 0.50-0.60

## 实验对比

| 模型版本 | CCC-V | CCC-A | CCC-Avg | MSE-V | MSE-A |
|---------|-------|-------|---------|-------|-------|
| 共享头   | 0.52  | 0.68  | 0.60    | 0.70  | 0.32  |
| 分离头   | ?     | ?     | ?       | ?     | ?     |

（待训练后更新）

## 文件修改

- `src/models/mamba_ema_model.py`: 主要修改
  - 添加 valence_attention, arousal_attention
  - 添加 valence_head, arousal_head
  - 修改 forward 实现特征分离加权

## 参考

- Valence-Arousal 模型: Russell's Circumplex Model of Affect
- Attention mechanism: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
- 特征加权: Zhang et al., "Attention-based Feature Fusion for Speech Emotion Recognition"
