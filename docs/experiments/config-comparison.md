# 微调配置对比

## 三种配置方案

根据你的 RTX 4090 24GB 显存，我创建了三种配置：

### 1. 标准配置 (configs/finetune.yaml) ✅ 推荐

**适用场景**: 平衡性能和稳定性

```yaml
batch_size: 8
accumulation_steps: 1
d_prosody_out: 64
d_hidden: 256
```

**资源占用**:
- 显存: ~16-18 GB
- 训练速度: ~12 分钟/epoch

**预期效果**:
- CCC-V: 0.59 → **0.64-0.68** (+8-15%)

---

### 2. 激进配置 (configs/finetune_aggressive.yaml)

**适用场景**: 与标准配置相同，这是为了兼容性创建的

```yaml
batch_size: 8
accumulation_steps: 1
d_prosody_out: 64
d_hidden: 256
```

实际上与标准配置相同。

---

### 3. 极限配置 (configs/finetune_max.yaml) 🚀 最高性能

**适用场景**: 榨干显存，追求极致性能

```yaml
batch_size: 12              # ⚡ 增大 50%
accumulation_steps: 1
d_prosody_out: 128          # ⚡ prosody 特征翻倍
d_hidden: 384               # ⚡ 隐藏层增大 50%
```

**资源占用**:
- 显存: ~20-22 GB（接近极限）
- 训练速度: ~15 分钟/epoch（更大的 batch）

**预期效果**:
- CCC-V: 0.59 → **0.66-0.70** (+12-19%)
- 更大的 batch_size 可能带来更稳定的训练
- 更大的模型容量可能学到更丰富的特征

**风险**:
- 可能 OOM（如果系统有其他进程占用显存）
- 训练时间略长

---

## 推荐使用

### 快速验证（当前正在运行）
```bash
# 已经在运行，等待完成即可
# 使用 configs/finetune.yaml (batch_size=8)
```

### 下次实验：尝试极限配置
```bash
# 停止当前训练（如果需要）
pkill -f "train.py.*finetune"

# 运行极限配置
uv run python src/train.py --config configs/finetune_max.yaml --fold 1 --gpu 0
```

---

## 显存占用对比

| 配置 | Batch Size | Prosody Dim | Hidden Dim | 显存 | 速度 | 预期 CCC-V |
|------|-----------|-------------|-----------|------|------|-----------|
| 保守 (旧) | 4 + acc=2 | 64 | 256 | 12 GB | ~12 min/ep | 0.62-0.65 |
| **标准 (当前)** | **8** | **64** | **256** | **16-18 GB** | **~12 min/ep** | **0.64-0.68** |
| 极限 | 12 | 128 | 384 | 20-22 GB | ~15 min/ep | 0.66-0.70 |

---

## 选择建议

**如果你想快速验证**:
- ✅ 继续使用当前的 `finetune.yaml` (batch_size=8)
- 预期 2-3 小时完成

**如果你想追求最高性能**:
- 🚀 停止当前训练，改用 `finetune_max.yaml`
- 可能多花 20% 时间，但 CCC-V 可能更高

**我的建议**:
- 先让当前训练跑完（已经开始了）
- 如果 CCC-V 达到 0.64-0.65，就很好了
- 如果想冲刺 0.68-0.70，下次用 `finetune_max.yaml`

---

## 极限配置的额外优化

如果使用 `finetune_max.yaml`，还可以尝试：

1. **增大 Mamba 层数**
```yaml
mamba_n_layers: 3  # 从 2 增加到 3
```

2. **使用更多 WavLM 层**
```yaml
speech_encoder_layers: [3, 6, 9, 12, 15, 18, 21, 24]  # 8 层
```

3. **增大 EMA state**
```yaml
d_state: 128  # 从 64 增大到 128
```

预期额外显存占用: +2-3 GB

---

*创建时间: 2026-01-14 18:15*
*当前训练: finetune.yaml (batch_size=8)*
*建议: 先完成当前训练，评估后决定是否使用极限配置*
