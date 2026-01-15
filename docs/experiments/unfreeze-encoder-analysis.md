# 解冻 WavLM Encoder 的资源消耗分析

## 硬件配置

- **GPU**: 8x NVIDIA RTX 4090 (24GB)
- **当前使用**: 单 GPU 训练

---

## 📊 参数量分析

### 当前模型（WavLM 冻结）

| 模块 | 参数量 | 可训练 |
|------|--------|--------|
| WavLM-Large | 315.45M | ❌ 冻结 |
| Prosody Encoder | 0.02M | ✓ |
| Speaker Encoder (ECAPA) | 0.07M | ✓ |
| FiLM | 0.58M | ✓ |
| Mamba Updater | 1.17M | ✓ |
| Valence/Arousal Heads | 1.24M | ✓ |
| **总计** | **318.54M** | **3.08M (1%)** |

### 解冻后

| 配置 | 总参数 | 可训练参数 | 可训练比例 |
|------|--------|-----------|----------|
| 当前（冻结） | 318.54M | 3.08M | 1.0% |
| **解冻全部** | 318.54M | **318.54M** | **100%** |
| **解冻顶 6 层** | 318.54M | ~80M | ~25% |

**结论**: 解冻会让可训练参数从 3M 增加到 315M，增加 **100 倍**！

---

## 💾 显存占用分析

### 显存构成

解冻训练的显存占用包含 4 部分：
1. **模型参数** (FP32): 318M × 4 bytes = 1.27 GB
2. **梯度** (FP32): 318M × 4 bytes = 1.27 GB（仅可训练参数）
3. **优化器状态** (AdamW): 318M × 8 bytes = 2.55 GB（momentum + variance）
4. **激活值**: 与 batch_size 成正比，WavLM 激活值很大

### 不同 Batch Size 的显存占用

| Batch Size | 冻结 (GB) | 解冻 (GB) | 增加 (GB) | RTX 4090 可行性 |
|-----------|-----------|-----------|----------|----------------|
| 2 | 2.26 | 8.88 | +6.63 | ✓✓ 充裕 |
| 4 | 3.23 | 12.79 | +9.56 | ✓✓ 充裕 |
| **8** | **5.19** | **20.60** | **+15.42** | **✓ 勉强** (留 3.4GB) |
| 16 | 9.09 | 36.23 | +27.13 | ✗ OOM |
| 32 | 16.91 | 67.48 | +50.57 | ✗ OOM |

### 结论

**RTX 4090 (24GB) 使用建议**:
- ✅ **推荐 batch_size=4**: 显存占用 12.79 GB，留有 11 GB 余量
- ⚠️ **勉强 batch_size=8**: 显存占用 20.60 GB，留有 3.4 GB 余量（险！）
- ❌ **不推荐 batch_size≥16**: 会 OOM

**注意**: 实际显存占用可能略高（CUDA kernels、临时缓存等），建议保守设置。

---

## ⏱️ 训练时间分析

### 单 Fold 训练时间（50 epochs）

假设 IEMOCAP 单 fold 约 7000 训练样本：

| 配置 | Batch Size | 单 epoch 时间 | 50 epochs 总时间 | 速度 |
|------|-----------|--------------|----------------|------|
| 冻结 | 8 | ~5 分钟 | **~4 小时** | 10 samples/sec |
| 解冻 | 8 | ~12 分钟 | **~10 小时** | 4 samples/sec |
| 解冻 | 4 | ~12 分钟 | **~10 小时** | 4 samples/sec |

**慢速倍数**: 解冻比冻结慢 **2.5-3 倍**

### 5 Fold 交叉验证总时间

| 配置 | 单 Fold | 5 Fold 总计 | 增加时间 |
|------|---------|------------|---------|
| 冻结 | 4 小时 | **20 小时** | - |
| 解冻 | 10 小时 | **50 小时** | **+30 小时** |

**实际可能更快**（Early Stopping 通常在 10-20 epochs 停止）：
- 冻结: 8-12 小时
- 解冻: 20-30 小时

---

## 🎯 优化方案对比

### 方案 1: 全解冻 + 差异学习率 ⭐⭐⭐⭐⭐

**配置**:
```python
freeze=False
optimizer = AdamW([
    {'params': speech_encoder.model.parameters(), 'lr': 1e-5},  # WavLM 小学习率
    {'params': other_params, 'lr': 1e-4},  # 其他正常学习率
])
```

**资源需求**:
- 显存: 20.60 GB (batch_size=8) 或 12.79 GB (batch_size=4)
- 时间: ~10 小时/fold
- 预期收益: CCC-V +8-15% → **0.64-0.68**

**优点**:
- 收益最大，最可能达到 0.68
- 实现简单

**缺点**:
- 显存紧张（batch_size=8 时）
- 训练时间增加 2.5x

---

### 方案 2: 渐进式解冻 ⭐⭐⭐⭐

**配置**:
- Epoch 1-10: 冻结 WavLM
- Epoch 11-20: 解冻 WavLM 顶 4 层
- Epoch 21-50: 全解冻

**资源需求**:
- 显存: 前期 5 GB，后期 20 GB
- 时间: ~7 小时/fold（平均）
- 预期收益: CCC-V +7-12% → **0.63-0.66**

**优点**:
- 更稳定，不易过拟合
- 前期训练快

**缺点**:
- 实现复杂
- 收益略低于全解冻

---

### 方案 3: 仅解冻顶层 ⭐⭐⭐

**配置**:
```python
freeze=True
# 手动解冻 layer 18-23（顶 6 层）
for i in range(18, 24):
    model.speech_encoder.model.encoder.layers[i].requires_grad = True
```

**资源需求**:
- 显存: ~10 GB (batch_size=8)
- 时间: ~6 小时/fold
- 预期收益: CCC-V +5-8% → **0.62-0.64**

**优点**:
- 显存占用适中
- 训练时间适中

**缺点**:
- 收益较小，可能达不到 0.68

---

### 方案 4: 梯度累积（省显存） ⭐⭐⭐⭐

**配置**:
```python
batch_size = 4
accumulation_steps = 2  # 等效 batch_size=8
```

**资源需求**:
- 显存: 12.79 GB（batch_size=4）
- 时间: ~10 小时/fold
- 预期收益: 与方案 1 相同

**优点**:
- 显存安全
- 等效大 batch_size（更稳定）

**缺点**:
- 实现稍复杂
- 训练时间略长（~10%）

---

## 💡 推荐策略

### 你的硬件（8x RTX 4090 24GB）

**最佳方案**: **方案 1（全解冻） + 方案 4（梯度累积）**

```python
# 配置
batch_size = 4
accumulation_steps = 2  # 等效 batch_size=8
freeze = False

optimizer = AdamW([
    {'params': model.speech_encoder.model.parameters(), 'lr': 1e-5},
    {'params': other_params, 'lr': 1e-4},
], weight_decay=1e-4)
```

**预期**:
- 显存: 12.79 GB（安全）
- 时间: ~10 小时/fold × 5 = **50 小时**
- 收益: CCC-V **0.64-0.68** (+8-15%)

**为什么推荐**:
1. ✅ 显存安全（12.79 GB，留 11 GB 余量）
2. ✅ 收益最大（最可能达到 0.68）
3. ✅ 可并行训练（你有 8 张卡，可同时跑多个 fold）

---

## 🚀 并行加速策略

你有 **8 张 RTX 4090**，可以**并行训练多个 fold**：

### 串行训练（当前）
- 5 fold × 10 小时 = **50 小时**

### 并行训练（推荐）
- 5 fold 同时跑 → **10 小时**（节省 40 小时！）

```bash
# 同时启动 5 个 fold
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --config configs/finetune.yaml --fold 1 &
CUDA_VISIBLE_DEVICES=1 uv run python src/train.py --config configs/finetune.yaml --fold 2 &
CUDA_VISIBLE_DEVICES=2 uv run python src/train.py --config configs/finetune.yaml --fold 3 &
CUDA_VISIBLE_DEVICES=3 uv run python src/train.py --config configs/finetune.yaml --fold 4 &
CUDA_VISIBLE_DEVICES=4 uv run python src/train.py --config configs/finetune.yaml --fold 5 &
```

**总时间**: 仅需 **10 小时**（而不是 50 小时）！

---

## 📋 投入产出比

| 指标 | 冻结（当前） | 解冻（推荐） | 差异 |
|------|-------------|-------------|------|
| CCC-V | 0.5896 | **0.64-0.68** | **+8-15%** |
| 显存 | 5 GB | 13 GB | +8 GB |
| 时间（5 fold 串行） | 20 小时 | 50 小时 | +30 小时 |
| 时间（5 fold 并行） | 4 小时 | **10 小时** | **+6 小时** |

**结论**: 并行训练的话，只需额外 **6 小时**就能获得 **+8-15%** 的提升，**非常值得**！

---

## ⚙️ 省显存技巧总结

如果显存仍然不够，可以尝试：

1. **混合精度训练 (FP16)**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
       output = model(batch)
       loss = loss_fn(...)
   scaler.scale(loss).backward()
   ```
   - 显存减少 ~40%
   - 速度提升 ~20%

2. **梯度检查点 (Gradient Checkpointing)**
   ```python
   model.speech_encoder.model.gradient_checkpointing_enable()
   ```
   - 显存减少 ~30%
   - 速度下降 ~20%

3. **减少 extract_layers**
   ```yaml
   speech_encoder_layers: [12, 24]  # 只用 2 层而不是 4 层
   ```
   - 显存减少 ~10%

---

*分析完成时间: 2026-01-14 17:30*
