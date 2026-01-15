# 解冻 WavLM 微调使用指南

## 快速开始

### 1. 快速验证（fold 1，20 epochs，2-3 小时）

```bash
# 使用自动化脚本
bash scripts/verify_finetune.sh

# 或手动运行
uv run python src/train.py --config configs/finetune.yaml --fold 1 --gpu 0
```

**预期结果**:
- 训练时间: ~2-3 小时
- 显存占用: ~12-13 GB
- CCC-V: 0.59 → **0.62-0.65**

### 2. 完整训练（5 fold，50 epochs）

如果快速验证效果好（CCC-V ≥ 0.62），运行完整训练：

```bash
# 创建完整训练配置
cp configs/finetune.yaml configs/finetune_full.yaml

# 修改 epochs 为 50
sed -i 's/epochs: 20/epochs: 50/' configs/finetune_full.yaml

# 训练所有 fold（串行）
for fold in {1..5}; do
    uv run python src/train.py --config configs/finetune_full.yaml --fold $fold --gpu 0
done
```

### 3. 并行训练（推荐，节省时间）

你有 8 张 RTX 4090，可以并行训练：

```bash
# 同时训练 5 个 fold（每个用一张卡）
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --config configs/finetune_full.yaml --fold 1 &
CUDA_VISIBLE_DEVICES=1 uv run python src/train.py --config configs/finetune_full.yaml --fold 2 &
CUDA_VISIBLE_DEVICES=2 uv run python src/train.py --config configs/finetune_full.yaml --fold 3 &
CUDA_VISIBLE_DEVICES=3 uv run python src/train.py --config configs/finetune_full.yaml --fold 4 &
CUDA_VISIBLE_DEVICES=4 uv run python src/train.py --config configs/finetune_full.yaml --fold 5 &

# 等待所有任务完成
wait
echo "所有 fold 训练完成！"
```

**时间对比**:
- 串行: ~10 小时/fold × 5 = 50 小时
- 并行: ~10 小时（同时运行）

---

## 配置说明

### configs/finetune.yaml

关键参数：

```yaml
data:
  loader:
    batch_size: 4  # 小 batch_size 节省显存

model:
  params:
    freeze_speech_encoder: false  # ⚡ 解冻 WavLM

train:
  epochs: 20  # 快速验证用 20，完整训练用 50

  optimizer:
    lr: 0.0001         # 其他参数学习率 (1e-4)
    encoder_lr: 0.00001  # WavLM 学习率 (1e-5，慢慢微调)

  accumulation_steps: 2  # 梯度累积，等效 batch_size=8
```

### 参数调优建议

如果显存不足（OOM）：
```yaml
batch_size: 2           # 减小 batch_size
accumulation_steps: 4   # 增加累积步数
```

如果训练不稳定：
```yaml
encoder_lr: 0.000005    # 降低 encoder 学习率 (5e-6)
grad_clip: 0.3          # 降低梯度裁剪阈值
```

如果想加快训练：
```yaml
batch_size: 8           # 增大 batch_size（显存够的话）
accumulation_steps: 1   # 不使用累积
```

---

## 监控训练

### 查看实时日志

```bash
# 找到最新的实验目录
LATEST_RUN=$(ls -td runs/ema_iemocap_finetune_*fold1* | head -1)

# 查看训练日志
tail -f "$LATEST_RUN/train.log"
```

### 检查显存使用

```bash
# 实时监控显存
watch -n 1 nvidia-smi
```

### 预期训练曲线

**正常情况**:
- Epoch 1-5: CCC-V 快速上升 (0.45 → 0.55)
- Epoch 6-10: 稳定上升 (0.55 → 0.60)
- Epoch 11-20: 缓慢上升 (0.60 → 0.62-0.65)

**异常情况**:
- CCC-V 不上升或下降 → 学习率过大，降低 encoder_lr
- Loss 爆炸 (NaN) → 梯度裁剪过大，降低 grad_clip
- 显存 OOM → 降低 batch_size

---

## 预期效果

### 基于快速验证（20 epochs）

| 指标 | 冻结（v1） | 解冻（预期） | 提升 |
|------|-----------|------------|------|
| CCC-V | 0.5896 | **0.62-0.65** | **+5-10%** |
| CCC-A | 0.6867 | 0.68-0.70 | 持平或略升 |
| CCC-Avg | 0.6381 | **0.65-0.67** | **+2-5%** |

### 基于完整训练（50 epochs）

| 指标 | 冻结（v1） | 解冻（预期） | 提升 |
|------|-----------|------------|------|
| CCC-V | 0.5896 | **0.64-0.68** | **+8-15%** |
| CCC-A | 0.6867 | 0.68-0.70 | 持平或略升 |
| CCC-Avg | 0.6381 | **0.66-0.69** | **+3-8%** |

---

## 故障排查

### 显存 OOM

**症状**: CUDA out of memory

**解决方案**:
1. 降低 batch_size 到 2
2. 增加 accumulation_steps 到 4
3. 使用混合精度训练（需要修改代码）

### 训练速度慢

**症状**: 每 epoch 超过 30 分钟

**可能原因**:
- IO 瓶颈（数据加载慢）
- 特征缓存未命中

**解决方案**:
```bash
# 检查特征缓存是否存在
ls -lh /tmp/wavlm_cache/ | wc -l

# 如果缓存为空，第一个 epoch 会很慢（正常）
```

### CCC-V 不提升

**症状**: 20 epochs 后 CCC-V < 0.60

**可能原因**:
- encoder_lr 太小，微调不够
- encoder_lr 太大，破坏预训练知识

**解决方案**:
```yaml
# 尝试调整 encoder_lr
encoder_lr: 0.00005  # 5e-5 (增大)
# 或
encoder_lr: 0.000005  # 5e-6 (减小)
```

---

## 下一步

如果快速验证（20 epochs）效果达到预期（CCC-V ≥ 0.62）：

1. ✅ **运行完整 50 epochs 训练**
2. ✅ **并行训练 5 fold**（节省时间）
3. ✅ **更新实验记录** (`runs/experiment_comparison.md`)
4. ✅ **如果 CCC-V 达到 0.68+，尝试进一步优化**：
   - 添加 Sigmoid 输出约束
   - 调整 Valence 损失权重
   - 数据增强

---

## 技术细节

### 差异学习率实现

```python
param_groups = [
    {'params': model.speech_encoder.model.parameters(), 'lr': 1e-5},  # WavLM
    {'params': other_params, 'lr': 1e-4},  # 其他
]
optimizer = AdamW(param_groups, weight_decay=1e-4)
```

### 梯度累积实现

```python
for batch_idx, batch in enumerate(loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

等效于更大的 batch_size，但显存占用不变。

---

*创建时间: 2026-01-14 18:00*
*验证配置: RTX 4090 24GB, batch_size=4, accumulation_steps=2*
