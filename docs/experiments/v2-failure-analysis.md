# V2 实验失败分析

## 实验结果

| Fold | CCC-V | CCC-A | CCC-Avg | 备注 |
|------|-------|-------|---------|------|
| 1 | 0.6496 | 0.7309 | 0.6902 | 最好 |
| 2 | 0.6779 | 0.6825 | 0.6802 | 较好 |
| 3 | 0.4720 | 0.6637 | 0.5678 | **异常差** |
| 4 | 0.5634 | 0.6569 | 0.6102 | 差 |
| 5 | 0.5464 | 0.7424 | 0.6444 | **严重过拟合** |
| **平均** | **0.5819** | **0.6953** | **0.6386** | |

### 与V1对比

- CCC-V: 0.5896 → 0.5819 (**-1.3%**, 反而下降！)
- CCC-A: 0.6867 → 0.6953 (+1.2%)
- CCC-Avg: 0.6381 → 0.6386 (+0.1%)

### 距离目标

- CCC-V: 0.5819 vs 0.68 (差 **0.0981**, 还差14%！)
- CCC-A: 0.6953 vs 0.68 (已达标 +2.2%)

## 问题分析

### 1. 训练极不稳定

**表现**：
- Fold 1-2: CCC-V 0.65-0.68 (正常)
- Fold 3-4: CCC-V 0.47-0.56 (异常低)
- Fold 5: 验证集0.69，测试集0.55 (过拟合14%！)

**原因**：
- batch_size=16 太大，梯度过于平滑
- WavLM微调本身就不稳定，需要小batch + 低学习率
- 不同fold数据分布差异被放大

### 2. 过拟合严重

**Fold 5**:
```
验证集: CCC-V 0.69, CCC-A 0.73
测试集: CCC-V 0.55, CCC-A 0.74
→ Valence 过拟合了 0.14 (25%)！
```

**原因**：
- dropout=0.15 太小
- d_prosody_out=128 + mamba_n_layers=4 容量过大
- weight_decay=0.0001 太小
- early_stopping patience=5 太小，没等到泛化

### 3. Weighted Loss 适得其反

```yaml
valence_weight: 1.5
arousal_weight: 1.0
```

**问题**：
- 强行提升Valence权重破坏了V-A的自然耦合
- 模型学到了"过度关注Valence"，反而泛化更差
- Arousal信息被边缘化

### 4. 注意力池化可能无效

CCC-V提升几乎为零，可能：
- 注意力层没有足够的数据学习（需要更多epochs）
- 或者对于pooling后的全局特征，attention意义不大
- 需要配合时序特征才能发挥作用

## V2配置的问题

```yaml
# ❌ 问题配置
batch_size: 16              # 太大，梯度不稳定
d_prosody_out: 128          # 容量过大
mamba_n_layers: 4           # 容量过大
dropout: 0.15               # 太小，过拟合
weight_decay: 0.0001        # 太小
early_stopping.patience: 5  # 太激进

loss:
  name: WeightedCCC
  params:
    valence_weight: 1.5     # 破坏V-A耦合
```

## 改进建议

### 方案1: 回归稳健基线（推荐优先尝试）

见 `configs/finetune_v3.yaml`:

```yaml
# ✅ 改进配置
batch_size: 8               # 更稳定
d_prosody_out: 64           # 避免过拟合
mamba_n_layers: 2           # 避免过拟合
dropout: 0.3                # 增强正则化
weight_decay: 0.001         # 增强正则化
lr: 0.00005                 # 降低学习率
encoder_lr: 0.000005        # encoder更保守
early_stopping.patience: 15 # 等待更久

loss:
  name: CCC                 # 不加权，自然学习
```

**预期效果**：
- 减少过拟合 (验证集/测试集gap缩小)
- 5个fold更稳定 (方差减小)
- CCC-V 可能提升到 0.62-0.64

### 方案2: 实现时序特征 + 交叉注意力

如果方案1效果仍不足，考虑架构改动：

1. **保留时序特征**：Speech Encoder输出 [B, T, 1024]
2. **交叉注意力**：Prosody查询Speech时序
3. **Mamba处理时序**：让Mamba真正学习时序依赖

**工作量**：2-3小时代码修改

### 方案3: 数据增强

- SpecAugment (频谱增强)
- 时间扰动 (time stretching/shifting)
- Mixup (样本混合)

**工作量**：1-2小时

### 方案4: 改用更好的预训练模型

- WavLM-Large → **Wav2Vec2-XLSR** (多语言，泛化更好)
- 或 **HuBERT-Large**
- 或 **Whisper encoder** (更强的语音理解)

**工作量**：30分钟（只需改配置）

## 下一步建议

### 立即执行（今天）

1. **跑 V3 配置**（稳健基线）
   ```bash
   # 先在Fold 1验证
   uv run python src/train.py --config configs/finetune_v3.yaml --fold 1

   # 如果效果好，再跑全部
   bash scripts/train_all_folds_parallel.sh configs/finetune_v3.yaml
   ```

2. **观察指标**：
   - 验证集/测试集gap是否缩小 (<5%为正常)
   - 5个fold的方差是否减小
   - CCC-V是否稳定在0.62+

### 短期计划（明天）

如果V3仍不达标：
1. 实现SpecAugment数据增强
2. 或尝试Whisper encoder（可能泛化更好）
3. 或实现时序特征+交叉注意力

### 关键问题

**为什么别人能到0.68？**

可能的原因：
1. 使用了数据增强
2. 使用了更好的预训练模型
3. 使用了ensemble (多模型集成)
4. 使用了时序特征（而不是pooled特征）
5. 超参数调得更好
6. 数据集可能不同（IEMOCAP有多个版本）

需要你提供原论文/参考实现的更多细节才能确定。
