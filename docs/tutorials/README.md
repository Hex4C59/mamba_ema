下面是一份可以直接放进仓库的 **README.md**（中文，偏工程落地），覆盖：数据格式、模块定义、前向流程、训练（TBPTT + scheduled sampling + CCC）、推理、个性化、消融与常见坑。你复制粘贴就能用。

---

# Mamba + EMA 状态记忆的端到端语音情绪回归（VA）

本项目实现一个**在线（streaming）**的 Valence/Arousal 回归模型，核心思想：

* **Speech encoder** 提取 utterance-level 语音表示
* **Prosody extractor** 提取显式韵律特征（eGeMAPS/F0/energy…）
* **Speaker encoder + FiLM** 用说话人条件调制语音表示（个性化友好）
* **Mamba updater** 从当前观测中提取“状态增量”
* **EMA state** 以显式时间尺度累积长期上下文
* **Regression head** 用（当前观测 + 长期状态）预测 VA
* 可选 **autoregressive feedback**：上一时刻预测参与状态更新（训练用 scheduled sampling 对齐线上）

---

## 1. 模型概览

### 1.1 符号约定

* `x_t`: 第 t 句音频波形（16kHz）
* `h_t ∈ R^{d_h}`: speech encoder 句向量（通常 768）
* `p_t ∈ R^{d_p}`: prosody 特征（例如 eGeMAPS 88 维）
* `s ∈ R^{d_s}`: speaker embedding（例如 ECAPA 192 维）
* `FiLM(s) → (γ, β) ∈ R^{d_h} × R^{d_h}`
* `h'_t = γ ⊙ h_t + β`
* `p̃_t = MLP_p(p_t) ∈ R^{d_p2}`（推荐 64）
* `z_t = [h'_t ; p̃_t] ∈ R^{d_z}`（d_z = d_h + d_p2）
* `ŷ_t = (v_t, a_t) ∈ R^2`
* `u_t ∈ R^{d_c}`: 状态增量（d_c=64/128）
* `c_t ∈ R^{d_c}`: EMA 状态

### 1.2 前向计算（核心）

**状态增量（Mamba）**
`u_t = MambaUpdater([z_t; ŷ_{t-1}])`

**EMA 状态更新**
`c_t = α c_{t-1} + (1-α) u_t`

**VA 预测**
`ŷ_t = Head([z_t; c_t])`

> 线上推理只需要保存 `(c_t, ŷ_t)`，不需要保存窗口历史。

---

## 2. 依赖与环境

建议 Python 3.10+。

* PyTorch（>=2.0）
* transformers（用于 wav2vec2/WavLM/HuBERT）
* torchaudio
* mamba-ssm（或你自己的 Mamba 实现）
* numpy, scipy
* （可选）openSMILE（提取 eGeMAPS），或 librosa/praat-parselmouth 提取 F0/energy

---

## 3. 数据准备

### 3.1 样本单位与切分

以 **utterance** 为基本样本，并按 **session / 对话** 排序形成序列（训练 TBPTT 时用）。

每条 utterance 至少包含：

* `audio_path` 或原始 `wav`
* `session_id`（同一段对话/录音）
* `speaker_id`
* `valence`, `arousal`（连续值，范围建议归一到 [0,1] 或 [-1,1]）

### 3.2 推荐 JSONL 格式

每行一个 utterance：

```json
{"utt_id":"s1_u0001","session_id":"s1","speaker_id":"A","audio_path":".../s1_u0001.wav","valence":0.62,"arousal":0.41}
```

> 训练时 DataLoader 按 `session_id` 分组，保证一个 batch 内是多个 session 的“时间片段”。

---

## 4. 特征模块实现细节

### 4.1 Speech Encoder（wav2vec2/WavLM）

实现建议：

* 输入：`waveform [B, L]`（16kHz）
* 输出：`h_t [B, d_h]`

常用做法：

1. `transformers` 模型输出 `last_hidden_state [B, T, d_h]`
2. pooling 得句向量：

   * mean pooling（简单）
   * attention pooling（更稳）

attention pooling 示意：

```python
att = softmax(W * H)  # [B, T, 1]
h = sum(att * H, dim=1)  # [B, d_h]
```

### 4.2 Prosody Extractor（显式韵律）

两条路线：

**A) 离线提取（推荐）**
用 openSMILE 提取 eGeMAPS（如 88 维），保存到文件（npz/pt），训练时直接加载。

**B) 在线提取（较慢）**
librosa/praat-parselmouth 提 F0、energy、voicing 等，训练阶段会更慢。

无论哪条路线，最终都映射到 `p̃_t ∈ R^{64}`：

```python
p_t = prosody_feat  # [B, d_p]
p_tilde = MLP_p(p_t)  # [B, 64]
```

### 4.3 Speaker Encoder（ECAPA / x-vector）

建议用现成预训练模型得到 `s ∈ R^{d_s}`，并做 L2 normalize：

```python
s = s / (s.norm(dim=-1, keepdim=True) + 1e-8)
```

两种使用方式：

* 训练时按 utterance 得 `s`
* 或按 speaker 聚合多句得到更稳定 `s`

---

## 5. FiLM（特征调制）实现

FiLM 本质是：用一个小 MLP 将 speaker embedding 映射成 `γ, β`，然后调制 `h_t`。

推荐实现（两层 MLP）：

```python
class FiLM(nn.Module):
    def __init__(self, speaker_dim=192, feat_dim=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(speaker_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * feat_dim),
        )

    def forward(self, h, s):
        gb = self.net(s)               # [B, 2*feat_dim]
        gamma, beta = gb.chunk(2, -1)  # each [B, feat_dim]
        gamma = torch.tanh(gamma) + 1.0  # optional: stabilize scale near 1
        return gamma * h + beta
```

最终融合得到观测向量：

```python
h_mod = film(h, s)        # [B, 768]
p_tilde = mlp_p(p)        # [B, 64]
z = torch.cat([h_mod, p_tilde], dim=-1)  # [B, 832]
```

---

## 6. Mamba Updater + EMA State

### 6.1 Mamba Updater 输入输出

输入：`x_upd = [z_t; ŷ_{t-1}] ∈ R^{d_z+2}`
输出：`u_t ∈ R^{d_c}`（建议 64/128）

建议用轻量结构：Linear 投影到 `d_model`，1–2 层 Mamba，再 Linear 回 `d_c`。

伪代码：

```python
x = proj_in(x_upd)      # [B, d_model]
x = mamba_block(x)      # 这里取决于你 mamba 实现（可能需要 [B, L, d_model]）
u = proj_out(x)         # [B, d_c]
u = layernorm(u)        # optional
```

> 实际 mamba-ssm 常以序列形式工作：你可以把时间维当成 `L`（TBPTT 片段长度），一次喂一个片段 `[B, L, d_model]`，更高效。

### 6.2 EMA State 更新

固定 α（推荐先用固定）：

```python
c = alpha * c + (1 - alpha) * u
```

推荐 α：

* 0.7–0.85（换成 Mamba 后不建议太大）
* Valence/Arousal 可用不同 α（A 往往更快）

---

## 7. Regression Head（VA 回归头）

输入：`[z_t; c_t] ∈ R^{d_z + d_c}`
输出：`ŷ_t ∈ R^2`

建议结构：两头 MLP（或共享 trunk + 两个线性头）

```python
shared = MLP_shared(torch.cat([z, c], -1))
v = torch.sigmoid(head_v(shared))
a = torch.sigmoid(head_a(shared))
y = torch.stack([v, a], dim=-1)  # [B, 2]
```

若标签是 [-1,1]，用 `tanh` 替代 sigmoid。

---

## 8. 损失函数（CCC + MSE）

### 8.1 CCC 定义（batch 级）

对一维输出 `y`、预测 `ŷ`：

[
CCC = \frac{2\rho \sigma_y \sigma_{\hat y}}{\sigma_y^2 + \sigma_{\hat y}^2 + (\mu_y - \mu_{\hat y})^2}
]

训练损失推荐：

```python
loss = 1 - 0.5 * (ccc_v + ccc_a) + 0.1 * (mse_v + mse_a)
```

实现注意：

* 加 `eps` 防止除零
* 用 float32/float64 都可，float64 更稳但更慢

---

## 9. 训练流程（TBPTT + Scheduled Sampling）

### 9.1 为什么需要 TBPTT

模型递归依赖 `c_t` 和 `ŷ_{t-1}`。若对整段对话反传，显存/时间会爆。

做法：按 session 形成序列，切成长度 `T` 的片段训练（T=20~50）。

### 9.2 Scheduled Sampling（对齐线上）

在 updater 输入中的 `ŷ_{t-1}`：

* 以概率 `p` 用真值 `y_{t-1}`
* 以概率 `1-p` 用模型预测 `ŷ_{t-1}`

`p` 随训练下降，例如线性：

* 前 30% steps: p=1
* 中间：线性降到 0
* 后 20% steps: p=0

### 9.3 训练伪代码（片段级）

核心结构：

```python
for session in sessions:
    c = zeros([B, d_c])
    y_prev = init_y  # [B, 2], e.g., 0.5

    for chunk in chunks(session, T):
        # optional TBPTT: detach state at chunk boundary
        c = c.detach()
        y_prev = y_prev.detach()

        loss_chunk = 0
        for t in chunk:
            z_t = fuse_features(x_t)  # speech+prosody+speaker

            # scheduled sampling
            y_in = y_gt_prev if rand() < p else y_prev

            u_t = mamba_updater(concat(z_t, y_in))
            c = alpha*c + (1-alpha)*u_t

            y_pred = head(concat(z_t, c))
            loss_chunk += loss_fn(y_pred, y_gt)

            y_prev = y_pred
            y_gt_prev = y_gt

        loss_chunk.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**强烈建议：**

* gradient clipping：`1.0`
* dropout（MLP、Mamba 输出）
* LayerNorm（u_t、或 z_t）

---

## 10. 推理（在线 Streaming）

推理只需要保存：

* `c_t`
* `ŷ_t`

步骤：

1. `h_t = speech_encoder(x_t)`
2. `p_t = prosody(x_t)`（或从缓存读取）
3. `s = speaker_encoder(x_t)`（或 speaker 注册向量）
4. `z_t = fuse(h_t, p_t, s)`
5. `u_t = updater([z_t; ŷ_{t-1}])`
6. `c_t = EMA(c_{t-1}, u_t)`
7. `ŷ_t = head([z_t; c_t])`

---

## 11. Few-shot 个性化（可选）

推荐最稳的个性化策略（10 句也不容易炸）：

* 冻住 speech encoder、mamba、head
* **只更新 FiLM 的参数**（或只更新 FiLM 的最后一层）
* 少量步数（20–100 steps），小学习率（1e-4～5e-4）

这会让模型“快速学会如何解释这个人的说话方式”。

---

## 12. 消融实验建议（写论文/做报告很关键）

1. 无状态：`ŷ_t = head(z_t)`
2. GRU updater + EMA vs Mamba updater + EMA
3. 去掉 prosody：`z_t = h'_t`
4. 去掉 FiLM：`h'_t = h_t`（或直接拼 s）
5. 固定 α vs 可学习 α（clip 到 [0.6, 0.9]）
6. scheduled sampling vs 纯 teacher forcing vs 纯自回归

指标：CCC-V / CCC-A / CCC-avg + RMSE。

---

## 13. 常见坑（以及怎么解决）

* **训练很抖 / CCC 不涨**：先关掉 autoregressive（不喂 ŷ_{t-1}），只用 `z_t → u_t`；稳定后再加 scheduled sampling。
* **误差累积**：p 下降要慢一点；或者只在 updater 输入里使用 `ŷ_{t-1}`，head 不用。
* **c_t 塌缩为常数**：对 `u_t` 加 LayerNorm；给 updater 输出加 dropout；减小 α。
* **speaker leakage**：speaker embedding 不要跨 session 误用；分割时 session/speaker-disjoint。
* **prosody 提取很慢**：离线预计算缓存；训练时只加载。

---

## 14. 推荐默认超参（可作为 config.yaml 起点）

* `d_h=768`
* `d_p=88`（eGeMAPS）
* `d_p2=64`
* `d_s=192`（ECAPA）
* `d_c=64`
* `alpha=0.8`
* TBPTT `T=30`
* optimizer：AdamW

  * 上层（FiLM、MLP、Mamba、Head）lr=1e-3
  * encoder 解冻后 lr=3e-5
* grad clip：1.0
* scheduled sampling：p 从 1 → 0（覆盖 50% steps）

---
