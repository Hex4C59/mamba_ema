"""Speech encoder using pre-trained WavLM/Wav2Vec2 models with caching."""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor, AutoModel


class SpeechEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        pooling: str = "mean",
        freeze: bool = True,
        d_output: int = 768,
        cache_dir: str = "/tmp/wavlm_cache",
        use_cache: bool = True,
        extract_layers: List[int] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.d_output = d_output
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.extract_layers = extract_layers

        # 创建缓存目录
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = AutoModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # Enable gradient checkpointing to save memory (trade compute for memory)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        if extract_layers is not None:
            self.model.config.output_hidden_states = True
            self.num_layers = len(extract_layers)
            # Learnable layer weights (initialized uniformly)
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        else:
            self.num_layers = 1

        if freeze:
            self.freeze()

        # Attention pooling layer
        if pooling == "attention":
            self.attention = nn.Linear(d_output, 1)

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(
        self,
        waveforms: List[torch.Tensor],
        names: List[str] | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # 这行代码获取模型参数所在的设备
        device = next(self.model.parameters()).device

        # 返回序列特征 (32, 100, 768)
        if return_sequence:
            return self._forward_sequence(waveforms, device)
        # 返回池化特征 (32, 768)
        else:
            return self._forward_pooled(waveforms, names, device)

    def _forward_pooled(
        self,
        waveforms: List[torch.Tensor],
        names: List[str] | None,
        device: torch.device,
    ) -> torch.Tensor:
        features = []

        for i, wf in enumerate(waveforms):
            cache_path = None
            # 如果有缓存特征，优先使用缓存
            if self.use_cache and names is not None:
                cache_path = self.cache_dir / f"{names[i]}.pt"
                if cache_path.exists():
                    try:
                        cached_feat = torch.load(cache_path, map_location=device, weights_only=True)
                        features.append(cached_feat)
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load cache {cache_path}: {e}")

            # 提取特征
            wf = wf.to(device)
            with torch.set_grad_enabled(self.training and not self._is_frozen()):
                # 在waveforms前添加batch维度
                outputs = self.model(wf.unsqueeze(0))

                if self.extract_layers is not None:
                    hidden_states = outputs.hidden_states  # 获取特征维度
                    # 提取指定层的特征
                    layer_features = [hidden_states[layer_idx] for layer_idx in self.extract_layers]
                    # 加权融合: sum(weight_i * layer_i)
                    weights = torch.softmax(self.layer_weights, dim=0)  # 归一化权重
                    hidden = sum(
                        w * layer for w, layer in zip(weights, layer_features)
                    )  # [1, T, d_output]
                else:
                    hidden = outputs.last_hidden_state  # [1, T, d_output]

            if self.pooling == "mean":
                pooled = hidden.mean(dim=1).squeeze(0)  # [d_output]
            elif self.pooling == "attention":
                attn_weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=1)  # [1, T]
                pooled = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1).squeeze(0)  # [d_output]

            features.append(pooled)

            if self.use_cache and cache_path is not None:
                try:
                    torch.save(pooled.cpu(), cache_path)
                except Exception as e:
                    print(f"Warning: Failed to save cache {cache_path}: {e}")

        # Stack batch
        return torch.stack(features)  # [B, d_output]

    def _forward_sequence(
        self, waveforms: List[torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract sequence features with batch processing and padding.

        Args:
            waveforms: List of audio tensors with different lengths
            device: Target device

        Returns:
            padded_sequences: [B, T_max, d_output] - Padded sequences
            attention_mask: [B, T_max] - Attention mask (True for padding)
        """
        # Move waveforms to device and get lengths
        waveforms = [wf.to(device) for wf in waveforms]
        lengths = [wf.shape[0] for wf in waveforms]
        max_len = max(lengths)

        # Pad waveforms to same length
        padded_waveforms = []
        for wf in waveforms:
            if wf.shape[0] < max_len:
                padding = torch.zeros(max_len - wf.shape[0], device=device)
                wf = torch.cat([wf, padding])
            padded_waveforms.append(wf)

        # Stack to batch: [B, T_max]
        batch_waveforms = torch.stack(padded_waveforms)

        # Create attention mask for WavLM (0 for valid, 1 for padding)
        attention_mask = torch.zeros(len(waveforms), max_len, device=device)
        for i, length in enumerate(lengths):
            if length < max_len:
                attention_mask[i, length:] = 1

        # Forward through model with batch processing
        with torch.set_grad_enabled(self.training and not self._is_frozen()):
            outputs = self.model(batch_waveforms, attention_mask=attention_mask)

            if self.extract_layers is not None:
                hidden_states = outputs.hidden_states
                layer_features = [hidden_states[layer_idx] for layer_idx in self.extract_layers]
                weights = torch.softmax(self.layer_weights, dim=0)
                hidden = sum(w * layer for w, layer in zip(weights, layer_features))
            else:
                hidden = outputs.last_hidden_state  # [B, T_feature, d_output]

        # Create padding mask for output (True for padding positions)
        # WavLM downsamples by ~320x, so we need to calculate feature lengths
        feature_lengths = [self._get_feat_extract_output_lengths(l) for l in lengths]
        max_feat_len = hidden.shape[1]

        padding_mask = torch.zeros(len(waveforms), max_feat_len, dtype=torch.bool, device=device)
        for i, feat_len in enumerate(feature_lengths):
            if feat_len < max_feat_len:
                padding_mask[i, feat_len:] = True

        return hidden, padding_mask

    def _get_feat_extract_output_lengths(self, input_length: int) -> int:
        """Calculate output length after WavLM's feature extraction.

        WavLM uses conv layers with stride, typically downsamples by ~320x for 16kHz audio.
        """
        # This is an approximation - exact calculation depends on model architecture
        # For WavLM: 7 conv layers with stride 2,2,2,2,2,2,2
        # Total stride = 2^7 = 128 for base, but with padding it's ~320
        return input_length // 320

    def _is_frozen(self) -> bool:
        """Check if model is frozen."""
        return not next(self.model.parameters()).requires_grad
