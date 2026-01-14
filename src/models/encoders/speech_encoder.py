"""Speech encoder using pre-trained WavLM/Wav2Vec2 models."""

from typing import List
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class SpeechEncoder(nn.Module):
    """Speech encoder with pooling layer.

    Uses pre-trained WavLM or Wav2Vec2 to extract utterance-level features.

    Args:
        model_name: HuggingFace model name
        pooling: Pooling method ("mean" or "attention")
        freeze: If True, freeze encoder weights
        d_output: Output dimension (default: 768)
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        pooling: str = "mean",
        freeze: bool = True,
        d_output: int = 768,
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.d_output = d_output

        # Load pre-trained model
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        # Freeze/unfreeze
        if freeze:
            self.freeze()

        # Attention pooling layer
        if pooling == "attention":
            self.attention = nn.Linear(d_output, 1)

    def freeze(self) -> None:
        """Freeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """Extract utterance-level features.

        Args:
            waveforms: List of [T_i] tensors (variable length)

        Returns:
            Tensor [B, d_output]
        """
        batch_size = len(waveforms)
        device = next(self.model.parameters()).device

        # Move waveforms to device
        waveforms = [wf.to(device) for wf in waveforms]

        # Process each waveform
        features = []
        for wf in waveforms:
            # Extract features [1, T, d_output]
            with torch.set_grad_enabled(self.training and not self._is_frozen()):
                outputs = self.model(wf.unsqueeze(0))
                hidden = outputs.last_hidden_state  # [1, T, d_output]

            # Pooling
            if self.pooling == "mean":
                pooled = hidden.mean(dim=1).squeeze(0)  # [d_output]
            elif self.pooling == "attention":
                # Attention pooling
                attn_weights = torch.softmax(
                    self.attention(hidden).squeeze(-1), dim=1
                )  # [1, T]
                pooled = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1).squeeze(
                    0
                )  # [d_output]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            features.append(pooled)

        # Stack batch
        return torch.stack(features)  # [B, d_output]

    def _is_frozen(self) -> bool:
        """Check if model is frozen."""
        return not next(self.model.parameters()).requires_grad
