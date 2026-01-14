"""FiLM (Feature-wise Linear Modulation) layer for speaker conditioning."""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """FiLM layer for conditioning features with speaker information.

    Applies affine transformation: h' = gamma * h + beta
    where gamma and beta are predicted from speaker embedding.

    Args:
        speaker_dim: Speaker embedding dimension
        feat_dim: Feature dimension to modulate
        hidden_dim: Hidden dimension for MLP (default: 256)
    """

    def __init__(
        self, speaker_dim: int = 192, feat_dim: int = 768, hidden_dim: int = 256
    ):
        super().__init__()
        self.speaker_dim = speaker_dim
        self.feat_dim = feat_dim

        # MLP to predict gamma and beta
        self.mlp = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feat_dim),  # gamma + beta
        )

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            h: Features to modulate [B, feat_dim]
            s: Speaker embeddings [B, speaker_dim]

        Returns:
            Modulated features [B, feat_dim]
        """
        # Predict gamma and beta
        params = self.mlp(s)  # [B, 2 * feat_dim]
        gamma, beta = params.chunk(2, dim=-1)  # each [B, feat_dim]

        # Stabilize gamma around 1.0
        gamma = torch.tanh(gamma) + 1.0

        # Apply modulation
        return gamma * h + beta
