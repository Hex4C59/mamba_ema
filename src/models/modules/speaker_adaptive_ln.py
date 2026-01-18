"""Speaker-Adaptive LayerNorm (SA-LN) module.

Uses speaker embedding to generate per-sample affine parameters,
similar to AdaIN but applied after LayerNorm.
"""

import torch
import torch.nn as nn


class SpeakerAdaptiveLayerNorm(nn.Module):
    """Speaker-Adaptive Layer Normalization.

    Generates scale (gamma) and bias (beta) from speaker embedding,
    then applies: y = gamma * LayerNorm(x) + beta

    This allows the model to adapt normalization to individual speakers,
    accounting for speaker-specific characteristics like voice pitch,
    loudness baseline, and speaking style.

    Args:
        d_feature: Feature dimension to normalize
        d_speaker: Speaker embedding dimension (default: 192 for ECAPA-TDNN)
        hidden_dim: Hidden dimension for affine parameter generation
    """

    def __init__(
        self,
        d_feature: int,
        d_speaker: int = 192,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.d_feature = d_feature

        # Standard LayerNorm without learnable affine parameters
        self.layer_norm = nn.LayerNorm(d_feature, elementwise_affine=False)

        # Generate gamma and beta from speaker embedding
        self.affine_generator = nn.Sequential(
            nn.Linear(d_speaker, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * d_feature),
        )

        # Initialize close to identity: gamma=1, beta=0
        nn.init.zeros_(self.affine_generator[-1].weight)
        nn.init.zeros_(self.affine_generator[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        speaker: torch.Tensor,
    ) -> torch.Tensor:
        """Apply speaker-adaptive layer normalization.

        Args:
            x: Input features [B, T, D] or [B, D]
            speaker: Speaker embedding [B, d_speaker]

        Returns:
            Normalized and transformed features, same shape as input
        """
        # Generate affine parameters: [B, 2*D]
        params = self.affine_generator(speaker)
        gamma, beta = params.chunk(2, dim=-1)  # each [B, D]

        # Apply LayerNorm (without learnable affine)
        x_norm = self.layer_norm(x)

        # Handle sequence dimension
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)  # [B, 1, D]
            beta = beta.unsqueeze(1)    # [B, 1, D]

        # Affine transform: gamma centered at 1 for stable training
        gamma = 1.0 + torch.tanh(gamma) * 0.5  # range [0.5, 1.5]

        return gamma * x_norm + beta
