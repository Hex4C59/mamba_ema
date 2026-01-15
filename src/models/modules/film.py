import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(
        self, speaker_dim: int = 192, feat_dim: int = 768, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.speaker_dim = speaker_dim
        self.feat_dim = feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feat_dim),  # gamma + beta
        )

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

        params = self.mlp(s)  # [B, 2 * feat_dim]
        gamma, beta = params.chunk(2, dim=-1)  # each [B, feat_dim]

        gamma = torch.tanh(gamma) + 1.0

        return gamma * h + beta
