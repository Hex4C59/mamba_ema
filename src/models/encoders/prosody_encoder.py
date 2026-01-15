"""Prosody encoder that loads cached eGeMAPS features."""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn


class ProsodyEncoder(nn.Module):
    """Prosody encoder using cached eGeMAPS features.
    """

    def __init__(
        self,
        feature_dir: str,
        d_input: int = 88,
        d_output: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.d_input = d_input
        self.d_output = d_output

        self.mlp = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_output),
        )

    def forward(self, names: List[str]) -> torch.Tensor:

        device = next(self.mlp.parameters()).device

        # Load features
        features = []
        for name in names:
            feature_path = self.feature_dir / f"{name}.pt"
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature not found: {feature_path}")

            feature = torch.load(feature_path, map_location=device, weights_only=True)  # [d_input]
            features.append(feature)

        features = torch.stack(features)  # [B, d_input]

        return self.mlp(features)  # [B, d_output]
