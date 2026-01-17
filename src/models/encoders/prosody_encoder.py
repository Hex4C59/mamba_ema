"""Prosody encoder that loads cached eGeMAPS features."""

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn


class ProsodyEncoder(nn.Module):
    """Prosody encoder using cached eGeMAPS features with in-memory cache."""

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
        self._feature_cache: Dict[str, torch.Tensor] = {}

        self.mlp = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_output),
        )

    def forward(self, names: List[str]) -> torch.Tensor:
        device = next(self.mlp.parameters()).device

        features = []
        for name in names:
            if name not in self._feature_cache:
                feature_path = self.feature_dir / f"{name}.pt"
                if not feature_path.exists():
                    raise FileNotFoundError(f"Feature not found: {feature_path}")
                feature = torch.load(feature_path, map_location="cpu", weights_only=True)
                self._feature_cache[name] = feature

            feature = self._feature_cache[name].to(device)
            features.append(feature)

        features = torch.stack(features)

        return self.mlp(features)
