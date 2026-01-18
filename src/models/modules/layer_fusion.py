"""Learnable layer fusion for multi-layer pretrained features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableLayerFusion(nn.Module):
    """Learnable weighted fusion of multi-layer features.

    Takes multi-layer features [B, num_layers, T, D] and produces
    a weighted combination [B, T, D] using learned softmax weights.

    Args:
        num_layers: Number of layers to fuse
        init_weights: Optional initial weights (default: uniform)
    """

    def __init__(self, num_layers: int = 4, init_weights: list[float] = None):
        super().__init__()
        if init_weights is not None:
            self.layer_weights = nn.Parameter(torch.tensor(init_weights))
        else:
            self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse multi-layer features.

        Args:
            x: [B, num_layers, T, D] multi-layer features

        Returns:
            [B, T, D] fused features
        """
        weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]
        # Weighted sum: [B, num_layers, T, D] * [num_layers, 1, 1] -> [B, T, D]
        return (x * weights.view(1, -1, 1, 1)).sum(dim=1)

    def get_weights(self) -> torch.Tensor:
        """Get current softmax weights for visualization."""
        return F.softmax(self.layer_weights, dim=0).detach()


class AttentiveLayerFusion(nn.Module):
    """Attention-based layer fusion that can adapt per-sample.

    Uses a small network to predict layer weights based on
    the global statistics of each layer's features.

    Args:
        num_layers: Number of layers to fuse
        d_feature: Feature dimension for each layer
        d_hidden: Hidden dimension for attention network
    """

    def __init__(self, num_layers: int = 4, d_feature: int = 1024, d_hidden: int = 64):
        super().__init__()
        self.num_layers = num_layers

        # Attention network: takes mean-pooled features, outputs layer weights
        self.attn_net = nn.Sequential(
            nn.Linear(d_feature * num_layers, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, num_layers),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Fuse multi-layer features with attention.

        Args:
            x: [B, num_layers, T, D] multi-layer features
            mask: [B, T] padding mask (True = padding)

        Returns:
            [B, T, D] fused features
        """
        B, L, T, D = x.shape

        # Mean pool each layer (with mask if provided)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(1).unsqueeze(-1).float()  # [B, 1, T, 1]
            x_masked = x * mask_expanded
            layer_means = x_masked.sum(dim=2) / mask_expanded.sum(dim=2).clamp(min=1)  # [B, L, D]
        else:
            layer_means = x.mean(dim=2)  # [B, L, D]

        # Compute attention weights
        layer_means_flat = layer_means.view(B, -1)  # [B, L*D]
        weights = F.softmax(self.attn_net(layer_means_flat), dim=-1)  # [B, L]

        # Weighted sum
        return (x * weights.view(B, L, 1, 1)).sum(dim=1)  # [B, T, D]
