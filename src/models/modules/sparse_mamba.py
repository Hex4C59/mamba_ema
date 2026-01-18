"""Sparse Global Mamba operating on keyframes only.

Takes keyframe indices and corresponding features, applies Mamba
to capture long-range dependencies with reduced computation.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")


class SparseGlobalMamba(nn.Module):
    """Sparse Global Mamba for long-range dependency modeling.

    Only processes keyframes selected by Sobel edge detection,
    significantly reducing computation while focusing on
    emotionally salient frames.

    Args:
        d_input: Input feature dimension
        d_model: Mamba model dimension
        d_output: Output dimension
        n_layers: Number of Mamba layers (default: 2)
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        d_output: int = 256,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_output

        self.input_proj = nn.Linear(d_input, d_model)

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, d_output)

    def forward(
        self,
        features: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process keyframes through Mamba.

        Args:
            features: [B, T, D] full sequence features from Local Mamba
            keyframe_indices: [B, K] indices of selected keyframes
            keyframe_mask: [B, K] True for padding in keyframes

        Returns:
            global_features: [B, d_output] aggregated global representation
        """
        B, T, D = features.shape
        K = keyframe_indices.shape[1]

        # Gather keyframe features: [B, K, D]
        indices_expanded = keyframe_indices.unsqueeze(-1).expand(-1, -1, D)
        keyframe_features = torch.gather(features, dim=1, index=indices_expanded)

        # Project input
        x = self.input_proj(keyframe_features)  # [B, K, d_model]

        # Apply Mamba layers with residual connections
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x) + residual

        # Output projection
        x = self.output_proj(x)  # [B, K, d_output]

        # Masked mean pooling
        if keyframe_mask is not None:
            # ~keyframe_mask: True for valid positions
            mask_expanded = (~keyframe_mask).unsqueeze(-1).float()  # [B, K, 1]
            valid_count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
            x = (x * mask_expanded).sum(dim=1) / valid_count  # [B, d_output]
        else:
            x = x.mean(dim=1)  # [B, d_output]

        return x
