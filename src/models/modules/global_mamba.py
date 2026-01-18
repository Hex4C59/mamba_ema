"""Global Mamba for full sequence processing with pooling."""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")


class GlobalMamba(nn.Module):
    """Global Mamba processing all frames with final pooling.

    Processes the full sequence through Mamba layers and pools to
    get a single global representation.

    Args:
        d_input: Input feature dimension
        d_model: Mamba model dimension
        d_output: Output dimension
        n_layers: Number of Mamba layers (default: 2)
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        dropout: Dropout rate (default: 0.1)
        pooling: Pooling method ("mean", "last", "max") (default: "mean")
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
        pooling: str = None,  # None = no pooling, return sequence
    ):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_output
        self.pooling = pooling

        self.input_proj = nn.Linear(d_input, d_model)

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, d_output)

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Process full sequence through Mamba with optional pooling.

        Args:
            features: [B, T, D] sequence features from Local Mamba
            mask: [B, T] padding mask (True = padding)

        Returns:
            If pooling is None: [B, T, d_output] sequence
            Otherwise: [B, d_output] pooled representation
        """
        # Project input
        x = self.input_proj(features)  # [B, T, d_model]

        # Apply Mamba layers with residual connections
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x) + residual

        # Output projection
        x = self.output_proj(x)  # [B, T, d_output]

        # No pooling - return sequence
        if self.pooling is None:
            return x  # [B, T, d_output]

        # Pooling with mask handling
        if self.pooling == "last":
            if mask is not None:
                lengths = (~mask).sum(dim=1) - 1  # [B]
                batch_indices = torch.arange(x.shape[0], device=x.device)
                x = x[batch_indices, lengths]  # [B, d_output]
            else:
                x = x[:, -1, :]  # [B, d_output]

        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
            x = x.max(dim=1)[0]  # [B, d_output]

        elif self.pooling == "mean":
            if mask is not None:
                mask_expanded = (~mask).unsqueeze(-1).float()  # [B, T, 1]
                valid_count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
                x = (x * mask_expanded).sum(dim=1) / valid_count  # [B, d_output]
            else:
                x = x.mean(dim=1)  # [B, d_output]

        return x
