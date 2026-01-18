"""Windowed Local Mamba for efficient local temporal modeling.

Splits input sequence into non-overlapping windows, applies Mamba
to each window independently, then concatenates results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")


class WindowedLocalMamba(nn.Module):
    """Windowed Local Mamba for local temporal modeling.

    Processes input in fixed-size windows, which:
    1. Reduces memory usage for long sequences
    2. Focuses on local temporal patterns
    3. Allows parallel processing of windows

    Args:
        d_input: Input feature dimension
        d_model: Mamba model dimension
        d_output: Output dimension
        window_size: Size of each window (default: 32)
        n_layers: Number of Mamba layers (default: 1)
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
        window_size: int = 32,
        n_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.d_output = d_output

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Mamba layers (shared across windows)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_output),
            nn.LayerNorm(d_output),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process input through windowed Mamba.

        Args:
            x: Input features [B, T, D]
            padding_mask: [B, T] True for padding positions (unused but kept for API)

        Returns:
            Output features [B, T, d_output]
        """
        B, T, D = x.shape

        # Project input
        x = self.input_proj(x)  # [B, T, d_model]

        # Pad to multiple of window_size
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # Pad T dimension

        T_padded = x.shape[1]
        num_windows = T_padded // self.window_size

        # Reshape to windows: [B, T, D] -> [B*num_windows, window_size, D]
        x = x.view(B, num_windows, self.window_size, self.d_model)
        x = x.reshape(B * num_windows, self.window_size, self.d_model)

        # Apply Mamba layers with residual connections
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x) + residual

        # Reshape back: [B*num_windows, window_size, D] -> [B, T_padded, D]
        x = x.view(B, num_windows, self.window_size, self.d_model)
        x = x.reshape(B, T_padded, self.d_model)

        # Remove padding
        if pad_len > 0:
            x = x[:, :T, :]

        # Output projection
        x = self.output_proj(x)  # [B, T, d_output]

        return x
