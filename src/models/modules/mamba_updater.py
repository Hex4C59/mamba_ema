"""Mamba-based updater for EMA state increment (using official implementation)."""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "mamba-ssm is required. Install with: pip install mamba-ssm"
    )


class MambaUpdater(nn.Module):
    """Mamba-based updater for EMA state increment.

    Extracts state increment u_t from current observation z_t using official Mamba blocks.

    Args:
        d_input: Input dimension (d_z = d_speech + d_prosody)
        d_model: Model dimension for Mamba blocks
        d_output: Output state increment dimension
        n_layers: Number of Mamba layers
        d_state: SSM state expansion factor (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Block expansion factor (default: 2)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_input: int = 1088,
        d_model: int = 256,
        d_output: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Official Mamba blocks
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )

        # Layer norms for pre-norm architecture
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        # Output projection with normalization
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_output), nn.LayerNorm(d_output)
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Extract features from observation.

        Args:
            z_t: Current observation [B, d_input] or [B, L, d_input]

        Returns:
            u_t: Output features [B, d_output] or [B, L, d_output]
        """
        # Project input (works for both 2D and 3D)
        x = self.input_proj(z_t)  # [B, d_model] or [B, L, d_model]

        # Handle 2D input (add sequence dimension for Mamba)
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, d_model] → [B, 1, d_model]

        # Pass through Mamba layers with residual connections
        for mamba_layer, norm, dropout in zip(self.mamba_layers, self.norms, self.dropouts):
            # Pre-norm architecture
            residual = x
            x = norm(x)
            x = mamba_layer(x)  # [B, L, d_model]
            x = dropout(x)
            x = x + residual  # Residual connection

        # Remove sequence dimension if input was 2D
        if is_2d:
            x = x.squeeze(1)  # [B, 1, d_model] → [B, d_model]

        # Project to output dimension
        u_t = self.output_proj(x)  # [B, d_output] or [B, L, d_output]

        return u_t
