"""Transformer-based updater with positional encoding."""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            x + positional encoding [B, T, D]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerUpdater(nn.Module):
    """Transformer-based updater with positional encoding.

    Drop-in replacement for MambaUpdater with same interface.

    Args:
        d_input: Input dimension
        d_model: Model dimension for Transformer
        d_output: Output dimension
        n_layers: Number of Transformer encoder layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout rate
        max_len: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        d_input: int = 1088,
        d_model: int = 256,
        d_output: int = 64,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output

        d_ff = d_ff or 4 * d_model

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_output), nn.LayerNorm(d_output)
        )

    def forward(
        self, z_t: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Process sequence through Transformer.

        Args:
            z_t: Input [B, T, d_input]
            mask: Padding mask [B, T], True = padding (optional)

        Returns:
            Output [B, T, d_output]
        """
        # Project input
        x = self.input_proj(z_t)  # [B, T, d_model]

        # Handle 2D input
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, d_model] -> [B, 1, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder (mask: True = ignore)
        x = self.transformer(x, src_key_padding_mask=mask)

        # Remove sequence dimension if input was 2D
        if is_2d:
            x = x.squeeze(1)

        # Project to output dimension
        return self.output_proj(x)
