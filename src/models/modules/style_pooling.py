"""Style Pooling Layer for emotion recognition.

Based on the architecture:
    LinearNorm (×2) → Conv1D GLU (×2) → Multi-Head Attention → LinearNorm → Temporal Average Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    """LayerNorm followed by Linear projection."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class Conv1dGLU(nn.Module):
    """1D Convolution with Gated Linear Unit activation."""

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Output 2x channels for GLU (split into value and gate)
        self.conv = nn.Conv1d(d_model, d_model * 2, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, D, T] for conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)  # [B, 2D, T]
        x = F.glu(x, dim=1)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        return self.dropout(x)


class StylePoolingLayer(nn.Module):
    """Style Pooling Layer for temporal aggregation with attention.

    Architecture:
        LinearNorm (×2) → Conv1D GLU (×2) → Multi-Head Attention →
        LinearNorm → Temporal Average Pooling → Output Linear

    Args:
        d_input: Input feature dimension (from Mamba/Transformer output)
        d_model: Internal dimension (default 1024 as in the figure)
        n_heads: Number of attention heads
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        n_outputs: Number of outputs (2 for V/A prediction)
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 1024,
        n_heads: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_outputs: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()

        # LinearNorm ×2 (dashed box indicates repetition)
        self.linear_norms_pre = nn.ModuleList([
            LinearNorm(d_model, d_model) for _ in range(2)
        ])

        # Conv1D GLU ×2 (dashed box indicates repetition)
        self.conv_glu_layers = nn.ModuleList([
            Conv1dGLU(d_model, kernel_size, dropout) for _ in range(2)
        ])

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(d_model)

        # LinearNorm after attention
        self.linear_norm_post = LinearNorm(d_model, d_model)

        # Output projection (after temporal average pooling)
        self.output_proj = nn.Linear(d_model, n_outputs)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, d_input]
            mask: Padding mask [B, T], True = padding position

        Returns:
            Output tensor [B, n_outputs]
        """
        # Input projection
        x = self.input_proj(x)  # [B, T, d_model]

        # LinearNorm ×2
        for ln in self.linear_norms_pre:
            x = x + ln(x)  # Residual connection

        # Conv1D GLU ×2
        for conv_glu in self.conv_glu_layers:
            x = x + conv_glu(x)  # Residual connection

        # Multi-Head Attention (self-attention)
        residual = x
        x_normed = self.mha_norm(x)
        attn_out, _ = self.mha(x_normed, x_normed, x_normed, key_padding_mask=mask)
        x = residual + self.mha_dropout(attn_out)

        # LinearNorm
        x = self.linear_norm_post(x)  # [B, T, d_model]

        # Temporal Average Pooling (with mask support)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()  # [B, T, 1]
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # [B, d_model]

        # Output projection
        return self.output_proj(x)  # [B, n_outputs]
