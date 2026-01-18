"""Speaker-Aware Cross-Attention module.

Uses Global Mamba output as Query and Speaker embedding as Key/Value
to calibrate the final representation against speaker baseline.
"""

import torch
import torch.nn as nn


class SpeakerAwareCrossAttention(nn.Module):
    """Cross-attention with speaker embedding as Key/Value.

    Global Mamba output queries speaker embedding to calibrate
    the final representation against speaker baseline.

    This ensures the emotion prediction is relative to the speaker's
    typical speaking patterns, not absolute acoustic values.

    Args:
        d_query: Query dimension (from Global Mamba output)
        d_speaker: Speaker embedding dimension
        d_hidden: Hidden/output dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_query: int = 256,
        d_speaker: int = 192,
        d_hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        self.query_proj = nn.Linear(d_query, d_hidden)
        self.kv_proj = nn.Linear(d_speaker, d_hidden)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        speaker: torch.Tensor,
    ) -> torch.Tensor:
        """Apply speaker-aware cross-attention.

        Args:
            query: [B, D_q] from Global Mamba (emotion representation)
            speaker: [B, D_s] speaker embedding (speaker baseline)

        Returns:
            output: [B, d_hidden] speaker-calibrated representation
        """
        # Expand to sequence dim: [B, D] -> [B, 1, D]
        q = self.query_proj(query.unsqueeze(1))  # [B, 1, d_hidden]

        # Speaker as single K/V token: [B, 1, d_hidden]
        kv = self.kv_proj(speaker.unsqueeze(1))

        # Cross-attention
        attn_out, _ = self.multihead_attn(query=q, key=kv, value=kv)

        # Residual + Norm
        out = self.layer_norm(attn_out + q)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out.squeeze(1)  # [B, d_hidden]
