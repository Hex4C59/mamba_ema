"""Cross-attention module for prosody-speech interaction."""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Cross-attention: Prosody (query) attends to Speech (key/value).

    This allows prosody features (pitch, energy) to actively query
    relevant moments in the speech sequence, learning fine-grained
    cross-modal interactions.

    Args:
        d_query: Dimension of query features (prosody)
        d_kv: Dimension of key/value features (speech)
        d_hidden: Hidden dimension for attention
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_query: int = 64,
        d_kv: int = 1024,
        d_hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        self.d_hidden = d_hidden
        self.num_heads = num_heads

        # Project query (prosody) to hidden dimension
        self.query_proj = nn.Linear(d_query, d_hidden)

        # Project key/value (speech) to hidden dimension
        self.kv_proj = nn.Linear(d_kv, d_hidden)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_hidden)

        # Output projection
        self.out_proj = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: [B, d_query] or [B, T_q, d_query] - Prosody features
            key_value: [B, T_kv, d_kv] - Speech sequence features
            key_padding_mask: [B, T_kv] - True for padding positions

        Returns:
            [B, d_hidden] - Cross-attended features
        """
        # Expand query to sequence if needed
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [B, 1, d_query]

        # Project to hidden dimension
        q = self.query_proj(query)  # [B, T_q, d_hidden]
        kv = self.kv_proj(key_value)  # [B, T_kv, d_hidden]

        # Multi-head cross-attention
        attn_out, attn_weights = self.multihead_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )  # attn_out: [B, T_q, d_hidden]

        # Add & Norm (residual connection)
        attn_out = self.layer_norm(attn_out + q)

        # Output projection
        out = self.out_proj(attn_out)  # [B, T_q, d_hidden]
        out = self.dropout(out)

        # Pool over query dimension (mean pooling)
        out = out.mean(dim=1)  # [B, d_hidden]

        return out
