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
        expand_query: If True, reshape prosody features [B, 64] to [B, 64, 1]
                     and project to [B, 64, 1024] to match speech dimension
    """

    def __init__(
        self,
        d_query: int = 64,
        d_kv: int = 1024,
        d_hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        expand_query: bool = True,  # 新增参数
    ):
        super().__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.expand_query = expand_query

        if expand_query:
            # 将prosody的64个特征看作序列，每个特征是1维，投影到1024维
            self.query_expand = nn.Linear(1, d_kv)  # 1 → 1024
            self.query_proj = nn.Linear(d_kv, d_hidden)   # 1024 → 256
        else:
            # 直接投影（原方案）
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
            query: [B, d_query] - Prosody features
            key_value: [B, T_kv, d_kv] - Speech sequence features
            key_padding_mask: [B, T_kv] - True for padding positions

        Returns:
            [B, d_hidden] - Cross-attended features
        """
        B, T = key_value.size(0), key_value.size(1)

        if self.expand_query:
            # 方案: 将prosody特征维度重新解释为序列维度
            # [B, 64] → [B, 64, 1]
            query = query.unsqueeze(-1)

            # [B, 64, 1] → [B, 64, 1024] 对齐到speech维度
            query = self.query_expand(query)

            # [B, 64, 1024] → [B, 64, d_hidden]
            q = self.query_proj(query)
        else:
            # 方案2: 直接扩展并投影（原方案）
            if query.dim() == 2:
                query = query.unsqueeze(1)  # [B, 1, d_query]
            q = self.query_proj(query)  # [B, 1 or T, d_hidden]

        # Project key/value
        kv = self.kv_proj(key_value)  # [B, T_kv, d_hidden]

        # Multi-head cross-attention
        attn_out, attn_weights = self.multihead_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )  # attn_out: [B, T or 1, d_hidden]

        # Add & Norm (residual connection)
        attn_out = self.layer_norm(attn_out + q)

        # Output projection
        out = self.out_proj(attn_out)  # [B, T or 1, d_hidden]
        out = self.dropout(out)

        # Pool over query dimension (mean pooling)
        out = out.mean(dim=1)  # [B, d_hidden]

        return out
