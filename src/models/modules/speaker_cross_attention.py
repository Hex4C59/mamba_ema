"""Speaker-Aware Cross-Attention module.

Uses Speaker embedding as Query and sequence features as Key/Value.
Speaker queries the sequence to aggregate relevant frame information.
"""

import torch
import torch.nn as nn


class SpeakerAwareCrossAttention(nn.Module):
    """Cross-attention with speaker embedding as Query.

    Speaker embedding queries sequence features to aggregate
    speaker-relevant information from all frames.

    Args:
        d_seq: Sequence feature dimension (K/V)
        d_speaker: Speaker embedding dimension (Q)
        d_hidden: Hidden/output dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_seq: int = 1152,
        d_speaker: int = 192,
        d_hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        self.query_proj = nn.Linear(d_speaker, d_hidden)  # Q: speaker
        self.kv_proj = nn.Linear(d_seq, d_hidden)  # K/V: sequence

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
        self, sequence: torch.Tensor, speaker: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Apply speaker-aware cross-attention.

        Args:
            sequence: [B, T, D_seq] sequence features (K/V)
            speaker: [B, D_speaker] speaker embedding (Q)
            mask: [B, T] padding mask (True = padding, will be ignored)

        Returns:
            output: [B, d_hidden] aggregated representation
        """
        # Q: speaker as single query token [B, 1, d_hidden]
        q = self.query_proj(speaker.unsqueeze(1))

        # K/V: sequence features [B, T, d_hidden]
        kv = self.kv_proj(sequence)

        # Cross-attention: speaker queries sequence
        # key_padding_mask expects True for positions to ignore
        attn_out, _ = self.multihead_attn(
            query=q, key=kv, value=kv, key_padding_mask=mask
        )

        # Residual + Norm + Output projection
        out = self.layer_norm(attn_out + q)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out.squeeze(1)  # [B, 1, D] -> [B, D]
