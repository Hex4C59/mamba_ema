"""Sequence-level cross-attention for pitch-speech interaction."""

import torch
import torch.nn as nn


class PitchCrossAttention(nn.Module):
    """Cross-attention where pitch (Q) attends to speech (K,V) at sequence level.

    Unlike utterance-level cross-attention, this module preserves the sequence
    dimension for downstream Mamba processing.

    Args:
        d_pitch: Pitch feature dimension (default: 1)
        d_speech: Speech feature dimension (h_seq_mod, default: 1024)
        d_hidden: Hidden dimension for attention (default: 1024, keep original dim)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        d_pitch: int = 1,
        d_speech: int = 1024,
        d_hidden: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        self.d_hidden = d_hidden
        self.num_heads = num_heads

        # Project pitch [B, T, 1] -> [B, T, d_hidden]
        self.pitch_proj = nn.Sequential(
            nn.Linear(d_pitch, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Project speech [B, T, d_speech] -> [B, T, d_hidden]
        self.speech_proj = nn.Linear(d_speech, d_hidden)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output layers
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.ffn = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        pitch: torch.Tensor,
        speech: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pitch: [B, T, 1] - Pitch values (aligned to speech frame rate)
            speech: [B, T, d_speech] - FiLM-modulated speech features (h_seq_mod)
            key_padding_mask: [B, T] - True for padding positions

        Returns:
            [B, T, d_hidden] - Cross-attended features for Mamba
        """
        # Project pitch to hidden dim
        q = self.pitch_proj(pitch)  # [B, T, d_hidden]

        # Project speech to hidden dim
        kv = self.speech_proj(speech)  # [B, T, d_hidden]

        # Cross-attention: pitch attends to speech
        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )  # [B, T, d_hidden]

        # Add & Norm (residual from query)
        x = self.layer_norm(attn_out + q)

        # FFN with residual
        x = self.ffn_norm(self.ffn(x) + x)

        return x
