"""MS-Mamba: Multi-Source Adaptive Sliding-Mamba for emotion recognition.

This model implements the "Tri-Stream Guided & Adaptive Injection" architecture:
1. WavLM (SSL features) - provides contextual semantic and acoustic representation
2. eGeMAPS LLDs (expert features) - provides explicit acoustic parameters
3. Speaker Embedding - provides personalized baseline for normalization

Key innovations:
- Speaker-Adaptive LayerNorm (SA-LN) for speaker conditioning
- Windowed Local Mamba for local temporal modeling
- Global Mamba for long-range dependencies (processes all frames)
- Speaker-Aware Cross-Attention for frame-level calibration
- Mean pooling after cross-attention
"""

from typing import Dict

import torch
import torch.nn as nn

from .modules.global_mamba import GlobalMamba
from .modules.layer_fusion import LearnableLayerFusion
from .modules.speaker_adaptive_ln import SpeakerAdaptiveLayerNorm
from .modules.speaker_cross_attention import SpeakerAwareCrossAttention
from .modules.windowed_mamba import WindowedLocalMamba


class MSMambaModel(nn.Module):
    """MS-Mamba: Multi-Source Adaptive Sliding-Mamba.

    Architecture flow:
    1. Layer Fusion: WavLM [B,L,T,1024] -> [B,T,1024] via learnable weights
    2. SA-LN: Apply speaker-adaptive normalization to both streams
    3. Concat: Fuse features [B,T,1024+prosody_proj]
    4. Windowed Local Mamba: Local temporal modeling -> [B,T,d_hidden]
    5. Global Mamba: Process all frames -> [B,T,d_hidden]
    6. Speaker-Aware Cross-Attention: Calibrate each frame -> [B,T,d_hidden]
    7. Mean Pooling: Aggregate frames -> [B,d_hidden]
    8. MLP Head: Regression -> [B,2] (Valence, Arousal)

    Args:
        d_speech: WavLM feature dimension (default: 1024)
        d_prosody_in: eGeMAPS LLD dimension (default: 23)
        d_speaker: Speaker embedding dimension (default: 192)
        d_prosody_proj: Projection dimension for LLDs (default: 128)
        d_hidden: Hidden dimension throughout the model (default: 256)
        num_wavlm_layers: Number of WavLM layers for fusion (default: 4)
        use_sa_ln: Whether to use Speaker-Adaptive LayerNorm (default: True)
        sa_ln_hidden: Hidden dimension for SA-LN MLP (default: 128)
        local_mamba: Config dict for Windowed Local Mamba
        global_mamba: Config dict for Global Mamba
        speaker_attn: Config dict for Speaker-Aware Cross-Attention
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        d_speech: int = 1024,
        d_prosody_in: int = 23,
        d_speaker: int = 192,
        d_prosody_proj: int = 128,
        d_hidden: int = 256,
        num_wavlm_layers: int = 4,
        use_sa_ln: bool = True,
        sa_ln_hidden: int = 128,
        local_mamba: dict = None,
        global_mamba: dict = None,
        speaker_attn: dict = None,
        dropout: float = 0.2,
        **kwargs,  # Ignore unused params like 'sobel'
    ):
        super().__init__()
        self.use_sa_ln = use_sa_ln
        self.d_speech = d_speech
        self.d_hidden = d_hidden

        # Default configs
        local_mamba = local_mamba or {}
        global_mamba = global_mamba or {}
        speaker_attn = speaker_attn or {}

        # 1. Learnable Layer Fusion for WavLM (no dimension reduction)
        self.layer_fusion = LearnableLayerFusion(num_layers=num_wavlm_layers)

        # 2. Prosody projection (small projection for LLDs)
        self.prosody_proj = nn.Sequential(
            nn.Linear(d_prosody_in, d_prosody_proj),
            nn.LayerNorm(d_prosody_proj),
            nn.Dropout(dropout),
        )

        # 3. Speaker-Adaptive LayerNorm
        if use_sa_ln:
            self.sa_ln_speech = SpeakerAdaptiveLayerNorm(d_speech, d_speaker, sa_ln_hidden)
            self.sa_ln_prosody = SpeakerAdaptiveLayerNorm(d_prosody_proj, d_speaker, sa_ln_hidden)

        # Fused dimension after concat: d_speech + d_prosody_proj
        d_fused = d_speech + d_prosody_proj

        # 4. Windowed Local Mamba
        self.local_mamba = WindowedLocalMamba(
            d_input=d_fused,
            d_model=local_mamba.get("d_model", 256),
            d_output=d_hidden,
            window_size=local_mamba.get("window_size", 32),
            n_layers=local_mamba.get("n_layers", 1),
            d_state=local_mamba.get("d_state", 16),
            d_conv=local_mamba.get("d_conv", 4),
            expand=local_mamba.get("expand", 2),
            dropout=dropout,
        )

        # 5. Global Mamba (no pooling - returns sequence)
        self.global_mamba = GlobalMamba(
            d_input=d_hidden,
            d_model=global_mamba.get("d_model", 256),
            d_output=d_hidden,
            n_layers=global_mamba.get("n_layers", 2),
            d_state=global_mamba.get("d_state", 16),
            d_conv=global_mamba.get("d_conv", 4),
            expand=global_mamba.get("expand", 2),
            pooling=None,  # No pooling, return sequence
            dropout=dropout,
        )

        # 6. Speaker-Aware Cross-Attention (Q=speaker, K/V=sequence)
        self.speaker_attn = SpeakerAwareCrossAttention(
            d_seq=d_hidden,  # K/V: sequence features
            d_speaker=d_speaker,  # Q: speaker embedding
            d_hidden=d_hidden,
            num_heads=speaker_attn.get("num_heads", 4),
            dropout=dropout,
        )

        # 7. MLP Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 2),  # [Valence, Arousal]
        )

    def forward(self, batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """Forward pass through MS-Mamba.

        Args:
            batch: Dictionary containing:
                - wavlm: [B, L, T, 1024] multi-layer WavLM features
                - wavlm_mask: [B, T] padding mask (True = padding)
                - egemaps_lld: [B, T, D] aligned LLD features
                - ecapa: [B, 192] speaker embedding

        Returns:
            Dictionary containing:
                - valence_pred: [B] predicted valence
                - arousal_pred: [B] predicted arousal
                - layer_weights: [L] learned layer fusion weights
        """
        device = next(self.parameters()).device

        # Load features
        wavlm = batch["wavlm"].to(device)          # [B, L, T, 1024]
        wavlm_mask = batch["wavlm_mask"].to(device)  # [B, T]
        egemaps = batch["egemaps_lld"].to(device)  # [B, T, D]
        ecapa = batch["ecapa"].to(device)          # [B, 192]

        # 1. Learnable layer fusion (no dimension reduction)
        h_speech = self.layer_fusion(wavlm)  # [B, T, 1024]

        # 2. Project prosody features
        h_prosody = self.prosody_proj(egemaps)  # [B, T, d_prosody_proj]

        # 3. Speaker-Adaptive LayerNorm
        if self.use_sa_ln:
            h_speech = self.sa_ln_speech(h_speech, ecapa)
            h_prosody = self.sa_ln_prosody(h_prosody, ecapa)

        # 4. Concatenate features
        h_fused = torch.cat([h_speech, h_prosody], dim=-1)  # [B, T, 1024+prosody_proj]

        # 5. Windowed Local Mamba
        h_local = self.local_mamba(h_fused, wavlm_mask)  # [B, T, d_hidden]

        # 6. Global Mamba (returns sequence)
        h_global = self.global_mamba(h_local, wavlm_mask)  # [B, T, d_hidden]

        # 7. Speaker-Aware Cross-Attention (Q=speaker, K/V=sequence)
        # Speaker queries sequence to aggregate relevant frames -> [B, d_hidden]
        h_pooled = self.speaker_attn(h_global, ecapa, wavlm_mask)

        # 8. Regression
        predictions = self.regression_head(h_pooled)

        return {
            "valence_pred": predictions[:, 0],
            "arousal_pred": predictions[:, 1],
            "layer_weights": self.layer_fusion.get_weights(),
        }
