"""MS-Mamba: Multi-Source Adaptive Sliding-Mamba for emotion recognition.

This model implements the "Tri-Stream Guided & Adaptive Injection" architecture:
1. WavLM (SSL features) - provides contextual semantic and acoustic representation
2. eGeMAPS LLDs (expert features) - provides explicit acoustic parameters
3. Speaker Embedding - provides personalized baseline for normalization

Key innovations:
- Speaker-Adaptive LayerNorm (SA-LN) for speaker conditioning
- Multi-Channel Sobel Edge Detector for keyframe selection
- Windowed Local Mamba for local temporal modeling
- Sparse Global Mamba for efficient long-range dependencies
- Speaker-Aware Cross-Attention for final calibration
"""

from typing import Dict

import torch
import torch.nn as nn

from .modules.sobel_edge_detector import MultiChannelSobelEdgeDetector
from .modules.sparse_mamba import SparseGlobalMamba
from .modules.speaker_adaptive_ln import SpeakerAdaptiveLayerNorm
from .modules.speaker_cross_attention import SpeakerAwareCrossAttention
from .modules.windowed_mamba import WindowedLocalMamba


class MSMambaModel(nn.Module):
    """MS-Mamba: Multi-Source Adaptive Sliding-Mamba.

    Architecture flow:
    1. Linear projection: WavLM [B,T,1024] -> [B,T,512], LLDs [B,T,23] -> [B,T,512]
    2. SA-LN: Apply speaker-adaptive normalization to both streams
    3. Concat: Fuse features [B,T,1024]
    4. Windowed Local Mamba: Local temporal modeling -> [B,T,256]
    5. Sobel Edge Detection: Select keyframes [B,K] where K << T
    6. Sparse Global Mamba: Process keyframes -> [B,256]
    7. Speaker-Aware Cross-Attention: Calibrate with speaker -> [B,256]
    8. MLP Head: Regression -> [B,2] (Valence, Arousal)

    Args:
        d_speech: WavLM feature dimension (default: 1024)
        d_prosody_in: eGeMAPS LLD dimension (default: 23)
        d_speaker: Speaker embedding dimension (default: 192)
        d_proj: Projection dimension for WavLM and LLDs (default: 512)
        d_hidden: Hidden dimension throughout the model (default: 256)
        use_sa_ln: Whether to use Speaker-Adaptive LayerNorm (default: True)
        sa_ln_hidden: Hidden dimension for SA-LN MLP (default: 128)
        local_mamba: Config dict for Windowed Local Mamba
        sobel: Config dict for Sobel Edge Detector
        global_mamba: Config dict for Sparse Global Mamba
        speaker_attn: Config dict for Speaker-Aware Cross-Attention
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        d_speech: int = 1024,
        d_prosody_in: int = 23,
        d_speaker: int = 192,
        d_proj: int = 512,
        d_hidden: int = 256,
        use_sa_ln: bool = True,
        sa_ln_hidden: int = 128,
        local_mamba: dict = None,
        sobel: dict = None,
        global_mamba: dict = None,
        speaker_attn: dict = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_sa_ln = use_sa_ln
        self.d_proj = d_proj
        self.d_hidden = d_hidden

        # Default configs
        local_mamba = local_mamba or {}
        sobel = sobel or {}
        global_mamba = global_mamba or {}
        speaker_attn = speaker_attn or {}

        # 1. Linear Projections
        self.speech_proj = nn.Sequential(
            nn.Linear(d_speech, d_proj),
            nn.LayerNorm(d_proj),
            nn.Dropout(dropout),
        )
        self.prosody_proj = nn.Sequential(
            nn.Linear(d_prosody_in, d_proj),
            nn.LayerNorm(d_proj),
            nn.Dropout(dropout),
        )

        # 2. Speaker-Adaptive LayerNorm
        if use_sa_ln:
            self.sa_ln_speech = SpeakerAdaptiveLayerNorm(d_proj, d_speaker, sa_ln_hidden)
            self.sa_ln_prosody = SpeakerAdaptiveLayerNorm(d_proj, d_speaker, sa_ln_hidden)

        # Fused dimension after concat
        d_fused = d_proj * 2  # 1024

        # 3. Windowed Local Mamba
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

        # 4. Sobel Edge Detector
        self.sobel_detector = MultiChannelSobelEdgeDetector(
            loudness_weight=sobel.get("loudness_weight", 0.6),
            pitch_weight=sobel.get("pitch_weight", 0.4),
            top_k_ratio=sobel.get("top_k_ratio", 0.1),
            min_keyframes=sobel.get("min_keyframes", 8),
            max_keyframes=sobel.get("max_keyframes", 64),
        )

        # 5. Sparse Global Mamba
        self.global_mamba = SparseGlobalMamba(
            d_input=d_hidden,
            d_model=global_mamba.get("d_model", 256),
            d_output=d_hidden,
            n_layers=global_mamba.get("n_layers", 2),
            d_state=global_mamba.get("d_state", 16),
            d_conv=global_mamba.get("d_conv", 4),
            expand=global_mamba.get("expand", 2),
            dropout=dropout,
        )

        # 6. Speaker-Aware Cross-Attention
        self.speaker_attn = SpeakerAwareCrossAttention(
            d_query=d_hidden,
            d_speaker=d_speaker,
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
                - wavlm: [B, T, 1024] WavLM sequence features
                - wavlm_mask: [B, T] padding mask (True = padding)
                - egemaps_lld: [B, T, D] aligned LLD features
                - ecapa: [B, 192] speaker embedding
                - loudness: [B, T] loudness channel
                - pitch: [B, T] pitch channel

        Returns:
            Dictionary containing:
                - valence_pred: [B] predicted valence
                - arousal_pred: [B] predicted arousal
                - saliency_map: [B, T] saliency for visualization
        """
        device = next(self.parameters()).device

        # Load features
        wavlm = batch["wavlm"].to(device)          # [B, T, 1024]
        wavlm_mask = batch["wavlm_mask"].to(device)  # [B, T]
        egemaps = batch["egemaps_lld"].to(device)  # [B, T, D]
        ecapa = batch["ecapa"].to(device)          # [B, 192]
        loudness = batch["loudness"].to(device)    # [B, T]
        pitch = batch["pitch"].to(device)          # [B, T]

        # 1. Project features
        h_speech = self.speech_proj(wavlm)     # [B, T, d_proj]
        h_prosody = self.prosody_proj(egemaps)  # [B, T, d_proj]

        # 2. Speaker-Adaptive LayerNorm
        if self.use_sa_ln:
            h_speech = self.sa_ln_speech(h_speech, ecapa)
            h_prosody = self.sa_ln_prosody(h_prosody, ecapa)

        # 3. Concatenate features
        h_fused = torch.cat([h_speech, h_prosody], dim=-1)  # [B, T, d_proj*2]

        # 4. Windowed Local Mamba
        h_local = self.local_mamba(h_fused, wavlm_mask)  # [B, T, d_hidden]

        # 5. Sobel Edge Detection for keyframes
        keyframe_indices, saliency, keyframe_mask = self.sobel_detector(
            loudness, pitch, wavlm_mask
        )

        # 6. Sparse Global Mamba
        h_global = self.global_mamba(h_local, keyframe_indices, keyframe_mask)  # [B, d_hidden]

        # 7. Speaker-Aware Cross-Attention
        h_final = self.speaker_attn(h_global, ecapa)  # [B, d_hidden]

        # 8. Regression
        predictions = self.regression_head(h_final)

        return {
            "valence_pred": predictions[:, 0],
            "arousal_pred": predictions[:, 1],
            "saliency_map": saliency,
        }
