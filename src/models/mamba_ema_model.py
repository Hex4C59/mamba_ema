"""Mamba + EMA model for emotion recognition (Stage 1: stateless baseline)."""

from typing import Dict, List
import torch
import torch.nn as nn
from .encoders.speech_encoder import SpeechEncoder
from .encoders.prosody_encoder import ProsodyEncoder
from .encoders.speaker_encoder import SpeakerEncoder
from .modules.film import FiLM


class MambaEMAModel(nn.Module):
    """Mamba + EMA model for VA regression.

    Stage 1 (baseline): Stateless version without EMA/Mamba.
    Forward: ŷ_t = Head(z_t) where z_t = [h'_t; p̃_t]

    Args:
        speech_encoder_name: HuggingFace model name for speech encoder
        d_speech: Speech encoder output dimension
        prosody_feature_dir: Directory with cached eGeMAPS features
        d_prosody_in: Input prosody dimension (88 for eGeMAPS)
        d_prosody_out: Output prosody dimension
        speaker_encoder_name: Model name for speaker encoder
        d_speaker: Speaker embedding dimension
        d_hidden: Hidden dimension for regression head
        dropout: Dropout rate
        use_ema: If True, use EMA state (Stage 2+)
    """

    def __init__(
        self,
        speech_encoder_name: str = "microsoft/wavlm-base-plus",
        d_speech: int = 768,
        prosody_feature_dir: str = "data/features/IEMOCAP/egemaps",
        d_prosody_in: int = 88,
        d_prosody_out: int = 64,
        speaker_encoder_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        d_speaker: int = 192,
        d_hidden: int = 256,
        dropout: float = 0.2,
        use_ema: bool = False,
    ):
        super().__init__()
        self.use_ema = use_ema

        # Feature extractors
        self.speech_encoder = SpeechEncoder(
            model_name=speech_encoder_name,
            pooling="mean",
            freeze=True,
            d_output=d_speech,
        )

        self.prosody_encoder = ProsodyEncoder(
            feature_dir=prosody_feature_dir,
            d_input=d_prosody_in,
            d_output=d_prosody_out,
        )

        self.speaker_encoder = SpeakerEncoder(
            model_name=speaker_encoder_name,
            d_output=d_speaker,
            normalize=True,
        )

        # FiLM modulation
        self.film = FiLM(
            speaker_dim=d_speaker, feat_dim=d_speech, hidden_dim=256
        )

        # Regression head
        d_fused = d_speech + d_prosody_out  # 768 + 64 = 832
        self.regression_head = nn.Sequential(
            nn.Linear(d_fused, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate heads for V and A
        self.head_v = nn.Linear(d_hidden // 2, 1)
        self.head_a = nn.Linear(d_hidden // 2, 1)

    def forward(self, batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Dict with keys:
                - waveforms: List[Tensor] [B] of [T_i]
                - names: List[str] [B]

        Returns:
            Dict with:
                - valence_pred: Tensor [B]
                - arousal_pred: Tensor [B]
        """
        waveforms = batch["waveforms"]
        names = batch["names"]

        # Extract features
        h = self.speech_encoder(waveforms)  # [B, 768]
        p = self.prosody_encoder(names)  # [B, 64]
        s = self.speaker_encoder(waveforms)  # [B, 192]

        # FiLM modulation
        h_mod = self.film(h, s)  # [B, 768]

        # Fuse features
        z = torch.cat([h_mod, p], dim=-1)  # [B, 832]

        # Regression
        shared = self.regression_head(z)  # [B, d_hidden//2]

        valence = torch.sigmoid(self.head_v(shared)).squeeze(-1)  # [B]
        arousal = torch.sigmoid(self.head_a(shared)).squeeze(-1)  # [B]

        return {"valence_pred": valence, "arousal_pred": arousal}
