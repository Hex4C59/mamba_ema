"""Mamba + EMA model for emotion recognition."""

from typing import Dict

import torch
import torch.nn as nn

from .encoders.prosody_encoder import ProsodyEncoder
from .encoders.speaker_encoder import SpeakerEncoder
from .encoders.speech_encoder import SpeechEncoder
from .modules.ema import EMAState
from .modules.film import FiLM
from .modules.mamba_updater import MambaUpdater


class MambaEMAModel(nn.Module):
    def __init__(
        self,
        speech_encoder_name: str = "pretrained_model/wav2vec2-base",
        d_speech: int = 768,
        speech_encoder_layers: list = None,  # Multi-layer extraction
        speech_encoder_pooling: str = "mean",  # Pooling method: "mean" or "attention"
        prosody_feature_dir: str = "data/features/IEMOCAP/egemaps",
        d_prosody_in: int = 88,
        d_prosody_out: int = 64,
        speaker_encoder_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        d_speaker: int = 192,
        d_hidden: int = 256,
        dropout: float = 0.2,
        use_ema: bool = False,
        d_state: int = 64,
        ema_alpha: float = 0.8,
        mamba_d_model: int = 256,
        mamba_n_layers: int = 2,
        freeze_speech_encoder: bool = True,  # 控制是否冻结 speech encoder
    ) -> None:
        super().__init__()
        self.use_ema = use_ema
        self.d_state = d_state

        self.speech_encoder = SpeechEncoder(
            model_name=speech_encoder_name,
            pooling=speech_encoder_pooling,
            freeze=freeze_speech_encoder,
            d_output=d_speech,
            extract_layers=speech_encoder_layers,
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
        self.film = FiLM(speaker_dim=d_speaker, feat_dim=d_speech, hidden_dim=256)

        d_fused = d_speech + d_prosody_out  # e.g., 1024 + 64 = 1088 for WavLM-Large
        if use_ema:
            self.mamba_updater = MambaUpdater(
                d_input=d_fused,
                d_model=mamba_d_model,
                d_output=d_state,
                n_layers=mamba_n_layers,
                dropout=dropout,
            )
            self.ema = EMAState(d_state=d_state, alpha=ema_alpha)

        # Feature fusion with separate attention for Valence and Arousal
        d_head_input = d_fused + d_state if use_ema else d_fused

        # Valence head: prosody-weighted fusion
        # Valence (正负情绪) 更依赖韵律特征
        self.valence_attention = nn.Sequential(
            nn.Linear(d_head_input, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 2),  # [prosody weight, other weight]
            nn.Softmax(dim=-1)
        )

        self.valence_head = nn.Sequential(
            nn.Linear(d_head_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),  # Output [0, 1]
        )

        # Arousal head: speech-weighted fusion
        # Arousal (激活度) 更依赖语音能量
        self.arousal_attention = nn.Sequential(
            nn.Linear(d_head_input, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 2),  # [speech weight, other weight]
            nn.Softmax(dim=-1)
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(d_head_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),  # Output [0, 1]
        )

        # Save dimensions for feature split
        self.d_speech = d_speech
        self.d_prosody_out = d_prosody_out

    def forward(
        self, batch: Dict[str, any], state: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        waveforms = batch["waveforms"]
        names = batch["names"]

        h = self.speech_encoder(waveforms, names)  # [B, 1024]
        p = self.prosody_encoder(names)  # [B, 64]
        s = self.speaker_encoder(waveforms)  # [B, 192]

        h_mod = self.film(h, s)  # [B, 1024]
        z = torch.cat([h_mod, p], dim=-1)  # [B, 1088]

        # Update EMA state if enabled
        if self.use_ema:
            u_t = self.mamba_updater(z)  # [B, d_state]
            c_t = self.ema(u_t, state)  # [B, d_state]

            head_input = torch.cat([z, c_t], dim=-1)  # [B, 1088+d_state]
        else:
            head_input = z
            c_t = None

        # Split features for separate weighting
        # head_input = [h_mod (speech), p (prosody), c_t (ema_state)]
        speech_feat = head_input[:, :self.d_speech]  # [B, 1024]
        prosody_feat = head_input[:, self.d_speech:self.d_speech + self.d_prosody_out]  # [B, 64]
        if self.use_ema:
            ema_feat = head_input[:, self.d_speech + self.d_prosody_out:]  # [B, d_state]

        # Valence: emphasize prosody features
        # Compute attention weights over prosody vs. other features
        valence_weights = self.valence_attention(head_input)  # [B, 2]
        w_prosody_v = valence_weights[:, 0:1]  # [B, 1]
        w_other_v = valence_weights[:, 1:2]  # [B, 1]

        # Weighted features
        weighted_prosody_v = prosody_feat * w_prosody_v  # [B, 64]
        weighted_speech_v = speech_feat * w_other_v  # [B, 1024]

        if self.use_ema:
            weighted_ema_v = ema_feat * w_other_v  # [B, d_state]
            valence_input = torch.cat([weighted_speech_v, weighted_prosody_v, weighted_ema_v], dim=-1)
        else:
            valence_input = torch.cat([weighted_speech_v, weighted_prosody_v], dim=-1)

        # Arousal: emphasize speech features
        arousal_weights = self.arousal_attention(head_input)  # [B, 2]
        w_speech_a = arousal_weights[:, 0:1]  # [B, 1]
        w_other_a = arousal_weights[:, 1:2]  # [B, 1]

        # Weighted features
        weighted_speech_a = speech_feat * w_speech_a  # [B, 1024]
        weighted_prosody_a = prosody_feat * w_other_a  # [B, 64]

        if self.use_ema:
            weighted_ema_a = ema_feat * w_other_a  # [B, d_state]
            arousal_input = torch.cat([weighted_speech_a, weighted_prosody_a, weighted_ema_a], dim=-1)
        else:
            arousal_input = torch.cat([weighted_speech_a, weighted_prosody_a], dim=-1)

        # Separate regression heads
        valence = self.valence_head(valence_input).squeeze(-1)  # [B]
        arousal = self.arousal_head(arousal_input).squeeze(-1)  # [B]

        result = {"valence_pred": valence, "arousal_pred": arousal}
        if self.use_ema:
            result["state"] = c_t

        return result
