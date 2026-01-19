"""Multimodal emotion recognition model with online/offline feature support."""

from typing import Dict

import torch
import torch.nn as nn

from .encoders.prosody_encoder import ProsodyEncoder
from .encoders.speaker_encoder import OfflineSpeakerEncoder, SpeakerEncoder
from .encoders.speech_encoder import OfflineSpeechEncoder, SpeechEncoder
from .modules.cross_attention import CrossAttention
from .modules.film import FiLM
from .modules.layer_fusion import LearnableLayerFusion
from .modules.mamba_updater import MambaUpdater


class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        # Online mode parameters
        speech_encoder_name: str = "pretrained_model/wav2vec2-base",
        speech_encoder_layers: list = None,
        speech_encoder_pooling: str = "mean",
        prosody_feature_dir: str = "data/features/IEMOCAP/egemaps",
        speaker_encoder_name: str = "speechbrain/spkrec-xvect-voxceleb",
        freeze_speech_encoder: bool = True,
        # Feature dimensions
        d_speech: int = 1024,
        d_prosody_in: int = 88,
        d_prosody_out: int = 64,
        d_speaker: int = 512,
        d_hidden: int = 256,
        dropout: float = 0.2,
        # Fusion modules
        use_cross_attention: bool = False,
        cross_attention_heads: int = 4,
        cross_attention_expand_query: bool = True,
        use_mamba: bool = False,
        mamba_d_model: int = 256,
        mamba_d_output: int = 128,
        mamba_n_layers: int = 2,
        # Offline mode
        use_offline_features: bool = False,
        num_wavlm_layers: int = 4,
        wavlm_layer_index: int = None,  # 固定使用某一层 (0-indexed in stored features)
        # Pitch (sequence-level prosody)
        use_pitch: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.use_mamba = use_mamba
        self.use_offline_features = use_offline_features
        self.use_pitch = use_pitch
        self.wavlm_layer_index = wavlm_layer_index

        # Initialize encoders based on mode
        if use_offline_features:
            # 如果指定了 wavlm_layer_index，则不需要 layer_fusion
            if wavlm_layer_index is not None:
                self.layer_fusion = None
            else:
                self.layer_fusion = LearnableLayerFusion(num_layers=num_wavlm_layers)
            self.speech_encoder = OfflineSpeechEncoder(
                d_input=d_speech, d_output=d_speech, dropout=dropout,
            )
            self.speaker_encoder = OfflineSpeakerEncoder(
                d_input=d_speaker, d_output=d_speaker, normalize=True,
            )

            if use_pitch:
                # Pitch: directly use [B, T, 1], no encoder needed
                self.pitch_encoder = None
                self.prosody_encoder = None
                d_fused = d_speech  # Mamba only processes speech
            else:
                # Original eGeMAPS path
                self.prosody_encoder = nn.Sequential(
                    nn.Linear(d_prosody_in, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, d_prosody_out),
                )
                self.pitch_encoder = None
                d_fused = d_speech + d_prosody_out
        else:
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
            self.pitch_encoder = None
            d_fused = d_speech + d_prosody_out

        self.film = FiLM(speaker_dim=d_speaker, feat_dim=d_speech, hidden_dim=256)

        if use_cross_attention and not use_offline_features:
            self.cross_attention = CrossAttention(
                d_query=d_prosody_out,
                d_kv=d_speech,
                d_hidden=d_hidden,
                num_heads=cross_attention_heads,
                dropout=dropout,
                expand_query=cross_attention_expand_query,
            )
            d_fused = d_hidden
        elif not use_offline_features:
            # Online mode without cross-attention
            self.cross_attention = None
            d_fused = d_speech + d_prosody_out
        else:
            # Offline mode: d_fused already set above
            self.cross_attention = None

        if use_mamba:
            self.mamba = MambaUpdater(
                d_input=d_fused,
                d_model=mamba_d_model,
                d_output=mamba_d_output,
                n_layers=mamba_n_layers,
                dropout=dropout,
            )
            d_final = mamba_d_output
        else:
            self.mamba = None
            d_final = d_fused

        # Add pitch dim (concat after mamba, before pool)
        if use_offline_features and use_pitch:
            d_final = d_final + 1  # pitch is [B, T, 1]

        self.regression_head = nn.Sequential(
            nn.Linear(d_final, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 2),
        )

    def forward(self, batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        if self.use_offline_features:
            return self._forward_offline(batch)
        else:
            return self._forward_online(batch)

    def _forward_online(self, batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """Forward pass with online feature extraction (original behavior)."""
        waveforms = batch["waveforms"]
        names = batch["names"]

        p = self.prosody_encoder(names)
        s = self.speaker_encoder(waveforms)

        device = next(self.parameters()).device
        if s.device != device:
            s = s.to(device)
        if p.device != device:
            p = p.to(device)

        h_seq, _ = self.speech_encoder(waveforms, names, return_sequence=True)
        s_expanded = s.unsqueeze(1).expand(-1, h_seq.size(1), -1)
        h_seq_mod = self.film(h_seq, s_expanded)

        if self.use_cross_attention:
            p_expanded = p.unsqueeze(1).expand(-1, h_seq.size(1), -1)
            z = torch.cat([h_seq_mod, p_expanded], dim=-1)
        else:
            p_expanded = p.unsqueeze(1).expand(-1, h_seq.size(1), -1)
            z = torch.cat([h_seq_mod, p_expanded], dim=-1)

        if self.use_mamba:
            z = self.mamba(z)

        z = z.mean(dim=1)

        predictions = self.regression_head(z)
        valence_pred = predictions[:, 0]
        arousal_pred = predictions[:, 1]

        return {"valence_pred": valence_pred, "arousal_pred": arousal_pred}

    def _forward_offline(self, batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """Forward pass with pre-extracted features."""
        device = next(self.parameters()).device

        # Load pre-extracted features from batch
        wavlm = batch["wavlm"].to(device)  # [B, L, T, D] multi-layer features
        wavlm_mask = batch["wavlm_mask"].to(device)  # [B, T]
        xvector = batch["xvector"].to(device)  # [B, 512]

        # Layer selection: 固定某一层 or 可学习融合
        if self.wavlm_layer_index is not None:
            wavlm_fused = wavlm[:, self.wavlm_layer_index, :, :]  # [B, T, D]
        else:
            wavlm_fused = self.layer_fusion(wavlm)  # [B, T, D]

        # Process features through lightweight encoders
        h_seq, _ = self.speech_encoder(wavlm_fused, wavlm_mask)  # [B, T, D]
        s = self.speaker_encoder(xvector)  # [B, d_speaker]

        # FiLM modulation
        s_expanded = s.unsqueeze(1).expand(-1, h_seq.size(1), -1)
        h_seq_mod = self.film(h_seq, s_expanded)

        # Prepare z for Mamba (pitch is added after pool)
        if self.use_pitch:
            z = h_seq_mod  # [B, T, d_speech]
        else:
            egemaps = batch["egemaps"].to(device)  # [B, 88]
            p = self.prosody_encoder(egemaps)  # [B, d_prosody_out]
            p_expanded = p.unsqueeze(1).expand(-1, h_seq.size(1), -1)
            z = torch.cat([h_seq_mod, p_expanded], dim=-1)

        if self.use_mamba:
            z = self.mamba(z)  # [B, T, 256]

        # Mean pooling mamba output (masked)
        if wavlm_mask is not None:
            mask_expanded = (~wavlm_mask).unsqueeze(-1).float()  # [B, T, 1]
            z_pooled = (z * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            z_pooled = z.mean(dim=1)  # [B, 256]

        # Concat with pitch: [B, 256] + [B, T] -> [B, 256 + T]
        if self.use_pitch:
            pitch = batch["pitch"].to(device)  # [B, T]
            z_final = torch.cat([z_pooled, pitch], dim=-1)  # [B, 256 + T]
        else:
            z_final = z_pooled

        predictions = self.regression_head(z_final)
        valence_pred = predictions[:, 0]
        arousal_pred = predictions[:, 1]

        return {"valence_pred": valence_pred, "arousal_pred": arousal_pred}
