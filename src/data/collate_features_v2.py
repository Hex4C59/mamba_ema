"""Collate functions for MS-Mamba with LLD alignment.

Handles:
- WavLM: Variable-length -> padded + mask
- eGeMAPS LLDs: Align to WavLM frame rate, extract Loudness/Pitch channels
- ECAPA: Fixed [192] -> stacked
"""

from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def align_lld_to_wavlm(
    lld: torch.Tensor,
    lld_length: int,
    wavlm_length: int,
    lld_hop_ms: float = 10.0,
    wavlm_hop_ms: float = 20.0,
) -> torch.Tensor:
    """Align eGeMAPS LLDs to WavLM frame rate via nearest-neighbor interpolation.

    Args:
        lld: [T_lld, D] frame-level LLD features
        lld_length: Original LLD length
        wavlm_length: Target WavLM length
        lld_hop_ms: LLD frame hop in ms (default 10ms for openSMILE)
        wavlm_hop_ms: WavLM frame hop in ms (default 20ms for 16kHz, stride=320)

    Returns:
        aligned: [T_wavlm, D] aligned LLD features
    """
    T_lld = lld.shape[0]

    # Calculate corresponding LLD indices for each WavLM frame
    ratio = wavlm_hop_ms / lld_hop_ms  # 2.0

    # WavLM frame i corresponds to LLD frame i * ratio
    indices = (torch.arange(wavlm_length, dtype=torch.float32) * ratio).long()
    indices = indices.clamp(max=T_lld - 1)

    aligned = lld[indices]  # [T_wavlm, D]
    return aligned


def collate_fn_features_v2(batch: List[Dict]) -> Dict[str, any]:
    """Collate function for MS-Mamba with LLD support.

    Args:
        batch: List of samples from FeatureDatasetV2

    Returns:
        dict with:
            - wavlm: Tensor [B, L, T_max, D] padded multi-layer sequences
            - wavlm_mask: Tensor [B, T_max] bool (True = padding)
            - ecapa: Tensor [B, 192]
            - egemaps_lld: Tensor [B, T_max, D_lld] aligned LLDs
            - loudness: Tensor [B, T_max] loudness channel
            - pitch: Tensor [B, T_max] pitch channel
            - valence: Tensor [B]
            - arousal: Tensor [B]
            - names: List[str]
            - sessions: List[str]
    """
    result = {}

    # Names and sessions
    result["names"] = [item["name"] for item in batch]
    result["sessions"] = [item.get("session", "unknown") for item in batch]

    # Labels
    result["valence"] = torch.tensor(
        [item["valence"] for item in batch], dtype=torch.float32
    )
    result["arousal"] = torch.tensor(
        [item["arousal"] for item in batch], dtype=torch.float32
    )

    # Get WavLM lengths and max length
    wavlm_lengths = [item["wavlm_length"] for item in batch]
    max_len = max(wavlm_lengths)

    # WavLM: check if multi-layer [L, T, D] or single-layer [T, D]
    first_wavlm = batch[0]["wavlm"]
    is_multilayer = first_wavlm.dim() == 3

    if is_multilayer:
        # Multi-layer: [L, T, D] -> padded [B, L, T_max, D]
        num_layers = first_wavlm.shape[0]
        d_model = first_wavlm.shape[2]

        padded_wavlm = torch.zeros(len(batch), num_layers, max_len, d_model)
        for i, item in enumerate(batch):
            T = item["wavlm_length"]
            padded_wavlm[i, :, :T, :] = item["wavlm"]
    else:
        # Single-layer (old format): [T, D] -> [B, 1, T_max, D] for compatibility
        d_model = first_wavlm.shape[1]
        wavlm_features = [item["wavlm"] for item in batch]
        padded_single = pad_sequence(wavlm_features, batch_first=True, padding_value=0.0)
        padded_wavlm = padded_single.unsqueeze(1)  # [B, 1, T, D]

    # Create padding mask
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(wavlm_lengths):
        if length < max_len:
            mask[i, length:] = True

    result["wavlm"] = padded_wavlm
    result["wavlm_mask"] = mask

    # ECAPA: fixed [192] -> stacked [B, 192]
    result["ecapa"] = torch.stack([item["ecapa"] for item in batch])

    # eGeMAPS LLDs: align to WavLM and pad
    aligned_llds = []
    loudness_list = []
    pitch_list = []

    for i, item in enumerate(batch):
        lld = item["egemaps_lld"]
        lld_length = item["lld_length"]
        wavlm_length = wavlm_lengths[i]
        loudness_idx = item["loudness_idx"]
        pitch_idx = item["pitch_idx"]

        # Align LLD to WavLM frame rate
        aligned = align_lld_to_wavlm(lld, lld_length, wavlm_length)
        aligned_llds.append(aligned)

        # Extract loudness and pitch channels
        loudness_list.append(aligned[:, loudness_idx])
        pitch_list.append(aligned[:, pitch_idx])

    # Pad aligned LLDs
    padded_lld = pad_sequence(aligned_llds, batch_first=True, padding_value=0.0)
    result["egemaps_lld"] = padded_lld

    # Pad loudness and pitch
    padded_loudness = pad_sequence(loudness_list, batch_first=True, padding_value=0.0)
    padded_pitch = pad_sequence(pitch_list, batch_first=True, padding_value=0.0)
    result["loudness"] = padded_loudness
    result["pitch"] = padded_pitch

    return result
