"""Collate functions for pre-extracted feature batching."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def resample_pitch(pitch: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample pitch sequence to target length using interpolation.

    Args:
        pitch: [T_orig] 1D pitch sequence
        target_len: Target sequence length

    Returns:
        [T_target] resampled pitch
    """
    if len(pitch) == target_len:
        return pitch
    if len(pitch) == 0:
        return torch.zeros(target_len)

    # Use F.interpolate for resampling
    pitch_2d = pitch.unsqueeze(0).unsqueeze(0)  # [1, 1, T_orig]
    resampled = F.interpolate(pitch_2d, size=target_len, mode="linear", align_corners=False)
    return resampled.squeeze(0).squeeze(0)  # [T_target]


def collate_fn_features(batch: List[Dict]) -> Dict[str, any]:
    """Collate function for offline feature datasets.

    Handles:
    - WavLM: Variable-length sequences -> padded + mask
    - X-Vector: Fixed [512] -> stacked
    - eGeMAPS: Fixed [88] -> stacked

    Args:
        batch: List of samples from FeatureDataset

    Returns:
        dict with:
            - wavlm: Tensor [B, T_max, D] padded sequences
            - wavlm_mask: Tensor [B, T_max] bool (True = padding)
            - wavlm_lengths: List[int]
            - xvector: Tensor [B, 512]
            - egemaps: Tensor [B, 88]
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
    result["valence"] = torch.tensor([item["valence"] for item in batch], dtype=torch.float32)
    result["arousal"] = torch.tensor([item["arousal"] for item in batch], dtype=torch.float32)

    # WavLM: variable-length, supports both [T, D] and [L, T, D] (multi-layer)
    if "wavlm" in batch[0]:
        lengths = [item["wavlm_length"] for item in batch]
        max_len = max(lengths)

        first_wavlm = batch[0]["wavlm"]
        is_multilayer = first_wavlm.dim() == 3

        if is_multilayer:
            # Multi-layer: [L, T, D] -> [B, L, T_max, D]
            num_layers, _, d_model = first_wavlm.shape
            padded = torch.zeros(len(batch), num_layers, max_len, d_model)
            for i, item in enumerate(batch):
                T = item["wavlm_length"]
                padded[i, :, :T, :] = item["wavlm"]
        else:
            # Single-layer: [T, D] -> [B, T_max, D]
            wavlm_features = [item["wavlm"] for item in batch]
            padded = pad_sequence(wavlm_features, batch_first=True, padding_value=0.0)

        # Create padding mask (True for padding positions)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            if length < max_len:
                mask[i, length:] = True

        result["wavlm"] = padded
        result["wavlm_mask"] = mask
        result["wavlm_lengths"] = lengths

    # X-Vector: fixed [512] -> stacked [B, 512]
    if "xvector" in batch[0]:
        result["xvector"] = torch.stack([item["xvector"] for item in batch])

    # eGeMAPS: fixed [88] -> stacked [B, 88]
    if "egemaps" in batch[0]:
        result["egemaps"] = torch.stack([item["egemaps"] for item in batch])

    # Pitch: variable-length, resample to WavLM frame rate and pad -> [B, T]
    if "pitch" in batch[0]:
        wavlm_lengths = result.get("wavlm_lengths", None)

        pitch_list = []
        for i, item in enumerate(batch):
            pitch = item["pitch"]  # [T_pitch]
            # Resample pitch to WavLM frame rate
            target_len = wavlm_lengths[i] if wavlm_lengths else len(pitch) // 2
            pitch_resampled = resample_pitch(pitch, target_len)
            pitch_list.append(pitch_resampled)  # [T]

        # Pad to max length
        max_len = max(p.size(0) for p in pitch_list)
        padded_pitch = torch.zeros(len(batch), max_len)
        for i, p in enumerate(pitch_list):
            padded_pitch[i, :p.size(0)] = p

        result["pitch"] = padded_pitch  # [B, T]

    return result
