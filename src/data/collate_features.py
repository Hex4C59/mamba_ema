"""Collate functions for pre-extracted feature batching."""

from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn_features(batch: List[Dict]) -> Dict[str, any]:
    """Collate function for offline feature datasets.

    Handles:
    - WavLM: Variable-length sequences -> padded + mask
    - ECAPA: Fixed [192] -> stacked
    - eGeMAPS: Fixed [88] -> stacked

    Args:
        batch: List of samples from FeatureDataset

    Returns:
        dict with:
            - wavlm: Tensor [B, T_max, D] padded sequences
            - wavlm_mask: Tensor [B, T_max] bool (True = padding)
            - wavlm_lengths: List[int]
            - ecapa: Tensor [B, 192]
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

    # ECAPA: fixed [192] -> stacked [B, 192]
    if "ecapa" in batch[0]:
        result["ecapa"] = torch.stack([item["ecapa"] for item in batch])

    # eGeMAPS: fixed [88] -> stacked [B, 88]
    if "egemaps" in batch[0]:
        result["egemaps"] = torch.stack([item["egemaps"] for item in batch])

    return result
