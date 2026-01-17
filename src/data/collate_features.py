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

    # WavLM: variable-length [T, D] -> padded [B, T_max, D]
    if "wavlm" in batch[0]:
        wavlm_features = [item["wavlm"] for item in batch]  # List of [T_i, D]
        lengths = [item["wavlm_length"] for item in batch]

        # Pad sequences: pad_sequence expects [T, D] tensors, pads along T
        padded = pad_sequence(wavlm_features, batch_first=True, padding_value=0.0)  # [B, T_max, D]

        # Create padding mask (True for padding positions)
        max_len = padded.shape[1]
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
