"""Collate functions for batching variable-length audio samples."""

from typing import Dict, List
import torch


def collate_fn_baseline(batch: List[Dict]) -> Dict[str, any]:
    """Collate function for baseline model (handles variable-length audio).

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict with:
            - waveforms: List[Tensor], [B] tensors of shape [T_i]
            - valence: Tensor [B]
            - arousal: Tensor [B]
            - names: List[str], [B]
            - sessions: List[str], [B]
            - lengths: List[int], [B] audio lengths
    """
    waveforms = [item["waveform"] for item in batch]
    valence = torch.tensor([item["valence"] for item in batch], dtype=torch.float32)
    arousal = torch.tensor([item["arousal"] for item in batch], dtype=torch.float32)
    names = [item["name"] for item in batch]
    sessions = [item["session"] for item in batch]
    lengths = [len(wf) for wf in waveforms]

    return {
        "waveforms": waveforms,
        "valence": valence,
        "arousal": arousal,
        "names": names,
        "sessions": sessions,
        "lengths": lengths,
    }
