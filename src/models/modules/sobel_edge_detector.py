"""Multi-Channel Sobel Edge Detector for keyframe selection.

Detects temporal edges in prosodic features (Loudness and Pitch)
to identify emotionally salient frames.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiChannelSobelEdgeDetector(nn.Module):
    """1D Sobel edge detector for prosodic features.

    Detects temporal edges (sudden changes) in Loudness and Pitch,
    which often correspond to emotional transitions.

    The Sobel filter highlights frames where prosodic features
    change rapidly, indicating potential emotion boundaries.

    Args:
        loudness_weight: Weight for loudness edges (default: 0.6)
        pitch_weight: Weight for pitch edges (default: 0.4)
        top_k_ratio: Ratio of frames to select as keyframes (default: 0.1)
        min_keyframes: Minimum number of keyframes (default: 8)
        max_keyframes: Maximum number of keyframes (default: 64)
    """

    def __init__(
        self,
        loudness_weight: float = 0.6,
        pitch_weight: float = 0.4,
        top_k_ratio: float = 0.1,
        min_keyframes: int = 8,
        max_keyframes: int = 64,
    ):
        super().__init__()
        self.loudness_weight = loudness_weight
        self.pitch_weight = pitch_weight
        self.top_k_ratio = top_k_ratio
        self.min_keyframes = min_keyframes
        self.max_keyframes = max_keyframes

        # 1D Sobel kernel: [-1, 0, 1] for gradient detection
        sobel_kernel = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3)
        self.register_buffer("sobel_kernel", sobel_kernel)

    def compute_edge_map(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude for a 1D signal.

        Args:
            signal: [B, T] input signal

        Returns:
            edge_map: [B, T] edge magnitude (non-negative)
        """
        # Add channel dim: [B, T] -> [B, 1, T]
        signal = signal.unsqueeze(1)

        # Apply Sobel filter with padding to maintain length
        edges = F.conv1d(signal, self.sobel_kernel, padding=1)  # [B, 1, T]

        # Take absolute value as edge magnitude
        edge_map = edges.abs().squeeze(1)  # [B, T]

        return edge_map

    def forward(
        self,
        loudness: torch.Tensor,
        pitch: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect keyframes based on prosodic edges.

        Args:
            loudness: [B, T] loudness contour
            pitch: [B, T] pitch (F0) contour
            padding_mask: [B, T] True for padding positions

        Returns:
            keyframe_indices: [B, K] indices of selected keyframes (sorted)
            saliency_map: [B, T] combined edge saliency
            keyframe_mask: [B, K] True for padding in keyframes
        """
        B, T = loudness.shape
        device = loudness.device

        # Compute edge maps
        loudness_edges = self.compute_edge_map(loudness)
        pitch_edges = self.compute_edge_map(pitch)

        # Normalize each edge map to [0, 1] per sample
        loudness_max = loudness_edges.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        pitch_max = pitch_edges.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        loudness_edges = loudness_edges / loudness_max
        pitch_edges = pitch_edges / pitch_max

        # Weighted combination
        saliency = (self.loudness_weight * loudness_edges +
                    self.pitch_weight * pitch_edges)

        # Mask out padding positions
        if padding_mask is not None:
            saliency = saliency.masked_fill(padding_mask, -float("inf"))

        # Determine valid lengths per sample
        if padding_mask is not None:
            valid_lengths = (~padding_mask).sum(dim=1)  # [B]
        else:
            valid_lengths = torch.full((B,), T, device=device)

        # Calculate per-sample K
        k_per_sample = (valid_lengths.float() * self.top_k_ratio).long()
        k_per_sample = k_per_sample.clamp(min=self.min_keyframes, max=self.max_keyframes)
        K = k_per_sample.max().item()  # Pad to max K

        # Ensure K doesn't exceed valid length
        K = min(K, T)

        # Select top-K indices per sample
        _, topk_indices = torch.topk(saliency, K, dim=1)  # [B, K]

        # Sort indices to maintain temporal order
        topk_indices, _ = torch.sort(topk_indices, dim=1)

        # Create mask for valid keyframes (some samples may have fewer than K)
        keyframe_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        for i in range(B):
            valid_k = min(k_per_sample[i].item(), K)
            if valid_k < K:
                keyframe_mask[i, valid_k:] = True

        return topk_indices, saliency, keyframe_mask
