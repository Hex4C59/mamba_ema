"""Speaker encoder using pre-trained ECAPA-TDNN."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    """Speaker encoder using pre-trained ECAPA-TDNN.
   """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        d_output: int = 192,
        normalize: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.d_output = d_output
        self.normalize = normalize

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, d_output)

    def forward(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """Extract speaker embeddings.
        """
        device = next(self.encoder.parameters()).device
        waveforms = [wf.to(device) for wf in waveforms]

        embeddings = []
        for wf in waveforms:
            # Add channel dim and batch dim: [1, 1, T]
            wf_input = wf.unsqueeze(0).unsqueeze(0)

            # Extract features
            features = self.encoder(wf_input)  # [1, 128, 1]
            features = features.squeeze(-1)  # [1, 128]

            # Project to d_output
            emb = self.fc(features).squeeze(0)  # [d_output]
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)  # [B, d_output]

        # L2 normalize
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)

        return embeddings
