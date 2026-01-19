"""Speaker encoder using pre-trained X-Vector."""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference import EncoderClassifier


class SpeakerEncoder(nn.Module):
    """Speaker encoder using pre-trained X-Vector from SpeechBrain."""

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-xvect-voxceleb",
        d_output: int = 512,
        normalize: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.d_output = d_output
        self.normalize = normalize
        self.xvector = None
        self._device = None

        d_xvector = 512
        if d_xvector != d_output:
            self.projection = nn.Linear(d_xvector, d_output)
        else:
            self.projection = None

    def _init_xvector(self, device):
        """Lazy initialization of X-Vector model on target device."""
        if self.xvector is not None:
            return

        model_path = Path(self.model_name)
        if not model_path.is_absolute():
            model_path = Path("pretrained_model") / self.model_name.split("/")[-1]

        device_str = "cuda" if "cuda" in str(device) else "cpu"
        self.xvector = EncoderClassifier.from_hparams(
            source=self.model_name if "/" in self.model_name else str(model_path),
            savedir=str(model_path) if model_path.exists() else None,
            run_opts={"device": device_str},
        )
        self.xvector.eval()
        self._device = device

    def forward(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """Extract speaker embeddings with batch processing."""
        if self.projection is not None:
            device = next(self.parameters()).device
        else:
            device = waveforms[0].device

        self._init_xvector(device)

        self.xvector.eval()

        max_len = max(wf.size(0) for wf in waveforms)
        batch_size = len(waveforms)

        padded_waveforms = torch.zeros(batch_size, max_len, device=device)
        wav_lens = torch.zeros(batch_size, device=device)

        for i, wf in enumerate(waveforms):
            wf = wf.to(device)
            length = wf.size(0)
            padded_waveforms[i, :length] = wf
            wav_lens[i] = length / max_len

        with torch.no_grad():
            embeddings = self.xvector.encode_batch(padded_waveforms, wav_lens)

            while embeddings.dim() > 2:
                embeddings = embeddings.squeeze(1)

        if embeddings.device != device:
            embeddings = embeddings.to(device)

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)

        return embeddings


class OfflineSpeakerEncoder(nn.Module):
    """Lightweight encoder for pre-extracted X-Vector embeddings.

    Processes pre-extracted [512] speaker embeddings with optional:
    - Linear projection to different output dimension
    - L2 normalization
    """

    def __init__(
        self,
        d_input: int = 512,
        d_output: int = 512,
        normalize: bool = True,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.normalize = normalize

        if d_input != d_output:
            self.projection = nn.Linear(d_input, d_output)
        else:
            self.projection = None

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Process pre-extracted speaker embeddings.

        Args:
            embeddings: [B, 512] pre-extracted X-Vector embeddings

        Returns:
            embeddings: [B, d_output] processed embeddings
        """
        if self.projection is not None:
            embeddings = self.projection(embeddings)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)

        return embeddings
