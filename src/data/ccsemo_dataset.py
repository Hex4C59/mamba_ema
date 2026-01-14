"""CCSEMO dataset for emotion recognition with VA regression.

This module provides a PyTorch Dataset for loading CCSEMO audio and labels
with support for 5-fold cross-validation.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class CCSEMODataset(Dataset):
    """CCSEMO (Chinese Conversational Speech Emotion) dataset.

    Loads audio waveforms and Valence/Arousal labels from CCSEMO.
    Supports 5-fold cross-validation by speaker with pre-defined train/val/test split.

    Args:
        label_file: Path to CSV label file (fold*.csv)
        audio_root: Root directory of audio files
        split: "train", "val", or "test"
        sample_rate: Target sample rate (default: 16000)
        normalize_vad: If True, normalize VA to [0, 1] (default: True)
    """

    def __init__(
        self,
        label_file: str,
        audio_root: str,
        split: str = "train",
        sample_rate: int = 16000,
        normalize_vad: bool = True,
    ):
        self.label_file = label_file
        self.audio_root = Path(audio_root)
        self.split = split
        self.sample_rate = sample_rate
        self.normalize_vad = normalize_vad

        # Load and preprocess labels
        self.data = self._load_labels()

        print(f"Loaded {len(self)} samples (split={split}, file={Path(label_file).name})")

    def _load_labels(self) -> pd.DataFrame:
        """Load and filter labels based on split.

        CCSEMO CSV format:
            audio_path,name,V,A,gender,duration,discrete_emotion,split_set,transcript

        Note: VA range is [-5, 5], need to normalize to [0, 1] if normalize_vad=True
        """
        df = pd.read_csv(self.label_file)

        # Filter by split_set
        df = df[df["split_set"] == self.split].copy()

        if len(df) == 0:
            raise ValueError(f"No samples found for split={self.split} in {self.label_file}")

        # Normalize VA from [-5, 5] to [0, 1] if needed
        if self.normalize_vad:
            df["V"] = (df["V"] + 5) / 10  # [-5, 5] -> [0, 1]
            df["A"] = (df["A"] + 5) / 10

        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary containing:
                - waveforms: Audio tensor [1, samples]
                - valence: Valence value (float)
                - arousal: Arousal value (float)
                - names: Audio filename
        """
        row = self.data.iloc[idx]

        # Load audio
        audio_path = self.audio_root / row["name"]
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "waveforms": waveform.squeeze(0),  # [samples]
            "valence": torch.tensor(row["V"], dtype=torch.float32),
            "arousal": torch.tensor(row["A"], dtype=torch.float32),
            "names": row["name"],
        }
