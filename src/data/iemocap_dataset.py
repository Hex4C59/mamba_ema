"""IEMOCAP dataset for emotion recognition with VA regression.

This module provides a PyTorch Dataset for loading IEMOCAP audio and labels
with support for 5-fold cross-validation.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class IEMOCAPDataset(Dataset):
    """IEMOCAP emotion recognition dataset.

    Loads audio waveforms and Valence/Arousal labels from IEMOCAP.
    Supports 5-fold cross-validation by session.

    Args:
        label_file: Path to CSV label file
        audio_root: Root directory of audio files
        split: "train" or "test"
        fold: Fold number (1-5), Session{fold} used as test set
        sample_rate: Target sample rate (default: 16000)
        normalize_vad: If True, normalize VAD to [0, 1] (default: True)
    """

    def __init__(
        self,
        label_file: str,
        audio_root: str,
        split: str = "train",
        fold: int = 1,
        sample_rate: int = 16000,
        normalize_vad: bool = True,
    ):
        self.label_file = label_file
        self.audio_root = Path(audio_root)
        self.split = split
        self.fold = fold
        self.sample_rate = sample_rate
        self.normalize_vad = normalize_vad

        # Load and preprocess labels
        self.data = self._load_labels()

        print(
            f"Loaded {len(self)} samples "
            f"(split={split}, fold={fold}, session={self._get_test_session()})"
        )

    def _get_test_session(self) -> str:
        """Get test session name for current fold."""
        return f"Session{self.fold}"

    def _load_labels(self) -> pd.DataFrame:
        """Load and filter labels based on split and fold."""
        # Read CSV
        df = pd.read_csv(self.label_file)

        # Replace audio path prefix
        df["audio_path"] = df["audio_path"].str.replace(
            "/tmp/IEMOCAP_full_release", str(self.audio_root), regex=False
        )

        # 5-fold cross-validation: Session{fold} as test
        test_session = self._get_test_session()
        if self.split == "train":
            df = df[df["session"] != test_session].copy()
        elif self.split == "test":
            df = df[df["session"] == test_session].copy()
        else:
            raise ValueError(f"Invalid split: {self.split}. Use 'train' or 'test'.")

        # Sort by (session, dialog, start_time) for temporal ordering
        df = df.sort_values(["session", "dialog", "start_time"]).reset_index(
            drop=True
        )

        return df

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Load audio and labels for a single sample.

        Returns:
            dict with keys:
                - waveform: Tensor [T], 16kHz mono audio
                - valence: float, [0, 1] if normalized, else [1, 5]
                - arousal: float, [0, 1] if normalized, else [1, 5]
                - name: str, sample identifier
                - session: str, session name
        """
        row = self.data.iloc[idx]

        # Load audio
        audio_path = Path(row["audio_path"])
        waveform, sr = torchaudio.load(str(audio_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # [T]

        # Extract VAD labels
        valence = float(row["V"])
        arousal = float(row["A"])

        # Normalize to [0, 1] (original range: 1-5)
        if self.normalize_vad:
            valence = (valence - 1.0) / 4.0
            arousal = (arousal - 1.0) / 4.0

        return {
            "waveform": waveform,
            "valence": valence,
            "arousal": arousal,
            "name": row["name"],
            "session": row["session"],
        }
