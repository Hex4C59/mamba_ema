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
    Supports 5-fold cross-validation by session with train/val/test split.

    Args:
        label_file: Path to CSV label file
        audio_root: Root directory of audio files
        split: "train", "val", or "test"
        fold: Fold number (1-5)
            - Session{fold} used as test set
            - Session{(fold % 5) + 1} used as validation set
            - Remaining 3 sessions used as training set
        sample_rate: Target sample rate (default: 16000)
        normalize_vad: If True, normalize VAD to [0, 1] (default: True)
        max_duration: If specified, truncate audio to this duration in seconds (default: None)
    """

    def __init__(
        self,
        label_file: str,
        audio_root: str,
        split: str = "train",
        fold: int = 1,
        sample_rate: int = 16000,
        normalize_vad: bool = True,
        max_duration: float = None,
    ):
        self.label_file = label_file
        self.audio_root = Path(audio_root)
        self.split = split
        self.fold = fold
        self.sample_rate = sample_rate
        self.normalize_vad = normalize_vad
        self.max_duration = max_duration

        # Load and preprocess labels
        self.data = self._load_labels()

        print(
            f"Loaded {len(self)} samples "
            f"(split={split}, fold={fold}, "
            f"test={self._get_test_session()}, val={self._get_val_session()})"
        )

    def _get_test_session(self) -> str:
        """Get test session name for current fold."""
        return f"Session{self.fold}"

    def _get_val_session(self) -> str:
        """Get validation session name for current fold.

        Validation session is selected from remaining sessions after excluding test.
        Use the next session in circular order from the remaining 4 sessions.
        """
        # Map fold to validation session (from remaining 4 sessions)
        # Fold 1: test=S1, remaining=[S2,S3,S4,S5], val=S2
        # Fold 2: test=S2, remaining=[S1,S3,S4,S5], val=S3
        # Fold 3: test=S3, remaining=[S1,S2,S4,S5], val=S4
        # Fold 4: test=S4, remaining=[S1,S2,S3,S5], val=S5
        # Fold 5: test=S5, remaining=[S1,S2,S3,S4], val=S1
        val_session_num = (self.fold % 5) + 1
        return f"Session{val_session_num}"

    def _load_labels(self) -> pd.DataFrame:
        """Load and filter labels based on split and fold.

        Split logic:
        - test: Load from pre-split fold{N}.csv (contains test session only)
        - train/val: Load from iemocap_label.csv, exclude test session samples,
                     then split remaining 4 sessions into train (3) and val (1)
        """
        if self.split == "test":
            # Load pre-split test set from fold{N}.csv
            label_dir = Path(self.label_file).parent
            fold_file = label_dir / f"fold{self.fold}.csv"

            if not fold_file.exists():
                raise FileNotFoundError(
                    f"Fold file not found: {fold_file}\n"
                    f"Please ensure fold{self.fold}.csv exists in {label_dir}"
                )

            df = pd.read_csv(fold_file)
        else:
            # Load full dataset and split train/val from remaining sessions
            df = pd.read_csv(self.label_file)

            # Load test set to get test session samples (for exclusion)
            label_dir = Path(self.label_file).parent
            fold_file = label_dir / f"fold{self.fold}.csv"

            if fold_file.exists():
                # Exclude test samples by name (more robust than session-based filtering)
                test_df = pd.read_csv(fold_file)
                test_names = set(test_df["name"].tolist())
                df = df[~df["name"].isin(test_names)].copy()
            else:
                # Fallback: exclude by test session
                test_session = self._get_test_session()
                df = df[df["session"] != test_session].copy()

            # Split remaining data into train/val
            val_session = self._get_val_session()

            if self.split == "train":
                # Use 3 sessions for training (exclude val)
                df = df[df["session"] != val_session].copy()
            elif self.split == "val":
                # Use 1 session for validation
                df = df[df["session"] == val_session].copy()
            else:
                raise ValueError(
                    f"Invalid split: {self.split}. Use 'train', 'val', or 'test'."
                )

        # Replace audio path prefix
        df["audio_path"] = df["audio_path"].str.replace(
            "/tmp/IEMOCAP_full_release", str(self.audio_root), regex=False
        )

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

        # Truncate to max_duration if specified
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.sample_rate)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]

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
