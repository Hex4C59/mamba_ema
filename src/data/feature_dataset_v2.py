"""Feature dataset for MS-Mamba with frame-level LLD support.

This module loads pre-extracted WavLM, ECAPA-TDNN, and eGeMAPS LLD features,
with time alignment between WavLM and LLDs.
"""

import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_pt(path: Path) -> torch.Tensor:
    """Load .pt file with warning suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
        return torch.load(path, map_location="cpu", weights_only=False)


class FeatureDatasetV2(Dataset):
    """Dataset for MS-Mamba with frame-level LLD support.

    Loads:
    - WavLM: [T, 1024] variable-length sequence features
    - ECAPA-TDNN: [192] fixed speaker embedding
    - eGeMAPS LLDs: [T', D] frame-level prosodic features (D ~ 23)

    LLDs are aligned to WavLM frame rate during collation.

    Args:
        label_file: Path to CSV label file
        feature_root: Root directory containing feature subdirs
        split: "train", "val", or "test"
        fold: Fold number for cross-validation
        normalize_vad: Normalize VA to [0, 1]
    """

    def __init__(
        self,
        label_file: str,
        feature_root: str,
        split: str = "train",
        fold: int = 1,
        normalize_vad: bool = True,
    ):
        self.label_file = label_file
        self.feature_root = Path(feature_root)
        self.split = split
        self.fold = fold
        self.normalize_vad = normalize_vad

        # Detect dataset type from label file path
        self.dataset_type = "IEMOCAP" if "IEMOCAP" in label_file else "CCSEMO"
        self.data = self._load_labels()

        print(f"FeatureDatasetV2: {len(self)} samples ({split}, fold={fold})")

    def _get_test_session(self) -> str:
        return f"Session{self.fold}"

    def _get_val_session(self) -> str:
        return f"Session{(self.fold % 5) + 1}"

    def _load_labels(self) -> pd.DataFrame:
        if self.dataset_type == "IEMOCAP":
            return self._load_iemocap_labels()
        else:
            return self._load_ccsemo_labels()

    def _load_iemocap_labels(self) -> pd.DataFrame:
        label_dir = Path(self.label_file).parent

        if self.split == "test":
            fold_file = label_dir / f"fold{self.fold}.csv"
            if not fold_file.exists():
                raise FileNotFoundError(f"Fold file not found: {fold_file}")
            df = pd.read_csv(fold_file)
        else:
            df = pd.read_csv(self.label_file)
            fold_file = label_dir / f"fold{self.fold}.csv"
            if fold_file.exists():
                test_df = pd.read_csv(fold_file)
                test_names = set(test_df["name"].tolist())
                df = df[~df["name"].isin(test_names)].copy()
            else:
                test_session = self._get_test_session()
                df = df[df["session"] != test_session].copy()

            val_session = self._get_val_session()
            if self.split == "train":
                df = df[df["session"] != val_session].copy()
            elif self.split == "val":
                df = df[df["session"] == val_session].copy()

        df = df.sort_values(["session", "dialog", "start_time"]).reset_index(drop=True)
        return df

    def _load_ccsemo_labels(self) -> pd.DataFrame:
        df = pd.read_csv(self.label_file)
        df = df[df["split_set"] == self.split].copy()
        df = df.reset_index(drop=True)
        return df

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Load features for a sample.

        Returns:
            dict with:
                - wavlm: Tensor [T, 1024]
                - wavlm_length: int
                - ecapa: Tensor [192]
                - egemaps_lld: Tensor [T', D] frame-level LLDs
                - lld_length: int
                - loudness_idx: int (column index for loudness)
                - pitch_idx: int (column index for pitch)
                - valence: float
                - arousal: float
                - name: str
                - session: str
        """
        row = self.data.iloc[idx]
        name = row["name"]
        result = {"name": name}

        # Load WavLM
        wavlm_path = self.feature_root / "wavlm" / f"{name}.pt"
        if wavlm_path.exists():
            data = _load_pt(wavlm_path)
            result["wavlm"] = data["features"]  # [T, D]
            result["wavlm_length"] = data["length"]
        else:
            raise FileNotFoundError(f"WavLM feature not found: {wavlm_path}")

        # Load ECAPA
        ecapa_path = self.feature_root / "ecapa" / f"{name}.pt"
        if ecapa_path.exists():
            result["ecapa"] = _load_pt(ecapa_path)
        else:
            raise FileNotFoundError(f"ECAPA feature not found: {ecapa_path}")

        # Load eGeMAPS LLDs
        lld_path = self.feature_root / "egemaps_lld" / f"{name}.pt"
        if lld_path.exists():
            lld_data = _load_pt(lld_path)
            result["egemaps_lld"] = lld_data["features"]  # [T', D]
            result["lld_length"] = lld_data["length"]
            result["loudness_idx"] = lld_data["loudness_idx"]
            result["pitch_idx"] = lld_data["pitch_idx"]
        else:
            raise FileNotFoundError(f"eGeMAPS LLD feature not found: {lld_path}")

        # Load labels
        valence = float(row["V"])
        arousal = float(row["A"])

        if self.normalize_vad:
            valence = (valence - 1.0) / 4.0
            arousal = (arousal - 1.0) / 4.0

        result["valence"] = valence
        result["arousal"] = arousal
        result["session"] = row.get("session", "unknown")

        return result
