"""Feature dataset for offline training with pre-extracted features.

This module loads pre-extracted WavLM, speaker embeddings (X-Vector or CAM++),
and eGeMAPS features instead of extracting them on-the-fly during training.
"""

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_pt(path: Path) -> torch.Tensor:
    """Load .pt file with warning suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
        return torch.load(path, map_location="cpu", weights_only=False)


class FeatureDataset(Dataset):
    """Dataset for loading pre-extracted features.

    Loads WavLM (variable-length), X-Vector (fixed), and eGeMAPS (fixed) features
    from pre-computed .pt files.

    Args:
        label_file: Path to CSV label file (IEMOCAP or CCSEMO format)
        feature_root: Root directory containing feature subdirs (wavlm/, xvector/, egemaps/)
        split: "train", "val", or "test"
        fold: Fold number for cross-validation (IEMOCAP: 1-5)
        normalize_vad: Normalize VA to [0, 1]
        features: List of features to load (default: ["wavlm", "xvector", "egemaps"])
        wavlm_layers: List of WavLM layers to use (e.g., [1,2,3,4] or [12]).
                      Single layer returns [T, D], multiple layers return [L, T, D].
        pitch_root: Root directory for pitch features (.npy files)
    """

    def __init__(
        self,
        label_file: str,
        feature_root: str,
        split: str = "train",
        fold: int = 1,
        normalize_vad: bool = True,
        features: list[str] = None,
        wavlm_layers: list[int] = None,
        pitch_root: str = None,
    ):
        self.label_file = label_file
        self.feature_root = Path(feature_root)
        self.split = split
        self.fold = fold
        self.normalize_vad = normalize_vad
        self.features = features or ["wavlm", "xvector", "egemaps"]
        self.wavlm_layers = wavlm_layers or [12]  # 默认单层兼容
        self.pitch_root = Path(pitch_root) if pitch_root else None

        # Detect dataset type from label file path
        if "IEMOCAP" in label_file:
            self.dataset_type = "IEMOCAP"
        elif "CCSEMO" in label_file:
            self.dataset_type = "CCSEMO"
        else:
            self.dataset_type = "IEMOCAP"  # Default

        self.data = self._load_labels()

        # Feature cache for faster loading
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}

        layer_info = f", wavlm_layers={self.wavlm_layers}" if "wavlm" in self.features else ""
        print(f"FeatureDataset: {len(self)} samples ({split}, fold={fold}{layer_info})")

    def _get_test_session(self) -> str:
        """Get test session for IEMOCAP."""
        return f"Session{self.fold}"

    def _get_val_session(self) -> str:
        """Get validation session for IEMOCAP."""
        return f"Session{(self.fold % 5) + 1}"

    def _load_labels(self) -> pd.DataFrame:
        """Load labels with split/fold filtering."""
        if self.dataset_type == "IEMOCAP":
            return self._load_iemocap_labels()
        else:
            return self._load_ccsemo_labels()

    def _load_iemocap_labels(self) -> pd.DataFrame:
        """Load IEMOCAP labels with 5-fold CV split."""
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
        """Load CCSEMO labels with split_set column."""
        df = pd.read_csv(self.label_file)
        # CCSEMO uses split_set column: train, val, test
        df = df[df["split_set"] == self.split].copy()
        df = df.reset_index(drop=True)
        return df

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Load features for a sample.

        Returns:
            dict with:
                - wavlm: Tensor [T, D] (single layer) or [L, T, D] (multi-layer)
                - wavlm_length: int
                - xvector: Tensor [512]
                - egemaps: Tensor [88]
                - valence: float
                - arousal: float
                - name: str
                - session: str (IEMOCAP only)
        """
        row = self.data.iloc[idx]
        name = row["name"]

        # Load features
        result = {"name": name}

        if "wavlm" in self.features:
            # Load from layer-specific directories: wavlm/layer_{n}/
            if len(self.wavlm_layers) == 1:
                # 单层模式：返回 [T, D]
                layer = self.wavlm_layers[0]
                wavlm_path = self.feature_root / "wavlm" / f"layer_{layer}" / f"{name}.pt"
                if wavlm_path.exists():
                    data = _load_pt(wavlm_path)
                    result["wavlm"] = data["features"]  # [T, D]
                    result["wavlm_length"] = data["length"]
                else:
                    raise FileNotFoundError(f"WavLM feature not found: {wavlm_path}")
            else:
                # 多层模式：返回 [L, T, D]
                layer_features = []
                for layer in self.wavlm_layers:
                    wavlm_path = self.feature_root / "wavlm" / f"layer_{layer}" / f"{name}.pt"
                    if wavlm_path.exists():
                        data = _load_pt(wavlm_path)
                        layer_features.append(data["features"])  # [T, D]
                    else:
                        raise FileNotFoundError(f"WavLM feature not found: {wavlm_path}")
                result["wavlm"] = torch.stack(layer_features, dim=0)  # [L, T, D]
                result["wavlm_length"] = layer_features[0].shape[0]

        if "xvector" in self.features:
            xvector_path = self.feature_root / "xvector" / f"{name}.pt"
            if xvector_path.exists():
                result["xvector"] = _load_pt(xvector_path)
            else:
                raise FileNotFoundError(f"X-Vector feature not found: {xvector_path}")

        if "campp" in self.features:
            campp_path = self.feature_root / "campp" / f"{name}.pt"
            if campp_path.exists():
                result["xvector"] = _load_pt(campp_path)  # 复用 xvector 键名，模型兼容
            else:
                raise FileNotFoundError(f"CAM++ feature not found: {campp_path}")

        if "egemaps" in self.features:
            egemaps_path = self.feature_root / "egemaps" / f"{name}.pt"
            if egemaps_path.exists():
                result["egemaps"] = _load_pt(egemaps_path)
            else:
                raise FileNotFoundError(f"eGeMAPS feature not found: {egemaps_path}")

        if "pitch" in self.features:
            if self.pitch_root is None:
                raise ValueError("pitch_root must be set when loading pitch features")
            # Remove .wav extension if present in name
            stem = name.replace(".wav", "")
            pitch_path = self.pitch_root / f"{stem}.npy"
            if pitch_path.exists():
                pitch_data = np.load(pitch_path)  # [T_pitch] 1D array
                result["pitch"] = torch.from_numpy(pitch_data).float()
                result["pitch_length"] = len(pitch_data)
            else:
                raise FileNotFoundError(f"Pitch feature not found: {pitch_path}")

        # Load labels
        valence = float(row["V"])
        arousal = float(row["A"])

        if self.normalize_vad:
            if self.dataset_type == "IEMOCAP":
                valence = (valence - 1.0) / 4.0  # [1, 5] -> [0, 1]
                arousal = (arousal - 1.0) / 4.0
            else:  # CCSEMO
                valence = (valence + 2.0) / 4.0  # [-2, 2] -> [0, 1]
                arousal = (arousal - 1.0) / 4.0  # [1, 5] -> [0, 1]

        result["valence"] = valence
        result["arousal"] = arousal

        if "session" in row:
            result["session"] = row["session"]
        else:
            result["session"] = "unknown"

        return result
