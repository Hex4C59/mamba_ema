"""Extract eGeMAPS prosody features from IEMOCAP audio files.

This script extracts 88-dimensional eGeMAPS features using openSMILE
and caches them as .pt files for fast loading during training.

Usage:
    uv run python scripts/extract_egemaps.py \
        --label_file data/labels/IEMOCAP/iemocap_label.csv \
        --audio_root /tmp/IEMOCAP_full_release \
        --output_dir data/features/IEMOCAP/egemaps
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

try:
    import opensmile
except ImportError:
    raise ImportError("opensmile not installed. Run: uv add opensmile")


def extract_egemaps(
    label_file: str, audio_root: str, output_dir: str, overwrite: bool = False
) -> None:
    """Extract eGeMAPS features for all audio files.

    Args:
        label_file: Path to IEMOCAP label CSV
        audio_root: Root directory of audio files
        output_dir: Output directory for .pt files
        overwrite: If True, overwrite existing features
    """
    # Load labels
    df = pd.read_csv(label_file)
    print(f"Loaded {len(df)} samples from {label_file}")

    # Replace audio path prefix
    df["audio_path"] = df["audio_path"].str.replace(
        "/tmp/IEMOCAP_full_release", audio_root, regex=False
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize openSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Extract features
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting eGeMAPS"):
        audio_path = Path(row["audio_path"])
        name = row["name"]
        output_file = output_path / f"{name}.pt"

        # Skip if already exists
        if output_file.exists() and not overwrite:
            skip_count += 1
            continue

        # Check audio file exists
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            error_count += 1
            continue

        try:
            # Extract features (returns DataFrame with 1 row)
            features_df = smile.process_file(str(audio_path))
            features = features_df.values.flatten()  # Shape: (88,)

            # Save as tensor
            features_tensor = torch.from_numpy(features).float()
            torch.save(features_tensor, output_file)
            success_count += 1

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            error_count += 1
            continue

    # Summary
    print(f"\nExtraction complete!")
    print(f"  Success: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract eGeMAPS features")
    parser.add_argument(
        "--label_file",
        type=str,
        required=True,
        help="Path to IEMOCAP label CSV",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        required=True,
        help="Root directory of audio files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features",
    )

    args = parser.parse_args()
    extract_egemaps(args.label_file, args.audio_root, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
