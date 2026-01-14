"""Extract eGeMAPS prosody features from CCSEMO audio files.

This script extracts 88-dimensional eGeMAPS features using openSMILE
and caches them as .pt files for fast loading during training.

Since 5-fold CV only changes train/val/test split but uses the same audio files,
we only need to extract features once for all unique audio files.

Usage:
    # Extract features for all unique audio files in CCSEMO
    uv run python scripts/extract_egemaps_ccsemo.py \
        --label_dir data/labels/CCSEMO/5fold \
        --audio_root /tmp/CCSEMO/audio \
        --output_dir data/features/CCSEMO/egemaps
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
    label_dir: str, audio_root: str, output_dir: str, overwrite: bool = False
) -> None:
    """Extract eGeMAPS features for all unique audio files across all folds.

    Args:
        label_dir: Directory containing fold*.csv files
        audio_root: Root directory of audio files
        output_dir: Output directory for .pt files
        overwrite: If True, overwrite existing features
    """
    # Load all fold CSV files and collect unique audio filenames
    label_path = Path(label_dir)
    fold_files = sorted(label_path.glob("fold*.csv"))

    if not fold_files:
        raise FileNotFoundError(f"No fold*.csv files found in {label_dir}")

    print(f"Found {len(fold_files)} fold files: {[f.name for f in fold_files]}")

    # Collect all unique audio filenames
    all_names = set()
    for fold_file in fold_files:
        df = pd.read_csv(fold_file)
        all_names.update(df["name"].tolist())

    print(f"Total unique audio files: {len(all_names)}")

    # CCSEMO uses 'name' field for audio filename
    # Audio path is constructed as: audio_root / name
    audio_root_path = Path(audio_root)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize openSMILE with eGeMAPS v02 (88 features)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Extract features for all unique audio files
    success_count = 0
    skip_count = 0
    error_count = 0

    for name in tqdm(sorted(all_names), desc="Extracting eGeMAPS"):
        audio_path = audio_root_path / name
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
            # Extract features (returns DataFrame with 1 row, 88 columns)
            features_df = smile.process_file(str(audio_path))
            features = features_df.values.flatten()  # Shape: (88,)

            # Verify feature dimension
            if len(features) != 88:
                print(f"Warning: Expected 88 features, got {len(features)} for {name}")
                error_count += 1
                continue

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
    parser = argparse.ArgumentParser(description="Extract eGeMAPS features for CCSEMO")
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing fold*.csv files (e.g., data/labels/CCSEMO/5fold)",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        required=True,
        help="Root directory of audio files (e.g., /tmp/CCSEMO/audio)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features/CCSEMO/egemaps",
        help="Output directory for .pt files (default: data/features/CCSEMO/egemaps)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features",
    )

    args = parser.parse_args()
    extract_egemaps(args.label_dir, args.audio_root, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
