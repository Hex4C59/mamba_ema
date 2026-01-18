"""Extract eGeMAPS LLDs (Low-Level Descriptors) for MS-Mamba.

LLDs provide frame-level prosodic features (23+ dim) instead of global functionals (88 dim).

Key LLDs for emotion:
- Loudness_sma3: Energy/loudness contour
- F0semitoneFrom27.5Hz_sma3nz: Pitch (semitones)
- jitterLocal_sma3nz: Voice quality
- shimmerLocaldB_sma3nz: Voice quality

Usage:
    uv run python scripts/extract_egemaps_llds.py \
        --label_file data/labels/IEMOCAP/iemocap_label.csv \
        --audio_root /path/to/IEMOCAP_full_release \
        --output_dir data/features/IEMOCAP/egemaps_lld
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

try:
    import opensmile
except ImportError:
    raise ImportError("opensmile not installed. Run: uv add opensmile")


# Key LLD features for Sobel edge detection
LOUDNESS_COL = "Loudness_sma3"
PITCH_COL = "F0semitoneFrom27.5Hz_sma3nz"


def extract_egemaps_llds(
    label_file: str, audio_root: str, output_dir: str, overwrite: bool = False
) -> None:
    """Extract eGeMAPS LLD features for all audio files.

    Args:
        label_file: Path to IEMOCAP label CSV
        audio_root: Root directory of audio files
        output_dir: Output directory for .pt files
        overwrite: If True, overwrite existing features
    """
    df = pd.read_csv(label_file)
    print(f"Loaded {len(df)} samples from {label_file}")

    df["audio_path"] = df["audio_path"].str.replace(
        "/tmp/IEMOCAP_full_release", audio_root, regex=False
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # LLDs mode: frame-level features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting eGeMAPS LLDs"):
        audio_path = Path(row["audio_path"])
        name = row["name"]
        output_file = output_path / f"{name}.pt"

        if output_file.exists() and not overwrite:
            skip_count += 1
            continue

        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            error_count += 1
            continue

        try:
            # Extract LLDs: DataFrame with shape [T, num_features]
            llds_df = smile.process_file(str(audio_path))
            llds = torch.from_numpy(llds_df.values).float()  # [T, D]

            columns = list(llds_df.columns)
            loudness_idx = columns.index(LOUDNESS_COL) if LOUDNESS_COL in columns else 0
            pitch_idx = columns.index(PITCH_COL) if PITCH_COL in columns else 1

            result = {
                "features": llds,             # [T, D] frame-level features
                "length": llds.shape[0],
                "loudness_idx": loudness_idx,
                "pitch_idx": pitch_idx,
                "columns": columns,
                "d_feature": llds.shape[1],
            }

            torch.save(result, output_file)
            success_count += 1

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            error_count += 1
            continue

    print(f"\nExtraction complete!")
    print(f"  Success: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_path}")

    # Print feature info from first sample
    sample_files = list(output_path.glob("*.pt"))
    if sample_files:
        sample = torch.load(sample_files[0])
        print(f"\nFeature info:")
        print(f"  Dimension: {sample['d_feature']}")
        print(f"  Sample length: {sample['length']} frames")
        print(f"  Loudness index: {sample['loudness_idx']}")
        print(f"  Pitch index: {sample['pitch_idx']}")
        print(f"  Columns: {sample['columns'][:5]}...")


def main():
    parser = argparse.ArgumentParser(description="Extract eGeMAPS LLD features")
    parser.add_argument(
        "--label_file", type=str, required=True, help="Path to IEMOCAP label CSV",
    )
    parser.add_argument(
        "--audio_root", type=str, required=True, help="Root directory of audio files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for .pt files",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features")

    args = parser.parse_args()
    extract_egemaps_llds(args.label_file, args.audio_root, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
