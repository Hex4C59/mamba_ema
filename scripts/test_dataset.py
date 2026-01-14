"""Test script for IEMOCAP dataset loading."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.iemocap_dataset import IEMOCAPDataset
from data.collate import collate_fn_baseline
from torch.utils.data import DataLoader


def test_dataset():
    """Test IEMOCAP dataset loading."""
    print("=" * 60)
    print("Testing IEMOCAP Dataset")
    print("=" * 60)

    # Create dataset
    dataset = IEMOCAPDataset(
        label_file="data/labels/IEMOCAP/iemocap_label.csv",
        audio_root="/tmp/IEMOCAP_full_release",
        split="train",
        fold=1,
        sample_rate=16000,
        normalize_vad=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  waveform shape: {sample['waveform'].shape}")
    print(f"  sample rate: 16000 Hz")
    print(f"  duration: {len(sample['waveform']) / 16000:.2f}s")
    print(f"  valence: {sample['valence']:.3f} (normalized)")
    print(f"  arousal: {sample['arousal']:.3f} (normalized)")
    print(f"  name: {sample['name']}")
    print(f"  session: {sample['session']}")

    # Test dataloader
    print(f"\n{'=' * 60}")
    print("Testing DataLoader")
    print("=" * 60)

    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_baseline
    )

    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  batch size: {len(batch['waveforms'])}")
    print(f"  waveform 0 shape: {batch['waveforms'][0].shape}")
    print(f"  waveform 1 shape: {batch['waveforms'][1].shape}")
    print(f"  valence shape: {batch['valence'].shape}")
    print(f"  arousal shape: {batch['arousal'].shape}")
    print(f"  lengths: {batch['lengths'][:4]}")
    print(f"  names: {batch['names'][:2]}")

    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()
