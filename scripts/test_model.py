"""Test script for MambaEMA model forward pass."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.mamba_ema_model import MambaEMAModel


def test_model():
    """Test model initialization and forward pass."""
    print("=" * 60)
    print("Testing MambaEMA Model")
    print("=" * 60)

    # Create model
    model = MambaEMAModel(
        speech_encoder_name="microsoft/wavlm-base-plus",
        d_speech=768,
        prosody_feature_dir="data/features/IEMOCAP/egemaps",
        d_prosody_in=88,
        d_prosody_out=64,
        speaker_encoder_name="speechbrain/spkrec-ecapa-voxceleb",
        d_speaker=192,
        d_hidden=256,
        dropout=0.2,
        use_ema=False,
    )

    print(f"\nModel initialized successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

    # Create dummy batch
    batch_size = 4
    dummy_batch = {
        "waveforms": [
            torch.randn(16000 * 3) for _ in range(batch_size)
        ],  # 3s audio
        "names": [
            f"Ses01F_impro01_F{i:03d}" for i in range(batch_size)
        ],  # dummy names
    }

    print(f"\n{'=' * 60}")
    print("Testing forward pass")
    print("=" * 60)
    print(f"\nInput batch size: {batch_size}")
    print(f"Waveform 0 shape: {dummy_batch['waveforms'][0].shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(dummy_batch)
            print(f"\nOutput:")
            print(f"  valence_pred shape: {output['valence_pred'].shape}")
            print(f"  arousal_pred shape: {output['arousal_pred'].shape}")
            print(f"  valence_pred: {output['valence_pred']}")
            print(f"  arousal_pred: {output['arousal_pred']}")

            print(f"\n{'=' * 60}")
            print("All tests passed!")
            print("=" * 60)
        except Exception as e:
            print(f"\nError during forward pass: {e}")
            print("Note: This test requires eGeMAPS features to be extracted first.")
            print("Run: uv run python scripts/extract_egemaps.py ...")
            raise


if __name__ == "__main__":
    test_model()
