"""Debug script to check model output distribution."""

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.iemocap_dataset import IEMOCAPDataset
from src.data.collate import collate_fn_baseline
from src.models.mamba_ema_model import MambaEMAModel


def main():
    # Load config from latest experiment
    config_path = "runs/baseline_iemocap_2026-01-14_08-27-39/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset = IEMOCAPDataset(
        label_file=config["data"]["params"]["label_file"],
        audio_root=config["data"]["params"]["audio_root"],
        split="train",
        fold=config["data"]["params"]["fold"],
        sample_rate=config["data"]["params"]["sample_rate"],
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_baseline)

    # Create model
    model = MambaEMAModel(**config["model"]["params"])
    model.eval()

    # Test on one batch
    batch = next(iter(loader))
    print(f"Batch size: {len(batch['waveforms'])}")
    print(f"Target valence: min={batch['valence'].min():.4f}, max={batch['valence'].max():.4f}, mean={batch['valence'].mean():.4f}")
    print(f"Target arousal: min={batch['arousal'].min():.4f}, max={batch['arousal'].max():.4f}, mean={batch['arousal'].mean():.4f}")

    with torch.no_grad():
        output = model(batch)

    print(f"\nModel output:")
    print(f"Valence pred: min={output['valence_pred'].min():.4f}, max={output['valence_pred'].max():.4f}, mean={output['valence_pred'].mean():.4f}, std={output['valence_pred'].std():.4f}")
    print(f"Arousal pred: min={output['arousal_pred'].min():.4f}, max={output['arousal_pred'].max():.4f}, mean={output['arousal_pred'].mean():.4f}, std={output['arousal_pred'].std():.4f}")
    print(f"\nFirst 10 predictions:")
    for i in range(min(10, len(output['valence_pred']))):
        print(f"  Sample {i}: V_pred={output['valence_pred'][i]:.4f}, A_pred={output['arousal_pred'][i]:.4f}, V_true={batch['valence'][i]:.4f}, A_true={batch['arousal'][i]:.4f}")


if __name__ == "__main__":
    main()
