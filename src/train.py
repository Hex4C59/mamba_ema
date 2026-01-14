"""Training script for Mamba + EMA emotion recognition model."""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_yaml, apply_overrides
from utils.seed import set_seed
from utils.experiment import init_experiment
from utils.checkpoint import save_checkpoint, load_checkpoint
from data.iemocap_dataset import IEMOCAPDataset
from data.collate import collate_fn_baseline
from models.mamba_ema_model import MambaEMAModel
from losses.combined_loss import CombinedLoss
from metrics.ccc_metric import CCCMetric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mamba+EMA model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")

    return parser.parse_args()


def build_dataloader(config: dict, split: str) -> DataLoader:
    """Build dataloader from config."""
    dataset = IEMOCAPDataset(
        label_file=config["data"]["params"]["label_file"],
        audio_root=config["data"]["params"]["audio_root"],
        split=split,
        fold=config["data"]["params"]["fold"],
        sample_rate=config["data"]["params"]["sample_rate"],
        normalize_vad=config["data"]["params"]["normalize_vad"],
    )

    loader = DataLoader(
        dataset,
        batch_size=config["data"]["loader"]["batch_size"],
        shuffle=config["data"]["loader"]["shuffle"] if split == "train" else False,
        num_workers=config["data"]["loader"]["num_workers"],
        collate_fn=collate_fn_baseline,
    )

    return loader


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer, loss_fn, device: str, logger
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        # Move batch to device
        batch["valence"] = batch["valence"].to(device)
        batch["arousal"] = batch["arousal"].to(device)

        # Forward
        optimizer.zero_grad()
        output = model(batch)
        loss_dict = loss_fn(
            output["valence_pred"],
            output["arousal_pred"],
            batch["valence"],
            batch["arousal"],
        )

        # Backward
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_dict["loss"].item()

    return {"loss": total_loss / len(loader)}


def validate(
    model: nn.Module, loader: DataLoader, loss_fn, metric, device: str
) -> dict:
    """Validate model."""
    model.eval()
    metric.reset()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch["valence"] = batch["valence"].to(device)
            batch["arousal"] = batch["arousal"].to(device)

            output = model(batch)

            # Compute loss
            loss_dict = loss_fn(
                output["valence_pred"],
                output["arousal_pred"],
                batch["valence"],
                batch["arousal"],
            )
            total_loss += loss_dict["loss"].item()

            # Update metric
            pred = (
                torch.stack([output["valence_pred"], output["arousal_pred"]], dim=-1)
                .cpu()
                .numpy()
            )
            target = (
                torch.stack([batch["valence"], batch["arousal"]], dim=-1)
                .cpu()
                .numpy()
            )
            metric.update(pred, target)

    metrics = metric.compute()
    metrics["loss"] = total_loss / len(loader)

    return metrics


def main():
    """Main training loop."""
    args = parse_args()

    # Load config
    config = load_yaml(args.config)
    config = apply_overrides(config, args.overrides)

    # Initialize experiment
    exp_dir, logger = init_experiment(config)

    # Set seed
    set_seed(config["train"]["seed"])

    # Device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.log_text(f"Using device: {device}")

    # Build model
    model = MambaEMAModel(**config["model"]["params"])
    model = model.to(device)

    # Build dataloaders
    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="test")  # Use test set as val

    # Loss and metric
    loss_fn = CombinedLoss(**config["loss"]["params"])
    metric = CCCMetric()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["optimizer"]["lr"],
        weight_decay=config["train"]["optimizer"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train"]["epochs"]
    )

    # Training loop
    best_metric = 0.0
    for epoch in range(config["train"]["epochs"]):
        logger.log_text(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, logger
        )
        logger.log(train_metrics, epoch, "train")
        logger.log_text(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, metric, device)
        logger.log(val_metrics, epoch, "val")
        logger.log_text(
            f"Val - CCC-V: {val_metrics['ccc_v']:.3f}, "
            f"CCC-A: {val_metrics['ccc_a']:.3f}, "
            f"CCC-Avg: {val_metrics['ccc_avg']:.3f}"
        )

        # Save checkpoint
        if val_metrics["ccc_avg"] > best_metric:
            best_metric = val_metrics["ccc_avg"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                },
                save_path=Path(exp_dir) / f"epoch_{epoch+1}.pth",
                is_best=True,
            )
            logger.log_text(f"Saved best model (CCC-Avg: {best_metric:.3f})")

        scheduler.step()

    # Plot curves
    logger.plot_curves()
    logger.log_text(f"\nTraining complete! Best CCC-Avg: {best_metric:.3f}")


if __name__ == "__main__":
    main()
