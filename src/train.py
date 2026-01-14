"""Training script for Mamba + EMA emotion recognition model."""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.collate import collate_fn_baseline
from data.iemocap_dataset import IEMOCAPDataset
from data.ccsemo_dataset import CCSEMODataset
from losses.ccc_loss import CCCLoss
from metrics.ccc_metric import CCCMetric
from models.mamba_ema_model import MambaEMAModel
from utils.checkpoint import save_checkpoint
from utils.config import apply_overrides, load_yaml
from utils.experiment import init_experiment
from utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mamba+EMA model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint")
    parser.add_argument("--fold", type=int, default=None, help="Fold number (1-5)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")

    return parser.parse_args()


def build_dataloader(config: dict, split: str) -> DataLoader:
    """Build dataloader from config."""
    dataset_name = config["data"]["name"]

    # Select dataset class based on config
    if dataset_name == "IEMOCAP":
        dataset = IEMOCAPDataset(
            label_file=config["data"]["params"]["label_file"],
            audio_root=config["data"]["params"]["audio_root"],
            split=split,
            fold=config["data"]["params"]["fold"],
            sample_rate=config["data"]["params"]["sample_rate"],
            normalize_vad=config["data"]["params"]["normalize_vad"],
        )
    elif dataset_name == "CCSEMO":
        dataset = CCSEMODataset(
            label_file=config["data"]["params"]["label_file"],
            audio_root=config["data"]["params"]["audio_root"],
            split=split,
            sample_rate=config["data"]["params"]["sample_rate"],
            normalize_vad=config["data"]["params"]["normalize_vad"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=config["data"]["loader"]["batch_size"],
        shuffle=config["data"]["loader"]["shuffle"] if split == "train" else False,
        num_workers=config["data"]["loader"]["num_workers"],
        collate_fn=collate_fn_baseline,
    )

    return loader


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer, loss_fn, device: str, logger, grad_clip: float
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        batch["valence"] = batch["valence"].to(device)
        batch["arousal"] = batch["arousal"].to(device)

        optimizer.zero_grad()
        output = model(batch)

        # CCC loss for both valence and arousal
        loss_v = loss_fn(output["valence_pred"], batch["valence"])
        loss_a = loss_fn(output["arousal_pred"], batch["arousal"])
        loss = loss_v + loss_a

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return {"loss": total_loss / len(loader)}


def validate(model: nn.Module, loader: DataLoader, loss_fn, metric, device: str) -> dict:
    """Validate model."""
    model.eval()
    metric.reset()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch["valence"] = batch["valence"].to(device)
            batch["arousal"] = batch["arousal"].to(device)

            output = model(batch)

            # CCC loss for both valence and arousal
            loss_v = loss_fn(output["valence_pred"], batch["valence"])
            loss_a = loss_fn(output["arousal_pred"], batch["arousal"])
            loss = loss_v + loss_a
            total_loss += loss.item()

            # Update metric
            pred = (
                torch.stack([output["valence_pred"], output["arousal_pred"]], dim=-1).cpu().numpy()
            )
            target = torch.stack([batch["valence"], batch["arousal"]], dim=-1).cpu().numpy()
            metric.update(pred, target)

    metrics = metric.compute()
    metrics["loss"] = total_loss / len(loader)

    return metrics


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)
    config = apply_overrides(config, args.overrides)

    # Apply command-line fold and gpu arguments (higher priority than config)
    if args.fold is not None:
        config["data"]["params"]["fold"] = args.fold
    if args.gpu is not None:
        config["train"]["gpu_id"] = args.gpu

    exp_dir, logger = init_experiment(config)

    set_seed(config["train"]["seed"])

    gpu_id = config["train"]["gpu_id"]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.log_text(f"Using device: {device}")
    logger.log_text(f"Using GPU: {gpu_id}")
    logger.log_text(f"Using Fold: {config['data']['params']['fold']}")

    model = MambaEMAModel(**config["model"]["params"])
    model = model.to(device)

    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="val")
    test_loader = build_dataloader(config, split="test")

    # Use CCC loss for training
    loss_fn = CCCLoss()
    metric = CCCMetric()

    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["optimizer"]["lr"],
        weight_decay=config["train"]["optimizer"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train"]["epochs"]
    )

    best_metric = -float("inf")
    patience = config["train"]["early_stopping"]["patience"]
    min_delta = config["train"]["early_stopping"]["min_delta"]
    patience_counter = 0

    for epoch in range(config["train"]["epochs"]):
        logger.log_text(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, logger, config["train"]["grad_clip"]
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

        # Save checkpoint (only if improvement > min_delta)
        improvement = val_metrics["ccc_avg"] - best_metric
        if improvement > min_delta:
            best_metric = val_metrics["ccc_avg"]
            patience_counter = 0  # Reset patience counter
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                },
                save_path=Path(exp_dir) / "best_model.pth",
                is_best=False,  # Already saving to best_model.pth
            )
            logger.log_text(
                f"Saved best model (CCC-Avg: {best_metric:.3f}, improvement: +{improvement:.4f})"
            )
        elif improvement > 0:
            # Small improvement but not enough to save
            logger.log_text(
                f"Small improvement (+{improvement:.4f}, threshold: {min_delta}). Not saving."
            )
            patience_counter += 1
        else:
            patience_counter += 1
            logger.log_text(f"No improvement. Patience: {patience_counter}/{patience}")

        logger.plot_curves()

        # Early stopping check
        if patience_counter >= patience:
            logger.log_text(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        scheduler.step()

    # Plot curves
    logger.plot_curves()
    logger.log_text(f"\nTraining complete! Best CCC-Avg: {best_metric:.3f}")

    # Evaluate on test set
    logger.log_text("\n" + "=" * 50)
    logger.log_text("Evaluating on Test Set...")
    logger.log_text("=" * 50)

    # Load best model
    checkpoint = torch.load(
        logger.exp_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint["epoch"]  # Extract best epoch

    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in test_loader:
            valence = batch["valence"].to(device)
            arousal = batch["arousal"].to(device)

            outputs = model(batch)
            valence_pred = outputs["valence_pred"]
            arousal_pred = outputs["arousal_pred"]

            preds = torch.stack([valence_pred, arousal_pred], dim=1)  # [B, 2]
            labels = torch.stack([valence, arousal], dim=1)  # [B, 2]

            all_preds.append(preds)
            all_labels.append(labels)
            all_names.extend(batch["names"])

    all_preds = torch.cat(all_preds, dim=0)  # [N, 2]
    all_labels = torch.cat(all_labels, dim=0)  # [N, 2]

    # Compute test metrics (CCC)
    metric.reset()
    metric.update(all_preds.cpu().numpy(), all_labels.cpu().numpy())
    test_metrics = metric.compute()

    test_ccc_v = test_metrics["ccc_v"]
    test_ccc_a = test_metrics["ccc_a"]
    test_ccc_avg = test_metrics["ccc_avg"]

    # Compute MSE
    import numpy as np

    preds_np = all_preds.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    mse_v = np.mean((preds_np[:, 0] - labels_np[:, 0]) ** 2)
    mse_a = np.mean((preds_np[:, 1] - labels_np[:, 1]) ** 2)
    mse_avg = (mse_v + mse_a) / 2.0

    logger.log_text(
        f"\nTest Results:\n"
        f"  CCC-V: {test_ccc_v:.4f}, CCC-A: {test_ccc_a:.4f}, CCC-Avg: {test_ccc_avg:.4f}\n"
        f"  MSE-V: {mse_v:.4f}, MSE-A: {mse_a:.4f}, MSE-Avg: {mse_avg:.4f}"
    )

    # Save best results
    with open(logger.exp_dir / "best_result.txt", "w") as f:
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"ccc_v: {test_ccc_v:.4f}\n")
        f.write(f"ccc_a: {test_ccc_a:.4f}\n")
        f.write(f"ccc_avg: {test_ccc_avg:.4f}\n")
        f.write(f"mse_v: {mse_v:.4f}\n")
        f.write(f"mse_a: {mse_a:.4f}\n")
        f.write(f"mse_avg: {mse_avg:.4f}\n")

    with open(logger.exp_dir / "test_prediction.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "valence_true", "arousal_true", "valence_pred", "arousal_pred"])
        for i, name in enumerate(all_names):
            writer.writerow(
                [
                    name,
                    f"{labels_np[i, 0]:.4f}",
                    f"{labels_np[i, 1]:.4f}",
                    f"{preds_np[i, 0]:.4f}",
                    f"{preds_np[i, 1]:.4f}",
                ]
            )


if __name__ == "__main__":
    main()
