import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.ccsemo_dataset import CCSEMODataset
from data.collate import collate_fn_baseline
from data.collate_features import collate_fn_features
from data.collate_features_v2 import collate_fn_features_v2
from data.feature_dataset import FeatureDataset
from data.feature_dataset_v2 import FeatureDatasetV2
from data.iemocap_dataset import IEMOCAPDataset
from losses.ccc_loss import CCCLoss
from metrics.ccc_metric import CCCMetric
from models.mamba_ema_model import MultimodalEmotionModel
from models.ms_mamba_model import MSMambaModel
from utils.checkpoint import save_checkpoint
from utils.config import load_config 
from utils.experiment import init_experiment
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mamba+EMA model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--fold", type=int, default=None, help="Fold number (1-5)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    return parser.parse_args()


def build_dataloader(config: dict, split: str) -> DataLoader:
    use_offline = config["data"].get("use_offline_features", False)
    use_lld = config["data"].get("use_lld_features", False)
    dataset_name = config["data"]["name"]

    if use_lld:
        # MS-Mamba mode: use LLD features
        dataset = FeatureDatasetV2(
            label_file=config["data"]["params"]["label_file"],
            feature_root=config["data"]["params"]["feature_root"],
            split=split,
            fold=config["data"]["params"].get("fold", 1),
            normalize_vad=config["data"]["params"].get("normalize_vad", True),
        )
        collate_fn = collate_fn_features_v2
    elif use_offline:
        # Offline mode: load pre-extracted features
        dataset = FeatureDataset(
            label_file=config["data"]["params"]["label_file"],
            feature_root=config["data"]["params"]["feature_root"],
            split=split,
            fold=config["data"]["params"].get("fold", 1),
            normalize_vad=config["data"]["params"].get("normalize_vad", True),
            features=config["data"]["params"].get("features", ["wavlm", "ecapa", "egemaps"]),
        )
        collate_fn = collate_fn_features
    elif dataset_name == "IEMOCAP":
        dataset = IEMOCAPDataset(
            label_file=config["data"]["params"]["label_file"],
            audio_root=config["data"]["params"]["audio_root"],
            split=split,
            fold=config["data"]["params"]["fold"],
            sample_rate=config["data"]["params"]["sample_rate"],
            normalize_vad=config["data"]["params"]["normalize_vad"],
        )
        collate_fn = collate_fn_baseline
    elif dataset_name == "CCSEMO":
        dataset = CCSEMODataset(
            label_file=config["data"]["params"]["label_file"],
            audio_root=config["data"]["params"]["audio_root"],
            split=split,
            sample_rate=config["data"]["params"]["sample_rate"],
            normalize_vad=config["data"]["params"]["normalize_vad"],
        )
        collate_fn = collate_fn_baseline
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=config["data"]["loader"]["batch_size"],
        shuffle=config["data"]["loader"]["shuffle"] if split == "train" else False,
        num_workers=config["data"]["loader"]["num_workers"],
        collate_fn=collate_fn,
    )

    return loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    grad_clip: float,
    accumulation_steps: int = 1,
    valence_weight: float = 1.0,
    arousal_weight: float = 1.0,
    scaler: GradScaler = None,
) -> dict:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch["valence"] = batch["valence"].to(device)
        batch["arousal"] = batch["arousal"].to(device)

        with autocast(enabled=(scaler is not None)):
            output = model(batch)

            loss_v = loss_fn(output["valence_pred"], batch["valence"])
            loss_a = loss_fn(output["arousal_pred"], batch["arousal"])
            loss = (valence_weight * loss_v + arousal_weight * loss_a) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return {"loss": total_loss / len(loader)}


def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, metric: nn.Module, device: str) -> dict:
    model.eval()
    metric.reset()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch["valence"] = batch["valence"].to(device)
            batch["arousal"] = batch["arousal"].to(device)

            output = model(batch)

            loss_v = loss_fn(output["valence_pred"], batch["valence"])
            loss_a = loss_fn(output["arousal_pred"], batch["arousal"])
            loss = loss_v + loss_a
            total_loss += loss.item()

            pred = (
                torch.stack([output["valence_pred"], output["arousal_pred"]], dim=-1).cpu().numpy()
            )
            target = torch.stack([batch["valence"], batch["arousal"]], dim=-1).cpu().numpy()
            metric.update(pred, target)

    metrics = metric.compute()
    metrics["loss"] = total_loss / len(loader)

    return metrics


def main():
    args = parse_args()

    config = load_config(args.config)

    if args.fold is not None:
        config["data"]["params"]["fold"] = args.fold
    if args.gpu is not None:
        config["train"]["gpu_id"] = args.gpu

    exp_dir, logger = init_experiment(config)

    set_seed(config["train"]["seed"])

    gpu_id = config["train"]["gpu_id"]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.log_text(f"Using device: {device}")
    logger.log_text(f"Using Fold: {config['data']['params']['fold']}")

    # 模型
    model_name = config["model"].get("name", "MambaEMA")
    if model_name == "MSMamba":
        model = MSMambaModel(**config["model"]["params"])
    else:
        model = MultimodalEmotionModel(**config["model"]["params"])
    model = model.to(device)

    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="val")
    test_loader = build_dataloader(config, split="test")

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

    loss_params = config.get("loss", {}).get("params", {})
    valence_weight = loss_params.get("valence_weight", 1.0)
    arousal_weight = loss_params.get("arousal_weight", 1.0)
    if valence_weight != 1.0 or arousal_weight != 1.0:
        logger.log_text(f"Using weighted loss: V={valence_weight}, A={arousal_weight}")

    for epoch in range(config["train"]["epochs"]):
        logger.log_text(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            config["train"]["grad_clip"],
            valence_weight,
            arousal_weight,
        )
        logger.log(train_metrics, epoch, "train")
        logger.log_text(f"Train Loss: {train_metrics['loss']:.4f}")

        val_metrics = validate(model, val_loader, loss_fn, metric, device)
        logger.log(val_metrics, epoch, "val")
        logger.log_text(
            f"Val - CCC-V: {val_metrics['ccc_v']:.3f}, "
            f"CCC-A: {val_metrics['ccc_a']:.3f}, "
            f"CCC-Avg: {val_metrics['ccc_avg']:.3f}"
        )

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
            logger.log_text(
                f"Small improvement (+{improvement:.4f}, threshold: {min_delta}). Not saving."
            )
            patience_counter += 1
        else:
            patience_counter += 1
            logger.log_text(f"No improvement. Patience: {patience_counter}/{patience}")

        logger.plot_curves()

        if patience_counter >= patience:
            logger.log_text(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        scheduler.step()

    logger.plot_curves()
    logger.log_text(f"\nTraining complete! Best CCC-Avg: {best_metric:.3f}")

    logger.log_text("\n" + "=" * 50)
    logger.log_text("Evaluating on Test Set...")
    logger.log_text("=" * 50)

    checkpoint = torch.load(
        logger.exp_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint["epoch"]

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

    metric.reset()
    metric.update(all_preds.cpu().numpy(), all_labels.cpu().numpy())
    test_metrics = metric.compute()

    test_ccc_v = test_metrics["ccc_v"]
    test_ccc_a = test_metrics["ccc_a"]
    test_ccc_avg = test_metrics["ccc_avg"]

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
