"""Checkpoint management for saving and loading model states."""

from pathlib import Path
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    state: Dict,
    save_path: Union[str, Path],
    is_best: bool = False,
) -> None:
    """Save model checkpoint.

    Args:
        state: Dict with keys:
            - epoch: int
            - model_state_dict: OrderedDict
            - optimizer_state_dict: dict
            - best_metric: float
            - config: dict
        save_path: Path to save checkpoint (str or Path)
        is_best: If True, also save as best_model.pth
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, save_path)

    if is_best:
        best_path = save_path.parent / "best_model.pth"
        torch.save(state, best_path)


def load_checkpoint(
    load_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    strict: bool = True,
) -> Dict:
    """Load model checkpoint.

    Args:
        load_path: Path to checkpoint (str or Path)
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        strict: If True, require exact key match

    Returns:
        Dict with checkpoint metadata (epoch, best_metric, config)
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location="cpu")

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
        "config": checkpoint.get("config", {}),
    }
