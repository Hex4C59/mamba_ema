import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """MSE loss wrapper for VA regression."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)
