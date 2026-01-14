"""Combined loss (CCC + MSE) for VA regression."""

import torch
import torch.nn as nn
from typing import Dict
from .ccc_loss import CCCLoss


class CombinedLoss(nn.Module):
    """Combined CCC and MSE loss for Valence/Arousal regression.

    Loss = 1 - ccc_weight * (CCC_v + CCC_a) / 2 + mse_weight * (MSE_v + MSE_a)

    Args:
        ccc_weight: Weight for CCC term (default: 0.5)
        mse_weight: Weight for MSE term (default: 0.1)
    """

    def __init__(self, ccc_weight: float = 0.5, mse_weight: float = 0.1):
        super().__init__()
        self.ccc_weight = ccc_weight
        self.mse_weight = mse_weight

        self.ccc_loss = CCCLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred_v: torch.Tensor,
        pred_a: torch.Tensor,
        target_v: torch.Tensor,
        target_a: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred_v: Predicted valence [B]
            pred_a: Predicted arousal [B]
            target_v: Target valence [B]
            target_a: Target arousal [B]

        Returns:
            Dict with keys: "loss", "ccc_v", "ccc_a", "mse"
        """
        # CCC losses (1 - CCC, higher is worse)
        ccc_v = self.ccc_loss(pred_v, target_v)
        ccc_a = self.ccc_loss(pred_a, target_a)

        # MSE losses
        mse_v = self.mse_loss(pred_v, target_v)
        mse_a = self.mse_loss(pred_a, target_a)

        # Combined loss
        ccc_term = (ccc_v + ccc_a) / 2.0
        mse_term = mse_v + mse_a

        total_loss = ccc_term * self.ccc_weight + mse_term * self.mse_weight

        return {
            "loss": total_loss,
            "ccc_v": ccc_v,
            "ccc_a": ccc_a,
            "mse": mse_term,
        }
