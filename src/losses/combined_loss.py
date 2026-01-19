"""Combined loss (CCC + MSE) for VA regression."""

import torch
import torch.nn as nn
from typing import Dict
from .ccc_loss import CCCLoss


class CombinedLoss(nn.Module):
    """Combined CCC and MSE loss for Valence/Arousal regression.

    支持两种模式：
    - 固定权重: Loss = ccc_weight * (CCC_v + CCC_a)/2 + mse_weight * (MSE_v + MSE_a)
    - 可学习权重 (Uncertainty Weighting): 基于 Kendall et al., 2018

    Args:
        ccc_weight: Weight for CCC term (default: 0.5)
        mse_weight: Weight for MSE term (default: 0.1)
        learnable_weights: Whether to use uncertainty weighting (default: False)
    """

    def __init__(
        self,
        ccc_weight: float = 0.5,
        mse_weight: float = 0.1,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.ccc_weight = ccc_weight
        self.mse_weight = mse_weight
        self.learnable_weights = learnable_weights

        self.ccc_loss = CCCLoss()
        self.mse_loss = nn.MSELoss()

        if learnable_weights:
            # 可学习的 log variance，初始化为 0 (即 σ=1, weight=1)
            self.log_var_v = nn.Parameter(torch.zeros(1))
            self.log_var_a = nn.Parameter(torch.zeros(1))

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
            Dict with keys: "loss", "ccc_v", "ccc_a", "mse", and optionally "weight_v", "weight_a"
        """
        ccc_v = self.ccc_loss(pred_v, target_v)
        ccc_a = self.ccc_loss(pred_a, target_a)
        mse_v = self.mse_loss(pred_v, target_v)
        mse_a = self.mse_loss(pred_a, target_a)
        mse_term = mse_v + mse_a

        if self.learnable_weights:
            # 单任务损失
            loss_v = ccc_v * self.ccc_weight + mse_v * self.mse_weight
            loss_a = ccc_a * self.ccc_weight + mse_a * self.mse_weight

            # Uncertainty weighting: L = 1/(2σ²) * loss + log(σ)
            # 等价于: L = 0.5 * exp(-log_var) * loss + 0.5 * log_var
            weight_v = torch.exp(-self.log_var_v)
            weight_a = torch.exp(-self.log_var_a)

            total_loss = (
                0.5 * weight_v * loss_v + 0.5 * self.log_var_v +
                0.5 * weight_a * loss_a + 0.5 * self.log_var_a
            ).squeeze()

            return {
                "loss": total_loss,
                "ccc_v": ccc_v,
                "ccc_a": ccc_a,
                "mse": mse_term,
                "weight_v": weight_v.squeeze(),
                "weight_a": weight_a.squeeze(),
            }
        else:
            ccc_term = (ccc_v + ccc_a) / 2.0
            total_loss = ccc_term * self.ccc_weight + mse_term * self.mse_weight
            return {"loss": total_loss, "ccc_v": ccc_v, "ccc_a": ccc_a, "mse": mse_term}
