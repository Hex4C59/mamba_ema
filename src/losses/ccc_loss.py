"""Concordance Correlation Coefficient (CCC) loss function."""

import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient loss.

    CCC measures agreement between predicted and target values.
    Formula: CCC = 2 * ρ * σ_y * σ_ŷ / (σ_y^2 + σ_ŷ^2 + (μ_y - μ_ŷ)^2)

    Returns 1 - CCC for minimization.

    Args:
        eps: Small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CCC loss.

        Args:
            pred: Predictions [B]
            target: Ground truth [B]

        Returns:
            Scalar loss (1 - CCC)
        """
        # Mean
        pred_mean = pred.mean()
        target_mean = target.mean()

        # Variance
        pred_var = pred.var(unbiased=False)
        target_var = target.var(unbiased=False)

        # Covariance
        covariance = ((pred - pred_mean) * (target - target_mean)).mean()

        # CCC
        numerator = 2.0 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

        ccc = numerator / (denominator + self.eps)

        # Return 1 - CCC for minimization
        return 1.0 - ccc
