import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient loss.
    Formula: CCC = 2 * ρ * σ_y * σ_ŷ / (σ_y^2 + σ_ŷ^2 + (μ_y - μ_ŷ)^2)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_var = pred.var(unbiased=False)
        target_var = target.var(unbiased=False)

        covariance = ((pred - pred_mean) * (target - target_mean)).mean()

        # 分子和分母
        numerator = 2.0 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

        # eps 防止除零
        ccc = numerator / (denominator + self.eps)

        return 1.0 - ccc
