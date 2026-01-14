"""CCC metric for evaluation."""

import numpy as np
from typing import Dict


class CCCMetric:
    """Concordance Correlation Coefficient metric.

    Accumulates predictions and targets to compute CCC for V and A.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds_v = []
        self.preds_a = []
        self.targets_v = []
        self.targets_a = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Add batch predictions.

        Args:
            pred: Predictions [B, 2] (valence, arousal)
            target: Targets [B, 2]
        """
        self.preds_v.extend(pred[:, 0].tolist())
        self.preds_a.extend(pred[:, 1].tolist())
        self.targets_v.extend(target[:, 0].tolist())
        self.targets_a.extend(target[:, 1].tolist())

    def compute(self) -> Dict[str, float]:
        """Compute CCC scores.

        Returns:
            Dict with "ccc_v", "ccc_a", "ccc_avg"
        """
        ccc_v = self._compute_ccc(
            np.array(self.preds_v), np.array(self.targets_v)
        )
        ccc_a = self._compute_ccc(
            np.array(self.preds_a), np.array(self.targets_a)
        )
        ccc_avg = (ccc_v + ccc_a) / 2.0

        return {"ccc_v": ccc_v, "ccc_a": ccc_a, "ccc_avg": ccc_avg}

    def _compute_ccc(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute CCC for a single dimension.

        Args:
            pred: Predictions [N]
            target: Targets [N]

        Returns:
            CCC score
        """
        # Mean
        pred_mean = pred.mean()
        target_mean = target.mean()

        # Variance
        pred_var = pred.var()
        target_var = target.var()

        # Covariance
        covariance = ((pred - pred_mean) * (target - target_mean)).mean()

        # CCC
        numerator = 2.0 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

        return numerator / (denominator + 1e-8)
