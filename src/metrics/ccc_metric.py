from typing import Dict

import numpy as np


class CCCMetric:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:

        self.preds_v = []
        self.preds_a = []
        self.targets_v = []
        self.targets_a = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        # extend 用于将一个可迭代对象中的所有元素逐个添加到列表末尾
        self.preds_v.extend(pred[:, 0].tolist())
        self.preds_a.extend(pred[:, 1].tolist())
        self.targets_v.extend(target[:, 0].tolist())
        self.targets_a.extend(target[:, 1].tolist())

    def compute(self) -> Dict[str, float]:
        ccc_v = self._compute_ccc(np.array(self.preds_v), np.array(self.targets_v))
        ccc_a = self._compute_ccc(np.array(self.preds_a), np.array(self.targets_a))
        ccc_avg = (ccc_v + ccc_a) / 2.0

        return {"ccc_v": ccc_v, "ccc_a": ccc_a, "ccc_avg": ccc_avg}

    def _compute_ccc(self, pred: np.ndarray, target: np.ndarray) -> float:

        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_var = pred.var()
        target_var = target.var()

        covariance = ((pred - pred_mean) * (target - target_mean)).mean()

        numerator = 2.0 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

        return numerator / (denominator + 1e-8)
