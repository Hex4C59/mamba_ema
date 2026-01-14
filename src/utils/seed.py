"""Random seed management for reproducibility.

This module provides utilities to set global random seeds across different
libraries (random, numpy, torch) to ensure reproducible experiments.
"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)

    Note:
        Setting deterministic=True may impact performance but ensures full
        reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
