"""
Seed and reproducibility utilities for Kinbridge-Sync experiments.

All random state flows through here to guarantee determinism:
    - Python random
    - NumPy random
    - PyTorch random (CPU and CUDA)

Usage:
    from core.seeds import seed_everything
    seed_everything(42)
"""

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42, torch_seed: bool = False) -> None:
    """Set all random seeds for reproducible experiments.

    Args:
        seed: Integer seed value.
        torch_seed: If True, also seed PyTorch (requires torch).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch_seed:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
