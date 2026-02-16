"""
Reproducibility utilities: deterministic seed setting.

Sets seeds for Python, NumPy, PyTorch, and CUDA to ensure
fully reproducible experiments across runs.
"""

import os
import random

# Default seed used across all SM-SIP experiments
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    """Set deterministic seeds for full reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU + CUDA)
    - CUBLAS (via environment variable)

    Should be called once at the start of each experiment.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Force deterministic CUBLAS operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
