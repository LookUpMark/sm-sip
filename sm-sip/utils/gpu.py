"""
GPU memory management utilities.
"""

import gc
import torch


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device() -> str:
    """Get the best available device.

    Returns:
        "cuda" if GPU is available, else "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
