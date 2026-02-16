"""Utility functions for GPU management, I/O, and reproducibility."""

from sm_sip.utils.io import save_results, load_results
from sm_sip.utils.seed import set_seed, DEFAULT_SEED


def clear_gpu_memory():
    """Clear GPU memory cache (lazy import of torch)."""
    from sm_sip.utils.gpu import clear_gpu_memory as _clear
    return _clear()


def get_device() -> str:
    """Get best available device (lazy import of torch)."""
    from sm_sip.utils.gpu import get_device as _get
    return _get()
