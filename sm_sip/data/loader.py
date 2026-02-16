"""
Dataset loading functions for WITS (Italian) and ArXiv (English).

Provides unified interface to load test data for inference/evaluation,
with configurable sample sizes and quality filtering.
"""

from typing import List, Dict, Optional
from datasets import load_dataset


def load_wits_dataset(
    split: str = "train",
    num_samples: int = 100,
    skip_samples: int = 25000,
    min_source_len: int = 500,
    max_source_len: int = 10000,
    min_summary_len: int = 50,
) -> List[Dict[str, str]]:
    """Load samples from the WITS (Wikipedia Italian) dataset.

    Args:
        split: Dataset split to load.
        num_samples: Number of samples to return.
        skip_samples: Number of samples to skip (for train/test separation).
        min_source_len: Minimum source text length (chars).
        max_source_len: Maximum source text length (chars).
        min_summary_len: Minimum summary length (chars).

    Returns:
        List of dicts with 'source' and 'reference' keys.
    """
    dataset = load_dataset("silvia-casola/WITS", split=split, streaming=True)
    dataset = dataset.skip(skip_samples)

    samples = []
    for entry in dataset:
        source = entry["source"]
        summary = entry["summary"]

        if len(source) < min_source_len or len(summary) < min_summary_len or len(source) > max_source_len:
            continue

        samples.append({"source": source, "reference": summary})
        if len(samples) >= num_samples:
            break

    print(f"  Loaded {len(samples)} WITS samples (skipped {skip_samples})")
    return samples


def load_arxiv_dataset(
    split: str = "test",
    num_samples: int = 100,
    skip_samples: int = 0,
    min_source_len: int = 500,
    max_source_len: int = 15000,
    min_summary_len: int = 50,
) -> List[Dict[str, str]]:
    """Load samples from the ArXiv summarization dataset.

    Args:
        split: Dataset split to load.
        num_samples: Number of samples to return.
        skip_samples: Number of samples to skip.
        min_source_len: Minimum source text length (chars).
        max_source_len: Maximum source text length (chars).
        min_summary_len: Minimum summary length (chars).

    Returns:
        List of dicts with 'source' and 'reference' keys.
    """
    dataset = load_dataset("ccdv/arxiv-summarization", split=split, streaming=True)
    dataset = dataset.skip(skip_samples)

    samples = []
    for entry in dataset:
        source = entry["article"]
        summary = entry["abstract"]

        if len(source) < min_source_len or len(summary) < min_summary_len or len(source) > max_source_len:
            continue

        samples.append({"source": source, "reference": summary})
        if len(samples) >= num_samples:
            break

    print(f"  Loaded {len(samples)} ArXiv samples (skipped {skip_samples})")
    return samples


def get_test_data(
    lang: str = "it",
    num_samples: int = 100,
    skip_samples: int = 25000,
) -> List[Dict[str, str]]:
    """Load test data based on language.

    Convenience wrapper that dispatches to language-specific loaders.

    Args:
        lang: "it" for WITS, "en" for ArXiv.
        num_samples: Number of test samples.
        skip_samples: Samples to skip for train/test separation.

    Returns:
        List of dicts with 'source' and 'reference' keys.
    """
    print(f"  Loading {lang.upper()} test data (skipping {skip_samples})...")
    if lang == "it":
        return load_wits_dataset(num_samples=num_samples, skip_samples=skip_samples)
    elif lang == "en":
        return load_arxiv_dataset(num_samples=num_samples, skip_samples=skip_samples)
    else:
        raise ValueError(f"Unsupported language: {lang}. Use 'it' or 'en'.")
