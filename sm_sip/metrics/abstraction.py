"""
Abstraction metrics: novel n-grams, compression ratio, and abstraction score.

These metrics quantify how much the generated summary rephrases
the source text rather than copying it verbatim.
"""

from typing import List, Tuple


def get_ngrams(text: str, n: int = 3) -> List[Tuple[str, ...]]:
    """Extract n-grams from text.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        List of n-gram tuples.
    """
    words = text.lower().split()
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def compute_abstraction_score(source: str, generated: str, n: int = 3) -> float:
    """Compute abstraction score: 1 - (n-gram overlap with source).

    Higher score = more abstractive (less copying from source).

    Args:
        source: Source text.
        generated: Generated summary.
        n: N-gram size for overlap computation.

    Returns:
        Abstraction score between 0.0 (fully extractive) and 1.0 (fully abstractive).
    """
    source_ngrams = set(get_ngrams(source, n))
    gen_ngrams = get_ngrams(generated, n)

    if not gen_ngrams:
        return 1.0  # Empty summary = no copying

    copied = sum(1 for ng in gen_ngrams if ng in source_ngrams)
    copy_ratio = copied / len(gen_ngrams)

    return 1.0 - copy_ratio


def compute_compression_ratio(source: str, generated: str) -> float:
    """Compute compression ratio (lower = more compressed).

    Args:
        source: Source text.
        generated: Generated summary.

    Returns:
        Ratio of generated length to source length.
    """
    if len(source) == 0:
        return 1.0
    return len(generated) / len(source)


def compute_novel_ngrams(source: str, generated: str, n: int = 2) -> float:
    """Compute percentage of n-grams in generated that are NOT in source.

    Higher = more novel content (more rephrasing).

    Args:
        source: Source text.
        generated: Generated summary.
        n: N-gram size.

    Returns:
        Fraction of novel n-grams (0.0 to 1.0).
    """
    source_ngrams = set(get_ngrams(source, n))
    gen_ngrams = get_ngrams(generated, n)

    if not gen_ngrams:
        return 0.0

    novel = sum(1 for ng in gen_ngrams if ng not in source_ngrams)
    return novel / len(gen_ngrams)
