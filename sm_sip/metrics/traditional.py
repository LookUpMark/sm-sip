"""
Traditional summarization metrics: BERT-Score, ROUGE, and KIR.

These are the standard automated metrics used for evaluating
summary quality against a reference.
"""

from typing import List, Dict
import numpy as np


def compute_bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "it",
    model_type: str = None,
) -> Dict[str, float]:
    """Compute BERTScore F1 between predictions and references.

    Args:
        predictions: List of generated summaries.
        references: List of reference summaries.
        lang: Language code for BERTScore model selection (ignored if model_type is set).
        model_type: Explicit model name (e.g. 'microsoft/mdeberta-v3-base') to override
                     the default language-based selection. Enables fair cross-lingual comparison.

    Returns:
        Dict with 'mean', 'std', and 'scores' (per-sample F1).
    """
    from bert_score import score as bert_score_fn

    # Force CPU to avoid VRAM contention with LLM on GPU
    if model_type:
        _, _, F1 = bert_score_fn(
            predictions, references, model_type=model_type, verbose=False, device="cpu"
        )
    else:
        _, _, F1 = bert_score_fn(
            predictions, references, lang=lang, verbose=False, device="cpu"
        )
    scores = F1.numpy().tolist()
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
    }


def compute_rouge(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE scores between predictions and references.

    Args:
        predictions: List of generated summaries.
        references: List of reference summaries.
        metrics: ROUGE variants to compute (default: ["rouge1", "rougeL"]).

    Returns:
        Dict mapping each metric to a dict with 'mean', 'std', 'scores'.
    """
    from rouge_score import rouge_scorer

    if metrics is None:
        metrics = ["rouge1", "rougeL"]

    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {m: [] for m in metrics}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for m in metrics:
            results[m].append(scores[m].fmeasure)

    return {
        m: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "scores": vals,
        }
        for m, vals in results.items()
    }


def compute_kir(
    predictions: List[str],
    salient_sentences: List[List[str]],
    word_overlap_threshold: float = 0.3,
    min_word_length: int = 4,
) -> Dict[str, float]:
    """Compute Key Information Retention (KIR).

    KIR measures what fraction of salient sentences from the source
    are represented in the generated summary (via word overlap).

    Args:
        predictions: List of generated summaries.
        salient_sentences: List of lists of salient sentences per sample.
        word_overlap_threshold: Minimum word overlap ratio for a hit.
        min_word_length: Minimum word length to consider.

    Returns:
        Dict with 'mean', 'std', and 'scores' (per-sample KIR).
    """
    scores = []
    for pred, salient in zip(predictions, salient_sentences):
        if not salient:
            scores.append(0.0)
            continue

        gen_lower = pred.lower()
        hits = 0
        for sent in salient:
            words = [w.lower() for w in sent.split() if len(w) > min_word_length]
            if words:
                word_hits = sum(1 for w in words if w in gen_lower)
                if word_hits / len(words) > word_overlap_threshold:
                    hits += 1
        scores.append(hits / len(salient))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
    }
