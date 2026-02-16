"""
Text preprocessing utilities for sentence segmentation and semantic labeling.

Handles language-specific sentence splitting via spaCy and semantic
similarity computation via Sentence-BERT for training label generation.
"""

from typing import List, Tuple
import numpy as np

# Lazy-loaded spaCy models (loaded on first use)
_NLP_MODELS = {}


def _get_nlp(lang: str):
    """Lazy-load spaCy model for the given language."""
    if lang not in _NLP_MODELS:
        import spacy
        model_name = "it_core_news_sm" if lang == "it" else "en_core_web_sm"
        _NLP_MODELS[lang] = spacy.load(model_name)
    return _NLP_MODELS[lang]


def extract_sentences(text: str, lang: str = "it", min_length: int = 20) -> List[str]:
    """Split text into sentences using spaCy.

    Args:
        text: Input text.
        lang: Language code ("it" or "en").
        min_length: Minimum sentence length in characters.

    Returns:
        List of sentence strings.
    """
    nlp = _get_nlp(lang)
    return [sent.text.strip() for sent in nlp(text).sents if len(sent.text.strip()) > min_length]


def compute_semantic_labels(
    source: str,
    summary: str,
    threshold: float = 0.60,
    sentence_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    lang: str = "it",
) -> Tuple[List[str], List[int]]:
    """Compute per-sentence salience labels using Sentence-BERT similarity.

    Each sentence is compared to the reference summary. If cosine
    similarity >= threshold, the sentence is labeled as salient (1).

    Args:
        source: Source text.
        summary: Reference summary.
        threshold: Similarity threshold for salience.
        sentence_model_name: Sentence-BERT model for embeddings.
        lang: Language for sentence segmentation.

    Returns:
        Tuple of (sentences, labels) where labels[i] in {0, 1}.
    """
    from sentence_transformers import SentenceTransformer

    sentences = extract_sentences(source, lang=lang)
    if not sentences:
        return [], []

    model = SentenceTransformer(sentence_model_name)
    summary_emb = model.encode([summary])
    sent_embs = model.encode(sentences)

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(sent_embs, summary_emb).flatten()

    labels = [1 if sim >= threshold else 0 for sim in similarities]
    return sentences, labels
