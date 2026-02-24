"""
Text preprocessing utilities for sentence segmentation and semantic labeling.

Handles language-specific sentence splitting via spaCy and semantic
similarity computation via Sentence-BERT for training label generation.
"""

from typing import List, Tuple, Optional
import numpy as np

# Lazy-loaded spaCy models (loaded on first use)
_NLP_MODELS = {}


def _get_nlp(lang: str):
    """Lazy-load spaCy model for the given language. Auto-downloads if missing."""
    if lang not in _NLP_MODELS:
        import spacy
        model_name = "it_core_news_sm" if lang == "it" else "en_core_web_sm"
        try:
            _NLP_MODELS[lang] = spacy.load(model_name)
        except OSError:
            print(f"  Downloading spaCy model '{model_name}'...")
            spacy.cli.download(model_name)
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


def compute_similarities(
    source: str,
    summary: str,
    lang: str = "it",
    sbert_model=None,
) -> Tuple[List[str], np.ndarray]:
    """Compute per-sentence cosine similarities to the summary.

    Splits source into sentences and computes similarity of each
    sentence to the summary using Sentence-BERT embeddings.

    Args:
        source: Source text.
        summary: Reference summary.
        lang: Language for sentence segmentation.
        sbert_model: Pre-loaded SentenceTransformer model. If None, loads default.

    Returns:
        Tuple of (sentences, similarities) where similarities is a 1D array.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = extract_sentences(source, lang=lang)
    if not sentences:
        return [], np.array([])

    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    summary_emb = sbert_model.encode([summary])
    sent_embs = sbert_model.encode(sentences)
    similarities = cosine_similarity(sent_embs, summary_emb).flatten()

    return sentences, similarities


def labels_from_similarities(similarities: np.ndarray, threshold: float) -> List[int]:
    """Derive binary salience labels from pre-computed similarities.

    Args:
        similarities: 1D array of cosine similarities.
        threshold: Similarity threshold for salience.

    Returns:
        List of labels where 1 = salient, 0 = non-salient.
    """
    return [1 if sim >= threshold else 0 for sim in similarities]


def compute_semantic_labels(
    source: str,
    summary: str,
    threshold: float = 0.60,
    sentence_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    lang: str = "it",
) -> Tuple[List[str], List[int]]:
    """Compute per-sentence salience labels using Sentence-BERT similarity.

    Convenience wrapper that combines compute_similarities + labels_from_similarities.
    Kept for backward compatibility.

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

    sbert = SentenceTransformer(sentence_model_name)
    sentences, similarities = compute_similarities(source, summary, lang=lang, sbert_model=sbert)
    if len(similarities) == 0:
        return [], []
    labels = labels_from_similarities(similarities, threshold)
    return sentences, labels
