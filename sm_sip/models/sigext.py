"""
SigExt model loading and salient sentence extraction.

SigExt is a token-classification model that identifies salient sentences
in source documents for use in guided summarization.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_sigext_model(
    model_id: str,
    max_length: int = 2048,
) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """Load a SigExt token-classification model.

    Args:
        model_id: HuggingFace model ID (e.g. 'LookUpMark/sigext-wits-it-10k-060t').
        max_length: Maximum sequence length.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"  Loading SigExt model: {model_id}...")
    model = AutoModelForTokenClassification.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def extract_salient_sentences(
    text: str,
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    lang: str = "it",
    max_length: int = 2048,
    top_k: int = 5,
) -> Tuple[List[str], str]:
    """Extract salient sentences from text using SigExt.

    Args:
        text: Source document text.
        model: SigExt model.
        tokenizer: SigExt tokenizer.
        lang: Language for sentence splitting.
        max_length: Maximum sequence length.
        top_k: Maximum number of salient sentences to return.

    Returns:
        Tuple of (salient_sentences, keyphrases_string).
    """
    import torch
    from sm_sip.data.preprocessing import extract_sentences

    sentences = extract_sentences(text, lang=lang)
    if not sentences:
        return [], ""

    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    ).to(model.device)

    offsets = encoding.pop("offset_mapping")[0].cpu().numpy()

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Map token predictions to sentence-level scores
    sentence_scores = []
    for sent in sentences:
        start_idx = text.find(sent)
        if start_idx < 0:
            sentence_scores.append(0.0)
            continue
        end_idx = start_idx + len(sent)

        # Count salient tokens in this sentence
        salient_count = 0
        total_count = 0
        for j, (s, e) in enumerate(offsets):
            if s >= start_idx and e <= end_idx and s != e:
                total_count += 1
                if predictions[j] == 1:
                    salient_count += 1

        score = salient_count / max(total_count, 1)
        sentence_scores.append(score)

    # Select top-k salient sentences
    ranked_indices = np.argsort(sentence_scores)[::-1][:top_k]
    salient = [sentences[i] for i in sorted(ranked_indices) if sentence_scores[i] > 0]

    keyphrases = " | ".join(salient) if salient else ""
    return salient, keyphrases


def preprocess_dataset(
    data: List[Dict],
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    lang: str = "it",
    top_k: int = 5,
) -> List[Dict]:
    """Preprocess a dataset by extracting salient sentences.

    Args:
        data: List of dicts with 'source', 'summary'/'reference' keys.
        model: SigExt model.
        tokenizer: SigExt tokenizer.
        lang: Language for processing.
        top_k: Max salient sentences per document.

    Returns:
        List of processed dicts with added 'keyphrases' and 'salient_sentences'.
    """
    from tqdm.auto import tqdm

    processed = []
    for item in tqdm(data, desc="  Extracting salient sentences"):
        source = item["source"]
        reference = item.get("summary", item.get("reference", ""))

        salient, keyphrases = extract_salient_sentences(
            source, model, tokenizer, lang=lang, top_k=top_k
        )

        processed.append({
            "source": source,
            "reference": reference,
            "keyphrases": keyphrases,
            "salient_sentences": salient,
        })

    return processed
