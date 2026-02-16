# Data

## Datasets

### Italian WITS (Wikipedia Italian Text Summarization)
- **Source**: HuggingFace [`silvia-casola/WITS`](https://huggingface.co/datasets/silvia-casola/WITS)
- **Training file**: `wits_train_25k.jsonl` — 25,000 samples with semantic salience labels
- **Test split**: 100 samples (skipping first 25,000 for train/test separation)
- **Fields**: `source` (article text), `summary` (reference summary)

### English ArXiv
- **Source**: HuggingFace [`ccdv/arxiv-summarization`](https://huggingface.co/datasets/ccdv/arxiv-summarization)
- **Loaded at runtime** via HuggingFace Datasets streaming
- **Training**: 1,000 samples, **Test**: 100 samples
- **Fields**: `article` (paper text), `abstract` (reference summary)

## Generating Training Data

The semantic labels are generated using Sentence-BERT similarity:

```python
from sm_sip.data.preprocessing import compute_semantic_labels

sentences, labels = compute_semantic_labels(
    source="...",
    summary="...",
    threshold=0.60,
    lang="it",
)
```

Each sentence with similarity ≥ threshold to the reference summary is labeled as salient (1).
