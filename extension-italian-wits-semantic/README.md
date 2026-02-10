# Italian Wikipedia (WITS) Extension

This folder contains the implementation and evaluation of the SM-SIP framework on the Italian Wikipedia (WITS) dataset.

## Directory Structure

```
extension-italian-wits-semantic/
├── data/
│   └── wits_train_25k.jsonl          # Semantically labeled training data (25k samples)
├── training/
│   └── training-standard.ipynb       # Unified training script for SigExt models
└── inference/
    ├── inference-standard.ipynb      # Baseline evaluation of 16 configurations
    ├── inference-prompt-enhanced.ipynb # Best-performing "Source-Aware" evaluation
    └── analyze_results.py            # Results aggregation and LaTeX table generator
```

## Components

### 1. Data (`/data`)
Contains the training set generated via **Semantic Supervision**. The labels represent sentence-level salience determined by Sentence-BERT similarity (mpnet-base-v2) between source sentences and reference summary sentences.

### 2. Training (`/training`)
The `training-standard.ipynb` notebook handles the fine-tuning of the **XLM-RoBERTa-Longformer** model. It produces 4 variants based on different similarity thresholds and sample sizes (10k-60t, 25k-60t, 25k-65t, 25k-70t).

### 3. Inference (`/inference`)
-   **Standard Pipeline**: Evaluates all combinations of quantization (4-bit, 8-bit) and prompting (zero-shot, few-shot) across the 4 trained models.
-   **Enhanced Pipeline**: Implements the "Source-Aware" prompt with manual auditing and LLM-as-judge scoring.
-   **Analysis**: `analyze_results.py` processes the JSON outputs to generate the summary metrics and LaTeX tables used in the paper.

## Requirements
See root `README.md` for full environment setup. Requires `it_core_news_sm` spacy model for Italian tokenization.
