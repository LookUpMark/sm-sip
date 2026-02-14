# English ArXiv Extension

This folder contains the implementation and evaluation of the SM-SIP framework on the English ArXiv dataset (ccdv/arxiv-summarization).

## Directory Structure

```
extension-english-arxiv-semantic/
├── training/
│   └── training-standard.ipynb         # SigExt model training for English
├── inference/
│   ├── inference-standard.ipynb        # ArXiv baseline evaluation (4 configurations)
│   ├── inference-prompt-enhanced.ipynb # Best-performing "Source-Aware" evaluation
│   ├── inference-judge.ipynb           # LLM-as-Judge evaluation pipeline
│   └── analyze_results.py              # Results aggregation script
└── results/                            # Symlink to root results/english folder
```

## Components

### 1. Training (`/training`)
The `training-standard.ipynb` notebook handles the fine-tuning of the **SigExt** model on 1,000 ArXiv samples with semantic supervision (Longformer-base-4096). The resulting model is `sigext-arxiv-en-1k-060t`.

### 2. Inference (`/inference`)
-   **Standard Pipeline**: Evaluates the model across 4 configurations (decoding type x quantization).
-   **Enhanced Pipeline**: Implemented in `inference-prompt-enhanced.ipynb`, this uses the optimized "Source-Aware" prompt and incorporates the **LLM-as-Judge** manual auditing.
-   **Analysis**: `analyze_results.py` summaries the JSON metrics, confirming the high faithfulness (5.0/5) achieved on the English domain.

## Requirements
See root `README.md` for full environment setup. Requires `en_core_web_sm` spacy model for English tokenization.
