# SM-SIP: Semantic & Multilingual Salient Information Prompting

Extension of the SigExt framework for controllable abstractive summarization with semantic supervision and multilingual support (Italian & English).

## Project Structure

```
sm-sip/
├── sm_sip/                        # Core Python package (pip install -e .)
│   ├── config.py                  # Centralized dataclass configurations
│   ├── data/                      # Data loading & preprocessing
│   ├── models/                    # SigExt + LLM loading
│   ├── prompts/                   # Summarization & judge templates (IT/EN)
│   ├── metrics/                   # BERT-Score, ROUGE-1/L, KIR, abstraction, judge
│   ├── pipelines/                 # Training, inference, evaluation orchestration
│   ├── analysis/                  # Results aggregation, LaTeX tables, plots
│   └── utils/                     # GPU, I/O, reproducibility (seed=42)
├── notebooks/
│   ├── training/                  # SigExt model training (IT/EN)
│   ├── inference/                 # Summary generation (IT/EN)
│   ├── evaluation/                # Full evaluation with judge (IT/EN)
│   └── ablation/                  # 5 ablation studies
├── scripts/analyze_results.py     # Unified CLI analysis
├── results/                       # Consolidated JSON results
│   ├── italian/
│   └── english/
├── data/                          # Training data (wits_train_25k.jsonl)
├── overleaf/                      # Paper drafts
├── setup.py                       # Package installer
└── requirements.txt               # Dependencies
```

## Key Results

### Italian WITS (100 samples)

| Metric | Value |
|--------|-------|
| **BERT Score** | 0.66 |
| **ROUGE-1** | 0.21 |
| **KIR** | 46% |
| **Faithfulness** | **4.80 / 5.00** |
| **Abstraction** | 4.65 / 5.00 |

### English ArXiv (100 samples)

| Metric | Value |
|--------|-------|
| **BERT Score** | **0.82** |
| **ROUGE-1** | 0.30 |
| **KIR** | 54% |
| **Faithfulness** | **5.00 / 5.00** |
| **Abstraction** | 4.98 / 5.00 |

### Key Findings
1. **Universal Robustness**: Unlike keyword-based methods (e.g., standard SIP) that struggle with morphologically rich languages like Italian, **SM-SIP** maintains high performance across languages.
2. **Zero Hallucinations**: Semantic supervision combined with grounded prompting achieves near-perfect faithfulness (5.0/5 on ArXiv, 4.8/5 on WITS).
3. **Abstraction Quality**: The LLM judge rates abstraction highly (4.6-5.0), confirming the model produces fluent, non-extractive summaries.

## Models (HuggingFace)

**Italian (WITS):**
- `LookUpMark/sigext-wits-it-10k-060t` ← Best performing
- `LookUpMark/sigext-wits-it-25k-060t`
- `LookUpMark/sigext-wits-it-25k-065t`
- `LookUpMark/sigext-wits-it-25k-070t`

**English (ArXiv):**
- `LookUpMark/sigext-arxiv-en-1k-060t`

## Quick Start

```bash
# Install the package
pip install -e .

# Run Italian inference (on Colab/GPU)
jupyter notebook notebooks/inference/infer_italian_wits.ipynb

# Run English evaluation
jupyter notebook notebooks/evaluation/eval_english_arxiv.ipynb

# Analyze existing results (CLI)
python scripts/analyze_results.py --lang all --latex
```

## Reproducibility

All experiments use `seed=42` by default (configurable). Seeds are set for Python, NumPy, PyTorch, CUDA, and CUBLAS to ensure fully deterministic results.