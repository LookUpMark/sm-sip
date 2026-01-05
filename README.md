# SM-SIP: Semantic & Multilingual Salient Information Prompting

Extension of the SigExt framework for controllable abstractive summarization with semantic supervision and multilingual support (Italian & English).

## Project Structure

```
DNLPProj/
├── extension-italian-wits-semantic/     # Italian Wikipedia (WITS)
│   ├── training/
│   │   └── training-standard.ipynb      # SigExt model training
│   └── inference/
│       ├── inference-standard.ipynb     # Standard inference (16 configs)
│       ├── inference-prompt-enhanced.ipynb  # Enhanced + LLM-as-Judge
│       └── results_enhanced/            # Results with justifications
├── extension-english-arxiv-semantic/    # English ArXiv
│   └── inference/
│       ├── inference-prompt-enhanced.ipynb  # ArXiv evaluation
│       └── results_enhanced.json        # ArXiv results
└── overleaf/
    └── paper-draft.tex                  # Paper manuscript
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
# Italian WITS inference
jupyter notebook extension-italian-wits-semantic/inference/inference-prompt-enhanced.ipynb

# English ArXiv inference
jupyter notebook extension-english-arxiv-semantic/inference/inference-prompt-enhanced.ipynb
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- transformers, accelerate, bitsandbytes
- spacy (`it_core_news_sm`, `en_core_web_sm`)
- langchain, langchain-huggingface
- ~8GB VRAM (8-bit) or ~6GB VRAM (4-bit)

## Citation

```bibtex
@inproceedings{smsip2024,
  title={SM-SIP: Semantic and Multilingual Salient Information Prompting},
  author={Lopez, Marc'Antonio et al.},
  booktitle={Deep NLP Course Project, Politecnico di Torino},
  year={2024}
}
```

## References

- [SigExt/SIP Paper (Xu et al., 2024)](https://aclanthology.org/2024.emnlp-industry.4/)
- [WITS Dataset (Casola & Lavelli, 2021)](https://github.com/silvia-casola/WITS)
- [ArXiv Summarization (Cohan et al.)](https://huggingface.co/datasets/ccdv/arxiv-summarization)