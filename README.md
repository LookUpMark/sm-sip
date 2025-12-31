# SM-SIP: Semantic & Multilingual Salient Information Prompting

Extension of the SigExt framework for controllable abstractive summarization with semantic supervision and Italian language support.

## Project Structure

```
DNLPProj/
├── extension-italian-wits-semantic/
│   ├── training/
│   │   └── training-standard.ipynb      # SigExt model training
│   └── inference/
│       ├── inference-standard.ipynb     # Main inference pipeline (16 configs)
│       ├── inference-prompt-enhanced.ipynb  # Enhanced prompts + LLM-as-Judge
│       ├── analyze_results.py           # Results analysis script
│       ├── results/                     # Standard inference results
│       └── results_enhanced/            # Enhanced prompt results
└── overleaf/
    └── paper-draft.tex                  # Paper manuscript
```

## Key Results (Italian WITS Dataset)

| Configuration | BERT Score | ROUGE-1 | KIR |
|--------------|------------|---------|-----|
| **10k-60t 8-bit few-shot** | **0.6649** | **0.2176** | 0.4396 |
| 10k-60t 8-bit zero-shot | 0.6592 | 0.2105 | **0.4908** |
| 25k-60t 4-bit few-shot | 0.6628 | 0.2133 | 0.4378 |

### Key Findings

1. **Data Efficiency**: 10k samples outperforms 25k samples
2. **Quantization**: 4-bit and 8-bit perform identically
3. **Prompting**: Zero-shot achieves higher KIR, few-shot achieves higher BERT Score
4. **Generator Dominance**: Llama-3.1-8B is the primary quality determinant

## Models (HuggingFace)

- `LookUpMark/sigext-wits-it-10k-060t`
- `LookUpMark/sigext-wits-it-25k-060t`
- `LookUpMark/sigext-wits-it-25k-065t`
- `LookUpMark/sigext-wits-it-25k-070t`

## Usage

### Training
```bash
# Run training notebook on Kaggle/Colab with GPU
jupyter notebook extension-italian-wits-semantic/training/training-standard.ipynb
```

### Inference
```bash
# Run inference (requires ~8GB VRAM for 4-bit, ~16GB for 8-bit)
jupyter notebook extension-italian-wits-semantic/inference/inference-standard.ipynb
```

### Analysis
```bash
cd extension-italian-wits-semantic/inference
python analyze_results.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers, accelerate, bitsandbytes
- spacy (it_core_news_sm)
- sentence-transformers
- langchain, langchain-huggingface

## Citation

```bibtex
@inproceedings{smsip2024,
  title={SM-SIP: Semantic and Multilingual Salient Information Prompting for Abstractive Summarization},
  author={Lopez, Marc'Antonio et al.},
  booktitle={Deep Natural Language Processing Course Project},
  year={2024},
  institution={Politecnico di Torino}
}
```

## References

- [SigExt/SIP Paper (Xu et al., 2024)](https://aclanthology.org/2024.emnlp-industry.4/)
- [WITS Dataset (Casola & Lavelli, 2021)](https://github.com/silvia-casola/WITS)