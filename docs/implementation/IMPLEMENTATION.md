# Complete Implementation Guide (Extended Technical Version 4.1)

## Deep Natural Language Processing - Polytechnic University of Turin

**Project:** SM-SIP (Semantic & Multilingual Salient Information Prompting)
**Version:** 4.1 - Neuro-Symbolic Architecture
**Target Infrastructure:** Google Colab (T4 GPU) / Kaggle
**Objective:** Extend the "Salient Information Prompting" paper with semantic supervision and multilingual support (Italian).

-----

## 1. Philosophy and Repository Structure

### 1.1 The Necessity of Modularity in ML Projects
Typical student projects fail because they exist as a single, giant `.ipynb` file. This leads to:
1.  **Global Variable Hell:** Variables defined in cell 3 interfering with cell 50.
2.  **Unreproducibility:** Running cells out of order breaks state.
3.  **Collaboration Blockers:** Git cannot merge binary `.ipynb` files.

**Our Solution:** We adopt a strict "Library pattern". `src/` functions as a pip-installable Python package. Notebooks function as "Consumer scripts" that merely import and run functions.

### 1.2 Detailed Directory Tree

```
DNLP-Project/
├── src/
│   ├── __init__.py
│   ├── config.py                      # Global parameters (paths, model names, thresholds)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wits_loader.py             # WITS dataset loader (Extension 2)
│   │   ├── labeling.py                # Semantic label generation (Extension 1)
│   │   └── fuzzy_labeling.py          # Original fuzzy baseline for comparison
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sigext_model.py            # XLM-R Longformer wrapper
│   │   └── llm_wrapper.py             # Llama-3 4-bit wrapper
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── sigext_trainer.py          # HF Trainer with Weighted Loss
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── generator.py               # LangChain orchestration
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                 # ROUGE, BERTScore, KIR, AlignScore
│   │
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py         # Italian sentence segmentation (Spacy)
│
├── notebooks/
│   ├── modulo1_architecture.ipynb     # Longformer architecture tests
│   ├── modulo2_semantic_labels.ipynb  # Extension 1: Semantic labeling
│   ├── modulo3_training.ipynb         # SigExt training
│   ├── modulo4_italian.ipynb          # Extension 2: Italian adaptation
│   └── master_evaluation.ipynb        # Final comparison: Baseline vs Fuzzy vs Semantic
```

-----

## 2. The "Neuro-Symbolic" Paradigm

We call this architecture "Neuro-Symbolic" because it bridges two worlds:
1.  **Symbolic/Discrete:** The `SigExt` model outputs discrete decisions (0 or 1) for each sentence.
2.  **Neural/Continuous:** The `Llama-3` model operates in the continuous space of token probabilities.

**The "Steering" Mechanism:**
We implement **Dynamic Prompting**: extracted sentences are programmatically injected into the `system_prompt` slot, acting as "Soft Constraints" that force the model to *attend* to specific content.

-----

## 3. Extension 1: Semantic Supervision — Full Implementation

> **Scientific Contribution:** Replace lexical (fuzzy) matching with semantic (embedding-based) matching for training label generation.

### 3.1 The Problem with Fuzzy Matching

The original paper uses:
$$\text{Fuzz}(p, q) = \frac{\text{LCS}(p, q)}{\max(|p|, |q|)}$$

**Why this fails:**
- "The company declared bankruptcy" vs "The firm failed" → Fuzzy = 0.15 (WRONG!)
- "The stock price increased" vs "Shares rose" → Fuzzy = 0.20 (WRONG!)

### 3.2 Implementation: `src/data/labeling.py`

```python
"""
Semantic Labeling for SigExt Training
=====================================
This module implements Extension 1: Semantic Supervision.

Instead of fuzzy matching, we use Sentence-BERT embeddings
to compute cosine similarity between source and summary sentences.
"""

import json
import torch
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import spacy

# Configuration
CONFIG = {
    "SBERT_MODEL": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "THRESHOLD": 0.60,  # Semantic similarity threshold
    "MIN_SENTENCE_LENGTH": 20,  # Ignore very short sentences
    "BATCH_SIZE": 32,  # For efficient GPU encoding
}


class SemanticLabeler:
    """
    Generates semantic labels for SigExt training.
    
    The core innovation: instead of lexical matching, we use
    embedding similarity to identify conceptually related sentences.
    """
    
    def __init__(self, threshold: float = 0.60):
        """
        Args:
            threshold: Cosine similarity threshold. 
                       Sentences with max_sim > threshold are labeled as salient.
                       Recommended: 0.55-0.65 (see threshold sweep experiments)
        """
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Sentence-BERT (multilingual for Italian support)
        print(f"Loading S-BERT on {self.device}...")
        self.sbert = SentenceTransformer(CONFIG["SBERT_MODEL"], device=self.device)
        
        # Load Spacy for Italian sentence segmentation
        self.nlp = spacy.load("it_core_news_sm")
        # Disable unused components for speed
        self.nlp.disable_pipes(["ner", "parser"])
    
    def segment_text(self, text: str) -> List[str]:
        """
        Split text into sentences using Spacy.
        
        Why Spacy instead of split('.')?
        - Handles abbreviations: "Art. 1", "Sig. Rossi", "ecc."
        - Handles quotes and complex punctuation
        - Language-aware sentence boundary detection
        """
        doc = self.nlp(text)
        sentences = [
            sent.text.strip() 
            for sent in doc.sents 
            if len(sent.text.strip()) > CONFIG["MIN_SENTENCE_LENGTH"]
        ]
        return sentences
    
    def compute_labels(
        self, 
        source_sentences: List[str], 
        summary_sentences: List[str]
    ) -> Tuple[List[int], List[float]]:
        """
        Compute semantic labels for source sentences.
        
        Algorithm:
        1. Encode all sentences into 768-dim vectors
        2. Compute cosine similarity matrix [N_source x M_summary]
        3. For each source sentence, find max similarity to any summary sentence
        4. Label = 1 if max_sim > threshold, else 0
        
        Returns:
            labels: List of 0/1 labels
            scores: List of max similarity scores (for debugging/analysis)
        """
        if not source_sentences or not summary_sentences:
            return [], []
        
        # Encode all sentences in batch (GPU-optimized)
        source_emb = self.sbert.encode(
            source_sentences, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        summary_emb = self.sbert.encode(
            summary_sentences, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        
        # Compute similarity matrix
        sim_matrix = util.cos_sim(source_emb, summary_emb)  # [N, M]
        
        # Generate labels
        labels = []
        scores = []
        for i in range(len(source_sentences)):
            max_sim = sim_matrix[i].max().item()
            scores.append(max_sim)
            labels.append(1 if max_sim > self.threshold else 0)
        
        return labels, scores
    
    def process_document(
        self, 
        source_text: str, 
        summary_text: str
    ) -> Dict:
        """
        Process a single document-summary pair.
        
        Returns:
            Dict with sentences, labels, and scores
        """
        source_sentences = self.segment_text(source_text)
        summary_sentences = self.segment_text(summary_text)
        
        labels, scores = self.compute_labels(source_sentences, summary_sentences)
        
        return {
            "sentences": source_sentences,
            "labels": labels,
            "scores": scores,  # For debugging
            "num_salient": sum(labels),
            "num_total": len(labels),
        }
    
    def process_dataset(
        self, 
        dataset_iterator, 
        output_path: str,
        num_samples: int = 10000
    ):
        """
        Process entire dataset and save to JSONL.
        
        Using JSONL format for:
        - Streaming read support (no need to load entire file)
        - Append-friendly (can resume if interrupted)
        - Human-readable for debugging
        """
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in tqdm(dataset_iterator, total=num_samples):
                source = entry["source"]
                summary = entry["summary"]
                
                # Skip invalid entries
                if len(source) < 500 or len(summary) < 50:
                    continue
                
                result = self.process_document(source, summary)
                
                # Only keep documents with at least one salient sentence
                if result["num_salient"] > 0:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    count += 1
                
                if count >= num_samples:
                    break
        
        print(f"Saved {count} labeled documents to {output_path}")


# ============================================================
# THRESHOLD SWEEP EXPERIMENT
# ============================================================

def threshold_sweep_experiment(sample_data: List[Tuple[str, str]]):
    """
    Experiment to find optimal threshold.
    
    Run this on a small sample (100 docs) before full dataset generation.
    """
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    results = {}
    
    for thresh in thresholds:
        labeler = SemanticLabeler(threshold=thresh)
        
        total_salient = 0
        total_sentences = 0
        
        for source, summary in sample_data:
            result = labeler.process_document(source, summary)
            total_salient += result["num_salient"]
            total_sentences += result["num_total"]
        
        salient_ratio = total_salient / total_sentences if total_sentences > 0 else 0
        results[thresh] = {
            "salient_ratio": salient_ratio,
            "total_salient": total_salient,
            "total_sentences": total_sentences,
        }
        
        print(f"θ={thresh}: {salient_ratio:.2%} sentences labeled as salient")
    
    return results
```

### 3.3 Fuzzy Baseline for Comparison: `src/data/fuzzy_labeling.py`

```python
"""
Fuzzy Labeling (Original Paper Method)
======================================
This implements the ORIGINAL paper's labeling method.
We use this as a BASELINE to compare against our semantic method.
"""

from fuzzywuzzy import fuzz
from typing import List, Tuple

def fuzzy_label(
    source_sentences: List[str], 
    summary_sentences: List[str],
    threshold: float = 0.70
) -> List[int]:
    """
    Original paper's labeling method.
    
    For each source sentence, compute fuzzy ratio with all summary sentences.
    If max ratio > threshold, label as 1.
    """
    labels = []
    for src_sent in source_sentences:
        max_score = 0
        for sum_sent in summary_sentences:
            # Character-level fuzzy matching
            score = fuzz.ratio(src_sent.lower(), sum_sent.lower()) / 100.0
            max_score = max(max_score, score)
        labels.append(1 if max_score > threshold else 0)
    return labels
```

### 3.4 Notebook: `notebooks/modulo2_semantic_labels.ipynb`

Key experiments to run:
1. **Threshold Sweep:** Test θ ∈ {0.50, 0.55, 0.60, 0.65, 0.70}
2. **Distribution Analysis:** Plot histogram of salient vs non-salient sentences
3. **Qualitative Check:** Manually verify 20 random labels for correctness
4. **Comparison:** Generate both Fuzzy and Semantic labels for same 1000 documents, compare overlap

-----

## 4. Extension 2: Italian Multilingual Adaptation — Full Implementation

> **Scientific Contribution:** Validate SIP on Italian, a morphologically rich language where fuzzy matching fails even more spectacularly.

### 4.1 Why Italian is a Good Test Case

| Challenge | Example | Impact |
|-----------|---------|--------|
| Verb conjugation | "mangiano" vs "mangiarono" vs "avrebbero mangiato" | Same verb, 3 forms |
| Article agreement | "il libro" / "i libri" / "dei libri" | Inflection changes |
| Pro-drop | "Vado" = "I go" (subject omitted) | Shorter sentences |
| Clitics | "glielo disse" = "he said it to him" | Merged words |

### 4.2 The WITS Dataset: `src/data/wits_loader.py`

```python
"""
WITS Dataset Loader
===================
Wikipedia for Italian Text Summarization

This dataset is ideal for Extension 2 because:
1. Native Italian (not translated from English)
2. Long documents (avg >1000 tokens, justifies Longformer)
3. Encyclopedic domain (factual, similar to ArXiv)
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import json


class WITSDataset(Dataset):
    """
    PyTorch Dataset for WITS.
    
    Loads pre-generated semantic labels from JSONL file.
    Handles tokenization and label alignment for Longformer.
    """
    
    def __init__(
        self, 
        labels_path: str,
        tokenizer_name: str = "markussagen/xlm-roberta-longformer-base-4096",
        max_length: int = 4096
    ):
        """
        Args:
            labels_path: Path to JSONL file with semantic labels
            tokenizer_name: HuggingFace tokenizer ID
            max_length: Maximum sequence length (4096 for Longformer)
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load labeled data
        self.data = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} labeled documents")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns tokenized input with aligned labels.
        
        Label Alignment Strategy:
        - Labels are per-sentence
        - Model outputs per-token
        - We assign sentence label to FIRST token of that sentence
        - All other tokens get label -100 (ignored in loss)
        """
        item = self.data[idx]
        sentences = item["sentences"]
        sentence_labels = item["labels"]
        
        # Join sentences for tokenization
        full_text = " ".join(sentences)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create token-level labels
        # Initialize all as -100 (ignore index)
        token_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        
        # Align sentence labels to first token of each sentence
        # This is an approximation - more sophisticated alignment is possible
        current_pos = 0
        for sent_idx, sentence in enumerate(sentences):
            if sent_idx >= len(sentence_labels):
                break
            
            # Tokenize individual sentence to find its tokens
            sent_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            
            if current_pos < self.max_length:
                # Assign label to first token of sentence
                token_labels[current_pos] = sentence_labels[sent_idx]
            
            current_pos += len(sent_tokens)
            if current_pos >= self.max_length:
                break
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": token_labels
        }


def load_wits_streaming(num_samples: int = 10000):
    """
    Load WITS dataset in streaming mode.
    
    Why streaming?
    - Full dataset is ~10GB
    - Streaming processes one example at a time
    - No memory explosion
    """
    dataset = load_dataset(
        "silvia-casola/WITS", 
        split="train", 
        streaming=True
    )
    return dataset.take(num_samples)
```

### 4.3 Italian Text Processing: `src/utils/text_processing.py`

```python
"""
Italian Text Processing Utilities
=================================
Specialized functions for Italian NLP preprocessing.
"""

import spacy
from typing import List

# Load Italian model globally (expensive, do once)
_nlp = None

def get_italian_nlp():
    """Lazy-load Italian Spacy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("it_core_news_sm")
        # Disable unused components for speed
        _nlp.disable_pipes(["ner"])
    return _nlp


def segment_italian(text: str, min_length: int = 20) -> List[str]:
    """
    Italian-aware sentence segmentation.
    
    Handles:
    - Abbreviations: Art., Sig., Dott., ecc.
    - Decimal numbers: 3.14
    - Dialogue punctuation: «Ciao», disse.
    """
    nlp = get_italian_nlp()
    doc = nlp(text)
    
    sentences = []
    for sent in doc.sents:
        clean = sent.text.strip()
        if len(clean) >= min_length:
            sentences.append(clean)
    
    return sentences


def normalize_italian(text: str) -> str:
    """
    Normalize Italian text for comparison.
    
    Operations:
    - Lowercase
    - Remove accents inconsistencies (è vs e')
    - Normalize quotes
    """
    text = text.lower()
    
    # Normalize apostrophe accents
    replacements = {
        "e'": "è",
        "a'": "à",
        "i'": "ì",
        "o'": "ò",
        "u'": "ù",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize quotes
    text = text.replace("«", '"').replace("»", '"')
    
    return text
```

### 4.4 Italian Prompt Template: `src/inference/generator.py`

```python
ITALIAN_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sei un esperto di sintesi testuale italiano. Il tuo compito è generare riassunti 
che siano:
- FEDELI: Non inventare informazioni non presenti nel testo
- COMPLETI: Includi tutti i concetti chiave indicati
- COERENTI: Il riassunto deve essere fluido e ben scritto
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
TESTO ORIGINALE:
{source_text}

---

ISTRUZIONI OBBLIGATORIE:
I seguenti concetti chiave sono stati estratti dal testo e DEVONO essere 
riflessi nel riassunto finale:

{keyphrases}

---

Genera un riassunto astrattivo in italiano (150-250 parole):
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
```

-----

## 5. Training Module: `src/training/sigext_trainer.py`

```python
"""
SigExt Trainer with Weighted Loss
=================================
Custom trainer that handles class imbalance in salience detection.

The Problem:
- Only ~10% of sentences are salient
- A lazy model predicts all zeros and gets 90% accuracy

The Solution:
- Weight the positive class 10x in the loss function
- Forces model to prioritize recall on salient sentences
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from typing import Dict, Any


class WeightedSigExtTrainer(Trainer):
    """
    Custom Trainer with weighted loss for imbalanced classification.
    """
    
    def __init__(self, pos_weight: float = 10.0, *args, **kwargs):
        """
        Args:
            pos_weight: Weight for positive class (salient sentences)
                        Higher = model tries harder to find salient sentences
                        Recommended: 8.0 - 12.0
        """
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
    
    def compute_loss(
        self, 
        model, 
        inputs, 
        return_outputs: bool = False,
        num_items_in_batch: int = None
    ):
        """
        Custom loss computation with class weighting.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get device
        device = logits.device
        
        # Create weighted loss function
        # Class weights: [1.0 for class 0, pos_weight for class 1]
        weight = torch.tensor([1.0, self.pos_weight]).to(device)
        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        
        # Compute loss
        # logits: [batch, seq_len, 2]
        # labels: [batch, seq_len]
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def get_training_args(output_dir: str = "./checkpoints") -> TrainingArguments:
    """
    Training configuration optimized for T4 GPU.
    
    Memory Optimization Strategy:
    - Batch size 2 (Longformer is memory-hungry)
    - Gradient accumulation 16 (effective batch = 32)
    - FP16 mixed precision
    - Gradient checkpointing
    """
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training duration
        num_train_epochs=2,
        
        # Batch configuration
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,  # Effective batch = 32
        
        # Optimization
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # Memory optimization
        fp16=True,
        gradient_checkpointing=True,
        
        # Logging
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        
        # Reproducibility
        seed=42,
        
        # Misc
        report_to="none",
        load_best_model_at_end=True,
    )
```

-----

## 6. Evaluation Module: `src/evaluation/metrics.py`

```python
"""
Evaluation Metrics for SM-SIP
=============================
Beyond ROUGE: semantic fidelity, controllability, and faithfulness.
"""

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_compute
from typing import List, Dict


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)
    
    return {k: np.mean(v) for k, v in scores.items()}


def compute_bertscore(
    predictions: List[str], 
    references: List[str],
    lang: str = "it"
) -> float:
    """
    Compute BERTScore (semantic similarity).
    
    Why BERTScore?
    - ROUGE only catches word overlap
    - BERTScore catches meaning ("car" ≈ "vehicle")
    - Critical for validating our semantic supervision extension
    """
    P, R, F1 = bert_score_compute(predictions, references, lang=lang, verbose=False)
    return F1.mean().item()


def compute_kir(
    keyphrases: List[List[str]], 
    summaries: List[str]
) -> float:
    """
    Compute Keyphrase Inclusion Rate (KIR).
    
    Definition:
    KIR = (# keyphrases appearing in summary) / (# keyphrases suggested)
    
    Why KIR?
    - Measures "Controllability" / "Steerability"
    - Did the LLM actually follow our instructions?
    - High KIR = the steering mechanism is working
    """
    inclusion_rates = []
    
    for kps, summary in zip(keyphrases, summaries):
        if not kps:
            continue
        
        summary_lower = summary.lower()
        hits = 0
        
        for kp in kps:
            # Normalize keyphrase
            kp_normalized = kp.lower().strip()
            
            # Check if keyphrase appears in summary
            # Using substring matching (simple but effective)
            if kp_normalized in summary_lower:
                hits += 1
        
        inclusion_rates.append(hits / len(kps))
    
    return np.mean(inclusion_rates) if inclusion_rates else 0.0


def run_full_evaluation(
    predictions: List[str],
    references: List[str],
    keyphrases: List[List[str]],
    lang: str = "it"
) -> Dict[str, float]:
    """
    Run complete evaluation suite.
    
    Returns dict with all metrics.
    """
    results = {}
    
    # ROUGE
    rouge_scores = compute_rouge(predictions, references)
    results.update(rouge_scores)
    
    # BERTScore
    results["bertscore"] = compute_bertscore(predictions, references, lang)
    
    # KIR
    results["kir"] = compute_kir(keyphrases, predictions)
    
    return results
```

-----

## 7. Timeline and Operational Execution

### Phase 1: Setup (Week 1)
- **Day 1-2:** Environment setup, download WITS, test Longformer loading
- **Day 3-4:** Implement `labeling.py`, run threshold sweep experiment

### Phase 2: Data Generation (Week 2)
- **Day 1-2:** Generate 10k Semantic Labels (takes ~4-5 hours on GPU)
- **Day 3-4:** Generate 10k Fuzzy Labels (for baseline comparison)
- **Day 5:** Quality check: manually verify 50 random labels

### Phase 3: Training (Week 3)
- **Day 1-2:** Train SigExt-Semantic (2 epochs, ~8 hours)
- **Day 3-4:** Train SigExt-Fuzzy (for comparison)
- **Day 5:** Evaluate both on validation set

### Phase 4: Integration & Evaluation (Week 4)
- **Day 1-2:** Set up Llama-3 inference pipeline
- **Day 3-4:** Run full evaluation: Baseline vs Fuzzy vs Semantic
- **Day 5:** Generate final report and figures

-----

## 8. Expected Results Summary

| Configuration | ROUGE-1 | BERTScore | KIR |
|---------------|---------|-----------|-----|
| Baseline (Zero-Shot) | 0.35 | 0.78 | N/A |
| SigExt-Fuzzy | 0.38 | 0.80 | 0.65 |
| **SigExt-Semantic** | **0.41** | **0.83** | **0.78** |

**Key Hypothesis:** Semantic supervision produces better extractors, leading to better steering, leading to better summaries—especially in morphologically rich Italian.
