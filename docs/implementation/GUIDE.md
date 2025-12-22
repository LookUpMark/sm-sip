# Technical Blueprint: SM-SIP (Semantic & Multilingual Salient Information Prompting)

## 1. Introduction: The Strategic Context

This document provides a comprehensive technical blueprint for the "Deep Natural Language Processing" course project. Its primary goal is not merely to "pass the exam" but to produce a piece of work that holds **scientific validity**, potentially leading to a workshop publication or a high-honors grade.

### 1.1 The "Controllability Crisis" in Modern Summarization

#### The Problem: Why Generative AI is not enough
The advent of Large Language Models (LLMs) like GPT-4, Llama 3, and Mistral has solved the *fluency* problem in abstractive summarization. These models can generate text that reads perfectly human. However, they have introduced a new, critical failure mode: **Lack of Control**.

When a user provides a long document (e.g., a 10-page scientific paper or a 2-hour meeting transcript) and asks an LLM to "summarize this," the model operates as a "Black Box". It relies on internal attention weights learned during pre-training to decide which tokens are important.
*   **The Flaw:** These attention weights often prioritize *frequent* words or simple sentence structures over *informationally dense* but rare entities (e.g., specific dates, chemical formulas, proper nouns).
*   **The Consequence:** The resulting summary is fluent but "hollow" (missing key facts) or, worse, "hallucinated" (inventing facts to fill gaps).

#### The Solution: Decoupling Selection from Generation
Our project adopts the **"Salient Information Prompting" (SIP)** paradigm (Xu et al., 2024). The core philosophy is to treat summarization as a two-step neuro-symbolic process:
1.  **Selection (The "What"):** A specialized, smaller model (The Extractor) acts as a high-precision filter, identifying exactly *which* sentences contain the core information. This model doesn't care about grammar or flow; it only cares about *salience*.
2.  **Generation (The "How"):** A massive LLM (The Generator) takes the original text *and* the extracted key sentences (as explicit constraints) and weaves them into a fluent narrative.

**Our Two Core Extensions:**
The original paper used *Lexical Overlap* (Fuzzy Matching) to train the extractor. This is scientifically flawed. We propose two major extensions:

| Extension | Problem Addressed | Our Innovation |
|-----------|-------------------|----------------|
| **Extension 1: Semantic Supervision** | Fuzzy Matching fails on synonyms ("car" vs "vehicle" = 0% overlap) | Train on Sentence-BERT embeddings (cosine similarity > 0.6) |
| **Extension 2: Italian Adaptation** | Original paper is English-only | Validate on WITS dataset using XLM-RoBERTa-Longformer |

---

## 2. Architectural Decisions: The "Why" Behind the Stack

We have carefully selected a "Tri-Model" architecture. Here is the rationale for each component.

### 2.1 The Generator: Meta-Llama-3-8B-Instruct
*   **The Choice:** We chose Llama-3-8B over Mistral-7B or Gemma-7B.
*   **The "Why":**
    *   **Instruction Following:** Llama 3 has shown superior capability in adhering to complex system prompts ("You must include these phrases..."). Older models often ignore these constraints after generating a few tokens ("Instruction Drift").
    *   **Context Window:** With an 8k native context window, it can ingest full Wikipedia articles without aggressive truncation, unlike older 4k models.
    *   **Efficiency:** By using 4-bit Normal Float (NF4) quantization, we fit this model into ~5.5GB of VRAM, leaving room on a standard T4 GPU (16GB) to run the Extractor simultaneously.

### 2.2 The Extractor: XLM-RoBERTa-Longformer
*   **The Choice:** A custom adaptation of XLM-R using Longformer attention.
*   **The "Why" (The Quadratic Bottleneck):**
    *   Standard BERT/RoBERTa models have a limit of 512 tokens because their Self-Attention mechanism computes a matrix of size $N^2$. For a 4000-token document, this would require $4000^2 = 16,000,000$ connections, causing immediate Out-Of-Memory (OOM) errors.
    *   **Longformer Solution:** It uses **Sparse Attention**. Each token only attends to a local window (e.g., 512 neighbors) and a few global attention tokens (e.g., `<s>`). This reduces complexity to $O(N)$, allowing us to process entire documents up to 4096 tokens.
    *   **Multilingualism:** We use the XLM-R base because it was pre-trained on 100 languages, including Italian. The standard Longformer (AllenAI) is English-only.

### 2.3 The Supervisor: Sentence-BERT (Paraphrase-Multilingual-Mpnet)
*   **The Choice:** `paraphrase-multilingual-mpnet-base-v2`.
*   **The "Why":**
    *   This component serves as the "Teacher" to generate training data.
    *   **Vector Space vs. String Matching:** In the sentence "La borsa è crollata" vs "I mercati finanziari sono in ribasso", the Levenshtein distance is huge. However, in the high-dimensional vector space produced by S-BERT, these two sentences are nearly identical vectors (Cosine Similarity > 0.7). This allows our extractor to learn that this sentence is *conceptually* important, even if the summary uses totally different words.

---

## 3. Extension 1: Semantic Supervision (Methodological Innovation)

> **Goal:** Replace the weak Fuzzy Matching labeling with robust Semantic Similarity labeling.

### 3.1 The Problem with Fuzzy Matching (Original Paper)

The original SigExt paper generates training labels using character-level fuzzy matching:

$$\text{Fuzz}(p, q) = \frac{\text{LongestCommonSubsequence}(p, q)}{\max(|p|, |q|)}$$

**Critical Failure Modes:**
| Source Sentence | Summary Sentence | Fuzzy Score | Semantic Score | Correct Label |
|-----------------|------------------|-------------|----------------|---------------|
| "The company declared bankruptcy" | "The firm failed" | 0.15 ❌ | 0.82 ✅ | 1 (Salient) |
| "Apple's stock price increased by 5%" | "Apple shares rose" | 0.28 ❌ | 0.91 ✅ | 1 (Salient) |
| "The weather was nice" | "It was sunny" | 0.12 ❌ | 0.75 ✅ | 1 (Salient) |

The fuzzy approach creates **noisy labels** that teach the extractor to ignore valid paraphrases.

### 3.2 Our Solution: Sentence-BERT Semantic Labeling

**The Core Algorithm:**
```python
from sentence_transformers import SentenceTransformer, util

# 1. Load the multilingual model
sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 2. Encode all sentences
doc_embeddings = sbert.encode(document_sentences)  # [N, 768]
sum_embeddings = sbert.encode(summary_sentences)   # [M, 768]

# 3. Compute similarity matrix
similarity_matrix = util.cos_sim(doc_embeddings, sum_embeddings)  # [N, M]

# 4. Assign labels
THRESHOLD = 0.60
labels = []
for i in range(len(document_sentences)):
    max_sim = similarity_matrix[i].max().item()
    labels.append(1 if max_sim > THRESHOLD else 0)
```

### 3.3 Threshold Selection: The Critical Hyperparameter

| Threshold $\theta$ | Effect | Risk |
|--------------------|--------|------|
| < 0.50 | Too permissive | Noisy labels (irrelevant sentences marked salient) |
| 0.55 - 0.65 | **Optimal Range** | Captures paraphrases, ignores noise |
| > 0.75 | Too strict | Misses valid conceptual matches |

**Experimental Protocol:**
1. Generate labels at $\theta \in \{0.50, 0.55, 0.60, 0.65, 0.70\}$
2. Train SigExt on each
3. Measure downstream BERTScore on validation set
4. Select $\theta$ with highest BERTScore

### 3.4 Implementation Files

| File | Purpose |
|------|---------|
| `src/data/labeling.py` | Core labeling logic with S-BERT |
| `src/data/semantic_labeler.py` | Batch processing for 10k+ documents |
| `notebooks/modulo2_semantic_labels.ipynb` | Threshold sweep experiments |

---

## 4. Extension 2: Italian Multilingual Adaptation (Domain Innovation)

> **Goal:** Prove that SIP generalizes beyond English to morphologically rich languages.

### 4.1 Why Italian is Challenging

Italian presents unique NLP challenges that make it an ideal test case:

| Challenge | English Example | Italian Example | Impact on Fuzzy Matching |
|-----------|-----------------|-----------------|--------------------------|
| **Verb Conjugation** | "he runs" | "lui corre / egli corre / correva" | Same meaning, different strings |
| **Article-Noun Agreement** | "the house" | "la casa / le case / della casa" | Inflection changes |
| **Pro-Drop** | "I am going" | "Vado" (subject implicit) | Sentence structure differs |
| **Compound Tenses** | "has eaten" | "ha mangiato / avrebbe mangiato" | Auxiliary variations |

Fuzzy matching catastrophically fails on Italian because the same concept can be expressed with vastly different character sequences.

### 4.2 The WITS Dataset

**WITS (Wikipedia for Italian Text Summarization)** is the ideal dataset for this extension:

| Property | Value | Why It Matters |
|----------|-------|----------------|
| **Language** | Italian | Native Italian summaries (not translated) |
| **Document Length** | ~1,500+ tokens | Justifies Longformer (not BERT) |
| **Domain** | Wikipedia | Encyclopedic, factual content |
| **Size** | 300k+ pairs | Sufficient for fine-tuning |

### 4.3 The XLM-RoBERTa-Longformer Challenge

**The Problem:** There is no pre-trained Italian Longformer.
**The Solution:** Use `markussagen/xlm-roberta-longformer-base-4096`.

| Model | Languages | Max Tokens | Source |
|-------|-----------|------------|--------|
| `allenai/longformer-base-4096` | English only | 4096 | ❌ Cannot use |
| `markussagen/xlm-roberta-longformer-base-4096` | 100+ (incl. Italian) | 4096 | ✅ Our choice |

### 4.4 Italian-Specific Preprocessing

```python
import spacy

# Load Italian model (NOT English!)
nlp = spacy.load("it_core_news_sm")

def segment_italian(text: str) -> list[str]:
    """
    Italian-aware sentence segmentation.
    Handles: Art., Sig., Dott., ecc.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text) > 20]
```

### 4.5 Italian Prompt Template

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sei un esperto di sintesi testuale italiano. Il tuo compito è generare riassunti 
fedeli, completi e coerenti.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Testo Originale:
{source_text}

ISTRUZIONI OBBLIGATORIE:
I seguenti concetti chiave DEVONO essere riflessi nel riassunto:
{extracted_keyphrases}

Genera un riassunto astrattivo in italiano:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### 4.6 Implementation Files

| File | Purpose |
|------|---------|
| `src/data/wits_loader.py` | WITS dataset ingestion and preprocessing |
| `src/utils/text_processing.py` | Italian-specific tokenization |
| `src/models/sigext_model.py` | XLM-R Longformer wrapper |
| `notebooks/modulo3_training.ipynb` | Italian fine-tuning |

---

## 5. Core Implementation Modules

### 5.1 Module 1: The Long Architecture (Foundation)
**Objective:** Build a model that can read long Italian texts without crashing.

**Step-by-Step Implementation:**
1.  **Architecture Conversion:** Since there is no `XLM-RoBERTa-Longformer` in the main HuggingFace library, we must construct it. We take the weights of `xlm-roberta-base` and inject them into a `LongformerConfig`.
2.  **Attention Injection:** We copy the semantic weights ($Q, K, V$) from XLM-R to Longformer. The *positional embeddings* must be expanded (from 512 to 4096) by copying and repeating the existing embeddings (a technique known as "Position Embedding Expansion").
3.  **Global Attention Masking:** In the forward pass, we must create a special `global_attention_mask`. We set this to `1` only for the first token (`<s>`) and potentially for the first token of every paragraph.

**Deliverables:**
- `src/models/sigext_model.py`
- `notebooks/modulo1_architecture.ipynb`

### 5.2 Module 2: The Semantic Pipeline (Data Engineering)
**Objective:** Create the "Silver Standard" dataset using semantic supervision.

**Step-by-Step Implementation:**
1.  **Segmentation:** Use `spacy` (`it_core_news_sm`) for Italian sentence boundary detection.
2.  **Embedding Computation:** Encode source and summary sentences into 768-dimensional vectors using S-BERT.
3.  **Threshold Application:** Label sentences with cosine similarity > 0.60 as salient.

**Deliverables:**
- `src/data/labeling.py`
- `notebooks/modulo2_semantic_labels.ipynb`

### 5.3 Module 3: Training the Extractor
**Objective:** Teach the Longformer to predict the semantic labels.

**Key Technical Decisions:**
- **Weighted Loss:** Apply $w=10.0$ to positive class to handle imbalance.
- **Batch Size:** 2 (VRAM constraint) with gradient accumulation of 16.
- **Precision:** FP16 mandatory.

**Deliverables:**
- `src/data/wits_loader.py`
- `src/training/sigext_trainer.py`
- `notebooks/modulo3_training.ipynb`

### 5.4 Module 4: Inference & Steering
**Objective:** Combine the Extractor and Generator into a complete pipeline.

**Key Technical Decisions:**
- **Quantization:** Llama-3 in 4-bit NF4.
- **Prompt Template:** Dynamic injection of keyphrases into system prompt.
- **LangChain:** For clean orchestration.

**Deliverables:**
- `src/inference/generator.py`
- `src/evaluation/metrics.py`
- `notebooks/master_evaluation.ipynb`

---

## 6. Evaluation Strategy: The Three Experiments

To prove scientific value, we run three configurations and compare:

| Configuration | Description | Purpose |
|---------------|-------------|---------|
| **Baseline Zero-Shot** | Llama-3 on WITS without keyphrases | Establish lower bound |
| **SigExt-Fuzzy** | Llama-3 + SigExt trained with Fuzzy labels | Reproduce original paper on Italian |
| **SigExt-Semantic** | Llama-3 + SigExt trained with Semantic labels | **Our innovation** |

### 6.1 Metrics

| Metric | Type | What It Measures |
|--------|------|------------------|
| **ROUGE-1/2/L** | Lexical | Word overlap with reference |
| **BERTScore** | Semantic | Meaning preservation |
| **KIR** | Controllability | Did the LLM include the keyphrases? |
| **AlignScore** | Faithfulness | Is the summary factually grounded? |

### 6.2 Expected Results

| Method | ROUGE-1 | BERTScore | KIR |
|--------|---------|-----------|-----|
| Baseline Zero-Shot | 0.35 | 0.78 | N/A |
| SigExt-Fuzzy | 0.38 | 0.80 | 0.65 |
| **SigExt-Semantic** | **0.41** | **0.83** | **0.78** |

---

## 7. Summary of Innovation

**SM-SIP** delivers two scientifically significant contributions:

1. **Extension 1 (Semantic Supervision):** Replaces noisy lexical labels with clean semantic labels, improving extractor quality.
2. **Extension 2 (Italian Adaptation):** Validates SIP on a morphologically rich, low-resource language, demonstrating cross-lingual generalization.

Together, these extensions elevate the project from "reproduction" to "original research contribution."