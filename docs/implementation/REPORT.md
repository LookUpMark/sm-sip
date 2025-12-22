# Strategic Analysis and Architectural Optimization for Abstractive Summarization Systems Based on Salient Information Prompting

## Towards a Semantic and Multilingual Approach

---

## 1. Introduction and Operational Context

This document constitutes a comprehensive research report aimed at the definition, critical review, and optimization of a project proposal for the "Deep Natural Language Processing" course at the Polytechnic University of Turin. The primary objective is to outline an implementation path that not only meets the standard academic requirements for passing the exam but is strategically structured to achieve the maximum score (10/10), with particular emphasis on acquiring the 5 additional points reserved for original "extensions." Furthermore, the proposal is formulated with the methodological rigor necessary to lay the foundations for a potential scientific publication in venues such as student workshops (e.g., Evalita) or sector conferences (e.g., EMNLP Findings, CLiC-it).

The core theme of the project focuses on **Text Summarization**, one of the most complex and relevant challenges in contemporary Natural Language Processing (NLP). Specifically, the work stems from the reference paper *Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization* by Xu et al. (2024), presented at the EMNLP Industry Track conference. This work introduces an innovative paradigm to control ("steer") the output of Large Language Models (LLMs) through the injection of keywords and salient phrases extracted directly from the source document, thereby mitigating hallucination and improving information coverage.

The following analysis is articulated in several phases:
1. A deep technical examination of the reference paper to understand its internal mechanisms and undeclared limitations.
2. A comparative evaluation of the two project drafts proposed by the user (Decoder-Only Architecture vs. Encoder-Decoder).
3. The synthesis of a Unified Project Proposal integrating the best components of the existing drafts with advanced methodological extensions.

### 1.1 The Problem of Controllability in Abstractive Generation

The advent of Large Language Models (LLMs) such as GPT-4, Llama 3, and Mistral has revolutionized the field of summarization. Unlike previous approaches based on supervised fine-tuning of sequence-to-sequence models (such as BART or PEGASUS), LLMs allow for the generation of fluid and coherent summaries in zero-shot or few-shot modes. However, this flexibility introduces a critical problem of **controllability**. When a user requests "Summarize this text," the model relies entirely on its internal weights and biases learned during pre-training to determine what is "salient."

This "black-box" mechanism often leads to two types of errors:

- **Omission of Critical Details:** The model might overlook entities or events that, while statistically rare, are fundamental to understanding the specific text.
- **Hallucination:** In the absence of strong constraints, the model may generate content that is plausible but not factual, or distort relationships between entities present in the source text.

The paper by Xu et al. (2024) proposes resolving this issue through **Salient Information Prompting (SIP)**. The central idea is to decouple the identification of salience from text generation. By using a dedicated extractor module (named **SigExt**) to identify key phrases a priori, it is possible to "guide" the LLM's generative process by injecting these phrases directly into the prompt. This hybrid approach, combining explicit extraction with abstractive generation, represents the current frontier of industrial research in the field of summarization.

---

## 2. Technical Deconstruction of the Reference Paper (SigExt)

To propose valid extensions, it is indispensable to thoroughly understand the architecture, design choices, and, above all, the intrinsic weaknesses of the SigExt system presented in the reference paper. It is not merely about replicating the code, but interrogating the theoretical foundations of the work.

### 2.1 The Extractor Architecture: Why Longformer?

The heart of the SIP system is the **SigExt (Keyphrase Signal Extractor)** module. Unlike classical unsupervised methods such as TextRank or YAKE, SigExt is a supervised neural model. The architectural backbone chosen is **Longformer-Large** (Beltagy et al., 2020).

This choice is not accidental but responds to a precise computational need. Standard Transformer models, like BERT or RoBERTa, use a full self-attention mechanism, whose computational and memory complexity grows quadratically with respect to the input sequence length ($O(N^2)$). This imposes a rigid limit, typically of 512 tokens, which is insufficient for processing long documents such as scientific papers (ArXiv dataset) or meeting transcripts (MeetingBank dataset) without resorting to aggressive truncation that loses information.

Longformer solves this problem by introducing a **"Sparse Attention"** mechanism, which combines:

- **Sliding Window Attention:** Each token attends only to neighboring tokens within a local window.
- **Global Attention:** Certain special tokens (like the `<s>` token) attend to all other tokens in the sequence.

This combination reduces complexity to linear ($O(N)$), allowing the processing of sequences up to 4,096 tokens. In the context of the university project, replicating this choice is fundamental if one intends to work on long document datasets, which are often the most challenging and interesting for analysis.

### 2.2 The Achilles' Heel: Training Label Generation

A crucial, and potentially weak, aspect of the paper concerns how SigExt is trained. Most summarization datasets (e.g., CNN/DailyMail) provide pairs of (Document, Summary) but do not explicitly provide the list of "key phrases" to be extracted.

To overcome this lack, the authors generate training labels (Ground Truth) heuristically. They compare every sentence of the source document ($p$) with the sentences of the reference summary ($q$) using a **Fuzzy Matching** metric at the character level:

$$\text{Fuzz}(p, q) = \frac{\text{LongestCommonSubsequence}(p, q)}{\max(|p|, |q|)}$$

If the maximum matching score exceeds a threshold $\epsilon$ (set at 0.7), the sentence $p$ is labeled as "salient" (Label = 1), otherwise as non-salient (Label = 0).

> **Methodological Critique:** This reliance on lexical matching represents a significant scientific limitation. Fuzzy Matching captures only character overlap. If the summary uses synonyms or complex paraphrasing (e.g., Source: "The company declared bankruptcy" vs. Summary: "The firm failed"), the fuzzy score will be low, and SigExt will erroneously learn not to select that phrase as important. This creates a "noisy" and suboptimal supervision signal, limiting the model's ability to learn true semantic salience. **Here lies one of the greatest opportunities for an innovative extension.**

### 2.3 Results and Risks: The Hallucination Paradox

The experimental analysis of the paper reveals a counterintuitive phenomenon. While adding keyphrases to the prompt consistently improves lexical overlap metrics (ROUGE), the impact on factual faithfulness (measured via AlignScore) is variable.

The authors explicitly note that:
> *"the impact on hallucination is not universally positive across LLMs. For certain models like Mistral, adding keyphrases can lead to increased hallucinations."*

This happens because the model, seeing a keyword in the prompt without its original context (for example, a negation), might "invent" a sentence that includes it incorrectly.

This result opens the way to a second line of extension: the implementation of **control and refinement mechanisms** (such as Self-Refine) to mitigate the risk introduced by explicit prompting.

---

## 3. Critical Evaluation of Project Drafts

In light of the paper's deconstruction, we now examine the two proposed drafts to determine which offers the best potential for the maximum score.

### 3.1 Analysis of Draft 1: Decoder-Only Architecture (DecLLM)

This proposal suggests the use of instruction-tuned generative models such as Gemma-7B-it, Mistral-7B-Instruct, or Llama-3, combined with a RAG (Retrieval-Augmented Generation) pipeline.

**Strengths:**
- **State-of-the-Art Alignment:** The use of Decoder-Only models reflects current industrial and academic standards. Working with Llama 3 or Mistral demonstrates up-to-date competence on the latest technologies.
- **RAG Integration:** The proposal to integrate RAG is excellent. While SigExt extracts sentences, RAG extracts contexts, providing a more solid factual basis and directly addressing the problem of hallucinations and long document management.
- **"Self-Refine" Extension:** The idea of using the model itself to critique and correct its own output is methodologically advanced and directly addresses the limitations cited in the paper regarding Mistral's faithfulness.

**Critical Issues:**
- **"Precision-Recall Analysis" Extension:** The draft proposes varying the number of keyphrases (K) as an extension. ⚠️ **Warning:** This is not a valid extension for the maximum score. The original paper already performs exactly this analysis (see Figure 2 in the original paper: "Effect of using different number of keyphrases on the precision-recall trade off"). Replicating a graph already present in the paper falls under the category of "Reproducibility" (2 points), not "Extensions" (5 points).

### 3.2 Analysis of Draft 2: Encoder-Decoder Architecture (RedLLM)

This proposal is based on classic sequence-to-sequence models like BART or PEGASUS, using KeyBERT for extraction.

**Strengths:**
- **Efficiency:** Encoder-Decoder models are often lighter and faster in the inference phase compared to Decoder-Only giants.
- **Advanced Metrics:** The implementation of ROUGE-W and ROUGE-S is a valid extension, albeit limited to the scope of evaluation.

**Critical Issues:**
- **Technological Obsolescence:** Models like BART (2019) and PEGASUS (2020) are consolidated technologies but less "appealing" for a Deep NLP project in 2025-2026 compared to generative LLMs.
- **Context Limitations:** BART typically has a limit of 1024 tokens. This clashes with the paper's emphasis on summarization of long documents. The use of Llama 3 (8k tokens) or Mistral (32k tokens) is much more natural for this task.
- **Methodological Inconsistency:** The draft suggests using KeyBERT (unsupervised) instead of SigExt (supervised). However, the paper explicitly claims that SigExt's supervised approach is superior. Replacing an advanced component with a simpler one without strong theoretical justification could weaken the project.

### 3.3 Verdict and Selection

**Draft 1 (Decoder-Only)** represents the clearly superior foundation. The use of models like Llama 3 and integration with modern frameworks like LangChain offer much more fertile ground for demonstrating advanced skills. However, to guarantee the "maximum score" and publishability, it is imperative to replace the weak extension (Precision-Recall Analysis) with a robust and innovative methodological extension.

---

## 4. Unified Project Proposal: "SM-SIP" (Semantic & Multilingual Salient Information Prompting)

To maximize the result, it is recommended to merge the structure of Draft 1 with two high-profile scientific extensions. The project will be titled:

**"Enhancing Prompt-Based Summarization via Semantically-Supervised Keyphrase Extraction in Low-Resource Settings"**

This configuration aims to cover all evaluation areas:
- **Clarity:** Clean architecture.
- **Reproducibility:** Open-source code.
- **Extensions:** Methodological innovation and domain adaptation.

### 4.1 Core Architecture (Project Base)

The base architecture will replicate the paper's pipeline but update the technological components to 2025 standards:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Generator (LLM)** | `Meta-Llama-3-8B-Instruct` | SOTA performance, 8k context window, excellent instruction following. Alternative: `Mistral-7B-v0.3`. |
| **Extractor (SigExt)** | XLM-RoBERTa-Longformer | Multilingual support (Italian), handles sequences up to 4096 tokens. |
| **Orchestration** | LangChain | Manages data flow: load documents → extract keyphrases → construct dynamic prompt → generate summary. |

### 4.2 Extension 1 (Methodological): Semantic Supervision (2.5 Points)

This extension addresses the methodological critique raised in section 2.2: the weakness of Fuzzy Matching.

**The Concept:** Instead of training SigExt on labels generated via simple character overlap, we will generate labels based on **semantic similarity**.

**Implementation:**
1. Use a Sentence-BERT model (e.g., `all-MiniLM-L6-v2` or `paraphrase-multilingual-mpnet-base-v2`) to encode all sentences of the source document ($P_{src}$) and all sentences of the target summary ($P_{ref}$) into dense vectors (embeddings).
2. Calculate the **Cosine Similarity** between the vectors.
3. Define a new labeling function:

$$\text{Label}(p) = 1 \iff \max_{q \in P_{ref}} \text{CosSim}(\vec{p}, \vec{q}) > \theta$$

where $\theta$ is an experimental threshold (e.g., 0.6).

**Scientific Value:** This modification transforms the extraction paradigm from "lexical" to "semantic." The hypothesis is that SigExt will learn to extract *concepts*, not just words, improving the quality of the signal sent to the LLM. This is an original scientific contribution that elevates the project above simple reproduction.

### 4.3 Extension 2 (Domain/Linguistic): Italian Multilingual Adaptation (2.5 Points)

This extension directly responds to the "Multilingual extension" suggestion present in the course slides and gives the project a strong local identity ("appealing" for an Italian university).

- **The Concept:** Demonstrate that the SIP method is not specific to English but generalizes to other morphologically richer languages like Italian.
- **Dataset:** Use of the **WITS (Wikipedia for Italian Text Summarization)** dataset. WITS is ideal because it contains long and complex Wikipedia articles, justifying the use of Longformer, unlike short news datasets.
- **Technical Challenge:** There is no standard pre-trained "Italian Longformer." We will have to use `XLM-RoBERTa-Longformer` (an adaptation of the multilingual XLM-R model to support sequences up to 4096 tokens). We will need to fine-tune this model on the WITS dataset using our "Semantic Labels."
- **Scientific Value:** Validate the effectiveness of prompting with salient information in a low-resource context (compared to English) and verify whether the LLM (Llama 3) respects content constraints even in the Italian language.

---

## 5. Detailed Implementation Plan

Below is a step-by-step technical roadmap for project realization, including details on datasets, model configurations, and evaluation metrics.

### 5.1 Data Preparation

To demonstrate both reproducibility and extension, a dual-dataset approach is recommended.

| Dataset | Language | Document Type | Project Scope | Complexity |
|---------|----------|---------------|---------------|------------|
| **ArXiv** | English | Scientific Papers | Reproduction. Validation of code against original paper results. | High (avg. 6,000 tokens) |
| **WITS** | Italian | Wikipedia Articles | Extension. Demonstration of multilingual capability. | Medium-High (avg. >1,000 tokens) |

**Pre-processing Pipeline for Semantic Extension:**
1. **Segmentation:** Split long documents into sentences (using `spacy` for Italian).
2. **Encoding:** Use `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` to generate embeddings. This model supports both English and Italian, ensuring consistency between the two parts of the project.
3. **Labeling:** Generate training JSONL files for SigExt where each sentence has a 0/1 label based on cosine similarity with the gold summary.

### 5.2 Model Configuration

#### 5.2.1 The Extractor (SigExt Improved)

For the Italian extension, we cannot use the English checkpoint `allenai/longformer-base-4096`. Instead, we will use `markussagen/xlm-roberta-longformer-base-4096`.

- **Training:** Fine-tuning for the Token Classification (or binary Sequence Labeling) task.
- **Imbalance Management:** Since key phrases are rare (few 1 labels compared to 0 labels), it is crucial to implement a **Weighted Cross-Entropy Loss** during training to penalize errors on positive classes more heavily.

#### 5.2.2 The Generator (LLM)

- **Model:** `Meta-Llama-3-8B-Instruct`. This model can be run on consumer GPUs (like Google Colab T4) using 4-bit quantization (QLoRA) via the `bitsandbytes` library.
- **Prompt Engineering:** The prompt must be adapted dynamically. Example template for Italian:

```text
Sei un esperto di sintesi testuale. Leggi attentamente il seguente articolo.
{source_text}

Inoltre, considera con particolare attenzione i seguenti concetti chiave estratti dal testo, 
che devono essere riflessi nel riassunto:
{extracted_keyphrases}

Genera un riassunto astrattivo completo, coerente e fedele al testo originale.
```

### 5.3 Experimental Protocol and Metrics

To obtain the maximum score in the "Analysis of Results" section (2 points), the analysis must not be limited to tables of numbers but must offer profound qualitative and quantitative insights.

#### 5.3.1 Evaluation Metrics

Do not limit to ROUGE. A project of this level requires a holistic set of metrics:

| Metric | Type | Purpose |
|--------|------|---------|
| **ROUGE-1/2/L** | Lexical | Measures lexical overlap. Indispensable for comparison with literature. |
| **BERTScore** | Semantic | Measures meaning similarity between the generated and reference summary using contextual embeddings. Fundamental to proving that the semantic extension works better than the fuzzy one. |
| **AlignScore or NLI** | Faithfulness | Use a Natural Language Inference model to verify if the summary is "entailed" by the source document. This serves to quantify hallucinations. |
| **Keyphrase Inclusion Rate (KIR)** | Custom | Calculates the percentage of keyphrases suggested in the prompt that actually appeared in the final summary. This measures the effectiveness of "steering." |

**KIR Formula:**

$$\text{KIR} = \frac{|k \in \text{Keyphrases} : k \in \text{Summary}|}{|\text{Keyphrases}|}$$

#### 5.3.2 Planned Experiments

The final report should present the results of three main configurations:

1. **Baseline Zero-Shot:** Llama-3 on WITS without any keyphrase prompt. (Starting point).
2. **SigExt-Fuzzy (Reproduction):** Llama-3 + SigExt trained with Fuzzy labels (replicating the paper's method, but on Italian data).
3. **SigExt-Semantic (Extension):** Llama-3 + SigExt trained with Semantic labels (our innovation).

**Hypothesis to Verify:** It is expected that SigExt-Semantic will produce a higher BERTScore and KIR compared to SigExt-Fuzzy, demonstrating that training the extractor on concepts (embeddings) is superior to training it on strings (fuzzy).

---

## 6. Strategy for the Report and Oral Exam

The quality of the final report (in LaTeX) is decisive. Here is how to structure the narrative to maximize impact.

### 6.1 Scientific Narrative

Do not present the project as a simple "completed assignment," but as a scientific investigation.

- ❌ Instead of saying: *"We applied the method to Italian."*
- ✅ Write: *"While Xu et al. (2024) demonstrated the efficacy of SIP for English, the method's robustness on morphologically rich languages like Italian remained unexplored. Our study bridges this gap by evaluating the cross-lingual adaptability of the SigExt architecture on the WITS dataset."*

### 6.2 Data and Table Management

Use clear comparative tables that highlight the improvement delta.

**Example of Expected Results Table (Simulation):**

| Method | Dataset | ROUGE-1 | BERTScore | KIR (Inclusion) | Notes |
|--------|---------|---------|-----------|-----------------|-------|
| Llama-3 Zero-Shot | WITS (IT) | 0.35 | 0.78 | N/A | Baseline |
| SigExt (Fuzzy) + Llama-3 | WITS (IT) | 0.38 | 0.80 | 0.65 | Original Paper Method |
| **SigExt (Semantic) + Llama-3** | **WITS (IT)** | **0.41** | **0.83** | **0.78** | **Our Extension** |

This table visually demonstrates that the proposed extension brings tangible value.

---

## 7. Conclusions and Immediate Roadmap

In conclusion, the analysis conducted unequivocally indicates that the path to the maximum score lies in adopting the **Decoder-Only Architecture (Draft 1)**, but enhanced by more rigorous extensions than those initially proposed.

The **"SM-SIP" (Semantic & Multilingual Salient Information Prompting)** proposal satisfies all criteria of excellence:

| Criterion | How SM-SIP Satisfies It |
|-----------|-------------------------|
| **Complexity** | Manages multiple models (Longformer + Llama 3) and embedding pipelines. |
| **Novelty** | Introduces a semantic supervision method not present in the original paper. |
| **Relevance** | Brings research to an Italian dataset (WITS), responding to a specific course request. |
| **Publishability** | The results of a "Fuzzy vs. Semantic Supervision" comparison in a low-resource context constitute valid material for scientific publication. |

### Next Operational Steps:

1. ✅ Download the WITS dataset and checkpoints for `Meta-Llama-3-8B-Instruct` and `markussagen/xlm-roberta-longformer-base-4096`.
2. ✅ Implement the Python script for generating semantic labels using `sentence-transformers`.
3. ✅ Configure the training environment (Hugging Face Trainer + PEFT/LoRA) for fine-tuning the extractor.
4. ✅ Start the first experimental runs on the WITS dataset.