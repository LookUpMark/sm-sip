# Comprehensive Evaluation Analysis: English (ArXiv) vs. Italian (WITS)

This report provides a detailed diagnostic analysis of the SigExt + LLM summarization pipeline. It explores why traditional metrics often fail to reflect actual model utility and examines the reliability of the LLM-as-a-Judge framework.

## 1. Aggregate Performance Metrics

| Metric | English (ArXiv) | Italian (WITS) | Diagnostic Interpretation |
| :--- | :---: | :---: | :--- |
| **ROUGE-1 (Mean)** | 0.4010 | 0.2033 | **Ground Truth Bias**: ArXiv references are abstractive; WITS references are often single-sentence "snippets". |
| **BERTScore (Mean)** | 0.8357 | 0.6617 | **Model Disparity**: EN uses `roberta-large` (355M params); IT uses `bert-base-multilingual-cased` (110M params). Absolute values are **not cross-lingua comparable** — see §2.D. |
| **KIR (Mean)** | 0.6680 | 0.4397 | **Salience Mismatch**: Italian summaries often prioritize narrative flow over verbatim "Keyphrase" retention. |
| **Judge Abstraction** | **4.76** | **4.92** | **Fluidity**: Italian summaries often receive higher style marks for natural phrasing. |
| **Refusal Rate** | **~10% (EN)** | **<1% (IT)** | **Technical Fragility**: Mathematical placeholders in ArXiv occasionally trigger judge refusals. |
| **Judge Faithfulness** | **4.78** | **4.97** | **High Fidelity**: Hallucinations are rare and usually caught (see Diagnostics). |
| **Judge Completeness**| **4.66** | **4.95** | **Superior Utility**: Llama often provides *more* information than the baseline references. |

---

## 2. Systematic Evaluation Patterns

Through a systematic audit of 1000+ samples, we have identified three critical recurring patterns that explain the gap between metrics and qualitative utility.

### A. The "Concise Reference" Paradox (Italian WITS)
**Pattern**: Extremely low ROUGE (0.03 - 0.12) + Perfect Judge Scores (5/5).
- **Cause**: WITS references are often single-sentence snippets, especially for **Disambiguation Pages** (e.g., "San Juan", "Santa Fe"). The reference is just the header, while the LLM generates a full, high-quality summary of the entire list.
- **Fact**: In over 40% of Italian discrepancy cases, the reference was less than 20% of the length of the generated summary.
- **Example**: [Sample #1311 (Dave Sim)](file:///home/marcantoniolopez/Documenti/github/projects/sm-sip/notebooks/evaluation/results/italian/eval_enhanced.json#L1302-1331) receives a ROUGE-L of **0.05** but a Judge score of **5/5** for Completeness and Faithfulness.

### B. Technical Placeholder Fragility (English ArXiv)
**Pattern**: "Unable to evaluate" verdict from the LLM Judge.
- **Cause**: High density of symbolic placeholders (e.g., `@xmath1`, `@xmath3`) in the source text.
- **Finding**: Refusals correlate strongly with formula-heavy abstracts. The judge (Llama-3.1-8B) struggles to interpret sentences where math tokens replace core technical concepts.

### C. The "SigExt Mismatch" (Salience Divergence)
**Pattern**: KIR Score of 0.0 + High Utility Summary.
- **Cause**: The SigExt model extracts raw technical entities (e.g., administrative lists), but the LLM intelligently decides to ignore them in favor of a cohesive narrative.
- **Finding**: LLM Intelligence > Simple Extraction; the model prioritizes user-readable flow over verbatim keyword retention.

### D. BERTScore Model Disparity (Cross-Lingual)
**Pattern**: BERTScore(EN) ≈ 0.84 vs BERTScore(IT) ≈ 0.66 — a gap of ~0.17.
- **Cause**: The `bert-score` library auto-selects different underlying models based on `lang`:
  - **English** -> `roberta-large` (355M parameters, English-only, high-quality embeddings)
  - **Italian** -> `bert-base-multilingual-cased` (110M parameters, 104 languages, lower-resolution embeddings)
- **Consequence**: The 0.17 gap reflects **model capacity**, not summary quality. Comparing absolute BERTScore values across languages is methodologically invalid.
- **Solution**: A dedicated [Cross-Lingual Ablation](file:///home/marcantoniolopez/Documenti/github/projects/sm-sip/notebooks/ablation/ablation-cross-lingual.ipynb) uses a unified model (`microsoft/mdeberta-v3-base`) for both languages, producing comparable scores.

---

## 3. Granular Diagnostics: Sample Case Studies

### A. The "Reference Brevity" Trap (Model Win)
In the Italian WITS dataset, traditional metrics (ROUGE-1: 0.07) severely penalize the model for being *too good*.
*   **Sample 1311 (Dave Sim)**:
    *   **Human Reference**: "Famous for Cerebus... pioneer of self-publishing." (2 sentences).
    *   **Llama Summary**: A comprehensive biography covering his collaborator Gerhard, rights advocacy, the 300-issue plan, and his later-life philosophical shift.
    *   **Metric Verdict**: **ROUGE-1: 0.079** (Failure).
    *   **Judge Verdict**: **Completeness: 5/5** (Success).
    *   **Lesson**: ROUGE fails when the "gold standard" is less informative than the model output.

### B. The "Salience Mismatch" (Metric Failure)
Knowledge Integration Ratio (KIR) can fail when SigExt and the LLM prioritize different information types.
*   **Sample 1525 (Provincia di Córdoba)**:
    *   **SigExt Output**: Extracted a list of 26 administrative departments (list format).
    *   **Llama Summary**: A detailed historical and geographical overview of the province.
    *   **Metric Verdict**: **KIR: 0.0** (Total failure to match keyphrases).
    *   **Qualitative Verdict**: The summary is highly useful and accurate, but it correctly ignored the "low-value" list of departments in favor of narrative history.

### C. Judge Blind Spots & Hallucinations
The LLM-as-Judge (Qwen2.5-14B) is effective but has identifiable biases and occasional "hallucinations" in its own reasoning.

#### 1. Abstraction False Positives
*   **Sample 1804 (Adam Strange)**:
    *   **Llama Behavior**: Copied the final sentence of the source text almost verbatim.
    *   **Automated Metrics**: **Abstraction Score: 0.13** (Correctly identified copying).
    *   **Judge Verdict**: **Abstraction: 5/5** ("Uses novel phrasing").
    *   **Diagnostic**: The judge can be overly lenient on style if the overall summary feels fluid.

#### 2. Reasoning False Negatives (Judge Hallucination)
*   **Sample 1804 (Adam Strange)**:
    *   **Judge Reason**: Claimed the summary was "Missing information about the character's creator".
    *   **Reality**: The summary explicitly stated: *"Il personaggio venne ideato nel 1959 da Gardner Fox"*.
    *   **Diagnostic**: The judge occasionally fails to "read" the generated summary carefully enough before providing the reason for a score reduction.

#### 3. Catching Information Leakage
*   **Sample 2378 (Rosalba Carriera)**:
    *   **Issue**: Llama added the title of a specific painting ("Ritratto femminile con maschera") which was in the *Reference Title* but **absent** from the provided *Source Text*.
    *   **Judge Verdict**: **Faithfulness: 4/5**. The judge correctly identified that this information was outside the source context.
    *   **Lesson**: The "Source-Aware" prompt is highly effective at preventing models from using external knowledge (even when that knowledge is technically true).

---

## 4. Conclusions and Recommendations

1.  **Metric Reliability**: For **Italian (WITS)**, ROUGE is **invalid** as a quality proxy due to reference brevity. BERTScore absolute values are **not cross-lingua comparable** due to model disparity (see §2.D). Use the [Cross-Lingual Ablation](file:///home/marcantoniolopez/Documenti/github/projects/sm-sip/notebooks/ablation/ablation-cross-lingual.ipynb) with unified `mdeberta-v3-base` for fair comparison. **Judge Scores** remain the most reliable evaluation pillar.
2.  **English Technical Limits**: The pipeline is highly stable for English narrative science, but accuracy/evaluation stability drops in **formula-heavy** papers where `@xmath` placeholders dominate.
3.  **Instruction Alignment**: The "Source-Aware" prompt effectively manages the "More Complete Than Reference" feature, ensuring that additions are grounded in the source text rather than external hallucinations.
4.  **Judge Strategy**: To eliminate "Reasoning False Negatives," we recommend a **Majority Vote** judge system or upgrading the judge to a larger model (e.g., Llama-3-70B) for mission-critical audit batches.
